"""Gradient-boosting baselines on the 54 summary features, and persistence.

    python scripts/train_xgboost.py --stage main | cv"""
import argparse
import json

import _path  # noqa: F401
import numpy as np
import xgboost as xgb

from doa import config as C
from doa.data import calibration_sample, load_arrays, make_split, samples, window_features
from doa.metrics import scores
from doa.training import log

GRID = [{"max_depth": dep, "learning_rate": lr, "objective": ob} for dep in (4, 6, 8) for lr in (0.02, 0.05)
        for ob in ("reg:squarederror", "reg:absoluteerror")]
DEFAULT = {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05, "early_stopping_rounds": 30}


def fit(F, S, seed, **params):
    p = {"n_estimators": 6000, "early_stopping_rounds": 40, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": 8,
         "tree_method": "hist", "eval_metric": "mae", "random_state": seed, **params}
    m = xgb.XGBRegressor(**p)
    m.fit(F["train"], S["train"][2], eval_set=[(F["val"], S["val"][2])], verbose=False)
    return m, float(np.mean(np.abs(m.predict(F["val"]) - S["val"][2])))


def prepare(d, split, H):
    S = samples(d, split, H, train=150_000, val=37_500)
    return S, {p: window_features(d, s[1]) for p, s in S.items()}


def save(tag, d, S, p):
    idx, start, y = S["test"]
    np.savez(C.RUNS / f"{tag}.npz", y=y, yp=p.astype(np.float32), idx=idx, start=start, case=d["win_case"][idx])
    (C.RUNS / f"{tag}.json").write_text(json.dumps({"tag": tag, "test": scores(y, p)}, indent=1))
    log(f"{tag}: {scores(y, p)}")


def persistence(d, S):
    return np.asarray(d["NUM"][S["test"][1] + C.W - 1, 0])


def main_stage(d):
    split, _ = make_split(d)
    tuned = {}
    for H in C.HORIZONS:
        S, F = prepare(d, split, H)
        best = min(((fit(F, S, 0, **g)[1], k) for k, g in enumerate(GRID)))
        tuned[str(H)] = GRID[best[1]]
        log(f"XGBoost H={H}: {tuned[str(H)]}")
        for s in C.SEEDS:
            m, _ = fit(F, S, s, **tuned[str(H)])
            m.save_model(C.RUNS / f"xgb_H{H}_s{s}.ubj")
            save(f"xgb_H{H}_s{s}", d, S, m.predict(F["test"]))
            save(f"xgb_default_H{H}_s{s}", d, S, fit(F, S, s, **DEFAULT)[0].predict(F["test"]))
        save(f"persistence_H{H}", d, S, persistence(d, S))
        cal = calibration_sample(d, split, H)
        q, _ = fit(F, S, 0, **DEFAULT, objective="reg:quantileerror", quantile_alpha=np.array([0.025, 0.975]), eval_metric=None)
        np.savez(C.RUNS / f"xgb_quantile_H{H}.npz", cal=q.predict(window_features(d, cal[1])), test=q.predict(F["test"]))
    (C.RUNS / "tuned_xgb.json").write_text(json.dumps(tuned, indent=1))


def cv_stage(d):
    tuned = json.loads((C.RUNS / "tuned_xgb.json").read_text())
    for f in range(C.FOLDS):
        split, _ = make_split(d, f)
        for H in C.HORIZONS:
            S, F = prepare(d, split, H)
            save(f"cv_xgb_f{f}_H{H}", d, S, fit(F, S, 0, **tuned[str(H)])[0].predict(F["test"]))
            save(f"cv_persistence_f{f}_H{H}", d, S, persistence(d, S))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["main", "cv"], default="main")
    {"main": main_stage, "cv": cv_stage}[ap.parse_args().stage](load_arrays())
