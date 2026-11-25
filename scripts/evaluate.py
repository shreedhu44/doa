"""Accuracy, paired comparisons, anesthetic phases and cross-validation.

    python scripts/evaluate.py --stage main | cv        # writes runs/results_<stage>.json"""
import argparse
import json

import _path  # noqa: F401
import numpy as np

from doa import config as C
from doa.data import load_arrays, phase_labels, subjects
from doa.metrics import band, growth, mean_scores, paired_difference, patient_weights, public, scores

SEEDED = ("proposed", "gru_nogaps", "gru_bis", "gru_bis_eeg", "gru_bis_drug", "gru_all", "transformer", "mlp",
          "anesnet", "xgb", "xgb_default")
PAIRS = [("proposed", k) for k in ("persistence", "pkpd", "xgb", "xgb_default", "anesnet", "transformer", "mlp", "gru_bis", "gru_all")] + \
        [("gru_bis_eeg", "gru_bis"), ("gru_bis_drug", "gru_bis"), ("gru_nogaps", "gru_bis"), ("proposed", "gru_nogaps"), ("gru_all", "gru_nogaps")]
PHASES = ("induction", "maintenance", "emergence")


def load(tags):
    z = [np.load(C.RUNS / f"{t}.npz") for t in tags]
    assert all(np.array_equal(a["idx"], z[0]["idx"]) for a in z), tags
    return z[0], np.stack([a["yp"].astype(float) for a in z])


def rmse_difference(y, inv, n, A, B_, reps=3000):
    rng = np.random.default_rng(0)
    sq = lambda P: np.stack([np.bincount(inv, (p - y) ** 2, len(n)) for p in P])
    a, b = sq(A), sq(B_)
    w = rng.multinomial(len(n), np.full(len(n), 1 / len(n)), size=reps).astype(float)
    ws = [rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=reps) / len(x) for x in (A, B_)]
    root = lambda s, x: np.sqrt(np.einsum("bp,bs,sp->b", w, s, x, optimize=True) / (w @ n))
    boot = root(ws[0], a) - root(ws[1], b)
    return {"diff": float(np.sqrt(a.mean(0).sum() / n.sum()) - np.sqrt(b.mean(0).sum() / n.sum())),
            "ci": np.percentile(boot, [2.5, 97.5]).tolist()}


def band_results(y, P, inv, n):
    t = band(y)
    conf = np.mean([[[np.sum((t == i) & (band(p) == j)) for j in range(4)] for i in range(4)] for p in P], 0)
    out = {"confusion": conf.tolist(), "recall": (np.diag(conf) / conf.sum(1)).tolist()}
    w = patient_weights(len(n), 3000)
    for name, true, flag in (("too_light", y > 60, np.mean(P > 60, 0)), ("too_deep", y < 40, np.mean(P < 40, 0))):
        hit, sup = np.bincount(inv, true * flag, len(n)), np.bincount(inv, true.astype(float), len(n))
        out[name] = {"recall": float(hit.sum() / sup.sum()), "ci": np.percentile((w @ hit) / (w @ sup), [2.5, 97.5]).tolist(),
                     "support": int(true.sum()), "precision": float((true * flag).sum() / flag.sum())}
    return out


def main_stage(d, subj):
    out = {}
    for H in C.HORIZONS:
        P = {}
        for k in SEEDED:
            z, P[k] = load([f"{k}_H{H}_s{s}" for s in C.SEEDS])
        for k in ("persistence", "pkpd"):
            P[k] = load([f"{k}_H{H}"])[1]
        y, case = z["y"].astype(float), z["case"]
        pat = np.array([subj[int(c)] for c in case])
        u, inv = np.unique(pat, return_inverse=True)
        n = np.bincount(inv).astype(float)
        phase = phase_labels(d, case, d["win_start"][z["idx"]], H)
        r = {"windows": len(y), "patients": len(u), "metrics": {k: mean_scores(y, p) for k, p in P.items()},
             "paired_MAE": {f"{a} - {b}": paired_difference(y, pat, P[a], P[b]) for a, b in PAIRS},
             "RMSE_proposed - xgb": rmse_difference(y, inv, n, P["proposed"], P["xgb"]), "phase": {}}
        for ph in PHASES:
            m = phase == ph
            r["phase"][ph] = {
                "windows": int(m.sum()),
                "MAE": {k: float(np.mean(np.abs(P[k][:, m] - y[m]))) for k in ("persistence", "gru_bis", "xgb", "anesnet", "proposed")},
                "paired_MAE": {k: paired_difference(y, pat, P["proposed"], P[k], mask=m) for k in ("gru_bis", "xgb", "anesnet")}}
        err = np.bincount(inv, np.abs(P["proposed"] - y).mean(0)) / n
        gain = err - np.bincount(inv, np.abs(P["xgb"] - y).mean(0)) / n
        r["per_patient"] = {"p10_p50_p90": np.percentile(err, [10, 50, 90]).tolist(), "mean_difference_to_xgb": float(gain.mean()),
                            "share_better_than_xgb": float(np.mean(gain < 0))}
        r["bands"] = band_results(y, P["proposed"], inv, n)
        out[str(H)] = r
    out["growth_30_to_120"] = {k: growth(out["30"]["paired_MAE"][f"proposed - {k}"], out["120"]["paired_MAE"][f"proposed - {k}"])
                               for k in ("xgb", "gru_bis")}
    return out


def cv_stage(d, subj):
    out = {}
    models = ("proposed", "gru_bis", "anesnet", "xgb", "persistence")
    for H in C.HORIZONS:
        Z = {k: [np.load(C.RUNS / f"cv_{k}_f{f}_H{H}.npz") for f in range(C.FOLDS)] for k in models}
        # the adapted AnesNet occasionally leaves the range of the index, so its forecasts are clipped
        pred = lambda k, z: np.clip(z["yp"].astype(float), 0, 100) if k == "anesnet" else z["yp"].astype(float)
        cat = lambda key: np.concatenate([z[key] for z in Z["proposed"]])
        y, case = cat("y").astype(float), cat("case")
        pat = np.array([subj[int(c)] for c in case])
        P = {k: np.concatenate([pred(k, z) for z in v])[None] for k, v in Z.items()}
        phase = phase_labels(d, case, d["win_start"][cat("idx")], H)
        folds = {k: [scores(z["y"].astype(float), pred(k, z)) for z in v] for k, v in Z.items()}
        r = {"windows": len(y), "patients": len(np.unique(pat)), "folds": folds,
             "mean_sd": {k: {m: [float(np.mean([s[m] for s in v])), float(np.std([s[m] for s in v], ddof=1))] for m in v[0]}
                         for k, v in folds.items()},
             "paired_MAE": {k: paired_difference(y, pat, P["proposed"], P[k]) for k in models[1:]}, "phase": {}}
        for ph in PHASES:
            m = phase == ph
            r["phase"][ph] = {"windows": int(m.sum()), "MAE": {k: float(np.mean(np.abs(p[0, m] - y[m]))) for k, p in P.items()},
                              "paired_MAE": {k: paired_difference(y, pat, P["proposed"], P[k], mask=m) for k in ("gru_bis", "xgb")}}
        out[str(H)] = r
    out["growth_30_to_120"] = {k: growth(out["30"]["paired_MAE"][k], out["120"]["paired_MAE"][k]) for k in ("xgb", "gru_bis")}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["main", "cv"], default="main")
    stage = ap.parse_args().stage
    res = public({"main": main_stage, "cv": cv_stage}[stage](load_arrays(), subjects()))
    (C.RUNS / f"results_{stage}.json").write_text(json.dumps(res, indent=1))
    for H in C.HORIZONS:
        print(f"H = {H} s")
        for k, v in (res[str(H)]["metrics"] if stage == "main" else res[str(H)]["mean_sd"]).items():
            m = (v["MAE"]["mean"], v["MAE"]["sd"]) if stage == "main" else v["MAE"]
            print(f"  {k:<14} MAE {m[0]:.3f} +/- {m[1]:.3f}")
