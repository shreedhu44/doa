"""Coverage, width and interval score of nominal 95% prediction intervals (writes runs/results_intervals.json)."""
import json

import _path  # noqa: F401
import numpy as np

from doa import config as C
from doa.data import cohort, load_arrays, phase_labels, subjects
from doa.metrics import conformal_quantile, patient_weights


def artifact_flags(d, start, train_cases):
    """BIS input frozen for 30 s, and EEG power outside the 0.5-99.5% range of the training cases."""
    rows = np.sort(np.random.RandomState(0).choice(len(d["CASE"]), 1_000_000, replace=False))
    p = np.asarray(d["EEG"][rows, 6])[np.isin(d["CASE"][rows], train_cases)]
    lo, hi = np.percentile(p[p > -18], [0.5, 99.5])
    bis = np.asarray(d["NUM"][:, 0])[start[:, None] + np.arange(C.W - 30, C.W)]
    power = np.asarray(d["EEG"][:, 6])[start[:, None] + np.arange(C.W - 60, C.W)]
    return (bis == bis[:, :1]).all(1), ((power < lo) | (power > hi)).any(1)


def interval_methods(H):
    z = np.load(C.RUNS / f"interval_proposed_H{H}.npz")
    x = np.load(C.RUNS / f"xgb_quantile_H{H}.npz")
    mc = np.load(C.RUNS / f"interval_mcdropout_H{H}.npz")
    yc, P = z["y_cal"].astype(float), z["test"].astype(float)
    qtest, qcal = z["qtest"].astype(float), z["qcal"].astype(float)
    lo, hi, lc, hc = qtest.min(-1), qtest.max(-1), qcal.min(-1), qcal.max(-1)
    q = np.array([conformal_quantile(np.maximum(a - yc, yc - b)) for a, b in zip(lc, hc)])
    c = np.array([conformal_quantile(np.abs(yc - p)) for p in z["cal"].astype(float)])
    qx = conformal_quantile(np.maximum(x["cal"].min(1) - yc, yc - x["cal"].max(1)))
    return z, q, {
        "proposed": (lo - q[:, None], hi + q[:, None]),
        "constant width": (P - c[:, None], P + c[:, None]),
        "quantile XGBoost": ((x["test"].min(1) - qx)[None], (x["test"].max(1) + qx)[None]),
        "dropout +/- 2 SD": (mc["test"] - 2 * mc["sd"], mc["test"] + 2 * mc["sd"]),
    }


def main():
    d, subj, out = load_arrays(), subjects(), {}
    train_cases = [c for c, (_, p) in cohort().items() if p == "train"]
    for H in C.HORIZONS:
        z, q, methods = interval_methods(H)
        y, case, start = z["y"].astype(float), z["case"], z["start"]
        u, inv = np.unique([subj[int(c)] for c in case], return_inverse=True)
        n, w = np.bincount(inv).astype(float), patient_weights(len(u))
        frozen, eeg = artifact_flags(d, start, train_cases)
        phase = phase_labels(d, case, start, H)
        strata = {"all": np.ones(len(y), bool), **{p: phase == p for p in ("induction", "maintenance", "emergence")},
                  "frozen BIS": frozen, "EEG artifact": eeg, "no artifact": ~(frozen | eeg)}
        r, score_by_patient = {}, {}
        for k, (a, b) in methods.items():
            a, b = np.clip(a, 0, 100), np.clip(b, 0, 100)
            hit, width = ((y >= a) & (y <= b)).mean(0), (b - a).mean(0)
            score = ((b - a) + 40 * np.maximum(a - y, 0) + 40 * np.maximum(y - b, 0)).mean(0)
            score_by_patient[k] = np.bincount(inv, score, len(u))
            r[k] = {"patients_below_90": float(np.mean(np.bincount(inv, hit, len(u)) / n < 0.9)), "strata": {}}
            for s, m in strata.items():
                num, den = np.bincount(inv[m], hit[m], len(u)), np.bincount(inv[m], minlength=len(u)).astype(float)
                boot = (w @ num) / np.maximum(w @ den, 1)
                r[k]["strata"][s] = {"windows": int(m.sum()), "coverage": float(hit[m].mean()),
                                     "ci": np.percentile(boot, [2.5, 97.5]).tolist(), "width": float(width[m].mean()),
                                     "interval_score": float(score[m].mean())}
        diff = score_by_patient["proposed"] - score_by_patient["quantile XGBoost"]
        ci = np.percentile((w @ diff) / (w @ n), [2.5, 97.5]).tolist()
        out[str(H)] = {"offsets": q.tolist(), "methods": r, "score_proposed_minus_quantile_xgb": [float(diff.sum() / n.sum())] + ci}
        print(f"H = {H} s")
        for k, v in r.items():
            a = v["strata"]["all"]
            print(f"  {k:<18} coverage {a['coverage']:.3f}  width {a['width']:.1f}  score {a['interval_score']:.1f}")
    (C.RUNS / "results_intervals.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
