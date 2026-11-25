"""Alerts for new BIS excursions on the complete test recordings (writes runs/results_alerts.json).

An excursion (too light: BIS > 60, too deep: BIS < 40) lasts at least 30 s, merging interruptions of up to 30 s, and is
new if the preceding 60 s lie outside it. Alerts are evaluated every 10 s while the patient is outside the excursion;
alerts closer than 60 s are one alarm and a persisting alert counts again every 180 s. An alarm is true if a new
excursion starts within 180 s. Thresholds are set on the validation recordings to a given number of false alarms per
at-risk hour."""
import json

import _path  # noqa: F401
import numpy as np

from doa import config as C
from doa.data import load_arrays, subjects
from doa.metrics import patient_weights

LOOK, MERGE, QUIET = 18, 6, 6          # 180 s, 60 s and 60 s in 10-s steps
METHODS = ("current", "gru_bis", "anesnet", "xgb", "proposed")
VARIANTS = {"main": (True, 3), "inclusive thresholds": (False, 3), "at least 10 s": (True, 1), "at least 60 s": (True, 6)}


def recordings(d, subj, part):
    """10-s grid of all recordings of a partition: quality-passed BIS, current BIS input and forecast positions."""
    z = np.load(C.RUNS / "stream_index.npz")
    mask = z["part"] == part
    S, K = z["start"][mask], z["case"][mask]
    cut = np.r_[0, np.flatnonzero(np.diff(d["CASE"])) + 1]
    span = {int(d["CASE"][a]): (int(a), int(b)) for a, b in zip(cut, np.r_[cut[1:], len(d["CASE"])])}
    target = dict(zip((d["win_start"] + C.W + C.BASE_H).tolist(), d["win_y"].tolist()))
    truth, case, offset, pos = [], [], 0, np.empty(len(S), np.int64)
    for c in dict.fromkeys(K.tolist()):
        a, b = span[c]
        n = (b - a) // C.STRIDE + 1
        truth.append(np.array([target.get(a + C.STRIDE * i, np.nan) for i in range(n)]))
        case.append(np.full(n, c))
        k = K == c
        pos[k] = offset + (S[k] - a) // C.STRIDE + C.W // C.STRIDE
        offset += n
    truth, case = np.concatenate(truth), np.concatenate(case)
    current = np.full(len(truth), np.nan)
    current[pos] = np.asarray(d["NUM"][:, 0])[S + C.W - 1]
    return {"bis": truth, "case": case, "current": current, "pos": pos, "mask": mask,
            "patient": np.array([subj[int(c)] for c in case])}


def shifted(case, x, k):
    """x delayed by k steps, nan across recording borders."""
    out = np.full(len(x), np.nan)
    out[k:] = x[:-k]
    out[np.r_[np.ones(k, bool), case[k:] != case[:-k]]] = np.nan
    return out


def beyond(x, event, strict):
    if event == "light":
        return x > 60 if strict else x >= 60
    return x < 40 if strict else x <= 40


def excursions(R, event, strict, min_samples):
    bis, case, current = R["bis"], R["case"], R["current"]
    valid = ~np.isnan(bis)
    ex = np.where(valid, beyond(bis, event, strict), False)
    I = np.flatnonzero(ex)
    new_run = np.r_[True, (np.diff(I) > 4) | (case[I[1:]] != case[I[:-1]])]
    first = I[new_run][np.bincount(np.cumsum(new_run) - 1) >= min_samples]
    before = lambda x: sum(np.nan_to_num(shifted(case, x.astype(float), k)) for k in range(1, QUIET + 1))
    onset = first[(before(valid)[first] >= 4) & (before(ex)[first] == 0)]
    latest = np.full(len(bis), np.nan)
    for k in (3, 2, 1):
        v = shifted(case, np.where(valid, ex.astype(float), np.nan), k)
        latest = np.where(np.isnan(v), latest, v)
    at_risk = (latest == 0) & ~np.isnan(current) & (current > 0) & ~beyond(current, event, strict)
    return {"onset": onset, "all": len(first), "at_risk": at_risk}


def evaluate(R, T, score, thr, event):
    case, N, onset = R["case"], len(R["case"]), T["onset"]
    alert = T["at_risk"] & ~np.isnan(score) & ((score > thr) if event == "light" else (score < thr))
    I = np.flatnonzero(alert)
    end = np.r_[np.flatnonzero(np.diff(case)) + 1, N]
    alarm, true = I, np.zeros(0, bool)
    if len(I):
        start = np.r_[True, (np.diff(I) > MERGE) | (case[I[1:]] != case[I[:-1]])]
        run = np.cumsum(start) - 1
        block = (I - I[start][run]) // LOOK
        alarm = I[np.r_[True, (np.diff(run) > 0) | (np.diff(block) > 0)]]
        is_onset = np.zeros(N, bool)
        is_onset[onset] = True
        cum = np.r_[0, np.cumsum(is_onset)]
        last = np.minimum(alarm + LOOK + 1, end[np.searchsorted(end, alarm, side="right")])
        true = cum[last] > cum[np.minimum(alarm + 1, N)]
    begin = np.r_[0, end[:-1]][np.searchsorted(end, onset, side="right")]
    lo = np.maximum(onset - LOOK, begin)
    cum_alert = np.r_[0, np.cumsum(alert)]
    found = cum_alert[onset] > cum_alert[lo]
    lead = np.full(len(onset), np.nan)
    if found.any():
        first_alert = I[np.minimum(np.searchsorted(I, lo), len(I) - 1)]
        lead[found] = ((onset - first_alert) * C.STRIDE)[found]
    hours = T["at_risk"].sum() * C.STRIDE / 3600
    return {"sensitivity": float(found.mean()), "precision": float(true.mean()) if len(alarm) else np.nan,
            "false_alarms_per_hour": float((~true).sum() / hours),
            "median_lead": float(np.nanmedian(lead)) if found.any() else np.nan, "_found": found.astype(float)}


def threshold(R, T, score, event, target):
    """Most sensitive threshold with at most `target` false alarms per hour on the validation recordings."""
    grid = np.arange(40, 75.01, 0.25)[::-1] if event == "light" else np.arange(25, 60.01, 0.25)
    best = float(grid[0])
    for t in grid:
        if evaluate(R, T, score, t, event)["false_alarms_per_hour"] > target:
            break
        best = float(t)
    return best


def scores_on_grid(R, name, H):
    if name == "current":
        return R["current"][None]
    yp = np.load(C.RUNS / f"stream_{name}_H{H}.npz")["yp"][:, R["mask"]]
    out = np.full((len(yp), len(R["bis"])), np.nan)
    out[:, R["pos"]] = yp
    return out


def average(runs):
    out = {}
    for k in runs[0]:
        if not k.startswith("_"):
            v = np.array([r[k] for r in runs], float)
            out[k] = float(np.nanmean(v)) if np.isfinite(v).any() else None
    return out


def main():
    d, subj = load_arrays(), subjects()
    V, E = recordings(d, subj, 0), recordings(d, subj, 1)
    patients = np.unique(E["patient"])
    w, out = patient_weights(len(patients), 2000), {}
    for variant, (strict, min_samples) in VARIANTS.items():
        for event in ("light", "deep"):
            TV, TE = excursions(V, event, strict, min_samples), excursions(E, event, strict, min_samples)
            who = np.searchsorted(patients, E["patient"][TE["onset"]])
            den = np.bincount(who, minlength=len(patients)).astype(float)
            res = {"new": len(TE["onset"]), "all": TE["all"], "at_risk_hours": float(TE["at_risk"].sum() * C.STRIDE / 3600)}
            for H in C.HORIZONS if variant == "main" else (60, 120):
                grid = {name: (scores_on_grid(V, name, H), scores_on_grid(E, name, H)) for name in METHODS}
                for rate in (1, 2, 4, 8) if variant == "main" else (1, 2):
                    block, hits = {}, {}
                    for name, (sv, se) in grid.items():
                        runs = [evaluate(E, TE, e, threshold(V, TV, v, event, rate), event) for v, e in zip(sv, se)]
                        block[name] = average(runs)
                        hits[name] = np.mean([np.bincount(who, r["_found"], len(patients)) for r in runs], axis=0)
                    for name in METHODS[:-1]:
                        boot = (w @ (hits["proposed"] - hits[name])) / (w @ den)
                        diff = block["proposed"]["sensitivity"] - block[name]["sensitivity"]
                        block["proposed"][f"sensitivity_minus_{name}"] = [diff] + np.percentile(boot, [2.5, 97.5]).tolist()
                    res[f"{H} s, {rate}/h"] = block
                    print(variant, event, f"{H} s, {rate}/h", {k: round(v["sensitivity"], 3) for k, v in block.items()}, flush=True)
                if variant == "main":
                    fixed = 60 if event == "light" else 40
                    res[f"{H} s, threshold {fixed}"] = average([evaluate(E, TE, e, fixed, event) for e in grid["proposed"][1]])
            out[f"{variant} | {event}"] = res
    (C.RUNS / "results_alerts.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
