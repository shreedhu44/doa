"""Cohort, patient-grouped splits and window access."""
import csv

import numpy as np

from . import config as C


def cohort():
    """{caseid: (subjectid, partition)} of the working cohort."""
    with open(C.SPLIT_CSV, newline="") as f:
        return {int(r["caseid"]): (int(r["subjectid"]), r["partition"]) for r in csv.DictReader(f)}


def subjects():
    return {c: s for c, (s, _) in cohort().items()}


def load_arrays():
    """Concatenated per-second arrays (NUM and EEG stay memory-mapped) and the window index."""
    d = {k: np.load(C.ARRAYS / f"{k}.npy", mmap_mode="r")
         for k in ("NUM", "EEG", "BIS", "CASE", "win_start", "win_case", "win_y")}
    for k in ("BIS", "CASE", "win_start", "win_case", "win_y"):
        d[k] = np.asarray(d[k])
    return d


def make_split(d, fold=None):
    """Window indices of train/val/test and the training cases; no patient is in two partitions.

    fold=None gives the fixed partition of data/cohort_split.csv, fold=0..4 one cross-validation fold."""
    table = cohort()
    if fold is None:
        sets = {p: [c for c, (_, q) in table.items() if q == p] for p in ("train", "val", "test")}
    else:
        s = sorted({v[0] for v in table.values()})
        np.random.RandomState(0).shuffle(s)
        test = set(np.array_split(s, C.FOLDS)[fold].tolist())
        rest = [x for x in s if x not in test]
        val = set(rest[:max(1, int(0.15 * len(rest)))])
        part = lambda x: "test" if x in test else "val" if x in val else "train"
        sets = {p: [c for c, (x, _) in table.items() if part(x) == p] for p in ("train", "val", "test")}
    return {p: np.where(np.isin(d["win_case"], v))[0] for p, v in sets.items()}, sets["train"]


def subsample(idx, k, seed):
    if k is None or len(idx) <= k:
        return idx
    return np.sort(np.random.RandomState(seed).choice(idx, k, replace=False))


def targets(d, idx, H):
    """Windows whose target at horizon H is a valid BIS of the same recording: (indices, starts, targets)."""
    start = d["win_start"][idx]
    t = np.minimum(start + C.W + H, len(d["BIS"]) - 1)
    y = d["BIS"][t]
    ok = (start + C.W + H < len(d["BIS"])) & (y >= 0) & (y <= 100) & (d["CASE"][t] == d["win_case"][idx])
    return idx[ok], start[ok], y[ok].astype(np.float32)


def samples(d, split, H, train=60_000, val=25_000, test=60_000):
    """Fixed window samples shared by all models at a horizon."""
    return {p: targets(d, subsample(split[p], n, seed), H)
            for p, n, seed in (("train", train, 1), ("val", val, 2), ("test", test, 3))}


def calibration_sample(d, split, H):
    return targets(d, subsample(split["val"], 25_000, 12), H)


def channel_stats(d, train_cases, key):
    """Mean and SD of every channel over the training cases (2 million random seconds)."""
    f = C.RUNS / f"stats_{key}.npz"
    if not f.exists():
        rows = np.where(np.isin(d["CASE"], train_cases))[0]
        if len(rows) > 2_000_000:
            rows = np.random.RandomState(0).choice(rows, 2_000_000, replace=False)
        rows = np.sort(rows)
        n, e = d["NUM"][rows], d["EEG"][rows]
        np.savez(f, nmu=n.mean(0, dtype=np.float64), nsd=n.std(0, dtype=np.float64) + 1e-6,
                 emu=e.mean(0, dtype=np.float64), esd=e.std(0, dtype=np.float64) + 1e-6)
    z = np.load(f)
    return {k: z[k].astype(np.float32) for k in ("nmu", "nsd", "emu", "esd")}


def summary_features(num, eeg):
    """54 features of raw windows (n, W, channels) for the tree-based and classical models."""
    return np.concatenate([num.mean(1), num.std(1), num[:, -1], num[:, -1] - num[:, 0],
                           eeg.mean(1), eeg[:, -1]], axis=1).astype(np.float32)


def window_features(d, starts, chunk=10_000):
    out = []
    for c in range(0, len(starts), chunk):
        ix = starts[c:c + chunk][:, None] + np.arange(C.W)
        out.append(summary_features(d["NUM"][ix], d["EEG"][ix]))
    return np.concatenate(out)


def phase_labels(d, case, starts, H):
    """Induction, maintenance or emergence of every target (used to stratify the evaluation only).

    Maintenance runs from the first to the last period of at least 60 s with BIS <= 60."""
    target = starts + C.W + H
    out = np.full(len(target), "unclassified", dtype=object)
    wc = d["win_case"]
    cut = np.flatnonzero(np.diff(wc)) + 1
    span = {int(wc[a]): (a, b) for a, b in zip(np.r_[0, cut], np.r_[cut, len(wc)])}
    for c in np.unique(case):
        a, b = span[int(c)]
        t = d["win_start"][a:b] + C.W + C.BASE_H
        low = d["win_y"][a:b] <= 60
        contiguous = np.r_[False, np.diff(t) == C.STRIDE]
        run = np.zeros(len(low), int)
        for i in range(len(low)):
            if low[i]:
                run[i] = run[i - 1] + 1 if i and contiguous[i] and low[i - 1] else 1
        end = np.flatnonzero(run >= 6)
        if len(end):
            m = case == c
            first, last = t[end[0] - 5], t[end[-1]]
            out[m] = np.where(target[m] < first, "induction", np.where(target[m] > last, "emergence", "maintenance"))
    return out
