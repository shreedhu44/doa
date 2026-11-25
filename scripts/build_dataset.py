"""Download the cohort from VitalDB, compute per-second features and write the concatenated arrays.

    python scripts/build_dataset.py [--workers 8]
Interrupted downloads resume at the first missing case."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import _path  # noqa: F401
import numpy as np

from doa import config as C
from doa.data import cohort
from doa.features import load_case, valid_windows


def build(caseid):
    out = C.CASES / f"{caseid}.npz"
    if not out.exists():
        c = load_case(caseid)
        starts, y = valid_windows(c["bis"], c["sqi"])
        np.savez(out, starts=starts, y=y, **c)


def assemble(cases):
    parts = {k: [] for k in ("NUM", "EEG", "BIS", "CASE", "win_start", "win_y", "win_case")}
    offset = 0
    for c in cases:
        z = np.load(C.CASES / f"{c}.npz")
        T, n = len(z["bis"]), len(z["starts"])
        for k, v in (("NUM", z["num"]), ("EEG", z["eeg"]), ("BIS", z["bis"]), ("CASE", np.full(T, c, np.int32)),
                     ("win_start", z["starts"] + offset), ("win_y", z["y"]), ("win_case", np.full(n, c, np.int32))):
            parts[k].append(v)
        offset += T
    for k, v in parts.items():
        np.save(C.ARRAYS / f"{k}.npy", np.concatenate(v))
    print(f"{len(cases):,} cases, {offset / 3600:,.1f} h, {sum(map(len, parts['win_start'])):,} windows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cases = sorted(cohort())
    with ThreadPoolExecutor(a.workers) as ex:
        jobs = {ex.submit(build, c): c for c in cases}
        for k, j in enumerate(as_completed(jobs), 1):
            try:
                j.result()
            except Exception as err:
                print(f"case {jobs[j]}: {err}", flush=True)
            if k % 50 == 0:
                print(f"{k}/{len(cases)}", flush=True)
    assemble(cases)
