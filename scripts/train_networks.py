"""Tune and train the proposed forecaster, its input variants and the network baselines.

    python scripts/train_networks.py --stage main    # fixed split, five seeds
    python scripts/train_networks.py --stage cv      # five-fold patient-grouped cross-validation"""
import argparse
import json

import _path  # noqa: F401

from doa import config as C
from doa.data import channel_stats, load_arrays, make_split
from doa.training import WindowStore, log, train

GRID = [{"width": w, "lr": lr, "dropout": dr} for w in (48, 96) for lr in (5e-4, 1e-3, 2e-3) for dr in (0.0, 0.15)]
ANESNET_GRID = [{"depth": dep, "mult": m, "lr": lr} for dep in (4, 7) for m in (1, 4) for lr in (5e-4, 1e-3, 2e-3)]


def tune(store, d, split, name, kind, inputs, grid):
    """Setting with the lowest validation MAE at every horizon (seed 0)."""
    f = C.RUNS / f"tuned_{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    best = {}
    for H in C.HORIZONS:
        runs = []
        for k, g in enumerate(grid):
            cfg = {"kind": kind, "inputs": inputs, "H": H, "seed": 0, **g}
            runs.append((train(store, d, split, cfg, f"tune_{name}_H{H}_{k}")["val_MAE"], k))
        best[str(H)] = grid[min(runs)[1]]
        log(f"{name} H={H}: {best[str(H)]}")
    f.write_text(json.dumps(best, indent=1))
    return best


def main_stage(d):
    split, cases = make_split(d)
    store = WindowStore(d, channel_stats(d, cases, "fixed"))
    I = C.INPUTS
    gru = tune(store, d, split, "gru", "gru", I["proposed"], GRID)
    bis = tune(store, d, split, "bis", "gru", I["bis"], GRID)
    tf = tune(store, d, split, "transformer", "transformer", I["all"], GRID)
    anes = tune(store, d, split, "anesnet", "anesnet", I["all"], ANESNET_GRID)
    for seed in C.SEEDS:
        for H in C.HORIZONS:
            h = str(H)

            def run(name, kind, inputs, setting, **extra):
                cfg = {"kind": kind, "inputs": inputs, "H": H, "seed": seed, **setting, **extra}
                train(store, d, split, cfg, f"{name}_H{H}_s{seed}")
            run("proposed", "gru", I["proposed"], gru[h], gaps=True)
            run("gru_nogaps", "gru", I["proposed"], gru[h])
            run("gru_bis", "gru", I["bis"], bis[h])
            for k in ("bis_eeg", "bis_drug", "all"):
                run(f"gru_{k}", "gru", I[k], gru[h])
            run("transformer", "transformer", I["all"], tf[h])
            run("mlp", "mlp", I["all"], tf[h])
            run("anesnet", "anesnet", I["all"], anes[h])
            run("mcdropout", "transformer", I["all"], {"width": 48, "lr": 1e-3, "dropout": 0.15})


def cv_stage(d):
    gru, bis, anes = (json.loads((C.RUNS / f"tuned_{k}.json").read_text()) for k in ("gru", "bis", "anesnet"))
    store = None
    for f in range(C.FOLDS):
        split, cases = make_split(d, f)
        stats = channel_stats(d, cases, f"fold{f}")
        if store is None:
            store = WindowStore(d, stats)
        store.set_stats(stats)
        for H in C.HORIZONS:
            for name, kind, inputs, setting, gaps in (("proposed", "gru", "proposed", gru, True), ("gru_bis", "gru", "bis", bis, False),
                                                      ("anesnet", "anesnet", "all", anes, False)):
                cfg = {"kind": kind, "inputs": C.INPUTS[inputs], "H": H, "seed": 0, **setting[str(H)], "gaps": gaps}
                train(store, d, split, cfg, f"cv_{name}_f{f}_H{H}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["main", "cv"], default="main")
    {"main": main_stage, "cv": cv_stage}[ap.parse_args().stage](load_arrays())
