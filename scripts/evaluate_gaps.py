"""Accuracy and interval coverage at 60 s when the last part of a stream is lost (writes runs/results_gaps.json)."""
import json

import _path  # noqa: F401
import numpy as np
import torch
import xgboost as xgb

from doa import config as C
from doa.data import channel_stats, load_arrays, make_split, samples, summary_features
from doa.training import DEV, WindowStore, fixed_gap, load_network, log, predict

H = 60
STREAMS = {"BIS": ("bis",), "EEG": ("eeg",), "BIS+EEG": ("bis", "eeg"), "drugs": ("drug",)}
GAPS = (10, 30, 60, 120, 300)


def main():
    d = load_arrays()
    split, cases = make_split(d)
    store = WindowStore(d, channel_stats(d, cases, "fixed"))
    _, start, y = samples(d, split, H)["test"]
    start, y = torch.from_numpy(start).to(DEV), y.astype(float)
    offsets = json.loads((C.RUNS / "results_intervals.json").read_text())[str(H)]["offsets"]
    nets = {k: [load_network(f"{k}_H{H}_s{s}") for s in C.SEEDS] for k in ("proposed", "gru_nogaps")}
    trees = []
    for s in C.SEEDS:
        m = xgb.XGBRegressor()
        m.load_model(C.RUNS / f"xgb_H{H}_s{s}.ubj")
        trees.append(m)
    rows = []
    for name, groups, gap in [("none", (), 0)] + [(k, g, t) for k, g in STREAMS.items() for t in GAPS]:
        F, last = [], []
        for b in torch.split(start, 4000):
            e, n = store.batch(b, fixed_gap(len(b), groups, gap))
            n = (n * store.s["nsd"] + store.s["nmu"]).cpu().numpy()
            e = (e * store.s["esd"] + store.s["emu"]).cpu().numpy()
            F.append(summary_features(n, e))
            last.append(n[:, -1, 0])
        F = np.concatenate(F)
        row = {"lost": name, "gap": gap, "persistence": float(np.mean(np.abs(np.concatenate(last) - y))),
               "xgb": float(np.mean([np.mean(np.abs(m.predict(F) - y)) for m in trees]))}
        for k, group in nets.items():
            mae, cover, width = [], [], []
            for (net, ck), offset in zip(group, offsets):
                p, q = predict(net, store, start, ck["scale"], groups, gap, batch=2048)
                lo, hi = np.clip(q.min(1) - offset, 0, 100), np.clip(q.max(1) + offset, 0, 100)
                mae.append(np.mean(np.abs(p - y)))
                cover.append(np.mean((y >= lo) & (y <= hi)))
                width.append(np.mean(hi - lo))
            row[k] = {"MAE": float(np.mean(mae))}
            if k == "proposed":                                # the offsets were calibrated for this model
                row[k].update(coverage=float(np.mean(cover)), width=float(np.mean(width)))
        rows.append(row)
        log(row)
    (C.RUNS / "results_gaps.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
