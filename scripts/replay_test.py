"""Offline versus streaming inputs: rebuild the inputs with every record after the forecast origin removed and
compare inputs and forecasts. --negative-control adds a non-causal filter, which the test must detect.

    python scripts/replay_test.py [--cases 20] [--origins 10] [--negative-control]"""
import argparse
import copy
import sys

import _path  # noqa: F401
import numpy as np
import torch
import vitaldb
from scipy.signal import butter, filtfilt

from doa import config as C
from doa.data import channel_stats, cohort, load_arrays, make_split
from doa.features import assemble_channels, valid_windows
from doa.training import DEV, load_network

TRACKS = list(dict.fromkeys([C.TRACK_BIS, C.TRACK_SQI, C.TRACK_EEG] + [t for tracks, _ in C.NUMERIC for t in tracks]))


def open_recording(caseid):
    api = sys.modules[vitaldb.load_case.__module__]
    return vitaldb.VitalFile(f"{api.api_url}/{api.DATASET_VERSION}/{caseid}.vital", TRACKS)


def channels(recording, noncausal):
    c = assemble_channels(recording.to_numpy)
    if noncausal:
        b, a = butter(2, 0.05)
        for k in ("num", "eeg"):
            c[k] = filtfilt(b, a, c[k], axis=0).astype(np.float32) if len(c[k]) > 30 else c[k]
    return c


def withhold_after(recording, end):
    """Copy of the recording without any record of second `end` or later; waveforms are cut sample by sample."""
    r = copy.deepcopy(recording)
    for track in r.trks.values():
        kept = []
        for rec in track.recs:
            if track.type == 1:
                n = int(round(end * track.srate)) - int(np.ceil((rec["dt"] - r.dtstart) * track.srate))
                if n <= 0:
                    continue
                rec["val"] = rec["val"][:n]
                kept.append(rec)
            elif max(0, int(rec["dt"] - r.dtstart)) < end:
                kept.append(rec)
        track.recs = kept
    return r


def main(n_cases, n_origins, noncausal):
    d = load_arrays()
    s = channel_stats(d, make_split(d)[1], "fixed")
    net, ck = load_network("proposed_H60_s0")

    def forecast(c, a, b):
        x = lambda v, m, sd: torch.from_numpy(((v[a:b] - m) / sd)[None].astype(np.float32)).to(DEV)
        with torch.no_grad():
            y, _, q = net(x(c["eeg"], s["emu"], s["esd"]), x(c["num"], s["nmu"], s["nsd"]))
        return torch.cat([y, q.ravel()]).cpu().numpy() * ck["scale"][1] + ck["scale"][0]

    test = sorted(c for c, (_, p) in cohort().items() if p == "test")
    worst_in = worst_out = 0.0
    for c in sorted(np.random.default_rng(1).choice(test, n_cases, replace=False).tolist()):
        rec = open_recording(c)
        full = channels(rec, noncausal)
        starts = valid_windows(full["bis"], full["sqi"])[0]
        for a in starts[np.unique(np.linspace(0, len(starts) - 1, n_origins).round().astype(int))]:
            b = int(a) + C.W
            part = channels(withhold_after(rec, b), noncausal)
            worst_in = max([worst_in] + [float(np.max(np.abs(full[k][a:b] - part[k][a:b]))) for k in ("num", "eeg")])
            worst_out = max(worst_out, float(np.max(np.abs(forecast(full, a, b) - forecast(part, a, b)))))
        print(f"case {c}: largest input difference {worst_in:.3g}, largest forecast difference {worst_out:.3g}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--origins", type=int, default=10)
    ap.add_argument("--negative-control", action="store_true")
    a = ap.parse_args()
    main(a.cases, a.origins, a.negative_control)
