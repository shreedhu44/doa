"""CPU time of one real-time forecast: features of the newest second, window assembly and one network pass."""
import time

import _path  # noqa: F401
import numpy as np
import torch

from doa import config as C
from doa.features import eeg_features
from doa.models import build


def main(iterations=300, warmup=30):
    ck = torch.load(C.RUNS / "proposed_H60_s0.pt", map_location="cpu", weights_only=True)
    net = build(ck["cfg"])
    net.load_state_dict(ck["state"])
    net.eval()
    z = np.load(C.RUNS / "stats_fixed.npz")
    s = {k: z[k].astype(np.float32) for k in z.files}
    x = lambda v, m, sd: torch.from_numpy(((v - m) / sd)[None].astype(np.float32))
    rng = np.random.default_rng(0)
    for threads in sorted({1, torch.get_num_threads()}):
        torch.set_num_threads(threads)
        num = rng.normal(50, 10, (C.W, C.N_NUM)).astype(np.float32)
        eeg = eeg_features(rng.normal(0, 20, C.W * C.EEG_FS))
        times = []
        for i in range(warmup + iterations):
            raw, values = rng.normal(0, 20, C.EEG_FS), rng.normal(50, 10, C.N_NUM).astype(np.float32)
            t = time.perf_counter()
            num = np.concatenate([num[1:], values[None]])
            eeg = np.concatenate([eeg[1:], eeg_features(raw)])
            with torch.no_grad():
                net(x(eeg, s["emu"], s["esd"]), x(num, s["nmu"], s["nsd"]))
            if i >= warmup:
                times.append(1000 * (time.perf_counter() - t))
        print(f"{threads} thread(s): median {np.median(times):.1f} ms, 99th percentile {np.percentile(times, 99):.1f} ms")


if __name__ == "__main__":
    main()
