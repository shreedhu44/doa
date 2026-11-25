"""Forecasts at every 10-s origin of the complete validation and test recordings, and the
calibration and test quantiles used for the prediction intervals."""
import _path  # noqa: F401
import numpy as np
import torch
import xgboost as xgb

from doa import config as C
from doa.data import calibration_sample, channel_stats, load_arrays, make_split, samples, window_features
from doa.training import DEV, WindowStore, load_network, log, predict


def origins(d, split):
    case = d["CASE"]
    cut = np.r_[0, np.flatnonzero(np.diff(case)) + 1]
    part = {int(c): p for p, k in enumerate(("val", "test")) for c in np.unique(d["win_case"][split[k]])}
    S, K, P = [], [], []
    for a, b in zip(cut, np.r_[cut[1:], len(case)]):
        c = int(case[a])
        if c in part and b - a >= C.W:
            s = np.arange(a, b - C.W + 1, C.STRIDE)
            S.append(s)
            K.append(np.full(len(s), c))
            P.append(np.full(len(s), part[c]))
    return np.concatenate(S), np.concatenate(K), np.concatenate(P)


def dropout_sd(net, store, starts, scale, seed, passes=20):
    """SD of the forecast over stochastic passes with dropout active."""
    torch.manual_seed(seed)
    net.train()
    with torch.no_grad():
        p = [torch.cat([net(*store.batch(b))[0] for b in torch.split(starts, 2048)]).cpu().numpy() for _ in range(passes)]
    return np.std(p, axis=0) * scale[1]


def main():
    d = load_arrays()
    split, cases = make_split(d)
    store = WindowStore(d, channel_stats(d, cases, "fixed"))
    start, case, part = origins(d, split)
    np.savez(C.RUNS / "stream_index.npz", start=start, case=case, part=part)
    to = lambda x: torch.from_numpy(x).to(DEV)
    feats = window_features(d, start)
    for H in C.HORIZONS:
        cal, test = calibration_sample(d, split, H), samples(d, split, H)["test"]
        for name in ("proposed", "gru_bis", "anesnet"):
            stream, P, Q = [], {"cal": [], "test": []}, {"cal": [], "test": []}
            for s in C.SEEDS:
                net, ck = load_network(f"{name}_H{H}_s{s}")
                stream.append(predict(net, store, to(start), ck["scale"], batch=2048)[0].astype(np.float32))
                for k, x in (("cal", cal), ("test", test)):
                    p, q = predict(net, store, to(x[1]), ck["scale"], batch=2048)
                    P[k].append(p)
                    Q[k].append(q)
            np.savez(C.RUNS / f"stream_{name}_H{H}.npz", yp=np.stack(stream))
            if name == "proposed":
                np.savez(C.RUNS / f"interval_proposed_H{H}.npz", cal=np.stack(P["cal"]), test=np.stack(P["test"]),
                         qcal=np.stack(Q["cal"]), qtest=np.stack(Q["test"]), y_cal=cal[2], y=test[2],
                         case=d["win_case"][test[0]], start=test[1])
        P, sd = [], []                                        # Monte-Carlo dropout baseline
        for s in C.SEEDS:
            net, ck = load_network(f"mcdropout_H{H}_s{s}")
            P.append(predict(net, store, to(test[1]), ck["scale"])[0])
            sd.append(dropout_sd(net, store, to(test[1]), ck["scale"], s))
        np.savez(C.RUNS / f"interval_mcdropout_H{H}.npz", test=np.stack(P), sd=np.stack(sd))
        yp = []
        for s in C.SEEDS:
            m = xgb.XGBRegressor()
            m.load_model(C.RUNS / f"xgb_H{H}_s{s}.ubj")
            yp.append(m.predict(feats).astype(np.float32))
        np.savez(C.RUNS / f"stream_xgb_H{H}.npz", yp=np.stack(yp))
        log(f"H={H} done")


if __name__ == "__main__":
    main()
