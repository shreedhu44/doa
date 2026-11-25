"""Per-second features of one VitalDB case."""
import numpy as np

from . import config as C

EEG_BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 47)]   # delta, theta, alpha, beta, gamma


def eeg_features(eeg, fs=C.EEG_FS):
    """Relative band powers, 95% spectral edge frequency and log power of every second: (T, 7)."""
    n = int(fs)
    T = len(eeg) // n
    x = np.nan_to_num(eeg[:T * n], nan=0.0).reshape(T, n)
    x = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(x, axis=1)) ** 2
    out = np.zeros((T, C.N_EEG), np.float32)
    keep = (f >= 0.5) & (f <= 47)
    total = psd[:, keep].sum(axis=1) + 1e-8
    for i, (lo, hi) in enumerate(EEG_BANDS):
        out[:, i] = psd[:, (f >= lo) & (f < hi)].sum(axis=1) / total
    cum = np.cumsum(psd[:, keep], axis=1) / (psd[:, keep].sum(axis=1, keepdims=True) + 1e-8)
    out[:, 5] = f[keep][(cum >= 0.95).argmax(axis=1)]
    out[:, 6] = np.log(total)
    return out


def forward_fill(a):
    """Carry the last value forward along the time axis; leading gaps become zero."""
    a = a.copy()
    for j in range(a.shape[1]):
        last = 0.0
        for i in range(len(a)):
            if np.isnan(a[i, j]):
                a[i, j] = last
            else:
                last = a[i, j]
    return np.nan_to_num(a)


def assemble_channels(read):
    """Aligned 1 Hz arrays of a recording. `read(tracks, interval)` returns a (rows, tracks) array."""
    arr = read([C.TRACK_BIS, C.TRACK_SQI] + [tracks[0] for tracks, _ in C.NUMERIC], 1)
    num = np.full((len(arr), len(C.NUMERIC)), np.nan, np.float32)
    for j, (tracks, _) in enumerate(C.NUMERIC):
        x = arr[:, 2 + j]
        if np.all(np.isnan(x)) and len(tracks) > 1:
            alt = read([tracks[1]], 1)
            if len(alt) >= len(arr):
                x = alt[:len(arr), 0]
        num[:, j] = x
    eeg = eeg_features(read([C.TRACK_EEG], 1.0 / C.EEG_FS)[:, 0])
    T = min(len(arr), len(eeg))
    bis, sqi = arr[:T, 0].astype(np.float32), arr[:T, 1].astype(np.float32)
    hist = bis.copy()
    hist[(hist < C.BIS_RANGE[0]) | (hist > C.BIS_RANGE[1])] = np.nan
    num = forward_fill(np.concatenate([hist[:, None], num[:T]], axis=1)).astype(np.float32)
    return {"bis": bis, "sqi": sqi, "num": num, "eeg": forward_fill(eeg[:T]).astype(np.float32)}


def load_case(caseid):
    import vitaldb
    return assemble_channels(lambda tracks, dt: vitaldb.load_case(caseid, tracks, dt))


def valid_windows(bis, sqi):
    """Window starts whose target (BIS at W + 60 s) passes the range and signal-quality rules."""
    starts, y = [], []
    for s in range(0, len(bis) - C.W - C.BASE_H, C.STRIDE):
        t = s + C.W + C.BASE_H
        if C.BIS_RANGE[0] <= bis[t] <= C.BIS_RANGE[1] and sqi[t] >= C.SQI_MIN:
            starts.append(s)
            y.append(bis[t])
    return np.asarray(starts, np.int64), np.asarray(y, np.float32)
