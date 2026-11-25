"""Accuracy metrics, patient-clustered bootstrap and conformal quantiles."""
import numpy as np

from . import config as C

BOOT = 5000


def band(v):
    return np.digitize(v, C.BANDS)


def scores(y, p):
    return {"MAE": float(np.mean(np.abs(p - y))), "RMSE": float(np.sqrt(np.mean((p - y) ** 2))),
            "r": float(np.corrcoef(p, y)[0, 1]), "band_acc": float(np.mean(band(p) == band(y)))}


def mean_scores(y, P):
    """Mean and SD of the metrics over the rows (seeds) of P."""
    s = [scores(y, p) for p in P]
    return {k: {"mean": float(np.mean([x[k] for x in s])), "sd": float(np.std([x[k] for x in s], ddof=1)) if len(s) > 1 else 0.0}
            for k in s[0]}


def patient_weights(n_patients, B=BOOT, seed=0):
    return np.random.default_rng(seed).multinomial(n_patients, np.full(n_patients, 1 / n_patients), size=B).astype(float)


def paired_difference(y, patient, A, B_, mask=None, B=BOOT, seed=0):
    """Mean |A - y| - mean |B - y| with a 95% CI from a joint bootstrap of patients and training seeds."""
    u, inv = np.unique(patient, return_inverse=True)
    if mask is not None:
        y, A, B_, inv = y[mask], A[:, mask], B_[:, mask], inv[mask]
    n = np.bincount(inv, minlength=len(u)).astype(float)
    err = lambda P: np.stack([np.bincount(inv, np.abs(p - y), len(u)) for p in P])
    SA, SB = err(A), err(B_)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=B).astype(float)
    wa = rng.multinomial(len(A), np.full(len(A), 1 / len(A)), size=B) / len(A)
    wb = rng.multinomial(len(B_), np.full(len(B_), 1 / len(B_)), size=B) / len(B_)
    agg = lambda ws, S: np.einsum("bp,bs,sp->b", w, ws, S, optimize=True)
    boot = (agg(wa, SA) - agg(wb, SB)) / (w @ n)
    per_patient = (SA.mean(0) - SB.mean(0))[n > 0] / n[n > 0]
    return {"diff": float((SA.mean(0).sum() - SB.mean(0).sum()) / n.sum()), "ci": np.percentile(boot, [2.5, 97.5]).tolist(),
            "share_patients_A_better": float(np.mean(per_patient < 0)), "_boot": boot}


def growth(short, long):
    """Change of a paired difference between two horizons."""
    return {"diff": long["diff"] - short["diff"], "ci": np.percentile(long["_boot"] - short["_boot"], [2.5, 97.5]).tolist()}


def public(o):
    """Drop bootstrap arrays before writing JSON."""
    if isinstance(o, dict):
        return {k: public(v) for k, v in o.items() if not str(k).startswith("_")}
    return o


def conformal_quantile(nonconformity, alpha=0.05):
    n = len(nonconformity)
    return float(np.quantile(nonconformity, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
