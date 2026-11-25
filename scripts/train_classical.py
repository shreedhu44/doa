"""PK-PD Emax model and classical regressors on the same test windows."""
import _path  # noqa: F401
import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from doa import config as C
from doa.data import load_arrays, make_split, samples, subsample, targets, window_features
from train_xgboost import save


def emax(x, e0, em, ce50, gamma, kappa):
    ce = np.maximum(x[0] + kappa * x[1], 1e-6)
    return e0 - em * ce ** gamma / (ce50 ** gamma + ce ** gamma)


def concentrations(d, start):
    """Propofol and remifentanil effect-site concentrations at the forecast origin."""
    return [np.asarray(d["NUM"][start + C.W - 1, c], float) for c in (7, 9)]


MODELS = {
    "adaboost": AdaBoostRegressor(n_estimators=150, random_state=0),
    "knn": KNeighborsRegressor(n_neighbors=15, n_jobs=8),
    "ridge": Ridge(alpha=1.0),
    "mlp_features": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=120, random_state=0),
    "histgb": HistGradientBoostingRegressor(max_iter=300, random_state=0),
    "random_forest": RandomForestRegressor(n_estimators=200, n_jobs=8, random_state=0),
    "extra_trees": ExtraTreesRegressor(n_estimators=200, n_jobs=8, random_state=0),
    "svr": SVR(kernel="rbf", C=10, gamma="scale"),
}


def main():
    d = load_arrays()
    split, _ = make_split(d)
    for H in C.HORIZONS:
        _, start, y = targets(d, subsample(split["train"], 200_000, 0), H)
        par, _ = curve_fit(emax, concentrations(d, start), y, p0=[90, 60, 3, 1.4, 1],
                           bounds=([50, 20, 0.5, 0.5, 0], [100, 90, 12, 6, 5]), maxfev=20_000)
        S = samples(d, split, H)
        save(f"pkpd_H{H}", d, S, emax(concentrations(d, S["test"][1]), *par))
    S = samples(d, split, 60, train=120_000)
    F = {p: window_features(d, S[p][1]) for p in ("train", "test")}
    some = np.random.RandomState(0).choice(len(F["train"]), 25_000, replace=False)
    for name, m in MODELS.items():
        rows = some if name == "svr" else slice(None)
        m.fit(F["train"][rows], S["train"][2][rows])
        save(f"{name}_H60", d, S, m.predict(F["test"]))


if __name__ == "__main__":
    main()
