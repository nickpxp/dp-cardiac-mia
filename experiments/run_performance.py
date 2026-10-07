"""Section 4.3: inference time and throughput, DP vs non-private, per family.

Timing is hardware-dependent and is not a reproducibility target; the CSV
records the environment it was taken on. One representative DP fit per
family (seed 42) is timed over N_TIMING_TRIALS full passes of the test set.
"""
from __future__ import annotations

import platform

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier as SkRF
from sklearn.linear_model import LogisticRegression as SkLR
from sklearn.naive_bayes import GaussianNB as SkNB

from _common import show, write_results
from dpcardiac import (
    fit_dp_gaussian_nb,
    fit_dp_logistic,
    fit_dp_random_forest,
    make_split,
    time_predict,
)
from dpcardiac.config import ARCH_EPSILON, DELTA, N_TIMING_TRIALS, RF_N_ESTIMATORS


def main() -> None:
    split = make_split()
    seed = 42
    models = {
        "NB": (
            SkNB().fit(split.x_train, split.y_train),
            fit_dp_gaussian_nb(split.x_train, split.y_train, ARCH_EPSILON, "laplace", delta=DELTA, seed=seed),
        ),
        "RF": (
            SkRF(n_estimators=RF_N_ESTIMATORS, random_state=0).fit(split.x_train, split.y_train),
            fit_dp_random_forest(split.x_train, split.y_train, ARCH_EPSILON, seed=seed),
        ),
        "LR": (
            SkLR(max_iter=1000).fit(split.x_train, split.y_train),
            fit_dp_logistic(split.x_train, split.y_train, ARCH_EPSILON, "gaussian", delta=DELTA, seed=seed),
        ),
    }
    env = f"{platform.machine()} | Python {platform.python_version()} | sklearn {sklearn.__version__}"
    rows = []
    for name, (base, dp) in models.items():
        base_ms, base_thr = time_predict(base.predict, split.x_test, N_TIMING_TRIALS)
        dp_ms, dp_thr = time_predict(dp.predict, split.x_test, N_TIMING_TRIALS)
        rows.append({"model": f"{name} (non-DP)", "ms_per_sample": base_ms,
                     "throughput": base_thr, "overhead_%": 0.0, "environment": env})
        rows.append({"model": f"{name} (DP, eps={ARCH_EPSILON})", "ms_per_sample": dp_ms,
                     "throughput": dp_thr, "overhead_%": (dp_ms - base_ms) / base_ms * 100.0,
                     "environment": env})
    res = pd.DataFrame(rows)
    path = write_results(res, "performance_results")
    show(res.drop(columns="environment"), decimals=6)
    print(f"\n{env}\nwrote {path}")


if __name__ == "__main__":
    main()
