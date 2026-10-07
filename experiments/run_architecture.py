"""Table 3: NB (both mechanisms), RF and LR under central DP at eps = 1.0.

Utility loss is measured against each family's own non-private accuracy on
the same scaled split. Seeds 42-46 for every configuration. Writes
results/architecture_per_seed.csv and results/architecture_results.csv.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SkRF
from sklearn.linear_model import LogisticRegression as SkLR
from sklearn.naive_bayes import GaussianNB as SkNB

from _common import show, write_results
from dpcardiac import (
    accuracy,
    f1,
    fit_dp_gaussian_nb,
    fit_dp_logistic,
    fit_dp_random_forest,
    make_split,
    mean_std,
    rule_based_mia,
    utility_loss_pct,
)
from dpcardiac.config import ARCH_EPSILON, DELTA, RF_N_ESTIMATORS, SEEDS

CONFIGS: tuple[tuple[str, str], ...] = (
    ("NB", "laplace"),
    ("NB", "gaussian"),
    ("RF", "diffprivlib"),
    ("LR", "output-pert (Gaussian)"),
)


def fit_dp(name: str, mechanism: str, split, eps: float, seed: int):
    """Dispatch to the DP estimator for one family."""
    if name == "NB":
        return fit_dp_gaussian_nb(split.x_train, split.y_train, eps, mechanism, delta=DELTA, seed=seed)
    if name == "RF":
        return fit_dp_random_forest(split.x_train, split.y_train, eps, seed=seed)
    if name == "LR":
        return fit_dp_logistic(split.x_train, split.y_train, eps, "gaussian", delta=DELTA, seed=seed)
    raise ValueError(f"unknown model {name!r}")


def main() -> None:
    split = make_split()
    baselines = {
        "NB": SkNB().fit(split.x_train, split.y_train),
        "RF": SkRF(n_estimators=RF_N_ESTIMATORS, random_state=0).fit(split.x_train, split.y_train),
        "LR": SkLR(max_iter=1000).fit(split.x_train, split.y_train),
    }
    base_acc = {k: accuracy(m, split.x_test, split.y_test) for k, m in baselines.items()}
    print("Non-private accuracy by family:", {k: round(v, 4) for k, v in base_acc.items()})

    per_seed = []
    for name, mech in CONFIGS:
        for seed in SEEDS:
            m = fit_dp(name, mech, split, ARCH_EPSILON, seed)
            per_seed.append({
                "model": name, "mechanism": mech, "epsilon": ARCH_EPSILON, "seed": seed,
                "accuracy": accuracy(m, split.x_test, split.y_test),
                "f1": f1(m, split.x_test, split.y_test),
                "mia": rule_based_mia(m, split.x_train, split.y_train, split.x_test, split.y_test),
            })
    ps = pd.DataFrame(per_seed)
    write_results(ps, "architecture_per_seed")

    rows = []
    for (name, mech), g in ps.groupby(["model", "mechanism"], sort=False):
        acc_m, acc_s = mean_std(g["accuracy"])
        rows.append({
            "model": name, "mechanism": mech,
            "np_acc": base_acc[name],
            "dp_acc": acc_m, "dp_acc_std": acc_s,
            "dp_f1": mean_std(g["f1"])[0],
            "util_loss_%": utility_loss_pct(base_acc[name], acc_m),
            "mia": mean_std(g["mia"])[0],
        })
    res = pd.DataFrame(rows)
    path = write_results(res, "architecture_results")
    show(res)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
