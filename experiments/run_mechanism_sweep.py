"""Table 1: Laplace and Gaussian central-DP NB across the privacy budget.

Writes results/central_dp_nb_per_seed.csv (one row per mechanism, epsilon,
seed) and results/central_dp_nb_results.csv (means and stds), plus the
non-private ceiling in results/baseline.csv.
"""
from __future__ import annotations

import pandas as pd

from _common import show, write_results
from dpcardiac import (
    accuracy,
    f1,
    fit_dp_gaussian_nb,
    fit_gaussian_nb,
    make_split,
    mean_std,
    rule_based_mia,
)
from dpcardiac.config import DELTA, EPSILONS, SEEDS
from dpcardiac.dp_naive_bayes import MECHANISMS


def main() -> None:
    split = make_split()
    base = fit_gaussian_nb(split.x_train, split.y_train)
    base_acc = accuracy(base, split.x_test, split.y_test)
    base_f1 = f1(base, split.x_test, split.y_test)
    base_mia = rule_based_mia(base, split.x_train, split.y_train, split.x_test, split.y_test)
    write_results(
        pd.DataFrame([{"model": "NB", "accuracy": base_acc, "f1": base_f1, "mia": base_mia}]),
        "baseline",
    )
    print(f"Non-private NB ceiling: acc {base_acc:.4f} | F1 {base_f1:.4f} | MIA {base_mia:.4f}\n")

    per_seed = []
    for mech in MECHANISMS:
        for eps in EPSILONS:
            for seed in SEEDS:
                m = fit_dp_gaussian_nb(split.x_train, split.y_train, eps, mech, delta=DELTA, seed=seed)
                per_seed.append({
                    "mechanism": mech,
                    "epsilon": eps,
                    "seed": seed,
                    "accuracy": accuracy(m, split.x_test, split.y_test),
                    "f1": f1(m, split.x_test, split.y_test),
                    "mia": rule_based_mia(m, split.x_train, split.y_train, split.x_test, split.y_test),
                })
    ps = pd.DataFrame(per_seed)
    write_results(ps, "central_dp_nb_per_seed")

    rows = []
    for (mech, eps), g in ps.groupby(["mechanism", "epsilon"], sort=False):
        acc_m, acc_s = mean_std(g["accuracy"])
        f1_m, f1_s = mean_std(g["f1"])
        mia_m, mia_s = mean_std(g["mia"])
        rows.append({
            "mechanism": mech, "epsilon": eps,
            "acc_mean": acc_m, "acc_std": acc_s,
            "f1_mean": f1_m, "f1_std": f1_s,
            "mia_mean": mia_m, "mia_std": mia_s,
        })
    res = pd.DataFrame(rows)
    path = write_results(res, "central_dp_nb_results")
    show(res)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
