"""Table 4: central vs feature-wise noise placement, NB, same budget and seeds.

Writes results/noise_placement_per_seed.csv and
results/noise_placement_results.csv. The central column repeats the Table 1
numbers by construction (same seeds, same code path); the feature-wise
column is the local-DP arm from dpcardiac.input_perturbation.
"""
from __future__ import annotations

import pandas as pd

from _common import show, write_results
from dpcardiac import (
    accuracy,
    f1,
    fit_dp_gaussian_nb,
    fit_featurewise_nb,
    make_split,
    mean_std,
    rule_based_mia,
)
from dpcardiac.config import DELTA, EPSILONS, SEEDS
from dpcardiac.dp_naive_bayes import MECHANISMS


def main() -> None:
    split = make_split()
    majority = max(split.y_train.mean(), 1 - split.y_train.mean())
    print(f"majority-class accuracy on train: {majority:.4f}\n")

    per_seed = []
    for mech in MECHANISMS:
        for eps in EPSILONS:
            for seed in SEEDS:
                central = fit_dp_gaussian_nb(split.x_train, split.y_train, eps, mech, delta=DELTA, seed=seed)
                local = fit_featurewise_nb(split.x_train, split.y_train, eps, mech, delta=DELTA, seed=seed)
                row = {"mechanism": mech, "epsilon": eps, "seed": seed}
                for tag, m in (("central", central), ("featurewise", local)):
                    row[f"{tag}_acc"] = accuracy(m, split.x_test, split.y_test)
                    row[f"{tag}_f1"] = f1(m, split.x_test, split.y_test)
                    row[f"{tag}_mia"] = rule_based_mia(
                        m, split.x_train, split.y_train, split.x_test, split.y_test
                    )
                per_seed.append(row)
    ps = pd.DataFrame(per_seed)
    write_results(ps, "noise_placement_per_seed")

    rows = []
    for (mech, eps), g in ps.groupby(["mechanism", "epsilon"], sort=False):
        rows.append({
            "mechanism": mech, "epsilon": eps,
            "central_acc": mean_std(g["central_acc"])[0],
            "featurewise_acc": mean_std(g["featurewise_acc"])[0],
            "central_f1": mean_std(g["central_f1"])[0],
            "featurewise_f1": mean_std(g["featurewise_f1"])[0],
            "central_mia": mean_std(g["central_mia"])[0],
            "featurewise_mia": mean_std(g["featurewise_mia"])[0],
        })
    res = pd.DataFrame(rows)
    path = write_results(res, "noise_placement_results")
    show(res)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
