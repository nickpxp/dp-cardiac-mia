"""Table 2: matched-seed paired t-test, Laplace vs Gaussian NB, at every epsilon.

Both arms use seeds 42-46, so the five accuracy differences are paired. The
paper reports the eps = 1.0 row (t = 1.598, p = 0.185) and cites eps = 0.5
(t = 3.18, p = 0.033) in the text; all seven are written to
results/mechanism_ttest.csv. Prediction consistency (agreement with the
non-private model on the test set) is written alongside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from _common import show, write_results
from dpcardiac import accuracy, fit_dp_gaussian_nb, fit_gaussian_nb, make_split
from dpcardiac.config import DELTA, EPSILONS, SEEDS
from dpcardiac.dp_naive_bayes import MECHANISMS


def main() -> None:
    split = make_split()
    base = fit_gaussian_nb(split.x_train, split.y_train)
    base_pred = base.predict(split.x_test)

    rows = []
    for eps in EPSILONS:
        acc = {m: [] for m in MECHANISMS}
        agree = {m: [] for m in MECHANISMS}
        for seed in SEEDS:
            for mech in MECHANISMS:
                m = fit_dp_gaussian_nb(split.x_train, split.y_train, eps, mech, delta=DELTA, seed=seed)
                acc[mech].append(accuracy(m, split.x_test, split.y_test))
                agree[mech].append(float((m.predict(split.x_test) == base_pred).mean()))
        lap, gau = np.asarray(acc["laplace"]), np.asarray(acc["gaussian"])
        diff = lap - gau
        t, p = stats.ttest_rel(lap, gau)
        sd = diff.std(ddof=1)
        rows.append({
            "epsilon": eps,
            "laplace_acc": lap.mean(), "laplace_std": lap.std(),
            "gaussian_acc": gau.mean(), "gaussian_std": gau.std(),
            "mean_diff": diff.mean(),
            "t": float(t), "p": float(p),
            "cohen_dz": float(diff.mean() / sd) if sd > 0 else float("nan"),
            "significant_0.05": bool(p < 0.05),
            "laplace_consistency": float(np.mean(agree["laplace"])),
            "gaussian_consistency": float(np.mean(agree["gaussian"])),
        })
    res = pd.DataFrame(rows)
    path = write_results(res, "mechanism_ttest")
    show(res)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
