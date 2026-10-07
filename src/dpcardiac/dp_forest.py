"""Central-DP random forest through diffprivlib (Fletcher and Islam 2017).

diffprivlib.models.RandomForestClassifier builds random decision trees on
the public feature box and labels leaves with the permute-and-flip mechanism,
so the budget is spent on the leaf labels only. It appears in Table 3 with a
single mechanism because the library offers no Gaussian variant.

The bounds are the [0, 1] box of the scaled features, which is public, so no
budget leaks through them (diffprivlib warns if bounds are omitted).
"""
from __future__ import annotations

import numpy as np

from dpcardiac.config import RF_N_ESTIMATORS


def fit_dp_random_forest(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    seed: int | None = None,
    n_estimators: int = RF_N_ESTIMATORS,
):
    """diffprivlib RandomForestClassifier on the [0, 1] feature box."""
    from diffprivlib.models import RandomForestClassifier as DPRF  # lazy import

    x = np.asarray(x, dtype=np.float64)
    if x.min() < 0.0 or x.max() > 1.0:
        raise ValueError("x must be scaled to [0, 1] before fitting.")
    d = x.shape[1]
    return DPRF(
        n_estimators=n_estimators,
        epsilon=epsilon,
        bounds=(np.zeros(d), np.ones(d)),
        classes=np.unique(y),
        random_state=seed,
    ).fit(x, np.asarray(y))
