"""Feature-wise (input) perturbation: the local-DP arm of Table 4.

Noise is added to every record's [0, 1] features before training a plain
Gaussian NB. The per-record budget eps is split evenly over the d features
(sequential composition), so each feature gets eps / d with sensitivity 1.
Only the placement differs from the central arm; the model, data, seeds and
budget are the same, which is what makes the two columns of Table 4
comparable.
"""
from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import GaussianNB

from dpcardiac.config import DELTA
from dpcardiac.dp_naive_bayes import MECHANISMS


def perturb_features(
    x: np.ndarray,
    epsilon: float,
    mechanism: str,
    delta: float = DELTA,
    seed: int | None = None,
) -> np.ndarray:
    """Return a noised copy of x, clipped back to [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be a 2-D array.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}.")
    d = x.shape[1]
    eps_j = epsilon / d
    rng = np.random.default_rng(seed)
    if mechanism == "laplace":
        noise = rng.laplace(0.0, 1.0 / eps_j, size=x.shape)
    else:
        sigma = np.sqrt(2.0 * np.log(1.25 / (delta / d))) / eps_j
        noise = rng.normal(0.0, sigma, size=x.shape)
    return np.clip(x + noise, 0.0, 1.0)


def fit_featurewise_nb(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    mechanism: str,
    delta: float = DELTA,
    seed: int | None = None,
) -> GaussianNB:
    """Noise the training features, then fit scikit-learn's GaussianNB."""
    x_noisy = perturb_features(x, epsilon, mechanism, delta=delta, seed=seed)
    return GaussianNB().fit(x_noisy, np.asarray(y))
