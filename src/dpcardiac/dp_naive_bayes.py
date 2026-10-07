"""Central-DP Gaussian Naive Bayes via privatised sufficient statistics.

Method: Vaidya, Shafiq, Basu and Hong (2013), "Differentially private naive
Bayes classification", IEEE/WIC/ACM WI-IAT, pp. 571-576. Per class the model
releases one count, one sum per feature and one sum of squares per feature,
so k(1 + 2d) statistics in total. Each has sensitivity 1 once the features sit
in [0, 1]. The budget is split equally by basic sequential composition
(Dwork and Roth 2014, Theorem 3.16): eps_q = eps / k(1 + 2d), and for the
Gaussian arm delta_q = delta / k(1 + 2d).

Both mechanisms run through the same code path. The only difference is the
noise distribution drawn on the statistics:

    Laplace  : b     = 1 / eps_q                                 (eps-DP)
    Gaussian : sigma = sqrt(2 ln(1.25 / delta_q)) / eps_q        ((eps, delta)-DP)

The Gaussian calibration is the classical one (Dwork and Roth 2014,
Theorem A.1) and needs eps_q < 1, which holds across the sweep because the
split divides eps by 30.

The RNG draw order (count, then d sums, then d sums of squares, per class in
sorted class order) is part of the reproducibility contract: changing it
changes every seeded number in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dpcardiac.config import DELTA, VAR_FLOOR

MECHANISMS: tuple[str, ...] = ("laplace", "gaussian")


@dataclass(frozen=True)
class GaussianNBModel:
    """Fitted parameters; predict() needs nothing else."""

    classes: np.ndarray
    log_priors: np.ndarray
    means: np.ndarray      # (k, d)
    variances: np.ndarray  # (k, d)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Argmax of log prior + summed Gaussian log-likelihood."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.means.shape[1]:
            raise ValueError(
                f"Expected (n, {self.means.shape[1]}) input, got {x.shape}."
            )
        ll = np.empty((x.shape[0], len(self.classes)))
        for ci in range(len(self.classes)):
            mu, var = self.means[ci], self.variances[ci]
            ll[:, ci] = self.log_priors[ci] - 0.5 * (
                np.log(2.0 * np.pi * var) + (x - mu) ** 2 / var
            ).sum(axis=1)
        return self.classes[np.argmax(ll, axis=1)]


def n_released_statistics(n_classes: int, n_features: int) -> int:
    """k(1 + 2d): one count, d sums, d sums of squares per class."""
    return int(n_classes * (1 + 2 * n_features))


def noise_scale(
    epsilon: float,
    mechanism: str,
    n_classes: int,
    n_features: int,
    delta: float = DELTA,
    sensitivity: float = 1.0,
) -> float:
    """Per-statistic Laplace scale b or Gaussian sigma after composition."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    n_q = n_released_statistics(n_classes, n_features)
    eps_q = epsilon / n_q
    if mechanism == "laplace":
        return sensitivity / eps_q
    if mechanism == "gaussian":
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1) for the Gaussian mechanism.")
        delta_q = delta / n_q
        return sensitivity * np.sqrt(2.0 * np.log(1.25 / delta_q)) / eps_q
    raise ValueError(f"mechanism must be one of {MECHANISMS}, got {mechanism!r}.")


def fit_dp_gaussian_nb(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    mechanism: str,
    delta: float = DELTA,
    seed: int | None = None,
    var_floor: float = VAR_FLOOR,
) -> GaussianNBModel:
    """Privatise the NB sufficient statistics and rebuild the classifier.

    `x` must already be scaled to [0, 1]; the sensitivity-1 claim depends on
    it, so the function checks rather than trusts.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if x.ndim != 2:
        raise ValueError("x must be a 2-D array.")
    if x.min() < 0.0 or x.max() > 1.0:
        raise ValueError("x must be scaled to [0, 1] before fitting.")
    classes = np.unique(y)
    k, d = len(classes), x.shape[1]
    scale = noise_scale(epsilon, mechanism, k, d, delta=delta)
    rng = np.random.default_rng(seed)

    def noise(size: int) -> np.ndarray:
        if mechanism == "laplace":
            return rng.laplace(0.0, scale, size=size)
        return rng.normal(0.0, scale, size=size)

    counts = np.zeros(k)
    means = np.zeros((k, d))
    variances = np.zeros((k, d))
    for ci, c in enumerate(classes):
        xc = x[y == c]
        n_noisy = max(len(xc) + float(noise(1)[0]), 1.0)
        s = xc.sum(axis=0) + noise(d)
        q = (xc ** 2).sum(axis=0) + noise(d)
        mu = np.clip(s / n_noisy, 0.0, 1.0)
        var = np.maximum(q / n_noisy - mu ** 2, var_floor)
        counts[ci], means[ci], variances[ci] = n_noisy, mu, var

    return GaussianNBModel(
        classes=classes,
        log_priors=np.log(counts / counts.sum()),
        means=means,
        variances=variances,
    )


def fit_gaussian_nb(
    x: np.ndarray, y: np.ndarray, var_floor: float = VAR_FLOOR
) -> GaussianNBModel:
    """Non-private reference through the same predict path (no noise)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    k, d = len(classes), x.shape[1]
    counts = np.zeros(k)
    means = np.zeros((k, d))
    variances = np.zeros((k, d))
    for ci, c in enumerate(classes):
        xc = x[y == c]
        n = float(len(xc))
        mu = xc.sum(axis=0) / n
        counts[ci] = n
        means[ci] = mu
        variances[ci] = np.maximum((xc ** 2).sum(axis=0) / n - mu ** 2, var_floor)
    return GaussianNBModel(
        classes=classes,
        log_priors=np.log(counts / counts.sum()),
        means=means,
        variances=variances,
    )
