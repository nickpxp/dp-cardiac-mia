"""Output-perturbation logistic regression (Chaudhuri, Monteleoni and Sarwate 2011).

"Differentially private empirical risk minimization", JMLR 12:1069-1109,
Algorithm 1. For L2-regularised logistic regression with ||x||_2 <= 1 the
L2 sensitivity of the minimiser is 2 / (n * lambda). The non-private
minimiser is found with scikit-learn and noise is added to the weight
vector once. Implemented directly because diffprivlib's LogisticRegression
(objective perturbation) does not import against scikit-learn 1.8.

Rows are augmented with a constant 1 for the intercept and divided by the
largest augmented row norm in the training set, so the norm bound holds for
every training row. The bound is a training-set statistic; the paper treats
it as public in the same way as the min-max scaling bounds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from dpcardiac.config import DELTA, LR_LAMBDA


@dataclass(frozen=True)
class DPLogisticModel:
    """Noised weight vector plus the public row-norm used for scaling."""

    weights: np.ndarray
    row_norm: float
    classes: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Sign of the noised linear score."""
        xa = augment(x, self.row_norm)
        return np.where(xa @ self.weights >= 0.0, self.classes[1], self.classes[0])


def augment(x: np.ndarray, row_norm: float) -> np.ndarray:
    """Append a 1 for the intercept and divide by the public row norm."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be a 2-D array.")
    return np.hstack([x, np.ones((len(x), 1))]) / row_norm


def max_row_norm(x: np.ndarray) -> float:
    """Largest L2 norm of the augmented rows (the scaling constant)."""
    xa = np.hstack([np.asarray(x, dtype=np.float64), np.ones((len(x), 1))])
    return float(np.linalg.norm(xa, axis=1).max())


def fit_dp_logistic(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    mechanism: str = "gaussian",
    lam: float = LR_LAMBDA,
    delta: float = DELTA,
    seed: int | None = None,
    row_norm: float | None = None,
) -> DPLogisticModel:
    """Fit regularised LR, then perturb the weights once."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("Output-perturbation LR is implemented for binary labels.")
    row_norm = max_row_norm(x) if row_norm is None else float(row_norm)
    xa = augment(x, row_norm)
    n, d = xa.shape
    w = (
        LogisticRegression(C=1.0 / (n * lam), fit_intercept=False, max_iter=2000)
        .fit(xa, y)
        .coef_.ravel()
        .copy()
    )
    sensitivity = 2.0 / (n * lam)
    rng = np.random.default_rng(seed)
    if mechanism == "gaussian":
        sigma = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
        b = rng.normal(0.0, sigma, size=d)
    elif mechanism == "laplace":
        # Spherical Laplace: Gamma(d, Delta/eps) radius on a uniform direction.
        g = rng.normal(size=d)
        g /= np.linalg.norm(g)
        b = rng.gamma(d, sensitivity / epsilon) * g
    else:
        raise ValueError("mechanism must be 'gaussian' or 'laplace'.")
    return DPLogisticModel(weights=w + b, row_norm=row_norm, classes=classes)
