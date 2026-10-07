"""Utility metrics and the summaries the tables are built from."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from sklearn.metrics import f1_score

from dpcardiac.config import N_TIMING_TRIALS


def accuracy(model, x: np.ndarray, y: np.ndarray) -> float:
    """Plain test accuracy."""
    return float((model.predict(x) == np.asarray(y)).mean())


def f1(model, x: np.ndarray, y: np.ndarray) -> float:
    """F1 on the positive class; 0 when the model never predicts it."""
    return float(f1_score(np.asarray(y), model.predict(x), zero_division=0))


def utility_loss_pct(baseline_acc: float, dp_acc: float) -> float:
    """(baseline - DP) / baseline * 100, as defined in Section 3.5."""
    if baseline_acc <= 0:
        raise ValueError("baseline accuracy must be positive.")
    return float((baseline_acc - dp_acc) / baseline_acc * 100.0)


def mean_std(values) -> tuple[float, float]:
    """Population mean and std (ddof=0), the convention used in the tables."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("mean_std of an empty sequence.")
    return float(arr.mean()), float(arr.std())


def time_predict(
    predict: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    n_trials: int = N_TIMING_TRIALS,
) -> tuple[float, float]:
    """Return (ms per sample, samples per second) over n_trials full passes."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    predict(x[:10])  # warm-up
    t0 = time.perf_counter()
    for _ in range(n_trials):
        predict(x)
    dt = (time.perf_counter() - t0) / n_trials
    return float(dt / len(x) * 1000.0), float(len(x) / dt)
