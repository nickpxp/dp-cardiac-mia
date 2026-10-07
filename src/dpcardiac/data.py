"""Load the synthetic cardiac dataset and build the scaled train/test split.

Features are min-max scaled to [0, 1] with bounds taken from the training
split. The bounds are treated as public (Section 3.2), which is what makes
every released sufficient statistic have sensitivity 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dpcardiac.config import (
    DATA_PATH,
    FEATURES,
    SPLIT_SEED,
    TARGET,
    TEST_SIZE,
)


@dataclass(frozen=True)
class Split:
    """Scaled train/test arrays plus the public scaling bounds."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    @property
    def n_features(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.y_train)


def load_dataframe(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read ddata.csv and fail loudly if the schema is wrong."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run tools/generate_data.py or place ddata.csv "
            "in data/."
        )
    df = pd.read_csv(path)
    missing = [c for c in (*FEATURES, TARGET) if c not in df.columns]
    if missing:
        raise ValueError(f"ddata.csv is missing columns: {missing}")
    if not set(np.unique(df[TARGET])) <= {0, 1}:
        raise ValueError(f"{TARGET} must be binary 0/1.")
    return df


def minmax_scale(
    x: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """Scale to [0, 1] with public bounds; constant features map to 0."""
    width = np.where((hi - lo) == 0, 1.0, hi - lo)
    return np.clip((np.asarray(x, dtype=np.float64) - lo) / width, 0.0, 1.0)


def make_split(
    df: pd.DataFrame | None = None,
    test_size: float = TEST_SIZE,
    seed: int = SPLIT_SEED,
) -> Split:
    """80/20 split with seed 42, then min-max scaling from the train bounds."""
    df = load_dataframe() if df is None else df
    x = df[list(FEATURES)].astype(np.float64)
    y = df[TARGET].to_numpy()
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )
    lo = x_tr.min(axis=0).to_numpy()
    hi = x_tr.max(axis=0).to_numpy()
    return Split(
        x_train=minmax_scale(x_tr.to_numpy(), lo, hi),
        x_test=minmax_scale(x_te.to_numpy(), lo, hi),
        y_train=np.asarray(y_tr),
        y_test=np.asarray(y_te),
        lo=lo,
        hi=hi,
    )
