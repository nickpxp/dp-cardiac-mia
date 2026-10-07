"""Shared runner plumbing: path setup, CSV writing, console tables."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dpcardiac.config import RESULTS_DIR  # noqa: E402


def write_results(df: pd.DataFrame, name: str) -> Path:
    """Write results/<name>.csv and return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def show(df: pd.DataFrame, decimals: int = 4) -> None:
    """Print a frame with floats rounded for reading."""
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(df.round(decimals).to_string(index=False))
