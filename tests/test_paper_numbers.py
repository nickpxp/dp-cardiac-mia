"""Regression test: the committed data must reproduce the paper's headline cells.

Runs the Table 1 and Table 2 cells at eps = 1.0 from data/ddata.csv with
seeds 42-46. Takes a few seconds; skipped when the data file is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dpcardiac import accuracy, fit_dp_gaussian_nb, fit_gaussian_nb, make_split  # noqa: E402
from dpcardiac.config import DATA_PATH, SEEDS  # noqa: E402

pytestmark = pytest.mark.skipif(not DATA_PATH.exists(), reason="data/ddata.csv not present")


@pytest.fixture(scope="module")
def split():
    return make_split()


def test_split_shape_and_scaling(split):
    assert split.x_train.shape == (80_000, 7)
    assert split.x_test.shape == (20_000, 7)
    assert split.x_train.min() >= 0.0 and split.x_train.max() <= 1.0
    assert split.y_train.mean() == pytest.approx(0.372, abs=0.002)


def test_nonprivate_ceiling(split):
    base = fit_gaussian_nb(split.x_train, split.y_train)
    assert accuracy(base, split.x_test, split.y_test) == pytest.approx(0.7555, abs=5e-5)


def test_table2_cells_at_eps_1(split):
    lap = np.array([
        accuracy(fit_dp_gaussian_nb(split.x_train, split.y_train, 1.0, "laplace", seed=s), split.x_test, split.y_test)
        for s in SEEDS
    ])
    gau = np.array([
        accuracy(fit_dp_gaussian_nb(split.x_train, split.y_train, 1.0, "gaussian", seed=s), split.x_test, split.y_test)
        for s in SEEDS
    ])
    assert lap.mean() == pytest.approx(0.7554, abs=5e-5)
    assert gau.mean() == pytest.approx(0.7421, abs=5e-5)
    assert gau.std() == pytest.approx(0.0170, abs=5e-5)
    t, p = stats.ttest_rel(lap, gau)
    assert t == pytest.approx(1.598, abs=5e-4)
    assert p == pytest.approx(0.185, abs=5e-4)
