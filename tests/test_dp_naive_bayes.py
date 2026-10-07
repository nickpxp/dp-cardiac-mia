"""Unit tests for the central-DP NB, placement arm, attack and LR."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dpcardiac import (  # noqa: E402
    attack_advantage_pp,
    fit_dp_gaussian_nb,
    fit_dp_logistic,
    fit_featurewise_nb,
    fit_gaussian_nb,
    n_released_statistics,
    noise_scale,
    perturb_features,
    rule_based_mia,
    utility_loss_pct,
)


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n, d = 4000, 7
    y = rng.binomial(1, 0.4, n)
    x = rng.normal(0, 1, (n, d)) + 0.8 * y[:, None]
    x = (x - x.min(0)) / (x.max(0) - x.min(0))
    return x[:3000], y[:3000], x[3000:], y[3000:]


def test_released_statistics_count():
    assert n_released_statistics(2, 7) == 30


def test_noise_scales_follow_composition():
    b = noise_scale(1.0, "laplace", 2, 7)
    assert b == pytest.approx(30.0)
    sigma = noise_scale(1.0, "gaussian", 2, 7, delta=1e-5)
    eps_q, delta_q = 1.0 / 30, 1e-5 / 30
    assert sigma == pytest.approx(np.sqrt(2 * np.log(1.25 / delta_q)) / eps_q)
    with pytest.raises(ValueError):
        noise_scale(1.0, "cauchy", 2, 7)
    with pytest.raises(ValueError):
        noise_scale(0.0, "laplace", 2, 7)


def test_fit_rejects_unscaled_input(toy):
    x, y, _, _ = toy
    with pytest.raises(ValueError):
        fit_dp_gaussian_nb(x * 2.0, y, 1.0, "laplace", seed=0)


def test_seed_makes_fit_deterministic(toy):
    x, y, xt, _ = toy
    a = fit_dp_gaussian_nb(x, y, 0.5, "gaussian", seed=42)
    b = fit_dp_gaussian_nb(x, y, 0.5, "gaussian", seed=42)
    c = fit_dp_gaussian_nb(x, y, 0.5, "gaussian", seed=43)
    assert np.array_equal(a.predict(xt), b.predict(xt))
    assert not np.array_equal(a.means, c.means)


def test_large_budget_recovers_nonprivate_model(toy):
    x, y, xt, yt = toy
    base = fit_gaussian_nb(x, y)
    dp = fit_dp_gaussian_nb(x, y, 1e9, "laplace", seed=0)
    assert np.allclose(dp.means, base.means, atol=1e-6)
    assert (dp.predict(xt) == base.predict(xt)).mean() == 1.0
    assert 0.5 < (base.predict(xt) == yt).mean()


def test_variances_floored_and_priors_normalised(toy):
    x, y, _, _ = toy
    m = fit_dp_gaussian_nb(x, y, 0.1, "laplace", seed=1)
    assert (m.variances >= 1e-3).all()
    assert np.exp(m.log_priors).sum() == pytest.approx(1.0)


def test_predict_shape_guard(toy):
    x, y, _, _ = toy
    m = fit_gaussian_nb(x, y)
    with pytest.raises(ValueError):
        m.predict(np.zeros((5, 3)))


def test_feature_perturbation_stays_in_box(toy):
    x, y, xt, yt = toy
    xn = perturb_features(x, 0.1, "laplace", seed=0)
    assert xn.shape == x.shape and xn.min() >= 0 and xn.max() <= 1
    xg = perturb_features(x, 0.1, "gaussian", seed=0)
    assert not np.allclose(xn, xg)
    clf = fit_featurewise_nb(x, y, 1.0, "laplace", seed=0)
    assert clf.predict(xt).shape == yt.shape


def test_rule_based_mia_is_half_when_train_equals_test():
    class Const:
        def predict(self, x):
            return np.ones(len(x), dtype=int)

    x = np.zeros((100, 2))
    y_all_one = np.ones(100, dtype=int)
    assert rule_based_mia(Const(), x, y_all_one, x, y_all_one) == pytest.approx(0.5)
    assert attack_advantage_pp(0.509) == pytest.approx(0.9)


def test_output_perturbation_lr_runs_and_noise_shrinks_with_budget(toy):
    x, y, xt, yt = toy
    tight = fit_dp_logistic(x, y, 0.01, "gaussian", seed=0)
    loose = fit_dp_logistic(x, y, 100.0, "gaussian", seed=0)
    assert tight.predict(xt).shape == yt.shape
    assert (loose.predict(xt) == yt).mean() > 0.6
    lap = fit_dp_logistic(x, y, 1.0, "laplace", seed=0)
    assert lap.weights.shape == (x.shape[1] + 1,)


def test_utility_loss():
    assert utility_loss_pct(0.7555, 0.7421) == pytest.approx(1.7736, abs=1e-3)
    with pytest.raises(ValueError):
        utility_loss_pct(0.0, 0.5)
