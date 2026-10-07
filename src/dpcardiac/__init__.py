"""Differential privacy against membership inference on a cardiac classifier.

Code for Pierrelouis and Liu (2026), "Protecting healthcare machine learning
from membership inference: An evaluation of differential privacy", Issues in
Information Systems 27(1):181-195. https://doi.org/10.48009/1_iis_2026_113
"""
from dpcardiac.attacks import attack_advantage_pp, rule_based_mia
from dpcardiac.data import Split, load_dataframe, make_split, minmax_scale
from dpcardiac.dp_forest import fit_dp_random_forest
from dpcardiac.dp_logistic import DPLogisticModel, fit_dp_logistic
from dpcardiac.dp_naive_bayes import (
    MECHANISMS,
    GaussianNBModel,
    fit_dp_gaussian_nb,
    fit_gaussian_nb,
    n_released_statistics,
    noise_scale,
)
from dpcardiac.input_perturbation import fit_featurewise_nb, perturb_features
from dpcardiac.metrics import (
    accuracy,
    f1,
    mean_std,
    time_predict,
    utility_loss_pct,
)

__version__ = "1.0.0"

__all__ = [
    "DPLogisticModel",
    "GaussianNBModel",
    "MECHANISMS",
    "Split",
    "accuracy",
    "attack_advantage_pp",
    "f1",
    "fit_dp_gaussian_nb",
    "fit_dp_logistic",
    "fit_dp_random_forest",
    "fit_featurewise_nb",
    "fit_gaussian_nb",
    "load_dataframe",
    "make_split",
    "mean_std",
    "minmax_scale",
    "n_released_statistics",
    "noise_scale",
    "perturb_features",
    "rule_based_mia",
    "time_predict",
    "utility_loss_pct",
]
