"""Black-box membership inference with the rule-based (loss-threshold) attack.

The attacker sees predictions only. A record is declared a member when the
model classifies it correctly, which is the 0-1 loss thresholded at its
decision boundary and is what ART ships as
MembershipInferenceBlackBoxRuleBased (Yeom et al. 2018, CSF, the "baseline"
attack). It is implemented directly here so that it runs against any object
with a predict method, including the sufficient-statistics NB.

Attack success is balanced accuracy on an equal number of members (first
MIA_SAMPLE training rows) and non-members (first MIA_SAMPLE test rows):

    success = (train accuracy + test error) / 2

which equals 0.5 for a model whose train and test accuracies coincide. The
reported number is attack accuracy, not advantage; the paper's "advantage"
is success - 0.5 in percentage points.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from dpcardiac.config import MIA_SAMPLE


class Predictor(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...


def rule_based_mia(
    model: Predictor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n: int = MIA_SAMPLE,
) -> float:
    """Balanced attack accuracy of the correct-classification rule."""
    n = min(n, len(x_train), len(x_test))
    if n == 0:
        raise ValueError("Attack needs at least one member and one non-member.")
    tr_correct = (model.predict(x_train[:n]) == np.asarray(y_train)[:n]).mean()
    te_correct = (model.predict(x_test[:n]) == np.asarray(y_test)[:n]).mean()
    return float((tr_correct + (1.0 - te_correct)) / 2.0)


def attack_advantage_pp(success: float) -> float:
    """Percentage points above the 0.5 coin-flip baseline."""
    return float((success - 0.5) * 100.0)
