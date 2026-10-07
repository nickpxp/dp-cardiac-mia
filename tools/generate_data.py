"""Regenerate data/ddata.csv: a leakage-audited source table, then CTGAN.

The committed ddata.csv is the exact file the paper's numbers were run on;
this script documents how it was made and lets a reader draw a fresh one.
CTGAN is stochastic and does not take a seed, so a regenerated file gives
new numbers, which is why the original is committed.

Source generator: every feature is derived from a hidden AF propensity with
its own noise, so each is correlated with the label but none is a copy. The
label carries irreducible noise, so the Bayes-optimal accuracy is below 100%
by construction. The audit refuses any feature whose correlation with the
label exceeds 0.9 or that alone yields 90% accuracy, which is the defect
that an earlier seed file had (a prior-diagnosis flag that was a near-copy
of the label).

Usage:
    python tools/generate_data.py --audit-only      # audit the committed file
    python tools/generate_data.py --regenerate      # needs `pip install ctgan`
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dpcardiac.config import (  # noqa: E402
    CTGAN_EPOCHS,
    DATA_PATH,
    FEATURES,
    GEN_SEED,
    N_SOURCE,
    N_SYNTH,
    SIGNAL_STRENGTH,
    SOURCE_PREVALENCE,
    TARGET,
)

CATEGORICAL: tuple[str, ...] = ("af_diagnosis", "af_detected", "signal_quality", "af_context_30min")
LEAK_CORR: float = 0.9
LEAK_SINGLE_FEATURE_ACC: float = 0.9


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_source(
    n: int = N_SOURCE,
    prevalence: float = SOURCE_PREVALENCE,
    signal: float = SIGNAL_STRENGTH,
    seed: int = GEN_SEED,
) -> pd.DataFrame:
    """Shared-latent cardiac table with a calibrated label prevalence."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(62, 11, n), 18, 95)
    chronic = 0.9 * ((age - 62) / 11) + rng.normal(0, 1.0, n)
    af_diagnosis = rng.binomial(1, sigmoid(-0.6 + 1.1 * chronic))
    episode = 0.8 * chronic + rng.normal(0, 1.3, n)
    sq = rng.choice([0, 1, 2], size=n, p=[0.20, 0.30, 0.50])
    qn = np.array([0.9, 0.6, 0.35])[sq]
    ecg = 0.60 + 0.12 * signal * episode + rng.normal(0, 0.10, n) * (1 + qn)
    ppg = 1.20 + 0.18 * signal * episode + rng.normal(0, 0.18, n) * (1 + qn)
    ctx = rng.binomial(1, sigmoid(-0.3 + 1.0 * episode + rng.normal(0, 0.8, n)))
    conf = np.clip(0.70 + 0.10 * sq - 0.10 * qn + rng.normal(0, 0.05, n), 0, 1)
    grid = np.linspace(-6, 6, 2401)
    lin = 1.2 * episode
    b0 = grid[np.argmin(np.abs(np.array([sigmoid(b + lin).mean() for b in grid]) - prevalence))]
    af = rng.binomial(1, sigmoid(b0 + lin))
    return pd.DataFrame({
        "age": age.round(0).astype(int),
        "af_diagnosis": af_diagnosis,
        "ecg_reading": ecg,
        "ppg_reading": ppg,
        "af_detected": af,
        "signal_quality": sq,
        "af_context_30min": ctx,
        "confidence_score": conf,
    })


def audit(df: pd.DataFrame, label: str = "") -> tuple[float, list[str]]:
    """Print per-feature correlations and refuse label leaks."""
    print("=" * 70 + f"\nAUDIT {label}\n" + "=" * 70)
    y = df[TARGET]
    print(f"rows {len(df)} | prevalence {y.mean():.3f}")
    leaks: list[str] = []
    for c in FEATURES:
        r = np.corrcoef(df[c].astype(float), y.astype(float))[0, 1]
        if abs(r) > LEAK_CORR:
            leaks.append(c)
        print(f"  corr {c:18s} {r:+.3f}{'  <-- LEAK' if abs(r) > LEAK_CORR else ''}")
    x_tr, x_te, y_tr, y_te = train_test_split(df[list(FEATURES)], y, test_size=0.3, random_state=0)
    for c in FEATURES:
        a = accuracy_score(
            y_te, DecisionTreeClassifier(max_depth=3, random_state=0).fit(x_tr[[c]], y_tr).predict(x_te[[c]])
        )
        if a > LEAK_SINGLE_FEATURE_ACC and c not in leaks:
            leaks.append(c)
    full = accuracy_score(
        y_te, RandomForestClassifier(n_estimators=200, random_state=0).fit(x_tr, y_tr).predict(x_te)
    )
    print(f"majority {max(y.mean(), 1 - y.mean()):.3f} | full-feature RF {full:.3f} | leaks: {leaks or 'none'}")
    if leaks:
        raise RuntimeError(f"label leakage through {leaks}; refusing to write the file")
    return float(full), leaks


def regenerate(out: Path = DATA_PATH) -> None:
    """Fit CTGAN on the audited source and sample N_SYNTH rows."""
    try:
        from ctgan import CTGAN
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("ctgan is not installed; `pip install ctgan` to regenerate.") from exc
    src = generate_source()
    audit(src, "clean source")
    model = CTGAN(epochs=CTGAN_EPOCHS)
    model.fit(src, list(CATEGORICAL))
    ddata = model.sample(N_SYNTH)
    audit(ddata, "CTGAN output")
    out.parent.mkdir(parents=True, exist_ok=True)
    ddata.to_csv(out, index=False)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit-only", action="store_true", help="audit data/ddata.csv and exit")
    g.add_argument("--regenerate", action="store_true", help="draw a fresh ddata.csv with CTGAN")
    args = ap.parse_args()
    if args.audit_only:
        audit(pd.read_csv(DATA_PATH), f"loaded {DATA_PATH.name}")
    else:
        regenerate()


if __name__ == "__main__":
    main()
