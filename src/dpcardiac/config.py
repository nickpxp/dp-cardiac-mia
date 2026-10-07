"""Experimental constants shared by every runner.

Every value here is stated in the paper (Section 3). Changing one changes
the tables, so the runners import from here rather than redefining them.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the repository root)
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_PATH: Path = REPO_ROOT / "data" / "ddata.csv"
RESULTS_DIR: Path = REPO_ROOT / "results"
FIGURES_DIR: Path = REPO_ROOT / "figures"

# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------
FEATURES: tuple[str, ...] = (
    "age",
    "af_diagnosis",
    "ecg_reading",
    "ppg_reading",
    "signal_quality",
    "af_context_30min",
    "confidence_score",
)
TARGET: str = "af_detected"
TEST_SIZE: float = 0.2
SPLIT_SEED: int = 42

# ---------------------------------------------------------------------------
# Privacy sweep
# ---------------------------------------------------------------------------
EPSILONS: tuple[float, ...] = (0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
ARCH_EPSILON: float = 1.0          # budget for the architecture comparison
DELTA: float = 1e-5                # Gaussian mechanism; delta << 1/n for n = 80,000
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)   # five matched runs
N_RUNS: int = len(SEEDS)

# ---------------------------------------------------------------------------
# Model and attack settings
# ---------------------------------------------------------------------------
VAR_FLOOR: float = 1e-3            # floor on reconstructed NB variances
MIA_SAMPLE: int = 2000             # records per side in the balanced attack set
RF_N_ESTIMATORS: int = 100
LR_LAMBDA: float = 1e-3            # L2 regularisation for output-perturbation LR
N_TIMING_TRIALS: int = 50          # inference-time trials (paper, Section 4.3)

# ---------------------------------------------------------------------------
# Synthetic-source generator (tools/generate_data.py)
# ---------------------------------------------------------------------------
N_SOURCE: int = 20_000
SOURCE_PREVALENCE: float = 0.30
SIGNAL_STRENGTH: float = 1.0
GEN_SEED: int = 7
CTGAN_EPOCHS: int = 300
N_SYNTH: int = 100_000
