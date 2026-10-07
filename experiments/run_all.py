"""Run every experiment in paper order, then the figures and paper numbers."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

STEPS = (
    HERE / "run_mechanism_sweep.py",
    HERE / "run_mechanism_ttest.py",
    HERE / "run_architecture.py",
    HERE / "run_noise_placement.py",
    HERE / "run_performance.py",
    ROOT / "tools" / "make_figures.py",
    ROOT / "tools" / "run_paper_numbers.py",
)


def main() -> None:
    for step in STEPS:
        print("\n" + "=" * 84 + f"\n{step.relative_to(ROOT)}\n" + "=" * 84)
        sys.argv = [str(step)]
        runpy.run_path(str(step), run_name="__main__")


if __name__ == "__main__":
    main()
