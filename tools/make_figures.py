"""Regenerate every figure from the CSVs in results/. No model training.

  figures/fig_central_dp_full.png   six-panel summary (utility, MIA, advantage,
                                    architecture accuracy and F1, throughput)
  figures/fig_noise_placement.png   Table 4 as lines, central vs feature-wise
  figures/fig_placement_bars.png    Table 4 as grouped bars with F1 panel
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dpcardiac.config import ARCH_EPSILON, FIGURES_DIR, RESULTS_DIR  # noqa: E402

C_LAP, C_GAU = "#1f4e79", "#b8341f"
ARCH_COLOURS = ("#1f4e79", "#b8341f", "#2e7d32", "#6a1b9a")
EPS_TICKS = ([0.1, 0.5, 1, 2, 5], ["0.1", "0.5", "1", "2", "5"])


def _load(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the experiments first.")
    return pd.read_csv(path)


def _log_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks(EPS_TICKS[0])
    ax.set_xticklabels(EPS_TICKS[1])
    ax.set_xlabel(r"$\epsilon$")
    ax.grid(alpha=0.3)


def fig_full(res: pd.DataFrame, arch: pd.DataFrame, perf: pd.DataFrame, base_acc: float) -> Path:
    lap = res[res.mechanism == "laplace"].sort_values("epsilon")
    gau = res[res.mechanism == "gaussian"].sort_values("epsilon")
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

    for d, c, mk, lab in ((lap, C_LAP, "o", r"Laplace ($\epsilon$-DP)"),
                          (gau, C_GAU, "s", r"Gaussian ($\epsilon,\delta$-DP)")):
        ax[0, 0].plot(d.epsilon, d.acc_mean, marker=mk, color=c, lw=2, ms=6, label=lab)
        ax[0, 0].fill_between(d.epsilon, d.acc_mean - d.acc_std, d.acc_mean + d.acc_std, color=c, alpha=0.15)
    ax[0, 0].axhline(base_acc, ls="--", lw=1.3, color="gray", label=f"Non-private ({base_acc:.3f})")
    _log_axis(ax[0, 0]); ax[0, 0].set_ylabel("Test accuracy"); ax[0, 0].set_title("(a) Utility vs budget"); ax[0, 0].legend(fontsize=8)

    for d, c, mk, lab in ((lap, C_LAP, "o", "Laplace"), (gau, C_GAU, "s", "Gaussian")):
        ax[0, 1].plot(d.epsilon, d.mia_mean, marker=mk, color=c, lw=2, ms=6, label=lab)
        ax[0, 1].fill_between(d.epsilon, d.mia_mean - d.mia_std, d.mia_mean + d.mia_std, color=c, alpha=0.15)
    ax[0, 1].axhline(0.5, ls=":", lw=1.5, color="green", label="Random")
    _log_axis(ax[0, 1]); ax[0, 1].set_ylim(0.45, 0.55); ax[0, 1].set_ylabel("MIA success"); ax[0, 1].set_title("(b) Membership inference"); ax[0, 1].legend(fontsize=8)

    for d, c, mk, lab in ((lap, C_LAP, "o", "Laplace"), (gau, C_GAU, "s", "Gaussian")):
        ax[0, 2].plot(d.epsilon, (d.mia_mean - 0.5) * 100, marker=mk, color=c, lw=2, ms=6, label=lab)
    ax[0, 2].axhline(0, ls=":", lw=1.5, color="green")
    _log_axis(ax[0, 2]); ax[0, 2].set_ylabel("Advantage over random (pp)"); ax[0, 2].set_title("(c) Attack advantage"); ax[0, 2].legend(fontsize=8)

    labels = [f"{r.model}\n{r.mechanism}" for r in arch.itertuples()]
    xpos = np.arange(len(arch))
    ax[1, 0].bar(xpos, arch.dp_acc, color=ARCH_COLOURS[: len(arch)], alpha=0.85)
    ax[1, 0].axhline(base_acc, ls="--", lw=1, color="gray")
    ax[1, 0].set_xticks(xpos); ax[1, 0].set_xticklabels(labels, fontsize=8); ax[1, 0].set_ylabel("Accuracy")
    ax[1, 0].set_title(f"(d) Architecture accuracy (eps={ARCH_EPSILON})"); ax[1, 0].grid(alpha=0.3, axis="y")

    ax[1, 1].bar(xpos, arch.dp_f1, color=ARCH_COLOURS[: len(arch)], alpha=0.85)
    ax[1, 1].set_xticks(xpos); ax[1, 1].set_xticklabels(labels, fontsize=8); ax[1, 1].set_ylabel("F1")
    ax[1, 1].set_title(f"(e) Architecture F1 (eps={ARCH_EPSILON})"); ax[1, 1].grid(alpha=0.3, axis="y")

    dp_perf = perf[perf.model.str.contains("DP,")]
    ax[1, 2].bar(np.arange(len(dp_perf)), dp_perf.throughput, color=("#1f4e79", "#2e7d32", "#6a1b9a"), alpha=0.85)
    ax[1, 2].set_xticks(np.arange(len(dp_perf))); ax[1, 2].set_xticklabels([m.split(" ")[0] for m in dp_perf.model], fontsize=9)
    ax[1, 2].set_ylabel("Samples / sec"); ax[1, 2].set_title("(f) Inference throughput (DP)"); ax[1, 2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIGURES_DIR / "fig_central_dp_full.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


def fig_placement_lines(place: pd.DataFrame, base_acc: float, majority: float) -> Path:
    cl = place[place.mechanism == "laplace"].sort_values("epsilon")
    cg = place[place.mechanism == "gaussian"].sort_values("epsilon")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(cl.epsilon, cl.central_acc, "-o", color="#1f4e79", lw=2.2, label="Central, Laplace")
    ax.plot(cg.epsilon, cg.central_acc, "-s", color="#3b7dd8", lw=2.2, label="Central, Gaussian")
    ax.plot(cl.epsilon, cl.featurewise_acc, "--o", color="#b8341f", lw=2, label="Feature-wise, Laplace")
    ax.plot(cg.epsilon, cg.featurewise_acc, "--s", color="#e07a5f", lw=2, label="Feature-wise, Gaussian")
    ax.axhline(base_acc, ls=":", color="gray", label=f"Non-private ({base_acc:.3f})")
    ax.axhline(majority, ls=":", color="#999999", label=f"Majority ({majority:.3f})")
    _log_axis(ax); ax.set_xlabel(r"Privacy budget $\epsilon$"); ax.set_ylabel("Test accuracy")
    ax.set_title("Noise placement: central preserves utility, feature-wise collapses")
    ax.legend(fontsize=8); plt.tight_layout()
    out = FIGURES_DIR / "fig_noise_placement.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


def fig_placement_bars(place: pd.DataFrame, base_acc: float) -> Path:
    eps = [0.1, 0.5, 1.0, 2.0, 5.0]
    lap = place[place.mechanism == "laplace"].set_index("epsilon")
    gau = place[place.mechanism == "gaussian"].set_index("epsilon")
    acc = {
        "Central DP (Laplace)": [lap.loc[e, "central_acc"] for e in eps],
        "Central DP (Gaussian)": [gau.loc[e, "central_acc"] for e in eps],
        "Input perturbation": [lap.loc[e, "featurewise_acc"] for e in eps],
    }
    f1 = {
        "Central DP (Laplace)": [lap.loc[e, "central_f1"] for e in eps],
        "Central DP (Gaussian)": [gau.loc[e, "central_f1"] for e in eps],
        "Input perturbation": [lap.loc[e, "featurewise_f1"] for e in eps],
    }
    colours = {"Central DP (Laplace)": "#2A6F97", "Central DP (Gaussian)": "#61A5C2", "Input perturbation": "#C9474B"}

    def grouped(ax, data, title, ylabel, baseline=None):
        series = list(data)
        x = np.arange(len(eps)); w = 0.8 / len(series)
        for j, name in enumerate(series):
            vals = np.asarray(data[name], dtype=float)
            pos = x + (j - (len(series) - 1) / 2) * w
            ax.bar(pos, vals, width=w, color=colours[name], label=name, edgecolor="white", linewidth=0.6)
            for p, v in zip(pos, vals):
                ax.annotate(f"{v:.3f}"[1:] if v < 1 else f"{v:.3f}", (p, v), ha="center", va="bottom",
                            fontsize=7, xytext=(0, 1.5), textcoords="offset points")
        if baseline is not None:
            ax.axhline(baseline, ls="--", lw=1.1, color="#444", label=f"Non-private baseline ({baseline:.3f})")
        ax.set_xticks(x); ax.set_xticklabels([f"ε={e}" for e in eps]); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, pad=8); ax.set_ylim(0, 0.85)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1)); ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
    grouped(ax1, acc, "Accuracy", "Accuracy", baseline=base_acc)
    grouped(ax2, f1, "F1 (positive class)", "F1")
    ax2.annotate("Input perturbation: F1 = 0 for ε ≤ 3\n(majority-class only)", xy=(len(eps) - 2 + 0.27, 0.02),
                 xytext=(len(eps) - 3.4, 0.45), fontsize=8, color="#C9474B",
                 arrowprops=dict(arrowstyle="->", color="#C9474B", lw=1))
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Noise placement decides usability: same model, same data", fontsize=12.5, y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = FIGURES_DIR / "fig_placement_bars.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    res = _load("central_dp_nb_results")
    arch = _load("architecture_results")
    perf = _load("performance_results")
    place = _load("noise_placement_results")
    base_acc = float(_load("baseline").loc[0, "accuracy"])
    per_seed = _load("noise_placement_per_seed")
    # majority-class accuracy equals the feature-wise floor; recover it from the data split
    from dpcardiac import make_split
    y = make_split().y_train
    majority = float(max(y.mean(), 1 - y.mean()))
    del per_seed
    for out in (fig_full(res, arch, perf, base_acc),
                fig_placement_lines(place, base_acc, majority),
                fig_placement_bars(place, base_acc)):
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
