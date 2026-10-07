"""Regenerate every table and prose number in the paper from results/*.csv.

No model is trained here. Each block names the paper table or section it
feeds, and the cell rounding matches the manuscript (four decimals for
accuracy and F1, three for MIA, two for utility loss, three for t and p).
Run after experiments/run_all.py, or on the committed CSVs as they are.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dpcardiac.config import ARCH_EPSILON, RESULTS_DIR  # noqa: E402

TABLE1_EPS = (0.1, 0.5, 1.0, 2.0, 5.0)


def load(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run experiments/run_all.py first.")
    return pd.read_csv(path)


def header(title: str) -> None:
    print("\n" + "=" * 84 + f"\n{title}\n" + "=" * 84)


def table1(res: pd.DataFrame, base: pd.Series) -> None:
    header("Table 1. Laplace and Gaussian mechanism performance (NB, 5 runs)")
    lap = res[res.mechanism == "laplace"].set_index("epsilon")
    gau = res[res.mechanism == "gaussian"].set_index("epsilon")
    print(f"{'eps':>5} {'Lap acc':>8} {'Lap F1':>8} {'Gau acc':>8} {'Gau F1':>8}")
    for e in TABLE1_EPS:
        print(f"{e:>5} {lap.loc[e, 'acc_mean']:>8.4f} {lap.loc[e, 'f1_mean']:>8.4f} "
              f"{gau.loc[e, 'acc_mean']:>8.4f} {gau.loc[e, 'f1_mean']:>8.4f}")
    print(f"Baseline (non-DP): accuracy {base.accuracy:.4f}, F1 {base.f1:.4f}")


def table2(res: pd.DataFrame, tt: pd.DataFrame, base: pd.Series) -> None:
    header(f"Table 2. Mechanism statistical summary at eps = {ARCH_EPSILON}")
    lap = res[(res.mechanism == "laplace") & (res.epsilon == ARCH_EPSILON)].iloc[0]
    gau = res[(res.mechanism == "gaussian") & (res.epsilon == ARCH_EPSILON)].iloc[0]
    t = tt[tt.epsilon == ARCH_EPSILON].iloc[0]
    loss_l = (base.accuracy - lap.acc_mean) / base.accuracy * 100
    loss_g = (base.accuracy - gau.acc_mean) / base.accuracy * 100
    rows = [
        ("Accuracy", f"{lap.acc_mean:.4f} ± {lap.acc_std:.4f}", f"{gau.acc_mean:.4f} ± {gau.acc_std:.4f}",
         f"{abs(lap.acc_mean - gau.acc_mean):.4f}"),
        ("F1-score", f"{lap.f1_mean:.4f}", f"{gau.f1_mean:.4f}", f"{abs(lap.f1_mean - gau.f1_mean):.4f}"),
        ("Utility loss (%)", f"{loss_l:.2f}", f"{loss_g:.2f}", f"{abs(loss_l - loss_g):.2f}"),
        ("MIA success", f"{lap.mia_mean:.3f}", f"{gau.mia_mean:.3f}", f"{abs(lap.mia_mean - gau.mia_mean):.3f}"),
        ("Paired t", "", "", f"{t.t:.3f}"),
        ("p-value", "", "", f"{t.p:.3f}"),
        ("Significant (0.05)", "", "", "Yes" if t["significant_0.05"] else "No"),
    ]
    print(f"{'Metric':<20} {'Laplace':>18} {'Gaussian':>18} {'|diff|':>8}")
    for r in rows:
        print(f"{r[0]:<20} {r[1]:>18} {r[2]:>18} {r[3]:>8}")
    print(f"Cohen d_z = {t.cohen_dz:.3f}; prediction consistency: "
          f"Laplace {t.laplace_consistency:.4f}, Gaussian {t.gaussian_consistency:.4f}")


def ttest_text(tt: pd.DataFrame) -> None:
    header("Section 4.1 prose: paired t-test at every epsilon")
    for _, r in tt.iterrows():
        flag = "significant" if r["significant_0.05"] else "ns"
        print(f"eps={r.epsilon:<4} t={r.t:+.3f} p={r.p:.3f} ({flag}), favours "
              f"{'Laplace' if r.mean_diff > 0 else 'Gaussian'} by {abs(r.mean_diff):.4f}")


def table3(arch: pd.DataFrame) -> None:
    header(f"Table 3. Model performance comparison at eps = {ARCH_EPSILON}")
    print(f"{'Model':<5} {'Mechanism':<24} {'np acc':>7} {'acc':>7} {'F1':>7} {'loss %':>7} {'MIA':>6}")
    for _, r in arch.iterrows():
        print(f"{r.model:<5} {r.mechanism:<24} {r.np_acc:>7.4f} {r.dp_acc:>7.4f} {r.dp_f1:>7.4f} "
              f"{r['util_loss_%']:>7.2f} {r.mia:>6.3f}")


def table4(place: pd.DataFrame) -> None:
    header("Table 4. Noise placement: test accuracy by eps")
    lap = place[place.mechanism == "laplace"].set_index("epsilon")
    gau = place[place.mechanism == "gaussian"].set_index("epsilon")
    print(f"{'eps':>5} {'C-Lap':>8} {'C-Gau':>8} {'FW-Lap':>8} {'FW-Gau':>8}")
    for e in lap.index:
        print(f"{e:>5} {lap.loc[e, 'central_acc']:>8.4f} {gau.loc[e, 'central_acc']:>8.4f} "
              f"{lap.loc[e, 'featurewise_acc']:>8.4f} {gau.loc[e, 'featurewise_acc']:>8.4f}")
    print(f"Feature-wise F1 (Laplace): "
          + ", ".join(f"{e}:{lap.loc[e, 'featurewise_f1']:.3f}" for e in lap.index))


def mia_text(res: pd.DataFrame, base: pd.Series) -> None:
    header("Section 4.2 prose: membership inference range")
    print(f"Non-DP NB attack success: {base.mia:.3f}")
    print(f"DP NB attack success range: {res.mia_mean.min():.3f} to {res.mia_mean.max():.3f} "
          f"(max advantage {(res.mia_mean.max() - 0.5) * 100:.1f} pp)")


def perf_text(perf: pd.DataFrame) -> None:
    header("Section 4.3 prose: inference overhead (hardware-dependent)")
    for _, r in perf.iterrows():
        print(f"{r.model:<20} {r.ms_per_sample * 1000:>8.3f} us/sample  {r.throughput:>14,.0f} samples/s  "
              f"overhead {r['overhead_%']:+.1f}%")
    print(f"environment: {perf.environment.iloc[0]}")


def main() -> None:
    res = load("central_dp_nb_results")
    base = load("baseline").iloc[0]
    tt = load("mechanism_ttest")
    arch = load("architecture_results")
    place = load("noise_placement_results")
    perf = load("performance_results")
    table1(res, base)
    table2(res, tt, base)
    ttest_text(tt)
    table3(arch)
    table4(place)
    mia_text(res, base)
    perf_text(perf)


if __name__ == "__main__":
    main()
