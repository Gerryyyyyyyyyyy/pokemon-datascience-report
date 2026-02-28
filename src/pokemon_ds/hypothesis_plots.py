from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Utility function to ensure output directory exists before saving plots
def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

# Save a boxplot comparing total_stats for mono-type vs dual-type Pokemon
def save_mono_vs_dual_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    _ensure_out_dir(out_dir)

    if "is_dual_type" not in df.columns or "total_stats" not in df.columns:
        raise ValueError("Required columns missing: 'is_dual_type', 'total_stats'")
    
    # Extract total_stats for mono and dual type groups
    mono = df.loc[~df["is_dual_type"], "total_stats"].dropna().to_numpy()
    dual = df.loc[df["is_dual_type"], "total_stats"].dropna().to_numpy()
   
    # Create and save the boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([mono, dual], tick_labels=["Mono-type", "Dual-type"])
    ax.set_title("Total stats: Mono-type vs Dual-type")
    ax.set_ylabel("total_stats")
    fig.tight_layout()
    fig.savefig(out_dir / "mono_vs_dual_total_stats_boxplot.png", dpi=200)
    plt.close(fig)

# Save a histogram of bootstrap mean differences with confidence interval lines
def save_bootstrap_mean_diff_hist(
    boot_diffs: np.ndarray, out_dir: Path, ci_low: float, ci_high: float) -> None:
    _ensure_out_dir(out_dir)
    
    # Create and save the histogram of bootstrap mean differences
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(boot_diffs, bins=30)
    ax.axvline(0.0)
    ax.axvline(ci_low, linestyle="--")
    ax.axvline(ci_high, linestyle="--")
    ax.set_title("Bootstrap distribution: mean(total_stats) dual - mono")
    ax.set_xlabel("mean difference (dual - mono)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "bootstrap_mean_diff_dual_minus_mono.png", dpi=200)
    plt.close(fig)

# Save a histogram of bootstrap median differences with confidence interval lines
def save_bootstrap_median_diff_hist(
    boot_diffs: np.ndarray, out_dir: Path, ci_low: float, ci_high: float) -> None:
    _ensure_out_dir(out_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(boot_diffs, bins=30)
    ax.axvline(0.0)
    ax.axvline(ci_low, linestyle="--")
    ax.axvline(ci_high, linestyle="--")
    ax.set_title("Bootstrap distribution: median(total_stats) dual - mono")
    ax.set_xlabel("median difference (dual - mono)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "bootstrap_median_diff_dual_minus_mono.png", dpi=200)
    plt.close(fig)