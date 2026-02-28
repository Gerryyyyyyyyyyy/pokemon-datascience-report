from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd



# Hypothesis: Dual-type Pokemon have higher average total_stats than mono-type Pokemon
def add_dual_type_feature(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["types"] = out["types"].fillna("unknown").astype(str)
    out["is_dual_type"] = out["types"].str.contains("|", regex=False)
    return out


def mono_vs_dual_summary(df: pd.DataFrame) -> dict[str, Any]:
    if "is_dual_type" not in df.columns:
        raise ValueError("Column 'is_dual_type' missing.")
    if "total_stats" not in df.columns:
        raise ValueError("Column 'total_stats' missing.")
    
    # Separate total_stats for mono and dual type Pokemon, dropping NaNs
    mono = df.loc[~df["is_dual_type"], "total_stats"].dropna()
    dual = df.loc[df["is_dual_type"], "total_stats"].dropna()

    # Compute summary statistics for each group and the difference in means and medians
    return {
        "mono": {
            "n": int(len(mono)),
            "mean_total_stats": round(float(mono.mean()), 2),
            "median_total_stats": round(float(mono.median()), 2),
            "std_total_stats": round(float(mono.std(ddof=1)), 2) if len(mono) > 1 else None,
        },
        "dual": {
            "n": int(len(dual)),
            "mean_total_stats": round(float(dual.mean()), 2),
            "median_total_stats": round(float(dual.median()), 2),
            "std_total_stats": round(float(dual.std(ddof=1)), 2) if len(dual) > 1 else None,
        },
        "mean_diff_dual_minus_mono": round(float(dual.mean() - mono.mean()), 2),
        "median_diff_dual_minus_mono": round(float(dual.median() - mono.median()), 2),
    }

# Bootstrap the mean difference in total_stats between dual and mono type Pokemon
def bootstrap_mean_diff(
    mono: np.ndarray,
    dual: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    
    # Check that input arrays are non-empty
    if len(mono) == 0 or len(dual) == 0:
        raise ValueError("mono and dual arrays must be non-empty")
    
    # Use numpy's random generator for reproducibility
    rng = np.random.default_rng(seed)

    # Array to hold bootstrap mean differences
    boot_diffs = np.empty(n_boot, dtype=float)

    # Perform bootstrap sampling
    for i in range(n_boot):
        mono_sample = rng.choice(mono, size=len(mono), replace=True)
        dual_sample = rng.choice(dual, size=len(dual), replace=True)
        boot_diffs[i] = dual_sample.mean() - mono_sample.mean()

    # Calculate 95% confidence interval from bootstrap distribution
    ci_low, ci_high = np.quantile(boot_diffs, [0.025, 0.975])

    # Return results in a dictionary
    return {
        "n_bootstrap": int(n_boot),
        "bootstrap_mean_diff_mean": round(float(boot_diffs.mean()), 4),
        "ci_95_low": round(float(ci_low), 4),
        "ci_95_high": round(float(ci_high), 4),
        "boot_diffs": boot_diffs,  # keep for plotting (not JSON-serializable directly)
    }
# Bootstrap the median difference in total_stats between dual and mono type Pokemon
def bootstrap_median_diff(
    mono: np.ndarray,
    dual: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    if len(mono) == 0 or len(dual) == 0:
        raise ValueError("mono and dual arrays must be non-empty")

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        mono_sample = rng.choice(mono, size=len(mono), replace=True)
        dual_sample = rng.choice(dual, size=len(dual), replace=True)
        boot_diffs[i] = np.median(dual_sample) - np.median(mono_sample)

    ci_low, ci_high = np.quantile(boot_diffs, [0.025, 0.975])

    return {
        "n_bootstrap": int(n_boot),
        "bootstrap_median_diff_mean": round(float(boot_diffs.mean()), 4),
        "ci_95_low": round(float(ci_low), 4),
        "ci_95_high": round(float(ci_high), 4),
        "boot_diffs": boot_diffs,  # for plotting only
    }

# Build a result dictionary for the hypothesis test, including group summaries and bootstrap results
def build_hypothesis_result(df: pd.DataFrame, n_boot: int = 5000, seed: int = 42) -> dict[str, Any]:
    if "is_dual_type" not in df.columns:
        raise ValueError("Column 'is_dual_type' missing.")
    if "total_stats" not in df.columns:
        raise ValueError("Column 'total_stats' missing.")

    mono = df.loc[~df["is_dual_type"], "total_stats"].dropna().to_numpy()
    dual = df.loc[df["is_dual_type"], "total_stats"].dropna().to_numpy()

    # Get summary statistics for mono and dual type groups and the difference in means/medians
    summary = mono_vs_dual_summary(df)
    boot = bootstrap_mean_diff(mono=mono, dual=dual, n_boot=n_boot, seed=seed)
    boot_median = bootstrap_median_diff(mono=mono, dual=dual, n_boot=n_boot, seed=seed)

    # Build a comprehensive result dictionary for the hypothesis test
    result = {
        "hypothesis": "Dual-type Pokemon have higher average total_stats than mono-type Pokemon",
        "metric": "total_stats",
        "groups": {
            "mono_type": "is_dual_type == False",
            "dual_type": "is_dual_type == True",
        },
        "group_summary": summary,
        "bootstrap_mean_difference_dual_minus_mono": {
            "n_bootstrap": boot["n_bootstrap"],
            "bootstrap_mean_diff_mean": boot["bootstrap_mean_diff_mean"],
            "ci_95_low": boot["ci_95_low"],
            "ci_95_high": boot["ci_95_high"],
        },
                "bootstrap_median_difference_dual_minus_mono": {
            "n_bootstrap": boot_median["n_bootstrap"],
            "bootstrap_median_diff_mean": boot_median["bootstrap_median_diff_mean"],
            "ci_95_low": boot_median["ci_95_low"],
            "ci_95_high": boot_median["ci_95_high"],
        },
        "interpretation_hint": (
            "If the 95% CI is entirely above 0, the data supports that dual-type Pokemon "
            "have a higher average total_stats in this dataset."
        ),
    }

    return result

# Utility function to save a dictionary as a JSON file, ensuring the output directory exists
def save_json(data: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")