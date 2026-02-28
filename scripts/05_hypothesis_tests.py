import pandas as pd
from pokemon_ds.config import PROCESSED_DIR, REPORTS_DIR, FIG_DIR
from pokemon_ds.hypothesis import (
    add_dual_type_feature,
    build_hypothesis_result,
    bootstrap_mean_diff,
    bootstrap_median_diff,   
    save_json,
)
from pokemon_ds.hypothesis_plots import (
    save_mono_vs_dual_boxplot,
    save_bootstrap_mean_diff_hist,
    save_bootstrap_median_diff_hist,  
)


def main() -> None:
    in_path = PROCESSED_DIR / "pokemon_clean.csv"
    df = pd.read_csv(in_path)

    # Add feature
    df = add_dual_type_feature(df)

    # Build JSON-safe result
    result = build_hypothesis_result(df, n_boot=5000, seed=42)
    out_json = REPORTS_DIR / "hypothesis_results.json"
    save_json(result, out_json)

    # Recompute bootstrap incl. raw distribution for plotting
    mono = df.loc[~df["is_dual_type"], "total_stats"].dropna().to_numpy()
    dual = df.loc[df["is_dual_type"], "total_stats"].dropna().to_numpy()
    boot = bootstrap_mean_diff(mono=mono, dual=dual, n_boot=5000, seed=42)
    boot_median = bootstrap_median_diff(mono=mono, dual=dual, n_boot=5000, seed=42)

    # Plots
    save_mono_vs_dual_boxplot(df, FIG_DIR)
    save_bootstrap_mean_diff_hist(
        boot_diffs=boot["boot_diffs"],
        out_dir=FIG_DIR,
        ci_low=boot["ci_95_low"],
        ci_high=boot["ci_95_high"],
    )
    save_bootstrap_median_diff_hist(
        boot_diffs=boot_median["boot_diffs"],
        out_dir=FIG_DIR,
        ci_low=boot_median["ci_95_low"],
        ci_high=boot_median["ci_95_high"],
    )

    # Console output
    group_summary = result["group_summary"]
    boot_summary = result["bootstrap_mean_difference_dual_minus_mono"]
    boot_median_summary = result["bootstrap_median_difference_dual_minus_mono"]

    print("=== HYPOTHESIS TEST: DUAL VS MONO TOTAL_STATS ===")
    print("\nGroup summary:")
    print(pd.Series({
        "mono_n": group_summary["mono"]["n"],
        "mono_mean_total_stats": group_summary["mono"]["mean_total_stats"],
        "dual_n": group_summary["dual"]["n"],
        "dual_mean_total_stats": group_summary["dual"]["mean_total_stats"],
        "mean_diff_dual_minus_mono": group_summary["mean_diff_dual_minus_mono"],
        "median_diff_dual_minus_mono": group_summary["median_diff_dual_minus_mono"],
    }))

    print("\nBootstrap CI (mean diff dual - mono):")
    print(pd.Series(boot_summary))

    print("\nBootstrap CI (median diff dual - mono):")
    print(pd.Series(boot_median_summary))

    print(f"\nSaved JSON: {out_json}")
    print(f"Saved plots to: {FIG_DIR}")


if __name__ == "__main__":
    main()