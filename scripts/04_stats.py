import pandas as pd
from pokemon_ds.config import PROCESSED_DIR, REPORTS_DIR, FIG_DIR
from pokemon_ds.stats_plots import (
    save_mono_vs_dual_count_plot,
    save_missing_values_plot,
    save_avg_total_stats_by_primary_type,
    save_avg_speed_by_primary_type,
)
from pokemon_ds.stats_analysis import (
    add_type_features,
    qa_checks,
    summary_metrics,
    primary_type_summary,
    save_json,
)


def main() -> None:
    in_path = PROCESSED_DIR / "pokemon_clean.csv"
    df = pd.read_csv(in_path)

    # Add derived features for analysis
    df = add_type_features(df)

    # QA + global metrics
    qa = qa_checks(df)
    metrics = summary_metrics(df)

    results = {
        "dataset": "pokemon_clean.csv",
        "qa_checks": qa,
        "summary_metrics": metrics,
    }

    # Save JSON report
    stats_json_path = REPORTS_DIR / "stats_results.json"
    save_json(results, stats_json_path)

    # Save plots and type summary CSV
    type_summary = primary_type_summary(df)
    save_mono_vs_dual_count_plot(df, FIG_DIR)
    save_missing_values_plot(df, FIG_DIR)
    save_avg_total_stats_by_primary_type(type_summary, FIG_DIR)
    save_avg_speed_by_primary_type(type_summary, FIG_DIR)
    type_summary_path = PROCESSED_DIR / "primary_type_summary.csv"
    type_summary.to_csv(type_summary_path, index=False)

    # Console preview of results
    print("=== QA CHECKS ===")
    print(pd.Series(qa))
    print("\n=== SUMMARY METRICS ===")
    print(pd.Series(metrics))
    print("\n=== PRIMARY TYPE SUMMARY (top 10 by mean_total_stats) ===")
    print(type_summary.head(10))

    print(f"\nSaved JSON: {stats_json_path}")
    print(f"Saved CSV:  {type_summary_path}")
    print(f"Saved avg. Speed by primary type plot to: {FIG_DIR}")
    print(f"Saved avg. Total Stats by primary type plot to: {FIG_DIR}")
    print(f"Saved mono vs dual type count plot to: {FIG_DIR}")

if __name__ == "__main__":
    main()