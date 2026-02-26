from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Make sure the output directory exists, then create and save a bar plot comparing the count of mono-type vs dual-type Pokemon
def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

# Create and save a bar plot comparing the count of mono-type vs dual-type Pokemon
def save_mono_vs_dual_count_plot(df: pd.DataFrame, out_dir: Path) -> None:
    _ensure_out_dir(out_dir)
    
    # Check if 'is_dual_type' column exists
    if "is_dual_type" not in df.columns:
        raise ValueError("Column 'is_dual_type' not found. Run add_type_features() first.")
    
    # Calculate counts
    counts = pd.Series(
        {
            "mono_type": int((~df["is_dual_type"]).sum()),
            "dual_type": int(df["is_dual_type"].sum()),
        }
    )
    
    # Create and save the bar plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title("Mono-type vs Dual-type Pokemon (count)")
    ax.set_xlabel("group")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "mono_vs_dual_count.png", dpi=200)
    plt.close(fig)

# Save missing values plot showing the count of missing values per column
def save_missing_values_plot(df: pd.DataFrame, out_dir: Path) -> None:
    _ensure_out_dir(out_dir)

    missing = df.isna().sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(missing.index, missing.values)
    ax.set_title("Missing values per column")
    ax.set_xlabel("column")
    ax.set_ylabel("missing count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "missing_values_per_column.png", dpi=200)
    plt.close(fig)

# Create and save a horizontal bar plot of average total stats by primary type, sorted by mean total stats
def save_avg_total_stats_by_primary_type(
    type_summary_df: pd.DataFrame, out_dir: Path, top_n: int | None = None
) -> None:
    _ensure_out_dir(out_dir)

    required = {"primary_type", "mean_total_stats"}
    missing = required - set(type_summary_df.columns)
    if missing:
        raise ValueError(f"Missing columns in type_summary_df: {sorted(missing)}")

    plot_df = type_summary_df.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    # Sort ascending for nicer horizontal bar plot (largest at top after invert)
    plot_df = plot_df.sort_values("mean_total_stats", ascending=True)

    labels = plot_df["primary_type"].astype(str).str.replace("-", " ", regex=False).str.title()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, plot_df["mean_total_stats"])
    ax.set_title("Average total stats by primary type")
    ax.set_xlabel("mean total_stats")
    ax.set_ylabel("primary type")
    fig.tight_layout()
    fig.savefig(out_dir / "avg_total_stats_by_primary_type.png", dpi=200)
    plt.close(fig)

# Create and save a horizontal bar plot of average speed by primary type, sorted by mean speed
def save_avg_speed_by_primary_type(
    type_summary_df: pd.DataFrame, out_dir: Path, top_n: int | None = None
) -> None:
    _ensure_out_dir(out_dir)

    # Validate required columns
    required = {"primary_type", "mean_speed"}
    missing = required - set(type_summary_df.columns)
    if missing:
        raise ValueError(f"Missing columns in type_summary_df: {sorted(missing)}")
   
    # Sort ascending for nicer horizontal bar plot (largest at top after invert)
    plot_df = type_summary_df.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    plot_df = plot_df.sort_values("mean_speed", ascending=True)
    labels = plot_df["primary_type"].astype(str).str.replace("-", " ", regex=False).str.title()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, plot_df["mean_speed"])
    ax.set_title("Average speed by primary type")
    ax.set_xlabel("mean speed")
    ax.set_ylabel("primary type")
    fig.tight_layout()
    fig.savefig(out_dir / "avg_speed_by_primary_type.png", dpi=200)
    plt.close(fig)