from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd

STAT_COLS = ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]

# Add derived features for analysis, such as primary type and dual type indicator
def add_type_features(df: pd.DataFrame) -> pd.DataFrame:
    
    out = df.copy()
    out["types"] = out["types"].fillna("unknown").astype(str)
    out["is_dual_type"] = out["types"].str.contains("|", regex=False)

    out["primary_type"] = (
        out["types"]
        .str.split("|", regex=False)
        .str[0]
        .fillna("unknown")
        .str.strip()
    )

    return out

# QA checks to validate dataset assumptions and identify potential issues
def qa_checks(df: pd.DataFrame) -> dict[str, Any]:
    
    checks: dict[str, Any] = {}
    checks["row_count"] = int(len(df))
    checks["column_count"] = int(df.shape[1])
    checks["missing_values_per_column"] = {
        col: int(val) for col, val in df.isna().sum().to_dict().items()
    }

    # Check if total_stats matches sum of stat columns
    if all(col in df.columns for col in STAT_COLS) and "total_stats" in df.columns:
        recomputed = df[STAT_COLS].sum(axis=1)
        mismatch_count = int((recomputed != df["total_stats"]).sum())
        checks["total_stats_mismatch_count"] = mismatch_count
    else:
        checks["total_stats_mismatch_count"] = None

    # Type formatting checks
    if "types" in df.columns:
        checks["types_missing_count"] = int(df["types"].isna().sum())
        checks["dual_type_count"] = int(df["types"].fillna("").str.contains("|", regex=False).sum())
        checks["mono_type_count"] = int(len(df) - checks["dual_type_count"])
    else:
        checks["types_missing_count"] = None
        checks["dual_type_count"] = None
        checks["mono_type_count"] = None

    # Duplicate checks
    checks["duplicate_rows_count"] = int(df.duplicated().sum())
    if "name" in df.columns:
        checks["duplicate_name_count"] = int(df["name"].duplicated().sum())
    else:
        checks["duplicate_name_count"] = None

    return checks


def summary_metrics(df: pd.DataFrame) -> dict[str, Any]:
   
    out: dict[str, Any] = {}
    out["n_pokemon"] = int(len(df))

    if "is_dual_type" in df.columns:
        dual_count = int(df["is_dual_type"].sum())
        out["n_dual_type"] = dual_count
        out["n_mono_type"] = int(len(df) - dual_count)
        out["pct_dual_type"] = round(dual_count / len(df) * 100, 2) if len(df) else None
    else:
        out["n_dual_type"] = None
        out["n_mono_type"] = None
        out["pct_dual_type"] = None

    if "total_stats" in df.columns:
        out["total_stats_mean"] = round(float(df["total_stats"].mean()), 2)
        out["total_stats_median"] = round(float(df["total_stats"].median()), 2)
        out["total_stats_min"] = int(df["total_stats"].min())
        out["total_stats_max"] = int(df["total_stats"].max())

        value_counts = df["total_stats"].value_counts().head(10)
        out["top_total_stats_frequencies"] = {
            str(int(k)): int(v) for k, v in value_counts.to_dict().items()
        }
    else:
        out["total_stats_mean"] = None
        out["total_stats_median"] = None
        out["total_stats_min"] = None
        out["total_stats_max"] = None
        out["top_total_stats_frequencies"] = {}

    return out

# Summarize stats by primary type, including count, mean/median total stats, and mean/median speed
def primary_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    
    required_cols = {"primary_type", "total_stats", "speed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for primary_type_summary: {sorted(missing)}")

    summary = (
        df.groupby("primary_type", dropna=False)
        .agg(
            count=("name", "count") if "name" in df.columns else ("primary_type", "count"),
            mean_total_stats=("total_stats", "mean"),
            median_total_stats=("total_stats", "median"),
            mean_speed=("speed", "mean"),
            median_speed=("speed", "median"),
        )
        .reset_index()
        .sort_values(["mean_total_stats", "count"], ascending=[False, False])
    )

    # rounding
    for col in ["mean_total_stats", "median_total_stats", "mean_speed", "median_speed"]:
        summary[col] = summary[col].round(2)

    return summary


def save_json(data: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")