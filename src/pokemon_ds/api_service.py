from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd
from pokemon_ds.config import PROCESSED_DIR
from pokemon_ds.modeling import (
    FEATURE_COLS,
    prepare_regression_data,
    train_linear_regression_pipeline,
)


ALLOWED_TOP_METRICS = {
    "total_stats",
    "hp",
    "attack",
    "defense",
    "special-attack",
    "special-defense",
    "speed",
    "base_experience",
}


@dataclass
class AppState:
    
    # In-memory state for API service. Loaded once on startup
    pokemon_df: pd.DataFrame
    regression_model: Any  # sklearn Pipeline


def load_clean_dataset() -> pd.DataFrame:

    # Load the cleaned dataset for use in API endpoints and model training
    path = PROCESSED_DIR / "pokemon_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {path}. Run scripts/02_clean.py first."
        )

    df = pd.read_csv(path)
    return df


def train_regression_model_for_api(df: pd.DataFrame):
   # Train a regression model to predict base_experience from stats, for use in API prediction endpoint
    X, y = prepare_regression_data(df)
    model = train_linear_regression_pipeline(X, y)
    return model


def build_app_state() -> AppState:
    
    # Load dataset and train regression model once on startup, to be used across API endpoints
    df = load_clean_dataset()
    model = train_regression_model_for_api(df)
    return AppState(pokemon_df=df, regression_model=model)


def get_top_pokemon_by_metric(df: pd.DataFrame, metric: str, n: int = 10) -> list[dict[str, Any]]:

    # Get top N Pokémon by a specified metric, for API endpoint. Validates metric and handles missing values.
    if metric not in ALLOWED_TOP_METRICS:
        raise ValueError(f"Unsupported metric '{metric}'. Allowed: {sorted(ALLOWED_TOP_METRICS)}")

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in dataset columns.")

    if "name" not in df.columns:
        raise ValueError("Column 'name' not found in dataset.")

    out = (
        df[["name", metric]]
        .dropna()
        .sort_values(metric, ascending=False)
        .head(n)
        .copy()
    )

    # Make names a bit nicer for API consumers
    out["name"] = out["name"].astype(str).str.replace("-", " ", regex=False).str.title()

    return out.to_dict(orient="records")


def summarize_types(df: pd.DataFrame) -> dict[str, Any]:
    
    # Summarize Pokémon types for API endpoint. Counts primary types and total occurrences across all Pokémon, handling missing values
    if "types" not in df.columns:
        raise ValueError("Column 'types' not found in dataset.")

    types = df["types"].fillna("unknown").astype(str)

    primary_counts = (
        types.str.split("|", regex=False).str[0].str.strip().value_counts()
    )

    occurrence_counts = (
        types.str.split("|", regex=False).explode().str.strip().value_counts()
    )

    return {
        "n_pokemon": int(len(df)),
        "primary_type_counts": {str(k): int(v) for k, v in primary_counts.to_dict().items()},
        "type_occurrence_counts": {str(k): int(v) for k, v in occurrence_counts.to_dict().items()},
    }


def predict_base_experience_from_stats(model, stats_payload: dict[str, float]) -> dict[str, Any]:
    
    # Predict base_experience from stats using the trained regression model, for API endpoint. Validates input and handles missing features.
    missing = [c for c in FEATURE_COLS if c not in stats_payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    X_pred = pd.DataFrame([{c: float(stats_payload[c]) for c in FEATURE_COLS}])
    pred = model.predict(X_pred)[0]

    return {
        "model": "linear_regression_pipeline",
        "features_used": FEATURE_COLS,
        "predicted_base_experience": round(float(pred), 4),
    }