from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Feature columns (inputs) used to predict base_experience
FEATURE_COLS = ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]

# Target column (output / label)
TARGET_COL = "base_experience"


# Prepare regression data by selecting features and target, dropping rows with missing target, and returning X (features) and y (target)
def prepare_regression_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    required = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Keep only columns needed for the regression task
    data = df[required].copy()

    # We must know the target to train/evaluate a supervised model
    data = data.dropna(subset=[TARGET_COL])

    X = data[FEATURE_COLS].copy()
    y = data[TARGET_COL].copy()

    return X, y

# Split data into train/test sets using sklearn's train_test_split, returning X_train, X_test, y_train, y_test
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
  
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Evaluate regression predictions using MAE, RMSE, and R² metrics, returning a dictionary of results
def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
   
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }

# Build a simple baseline predictor that always predicts the mean of the training target values
def baseline_mean_predictor(y_train: pd.Series, n_test: int) -> np.ndarray:
   
    mean_value = float(y_train.mean())
    return np.full(shape=n_test, fill_value=mean_value, dtype=float)

# Build a regression pipeline for Linear Regression
def build_linear_regression_pipeline() -> Pipeline:
   
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

# Build a regression pipeline for Random Forest Regressor
def build_random_forest_pipeline(random_state: int = 42) -> Pipeline:
    
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    random_state=random_state,
                ),
            ),
        ]
    )

# Build and fit the Linear Regression pipeline on training data
def train_linear_regression_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Build and fit the Linear Regression pipeline on training data.
    """
    pipe = build_linear_regression_pipeline()
    pipe.fit(X_train, y_train)
    return pipe

# Build and fit the Random Forest pipeline on training data
def train_random_forest_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Pipeline:

    pipe = build_random_forest_pipeline(random_state=random_state)
    pipe.fit(X_train, y_train)
    return pipe

# Extract feature importance from a fitted Random Forest pipeline and return as a sorted DataFrame
def feature_importance_table_rf_from_pipeline(
    rf_pipeline: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
  
    rf_model = rf_pipeline.named_steps["model"]

    imp = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    imp["importance"] = imp["importance"].round(6)
    return imp

# Run cross-validation for a regression pipeline and return mean/std of metrics across folds
def cross_validate_regression_model(
    model_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # sklearn uses NEGATIVE error metrics for scorers by convention
    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }
    # Run cross-validation and get scores for each fold
    scores = cross_validate(
        estimator=model_pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=None,  # keep simple / portable
    )

    # Convert negative sklearn error scores back to positive error values
    mae_values = -scores["test_mae"]
    mse_values = -scores["test_mse"]
    rmse_values = np.sqrt(mse_values)
    r2_values = scores["test_r2"]

    return {
        "cv_n_splits": int(n_splits),
        "mae_mean": round(float(np.mean(mae_values)), 4),
        "mae_std": round(float(np.std(mae_values, ddof=1)), 4) if len(mae_values) > 1 else 0.0,
        "rmse_mean": round(float(np.mean(rmse_values)), 4),
        "rmse_std": round(float(np.std(rmse_values, ddof=1)), 4) if len(rmse_values) > 1 else 0.0,
        "r2_mean": round(float(np.mean(r2_values)), 4),
        "r2_std": round(float(np.std(r2_values, ddof=1)), 4) if len(r2_values) > 1 else 0.0,
    }

# Build a comprehensive regression results dictionary for reporting and saving to JSON
def build_regression_results_dict(
    y_train: pd.Series,
    y_test: pd.Series,
    baseline_metrics: dict[str, float],
    linear_metrics: dict[str, float],
    rf_metrics: dict[str, float],
) -> dict[str, Any]:
  
    return {
        "task": "Regression: predict base_experience from base stats",
        "target": TARGET_COL,
        "features": FEATURE_COLS,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "preprocessing": {
            "linear_regression": [
                "SimpleImputer(strategy='median')",
                "StandardScaler()",
                "LinearRegression()",
            ],
            "random_forest_regressor": [
                "SimpleImputer(strategy='median')",
                "RandomForestRegressor(...)",
            ],
            "notes": [
                "Preprocessing is fit only on training data via sklearn Pipeline.",
                "Scaling is used for Linear Regression but not for Random Forest.",
            ],
        },
        "baseline_mean_predictor": baseline_metrics,
        "linear_regression": linear_metrics,
        "random_forest_regressor": rf_metrics,
        "notes": [
            "This is a small educational dataset (Gen 1 Pokemon).",
            "The goal is to practice ML workflow and model comparison, not maximize performance.",
        ],
    }

# Add cross-validation metrics to an existing regression result dictionary
def add_cv_results_to_regression_dict(
    results: dict[str, Any],
    linear_cv_metrics: dict[str, float],
    rf_cv_metrics: dict[str, float],
) -> dict[str, Any]:
  
    out = dict(results)
    out["cross_validation"] = {
        "linear_regression_pipeline": linear_cv_metrics,
        "random_forest_pipeline": rf_cv_metrics,
        "notes": [
            "Cross-validation is performed on the full prepared dataset (X, y).",
            "Pipelines ensure preprocessing is fit separately inside each fold.",
            "This complements the single train/test split evaluation.",
        ],
    }
    return out

def save_json(data: dict[str, Any], out_path: Path) -> None:

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")