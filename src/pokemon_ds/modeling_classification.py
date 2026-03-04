from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Features used for the classification task
FEATURE_COLS = ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]

# Target label
TARGET_COL = "is_dual_type"

# Create classification target `is_dual_type` from the `types` column.
def add_dual_type_target(df: pd.DataFrame) -> pd.DataFrame:
  
    out = df.copy()
    out["types"] = out["types"].fillna("unknown").astype(str)
    out[TARGET_COL] = out["types"].str.contains("|", regex=False)
    return out

# Prepare classification data by selecting features and target, dropping rows with missing target, and returning X (features) and y (target)
def prepare_classification_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    
    required = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = df[required].copy()

    # Drop rows with missing target (unlikely, but safe)
    data = data.dropna(subset=[TARGET_COL])

    X = data[FEATURE_COLS].copy()
    y = data[TARGET_COL].astype(int).copy()  # convert bool -> 0/1 for convenience

    return X, y

# Split data into train/test sets using sklearn's train_test_split, returning X_train, X_test, y_train, y_test
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Evaluate classification predictions using accuracy, precision, recall, and F1 metrics, returning a dictionary of results
def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
   
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }

# Compute confusion matrix counts for binary classification, returning a dictionary of TN, FP, FN, TP
def confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
   
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "tn_mono_pred_mono": int(tn),
        "fp_mono_pred_dual": int(fp),
        "fn_dual_pred_mono": int(fn),
        "tp_dual_pred_dual": int(tp),
    }

# Baseline classifier: always predict the majority class from y_train. Returns an array of predictions for the test set.
def baseline_majority_predictor(y_train: pd.Series, n_test: int) -> np.ndarray:
   
    majority_class = int(y_train.mode().iloc[0])
    return np.full(shape=n_test, fill_value=majority_class, dtype=int)

# Build a classification pipeline for Logistic Regression
def build_logistic_regression_pipeline() -> Pipeline:
   
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

# Build a classification pipeline for Random Forest Classifier
def build_random_forest_classifier_pipeline(random_state: int = 42) -> Pipeline:
   
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=random_state,
            )),
        ]
    )

# Build and fit the Logistic Regression pipeline on training data
def train_logistic_regression_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe = build_logistic_regression_pipeline()
    pipe.fit(X_train, y_train)
    return pipe

# Build and fit the Random Forest Classifier pipeline on training data
def train_random_forest_classifier_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Pipeline:
    pipe = build_random_forest_classifier_pipeline(random_state=random_state)
    pipe.fit(X_train, y_train)
    return pipe

# Extract feature importances from fitted RF classifier pipeline and return a DataFrame sorted by importance
def feature_importance_table_rf_classifier_from_pipeline(
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

# Cross-validate a classifier pipeline using K-Fold CV and return mean/std for accuracy, precision, recall, and F1 metrics
def cross_validate_classifier_model(
    model_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # We use multiple metrics to get a comprehensive view of classification performance.
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    # cross_validate returns a dict with keys like 'test_accuracy', 'test_precision', etc., each containing an array of scores for the folds.
    scores = cross_validate(
        estimator=model_pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=None,
    )
    # compute the mean and standard deviation for each metric across the folds. If there's only one fold, std is set to 0.0.
    def mstd(key: str) -> tuple[float, float]:
        vals = scores[f"test_{key}"]
        mean = round(float(np.mean(vals)), 4)
        std = round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0
        return mean, std

    acc_mean, acc_std = mstd("accuracy")
    prec_mean, prec_std = mstd("precision")
    rec_mean, rec_std = mstd("recall")
    f1_mean, f1_std = mstd("f1")

    return {
        "cv_n_splits": int(n_splits),
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "precision_mean": prec_mean,
        "precision_std": prec_std,
        "recall_mean": rec_mean,
        "recall_std": rec_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
    }

# Build a JSON summary dictionary for classification experiment results, including holdout metrics and confusion matrices for baseline, logistic regression, and random forest models.
def build_classification_results_dict(
    y_train: pd.Series,
    y_test: pd.Series,
    baseline_metrics: dict[str, float],
    logreg_metrics: dict[str, float],
    rf_metrics: dict[str, float],
    baseline_cm: dict[str, int],
    logreg_cm: dict[str, int],
    rf_cm: dict[str, int],
) -> dict[str, Any]:
 
    class_counts_train = pd.Series(y_train).value_counts().to_dict()
    class_counts_test = pd.Series(y_test).value_counts().to_dict()

    return {
        "task": "Classification: predict is_dual_type from base stats",
        "target": TARGET_COL,
        "target_mapping": {"0": "mono_type", "1": "dual_type"},
        "features": FEATURE_COLS,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "class_balance": {
            "train": {str(int(k)): int(v) for k, v in class_counts_train.items()},
            "test": {str(int(k)): int(v) for k, v in class_counts_test.items()},
        },
        "preprocessing": {
            "logistic_regression": [
                "SimpleImputer(strategy='median')",
                "StandardScaler()",
                "LogisticRegression(max_iter=1000)",
            ],
            "random_forest_classifier": [
                "SimpleImputer(strategy='median')",
                "RandomForestClassifier(...)",
            ],
        },
        "baseline_majority_predictor": {
            "metrics": baseline_metrics,
            "confusion_matrix": baseline_cm,
        },
        "logistic_regression": {
            "metrics": logreg_metrics,
            "confusion_matrix": logreg_cm,
        },
        "random_forest_classifier": {
            "metrics": rf_metrics,
            "confusion_matrix": rf_cm,
        },
        "notes": [
            "This is a learning-focused classification task on a small dataset.",
            "Preprocessing is handled via sklearn Pipelines to avoid leakage.",
        ],
    }

# Add CV results to an existing classification result dictionary
def add_cv_results_to_classification_dict(
    results: dict[str, Any],
    logreg_cv_metrics: dict[str, float],
    rf_cv_metrics: dict[str, float],
) -> dict[str, Any]:
   
    out = dict(results)
    out["cross_validation"] = {
        "logistic_regression_pipeline": logreg_cv_metrics,
        "random_forest_classifier_pipeline": rf_cv_metrics,
        "notes": [
            "Cross-validation complements the single holdout split.",
            "Pipelines ensure fold-wise preprocessing without leakage.",
        ],
    }
    return out


def save_json(data: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")