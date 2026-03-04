import pandas as pd
from pokemon_ds.config import PROCESSED_DIR, REPORTS_DIR, FIG_DIR
from pokemon_ds.modeling import (
    FEATURE_COLS,
    prepare_regression_data,
    split_data,
    evaluate_regression,
    baseline_mean_predictor,
    train_linear_regression_pipeline,
    train_random_forest_pipeline,
    build_linear_regression_pipeline,
    build_random_forest_pipeline,
    cross_validate_regression_model,
    add_cv_results_to_regression_dict,
    feature_importance_table_rf_from_pipeline,
    build_regression_results_dict,
    save_json,
)
from pokemon_ds.modeling_plots import (
    save_pred_vs_actual_scatter,
    save_rf_feature_importance_plot,
)

def main() -> None:
    
    # Load the cleaned dataset
    # This file is produced by scripts/02_clean.py and contains:
    # - the base stats
    # - base_experience
    # - engineered total_stats
    
    in_path = PROCESSED_DIR / "pokemon_clean.csv"
    df = pd.read_csv(in_path)

    # Prepare features (X) and target (y) for the regression task
    # Task: predict base_experience from the six base stats.
    #
    # IMPORTANT:
    # - This function only selects columns and drops rows with missing target.
    # - Preprocessing (imputation/scaling) is done later inside sklearn Pipelines
    #   to avoid data leakage.
    
    X, y = prepare_regression_data(df)

    # Train/test split (holdout evaluation)
    #
    # IMPORTANT:
    # We split BEFORE fitting any preprocessing transformers.
    # If we scaled/imputed before the split, we would leak information from the
    # test set into training.
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    
    # Baseline model: mean predictor
    #
    # This predicts the mean of y_train for every test sample.
    # It is a simple reference point. Real ML models should beat this baseline.
    
    y_pred_baseline = baseline_mean_predictor(y_train, n_test=len(y_test))
    baseline_metrics = evaluate_regression(y_test, y_pred_baseline)

    
    # Linear Regression (with preprocessing pipeline)
    # Pipeline = SimpleImputer(median) -> StandardScaler -> LinearRegression
    
    linear_pipe = train_linear_regression_pipeline(X_train, y_train)
    y_pred_linear = linear_pipe.predict(X_test)
    linear_metrics = evaluate_regression(y_test, y_pred_linear)

    
    # Random Forest Regressor (with preprocessing pipeline)
    # Pipeline = SimpleImputer(median) -> RandomForestRegressor
  
    
    rf_pipe = train_random_forest_pipeline(X_train, y_train, random_state=42)
    y_pred_rf = rf_pipe.predict(X_test)
    rf_metrics = evaluate_regression(y_test, y_pred_rf)

    
    # 5-Fold Cross-Validation (more robust model comparison)
    #
    # Holdout metrics depend on one particular split.
    # Cross-validation evaluates model performance across multiple folds and gives:
    # - mean performance
    # - standard deviation (stability / variance)
    #
    # IMPORTANT:
    # We create fresh pipelines here and evaluate them via CV on the full X/y.
    # Pipelines ensure preprocessing is fit separately inside each fold (no leakage).
    
    linear_cv_pipe = build_linear_regression_pipeline()
    rf_cv_pipe = build_random_forest_pipeline(random_state=42)

    linear_cv_metrics = cross_validate_regression_model(
        model_pipeline=linear_cv_pipe,
        X=X,
        y=y,
        n_splits=5,
        random_state=42,
    )

    rf_cv_metrics = cross_validate_regression_model(
        model_pipeline=rf_cv_pipe,
        X=X,
        y=y,
        n_splits=5,
        random_state=42,
    )

    
    # Build and save experiment summary as JSON
    #
    # We combine:
    # - holdout metrics (baseline / linear / RF)
    # - CV metrics (linear / RF)
    # - preprocessing notes
    
    results = build_regression_results_dict(
        y_train=y_train,
        y_test=y_test,
        baseline_metrics=baseline_metrics,
        linear_metrics=linear_metrics,
        rf_metrics=rf_metrics,
    )

    results = add_cv_results_to_regression_dict(
        results=results,
        linear_cv_metrics=linear_cv_metrics,
        rf_cv_metrics=rf_cv_metrics,
    )

    out_json = REPORTS_DIR / "ml_regression_results.json"
    save_json(results, out_json)

    
    # Diagnostic plots: Predicted vs Actual
    #
    # These plots help to visually inspect:
    # - whether predictions follow the identity line
    # - systematic under/overestimation
    # - spread of prediction errors
    
    save_pred_vs_actual_scatter(
        y_true=y_test,
        y_pred=y_pred_linear,
        out_dir=FIG_DIR,
        filename="regression_pred_vs_actual_linear.png",
        title="Linear Regression: predicted vs actual",
    )

    save_pred_vs_actual_scatter(
        y_true=y_test,
        y_pred=y_pred_rf,
        out_dir=FIG_DIR,
        filename="regression_pred_vs_actual_rf.png",
        title="Random Forest: predicted vs actual",
    )

    
    # Random Forest feature importance
    #
    # We extract the feature importances from the fitted RF model inside the pipeline.
    # This is useful for interpretation (which stats matter most for the prediction).
    
    rf_importance = feature_importance_table_rf_from_pipeline(rf_pipe, FEATURE_COLS)

    rf_importance_path = PROCESSED_DIR / "rf_feature_importance.csv"
    rf_importance.to_csv(rf_importance_path, index=False)

    save_rf_feature_importance_plot(rf_importance, FIG_DIR)

    
    # Console output (quick development feedback)
    #
    # Prints all key metrics so you can quickly compare models without opening files.
    
    print("=== REGRESSION TASK: base_experience from base stats ===")
    print(f"Train size: {len(y_train)}")
    print(f"Test size:  {len(y_test)}")

    print("\nBaseline (mean predictor):")
    print(pd.Series(baseline_metrics))

    print("\nLinear Regression (holdout, pipeline with scaler):")
    print(pd.Series(linear_metrics))

    print("\nRandom Forest Regressor (holdout, pipeline):")
    print(pd.Series(rf_metrics))

    print("\n=== 5-Fold Cross-Validation (Linear Regression Pipeline) ===")
    print(pd.Series(linear_cv_metrics))

    print("\n=== 5-Fold Cross-Validation (Random Forest Pipeline) ===")
    print(pd.Series(rf_cv_metrics))

    print("\nRandom Forest feature importance:")
    print(rf_importance)

    print(f"\nSaved JSON: {out_json}")
    print(f"Saved feature importance CSV: {rf_importance_path}")
    print(f"Saved plots to: {FIG_DIR}")


if __name__ == "__main__":
    main()