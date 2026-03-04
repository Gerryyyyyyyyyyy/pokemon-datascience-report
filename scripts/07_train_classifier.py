import pandas as pd

from pokemon_ds.config import PROCESSED_DIR, REPORTS_DIR, FIG_DIR
from pokemon_ds.modeling_classification import (
    FEATURE_COLS,
    add_dual_type_target,
    prepare_classification_data,
    split_data,
    evaluate_classification,
    confusion_matrix_dict,
    baseline_majority_predictor,
    train_logistic_regression_pipeline,
    train_random_forest_classifier_pipeline,
    build_logistic_regression_pipeline,
    build_random_forest_classifier_pipeline,
    cross_validate_classifier_model,
    feature_importance_table_rf_classifier_from_pipeline,
    build_classification_results_dict,
    add_cv_results_to_classification_dict,
    save_json,
)
from pokemon_ds.modeling_classification_plots import (
    save_confusion_matrix_barplot,
    save_rf_classifier_feature_importance_plot,
    save_predicted_class_distribution_plot,
)


def main() -> None:
   
    # Load cleaned dataset and create classification target
    # Target: is_dual_type (0=mono, 1=dual)
  
    in_path = PROCESSED_DIR / "pokemon_clean.csv"
    df = pd.read_csv(in_path)
    df = add_dual_type_target(df)

    # Prepare features and target
    X, y = prepare_classification_data(df)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # Baseline classifier: majority-class predictor
    y_pred_baseline = baseline_majority_predictor(y_train, n_test=len(y_test))
    baseline_metrics = evaluate_classification(y_test, y_pred_baseline)
    baseline_cm = confusion_matrix_dict(y_test, y_pred_baseline)

    # Logistic Regression pipeline (imputer + scaler + model)
    logreg_pipe = train_logistic_regression_pipeline(X_train, y_train)
    y_pred_logreg = logreg_pipe.predict(X_test)
    logreg_metrics = evaluate_classification(y_test, y_pred_logreg)
    logreg_cm = confusion_matrix_dict(y_test, y_pred_logreg)

    # Random Forest Classifier pipeline (imputer + model)
    rf_pipe = train_random_forest_classifier_pipeline(X_train, y_train, random_state=42)
    y_pred_rf = rf_pipe.predict(X_test)
    rf_metrics = evaluate_classification(y_test, y_pred_rf)
    rf_cm = confusion_matrix_dict(y_test, y_pred_rf)

    # 5-Fold Cross-Validation (robust comparison)
    logreg_cv_pipe = build_logistic_regression_pipeline()
    rf_cv_pipe = build_random_forest_classifier_pipeline(random_state=42)

    logreg_cv_metrics = cross_validate_classifier_model(
        model_pipeline=logreg_cv_pipe,
        X=X,
        y=y,
        n_splits=5,
        random_state=42,
    )

    rf_cv_metrics = cross_validate_classifier_model(
        model_pipeline=rf_cv_pipe,
        X=X,
        y=y,
        n_splits=5,
        random_state=42,
    )

    # Save results to JSON and create plots
    results = build_classification_results_dict(
        y_train=y_train,
        y_test=y_test,
        baseline_metrics=baseline_metrics,
        logreg_metrics=logreg_metrics,
        rf_metrics=rf_metrics,
        baseline_cm=baseline_cm,
        logreg_cm=logreg_cm,
        rf_cm=rf_cm,
    )

    results = add_cv_results_to_classification_dict(
        results=results,
        logreg_cv_metrics=logreg_cv_metrics,
        rf_cv_metrics=rf_cv_metrics,
    )

    out_json = REPORTS_DIR / "ml_classification_results.json"
    save_json(results, out_json)

    # Feature importance (RF classifier)
    rf_importance = feature_importance_table_rf_classifier_from_pipeline(rf_pipe, FEATURE_COLS)
    rf_importance_path = PROCESSED_DIR / "rf_classifier_feature_importance.csv"
    rf_importance.to_csv(rf_importance_path, index=False)

    # Plots
    save_confusion_matrix_barplot(
        baseline_cm,
        FIG_DIR,
        filename="clf_baseline_confusion_components.png",
        title="Baseline majority classifier: confusion matrix components",
    )

    save_confusion_matrix_barplot(
        logreg_cm,
        FIG_DIR,
        filename="clf_logreg_confusion_components.png",
        title="Logistic Regression: confusion matrix components",
    )

    save_confusion_matrix_barplot(
        rf_cm,
        FIG_DIR,
        filename="clf_rf_confusion_components.png",
        title="Random Forest Classifier: confusion matrix components",
    )

    save_predicted_class_distribution_plot(
        y_pred_logreg,
        FIG_DIR,
        filename="clf_logreg_predicted_class_distribution.png",
        title="Logistic Regression: predicted class distribution",
    )

    save_predicted_class_distribution_plot(
        y_pred_rf,
        FIG_DIR,
        filename="clf_rf_predicted_class_distribution.png",
        title="Random Forest Classifier: predicted class distribution",
    )

    save_rf_classifier_feature_importance_plot(rf_importance, FIG_DIR)

    # Console output
    print("=== CLASSIFICATION TASK: is_dual_type from base stats ===")
    print(f"Train size: {len(y_train)}")
    print(f"Test size:  {len(y_test)}")

    print("\nBaseline (majority predictor):")
    print(pd.Series(baseline_metrics))
    print(pd.Series(baseline_cm))

    print("\nLogistic Regression (holdout):")
    print(pd.Series(logreg_metrics))
    print(pd.Series(logreg_cm))

    print("\nRandom Forest Classifier (holdout):")
    print(pd.Series(rf_metrics))
    print(pd.Series(rf_cm))

    print("\n=== 5-Fold Cross-Validation (Logistic Regression Pipeline) ===")
    print(pd.Series(logreg_cv_metrics))

    print("\n=== 5-Fold Cross-Validation (Random Forest Classifier Pipeline) ===")
    print(pd.Series(rf_cv_metrics))

    print("\nRandom Forest classifier feature importance:")
    print(rf_importance)

    print(f"\nSaved JSON: {out_json}")
    print(f"Saved feature importance CSV: {rf_importance_path}")
    print(f"Saved plots to: {FIG_DIR}")


if __name__ == "__main__":
    main()