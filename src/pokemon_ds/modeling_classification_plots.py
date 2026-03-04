from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

# Save a confusion matrix bar plot given a dictionary of counts for each component (TP, FP, TN, FN)
def save_confusion_matrix_barplot(
    cm_dict: dict[str, int],
    out_dir: Path,
    filename: str,
    title: str,
) -> None:
  
    _ensure_out_dir(out_dir)
    
    # cm_dict should have keys like 'TP', 'FP', 'TN', 'FN' with integer counts. We create a bar plot to visualize these counts.
    labels = list(cm_dict.keys())
    values = list(cm_dict.values())
   
    # Create and save the bar plot for the confusion matrix components
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel("confusion matrix component")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)

# Save a horizontal bar plot of feature importances from a random forest classifier, given a DataFrame with 'feature' and 'importance' columns. The plot is saved to the specified output directory.
def save_rf_classifier_feature_importance_plot(
    importance_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    
    _ensure_out_dir(out_dir)

    # importance_df should have columns 'feature' and 'importance'.sort by importance and create a horizontal bar plot to visualize feature importance for the random forest classifier.
    plot_df = importance_df.sort_values("importance", ascending=True)
    labels = plot_df["feature"].astype(str)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, plot_df["importance"])
    ax.set_title("Random Forest classifier feature importance")
    ax.set_xlabel("importance")
    ax.set_ylabel("feature")
    fig.tight_layout()
    fig.savefig(out_dir / "rf_classifier_feature_importance.png", dpi=200)
    plt.close(fig)

# Save a bar plot of the predicted class distribution (0=mono, 1=dual) from a classifier's predictions, given an array of predicted classes and an output directory to save the plot.
def save_predicted_class_distribution_plot(
    y_pred: np.ndarray | pd.Series,
    out_dir: Path,
    filename: str,
    title: str,
) -> None:
   
    _ensure_out_dir(out_dir)
   
    # y_pred should be an array of predicted class labels (0 for mono-type, 1 for dual-type). We count the occurrences of each predicted class and create a bar plot to visualize the predicted class distribution.
    y_pred_arr = np.asarray(y_pred, dtype=int)
    counts = pd.Series(y_pred_arr).value_counts().sort_index()
    counts = counts.reindex([0, 1], fill_value=0)

    labels = ["mono_type (0)", "dual_type (1)"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts.values)
    ax.set_title(title)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)