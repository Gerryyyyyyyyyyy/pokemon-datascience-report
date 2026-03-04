from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def save_pred_vs_actual_scatter(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    out_dir: Path,
    filename: str,
    title: str,
) -> None:
    
    _ensure_out_dir(out_dir)

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_arr, y_pred_arr)

    # Identity line (perfect predictions)
    min_val = min(y_true_arr.min(), y_pred_arr.min())
    max_val = max(y_true_arr.max(), y_pred_arr.max())
    ax.plot([min_val, max_val], [min_val, max_val])

    ax.set_title(title)
    ax.set_xlabel("Actual base_experience")
    ax.set_ylabel("Predicted base_experience")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)


def save_rf_feature_importance_plot(
    importance_df: pd.DataFrame, out_dir: Path
) -> None:
   
    _ensure_out_dir(out_dir)

    plot_df = importance_df.sort_values("importance", ascending=True)
    labels = plot_df["feature"].astype(str)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, plot_df["importance"])
    ax.set_title("Random Forest feature importance")
    ax.set_xlabel("importance")
    ax.set_ylabel("feature")
    fig.tight_layout()
    fig.savefig(out_dir / "rf_feature_importance.png", dpi=200)
    plt.close(fig)