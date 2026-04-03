"""
Evaluation: Threshold optimisation, metrics, plots, and model persistence.

Responsibilities
----------------
- Optimise the decision threshold on the VALIDATION set (X_val / y_val).
  The threshold is found by minimising total business cost across a fine
  grid of probability cutoffs.
- Report final metrics on the TEST set (X_test / y_test) using the
  optimised threshold. The test set is touched exactly once — here.
- Save confusion matrix, ROC curve, and feature-importance plots to disk.
- Persist the final production model with joblib.

Why separate val/test?
----------------------
If the threshold were optimised on the test set, the reported cost would
be the minimum achievable on that specific sample — an overly optimistic
figure that does not generalise. Using a held-out validation set for the
search and the test set for reporting gives an unbiased estimate of the
cost you'd expect to see in production.
"""

import logging
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)

from src.config import (
    ARTIFACTS_DIR,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. THRESHOLD OPTIMISATION  (runs on VALIDATION set only)
# ---------------------------------------------------------------------------

def optimize_threshold(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold_step: float = 0.01,
) -> Tuple[float, float]:
    """Find the probability threshold that minimises total business cost.

    The search is performed on the VALIDATION set so that the TEST set
    remains a pristine, unbiased benchmark.

    Parameters
    ----------
    model :
        Fitted sklearn-compatible pipeline with a predict_proba method.
    X_val, y_val :
        Validation features and labels — used ONLY for threshold search.
    threshold_step :
        Granularity of the threshold grid. Default 0.01 (101 candidates).

    Returns
    -------
    best_threshold : float
        Probability cutoff that minimises cost on the validation set.
    min_cost : float
        Business cost achieved at best_threshold on the validation set.

    Raises
    ------
    ValueError
        If the model does not support predict_proba.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Threshold optimisation requires a model with predict_proba. "
            "Ensure probability=True is set for SVC."
        )

    y_prob_val = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.0, 1.0 + threshold_step, threshold_step)
    costs = []

    for t in thresholds:
        y_pred_t = (y_prob_val >= t).astype(int)
        cm = confusion_matrix(y_val, y_pred_t)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            cost = fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE
        else:
            # Degenerate prediction (single class) — assign worst-case cost
            cost = np.inf
        costs.append(cost)

    min_cost_idx  = int(np.argmin(costs))
    best_threshold = float(thresholds[min_cost_idx])
    min_cost       = float(costs[min_cost_idx])

    logger.info(
        "Threshold optimised on validation set — "
        "best threshold: %.2f | val cost: $%s",
        best_threshold,
        f"{min_cost:,.0f}",
    )
    return best_threshold, min_cost


# ---------------------------------------------------------------------------
# 2. MAIN EVALUATION FUNCTION  (reports on TEST set)
# ---------------------------------------------------------------------------

def evaluate_and_plot(
    model,
    model_name: str,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    """Evaluate the tuned model and persist all diagnostic artefacts.

    Workflow
    --------
    1. Find optimal threshold on (X_val, y_val) — threshold search.
    2. Apply that threshold to (X_test, y_test) — unbiased final report.
    3. Print classification report and business-impact summary.
    4. Save confusion matrix, ROC curve, and feature-importance plots.

    Parameters
    ----------
    model :
        Tuned, fitted pipeline returned by modeling.tune_champion_model.
    model_name : str
        Human-readable model name used in plot titles and log messages.
    X_val, y_val :
        Validation set — used ONLY for threshold optimisation.
    X_test, y_test :
        Test set — used ONLY for final metric reporting. Never used
        before this function is called in the pipeline.
    """
    logger.info("[6/7] Evaluating champion '%s'...", model_name)

    # --- Step 1: Threshold optimisation on VALIDATION set ---
    best_thresh, val_cost = optimize_threshold(model, X_val, y_val)

    # --- Step 2: Final predictions on TEST set ---
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= best_thresh).astype(int)

    # --- Step 3: Classification report ---
    separator = "=" * 70
    print(f"\n{separator}")
    print(
        f"  FINAL TEST-SET REPORT  |  Model: {model_name}  |  "
        f"Threshold: {best_thresh:.2f}"
    )
    print(separator)
    print(classification_report(y_test, y_pred_test, digits=4))

    # --- Step 4: Business impact summary ---
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    total_cost = fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE

    fn_cost = fn * COST_FALSE_NEGATIVE
    fp_cost = fp * COST_FALSE_POSITIVE

    print(f"\n  BUSINESS IMPACT ANALYSIS (test set):")
    print(f"    Optimal threshold (from val set)  : {best_thresh:.2f}")
    print(f"    False Negatives  (missed failures): {fn:>4d}  →  cost ${fn_cost:>10,.0f}")
    print(f"    False Positives  (false alarms)   : {fp:>4d}  →  cost ${fp_cost:>10,.0f}")
    print(f"    ─────────────────────────────────────────────────────")
    print(f"    TOTAL PROJECTED COST              :        ${total_cost:>10,.0f}")
    print(f"{separator}\n")

    # --- Step 5: Save plots ---
    _plot_confusion_matrix(cm, model_name, best_thresh, total_cost)
    _plot_roc_curve(y_test, y_prob_test, model_name)
    _plot_feature_importance(model, model_name)

    logger.info(
        "Evaluation complete. Test cost: $%s | Threshold: %.2f",
        f"{total_cost:,.0f}",
        best_thresh,
    )


# ---------------------------------------------------------------------------
# 3. MODEL PERSISTENCE
# ---------------------------------------------------------------------------

def save_model(model, model_name: str) -> None:
    """Serialise the final production model to disk with joblib.

    The model is saved under artifacts/models/<safe_name>_champion.pkl.
    The models directory is created defensively in case config.py's
    directory setup was somehow bypassed.

    Parameters
    ----------
    model :
        Tuned, fitted pipeline to persist.
    model_name : str
        Human-readable name; spaces are replaced with underscores.
    """
    safe_name = model_name.replace(" ", "_").lower()
    model_dir = ARTIFACTS_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    path = model_dir / f"{safe_name}_champion.pkl"

    try:
        joblib.dump(model, path)
        logger.info("[7/7] Production model saved: %s", path)
    except Exception as exc:
        raise IOError(
            f"Failed to save model to {path}. "
            f"Check disk space and write permissions.\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# 4. PRIVATE PLOT HELPERS
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    threshold: float,
    total_cost: float,
) -> None:
    """Save a confusion-matrix heatmap to the graphs directory."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        annot_kws={"size": 14},
    )
    ax.set_title(
        f"Confusion Matrix — {model_name}\n"
        f"Threshold: {threshold:.2f}  |  Projected cost: ${total_cost:,.0f}",
        fontsize=13,
        pad=12,
    )
    ax.set_ylabel("Actual label",    fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_xticklabels(["No Failure (0)", "Failure (1)"], fontsize=10)
    ax.set_yticklabels(["No Failure (0)", "Failure (1)"], fontsize=10, rotation=0)

    save_path = ARTIFACTS_DIR / "graphs" / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", save_path)


def _plot_roc_curve(
    y_test: pd.Series,
    y_prob: np.ndarray,
    model_name: str,
) -> None:
    """Save a ROC curve plot to the graphs directory."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        fpr, tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    save_path = ARTIFACTS_DIR / "graphs" / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("ROC curve saved: %s", save_path)


def _plot_feature_importance(model, model_name: str, top_n: int = 10) -> None:
    """Save a horizontal bar chart of the top-N most important features.

    Supports tree-based models (feature_importances_) and linear models
    (coef_). Logs a warning and exits cleanly if the model type supports
    neither, so the pipeline does not crash on unsupported models.

    Parameters
    ----------
    model :
        Fitted pipeline (must contain 'preprocessor' and 'model' named steps).
    model_name : str
        Used in the plot title.
    top_n : int
        Number of top features to display.
    """
    try:
        classifier    = model.named_steps["model"]
        preprocessor  = model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()

        importances: Optional[np.ndarray] = None  # noqa: F821

        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])

        if importances is None:
            logger.warning(
                "Feature importance plot skipped — '%s' exposes neither "
                "feature_importances_ nor coef_.",
                model_name,
            )
            return

        if len(importances) != len(feature_names):
            logger.warning(
                "Feature importance length mismatch (%d importances vs %d names). "
                "Skipping plot.",
                len(importances),
                len(feature_names),
            )
            return

        # Build a tidy DataFrame for clean plotting
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .sort_values("importance", ascending=True)   # ascending for horizontal bar
        )

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(
            importance_df["feature"],
            importance_df["importance"],
            color=sns.color_palette("viridis", len(importance_df)),
        )
        ax.set_xlabel("Importance score", fontsize=12)
        ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13)
        ax.grid(axis="x", alpha=0.3)

        save_path = ARTIFACTS_DIR / "graphs" / "feature_importance.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info("Feature importance plot saved: %s", save_path)

    except Exception as exc:
        logger.warning(
            "Could not generate feature importance plot: %s", exc, exc_info=True
        )