"""
Modeling: Model Zoo, Cross-Validated Benchmarking, and Champion Selection.

Responsibilities
----------------
- Define the full model zoo (classical ML + advanced boosting).
- Benchmark every model using 5-fold stratified cross-validation on the
  training set so champion selection is not sensitive to a single split.
- Fit the champion on the full training set for final use.
- Tune the champion's hyperparameters using a custom business-cost scorer
  via GridSearchCV.

Design decisions
----------------
SMOTE placement: SMOTE is applied inside the imblearn Pipeline, AFTER the
preprocessor step but BEFORE the classifier. This guarantees that:
  (a) synthetic samples are generated only from scaled, imputed data, and
  (b) no synthetic data leaks into the validation folds during CV.

Champion selection: models are ranked by mean cross-validated F1 score
(not test-set F1) to avoid selection bias from a single holdout split.
The standard deviation of CV scores is also logged — a model with slightly
lower mean but much lower std is often preferable in production.

Custom scorer: GridSearchCV optimises for MINIMUM total business cost
(FP × $500 + FN × $10,000). greater_is_better=False tells sklearn that
lower values are better.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# --- Config import MUST come before any conditional imports that use logger ---
from src.config import (
    ARTIFACTS_DIR,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. OPTIONAL DEPENDENCY GUARDS
# ---------------------------------------------------------------------------
# imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    IMBLEARN_AVAILABLE = True
    logger.info("imbalanced-learn detected — SMOTE will be applied in pipeline.")
except ImportError:
    from sklearn.pipeline import Pipeline  # type: ignore[assignment]
    IMBLEARN_AVAILABLE = False
    logger.warning(
        "imbalanced-learn not installed. SMOTE will be skipped. "
        "Run: pip install imbalanced-learn"
    )

# Advanced boosting libraries
try:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    ADVANCED_BOOSTING = True
    logger.info("XGBoost / CatBoost / LightGBM detected — advanced models enabled.")
except ImportError:
    ADVANCED_BOOSTING = False
    logger.warning(
        "XGBoost / CatBoost / LightGBM not found. Advanced models will be skipped. "
        "Run: pip install xgboost catboost lightgbm"
    )


# ---------------------------------------------------------------------------
# 2. CUSTOM BUSINESS-COST SCORER
# ---------------------------------------------------------------------------

def total_cost_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate total business cost from a confusion matrix.

    This metric drives hyperparameter tuning. The goal is to MINIMISE cost.

    Cost model
    ----------
    False Negative (missed failure) → $10,000 unplanned downtime
    False Positive (false alarm)    → $500  unnecessary inspection

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (not probabilities).

    Returns
    -------
    float
        Total cost. Returns np.inf if the confusion matrix is degenerate
        (model predicts only one class), penalising such models heavily.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        # Degenerate case: model predicts a single class — penalise heavily
        return np.inf
    tn, fp, fn, tp = cm.ravel()
    return float(fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE)


# greater_is_better=False → sklearn minimises the scorer value
business_cost_scorer = make_scorer(total_cost_metric, greater_is_better=False)


# ---------------------------------------------------------------------------
# 3. HELPER — BUILD A SINGLE FULL PIPELINE
# ---------------------------------------------------------------------------

def _build_pipeline(preprocessor, model) -> Pipeline:
    """Assemble a full imblearn/sklearn Pipeline for one model.

    Pipeline order
    --------------
    preprocessor → (SMOTE if available) → model

    SMOTE must come after the preprocessor so it operates on imputed and
    scaled data. It must come before the model so it is transparent to the
    classifier. When embedded inside cross_validate, sklearn/imblearn fits
    the entire pipeline on each training fold only — SMOTE never sees
    validation-fold samples.

    Parameters
    ----------
    preprocessor:
        Unfitted ColumnTransformer from feature_engineering.get_preprocessor().
    model:
        An unfitted sklearn-compatible classifier.

    Returns
    -------
    Pipeline
        Assembled but unfitted pipeline.
    """
    steps = [("preprocessor", preprocessor)]
    if IMBLEARN_AVAILABLE:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps=steps)


# ---------------------------------------------------------------------------
# 4. MAIN BENCHMARKING FUNCTION
# ---------------------------------------------------------------------------

def train_and_benchmark(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor,
) -> Tuple[Pipeline, str, pd.DataFrame]:
    """Benchmark the full model zoo and select the best champion.

    Methodology
    -----------
    Each model is evaluated via 5-fold stratified cross-validation on
    X_train / y_train. The CV mean F1 score is used for champion selection
    — NOT the holdout test-set score — to ensure the selection is robust
    across different random splits.

    After champion selection, the winning pipeline is re-fitted on the
    ENTIRE training set (X_train, y_train) so it benefits from all available
    labelled data before moving to tuning.

    A held-out test-set evaluation (for the leaderboard display only) is also
    computed so practitioners can sanity-check that CV scores are realistic.

    Parameters
    ----------
    X_train, y_train : training features and labels.
    X_test,  y_test  : held-out test features and labels (read-only here;
                       used only for the leaderboard display).
    preprocessor     : unfitted ColumnTransformer.

    Returns
    -------
    champion_model : Pipeline
        Best pipeline, refitted on the full training set.
    champion_name : str
        Name of the winning model.
    leaderboard : pd.DataFrame
        Full results table sorted by CV F1 (descending).
    """
    logger.info("[4/7] Benchmarking model zoo with 5-fold stratified CV + SMOTE...")

    # --- Define model zoo ---
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "SVC (Prob)": SVC(probability=True, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Gaussian NB": GaussianNB(),
    }

    if ADVANCED_BOOSTING:
        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        models["CatBoost"] = CatBoostClassifier(
            verbose=0,
            random_state=RANDOM_STATE,
        )
        models["LightGBM"] = LGBMClassifier(
            verbosity=-1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    cv_strategy = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )

    cv_scoring = {
        "f1":        "f1",
        "precision": "precision",
        "recall":    "recall",
        "roc_auc":   "roc_auc",
    }

    results_list = []
    trained_models: Dict[str, Pipeline] = {}

    for name, model in models.items():
        logger.info("  Training: %s", name)
        try:
            pipeline = _build_pipeline(preprocessor, model)

            # --- Cross-validated metrics on training set ---
            cv_results = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv_strategy,
                scoring=cv_scoring,
                n_jobs=-1,
                return_train_score=False,
            )

            cv_f1_mean  = cv_results["test_f1"].mean()
            cv_f1_std   = cv_results["test_f1"].std()
            cv_auc_mean = cv_results["test_roc_auc"].mean()

            # --- Re-fit on full training set for test-set display ---
            pipeline.fit(X_train, y_train)
            y_pred_test = pipeline.predict(X_test)

            try:
                y_prob_test = pipeline.predict_proba(X_test)[:, 1]
            except AttributeError:
                logger.warning(
                    "%s does not support predict_proba — ROC-AUC set to 0.", name
                )
                y_prob_test = np.zeros(len(y_test))

            test_f1  = f1_score(y_test, y_pred_test)
            test_auc = roc_auc_score(y_test, y_prob_test)

            results_list.append(
                {
                    "Model":           name,
                    "CV_F1_Mean":      round(cv_f1_mean,  4),
                    "CV_F1_Std":       round(cv_f1_std,   4),
                    "CV_AUC_Mean":     round(cv_auc_mean, 4),
                    "Test_F1":         round(test_f1,     4),
                    "Test_AUC":        round(test_auc,    4),
                    "Accuracy":        round(accuracy_score(y_test, y_pred_test),         4),
                    "Precision":       round(precision_score(y_test, y_pred_test, zero_division=0), 4),
                    "Recall":          round(recall_score(y_test, y_pred_test),            4),
                }
            )
            trained_models[name] = pipeline

        except Exception as exc:
            logger.error("Error training %s: %s", name, exc, exc_info=True)

    if not results_list:
        raise RuntimeError(
            "All models failed to train. Check logs for individual errors."
        )

    # --- Build and save leaderboard ---
    leaderboard = (
        pd.DataFrame(results_list)
        .sort_values(by="CV_F1_Mean", ascending=False)
        .reset_index(drop=True)
    )

    leaderboard_path = ARTIFACTS_DIR / "model_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    logger.info("Leaderboard saved to: %s", leaderboard_path)

    _print_leaderboard(leaderboard)

    # --- Select champion by CV F1 ---
    champion_name = leaderboard.iloc[0]["Model"]
    champion_model = trained_models[champion_name]

    logger.info(
        "Champion: %s | CV F1: %.4f ± %.4f | Test F1: %.4f",
        champion_name,
        leaderboard.iloc[0]["CV_F1_Mean"],
        leaderboard.iloc[0]["CV_F1_Std"],
        leaderboard.iloc[0]["Test_F1"],
    )

    return champion_model, champion_name, leaderboard


# ---------------------------------------------------------------------------
# 5. HYPERPARAMETER TUNING
# ---------------------------------------------------------------------------

def tune_champion_model(
    model_pipeline: Pipeline,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Tune the champion pipeline using GridSearchCV with the business-cost scorer.

    The tuning objective is to MINIMISE total business cost
    (FP × $500 + FN × $10,000) rather than maximising a symmetric metric
    like F1. This reflects the real-world asymmetry: missing a failure is
    20× more expensive than a false alarm.

    Param grids are defined for the four most commonly winning models.
    If the champion is not in the grid (e.g. GaussianNB), the original
    pipeline is returned with a clear log message — it is NOT silently
    labelled as "tuned".

    Parameters
    ----------
    model_pipeline : Pipeline
        Champion pipeline fitted on the full training set.
    model_name : str
        Name of the champion model (used to look up the param grid).
    X_train, y_train : full training data for refitting.

    Returns
    -------
    Pipeline
        Either the best estimator from GridSearchCV, or the original
        pipeline if no grid is defined for this model.
    """
    logger.info("[5/7] Hyperparameter tuning: %s", model_name)

    param_grids: Dict[str, Dict] = {
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth":    [10, 20, None],
            "model__min_samples_split": [2, 5],
        },
        "XGBoost": {
            "model__n_estimators":  [100, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth":     [3, 5, 6],
            "model__subsample":     [0.8, 1.0],
        },
        "CatBoost": {
            "model__iterations":    [200, 500],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__depth":         [4, 6, 8],
        },
        "LightGBM": {
            "model__n_estimators":  [100, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth":     [3, 5, -1],
            "model__num_leaves":    [31, 63],
        },
        "Gradient Boosting": {
            "model__n_estimators":  [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth":     [3, 5],
        },
        "Logistic Regression": {
            "model__C":       [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
        },
    }

    # Match champion name against the grid keys (substring match for safety)
    grid_params: Optional[Dict] = None
    for key, params in param_grids.items():
        if key in model_name:
            grid_params = params
            break

    if grid_params is None:
        logger.info(
            "No tuning grid defined for '%s'. "
            "Returning base champion (untuned — metrics will be reported as-is).",
            model_name,
        )
        return model_pipeline

    logger.info(
        "Optimising for MINIMUM BUSINESS COST (FP×$%s + FN×$%s)...",
        f"{COST_FALSE_POSITIVE:,}",
        f"{COST_FALSE_NEGATIVE:,}",
    )

    cv_strategy = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=grid_params,
        scoring=business_cost_scorer,
        cv=cv_strategy,
        n_jobs=-1,
        verbose=1,
        refit=True,   # refit best estimator on full X_train after search
        error_score="raise",
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_cost   = -grid_search.best_score_   # negate: scorer stores negated cost

    logger.info("Tuning complete.")
    logger.info("Best params : %s", best_params)
    logger.info("Best CV cost: $%s", f"{best_cost:,.0f}")

    return grid_search.best_estimator_


# ---------------------------------------------------------------------------
# 6. PRIVATE HELPERS
# ---------------------------------------------------------------------------

def _print_leaderboard(leaderboard: pd.DataFrame) -> None:
    """Pretty-print the model leaderboard to stdout."""
    separator = "=" * 100
    print(f"\n{separator}")
    print("   MASTER MODEL LEADERBOARD  (ranked by 5-fold CV F1 — higher is better)")
    print(separator)
    print(leaderboard.to_string(index=True))
    print(f"{separator}\n")