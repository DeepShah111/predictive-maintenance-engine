"""
Feature engineering: Physics-based feature creation and preprocessing pipeline.

Responsibilities
----------------
- Engineer domain-informed features from raw sensor readings.
- Build a reproducible sklearn ColumnTransformer (preprocessor) that is
  fitted ONLY on training data — never on validation or test data.
- Perform a clean 60 / 20 / 20 stratified three-way split:
    • X_train / y_train  → model fitting and cross-validation
    • X_val  / y_val     → threshold optimisation (ONLY)
    • X_test / y_test    → final evaluation reporting (NEVER touched earlier)

Why three splits?
-----------------
Searching for the optimal decision threshold on the test set and then
reporting metrics on the same test set is a form of data leakage — the
threshold has been 'informed' by the test labels, producing optimistically
biased metrics. The validation split is used exclusively for threshold
search so the test split remains a truly held-out, unbiased benchmark.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.config import (
    CAT_FEATURES,
    LEAKAGE_COLS,
    NUM_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

# Type alias for the six-tuple returned by build_features_and_split
SplitData = Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series,    pd.Series,    pd.Series,
]


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def create_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer three physics-motivated features from raw sensor readings.

    Feature rationale
    -----------------
    Temp_Diff   : Process temperature − Air temperature.
                  A rising thermal gradient signals increased heat retention
                  which is a known precursor to tool-wear failures.

    Power       : Torque [Nm] × Rotational speed [rpm].
                  Mechanical power input to the spindle. Sustained high power
                  correlates with accelerated wear on cutting tools.

    Force_Ratio : Torque [Nm] / (Rotational speed [rpm] + ε).
                  Approximates the load experienced per revolution. A high
                  ratio at low speed indicates heavy cutting conditions.
                  The ε = 1e-5 guard prevents division-by-zero; it has no
                  meaningful effect because RPM values in this dataset are
                  in the range 1168–2886.

    Parameters
    ----------
    df:
        Cleaned DataFrame containing the raw sensor columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with three additional columns appended.
    """
    logger.info("Generating physics-based features...")
    df_eng = df.copy()

    df_eng["Temp_Diff"] = (
        df_eng["Process temperature [K]"] - df_eng["Air temperature [K]"]
    )

    df_eng["Power"] = df_eng["Torque [Nm]"] * df_eng["Rotational speed [rpm]"]

    df_eng["Force_Ratio"] = df_eng["Torque [Nm]"] / (
        df_eng["Rotational speed [rpm]"] + 1e-5
    )

    # Sanitise any infinities that may arise from extreme sensor readings
    df_eng.replace([np.inf, -np.inf], 0, inplace=True)

    logger.info(
        "Physics features created: Temp_Diff, Power, Force_Ratio. "
        "New shape: %s",
        df_eng.shape,
    )
    return df_eng


def get_preprocessor() -> ColumnTransformer:
    """Build and return the sklearn ColumnTransformer (unfitted).

    Numerical pipeline  : median imputation → StandardScaler.
    Categorical pipeline: mode imputation   → OrdinalEncoder.

    Why OrdinalEncoder for 'Type'?
    ------------------------------
    The 'Type' column encodes a genuine quality tier: L (Low) < M (Medium)
    < H (High). This natural ordinal relationship is explicitly specified via
    the `categories` argument, making the encoded integers meaningful (0, 1, 2)
    rather than arbitrary. Tree-based models exploit this ordering directly;
    linear models benefit from the single-column compactness. If the dataset
    were to include a nominal categorical (no ordering), OneHotEncoder would
    be the correct choice instead.

    Returns
    -------
    ColumnTransformer
        An unfitted preprocessor. It must be embedded inside a Pipeline
        that is fitted only on training data.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[["L", "M", "H"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUM_FEATURES),
            ("cat", cat_pipeline, CAT_FEATURES),
        ],
        remainder="drop",   # explicitly drop any column not listed above
    )

    return preprocessor


def build_features_and_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> SplitData:
    """Engineer features and produce a clean three-way stratified split.

    Split strategy
    --------------
    The dataset is divided into three non-overlapping subsets:

        Train (60 %)  → fit models and cross-validation
        Val   (20 %)  → threshold optimisation ONLY
        Test  (20 %)  → final, unbiased metric reporting

    All splits are stratified on *target_col* to preserve the minority-class
    ratio (~3.4 %) across all three subsets, which is critical for an
    imbalanced classification task.

    Parameters
    ----------
    df:
        Cleaned DataFrame returned by :func:`~data_ingestion.clean_data`.
    target_col:
        Name of the binary target column.

    Returns
    -------
    tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        Six objects: three feature DataFrames and three target Series.

    Raises
    ------
    ValueError
        If *target_col* is not present in *df* after leakage columns are
        dropped, or if feature columns defined in config are missing.
    """
    logger.info("[3/7] Engineering features and creating train/val/test split...")

    # --- Step 1: Physics feature creation ---
    df_enriched = create_physics_features(df)

    # --- Step 2: Drop leakage columns (identifiers + failure-mode sub-flags) ---
    df_model = df_enriched.drop(columns=LEAKAGE_COLS, errors="ignore")

    # --- Step 3: Validate that all configured feature columns are present ---
    all_features = NUM_FEATURES + CAT_FEATURES
    missing_features = [f for f in all_features if f not in df_model.columns]
    if missing_features:
        raise ValueError(
            f"The following feature columns are missing after leakage drop: "
            f"{missing_features}. Check NUM_FEATURES / CAT_FEATURES in config.py."
        )

    if target_col not in df_model.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {df_model.columns.tolist()}"
        )

    # --- Step 4: Separate features and target ---
    X = df_model[all_features]   # only configured features — no stray columns
    y = df_model[target_col]

    # --- Step 5: Three-way stratified split ---
    # First split: separate out the test set (20 %)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Second split: from the remaining 80 %, carve out the validation set.
    # val_ratio_of_temp = 20% of total / 80% of total = 0.25
    val_ratio_of_temp = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio_of_temp,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    logger.info(
        "Split complete — Train: %s | Val: %s | Test: %s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )
    logger.info(
        "Class-1 rate — Train: %.2f%% | Val: %.2f%% | Test: %.2f%%",
        100 * y_train.mean(),
        100 * y_val.mean(),
        100 * y_test.mean(),
    )

    return X_train, X_val, X_test, y_train, y_val, y_test