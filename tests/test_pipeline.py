"""
Unit tests for the AI4I Predictive Maintenance Pipeline.

Run with:
    pytest tests/ -v

Each test is self-contained and does not require Google Drive, a GPU,
or any downloaded data. All inputs are constructed inline using small
synthetic DataFrames that mirror the shape and types of the real dataset.

Test coverage
-------------
1.  Physics feature creation — correct column names and values.
2.  Physics features — infinity sanitisation.
3.  Leakage column removal — all leakage cols are dropped.
4.  Preprocessor — builds without error (smoke test).
5.  Clean data — duplicate removal.
6.  Clean data — index is reset to contiguous integers after dedup.
7.  Three-way split — correct number of return values.
8.  Three-way split — sizes add up to the input size.
9.  Three-way split — class-balance is preserved across all three splits.
10. Business cost metric — correct calculation on a known confusion matrix.
11. Business cost metric — returns inf for degenerate single-class predictions.
12. Schema validation — raises ValueError on missing columns.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline as SklearnPipeline

# ---------------------------------------------------------------------------
# Helpers — build minimal synthetic DataFrames that look like the real dataset
# ---------------------------------------------------------------------------

def _make_raw_df(n: int = 200, failure_rate: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic DataFrame with the same schema as ai4i2020.csv."""
    rng = np.random.default_rng(seed)
    n_failures = max(2, int(n * failure_rate))
    y = np.zeros(n, dtype=int)
    y[:n_failures] = 1
    rng.shuffle(y)

    return pd.DataFrame({
        "UDI":                          np.arange(1, n + 1),
        "Product ID":                   [f"L{i}" for i in range(n)],
        "Type":                         rng.choice(["L", "M", "H"], size=n),
        "Air temperature [K]":          rng.uniform(295, 305, size=n),
        "Process temperature [K]":      rng.uniform(305, 315, size=n),
        "Rotational speed [rpm]":       rng.uniform(1168, 2886, size=n),
        "Torque [Nm]":                  rng.uniform(3.8, 76.6, size=n),
        "Tool wear [min]":              rng.uniform(0, 253, size=n),
        "Machine failure":              y,
        "TWF":                          y,   # failure-mode sub-flags mirror target
        "HDF":                          y,
        "PWF":                          y,
        "OSF":                          y,
        "RNF":                          y,
    })


# ---------------------------------------------------------------------------
# 1 — Physics feature creation: correct column names and values
# ---------------------------------------------------------------------------

def test_physics_features_columns_created():
    from src.feature_engineering import create_physics_features
    df = _make_raw_df(n=50)
    result = create_physics_features(df)
    for col in ("Temp_Diff", "Power", "Force_Ratio"):
        assert col in result.columns, f"Expected column '{col}' not found."


def test_physics_features_temp_diff_value():
    from src.feature_engineering import create_physics_features
    df = pd.DataFrame({
        "Air temperature [K]":     [300.0],
        "Process temperature [K]": [310.0],
        "Rotational speed [rpm]":  [1500.0],
        "Torque [Nm]":             [40.0],
    })
    result = create_physics_features(df)
    assert result["Temp_Diff"].iloc[0] == pytest.approx(10.0, abs=1e-6)


def test_physics_features_power_value():
    from src.feature_engineering import create_physics_features
    df = pd.DataFrame({
        "Air temperature [K]":     [300.0],
        "Process temperature [K]": [310.0],
        "Rotational speed [rpm]":  [2000.0],
        "Torque [Nm]":             [50.0],
    })
    result = create_physics_features(df)
    # Power = Torque × RPM = 50 × 2000 = 100,000
    assert result["Power"].iloc[0] == pytest.approx(100_000.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 2 — Physics features: infinity sanitisation
# ---------------------------------------------------------------------------

def test_physics_features_no_infinities():
    from src.feature_engineering import create_physics_features
    df = _make_raw_df(n=100)
    # Force extremely high torque to provoke potential overflow
    df["Torque [Nm]"] = 1e15
    result = create_physics_features(df)
    assert not np.isinf(result["Force_Ratio"]).any(), "Force_Ratio contains inf."
    assert not np.isinf(result["Power"]).any(), "Power contains inf."


# ---------------------------------------------------------------------------
# 3 — Leakage column removal
# ---------------------------------------------------------------------------

def test_leakage_cols_dropped_after_split():
    from src.config import LEAKAGE_COLS
    from src.feature_engineering import build_features_and_split
    df = _make_raw_df(n=300)
    X_train, X_val, X_test, y_train, y_val, y_test = build_features_and_split(df)
    for col in LEAKAGE_COLS:
        assert col not in X_train.columns, f"Leakage col '{col}' still in X_train."
        assert col not in X_val.columns,   f"Leakage col '{col}' still in X_val."
        assert col not in X_test.columns,  f"Leakage col '{col}' still in X_test."


# ---------------------------------------------------------------------------
# 4 — Preprocessor smoke test
# ---------------------------------------------------------------------------

def test_get_preprocessor_returns_column_transformer():
    from sklearn.compose import ColumnTransformer
    from src.feature_engineering import get_preprocessor
    preprocessor = get_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)


# ---------------------------------------------------------------------------
# 5 — clean_data: duplicate removal
# ---------------------------------------------------------------------------

def test_clean_data_removes_duplicates():
    from src.data_ingestion import clean_data
    df = _make_raw_df(n=20)
    # Manually duplicate the first 5 rows
    df_with_dups = pd.concat([df, df.iloc[:5]], ignore_index=True)
    cleaned = clean_data(df_with_dups)
    assert len(cleaned) == len(df), (
        f"Expected {len(df)} rows after dedup, got {len(cleaned)}."
    )


# ---------------------------------------------------------------------------
# 6 — clean_data: index is contiguous after deduplication
# ---------------------------------------------------------------------------

def test_clean_data_index_is_contiguous():
    from src.data_ingestion import clean_data
    df = _make_raw_df(n=30)
    df_with_dups = pd.concat([df, df.iloc[:5]], ignore_index=True)
    cleaned = clean_data(df_with_dups)
    expected_index = list(range(len(cleaned)))
    assert cleaned.index.tolist() == expected_index, (
        "Index is not contiguous after clean_data."
    )


# ---------------------------------------------------------------------------
# 7 — Three-way split: correct return signature (6 objects)
# ---------------------------------------------------------------------------

def test_build_features_and_split_returns_six_objects():
    from src.feature_engineering import build_features_and_split
    df = _make_raw_df(n=300)
    result = build_features_and_split(df)
    assert len(result) == 6, f"Expected 6 return values, got {len(result)}."


# ---------------------------------------------------------------------------
# 8 — Three-way split: sizes add up
# ---------------------------------------------------------------------------

def test_build_features_and_split_sizes():
    from src.feature_engineering import build_features_and_split
    df = _make_raw_df(n=500)
    X_train, X_val, X_test, y_train, y_val, y_test = build_features_and_split(df)
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(df), (
        f"Split sizes sum to {total} but input has {len(df)} rows."
    )


# ---------------------------------------------------------------------------
# 9 — Three-way split: class balance preserved across all splits
# ---------------------------------------------------------------------------

def test_build_features_and_split_class_balance():
    from src.feature_engineering import build_features_and_split
    # Use a large n so stratification can work accurately
    df = _make_raw_df(n=1000, failure_rate=0.05)
    X_train, X_val, X_test, y_train, y_val, y_test = build_features_and_split(df)
    original_rate = df["Machine failure"].mean()
    tolerance = 0.03   # allow ±3 percentage points
    for split_y, name in [(y_train, "train"), (y_val, "val"), (y_test, "test")]:
        rate = split_y.mean()
        assert abs(rate - original_rate) < tolerance, (
            f"Class-1 rate in '{name}' split ({rate:.3f}) deviates more than "
            f"{tolerance} from original ({original_rate:.3f})."
        )


# ---------------------------------------------------------------------------
# 10 — Business cost metric: correct calculation
# ---------------------------------------------------------------------------

def test_total_cost_metric_correct_value():
    from src.modeling import total_cost_metric
    from src.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE

    # Construct a known confusion matrix:
    # TN=90, FP=5, FN=3, TP=2  (4 classes: 0,0 → TN; 0,1 → FP; 1,0 → FN; 1,1 → TP)
    y_true = np.array([0]*90 + [0]*5 + [1]*3 + [1]*2)
    y_pred = np.array([0]*90 + [1]*5 + [0]*3 + [1]*2)

    expected_cost = (5 * COST_FALSE_POSITIVE) + (3 * COST_FALSE_NEGATIVE)
    result = total_cost_metric(y_true, y_pred)
    assert result == pytest.approx(expected_cost), (
        f"Expected cost {expected_cost}, got {result}."
    )


# ---------------------------------------------------------------------------
# 11 — Business cost metric: degenerate prediction returns inf
# ---------------------------------------------------------------------------

# REPLACE WITH THIS:
def test_total_cost_metric_degenerate_returns_inf():
    from src.modeling import total_cost_metric
    # Degenerate case: y_true has only one class → confusion matrix is 1×1
    # This happens when an entire fold contains no positive samples
    y_true = np.array([0, 0, 0, 0, 0])   # only class 0 in ground truth
    y_pred = np.array([0, 0, 0, 0, 0])
    result = total_cost_metric(y_true, y_pred)
    assert result == np.inf, (
        f"Expected np.inf for degenerate prediction, got {result}."
    )

# ---------------------------------------------------------------------------
# 12 — Schema validation: missing columns raise ValueError
# ---------------------------------------------------------------------------

def test_schema_validation_raises_on_missing_columns():
    from src.data_ingestion import _validate_schema
    # DataFrame with a critical column missing
    incomplete_df = pd.DataFrame({
        "UDI": [1, 2],
        "Type": ["L", "M"],
        # Deliberately omitting all sensor columns and the target
    })
    with pytest.raises(ValueError, match="missing"):
        _validate_schema(incomplete_df)