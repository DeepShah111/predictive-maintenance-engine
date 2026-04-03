"""
Data ingestion and cleaning module.

Responsibilities
----------------
- Download the raw CSV from Google Drive if not already cached locally.
- Perform lightweight cleaning (deduplication, index reset, schema validation).
- Return a clean DataFrame ready for feature engineering.

No feature creation or splitting happens here — single responsibility.
"""

import logging
from pathlib import Path

import pandas as pd
import gdown

from src.config import DOWNLOAD_URL, FILEPATH, TARGET_COL, LEAKAGE_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected raw columns — used for schema validation after loading.
# If the dataset changes format, the pipeline fails fast with a clear message
# rather than silently producing wrong results downstream.
# ---------------------------------------------------------------------------
_REQUIRED_COLUMNS: list[str] = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    TARGET_COL,
    # Failure-mode sub-flags (dropped later as leakage, but must be present)
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def load_data(
    filepath: Path = FILEPATH,
    url: str = DOWNLOAD_URL,
) -> pd.DataFrame:
    """Download (if needed) and load the AI4I 2020 dataset.

    The file is cached at *filepath* after the first download so that
    subsequent runs do not hit the network.

    Parameters
    ----------
    filepath:
        Local path where the CSV is stored / should be stored.
    url:
        Google Drive direct-download URL for the dataset.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified dataset as loaded from disk.

    Raises
    ------
    RuntimeError
        If the download fails or the file cannot be read.
    FileNotFoundError
        If the file is missing after a download attempt.
    ValueError
        If the loaded CSV is missing required columns.
    """
    logger.info("[1/7] Loading data source...")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --- Fetch from Google Drive if not cached ---
    if filepath.exists():
        logger.info("Cache hit — loading from local file: %s", filepath)
    else:
        logger.info("Cache miss — downloading dataset from Google Drive...")
        try:
            gdown.download(url, str(filepath), quiet=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download dataset from {url}. "
                f"Check your internet connection or the FILE_ID in config.py.\n"
                f"Original error: {exc}"
            ) from exc

        if not filepath.exists():
            raise FileNotFoundError(
                f"Download appeared to succeed but file not found at: {filepath}"
            )

    # --- Read CSV ---
    try:
        raw_df = pd.read_csv(filepath)
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse CSV at {filepath}. "
            f"The file may be corrupt — delete it and re-run to trigger a fresh download.\n"
            f"Original error: {exc}"
        ) from exc

    logger.info("Raw data loaded. Shape: %s", raw_df.shape)

    # --- Schema validation (fail fast) ---
    _validate_schema(raw_df)

    return raw_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform lightweight, dataset-agnostic cleaning.

    Steps
    -----
    1. Drop exact duplicate rows.
    2. Report and log null counts per column (does NOT impute — that belongs
       inside the sklearn preprocessing pipeline to prevent leakage).
    3. Validate that the target column contains only binary values {0, 1}.
    4. Reset index so downstream CV splitting indices are contiguous.

    Parameters
    ----------
    df:
        Raw DataFrame returned by :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    logger.info("[2/7] Cleaning data...")
    cleaned = df.copy()

    # 1. Deduplication
    n_before = len(cleaned)
    cleaned.drop_duplicates(inplace=True)
    n_dropped = n_before - len(cleaned)
    if n_dropped > 0:
        logger.info("Dropped %d duplicate rows.", n_dropped)
    else:
        logger.info("No duplicate rows found.")

    # 2. Null audit — log only; imputation happens inside the sklearn Pipeline
    null_counts = cleaned.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        logger.warning(
            "Null values detected (will be handled by pipeline imputer):\n%s",
            cols_with_nulls.to_string(),
        )
    else:
        logger.info("No null values found in dataset.")

    # 3. Target column sanity check
    unique_target_vals = set(cleaned[TARGET_COL].unique())
    if not unique_target_vals.issubset({0, 1}):
        raise ValueError(
            f"Target column '{TARGET_COL}' must contain only {{0, 1}}. "
            f"Found: {unique_target_vals}"
        )
    class_counts = cleaned[TARGET_COL].value_counts()
    logger.info(
        "Target distribution — class 0: %d (%.1f%%), class 1: %d (%.1f%%)",
        class_counts.get(0, 0),
        100 * class_counts.get(0, 0) / len(cleaned),
        class_counts.get(1, 0),
        100 * class_counts.get(1, 0) / len(cleaned),
    )

    # 4. Reset index — contiguous indices are required by StratifiedKFold
    cleaned.reset_index(drop=True, inplace=True)

    logger.info("Cleaning complete. Final shape: %s", cleaned.shape)
    return cleaned


# ---------------------------------------------------------------------------
# PRIVATE HELPERS
# ---------------------------------------------------------------------------

def _validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing from *df*.

    Parameters
    ----------
    df:
        DataFrame to validate.

    Raises
    ------
    ValueError
        Lists every missing column so the user knows exactly what to fix.
    """
    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing {len(missing)} required column(s): {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    logger.info("Schema validation passed. All %d required columns present.", len(_REQUIRED_COLUMNS))