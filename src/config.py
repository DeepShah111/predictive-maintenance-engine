"""
Configuration settings for the AI4I Predictive Maintenance Project.

All constants, paths, logging setup, and business parameters are
centralized here. No other module should hardcode any of these values.
"""

import logging
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# 1. GLOBAL PLOT & WARNING SETTINGS
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 300

# ---------------------------------------------------------------------------
# 2. REPRODUCIBILITY
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# 3. DIRECTORY SETUP
# ---------------------------------------------------------------------------
# ARTIFACTS_DIR is resolved relative to the project root so the pipeline
# works regardless of the working directory the caller uses.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR: Path = _PROJECT_ROOT / "artifacts"

_SUB_DIRS = ["graphs", "models", "data"]
for _d in _SUB_DIRS:
    (ARTIFACTS_DIR / _d).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 4. LOGGING
# The root logger is configured once here. Every other module calls
# logging.getLogger(__name__) and inherits these handlers automatically.
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
_log_file = ARTIFACTS_DIR / "pipeline_run.log"

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info("Config loaded. Project root: %s", _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# 5. DATA SOURCE CONSTANTS
# Google Drive file ID is kept here so it is the single place to update
# when the dataset changes. The filepath is derived from ARTIFACTS_DIR so
# it is always consistent with the directory setup above.
# ---------------------------------------------------------------------------
FILE_ID: str = "1lbc7JuoDQMTRengP2CVtnl-oohz-IE8x"
DOWNLOAD_URL: str = f"https://drive.google.com/uc?id={FILE_ID}"
FILEPATH: Path = ARTIFACTS_DIR / "data" / "ai4i2020.csv"

# ---------------------------------------------------------------------------
# 6. MODELLING CONSTANTS
# ---------------------------------------------------------------------------
TARGET_COL: str = "Machine failure"

# Split ratios  →  60 % train | 20 % validation | 20 % test
# Validation set is used ONLY for threshold optimisation.
# Test set is used ONLY for final reporting — never for any decision making.
TRAIN_SIZE: float = 0.60
VAL_SIZE: float = 0.20   # fraction of the FULL dataset
TEST_SIZE: float = 0.20  # fraction of the FULL dataset

# ---------------------------------------------------------------------------
# 7. BUSINESS COST PARAMETERS
# ---------------------------------------------------------------------------
# A missed machine failure (False Negative) → unplanned downtime ≈ $10,000
# A false alarm       (False Positive)      → unnecessary inspection ≈ $500
# These drive threshold optimisation and the custom business-cost scorer.
COST_FALSE_NEGATIVE: int = 10_000
COST_FALSE_POSITIVE: int = 500

# ---------------------------------------------------------------------------
# 8. FEATURE DEFINITIONS
# ---------------------------------------------------------------------------
# Columns that must be dropped BEFORE modelling:
#   - UDI, Product ID  → row identifiers (no predictive value)
#   - TWF, HDF, PWF, OSF, RNF → individual failure-mode flags that are
#     *consequences* of the target variable; keeping them would be target
#     leakage (they are only set to 1 when Machine failure = 1).
LEAKAGE_COLS: list[str] = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

# 'Type' is genuinely ordinal (L < M < H quality tier) so OrdinalEncoder
# is the correct choice here. A comment is placed in feature_engineering.py
# as well to make the rationale explicit during code review.
CAT_FEATURES: list[str] = ["Type"]

NUM_FEATURES: list[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    # --- Physics-derived features (engineered in feature_engineering.py) ---
    "Temp_Diff",    # Process temp − Air temp  (thermal gradient proxy)
    "Power",        # Torque × RPM             (mechanical power input)
    "Force_Ratio",  # Torque / RPM             (load-per-revolution proxy)
]