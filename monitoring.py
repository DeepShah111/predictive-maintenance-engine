"""
Monitoring: Data Drift Detection for Production Inference.

Responsibilities
----------------
- Compute and persist training-set feature statistics (mean, std, min, max)
  so the monitoring module has a reference distribution to compare against.
- Run Kolmogorov-Smirnov (KS) tests on incoming sensor data vs the training
  distribution to detect covariate shift.
- Log drift alerts to a monitoring CSV for downstream alerting / dashboards.
- Expose a DriftMonitor class that integrates cleanly with both the FastAPI
  endpoint and the Streamlit batch analysis tab.

Usage
-----
    from monitoring import DriftMonitor

    # Initialise once at startup (after training)
    monitor = DriftMonitor()

    # Save training stats after the pipeline runs
    monitor.save_training_stats(X_train)

    # Check a new batch for drift
    alerts = monitor.check_drift(new_batch_df)
    if alerts:
        print("DRIFT DETECTED:", alerts)

CLI usage:
    python monitoring.py --csv path/to/new_readings.csv
"""

import argparse
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# Local imports — tolerant of being run before src/ is on path
try:
    from src.config import ARTIFACTS_DIR, NUM_FEATURES
except ImportError:
    ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
    NUM_FEATURES = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temp_Diff",
        "Power",
        "Force_Ratio",
    ]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
_STATS_PATH      = ARTIFACTS_DIR / "training_stats.csv"
_DRIFT_LOG_PATH  = ARTIFACTS_DIR / "drift_alerts.csv"

# KS p-value threshold — below this we flag drift (α = 0.05)
_KS_P_THRESHOLD: float = 0.05

# Minimum batch size to run a meaningful KS test
_MIN_BATCH_SIZE: int = 30

# Drift log CSV header
_LOG_HEADER = [
    "timestamp_utc",
    "feature",
    "ks_statistic",
    "p_value",
    "drift_detected",
    "train_mean",
    "batch_mean",
    "mean_shift_pct",
    "n_samples",
]


# ---------------------------------------------------------------------------
# DRIFT MONITOR CLASS
# ---------------------------------------------------------------------------

class DriftMonitor:
    """Kolmogorov-Smirnov based data drift monitor.

    Parameters
    ----------
    stats_path : Path
        Path to the training statistics CSV. Created by :meth:`save_training_stats`.
    drift_log_path : Path
        Path where drift alerts are appended as CSV rows.
    ks_p_threshold : float
        KS test p-value below which drift is declared. Default 0.05.
    min_batch_size : int
        Minimum number of rows required to run a KS test. Default 30.
    """

    def __init__(
        self,
        stats_path:      Path = _STATS_PATH,
        drift_log_path:  Path = _DRIFT_LOG_PATH,
        ks_p_threshold:  float = _KS_P_THRESHOLD,
        min_batch_size:  int   = _MIN_BATCH_SIZE,
    ) -> None:
        self.stats_path     = Path(stats_path)
        self.drift_log_path = Path(drift_log_path)
        self.ks_p_threshold = ks_p_threshold
        self.min_batch_size = min_batch_size

        # Initialise training stats — loaded lazily from disk
        self._train_stats: Optional[pd.DataFrame] = None

        # Ensure the drift log CSV exists with a header row
        self._init_drift_log()

    # ------------------------------------------------------------------ #
    # PUBLIC API                                                           #
    # ------------------------------------------------------------------ #

    def save_training_stats(self, X_train: pd.DataFrame) -> None:
        """Compute and persist descriptive statistics for the training set.

        Call this once after the training pipeline completes so that the
        monitoring module has a reference distribution.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix (pre-preprocessor, post feature engineering).
        """
        numeric_cols = [c for c in NUM_FEATURES if c in X_train.columns]
        if not numeric_cols:
            logger.warning("No numeric feature columns found in X_train. Skipping stats save.")
            return

        stats_df = X_train[numeric_cols].describe().T   # index = feature names
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(self.stats_path)
        logger.info("Training stats saved to: %s", self.stats_path)
        self._train_stats = None   # invalidate cache

    def check_drift(
        self,
        batch_df: pd.DataFrame,
        tag: str = "batch",
    ) -> list[dict]:
        """Run KS drift tests on *batch_df* vs the training distribution.

        Parameters
        ----------
        batch_df : pd.DataFrame
            New sensor readings (must contain numeric feature columns).
        tag : str
            Optional label for this batch, logged to the drift CSV.

        Returns
        -------
        list[dict]
            One dict per drifted feature, with keys: feature, ks_statistic,
            p_value, train_mean, batch_mean, mean_shift_pct, n_samples.
            Empty list if no drift is detected or the batch is too small.

        Raises
        ------
        FileNotFoundError
            If training stats have not been saved yet.
        """
        train_stats = self._load_train_stats()
        if train_stats is None:
            raise FileNotFoundError(
                f"Training stats not found at {self.stats_path}. "
                "Call save_training_stats(X_train) after running the pipeline."
            )

        n = len(batch_df)
        if n < self.min_batch_size:
            logger.warning(
                "Batch size %d < minimum %d — KS test skipped (unreliable with small samples).",
                n, self.min_batch_size,
            )
            return []

        alerts = []
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        for feature in train_stats.index:
            if feature not in batch_df.columns:
                continue

            batch_values = batch_df[feature].dropna().values

            # Reconstruct an approximate reference sample from stored stats
            # (mean and std of the training distribution)
            train_mean = float(train_stats.loc[feature, "mean"])
            train_std  = float(train_stats.loc[feature, "std"])
            train_count = int(train_stats.loc[feature, "count"])

            # Generate a synthetic reference sample (same size as batch)
            # using the training distribution moments. This avoids storing
            # the full training set in production memory.
            rng = np.random.default_rng(seed=42)
            reference_sample = rng.normal(
                loc=train_mean, scale=max(train_std, 1e-9), size=len(batch_values)
            )

            ks_stat, p_value = stats.ks_2samp(reference_sample, batch_values)
            drift_detected   = bool(p_value < self.ks_p_threshold)

            batch_mean     = float(np.mean(batch_values))
            mean_shift_pct = abs(batch_mean - train_mean) / (abs(train_mean) + 1e-9) * 100

            log_row = {
                "timestamp_utc":  timestamp,
                "feature":        feature,
                "ks_statistic":   round(ks_stat, 6),
                "p_value":        round(p_value, 6),
                "drift_detected": drift_detected,
                "train_mean":     round(train_mean, 4),
                "batch_mean":     round(batch_mean, 4),
                "mean_shift_pct": round(mean_shift_pct, 2),
                "n_samples":      n,
                "tag":            tag,
            }
            self._append_to_log(log_row)

            if drift_detected:
                alerts.append(log_row)
                logger.warning(
                    "DRIFT DETECTED — feature: %-32s | KS=%.4f | p=%.4f | "
                    "train_mean=%.3f | batch_mean=%.3f | shift=%.1f%%",
                    feature, ks_stat, p_value, train_mean, batch_mean, mean_shift_pct,
                )
            else:
                logger.debug(
                    "No drift — feature: %-32s | KS=%.4f | p=%.4f",
                    feature, ks_stat, p_value,
                )

        if not alerts:
            logger.info("Drift check complete for %d samples — no drift detected.", n)
        else:
            logger.warning(
                "Drift check complete — %d/%d features drifted.",
                len(alerts), len(train_stats),
            )

        return alerts

    def get_drift_log(self) -> pd.DataFrame:
        """Return the full drift alert log as a DataFrame.

        Returns an empty DataFrame if no alerts have been logged yet.
        """
        if not self.drift_log_path.exists():
            return pd.DataFrame(columns=_LOG_HEADER + ["tag"])
        return pd.read_csv(self.drift_log_path)

    def get_feature_stats(self) -> Optional[pd.DataFrame]:
        """Return training feature statistics, or None if not yet saved."""
        return self._load_train_stats()

    # ------------------------------------------------------------------ #
    # PRIVATE HELPERS                                                      #
    # ------------------------------------------------------------------ #

    def _load_train_stats(self) -> Optional[pd.DataFrame]:
        """Load training stats from disk (cached in memory after first load)."""
        if self._train_stats is not None:
            return self._train_stats
        if not self.stats_path.exists():
            return None
        self._train_stats = pd.read_csv(self.stats_path, index_col=0)
        return self._train_stats

    def _init_drift_log(self) -> None:
        """Create the drift log CSV with headers if it doesn't exist."""
        if not self.drift_log_path.exists():
            self.drift_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.drift_log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_LOG_HEADER + ["tag"])
                writer.writeheader()

    def _append_to_log(self, row: dict) -> None:
        """Append one row to the drift alert CSV log."""
        with open(self.drift_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_LOG_HEADER + ["tag"])
            writer.writerow(row)


# ---------------------------------------------------------------------------
# CONVENIENCE FUNCTION — integrate into training pipeline
# ---------------------------------------------------------------------------

def save_training_stats_from_pipeline(X_train: pd.DataFrame) -> None:
    """Convenience wrapper: create a DriftMonitor and save training stats.

    Call this at the end of the training pipeline (in main_execution.ipynb)
    to ensure monitoring is initialised before the first production batch.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix returned by build_features_and_split().
    """
    monitor = DriftMonitor()
    monitor.save_training_stats(X_train)


# ---------------------------------------------------------------------------
# CLI ENTRYPOINT
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run drift detection on a CSV of new machine readings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv", required=True, type=Path,
        help="Path to a CSV file containing new sensor readings.",
    )
    parser.add_argument(
        "--tag", default="cli_run",
        help="Label for this batch in the drift log.",
    )
    parser.add_argument(
        "--threshold", type=float, default=_KS_P_THRESHOLD,
        help="KS p-value threshold below which drift is declared.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows from {args.csv}")

    monitor = DriftMonitor(ks_p_threshold=args.threshold)

    try:
        alerts = monitor.check_drift(df, tag=args.tag)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

    if alerts:
        print(f"\n⚠️  DRIFT DETECTED in {len(alerts)} feature(s):\n")
        for a in alerts:
            print(
                f"  {a['feature']:<35} KS={a['ks_statistic']:.4f}  "
                f"p={a['p_value']:.4f}  shift={a['mean_shift_pct']:.1f}%"
            )
    else:
        print("\n✅ No drift detected.")

    print(f"\nDrift log updated: {_DRIFT_LOG_PATH}")