"""
FastAPI Predictive Maintenance API
====================================
Endpoints:
  GET  /health           — liveness check
  POST /predict          — single sensor reading → failure probability
  POST /predict-batch    — list of readings → batch predictions

Run locally:
    uvicorn api.main:app --reload --port 8000

Or via Docker:
    docker compose up
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# PATH SETUP — allows `uvicorn api.main:app` from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import ARTIFACTS_DIR, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE
from src.feature_engineering import create_physics_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# APP INITIALISATION
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Real-time machine failure prediction powered by LightGBM. "
        "Trained on the AI4I 2020 Predictive Maintenance dataset. "
        "Business-cost optimised threshold (FN=$10k, FP=$500)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# MODEL LOADING (at startup, once)
# ---------------------------------------------------------------------------
_model = None
_model_name: str = "unknown"
_decision_threshold: float = 0.32   # optimised on val set during training


@app.on_event("startup")
async def load_model() -> None:
    """Load the serialised champion model on application startup."""
    global _model, _model_name, _decision_threshold

    model_dir = ARTIFACTS_DIR / "models"
    candidates = list(model_dir.glob("*_champion.pkl")) if model_dir.exists() else []

    if not candidates:
        logger.error(
            "No champion model found in %s. "
            "Run the training pipeline before starting the API.",
            model_dir,
        )
        return   # API starts but /predict will return 503

    path = candidates[0]
    _model = joblib.load(path)
    _model_name = path.stem.replace("_champion", "").replace("_", " ").title()
    logger.info("Model loaded: %s from %s", _model_name, path)


# ---------------------------------------------------------------------------
# PYDANTIC INPUT/OUTPUT MODELS
# ---------------------------------------------------------------------------

class SensorReading(BaseModel):
    """Single machine sensor snapshot for failure prediction."""

    machine_type: str = Field(
        ...,
        pattern="^[LMHlmh]$",
        description="Quality tier of the machine: 'L' (Low), 'M' (Medium), or 'H' (High)",
        examples=["M"],
    )
    air_temperature_k: float = Field(
        ...,
        ge=290.0, le=315.0,
        alias="air_temperature_K",
        description="Ambient air temperature in Kelvin (typical range 295–305 K)",
        examples=[300.0],
    )
    process_temperature_k: float = Field(
        ...,
        ge=300.0, le=325.0,
        alias="process_temperature_K",
        description="Process temperature in Kelvin (typical range 305–315 K)",
        examples=[310.0],
    )
    rotational_speed_rpm: int = Field(
        ...,
        ge=1000, le=3000,
        description="Tool rotational speed in RPM (typical range 1168–2886)",
        examples=[1500],
    )
    torque_nm: float = Field(
        ...,
        ge=1.0, le=80.0,
        alias="torque_Nm",
        description="Applied torque in Newton-metres (typical range 3.8–76.6 Nm)",
        examples=[40.0],
    )
    tool_wear_min: int = Field(
        ...,
        ge=0, le=300,
        description="Cumulative tool wear in minutes (typical range 0–253 min)",
        examples=[100],
    )
    machine_id: Optional[str] = Field(
        None,
        description="Optional machine identifier (echoed back in the response)",
        examples=["MACHINE-001"],
    )

    @field_validator("machine_type")
    @classmethod
    def normalise_type(cls, v: str) -> str:
        return v.upper()

    model_config = {"populate_by_name": True}


class PredictionResult(BaseModel):
    """Failure prediction result for a single machine reading."""

    machine_id:          Optional[str]
    failure_probability: float = Field(..., description="Predicted failure probability [0, 1]")
    failure_probability_pct: float = Field(..., description="Failure probability as percentage [0, 100]")
    risk_level:          str  = Field(..., description="One of: SAFE | MONITOR | DANGER")
    recommended_action:  str
    expected_cost_if_ignored: float = Field(..., description="Expected cost ($) if no action taken")
    physics_features:    dict = Field(..., description="Derived physics features used by the model")
    model_name:          str
    threshold_used:      float


class BatchRequest(BaseModel):
    """Batch prediction request containing multiple sensor readings."""

    readings: list[SensorReading] = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="List of sensor readings (1–10,000 items)",
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="Override the default decision threshold (optional)",
    )


class BatchPredictionResult(BaseModel):
    """Summary result for a batch prediction request."""

    total_readings:  int
    n_danger:        int
    n_monitor:       int
    n_safe:          int
    total_cost_at_risk: float
    predictions:     list[PredictionResult]


class HealthResponse(BaseModel):
    status:     str
    model_name: str
    model_loaded: bool
    threshold:  float
    version:    str


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

def _reading_to_df(reading: SensorReading) -> pd.DataFrame:
    """Convert a Pydantic SensorReading to the DataFrame the model expects."""
    return pd.DataFrame([{
        "Air temperature [K]":     reading.air_temperature_k,
        "Process temperature [K]": reading.process_temperature_k,
        "Rotational speed [rpm]":  reading.rotational_speed_rpm,
        "Torque [Nm]":             reading.torque_nm,
        "Tool wear [min]":         reading.tool_wear_min,
        "Type":                    reading.machine_type,
    }])


_FEATURE_COLS = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "Temp_Diff", "Power", "Force_Ratio", "Type",
]

_RISK_ACTIONS = {
    "SAFE":    "No action required. Continue normal operations.",
    "MONITOR": "Schedule a maintenance inspection within 48 hours.",
    "DANGER":  "IMMEDIATE maintenance required. Take machine offline.",
}


def _classify_risk(prob: float) -> str:
    if prob < 0.25:
        return "SAFE"
    if prob < 0.55:
        return "MONITOR"
    return "DANGER"


def _run_inference(reading: SensorReading) -> PredictionResult:
    """Core inference logic — shared by single and batch endpoints."""
    df_raw  = _reading_to_df(reading)
    df_phys = create_physics_features(df_raw)

    # Select only the features the pipeline expects
    from src.config import NUM_FEATURES, CAT_FEATURES
    feature_df = df_phys[NUM_FEATURES + CAT_FEATURES]

    prob       = float(_model.predict_proba(feature_df)[:, 1][0])
    risk       = _classify_risk(prob)
    phys_row   = df_phys.iloc[0]

    return PredictionResult(
        machine_id=reading.machine_id,
        failure_probability=round(prob, 6),
        failure_probability_pct=round(prob * 100, 2),
        risk_level=risk,
        recommended_action=_RISK_ACTIONS[risk],
        expected_cost_if_ignored=round(prob * COST_FALSE_NEGATIVE, 2),
        physics_features={
            "Temp_Diff":   round(float(phys_row["Temp_Diff"]),   4),
            "Power":       round(float(phys_row["Power"]),       2),
            "Force_Ratio": round(float(phys_row["Force_Ratio"]), 6),
        },
        model_name=_model_name,
        threshold_used=_decision_threshold,
    )


# ---------------------------------------------------------------------------
# EXCEPTION HANDLER
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Liveness check. Returns 200 if the API is running and the model is loaded."""
    return HealthResponse(
        status="healthy" if _model is not None else "degraded — model not loaded",
        model_name=_model_name,
        model_loaded=_model is not None,
        threshold=_decision_threshold,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_single(reading: SensorReading) -> PredictionResult:
    """
    Predict failure probability for a **single** machine sensor reading.

    Returns the failure probability, risk level (SAFE / MONITOR / DANGER),
    recommended action, and derived physics features.

    **Example curl:**
    ```bash
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "machine_type": "M",
        "air_temperature_K": 300.5,
        "process_temperature_K": 310.2,
        "rotational_speed_rpm": 1450,
        "torque_Nm": 42.5,
        "tool_wear_min": 195,
        "machine_id": "MACHINE-001"
      }'
    ```
    """
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run the training pipeline and restart the API.",
        )
    try:
        return _run_inference(reading)
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc


@app.post("/predict-batch", response_model=BatchPredictionResult, tags=["Prediction"])
async def predict_batch(batch: BatchRequest) -> BatchPredictionResult:
    """
    Predict failure probability for **multiple** machine readings in one request.

    Returns individual predictions plus a fleet-wide summary (total cost at risk,
    breakdown by risk level).

    **Example curl:**
    ```bash
    curl -X POST http://localhost:8000/predict-batch \\
      -H "Content-Type: application/json" \\
      -d '{
        "readings": [
          {
            "machine_type": "L",
            "air_temperature_K": 298.1,
            "process_temperature_K": 308.6,
            "rotational_speed_rpm": 1861,
            "torque_Nm": 24.8,
            "tool_wear_min": 221,
            "machine_id": "MACHINE-042"
          },
          {
            "machine_type": "M",
            "air_temperature_K": 302.3,
            "process_temperature_K": 312.1,
            "rotational_speed_rpm": 1350,
            "torque_Nm": 60.0,
            "tool_wear_min": 240,
            "machine_id": "MACHINE-099"
          }
        ]
      }'
    ```
    """
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run the training pipeline and restart the API.",
        )
    try:
        predictions = [_run_inference(r) for r in batch.readings]
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch prediction failed: {str(exc)}",
        ) from exc

    n_danger  = sum(1 for p in predictions if p.risk_level == "DANGER")
    n_monitor = sum(1 for p in predictions if p.risk_level == "MONITOR")
    n_safe    = sum(1 for p in predictions if p.risk_level == "SAFE")
    total_cost = sum(p.expected_cost_if_ignored for p in predictions)

    # Apply optional custom threshold (re-classify if provided)
    if batch.threshold is not None and batch.threshold != _decision_threshold:
        for pred in predictions:
            pred.risk_level = (
                "SAFE"    if pred.failure_probability < batch.threshold * 0.45 else
                "MONITOR" if pred.failure_probability < batch.threshold else
                "DANGER"
            )
            pred.recommended_action = _RISK_ACTIONS[pred.risk_level]
            pred.threshold_used = batch.threshold

    # Sort by failure probability descending
    predictions.sort(key=lambda p: p.failure_probability, reverse=True)

    return BatchPredictionResult(
        total_readings=len(predictions),
        n_danger=n_danger,
        n_monitor=n_monitor,
        n_safe=n_safe,
        total_cost_at_risk=round(total_cost, 2),
        predictions=predictions,
    )