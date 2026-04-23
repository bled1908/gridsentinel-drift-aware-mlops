"""
GridSentinel Inference API

Endpoints:
    GET  /health        — liveness check
    GET  /model/info    — current model metadata
    POST /predict       — hourly load forecast
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add src/ to path so we can import project modules when running from root
sys.path.insert(0, str(Path(__file__).parent))

from forecasting_model import LoadForecaster
from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GridSentinel",
    description="Drift-aware electricity load forecasting API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Globals — model loaded once at startup
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "models/xgboost_baseline.json")
_forecaster: Optional[LoadForecaster] = None


def _load_model() -> LoadForecaster:
    """Load (or reload) the XGBoost model from disk."""
    forecaster = LoadForecaster()
    forecaster.load_model(MODEL_PATH)
    log.info("Model loaded from %s | Features: %s", MODEL_PATH, forecaster.feature_names)
    return forecaster


@app.on_event("startup")
async def startup() -> None:
    global _forecaster
    _forecaster = _load_model()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Feature vector for a single hourly prediction.

    All numeric fields correspond to the features produced by
    ``data_processing.create_features()``.
    """
    load_lag_1h: float = Field(..., description="Load 1 hour ago (kW)")
    load_lag_2h: float = Field(..., description="Load 2 hours ago (kW)")
    load_lag_3h: float = Field(..., description="Load 3 hours ago (kW)")
    load_lag_6h: float = Field(..., description="Load 6 hours ago (kW)")
    load_lag_12h: float = Field(..., description="Load 12 hours ago (kW)")
    load_lag_24h: float = Field(..., description="Load 24 hours ago (kW)")
    load_lag_48h: float = Field(..., description="Load 48 hours ago (kW)")
    load_lag_72h: float = Field(..., description="Load 72 hours ago (kW)")
    load_lag_168h: float = Field(..., description="Load 168 hours (1 week) ago (kW)")
    load_roll_mean_24h: float = Field(..., description="24-hour rolling mean (kW)")
    load_roll_std_24h: float = Field(..., description="24-hour rolling std dev (kW)")
    hour_sin_daily: float = Field(..., description="Sine of hour within day")
    hour_cos_daily: float = Field(..., description="Cosine of hour within day")
    hour_sin_weekly: float = Field(..., description="Sine of hour within week")
    hour_cos_weekly: float = Field(..., description="Cosine of hour within week")
    day_sin_yearly: float = Field(..., description="Sine of day within year")
    day_cos_yearly: float = Field(..., description="Cosine of day within year")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day (0–23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    day_of_month: int = Field(..., ge=1, le=31, description="Day of the month")
    month: int = Field(..., ge=1, le=12, description="Month (1–12)")
    is_weekend: int = Field(..., ge=0, le=1, description="1 if weekend, else 0")
    season: int = Field(..., ge=0, le=3, description="0=Winter 1=Spring 2=Summer 3=Autumn")
    is_holiday: int = Field(..., ge=0, le=1, description="1 if public holiday, else 0")


class PredictResponse(BaseModel):
    forecast_kw: float = Field(..., description="Predicted load in kilowatts")
    model_version: str = Field(..., description="Loaded model file path")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_path: str
    feature_count: int
    features: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health() -> HealthResponse:
    """Liveness check — returns 200 when the service and model are ready."""
    return HealthResponse(status="ok", model_loaded=_forecaster is not None)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Operations"])
async def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    if _forecaster is None or _forecaster.feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_path=MODEL_PATH,
        feature_count=len(_forecaster.feature_names),
        features=_forecaster.feature_names,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Generate an hourly electricity load forecast.

    Accepts a JSON body with all 24 feature fields. Returns the predicted
    load in kilowatts alongside the model version identifier.
    """
    if _forecaster is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import pandas as pd

    feature_dict = request.model_dump()

    if _forecaster.feature_names is None:
        raise HTTPException(status_code=500, detail="Model feature names not available")

    # Build DataFrame in the exact column order the model expects
    try:
        X = pd.DataFrame([feature_dict])[_forecaster.feature_names]
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing feature: {exc}") from exc

    prediction: float = float(_forecaster.predict(X)[0])
    log.debug("Prediction: %.4f kW", prediction)

    return PredictResponse(forecast_kw=prediction, model_version=MODEL_PATH)
