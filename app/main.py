"""
ParkCast SF — FastAPI Service
Serves parking occupancy predictions from the best trained model.

Endpoints:
  GET  /         → welcome message
  GET  /health   → model health check
  POST /predict  → predict parking occupancy %
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging


# ── Setup logging ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import mlflow.sklearn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://34.133.160.231:5000")
MODEL_NAME = "parkcast-occupancy-model"

# tracking 
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None

def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
        print(f"Model loaded from MLflow registry: {MODEL_NAME}@champion")
    except Exception as e:
        print(f"Failed to load from MLflow registry: {e}")
        # Fallback to local model
        try:
            model = joblib.load("models/RandomForest.pkl")
            print("Loaded fallback local model")
        except Exception as e2:
            print(f"Failed to load fallback model: {e2}")
            model = None

load_model()

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="ParkCast SF API",
    description="Predicts parking availability in San Francisco 30-60 minutes ahead.",
    version="1.0.0",
)

# ── Feature order must match training ────────────────────────
FEATURE_ORDER = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    "is_street_cleaning",
    "has_nearby_event",
    "is_holiday",
    "is_school_day",
    "is_raining",
    "bad_weather",
    "temperature",
    "total_spaces",
    "neighborhood_encoded",
]

# Neighborhood encoding (must match LabelEncoder from training)
NEIGHBORHOOD_MAP = {
    "castro":      0,
    "haight":      1,
    "marina":      2,
    "mission":     3,
    "noe valley":  4,
    "richmond":    5,
    "soma":        6,
    "sunset":      7,
    "tenderloin":  8,
    "unknown":     9,
}


# ── Pydantic models ───────────────────────────────────────────

class ParkingInput(BaseModel):
    """Input features for parking occupancy prediction."""

    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    neighborhood: str = Field(..., description="SF neighborhood name")
    total_spaces: int = Field(..., ge=1, description="Total parking spaces on block")
    is_raining: int = Field(0, ge=0, le=1, description="Is it raining? (0=No, 1=Yes)")
    has_nearby_event: int = Field(0, ge=0, le=1, description="Is there a nearby event? (0=No, 1=Yes)")
    is_holiday: int = Field(0, ge=0, le=1, description="Is it a federal holiday? (0=No, 1=Yes)")
    is_school_day: int = Field(1, ge=0, le=1, description="Is it a school day? (0=No, 1=Yes)")
    temperature: float = Field(60.0, description="Temperature in Fahrenheit")

    class Config:
        json_schema_extra = {
            "example": {
                "hour": 19,
                "day_of_week": 4,
                "month": 4,
                "neighborhood": "mission",
                "total_spaces": 45,
                "is_raining": 0,
                "has_nearby_event": 1,
                "is_holiday": 0,
                "is_school_day": 1,
                "temperature": 62.5,
            }
        }


class ParkingPrediction(BaseModel):
    """Output prediction response."""
    neighborhood: str
    hour: int
    day_of_week: int
    predicted_occupancy_pct: float
    available_spaces_estimate: int
    demand_level: str
    recommendation: str
    model_version: str = "RandomForest-v1"


# ── Helper functions ──────────────────────────────────────────

def get_demand_level(occupancy_pct: float) -> str:
    if occupancy_pct < 40:
        return "Low"
    elif occupancy_pct < 70:
        return "Medium"
    elif occupancy_pct < 85:
        return "High"
    else:
        return "Very High"


def get_recommendation(occupancy_pct: float) -> str:
    if occupancy_pct < 40:
        return "Easy to park — plenty of spaces available."
    elif occupancy_pct < 70:
        return "Good chance of finding parking — head over."
    elif occupancy_pct < 85:
        return "Limited spots — arrive early or consider nearby blocks."
    else:
        return "Very hard to park — consider public transit or a garage."


def prepare_features(input: ParkingInput) -> np.ndarray:
    """Convert input to feature array matching training order."""
    is_weekend = 1 if input.day_of_week >= 5 else 0
    is_rush_hour = 1 if (7 <= input.hour <= 9 or 17 <= input.hour <= 19) else 0
    is_street_cleaning = 1 if 8 <= input.hour <= 12 else 0
    bad_weather = input.is_raining
    neighborhood_encoded = NEIGHBORHOOD_MAP.get(
        input.neighborhood.lower(), NEIGHBORHOOD_MAP["unknown"]
    )

    features = [
        input.hour,
        input.day_of_week,
        input.month,
        is_weekend,
        is_rush_hour,
        is_street_cleaning,
        input.has_nearby_event,
        input.is_holiday,
        input.is_school_day,
        input.is_raining,
        bad_weather,
        input.temperature,
        input.total_spaces,
        neighborhood_encoded,
    ]

    return np.array(features).reshape(1, -1)


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/")
def root():
    """Welcome endpoint."""
    return {
        "message": "Welcome to ParkCast SF API",
        "description": "Predicts parking occupancy in San Francisco 30-60 min ahead",
        "version": "1.0.0",
        "endpoints": {
            "health":  "GET /health",
            "predict": "POST /predict",
            "docs":    "GET /docs",
        }
    }


@app.get("/health")
def health():
    """Health check — confirms model is loaded."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(model).__name__,
        "features": FEATURE_ORDER,
        "num_features": len(FEATURE_ORDER),
    }


@app.post("/predict", response_model=ParkingPrediction)
def predict(input: ParkingInput):
    """
    Predict parking occupancy % for a given location and time.

    Returns predicted occupancy percentage and a parking recommendation.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )

    try:
        # Prepare features
        features = prepare_features(input)

        # Run prediction
        occupancy_pct = float(model.predict(features)[0])
        occupancy_pct = round(min(100.0, max(0.0, occupancy_pct)), 2)

        # Estimate available spaces
        available = int(input.total_spaces * (1 - occupancy_pct / 100))

        return ParkingPrediction(
            neighborhood=input.neighborhood,
            hour=input.hour,
            day_of_week=input.day_of_week,
            predicted_occupancy_pct=occupancy_pct,
            available_spaces_estimate=available,
            demand_level=get_demand_level(occupancy_pct),
            recommendation=get_recommendation(occupancy_pct),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")