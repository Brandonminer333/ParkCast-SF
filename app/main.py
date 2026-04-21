"""
ParkCast SF — FastAPI Service

Serves the parkcast-frontend UI (https://parkcast-frontend.vercel.app).

Public endpoints:
  GET  /                  → welcome / endpoint map
  GET  /health            → liveness + asset summary
  POST /predict/blocks    → block-by-block occupancy forecast around a lat/lon

The frontend calls /health on load and /predict/blocks when the user taps
"Find Parking". Everything else the browser needs (geocoding, routing,
weather) is fetched directly from third-party services (Nominatim/OSRM/
Open-Meteo) and does not go through this API.

The prediction stack is the hybrid LightGBM model trained in
dev/train_lightgbm.ipynb. At inference:

  final_occupancy = clip(block_hour_dow_mean + LightGBM_residual, 0, 100)

where `block_hour_dow_mean` is a per-block historical baseline and the
LightGBM residual is learned over SFpark meter-hour data. Weather is
pulled from Open-Meteo on demand, cached by day.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ── Paths ────────────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")
DATA_DIR = os.path.join(os.path.dirname(APP_DIR), "data")

MODEL_PATH = os.path.join(MODEL_DIR, "LightGBM.pkl")
BLOCK_AGG_PATH = os.path.join(MODEL_DIR, "LightGBM.block_aggs.parquet")
BLOCKS_PATH = os.path.join(MODEL_DIR, "blocks.parquet")
MASTER_PATH = os.path.join(MODEL_DIR, "master_blocks.parquet")
LAG_PATH = os.path.join(MODEL_DIR, "lag_history.parquet")
CIT_LOOKUP_PATH = os.path.join(MODEL_DIR, "citations_hourly_median.parquet")
META_PATH = os.path.join(MODEL_DIR, "LightGBM.meta.json")
EVENTS_PATH = os.path.join(DATA_DIR, "events.csv")


# ── Feature schema (must stay in sync with dev/train_lightgbm.ipynb) ─────────
FEATURES_NUMERIC = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "is_school_day", "is_raining", "temperature", "event_intensity",
    "citation_count", "citations_hourly_median",
    "lat", "lon", "total_spaces",
    "block_mean", "block_hour_mean",
    "lag_7d", "lag_14d", "lag_28d",
]
FEATURES_CATEGORICAL = ["neighborhood"]
FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL

US_HOLIDAYS = {
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 5, 26),
    date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1), date(2025, 10, 13),
    date(2025, 11, 11), date(2025, 11, 27), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 4), date(2026, 9, 7), date(2026, 10, 12),
    date(2026, 11, 11), date(2026, 11, 26), date(2026, 12, 25),
}


# ── Globals populated at startup ─────────────────────────────────────────────
model = None
block_aggs: Optional[pd.DataFrame] = None
blocks: Optional[pd.DataFrame] = None
master: Optional[pd.DataFrame] = None
lag_hist: Optional[pd.DataFrame] = None
cit_lookup: Optional[pd.DataFrame] = None
events: Optional[pd.DataFrame] = None
meta: dict = {}
global_baseline_mean: float = 50.0


def _load_all() -> None:
    """Load model + lookup tables into module globals. Missing optional assets
    (events, master catalog) degrade gracefully; missing required assets
    leave `model` as None and /health will report 503."""
    global model, block_aggs, blocks, master, lag_hist, cit_lookup, events
    global meta, global_baseline_mean

    model = joblib.load(MODEL_PATH)
    block_aggs = pd.read_parquet(BLOCK_AGG_PATH)
    blocks = pd.read_parquet(BLOCKS_PATH)
    lag_hist = (pd.read_parquet(LAG_PATH)
                  .sort_values(["lat", "lon", "timestamp"])
                  .reset_index(drop=True))
    cit_lookup = pd.read_parquet(CIT_LOOKUP_PATH)

    if os.path.exists(MASTER_PATH):
        master = pd.read_parquet(MASTER_PATH)

    if os.path.exists(EVENTS_PATH):
        events = pd.read_csv(EVENTS_PATH, parse_dates=["date"])
    else:
        events = pd.DataFrame(columns=["date", "venue_lat", "venue_lon",
                                       "start_hour", "end_hour"])

    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
            global_baseline_mean = float(meta.get("global_mean", 50.0))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_all()
    yield


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="ParkCast SF API",
    description="Block-by-block parking occupancy forecasts for San Francisco.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Weather: on-demand Open-Meteo with daily cache ───────────────────────────
@lru_cache(maxsize=256)
def _weather_day(day_iso: str) -> Optional[dict]:
    """SF hourly temperature (°F) + precipitation for `day_iso`.
    Returns {hour: (temp_f, is_raining)} or None on failure."""
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=37.7749&longitude=-122.4194"
        f"&start_date={day_iso}&end_date={day_iso}"
        "&hourly=temperature_2m,precipitation"
        "&temperature_unit=fahrenheit"
        "&timezone=America/Los_Angeles"
    )
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.load(r)
        hours = [datetime.fromisoformat(t).hour for t in data["hourly"]["time"]]
        temps = data["hourly"]["temperature_2m"]
        precip = data["hourly"]["precipitation"]
        return {h: (float(t), 1 if (p or 0) > 0.1 else 0)
                for h, t, p in zip(hours, temps, precip)}
    except Exception:
        return None


def weather_for(ts: datetime) -> tuple[float, int]:
    day_data = _weather_day(ts.date().isoformat())
    if day_data and ts.hour in day_data:
        return day_data[ts.hour]
    return 60.0, 0


# ── Geo helpers ──────────────────────────────────────────────────────────────
def haversine_vec(lat1: float, lon1: float,
                  lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in meters, scalar origin → array dest."""
    R = 6_371_000
    lat1r = math.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + math.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


# ── Feature engineering ──────────────────────────────────────────────────────
def is_school_day(d: date) -> int:
    """Rough SF school-year heuristic: weekdays, Sep–May, not a US holiday."""
    if d.weekday() >= 5 or d in US_HOLIDAYS:
        return 0
    if d.month in (6, 7, 8):
        return 0
    return 1


def event_intensity_at(lat: float, lon: float, ts: datetime,
                       radius_m: float = 800.0) -> float:
    """max(exp(-dist_m / 300)) over active events within `radius_m`. Matches
    the encoding used during training (see dev/preprocess_real_data.ipynb)."""
    if events is None or events.empty:
        return 0.0
    same_day = events[events["date"].dt.date == ts.date()]
    if same_day.empty:
        return 0.0
    in_window = same_day[(same_day["start_hour"] <= ts.hour)
                         & (ts.hour <= same_day["end_hour"])]
    if in_window.empty:
        return 0.0
    dists = haversine_vec(lat, lon,
                          in_window["venue_lat"].values,
                          in_window["venue_lon"].values)
    close = dists <= radius_m
    if not close.any():
        return 0.0
    return float(np.exp(-dists[close] / 300.0).max())


def lag_value(lat: float, lon: float, ts: datetime, days: int) -> float:
    target = ts - timedelta(days=days)
    mask = ((lag_hist["lat"] == lat)
            & (lag_hist["lon"] == lon)
            & (lag_hist["timestamp"] == target))
    hit = lag_hist.loc[mask, "occupancy_pct"]
    return float(hit.iloc[0]) if len(hit) else np.nan


def block_aggregates_for(lat: float, lon: float, hour: int, dow: int
                         ) -> tuple[float, float, float]:
    """(block_mean, block_hour_mean, block_hour_dow_mean) with progressive
    fallbacks. block_hour_dow_mean is the additive baseline the residual
    LightGBM was trained against."""
    exact = block_aggs[
        (block_aggs["lat"] == lat) & (block_aggs["lon"] == lon)
        & (block_aggs["hour"] == hour) & (block_aggs["day_of_week"] == dow)
    ]
    if len(exact):
        r = exact.iloc[0]
        return (float(r["block_mean"]), float(r["block_hour_mean"]),
                float(r["block_hour_dow_mean"]))

    hour_only = block_aggs[
        (block_aggs["lat"] == lat) & (block_aggs["lon"] == lon)
        & (block_aggs["hour"] == hour)
    ]
    if len(hour_only):
        bhm = float(hour_only["block_hour_mean"].iloc[0])
        bm = float(hour_only["block_mean"].iloc[0])
        return bm, bhm, bhm

    block_only = block_aggs[
        (block_aggs["lat"] == lat) & (block_aggs["lon"] == lon)
    ]
    if len(block_only):
        bm = float(block_only["block_mean"].iloc[0])
        return bm, bm, bm

    gm = global_baseline_mean
    return gm, gm, gm


def citations_median(hour: int, dow: int) -> float:
    row = cit_lookup[
        (cit_lookup["hour"] == hour) & (cit_lookup["day_of_week"] == dow)
    ]
    return float(row["citations_hourly_median"].iloc[0]) if len(row) else 0.0


def build_feature_row(block_row: pd.Series, ts: datetime,
                      temp_f: float, raining: int) -> dict:
    lat = float(block_row["lat"])
    lon = float(block_row["lon"])
    hour = ts.hour
    dow = ts.weekday()
    bm, bhm, baseline = block_aggregates_for(lat, lon, hour, dow)
    return {
        "hour": hour,
        "day_of_week": dow,
        "month": ts.month,
        "is_weekend": 1 if dow >= 5 else 0,
        "is_holiday": 1 if ts.date() in US_HOLIDAYS else 0,
        "is_school_day": is_school_day(ts.date()),
        "is_raining": raining,
        "temperature": temp_f,
        "event_intensity": event_intensity_at(lat, lon, ts),
        "citation_count": 0.0,
        "citations_hourly_median": citations_median(hour, dow),
        "lat": lat,
        "lon": lon,
        "total_spaces": int(block_row["total_spaces"]),
        "block_mean": bm,
        "block_hour_mean": bhm,
        "lag_7d": lag_value(lat, lon, ts, 7),
        "lag_14d": lag_value(lat, lon, ts, 14),
        "lag_28d": lag_value(lat, lon, ts, 28),
        "neighborhood": str(block_row["neighborhood"]),
        "_baseline": baseline,
    }


def score_blocks(block_df: pd.DataFrame, ts: datetime) -> pd.DataFrame:
    """Run the residual LightGBM and add occupancy + available-spaces columns.
    Residual is added to block_hour_dow_mean baseline (see train_lightgbm)."""
    temp_f, raining = weather_for(ts)
    rows = [build_feature_row(r, ts, temp_f, raining)
            for _, r in block_df.iterrows()]
    feat = pd.DataFrame(rows)
    baselines = feat.pop("_baseline").values
    feat["neighborhood"] = feat["neighborhood"].astype("category")
    for col in FEATURES_NUMERIC:
        feat[col] = pd.to_numeric(feat[col], errors="coerce")

    residual = model.predict(feat[FEATURES])
    occupancy = np.clip(baselines + residual, 0, 100)

    out = block_df.copy().reset_index(drop=True)
    out["predicted_occupancy_pct"] = occupancy.round(2)
    out["available_spaces"] = (out["total_spaces"]
                               * (1 - occupancy / 100)).round().astype(int)
    return out


# ── Response classification ──────────────────────────────────────────────────
def demand_level(pct: float) -> str:
    if pct < 40:
        return "Low"
    if pct < 70:
        return "Medium"
    if pct < 85:
        return "High"
    return "Very High"


def color_for(pct: float) -> str:
    """Hex color used by the frontend map markers. Matches the UI legend."""
    if pct < 40:
        return "#22c55e"   # green  — Easy
    if pct < 70:
        return "#f59e0b"   # amber  — Moderate
    if pct < 85:
        return "#f97316"   # orange — Hard
    return "#ef4444"       # red    — Very Hard


def street_label_for(lat: float, lon: float, neighborhood: str) -> str:
    """Prefer a human-readable street name from master_blocks if available,
    otherwise fall back to a neighborhood-tagged label."""
    if master is not None and not master.empty:
        candidate_cols = [c for c in ("corridor", "limits") if c in master.columns]
        if candidate_cols:
            # Nearest master block within ~80m (street-segment centers can
            # drift from metered centroids).
            dists = haversine_vec(lat, lon,
                                  master["lat"].values, master["lon"].values)
            nearest = int(np.argmin(dists))
            if dists[nearest] <= 80.0:
                for col in candidate_cols:
                    val = master.iloc[nearest][col]
                    if pd.notna(val) and str(val).strip():
                        return str(val).strip()
    pretty_nbh = neighborhood.replace("_", " ").title() if neighborhood else "Unknown"
    return f"{pretty_nbh} block"


# ── Request / response schemas ───────────────────────────────────────────────
class BlockPredictionRequest(BaseModel):
    """Matches the POST body sent by parkcast-frontend/app/page.js.

    Most "condition" fields (is_holiday, is_school_day, is_raining, temperature,
    has_nearby_event) are resolved server-side from the arrival timestamp and
    Open-Meteo + the events catalog, so the client-supplied values are accepted
    for backwards compatibility but deliberately ignored — the server's answer
    stays self-consistent even if the browser's weather fetch failed."""
    lat: float = Field(..., description="Destination latitude")
    lon: float = Field(..., description="Destination longitude")
    radius_meters: int = Field(1500, ge=100, le=3000)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    is_raining: int = Field(0, ge=0, le=1)
    has_nearby_event: int = Field(0, ge=0, le=1)
    is_holiday: int = Field(0, ge=0, le=1)
    is_school_day: int = Field(1, ge=0, le=1)
    temperature: float = Field(60.0)
    minutes_away: int = Field(0, ge=0, le=180)


class BlockPrediction(BaseModel):
    block_id: str
    street: str
    lat: float
    lon: float
    total_spaces: int
    neighborhood: str
    distance_meters: int
    predicted_occupancy_pct: float
    available_spaces_estimate: int
    demand_level: str
    color: str


class BlockPredictionResponse(BaseModel):
    destination_lat: float
    destination_lon: float
    radius_meters: int
    predicted_at_hour: int
    minutes_away: int
    total_blocks_found: int
    blocks: List[BlockPrediction]


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Welcome to ParkCast SF API v2",
        "description": "Block-by-block parking prediction for San Francisco",
        "version": "2.0.0",
        "endpoints": {
            "health": "GET /health",
            "predict_blocks": "POST /predict/blocks",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health():
    if model is None or blocks is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(model).__name__,
        "num_features": len(FEATURES),
        "total_blocks_in_db": int(len(blocks)),
        "trained_test_mae": meta.get("metrics", {})
                                 .get("residual_model", {}).get("mae"),
    }


@app.post("/predict/blocks", response_model=BlockPredictionResponse)
def predict_blocks_endpoint(req: BlockPredictionRequest):
    """Rank every metered block within `radius_meters` of (lat, lon) by
    predicted occupancy at arrival time. If no blocks fall inside the radius,
    return the 8 closest so the map still has something to render."""
    if model is None or blocks is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    arrival_ts = datetime.now() + timedelta(minutes=req.minutes_away)
    arrival_hour = arrival_ts.hour

    dists = haversine_vec(req.lat, req.lon,
                          blocks["lat"].values, blocks["lon"].values)
    mask = dists <= req.radius_meters

    if mask.any():
        near = blocks.loc[mask].copy()
        near_dists = dists[mask]
    else:
        nearest_idx = np.argsort(dists)[:8]
        near = blocks.iloc[nearest_idx].copy()
        near_dists = dists[nearest_idx]

    if near.empty:
        return BlockPredictionResponse(
            destination_lat=req.lat, destination_lon=req.lon,
            radius_meters=req.radius_meters, predicted_at_hour=arrival_hour,
            minutes_away=req.minutes_away, total_blocks_found=0, blocks=[],
        )

    near = near.reset_index(drop=True)
    near["distance_m"] = near_dists

    scored = score_blocks(near.drop(columns=["distance_m"]), arrival_ts)
    scored["distance_m"] = near["distance_m"].values

    out: List[BlockPrediction] = []
    for _, r in scored.iterrows():
        lat = float(r["lat"])
        lon = float(r["lon"])
        occ = float(r["predicted_occupancy_pct"])
        nbh = str(r["neighborhood"])
        out.append(BlockPrediction(
            block_id=f"b_{lat:.5f}_{lon:.5f}",
            street=street_label_for(lat, lon, nbh),
            lat=lat,
            lon=lon,
            total_spaces=int(r["total_spaces"]),
            neighborhood=nbh,
            distance_meters=int(round(float(r["distance_m"]))),
            predicted_occupancy_pct=round(occ, 2),
            available_spaces_estimate=int(r["available_spaces"]),
            demand_level=demand_level(occ),
            color=color_for(occ),
        ))

    out.sort(key=lambda b: b.distance_meters)

    return BlockPredictionResponse(
        destination_lat=req.lat,
        destination_lon=req.lon,
        radius_meters=req.radius_meters,
        predicted_at_hour=arrival_hour,
        minutes_away=req.minutes_away,
        total_blocks_found=len(out),
        blocks=out,
    )
