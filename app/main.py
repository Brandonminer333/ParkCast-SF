"""
ParkCast SF — FastAPI inference service.

Loads the LightGBM residual model + per-block lookup parquets from GCS at
startup (falls back to local `app/models/` for dev). Every request builds
the full 33-feature matrix the model was trained on; lags are NaN at
inference (training lag_history ends at the temporal split — LightGBM uses
surrogate splits for missing features).

Env vars:
  GCS_BUCKET   — bucket name. If unset, loads from local app/models/.
  GCS_PREFIX   — optional key prefix inside the bucket (e.g. "Data/"). Must
                 end with "/" or be empty.

Endpoints:
  GET  /                → service info
  GET  /health          → model-loaded check + metrics
  POST /predict         → single aggregate prediction for a neighborhood
  POST /predict/blocks  → block-by-block predictions around (lat, lon)
  GET  /geocode_proxy   → Nominatim address lookup
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.bundle import LOCAL_MODEL_DIR, ModelBundle, resolve_model_dir
from app.constants import (
    MAX_BLOCKS_RETURNED,
    classify_occupancy,
    color_for,
    demand_level,
    recommendation_for,
)
from app.features import build_features, haversine_vec, predict_rows
from app.schemas import (
    BlockPrediction,
    BlocksRequest,
    BlocksResponse,
    ParkingInput,
    ParkingPrediction,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Backward-compatible aliases for test imports ───────────────────
_demand_level = demand_level
_color = color_for
_recommendation = recommendation_for

# ── Model bundle (loaded once at import time) ──────────────────────
BUNDLE: Optional[ModelBundle] = None
try:
    BUNDLE = ModelBundle(resolve_model_dir())
except Exception as e:  # noqa: BLE001
    logger.error(f"❌ Failed to load ModelBundle: {e}")


# ── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="ParkCast SF API",
    description="Block-level parking occupancy predictions for San Francisco.",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "ParkCast SF API",
        "version": "3.0.0",
        "endpoints": [
            "GET /health",
            "POST /predict",
            "POST /predict/blocks",
            "GET /geocode_proxy",
            "GET /docs",
        ],
    }


@app.get("/health")
def health():
    if BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    metrics = BUNDLE.meta.get("metrics", {})
    return {
        "status": "healthy",
        "artifact_source": BUNDLE.src,
        "model_features": len(BUNDLE.features),
        "total_blocks": int(len(BUNDLE.blocks)),
        "train_split_time": BUNDLE.meta.get("split_time"),
        "test_mae": metrics.get("residual_model", {}).get("mae"),
        "test_r2": metrics.get("residual_model", {}).get("r2"),
        "mlflow_run_id": BUNDLE.meta.get("mlflow_run_id"),
    }


@app.post("/predict/blocks", response_model=BlocksResponse)
def predict_blocks(req: BlocksRequest):
    if BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    target_hour = (req.hour + req.minutes_away // 60) % 24

    blocks = BUNDLE.blocks.copy()
    blocks["distance_m"] = haversine_vec(
        req.lat, req.lon, blocks["lat"].to_numpy(), blocks["lon"].to_numpy()
    )

    nearby = blocks[blocks["distance_m"] <= req.radius_meters].sort_values("distance_m")
    if nearby.empty:
        # Nothing in radius → snap to the 8 closest trained blocks so the
        # user still gets a usable answer.
        nearby = blocks.sort_values("distance_m").head(8)

    nearby = nearby.head(MAX_BLOCKS_RETURNED)

    feat_df = build_features(nearby, target_hour, req, BUNDLE)
    preds = predict_rows(feat_df, BUNDLE)

    out = _build_block_predictions(feat_df, preds)

    return BlocksResponse(
        destination_lat=req.lat,
        destination_lon=req.lon,
        radius_meters=req.radius_meters,
        predicted_at_hour=target_hour,
        minutes_away=req.minutes_away,
        total_blocks_found=len(out),
        blocks=out,
    )


def _build_block_predictions(
    feat_df: pd.DataFrame,
    preds: np.ndarray,
) -> List[BlockPrediction]:
    """Vectorized construction of BlockPrediction list from feature df + preds."""
    result_df = feat_df[
        ["lat", "lon", "street", "neighborhood", "total_spaces", "distance_m", "coverage"]
    ].copy()
    result_df["predicted_occupancy_pct"] = np.round(preds, 2)
    result_df["total_spaces"] = result_df["total_spaces"].fillna(0).astype(int)
    result_df["available_spaces_estimate"] = (
        result_df["total_spaces"] * (1 - result_df["predicted_occupancy_pct"] / 100)
    ).astype(int)

    out: List[BlockPrediction] = []
    for row in result_df.itertuples(index=False):
        pct = float(row.predicted_occupancy_pct)
        info = classify_occupancy(pct)
        neigh = row.neighborhood
        out.append(
            BlockPrediction(
                lat=float(row.lat),
                lon=float(row.lon),
                street=str(row.street) if pd.notna(row.street) else None,
                neighborhood=str(neigh) if pd.notna(neigh) else None,
                total_spaces=int(row.total_spaces),
                distance_meters=int(row.distance_m),
                predicted_occupancy_pct=pct,
                available_spaces_estimate=int(row.available_spaces_estimate),
                demand_level=info.label,
                color=info.color,
                coverage=str(row.coverage) if pd.notna(row.coverage) else "metered",
            )
        )
    return out


@app.post("/predict", response_model=ParkingPrediction)
def predict(body: ParkingInput):
    """Aggregate prediction across every trained block in a neighborhood."""
    if BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    mask = BUNDLE.blocks["neighborhood"].str.lower() == body.neighborhood.strip().lower()
    neigh_blocks = BUNDLE.blocks[mask].copy()
    if neigh_blocks.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No trained blocks found in neighborhood '{body.neighborhood}'.",
        )

    synthetic_req = BlocksRequest(
        lat=float(neigh_blocks["lat"].mean()),
        lon=float(neigh_blocks["lon"].mean()),
        radius_meters=2000,
        hour=body.hour,
        day_of_week=body.day_of_week,
        month=body.month,
        is_raining=body.is_raining,
        is_holiday=body.is_holiday,
        is_school_day=body.is_school_day,
        temperature=body.temperature,
        event_intensity=body.event_intensity,
        minutes_away=0,
    )
    neigh_blocks["distance_m"] = 0.0
    feat_df = build_features(neigh_blocks, body.hour, synthetic_req, BUNDLE)
    preds = predict_rows(feat_df, BUNDLE)
    pct = float(round(float(np.median(preds)), 2))
    avail = int(body.total_spaces * (1 - pct / 100))
    info = classify_occupancy(pct)

    return ParkingPrediction(
        neighborhood=body.neighborhood,
        hour=body.hour,
        day_of_week=body.day_of_week,
        predicted_occupancy_pct=pct,
        available_spaces_estimate=avail,
        demand_level=info.label,
        recommendation=info.recommendation,
        blocks_aggregated=int(len(neigh_blocks)),
    )


@app.get("/geocode_proxy")
def geocode_proxy(q: str):
    import requests

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": f"{q} San Francisco CA",
                "format": "json",
                "limit": 6,
                "addressdetails": 1,
            },
            headers={
                "User-Agent": "ParkCastSF/3.0 (university project usfca.edu)",
                "Accept-Language": "en",
                "Referer": "https://parkcast-frontend.vercel.app",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        filtered = [
            d
            for d in data
            if "san francisco" in d.get("display_name", "").lower()
            or "california" in d.get("display_name", "").lower()
        ]
        return filtered or data
    except Exception as e:  # noqa: BLE001
        logger.error(f"Geocode error: {e}")
        return []
