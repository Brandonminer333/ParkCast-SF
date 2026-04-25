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

import json
import logging
import math
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Artifact location ──────────────────────────────────────────────
GCS_BUCKET = os.getenv("GCS_BUCKET", "").strip()
GCS_PREFIX = os.getenv("GCS_PREFIX", "").strip()
if GCS_PREFIX and not GCS_PREFIX.endswith("/"):
    GCS_PREFIX += "/"

LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CACHE_DIR = os.getenv("PARKCAST_CACHE_DIR", "/tmp/parkcast_models")

ARTIFACT_FILES = [
    "LightGBM.pkl",
    "LightGBM.block_aggs.parquet",
    "LightGBM.meta.json",
    "blocks.parquet",
    "lag_history.parquet",
    "citations_hourly_median.parquet",
    "sfpark_calibration.parquet",
    "block_static_features.parquet",
]

# Optional: enriches blocks with corridor/limits (street name + cross-streets).
# Missing in GCS won't break the service; the `street` field just stays None.
OPTIONAL_ARTIFACT_FILES = ["master_blocks.parquet"]


def _download_from_gcs(bucket: str, prefix: str, files: List[str], dest: str, optional: bool = False) -> None:
    from google.cloud import storage  # lazy — local dev doesn't need it

    os.makedirs(dest, exist_ok=True)
    client = storage.Client()
    b = client.bucket(bucket)
    for f in files:
        key = f"{prefix}{f}"
        try:
            b.blob(key).download_to_filename(os.path.join(dest, f))
            logger.info(f"  downloaded gs://{bucket}/{key}")
        except Exception as e:
            if optional:
                logger.info(f"  skipped optional gs://{bucket}/{key}: {e}")
            else:
                raise


def _resolve_model_dir() -> str:
    if GCS_BUCKET:
        try:
            logger.info(f"Fetching artifacts from gs://{GCS_BUCKET}/{GCS_PREFIX} …")
            _download_from_gcs(GCS_BUCKET, GCS_PREFIX, ARTIFACT_FILES, CACHE_DIR)
            _download_from_gcs(GCS_BUCKET, GCS_PREFIX, OPTIONAL_ARTIFACT_FILES, CACHE_DIR, optional=True)
            return CACHE_DIR
        except Exception as e:  # noqa: BLE001 — any download error → fallback
            logger.warning(f"GCS download failed ({e}); falling back to {LOCAL_MODEL_DIR}")
    else:
        logger.info(f"GCS_BUCKET unset; loading from {LOCAL_MODEL_DIR}")
    return LOCAL_MODEL_DIR


# ── Model bundle ───────────────────────────────────────────────────
class ModelBundle:
    """Model + every lookup parquet inference needs, loaded once at startup."""

    def __init__(self, src: str):
        self.src = src

        self.model = joblib.load(os.path.join(src, "LightGBM.pkl"))
        with open(os.path.join(src, "LightGBM.meta.json")) as f:
            self.meta = json.load(f)
        self.features: List[str] = self.meta["features"]
        self.global_mean: float = float(self.meta.get("global_mean", 45.0))

        self.blocks = pd.read_parquet(os.path.join(src, "blocks.parquet"))
        self.block_aggs = pd.read_parquet(os.path.join(src, "LightGBM.block_aggs.parquet"))
        self.static = pd.read_parquet(os.path.join(src, "block_static_features.parquet"))
        self.cit_med = pd.read_parquet(os.path.join(src, "citations_hourly_median.parquet"))

        # Optional enrichment: master_blocks.parquet has human-readable
        # street names ("corridor") and cross-streets ("limits"). Merge on
        # (lat, lon); if the file isn't present, `street` stays NaN and the
        # API returns null for that field.
        master_path = os.path.join(src, "master_blocks.parquet")
        if os.path.exists(master_path):
            try:
                master = pd.read_parquet(master_path)[["lat", "lon", "corridor", "limits"]].dropna(
                    subset=["lat", "lon"]
                )
                # blocks.parquet and master_blocks.parquet use different lat/lon
                # precisions and the rows don't share IDs, so an exact merge
                # produces 0 matches. Use a KDTree nearest-neighbor join with
                # a ~50m tolerance instead. SF span is small enough that
                # treating degrees as a flat plane is fine for matching.
                from scipy.spatial import cKDTree

                tree = cKDTree(master[["lat", "lon"]].values)
                dist, idx = tree.query(self.blocks[["lat", "lon"]].values, k=1)
                near = dist < 0.0005  # ~55m at SF latitude
                corridor = pd.Series([None] * len(self.blocks))
                limits = pd.Series([None] * len(self.blocks))
                corridor.loc[near] = master["corridor"].values[idx[near]]
                limits.loc[near] = master["limits"].values[idx[near]]

                def _street(c, l):
                    if pd.notna(c) and pd.notna(l):
                        return f"{c} ({l})"
                    if pd.notna(c):
                        return str(c)
                    return None

                self.blocks["street"] = [_street(c, l) for c, l in zip(corridor, limits)]
                logger.info(f"  street enrichment matched {near.sum():,}/{len(self.blocks):,} blocks")
            except Exception as e:
                logger.warning(f"master_blocks.parquet unreadable ({e}); street will be None")
                self.blocks["street"] = None
        else:
            self.blocks["street"] = None

        # Freeze the neighborhood category set to what training saw.
        # Unknown categories at inference become NaN → surrogate splits.
        neighborhoods = sorted(self.blocks["neighborhood"].dropna().unique().tolist())
        self.neighborhood_dtype = pd.CategoricalDtype(categories=neighborhoods)

        logger.info(f"ModelBundle loaded from {src}: " f"{len(self.blocks):,} blocks · {len(self.features)} features")


BUNDLE: Optional[ModelBundle] = None
try:
    BUNDLE = ModelBundle(_resolve_model_dir())
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


# ── Pydantic models ────────────────────────────────────────────────
class BlocksRequest(BaseModel):
    lat: float = Field(..., description="Destination latitude")
    lon: float = Field(..., description="Destination longitude")
    radius_meters: int = Field(500, ge=100, le=2000)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    is_raining: int = Field(0, ge=0, le=1)
    is_holiday: int = Field(0, ge=0, le=1)
    is_school_day: int = Field(1, ge=0, le=1)
    temperature: float = Field(60.0)
    event_intensity: float = Field(0.0, ge=0.0, le=1.0)
    minutes_away: int = Field(0, ge=0, le=120)

    class Config:
        json_schema_extra = {
            "example": {
                "lat": 37.7816,
                "lon": -122.3975,
                "radius_meters": 500,
                "hour": 19,
                "day_of_week": 4,
                "month": 4,
                "is_raining": 0,
                "is_holiday": 0,
                "is_school_day": 1,
                "temperature": 62.5,
                "event_intensity": 0.6,
                "minutes_away": 20,
            }
        }


class BlockPrediction(BaseModel):
    lat: float
    lon: float
    street: Optional[str] = None
    neighborhood: Optional[str] = None
    total_spaces: int
    distance_meters: int
    predicted_occupancy_pct: float
    available_spaces_estimate: int
    demand_level: str
    color: str


class BlocksResponse(BaseModel):
    destination_lat: float
    destination_lon: float
    radius_meters: int
    predicted_at_hour: int
    minutes_away: int
    total_blocks_found: int
    blocks: List[BlockPrediction]


class ParkingInput(BaseModel):
    """Aggregate neighborhood-level prediction (backward-compat with v2)."""

    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    neighborhood: str = Field(...)
    total_spaces: int = Field(40, ge=1)
    is_raining: int = Field(0, ge=0, le=1)
    is_holiday: int = Field(0, ge=0, le=1)
    is_school_day: int = Field(1, ge=0, le=1)
    temperature: float = Field(60.0)
    event_intensity: float = Field(0.0, ge=0.0, le=1.0)


class ParkingPrediction(BaseModel):
    neighborhood: str
    hour: int
    day_of_week: int
    predicted_occupancy_pct: float
    available_spaces_estimate: int
    demand_level: str
    recommendation: str
    blocks_aggregated: int

    class Config:
        protected_namespaces = ()


# ── Helpers ────────────────────────────────────────────────────────
def _haversine_vec(lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    R = 6_371_000
    phi0 = math.radians(lat0)
    phi1 = np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlam = np.radians(lons - lon0)
    a = np.sin(dphi / 2) ** 2 + math.cos(phi0) * np.cos(phi1) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _demand_level(pct: float) -> str:
    if pct < 40:
        return "Low"
    if pct < 70:
        return "Medium"
    if pct < 85:
        return "High"
    return "Very High"


def _color(pct: float) -> str:
    if pct < 40:
        return "#22c55e"  # green
    if pct < 70:
        return "#f59e0b"  # amber
    if pct < 85:
        return "#f97316"  # orange
    return "#ef4444"  # red


def _recommendation(pct: float) -> str:
    if pct < 40:
        return "Easy to park — plenty of spaces."
    if pct < 70:
        return "Good chance of parking — head over."
    if pct < 85:
        return "Limited spots — arrive early or check nearby blocks."
    return "Very hard to park — consider transit or a garage."


LAG_COLS = ["lag_1d", "lag_2d", "lag_7d", "lag_14d", "lag_28d", "lag_3d_mean", "lag_7d_mean"]


def _build_features(
    blocks_df: pd.DataFrame,
    target_hour: int,
    req: BlocksRequest,
    bundle: ModelBundle,
) -> pd.DataFrame:
    """Attach every feature LightGBM needs to each block row.

    Returns a DataFrame with all `bundle.features` columns plus a `_baseline`
    column (block_hour_dow_mean with fallbacks) used to reconstruct the
    non-residual prediction.
    """
    df = blocks_df.copy()

    df["hour"] = target_hour
    df["day_of_week"] = req.day_of_week
    df["month"] = req.month
    df["is_weekend"] = 1 if req.day_of_week >= 5 else 0
    df["is_holiday"] = req.is_holiday
    df["is_school_day"] = req.is_school_day
    df["is_raining"] = req.is_raining
    df["temperature"] = req.temperature
    df["event_intensity"] = req.event_intensity

    df = df.merge(bundle.block_aggs, on=["lat", "lon", "hour", "day_of_week"], how="left")
    df = df.merge(bundle.static, on=["lat", "lon"], how="left")
    df = df.merge(bundle.cit_med, on=["hour", "day_of_week"], how="left")

    # Lags stay NaN: lag_history ends at the training split and no live feed
    # is wired yet. LightGBM uses surrogate splits for missing features.
    for col in LAG_COLS:
        df[col] = np.nan

    df["neighborhood"] = df["neighborhood"].astype(bundle.neighborhood_dtype)

    df["_baseline"] = (
        df["block_hour_dow_mean"].fillna(df["block_hour_mean"]).fillna(df["block_mean"]).fillna(bundle.global_mean)
    )
    return df


def _predict_rows(df: pd.DataFrame, bundle: ModelBundle) -> np.ndarray:
    X = df[bundle.features]
    residual = bundle.model.predict(X)
    return np.clip(df["_baseline"].values + residual, 0.0, 100.0)


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
    blocks["distance_m"] = _haversine_vec(req.lat, req.lon, blocks["lat"].to_numpy(), blocks["lon"].to_numpy())

    nearby = blocks[blocks["distance_m"] <= req.radius_meters].sort_values("distance_m")
    if nearby.empty:
        # Nothing in radius → snap to the 8 closest trained blocks so the user
        # still gets a usable answer.
        nearby = blocks.sort_values("distance_m").head(8)

    feat_df = _build_features(nearby, target_hour, req, BUNDLE)
    preds = _predict_rows(feat_df, BUNDLE)

    out: List[BlockPrediction] = []
    for (_, row), pct in zip(feat_df.iterrows(), preds):
        pct = float(round(pct, 2))
        total = int(row["total_spaces"]) if pd.notna(row["total_spaces"]) else 0
        avail = int(total * (1 - pct / 100))
        neigh = row["neighborhood"]
        out.append(
            BlockPrediction(
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                street=str(row["street"]) if "street" in row and pd.notna(row.get("street")) else None,
                neighborhood=str(neigh) if pd.notna(neigh) else None,
                total_spaces=total,
                distance_meters=int(row["distance_m"]),
                predicted_occupancy_pct=pct,
                available_spaces_estimate=avail,
                demand_level=_demand_level(pct),
                color=_color(pct),
            )
        )

    return BlocksResponse(
        destination_lat=req.lat,
        destination_lon=req.lon,
        radius_meters=req.radius_meters,
        predicted_at_hour=target_hour,
        minutes_away=req.minutes_away,
        total_blocks_found=len(out),
        blocks=out,
    )


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
    feat_df = _build_features(neigh_blocks, body.hour, synthetic_req, BUNDLE)
    preds = _predict_rows(feat_df, BUNDLE)
    pct = float(round(float(np.median(preds)), 2))
    avail = int(body.total_spaces * (1 - pct / 100))

    return ParkingPrediction(
        neighborhood=body.neighborhood,
        hour=body.hour,
        day_of_week=body.day_of_week,
        predicted_occupancy_pct=pct,
        available_spaces_estimate=avail,
        demand_level=_demand_level(pct),
        recommendation=_recommendation(pct),
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
            if "san francisco" in d.get("display_name", "").lower() or "california" in d.get("display_name", "").lower()
        ]
        return filtered or data
    except Exception as e:  # noqa: BLE001
        logger.error(f"Geocode error: {e}")
        return []
