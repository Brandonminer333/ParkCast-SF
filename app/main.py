"""
ParkCast SF — FastAPI Service

Citywide curb-availability advisory service.

/nearby returns every SF block within a radius, classified by what signal
we have for it:

  metered       → LightGBM-predicted occupancy + available spaces
  rpp           → residential permit zone (permit required)
  no_parking    → no parking / no stopping
  time_limited  → posted time limit (e.g., 2 hr)
  unmetered     → residential / free curb, no real-time signal

All "user-toggled" features (weather, holidays, school) are resolved
server-side from timestamp + Open-Meteo.
"""

import os
import json
import math
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, date
from functools import lru_cache
from typing import Optional, List

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
SFPARK_CAL_PATH = os.path.join(MODEL_DIR, "sfpark_calibration.parquet")
SFPARK_META_PATH = os.path.join(MODEL_DIR, "sfpark_calibration.meta.json")
META_PATH = os.path.join(MODEL_DIR, "LightGBM.meta.json")
EVENTS_PATH = os.path.join(DATA_DIR, "events.csv")

# ── Feature schema (must match train_lightgbm.py) ────────────────────────────
FEATURES_NUMERIC = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "is_school_day",
    "is_raining",
    "temperature",
    "event_intensity",
    "citation_count",
    "citations_hourly_median",
    "lat",
    "lon",
    "total_spaces",
    "block_mean",
    "block_hour_mean",
    "lag_7d",
    "lag_14d",
    "lag_28d",
]
FEATURES_CATEGORICAL = ["neighborhood"]
FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL
BASELINE_COL = "block_hour_dow_mean"

# US federal holidays observed in SF for training window. Keep hardcoded —
# small set, no extra dependency.
US_HOLIDAYS = {
    date(2025, 1, 1),
    date(2025, 1, 20),
    date(2025, 2, 17),
    date(2025, 5, 26),
    date(2025, 6, 19),
    date(2025, 7, 4),
    date(2025, 9, 1),
    date(2025, 10, 13),
    date(2025, 11, 11),
    date(2025, 11, 27),
    date(2025, 12, 25),
    date(2026, 1, 1),
    date(2026, 1, 19),
    date(2026, 2, 16),
    date(2026, 5, 25),
    date(2026, 6, 19),
    date(2026, 7, 4),
    date(2026, 9, 7),
    date(2026, 10, 12),
    date(2026, 11, 11),
    date(2026, 11, 26),
    date(2026, 12, 25),
}


# ── Globals populated at startup ─────────────────────────────────────────────
model = None
block_aggs: Optional[pd.DataFrame] = None
blocks: Optional[pd.DataFrame] = None
master: Optional[pd.DataFrame] = None
lag_hist: Optional[pd.DataFrame] = None
cit_lookup: Optional[pd.DataFrame] = None
events: Optional[pd.DataFrame] = None
sfpark_lookup: dict = {}  # {(pm_district, hour, is_weekend): occ_pct}
neighborhood_to_district: dict = {}  # {neighborhood: pm_district}
meta: dict = {}
global_baseline_mean: float = 50.0


def _load_all():
    global model, block_aggs, blocks, master, lag_hist, cit_lookup
    global events, meta, global_baseline_mean
    global sfpark_lookup, neighborhood_to_district

    model = joblib.load(MODEL_PATH)
    block_aggs = pd.read_parquet(BLOCK_AGG_PATH)
    blocks = pd.read_parquet(BLOCKS_PATH)
    master = pd.read_parquet(MASTER_PATH)
    lag_hist = pd.read_parquet(LAG_PATH)
    # Sort for faster timestamp slicing
    lag_hist = lag_hist.sort_values(["lat", "lon", "timestamp"]).reset_index(drop=True)
    cit_lookup = pd.read_parquet(CIT_LOOKUP_PATH)

    if os.path.exists(SFPARK_CAL_PATH):
        cal = pd.read_parquet(SFPARK_CAL_PATH)
        sfpark_lookup = {
            (r["pm_district"], int(r["hour"]), int(r["is_weekend"])): float(r["sfpark_occ_pct"])
            for _, r in cal.iterrows()
        }
    if os.path.exists(SFPARK_META_PATH):
        with open(SFPARK_META_PATH, encoding="utf-8") as f:
            mapping = json.load(f).get("district_to_neighborhoods", {})
        neighborhood_to_district = {
            nbh: district for district, nbhs in mapping.items() for nbh in nbhs
        }

    if os.path.exists(EVENTS_PATH):
        events = pd.read_csv(EVENTS_PATH, parse_dates=["date"])
    else:
        events = pd.DataFrame(columns=["date", "venue_lat", "venue_lon", "start_hour", "end_hour"])

    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
            global_baseline_mean = float(meta.get("global_mean", 50.0))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_all()
    yield


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="ParkCast SF API",
    description="Predicts SF curb parking occupancy by block coordinates.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Weather: on-demand Open-Meteo with daily cache ───────────────────────────
@lru_cache(maxsize=256)
def _weather_day(day_iso: str):
    """Fetch SF hourly temperature + precipitation for a given date.
    Returns dict {hour: (temp_f, is_raining)} or None on failure."""
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
        return {h: (float(t), 1 if (p or 0) > 0.1 else 0) for h, t, p in zip(hours, temps, precip)}
    except Exception:
        return None


def weather_for(ts: datetime) -> tuple[float, int]:
    day_data = _weather_day(ts.date().isoformat())
    if day_data and ts.hour in day_data:
        return day_data[ts.hour]
    return 60.0, 0  # sensible SF defaults


# ── Feature helpers ──────────────────────────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized haversine, scalar origin → arrays."""
    R = 6_371_000
    lat1r = math.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + math.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def is_school_day(d: date) -> int:
    # SF school year: rough heuristic — weekdays, Sep–May, not holiday.
    if d.weekday() >= 5 or d in US_HOLIDAYS:
        return 0
    if d.month in (6, 7, 8):
        return 0
    return 1


def event_intensity_at(lat: float, lon: float, ts: datetime, radius_m: float = 800.0) -> float:
    """Max over all active events of exp(-dist/300) within `radius_m`.
    Matches preprocess_real_data.py encoding so training and inference agree."""
    if events is None or events.empty:
        return 0.0
    same_day = events[events["date"].dt.date == ts.date()]
    if same_day.empty:
        return 0.0
    in_window = same_day[(same_day["start_hour"] <= ts.hour) & (ts.hour <= same_day["end_hour"])]
    if in_window.empty:
        return 0.0
    dists = haversine_vec(lat, lon, in_window["venue_lat"].values, in_window["venue_lon"].values)
    close = dists <= radius_m
    if not close.any():
        return 0.0
    return float(np.exp(-dists[close] / 300.0).max())


def lag_value(lat: float, lon: float, ts: datetime, days: int) -> float:
    """Return occupancy_pct at (lat, lon) at ts - `days`*24h, or NaN."""
    target = ts - timedelta(days=days)
    mask = (lag_hist["lat"] == lat) & (lag_hist["lon"] == lon) & (lag_hist["timestamp"] == target)
    hit = lag_hist.loc[mask, "occupancy_pct"]
    return float(hit.iloc[0]) if len(hit) else np.nan


def block_aggregates_for(lat: float, lon: float, hour: int, dow: int):
    """Return (block_mean, block_hour_mean, baseline) for a block. Falls back
    to global mean for unknown blocks."""
    row = block_aggs[
        (block_aggs["lat"] == lat)
        & (block_aggs["lon"] == lon)
        & (block_aggs["hour"] == hour)
        & (block_aggs["day_of_week"] == dow)
    ]
    if len(row):
        r = row.iloc[0]
        return (
            float(r["block_mean"]),
            float(r["block_hour_mean"]),
            float(r["block_hour_dow_mean"]),
        )
    # Hour-only fallback
    row = block_aggs[
        (block_aggs["lat"] == lat) & (block_aggs["lon"] == lon) & (block_aggs["hour"] == hour)
    ]
    if len(row):
        bhm = float(row["block_hour_mean"].iloc[0])
        bm = float(row["block_mean"].iloc[0])
        return bm, bhm, bhm
    # Block-only fallback
    row = block_aggs[(block_aggs["lat"] == lat) & (block_aggs["lon"] == lon)]
    if len(row):
        bm = float(row["block_mean"].iloc[0])
        return bm, bm, bm
    gm = global_baseline_mean
    return gm, gm, gm


def citations_median(hour: int, dow: int) -> float:
    row = cit_lookup[(cit_lookup["hour"] == hour) & (cit_lookup["day_of_week"] == dow)]
    return float(row["citations_hourly_median"].iloc[0]) if len(row) else 0.0


def build_feature_row(block_row, ts: datetime, temp_f: float, raining: int) -> dict:
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
        "citation_count": 0.0,  # unknown at inference, low importance
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


def sfpark_anchor(neighborhood: str, hour: int, is_weekend: int) -> Optional[float]:
    """Return SFpark ground-truth occupancy for this (neighborhood, hour,
    is_weekend) slot, or None if neighborhood isn't in the mapping."""
    district = neighborhood_to_district.get(neighborhood)
    if district is None:
        return None
    return sfpark_lookup.get((district, hour, is_weekend))


def predict_blocks(block_df: pd.DataFrame, ts: datetime) -> pd.DataFrame:
    temp_f, raining = weather_for(ts)
    rows = [build_feature_row(r, ts, temp_f, raining) for _, r in block_df.iterrows()]
    feat = pd.DataFrame(rows)
    baselines = feat.pop("_baseline").values
    feat["neighborhood"] = feat["neighborhood"].astype("category")
    for col in FEATURES_NUMERIC:
        feat[col] = pd.to_numeric(feat[col], errors="coerce")
    X = feat[FEATURES]
    residual = model.predict(X)
    # LightGBM is trained on 2025-2026 paid-session data; its absolute level
    # is what current meter data actually shows. Earlier we shifted outputs
    # to a 2011-2013 SFpark district anchor on the theory that paid-session
    # undercounts physical occupancy, but dev/validate_sfpark_calibration.py
    # shows the district anchor has 19.5 MAE at the block level and we have
    # no modern physical-occupancy ground truth to justify the shift. So we
    # keep LightGBM's native output for metered blocks. SFpark anchor is
    # still used in prior_for() for non-metered blocks where it's the best
    # available prior.
    occupancy = np.clip(baselines + residual, 0, 100)

    out = block_df.copy().reset_index(drop=True)
    out["predicted_occupancy_pct"] = occupancy.round(2)
    out["available_spaces"] = (out["total_spaces"] * (1 - occupancy / 100)).round().astype(int)
    return out


# ── Pydantic schemas ─────────────────────────────────────────────────────────
class PredictIn(BaseModel):
    lat: float = Field(..., description="Block latitude")
    lon: float = Field(..., description="Block longitude")
    timestamp: Optional[datetime] = Field(None, description="Prediction time (default: now)")


class PredictOut(BaseModel):
    lat: float
    lon: float
    neighborhood: str
    total_spaces: int
    timestamp: datetime
    predicted_occupancy_pct: float
    available_spaces: int
    demand_level: str


class NearbyIn(BaseModel):
    dest_lat: float = Field(..., description="Destination latitude")
    dest_lon: float = Field(..., description="Destination longitude")
    radius_m: float = Field(500.0, ge=50, le=5000)
    timestamp: Optional[datetime] = Field(None)
    limit: int = Field(20, ge=1, le=200)


class NearbyBlock(BaseModel):
    cnn: Optional[float] = None
    lat: float
    lon: float
    block_class: str
    corridor: Optional[str] = None
    limits: Optional[str] = None
    distance_m: float
    advisory: str
    score: float
    # Metered-only fields:
    neighborhood: Optional[str] = None
    total_spaces: Optional[int] = None
    predicted_occupancy_pct: Optional[float] = None
    available_spaces: Optional[int] = None
    demand_level: Optional[str] = None
    # Regulation-only fields:
    rpp_area: Optional[str] = None
    hrlimit: Optional[float] = None
    # Capacity/zone flags (from parking census + zone datasets):
    has_blue_zone: Optional[bool] = None
    has_bus_zone: Optional[bool] = None
    has_shuttle_stop: Optional[bool] = None


class NearbyOut(BaseModel):
    timestamp: datetime
    dest_lat: float
    dest_lon: float
    weather: dict
    blocks: List[NearbyBlock]


def demand_level(pct: float) -> str:
    if pct < 40:
        return "Low"
    if pct < 70:
        return "Medium"
    if pct < 85:
        return "High"
    return "Very High"


def advisory_for(
    cls: str, occ: Optional[float] = None, rpp_area: str = "", hrlimit: Optional[float] = None
) -> str:
    if cls == "metered":
        if occ is None:
            return "Metered — no prediction"
        return f"Paid meter — {demand_level(occ)} demand (~{occ:.0f}% full)"
    if cls == "rpp":
        area = f" (Area {rpp_area})" if rpp_area else ""
        return f"Residential permit required{area}"
    if cls == "no_parking":
        return "No parking / no stopping"
    if cls == "time_limited":
        if hrlimit:
            return f"Time-limited parking ({hrlimit:g} hr)"
        return "Time-limited parking"
    return "Free / unmetered curb — no real-time signal"


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "ParkCast SF API",
        "version": "2.0.0",
        "model": "LightGBM residual on block×hour×dow baseline",
        "endpoints": ["/health", "/predict", "/nearby", "/docs"],
    }


@app.get("/health")
def health():
    ok = all(x is not None for x in (model, block_aggs, blocks, master, lag_hist, cit_lookup))
    if not ok:
        raise HTTPException(503, "Model or assets not loaded")
    class_counts = master["block_class"].value_counts().to_dict() if master is not None else {}
    return {
        "status": "healthy",
        "model_features": FEATURES,
        "n_metered_blocks": int(len(blocks)),
        "n_master_blocks": int(len(master)),
        "block_classes": {k: int(v) for k, v in class_counts.items()},
        "lag_history_rows": int(len(lag_hist)),
        "events_loaded": int(len(events)) if events is not None else 0,
        "trained_test_mae": meta.get("metrics", {}).get("residual_model", {}).get("mae"),
    }


@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    ts = inp.timestamp or datetime.now()
    row = blocks[(blocks["lat"] == inp.lat) & (blocks["lon"] == inp.lon)]
    if row.empty:
        raise HTTPException(404, f"No known block at ({inp.lat}, {inp.lon})")
    result = predict_blocks(row, ts).iloc[0]
    return PredictOut(
        lat=float(result["lat"]),
        lon=float(result["lon"]),
        neighborhood=str(result["neighborhood"]),
        total_spaces=int(result["total_spaces"]),
        timestamp=ts,
        predicted_occupancy_pct=float(result["predicted_occupancy_pct"]),
        available_spaces=int(result["available_spaces"]),
        demand_level=demand_level(float(result["predicted_occupancy_pct"])),
    )


# Per-class score penalties (lower = more attractive). Metered uses predicted
# occupancy. Unmetered/time_limited borrow the SFpark district mean for the
# block's neighborhood at the requested hour so dense downtown curb is scored
# realistically instead of with a universal "55" residential prior. When no
# SFpark mapping exists (outer neighborhoods), we fall back to the constants
# below. RPP and no-parking are always penalized — they're not usable to a
# driver without a permit regardless of actual occupancy.
CLASS_PRIOR_FALLBACK = {
    "metered": None,  # use model output
    "unmetered": 55.0,  # residential curb fallback
    "time_limited": 60.0,
    "rpp": 90.0,
    "no_parking": 99.0,
}


def prior_for(
    cls: str,
    neighborhood: Optional[str],
    hour: int,
    is_weekend: int,
    event_intensity: float = 0.0,
    is_raining: int = 0,
    temperature_f: Optional[float] = None,
) -> float:
    """Baseline occupancy prior for non-metered blocks. LightGBM ignores
    event_intensity (event rows are 0.02% of metered training data and
    recurring-venue patterns are baked into block×hr×dow means). The
    non-metered path has no such baseline, so we apply three bumps:
      - event_intensity ×25  (festival/parade proximity)
      - rain +4              (fewer walk/cycle trips → more curb use)
      - extreme temp +2      (< 45°F or > 85°F; people drive more)
    """
    if cls in ("rpp", "no_parking"):
        base = CLASS_PRIOR_FALLBACK[cls]
    elif neighborhood:
        anchor = sfpark_anchor(neighborhood, hour, is_weekend)
        base = anchor if anchor is not None else CLASS_PRIOR_FALLBACK.get(cls, 70.0)
    else:
        base = CLASS_PRIOR_FALLBACK.get(cls, 70.0)
    bump = 25.0 * max(0.0, min(1.0, event_intensity))
    if is_raining:
        bump += 4.0
    if temperature_f is not None and (temperature_f < 45.0 or temperature_f > 85.0):
        bump += 2.0
    return min(99.0, base + bump)


@app.post("/nearby", response_model=NearbyOut)
def nearby(inp: NearbyIn):
    if model is None or master is None:
        raise HTTPException(503, "Model not loaded")
    ts = inp.timestamp or datetime.now()

    # Distance filter over the citywide master catalog
    dists = haversine_vec(inp.dest_lat, inp.dest_lon, master["lat"].values, master["lon"].values)
    mask = dists <= inp.radius_m
    if not mask.any():
        return NearbyOut(
            timestamp=ts, dest_lat=inp.dest_lat, dest_lon=inp.dest_lon, weather={}, blocks=[]
        )
    near = master.loc[mask].copy()
    near["distance_m"] = dists[mask]

    # Metered subset → run LightGBM keyed by metered_lat/metered_lon
    metered_mask = (near["block_class"] == "metered") & near["metered_lat"].notna()
    pred_map: dict = {}
    if metered_mask.any():
        met = near.loc[metered_mask, ["metered_lat", "metered_lon"]].rename(
            columns={"metered_lat": "lat", "metered_lon": "lon"}
        )
        met = met.merge(blocks, on=["lat", "lon"], how="left")
        preds = predict_blocks(met, ts)
        for i, idx in enumerate(near.index[metered_mask]):
            pred_map[idx] = preds.iloc[i]

    hour = ts.hour
    is_wknd = 1 if ts.weekday() >= 5 else 0
    temp_f, raining = weather_for(ts)
    results = []
    for idx, r in near.iterrows():
        cls = str(r["block_class"])
        dist = float(r["distance_m"])
        master_nbh = str(r["neighborhood"]) if pd.notna(r.get("neighborhood")) else None
        occ = avail = total = nbh = None
        dlvl = None
        if cls == "metered" and idx in pred_map:
            p = pred_map[idx]
            occ = float(p["predicted_occupancy_pct"])
            avail = int(p["available_spaces"])
            total = int(p["total_spaces"])
            nbh = str(p["neighborhood"])
            dlvl = demand_level(occ)
            occ_score = occ
        else:
            evi = event_intensity_at(float(r["lat"]), float(r["lon"]), ts)
            occ_score = prior_for(
                cls, master_nbh, hour, is_wknd, evi, is_raining=raining, temperature_f=temp_f
            )
            nbh = master_nbh
            if pd.notna(r.get("total_spaces")):
                total = int(r["total_spaces"])
                avail = int(round(total * (1 - occ_score / 100)))
                dlvl = demand_level(occ_score)

        # Zone penalties: bus/shuttle zones remove spaces; blue curb is
        # permit-only. Each adds a small score penalty so blocks with
        # posted no-park zones get ranked below equivalently-full blocks
        # without them.
        zone_penalty = 0.0
        if bool(r.get("has_bus_zone")):
            zone_penalty += 3.0
        if bool(r.get("has_shuttle_stop")):
            zone_penalty += 2.0
        if bool(r.get("has_blue_zone")):
            zone_penalty += 2.0

        score = 0.7 * occ_score + 0.3 * (dist / inp.radius_m * 100) + zone_penalty
        rpp_area = str(r.get("rpp_area") or "") if pd.notna(r.get("rpp_area")) else ""
        hrl = float(r["hrlimit"]) if pd.notna(r.get("hrlimit")) else None
        results.append(
            (
                score,
                NearbyBlock(
                    cnn=float(r["cnn"]) if pd.notna(r.get("cnn")) else None,
                    lat=float(r["lat"]),
                    lon=float(r["lon"]),
                    block_class=cls,
                    corridor=str(r["corridor"]) if pd.notna(r.get("corridor")) else None,
                    limits=str(r["limits"]) if pd.notna(r.get("limits")) else None,
                    distance_m=round(dist, 1),
                    advisory=advisory_for(cls, occ, rpp_area, hrl),
                    score=round(score, 2),
                    neighborhood=nbh,
                    total_spaces=total,
                    predicted_occupancy_pct=round(occ, 2) if occ is not None else None,
                    available_spaces=avail,
                    demand_level=dlvl,
                    rpp_area=rpp_area or None,
                    hrlimit=hrl,
                    has_blue_zone=(
                        bool(r.get("has_blue_zone")) if pd.notna(r.get("has_blue_zone")) else None
                    ),
                    has_bus_zone=(
                        bool(r.get("has_bus_zone")) if pd.notna(r.get("has_bus_zone")) else None
                    ),
                    has_shuttle_stop=(
                        bool(r.get("has_shuttle_stop"))
                        if pd.notna(r.get("has_shuttle_stop"))
                        else None
                    ),
                ),
            )
        )
    results.sort(key=lambda x: x[0])
    top = [b for _, b in results[: inp.limit]]

    return NearbyOut(
        timestamp=ts,
        dest_lat=inp.dest_lat,
        dest_lon=inp.dest_lon,
        weather={"temperature_f": temp_f, "is_raining": int(raining)},
        blocks=top,
    )
