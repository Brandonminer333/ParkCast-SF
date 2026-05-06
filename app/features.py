"""Feature-building and prediction helpers for ParkCast SF inference.

These are pure functions that operate on DataFrames — no FastAPI, no
global state, easy to unit-test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.bundle import ModelBundle
from app.constants import LAG_COLS
from app.schemas import BlocksRequest


def haversine_vec(
    lat0: float,
    lon0: float,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine distance in meters from a single point to arrays."""
    R = 6_371_000
    phi0 = math.radians(lat0)
    phi1 = np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlam = np.radians(lons - lon0)
    a = np.sin(dphi / 2) ** 2 + math.cos(phi0) * np.cos(phi1) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def build_features(
    blocks_df: pd.DataFrame,
    target_hour: int,
    req: BlocksRequest,
    bundle: ModelBundle,
) -> pd.DataFrame:
    """Attach every feature LightGBM needs to each block row.

    Returns a DataFrame with all ``bundle.features`` columns plus a
    ``_baseline`` column used to reconstruct the non-residual prediction.
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

    # Fill inferred (non-metered) blocks with KNN-derived baselines.
    if bundle.inferred_aggs is not None and "cnn" in df.columns:
        infa = bundle.inferred_aggs.rename(
            columns={
                "block_hour_dow_mean": "_inf_bhd",
                "block_hour_mean": "_inf_bh",
                "block_mean": "_inf_bm",
            }
        )
        df = df.merge(infa, on=["cnn", "hour", "day_of_week"], how="left")
        df["block_hour_dow_mean"] = df["block_hour_dow_mean"].fillna(df["_inf_bhd"])
        df["block_hour_mean"] = df["block_hour_mean"].fillna(df["_inf_bh"])
        df["block_mean"] = df["block_mean"].fillna(df["_inf_bm"])
        df = df.drop(columns=["_inf_bhd", "_inf_bh", "_inf_bm"])

    lag_lookup = getattr(bundle, "lag_lookup", None)
    if lag_lookup is not None:
        df = df.merge(lag_lookup, on=["lat", "lon", "hour", "day_of_week"], how="left")
    for col in LAG_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df["neighborhood"] = df["neighborhood"].astype(bundle.neighborhood_dtype)

    df["_baseline"] = (
        df["block_hour_dow_mean"]
        .fillna(df["block_hour_mean"])
        .fillna(df["block_mean"])
        .fillna(bundle.global_mean)
    )
    return df


def predict_rows(df: pd.DataFrame, bundle: ModelBundle) -> np.ndarray:
    """Run the residual model and reconstruct absolute occupancy %."""
    X = df[bundle.features]
    residual = bundle.model.predict(X)
    return np.clip(df["_baseline"].values + residual, 0.0, 100.0)
