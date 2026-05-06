"""Model-layer tests for ParkCast SF.

These tests target the ML model + feature pipeline directly — `build_features`,
`predict_rows`, and the `ModelBundle` invariants — without going through the
FastAPI app. Most run against the in-memory `fake_bundle` fixture from
`conftest.py`; the final two exercise the real `ModelBundle` loaded from
`app/models/` and auto-skip when those artifacts are absent.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

_repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.features import build_features, predict_rows  # noqa: E402
from app.schemas import BlocksRequest  # noqa: E402

pytestmark = pytest.mark.model


def _make_request(**overrides) -> BlocksRequest:
    base = dict(
        lat=37.7816,
        lon=-122.3975,
        radius_meters=500,
        hour=14,
        day_of_week=2,
        month=4,
        is_raining=0,
        is_holiday=0,
        is_school_day=1,
        temperature=62.0,
        event_intensity=0.0,
        minutes_away=0,
    )
    base.update(overrides)
    return BlocksRequest(**base)


# ── 1. build_features produces every feature the model expects ─────


def test_build_features_emits_all_required_columns(fake_bundle):
    req = _make_request()
    feat_df = build_features(fake_bundle.blocks, target_hour=req.hour, req=req, bundle=fake_bundle)

    missing = [c for c in fake_bundle.features if c not in feat_df.columns]
    assert not missing, f"missing feature columns: {missing}"
    assert "_baseline" in feat_df.columns
    assert len(feat_df) == len(fake_bundle.blocks)


# ── 2. Baseline fallback hierarchy ─────────────────────────────────


def test_build_features_baseline_falls_back_to_global_mean(fake_bundle):
    """The Sunset block has no entry in `block_aggs`, so its `_baseline`
    must come from `bundle.global_mean`."""
    req = _make_request()
    feat_df = build_features(fake_bundle.blocks, target_hour=req.hour, req=req, bundle=fake_bundle)

    sunset_row = feat_df[feat_df["neighborhood"] == "sunset"].iloc[0]
    assert sunset_row["_baseline"] == pytest.approx(fake_bundle.global_mean)

    soma_row = feat_df[(feat_df["lat"] == 37.7816) & (feat_df["lon"] == -122.3975)].iloc[0]
    # block_hour_dow_mean = 65.0 wins over the other (lower-priority) means.
    assert soma_row["_baseline"] == pytest.approx(65.0)


# ── 3. predict_rows reconstructs absolute occupancy from residual ──


def test_predict_rows_adds_residual_to_baseline(fake_bundle):
    req = _make_request()
    feat_df = build_features(fake_bundle.blocks, target_hour=req.hour, req=req, bundle=fake_bundle)

    preds = predict_rows(feat_df, fake_bundle)
    expected = np.clip(feat_df["_baseline"].to_numpy() + 5.0, 0.0, 100.0)

    assert preds.shape == (len(fake_bundle.blocks),)
    np.testing.assert_allclose(preds, expected)


# ── 4. predict_rows clips into [0, 100] ────────────────────────────


def test_predict_rows_clips_to_valid_occupancy_range(fake_bundle):
    """A residual that would push the result above 100% must be clipped."""

    class ExplodingModel:
        def predict(self, X):
            return np.full(len(X), 250.0)

    fake_bundle.model = ExplodingModel()
    req = _make_request()
    feat_df = build_features(fake_bundle.blocks, target_hour=req.hour, req=req, bundle=fake_bundle)
    preds = predict_rows(feat_df, fake_bundle)

    assert (preds >= 0.0).all()
    assert (preds <= 100.0).all()


# ── 5. Lag columns stay NaN at inference ───────────────────────────


def test_build_features_leaves_lag_columns_nan_when_lookup_missing(fake_bundle):
    """When the bundle has no lag_lookup (e.g. lag_history.parquet absent),
    every lag column falls back to NaN and LightGBM uses surrogate splits."""
    from app.constants import LAG_COLS

    req = _make_request()
    feat_df = build_features(fake_bundle.blocks, target_hour=req.hour, req=req, bundle=fake_bundle)

    for col in LAG_COLS:
        assert feat_df[col].isna().all(), f"{col} should be NaN at inference"


def test_build_features_uses_lag_lookup_when_present(fake_bundle):
    """When a lag_lookup is attached, build_features merges its values onto
    matching (lat, lon, hour, dow) rows instead of leaving them NaN."""
    req = _make_request()
    lookup = pd.DataFrame(
        [
            {
                "lat": 37.7816,
                "lon": -122.3975,
                "hour": req.hour,
                "day_of_week": req.day_of_week,
                "lag_1d": 41.0,
                "lag_2d": 42.0,
                "lag_7d": 47.0,
                "lag_14d": 48.0,
                "lag_28d": 49.0,
            }
        ]
    )
    fake_bundle.lag_lookup = lookup

    feat_df = build_features(fake_bundle.blocks, target_hour=req.hour, req=req, bundle=fake_bundle)
    soma = feat_df[(feat_df["lat"] == 37.7816) & (feat_df["lon"] == -122.3975)].iloc[0]
    assert soma["lag_1d"] == 41.0
    assert soma["lag_28d"] == 49.0
    # Blocks not in the lookup keep NaN.
    other = feat_df[feat_df["neighborhood"] == "sunset"].iloc[0]
    assert pd.isna(other["lag_1d"])


# ── 6. ModelBundle exposes a usable model ──────────────────────────


def test_real_model_bundle_predicts_on_single_row():
    """Load the real bundle and run one prediction through the full pipeline."""
    import app.main as main_mod

    if main_mod.BUNDLE is None:
        pytest.skip("ModelBundle artifacts unavailable")

    bundle = main_mod.BUNDLE
    one_block = bundle.blocks.head(1).copy()
    req = _make_request(lat=float(one_block.iloc[0]["lat"]), lon=float(one_block.iloc[0]["lon"]))

    feat_df = build_features(one_block, target_hour=req.hour, req=req, bundle=bundle)
    preds = predict_rows(feat_df, bundle)

    assert preds.shape == (1,)
    assert 0.0 <= float(preds[0]) <= 100.0


# ── 7. Model feature order matches metadata ────────────────────────


def test_real_model_feature_list_matches_meta():
    """The model was trained on `meta['features']`; that list must equal
    `bundle.features` so `predict_rows` selects columns in the right order."""
    import app.main as main_mod

    if main_mod.BUNDLE is None:
        pytest.skip("ModelBundle artifacts unavailable")

    bundle = main_mod.BUNDLE
    assert bundle.features == bundle.meta["features"]
    assert len(bundle.features) > 0
    # neighborhood is always part of the feature set for this model.
    assert "neighborhood" in bundle.features
    # global_mean must be a finite occupancy-ish percentage.
    assert 0.0 <= float(bundle.global_mean) <= 100.0
