"""Shared fixtures for the functional test suite.

`app.main` loads a real ModelBundle (model + parquets) at import time. The
functional tests swap the module-level `BUNDLE` with a minimal in-memory
fake so the API can be exercised without real parquet fixtures or a warm model.
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

_repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Minimal feature list matching what _build_features() produces.
FAKE_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "is_school_day",
    "is_raining",
    "temperature",
    "event_intensity",
    "neighborhood",
    "block_hour_dow_mean",
    "block_hour_mean",
    "block_mean",
    "citations_hourly_median",
    "lag_1d",
    "lag_2d",
    "lag_7d",
    "lag_14d",
    "lag_28d",
    "lag_3d_mean",
    "lag_7d_mean",
]


class DummyModel:
    """Predict a constant residual of +5.0 for every row."""

    def predict(self, X):
        return np.full(len(X), 5.0)


@pytest.fixture(scope="function")
def fake_bundle():
    """Minimal in-memory ModelBundle substitute for functional tests.

    Three blocks: two in SoMa (with block_aggs) and one in Sunset (no
    block_aggs → falls back to global_mean).

    DummyModel returns residual=+5.0, so:
      SoMa block 1: baseline 65 + 5 = 70% → High
      SoMa block 2: baseline 55 + 5 = 60% → Medium
      Sunset block:  baseline 45 + 5 = 50% → Medium  (global_mean fallback)
    """
    bundle = SimpleNamespace()
    bundle.src = "test-fixture"
    bundle.model = DummyModel()
    bundle.features = list(FAKE_FEATURES)
    bundle.meta = {
        "features": FAKE_FEATURES,
        "global_mean": 45.0,
        "metrics": {"residual_model": {"mae": 4.95, "r2": 0.72}},
    }
    bundle.global_mean = 45.0

    bundle.blocks = pd.DataFrame(
        [
            {
                "lat": 37.7816,
                "lon": -122.3975,
                "neighborhood": "soma",
                "total_spaces": 42,
                "coverage": "metered",
                "street": "Howard St (2nd-3rd)",
                "cnn": 1001,
            },
            {
                "lat": 37.7820,
                "lon": -122.3968,
                "neighborhood": "soma",
                "total_spaces": 38,
                "coverage": "metered",
                "street": "Folsom St (3rd-4th)",
                "cnn": 1002,
            },
            {
                "lat": 37.7500,
                "lon": -122.4800,
                "neighborhood": "sunset",
                "total_spaces": 30,
                "coverage": "metered",
                "street": "Irving St (9th-10th)",
                "cnn": 1003,
            },
        ]
    )

    bundle.block_aggs = pd.DataFrame(
        [
            {
                "lat": lat,
                "lon": lon,
                "hour": h,
                "day_of_week": d,
                "block_mean": bm,
                "block_hour_mean": bhm,
                "block_hour_dow_mean": bhdm,
            }
            for lat, lon, bm, bhm, bhdm in [
                (37.7816, -122.3975, 60.0, 62.0, 65.0),
                (37.7820, -122.3968, 55.0, 55.0, 55.0),
            ]
            for h in range(24)
            for d in range(7)
        ]
    )

    bundle.static = pd.DataFrame(
        [
            {"lat": 37.7816, "lon": -122.3975},
            {"lat": 37.7820, "lon": -122.3968},
            {"lat": 37.7500, "lon": -122.4800},
        ]
    )

    bundle.cit_med = pd.DataFrame(
        [
            {"hour": h, "day_of_week": d, "citations_hourly_median": 0.5}
            for h in range(24)
            for d in range(7)
        ]
    )

    bundle.inferred_aggs = None

    neighborhoods = sorted(bundle.blocks["neighborhood"].dropna().unique().tolist())
    bundle.neighborhood_dtype = pd.CategoricalDtype(categories=neighborhoods)

    return bundle


@pytest.fixture()
def client(fake_bundle, monkeypatch):
    """TestClient with BUNDLE swapped for the fake."""
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "BUNDLE", fake_bundle)
    return TestClient(main_mod.app)
