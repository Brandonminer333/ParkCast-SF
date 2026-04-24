"""Shared fixtures for the functional test suite.

`app.main` loads real parquet assets + a trained model at FastAPI startup. The
functional tests swap those globals with minimal in-memory fakes so the API can
be exercised without the ~1.5 MB of parquet fixtures or a warm model.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="function")
def main_module(monkeypatch):
    """Import `app.main` with `_load_all` neutered, then swap in fake assets."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import app.main as main_mod

    main_mod = importlib.reload(main_mod)
    monkeypatch.setattr(main_mod, "_load_all", lambda: None)

    class DummyModel:
        def predict(self, X):
            return np.full(len(X), 5.0)

    blocks = pd.DataFrame(
        [
            {
                "lat": 37.7816,
                "lon": -122.3975,
                "neighborhood": "soma",
                "total_spaces": 42,
            },
            {
                "lat": 37.7820,
                "lon": -122.3968,
                "neighborhood": "soma",
                "total_spaces": 38,
            },
            {
                "lat": 37.7500,
                "lon": -122.4800,
                "neighborhood": "sunset",
                "total_spaces": 30,
            },
        ]
    )

    block_aggs = pd.DataFrame(
        [
            {
                "lat": 37.7816,
                "lon": -122.3975,
                "hour": h,
                "day_of_week": d,
                "block_mean": 60.0,
                "block_hour_mean": 62.0,
                "block_hour_dow_mean": 65.0,
            }
            for h in range(24)
            for d in range(7)
        ]
        + [
            {
                "lat": 37.7820,
                "lon": -122.3968,
                "hour": h,
                "day_of_week": d,
                "block_mean": 55.0,
                "block_hour_mean": 55.0,
                "block_hour_dow_mean": 55.0,
            }
            for h in range(24)
            for d in range(7)
        ]
    )

    lag_hist = pd.DataFrame(columns=["lat", "lon", "timestamp", "occupancy_pct"])
    cit_lookup = pd.DataFrame(
        [
            {"hour": h, "day_of_week": d, "citations_hourly_median": 0.0}
            for h in range(24)
            for d in range(7)
        ]
    )
    events = pd.DataFrame(columns=["date", "venue_lat", "venue_lon", "start_hour", "end_hour"])

    main_mod.model = DummyModel()
    main_mod.blocks = blocks
    main_mod.block_aggs = block_aggs
    main_mod.master = None
    main_mod.lag_hist = lag_hist
    main_mod.cit_lookup = cit_lookup
    main_mod.events = events
    main_mod.meta = {"metrics": {"residual_model": {"mae": 4.95}}}
    main_mod.global_baseline_mean = 50.0

    monkeypatch.setattr(main_mod, "weather_for", lambda ts: (62.0, 0))

    return main_mod


@pytest.fixture()
def client(main_module):
    return TestClient(main_module.app)
