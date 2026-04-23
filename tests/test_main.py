"""Tests for app.main.

The service loads real parquet assets at startup. These tests swap the module
globals with minimal fakes so the FastAPI app can be exercised without
shipping the ~1.5 MB of parquet fixtures.
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
    events = pd.DataFrame(
        columns=["date", "venue_lat", "venue_lon", "start_hour", "end_hour"]
    )

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


def test_root_returns_welcome(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"].startswith("Welcome to ParkCast SF API")
    assert "predict_blocks" in body["endpoints"]


def test_health_503_when_model_missing(client, main_module):
    main_module.model = None
    resp = client.get("/health")
    assert resp.status_code == 503
    assert "Model not loaded" in resp.json()["detail"]


def test_health_ok_reports_block_count(client, main_module):
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True
    assert payload["num_features"] == len(main_module.FEATURES)
    assert payload["total_blocks_in_db"] == 3
    assert payload["trained_test_mae"] == 4.95


@pytest.mark.parametrize(
    "occ,expected",
    [
        (0, "Low"),
        (39.9, "Low"),
        (40, "Medium"),
        (69.9, "Medium"),
        (70, "High"),
        (84.9, "High"),
        (85, "Very High"),
        (100, "Very High"),
    ],
)
def test_demand_level_boundaries(main_module, occ, expected):
    assert main_module.demand_level(occ) == expected


@pytest.mark.parametrize(
    "occ,expected_hex",
    [
        (10, "#22c55e"),
        (55, "#f59e0b"),
        (77, "#f97316"),
        (95, "#ef4444"),
    ],
)
def test_color_matches_legend(main_module, occ, expected_hex):
    assert main_module.color_for(occ) == expected_hex


def test_is_school_day_heuristic(main_module):
    from datetime import date as _date

    assert main_module.is_school_day(_date(2026, 3, 4)) == 1
    assert main_module.is_school_day(_date(2026, 3, 7)) == 0
    assert main_module.is_school_day(_date(2026, 7, 15)) == 0
    assert main_module.is_school_day(_date(2026, 1, 1)) == 0


def test_predict_blocks_returns_ranked_candidates_within_radius(client):
    payload = {
        "lat": 37.7816,
        "lon": -122.3975,
        "radius_meters": 1500,
        "hour": 18,
        "day_of_week": 4,
        "month": 4,
        "is_raining": 0,
        "has_nearby_event": 0,
        "is_holiday": 0,
        "is_school_day": 1,
        "temperature": 62.0,
        "minutes_away": 0,
    }
    resp = client.post("/predict/blocks", json=payload)
    assert resp.status_code == 200
    body = resp.json()

    assert body["destination_lat"] == payload["lat"]
    assert body["radius_meters"] == 1500
    assert body["total_blocks_found"] == 2
    returned_blocks = body["blocks"]
    assert len(returned_blocks) == 2

    distances = [b["distance_meters"] for b in returned_blocks]
    assert distances == sorted(distances)

    required_keys = {
        "block_id",
        "street",
        "lat",
        "lon",
        "total_spaces",
        "neighborhood",
        "distance_meters",
        "predicted_occupancy_pct",
        "available_spaces_estimate",
        "demand_level",
        "color",
    }
    for b in returned_blocks:
        assert required_keys <= set(b.keys())

    first = returned_blocks[0]
    assert first["predicted_occupancy_pct"] == pytest.approx(70.0, abs=0.01)
    assert first["demand_level"] == "High"
    assert first["color"] == "#f97316"
    assert first["available_spaces_estimate"] == int(round(42 * 0.30))


def test_predict_blocks_falls_back_to_nearest_when_radius_empty(client):
    payload = {
        "lat": 37.0,
        "lon": -121.0,
        "radius_meters": 100,
        "hour": 12,
        "day_of_week": 2,
        "month": 4,
        "is_raining": 0,
        "has_nearby_event": 0,
        "is_holiday": 0,
        "is_school_day": 1,
        "temperature": 62.0,
        "minutes_away": 0,
    }
    resp = client.post("/predict/blocks", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_blocks_found"] == 3


def test_predict_blocks_503_when_model_missing(client, main_module):
    main_module.model = None
    payload = {
        "lat": 37.7816,
        "lon": -122.3975,
        "radius_meters": 500,
        "hour": 10,
        "day_of_week": 1,
        "month": 1,
        "is_raining": 0,
        "has_nearby_event": 0,
        "is_holiday": 0,
        "is_school_day": 1,
        "temperature": 60.0,
        "minutes_away": 0,
    }
    resp = client.post("/predict/blocks", json=payload)
    assert resp.status_code == 503
