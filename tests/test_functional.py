"""Functional tests for the ParkCast SF API.

Each test drives a single endpoint through FastAPI's `TestClient` while the
`main_module` fixture (see `conftest.py`) substitutes in a `DummyModel` and a
tiny hand-rolled parking database. No parquet files, no trained model, and no
external HTTP required.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.functional


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
