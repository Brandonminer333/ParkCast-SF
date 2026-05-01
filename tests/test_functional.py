"""Functional tests for the ParkCast SF API.

Each test drives a single endpoint through FastAPI's `TestClient` while the
`client` fixture (see `conftest.py`) substitutes in a fake ModelBundle with a
`DummyModel` and a tiny hand-rolled parking database. No parquet files, no
trained model, and no external HTTP required.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.functional


# ── Root ────────────────────────────────────────────────────────────


def test_root_returns_service_info(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "ParkCast SF API"
    assert "POST /predict/blocks" in body["endpoints"]


# ── Health ──────────────────────────────────────────────────────────


def test_health_503_when_bundle_missing(client, monkeypatch):
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "BUNDLE", None)
    resp = client.get("/health")
    assert resp.status_code == 503
    assert "Model not loaded" in resp.json()["detail"]


def test_health_ok_reports_block_count(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "healthy"
    assert payload["model_features"] == 21  # len(FAKE_FEATURES)
    assert payload["total_blocks"] == 3
    assert payload["test_mae"] == 4.95


# ── POST /predict/blocks ───────────────────────────────────────────


def test_predict_blocks_returns_ranked_candidates_within_radius(client):
    """SoMa blocks are ~80m apart. Sunset block is ~8km away.
    With radius=1500 from the first SoMa block, only the two SoMa blocks
    are returned, sorted by distance.
    """
    payload = {
        "lat": 37.7816,
        "lon": -122.3975,
        "radius_meters": 1500,
        "hour": 18,
        "day_of_week": 4,
        "month": 4,
        "is_raining": 0,
        "event_intensity": 0.0,
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

    # Must be sorted by distance
    distances = [b["distance_meters"] for b in returned_blocks]
    assert distances == sorted(distances)

    # Every block must have these keys
    required_keys = {
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
        "coverage",
    }
    for b in returned_blocks:
        assert required_keys <= set(b.keys())

    # First block: baseline 65 + residual 5 = 70%
    first = returned_blocks[0]
    assert first["predicted_occupancy_pct"] == pytest.approx(70.0, abs=0.01)
    assert first["demand_level"] == "High"
    assert first["color"] == "#f97316"
    assert first["available_spaces_estimate"] == int(42 * 0.30)


def test_predict_blocks_falls_back_to_nearest_when_radius_empty(client):
    """Querying far from any block with a tiny radius returns the 8 nearest
    (or all 3 in our fixture)."""
    payload = {
        "lat": 37.0,
        "lon": -121.0,
        "radius_meters": 100,
        "hour": 12,
        "day_of_week": 2,
        "month": 4,
        "is_raining": 0,
        "event_intensity": 0.0,
        "is_holiday": 0,
        "is_school_day": 1,
        "temperature": 62.0,
        "minutes_away": 0,
    }
    resp = client.post("/predict/blocks", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_blocks_found"] == 3


def test_predict_blocks_503_when_bundle_missing(client, monkeypatch):
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "BUNDLE", None)
    payload = {
        "lat": 37.7816,
        "lon": -122.3975,
        "radius_meters": 500,
        "hour": 10,
        "day_of_week": 1,
        "month": 1,
        "is_raining": 0,
        "event_intensity": 0.0,
        "is_holiday": 0,
        "is_school_day": 1,
        "temperature": 60.0,
        "minutes_away": 0,
    }
    resp = client.post("/predict/blocks", json=payload)
    assert resp.status_code == 503


# ── POST /predict (aggregate neighborhood) ─────────────────────────


def test_predict_neighborhood_soma(client):
    """Aggregate prediction for 'soma' should succeed and return
    a plausible response."""
    payload = {
        "hour": 14,
        "day_of_week": 3,
        "month": 5,
        "neighborhood": "soma",
        "total_spaces": 40,
        "event_intensity": 0.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["neighborhood"] == "soma"
    assert 0.0 <= body["predicted_occupancy_pct"] <= 100.0
    assert body["demand_level"] in {"Low", "Medium", "High", "Very High"}
    assert body["blocks_aggregated"] == 2


def test_predict_neighborhood_404_for_unknown(client):
    payload = {
        "hour": 10,
        "day_of_week": 0,
        "month": 1,
        "neighborhood": "atlantis",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 404
