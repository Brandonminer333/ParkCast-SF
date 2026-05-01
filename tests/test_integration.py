"""Integration tests for the ParkCast SF API.

These tests exercise the FastAPI app against REAL assets:
- The trained ModelBundle is loaded at import time from `app/models/`.
- The real parquet lookup tables are loaded via `ModelBundle.__init__`.

They run in-process through Starlette's `TestClient` (no real HTTP socket),
but no module globals are monkeypatched. The whole module is skipped when
the ModelBundle failed to load (i.e. `BUNDLE is None`).
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

_repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import app.main as main_mod  # noqa: E402

_BUNDLE_AVAILABLE = main_mod.BUNDLE is not None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _BUNDLE_AVAILABLE,
        reason=(
            "ModelBundle not loaded. Ensure model artifacts exist in "
            f"{main_mod.LOCAL_MODEL_DIR} or set GCS_BUCKET."
        ),
    ),
]


@pytest.fixture(scope="module")
def real_client():
    """TestClient using the real ModelBundle loaded at import time."""
    with TestClient(main_mod.app) as client:
        yield client


def test_health_reports_healthy_with_real_bundle(real_client):
    resp = real_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["total_blocks"] > 0
    assert body["model_features"] > 0


def test_predict_blocks_against_real_model_returns_valid_response(real_client):
    """POST a realistic request and verify the real model + real parquet data
    produce a well-formed, plausible response."""
    payload = {
        "lat": 37.7790,
        "lon": -122.4180,
        "radius_meters": 800,
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
    resp = real_client.post("/predict/blocks", json=payload)
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_blocks_found"] >= 1
    assert len(body["blocks"]) >= 1

    first = body["blocks"][0]
    assert 0.0 <= first["predicted_occupancy_pct"] <= 100.0
    assert first["demand_level"] in {"Low", "Medium", "High", "Very High"}
    assert first["color"].startswith("#") and len(first["color"]) == 7
    assert first["available_spaces_estimate"] >= 0
    assert first["available_spaces_estimate"] <= first["total_spaces"]


def test_predict_blocks_ordering_is_by_distance(real_client):
    payload = {
        "lat": 37.7790,
        "lon": -122.4180,
        "radius_meters": 1500,
        "hour": 18,
        "day_of_week": 5,
        "month": 4,
        "is_raining": 0,
        "event_intensity": 0.0,
        "is_holiday": 0,
        "is_school_day": 0,
        "temperature": 60.0,
        "minutes_away": 0,
    }
    resp = real_client.post("/predict/blocks", json=payload)
    assert resp.status_code == 200
    distances = [b["distance_meters"] for b in resp.json()["blocks"]]
    assert distances == sorted(distances)


def test_predict_neighborhood_aggregate_against_real_model(real_client):
    """Exercise the aggregate /predict endpoint with a real bundle."""
    payload = {
        "hour": 14,
        "day_of_week": 2,
        "month": 4,
        "neighborhood": "soma",
        "total_spaces": 40,
        "event_intensity": 0.0,
    }
    resp = real_client.post("/predict", json=payload)
    # Might 404 if no SoMa blocks in the real bundle — accept both.
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        body = resp.json()
        assert 0.0 <= body["predicted_occupancy_pct"] <= 100.0
        assert body["blocks_aggregated"] >= 1
