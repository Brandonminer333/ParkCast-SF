"""Integration tests for the ParkCast SF API.

These tests exercise the FastAPI app against REAL assets:
- The trained model is loaded via MLflow when `MLFLOW_TRACKING_URI` is set,
  and via the legacy on-disk `app/models/LightGBM.pkl` otherwise.
- The real parquet lookup tables are loaded via `_load_all`.
- The real weather provider (`weather_for` -> Open-Meteo) is invoked over the
  network.

They run in-process through Starlette's `TestClient` (no real HTTP socket),
but no module globals are monkeypatched. The whole module is skipped when
neither model source is available. Weather-dependent assertions are softened
/ skipped when the Open-Meteo API is unreachable, so the suite stays green
offline as long as a model source is present.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import socket
import sys

import pytest
from fastapi.testclient import TestClient

_repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.main import MODEL_PATH  # noqa: E402

_MODEL_AVAILABLE = bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip()) or os.path.exists(
    MODEL_PATH
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _MODEL_AVAILABLE,
        reason=(
            "No model source available. Set MLFLOW_TRACKING_URI to load the "
            f"champion from the registry, or place a legacy model at {MODEL_PATH}."
        ),
    ),
]


def _weather_api_reachable(
    host: str = "api.open-meteo.com", port: int = 443, timeout: float = 2.0
) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def real_client():
    """Fresh `app.main` import with the real `_load_all` running via lifespan."""
    import app.main as main_mod

    main_mod = importlib.reload(main_mod)
    with TestClient(main_mod.app) as client:
        yield client


def test_lifespan_loads_real_model_and_blocks(real_client):
    resp = real_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["total_blocks_in_db"] > 0
    assert body["num_features"] > 0


@pytest.mark.skipif(
    not os.environ.get("MLFLOW_TRACKING_URI", "").strip(),
    reason="MLFLOW_TRACKING_URI not set; skipping MLflow fetch smoke test.",
)
def test_model_is_fetched_from_mlflow_registry():
    """Smoke-test the MLflow loading path end-to-end.

    Reloads `app.main` with the current environment, drives FastAPI's lifespan
    (so `_load_all` runs for real), and verifies:

    - `model_source` starts with `mlflow:` (i.e. the registry was used, not
      the local joblib fallback).
    - The loaded model is a non-None estimator bound to the expected registry
      URI of `models:/<MLFLOW_MODEL_NAME>@<MLFLOW_MODEL_STAGE>`.
    - `/health` reports `healthy` with a feature count that matches the
      booster's actual feature list.

    Skipped when `MLFLOW_TRACKING_URI` is unset so offline dev + CI aren't
    broken by a missing registry.
    """
    import app.main as main_mod

    main_mod = importlib.reload(main_mod)
    assert main_mod.MLFLOW_TRACKING_URI, "MLFLOW_TRACKING_URI should be non-empty here"

    with TestClient(main_mod.app) as client:
        assert main_mod.model is not None, "model should be populated after lifespan"

        expected_uri = f"mlflow:models:/{main_mod.MLFLOW_MODEL_NAME}@{main_mod.MLFLOW_MODEL_STAGE}"
        assert main_mod.model_source == expected_uri, (
            f"expected model to be fetched from {expected_uri!r}, " f"got {main_mod.model_source!r}"
        )

        booster = getattr(main_mod.model, "booster_", None)
        assert booster is not None, "loaded MLflow model must expose a LightGBM booster"
        booster_features = list(booster.feature_name())
        assert booster_features, "booster reported empty feature list"
        assert (
            main_mod.FEATURES == booster_features
        ), "app.main.FEATURES should mirror the booster's feature order after load"

        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["model_source"] == expected_uri
        assert body["num_features"] == len(booster_features)


def test_predict_blocks_against_real_model_returns_valid_response(real_client):
    """POST a realistic request and verify the real model + real parquet data
    produce a well-formed, plausible response. `weather_for` will hit Open-Meteo
    unless the network is down, in which case it falls back to 60F / dry."""
    payload = {
        "lat": 37.7790,
        "lon": -122.4180,
        "radius_meters": 800,
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
        "has_nearby_event": 0,
        "is_holiday": 0,
        "is_school_day": 0,
        "temperature": 60.0,
        "minutes_away": 0,
    }
    resp = real_client.post("/predict/blocks", json=payload)
    assert resp.status_code == 200
    distances = [b["distance_meters"] for b in resp.json()["blocks"]]
    assert distances == sorted(distances)


def test_real_weather_lookup_returns_usable_data():
    """Directly exercise the real `weather_for` call against Open-Meteo.

    Skipped when the Open-Meteo host isn't reachable (e.g. offline CI).
    """
    if not _weather_api_reachable():
        pytest.skip("Open-Meteo API unreachable; skipping live weather test.")

    from datetime import datetime

    import app.main as main_mod

    main_mod = importlib.reload(main_mod)
    temp_f, is_raining = main_mod.weather_for(datetime(2026, 5, 1, 12, 0))
    assert isinstance(temp_f, float)
    assert -20.0 <= temp_f <= 130.0
    assert is_raining in (0, 1)
