"""Application-level (end-to-end) tests for the ParkCast SF API.

A real `uvicorn` server is launched in a background thread on a free localhost
port, and each test hits the live HTTP endpoint through `httpx`. This is the
closest we can get to a production smoke test without actually deploying:
real ASGI server, real HTTP socket, real lifespan, real model, real parquet.

The whole module is skipped when the trained model isn't available on disk.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import socket
import sys
import threading
import time

import httpx
import pytest
import uvicorn

_repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.main import MODEL_PATH  # noqa: E402

_MODEL_AVAILABLE = bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip()) or os.path.exists(
    MODEL_PATH
)

pytestmark = [
    pytest.mark.application,
    pytest.mark.skipif(
        not _MODEL_AVAILABLE,
        reason=(
            "No model source available. Set MLFLOW_TRACKING_URI to load the "
            f"champion from the registry, or place a legacy model at {MODEL_PATH}."
        ),
    ),
]


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    """Spin up `uvicorn` in a background thread and yield its base URL."""
    port = _free_port()
    config = uvicorn.Config(
        "app.main:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30.0
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=1.0)
            if r.status_code in (200, 503):
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.fail("uvicorn server did not become reachable within 30s")

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_server_root_returns_welcome(live_server):
    resp = httpx.get(f"{live_server}/", timeout=5.0)
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"].startswith("Welcome to ParkCast SF API")
    assert "predict_blocks" in body["endpoints"]


def test_server_health_is_healthy(live_server):
    resp = httpx.get(f"{live_server}/health", timeout=5.0)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_server_predict_blocks_roundtrip(live_server):
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
    resp = httpx.post(f"{live_server}/predict/blocks", json=payload, timeout=15.0)
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_blocks_found"] >= 1
    first = body["blocks"][0]
    assert 0.0 <= first["predicted_occupancy_pct"] <= 100.0
    assert first["demand_level"] in {"Low", "Medium", "High", "Very High"}


def test_server_rejects_malformed_predict_request(live_server):
    resp = httpx.post(f"{live_server}/predict/blocks", json={"lat": "nope"}, timeout=5.0)
    assert resp.status_code == 422


def test_server_cors_headers_present(live_server):
    resp = httpx.options(
        f"{live_server}/predict/blocks",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
        timeout=5.0,
    )
    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "*"
