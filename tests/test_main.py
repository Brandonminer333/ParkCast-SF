import importlib
import pathlib
import sys
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="function")
def main_module(monkeypatch):
    """Reload app.main with patched model loading to avoid external dependencies."""

    # Ensure repository root is on sys.path so `import app.main` works when pytest
    # is run from within tests/.
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import mlflow.sklearn
    import joblib

    class DummyModel:
        def predict(self, X):
            return [50.0]

    # Prevent real registry/disk access during import-time load_model()
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda *_, **__: DummyModel())
    monkeypatch.setattr(joblib, "load", lambda *_, **__: DummyModel())

    main_mod = importlib.reload(importlib.import_module("app.main"))
    return main_mod


@pytest.fixture()
def client(main_module):
    return TestClient(main_module.app)


def test_root_returns_welcome(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"].startswith("Welcome to ParkCast SF API")
    assert "health" in body["endpoints"]


def test_health_returns_service_unavailable_when_model_missing(client, main_module):
    main_module.model = None
    resp = client.get("/health")
    assert resp.status_code == 503
    assert "Model not loaded" in resp.json()["detail"]


def test_health_returns_ok_with_model_loaded(client, main_module):
    class DummyModel:
        pass

    main_module.model = DummyModel()
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["model_loaded"] is True
    assert payload["num_features"] == len(main_module.FEATURE_ORDER)
    assert payload["model_type"] == "DummyModel"


def test_prepare_features_sets_weekend_rush_and_street_cleaning(main_module):
    input_obj = main_module.ParkingInput(
        hour=8,
        day_of_week=6,  # Sunday
        month=5,
        neighborhood="mission",
        total_spaces=20,
        is_raining=1,
        has_nearby_event=0,
        is_holiday=0,
        is_school_day=0,
        temperature=55.0,
    )
    features = main_module.prepare_features(input_obj)
    assert features.shape == (1, len(main_module.FEATURE_ORDER))
    # weekend, rush hour (8am), and street cleaning (8-12) all flagged
    assert features[0, 3] == 1  # is_weekend
    assert features[0, 4] == 1  # is_rush_hour
    assert features[0, 5] == 1  # is_street_cleaning


def test_prepare_features_handles_unknown_neighborhood(main_module):
    input_obj = main_module.ParkingInput(
        hour=12,
        day_of_week=2,
        month=1,
        neighborhood="unknown-neighborhood",
        total_spaces=10,
        is_raining=0,
        has_nearby_event=0,
        is_holiday=0,
        is_school_day=1,
        temperature=60.0,
    )
    features = main_module.prepare_features(input_obj)
    encoded_value = features[0, -1]
    assert encoded_value == main_module.NEIGHBORHOOD_MAP["unknown"]


@pytest.mark.parametrize(
    "occupancy, expected",
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
def test_get_demand_level_boundaries(main_module, occupancy, expected):
    assert main_module.get_demand_level(occupancy) == expected


@pytest.mark.parametrize(
    "occupancy, snippet",
    [
        (10, "plenty of spaces"),
        (50, "Good chance"),
        (75, "Limited spots"),
        (95, "Very hard to park"),
    ],
)
def test_get_recommendation_text(main_module, occupancy, snippet):
    assert snippet in main_module.get_recommendation(occupancy)


def test_predict_returns_enriched_response(client, main_module):
    class PredictModel:
        def predict(self, X):
            # include negative value to ensure clamping to 0-100 is not needed here
            return [75.0]

    main_module.model = PredictModel()
    payload = {
        "hour": 18,
        "day_of_week": 4,
        "month": 9,
        "neighborhood": "mission",
        "total_spaces": 40,
        "is_raining": 0,
        "has_nearby_event": 1,
        "is_holiday": 0,
        "is_school_day": 1,
        "temperature": 65.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_occupancy_pct"] == 75.0
    assert body["available_spaces_estimate"] == 10  # 40 * (1 - 0.75)
    assert body["demand_level"] == "High"
    assert "recommendation" in body


def test_predict_returns_service_unavailable_when_model_missing(client, main_module):
    main_module.model = None
    payload = {
        "hour": 10,
        "day_of_week": 1,
        "month": 1,
        "neighborhood": "mission",
        "total_spaces": 20,
        "is_raining": 0,
        "has_nearby_event": 0,
        "is_holiday": 0,
        "is_school_day": 1,
        "temperature": 60.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 503
    assert "Model not loaded" in resp.json()["detail"]
