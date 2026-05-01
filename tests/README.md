# `tests/` — ParkCast SF API test suite

The suite is split into four files, one per test category. Every test in every
file is tagged with a marker so you can run them as a group, and each category
has its own scope, dependencies, and runtime profile.

## Layout

| File                       | Marker         | What it covers                                                                                          |
|----------------------------|----------------|---------------------------------------------------------------------------------------------------------|
| `test_unit.py`             | `unit`         | Pure helper functions in `app.main` (`demand_level`, `color_for`, `is_school_day`). No app, no fixtures. |
| `test_functional.py`       | `functional`   | Single endpoints via Starlette's `TestClient`, with a `DummyModel` and tiny in-memory parking DB.       |
| `test_integration.py`      | `integration`  | Real model + real parquet assets + real Open-Meteo weather API, still in-process via `TestClient`.       |
| `test_application.py`      | `application`  | A live `uvicorn` server bound to a free localhost port, exercised over real HTTP with `httpx`.          |
| `conftest.py`              | n/a            | Hosts the shared `main_module` + `client` fixtures used by `test_functional.py`.                         |

## Prerequisites

- Python 3.10 or 3.11 (matches `app/main.py` + the Cloud Run image).
- A model source. Either:
  - `MLFLOW_TRACKING_URI` pointing at the registry that hosts
    `models:/parkcast-occupancy-model@champion` (the production source of
    truth), or
  - the legacy `app/models/LightGBM.pkl` + sibling parquet lookup tables,
    which are committed to the repo so a fresh checkout already has a
    working fallback.

  Integration and application tests auto-skip when neither source is
  reachable.
- Pip-installable packages:

  ```bash
  pip install \
      "fastapi>=0.110" "pydantic>=2.6" "httpx>=0.27" \
      "numpy>=1.26" "pandas>=2.2" "joblib>=1.3" \
      "lightgbm>=4.0" "pyarrow>=15.0" "mlflow>=2.11" \
      "uvicorn[standard]>=0.27" "pytest>=8.0"
  ```

  If you're already in the `parkcast` conda env that the runtime uses, all
  of these are present and you can skip the pip install.

### Per-block inference lookups

`app/main.py` needs three per-block parquet lookups to populate the static +
hour/dow-aggregate features the trained model expects
(`poi_*_200m`, `complaints_311_*`, `sfpark_block_occ`, `permit_*`). They're
checked into the repo at `app/models/block_static.parquet`,
`app/models/complaints_lookup.parquet`, and `app/models/sfpark_lookup.parquet`,
and they're generated from `dev/processed_training_data.csv` by the one-off
script at `dev/build_block_lookups.py`:

```bash
python dev/build_block_lookups.py
```

Re-run it whenever `processed_training_data.csv` is updated so inference
stays aligned with what the next retrain will see.

## Running the suite

From the repo root:

```bash
# Everything (this is what CI runs)
pytest

# Just one category
pytest -m unit
pytest -m functional
pytest -m integration
pytest -m application

# Fast local feedback loop — skip the slow / network-touching layers
pytest -m "unit or functional"

# Everything except the live-server tests
pytest -m "not application"

# A single file
pytest tests/test_functional.py

# A single test
pytest tests/test_functional.py::test_health_ok_reports_block_count
```

`pyproject.toml` registers the four markers under `[tool.pytest.ini_options]`
and enables `--strict-markers`, so any typo in `-m "..."` (or in a `pytestmark`
inside a test file) will fail loudly instead of silently selecting nothing.

## What each category does (and what it costs)

### `unit` — `tests/test_unit.py`
Imports the helper functions directly from `app.main` and asserts on their
return values across parametrized inputs. No FastAPI app is constructed, no
fixtures are loaded, and nothing touches disk or the network. Sub-second.

### `functional` — `tests/test_functional.py`
Uses the `client` and `main_module` fixtures from `conftest.py`. The fixture
reloads `app.main`, monkeypatches `_load_all` into a no-op, swaps in a
`DummyModel` whose `.predict()` returns a constant, and replaces
`weather_for(...)` with a lambda that returns `(62.0, 0)`. Every test then
hits a single endpoint via `TestClient` and asserts on the JSON shape /
specific arithmetic. No real model, no parquet, no network.

### `integration` — `tests/test_integration.py`
Reloads `app.main` and lets FastAPI's lifespan fire `_load_all`. The model is
loaded from MLflow when `MLFLOW_TRACKING_URI` is set (production path) and
otherwise from the legacy local `.pkl` (CI / offline dev fallback). The
real per-block parquet lookups under `app/models/*.parquet` are read from
disk either way. Endpoints are exercised through `TestClient` (in-process
ASGI, no real socket). `weather_for(...)` is **not** monkeypatched, so the
prediction endpoint hits `api.open-meteo.com` over HTTPS for live weather
data. The whole module auto-skips if no model source is available, and the
dedicated weather lookup test is individually skipped if Open-Meteo is
unreachable.

### `application` — `tests/test_application.py`
Starts a real `uvicorn.Server` against `app.main:app` in a background thread
on a randomly chosen free localhost port, polls `/health` until startup
finishes, and then drives the API with `httpx` over a real TCP socket: the
welcome page, the health check, a happy-path `POST /predict/blocks`, a
malformed request that should produce a 422, and a CORS preflight. This is
the closest the suite gets to a production smoke test without actually
deploying. Auto-skips when no model source is available (neither
`MLFLOW_TRACKING_URI` set nor a local legacy model present).

## CI

`.github/workflows/jobs.yml` runs **all** markers — there is no filtering. The
`test` job:

1. Sets up Python 3.11 with pip caching.
2. Installs the dependency list listed above (including `mlflow` so the
   production loading path can be exercised once `MLFLOW_TRACKING_URI` is
   configured as a workflow secret).
3. Runs `pytest -q`.

CI does not currently set `MLFLOW_TRACKING_URI`, so integration and
application tests load the legacy local model committed to `app/models/`.
The model loads via `joblib`, the per-block lookups load from parquet, and
`app.main`'s feature-subset logic (see `_load_all` / `_model_feature_order`)
picks the correct feature list from the loaded model's booster so the
31-feature legacy and the 33-feature champion both work without code
changes. The `lint` job runs `black --check --diff app tests` in parallel
and uses the `[tool.black]` config from `pyproject.toml` (`line-length = 100`).

## Adding new tests

1. Decide which layer the test belongs to:
   - Pure function with no app? → `test_unit.py` (`unit`).
   - Endpoint behaviour with cheap fakes? → `test_functional.py` (`functional`).
   - Real model / real parquet / real third-party HTTP? → `test_integration.py`
     (`integration`).
   - Must verify the actual HTTP server boots and serves correctly? →
     `test_application.py` (`application`).
2. Don't add any new top-level marker without first registering it in
   `pyproject.toml` under `[tool.pytest.ini_options].markers` — the strict
   markers setting will reject anything unknown.
