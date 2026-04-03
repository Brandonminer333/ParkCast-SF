# ParkCast SF

> Predict where to park in San Francisco — before you leave.

ParkCast SF is a machine learning web service that predicts parking occupancy 30–60 minutes ahead using historical sensor data, weather, events, and time patterns.

---

## Team

**Group:** Parkcast SF
**Members:** 

**Brandon Miner**
HEAD
**Kyvan Zahiri**


**Kayvan Zahiri**

refs/remotes/origin/main
**Temesghen Kahsay**

---

## Project Structure

```
parcast/
├── app/
│   └── main.py              # FastAPI application
├── models/
│   └── RandomForest.pkl     # Trained model
├── data/
│   └── parkcast_raw.csv     # Ingested dataset
├── data_ingestion.ipynb     # Data ingestion script
├── train_model.ipynb        # Model training script
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Model Performance

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **RandomForest** ← best | **7.88%** | **9.80%** | **0.87** |
| GradientBoosting | 7.93% | 9.86% | 0.87 |
| Ridge | 17.31% | 20.52% | 0.43 |

**Target:** Predict parking occupancy % (0–100%)
**Features:** Hour, day of week, neighborhood, weather, events, holidays, school days

---

## Data Sources

| Source | What it provides |
|--------|-----------------|
| SFpark Sensor API | Real-time + historical occupancy from 12,000+ sensors |
| SF Open Data | Street cleaning schedules, meter locations |
| SF Events API | Giants games, concerts, public events |
| Open-Meteo | Hourly SF weather (rain, temperature) |
| US Federal Holidays | Holiday parking patterns |
| SF School Calendar | School day demand patterns |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Model health check |
| POST | `/predict` | Predict parking occupancy |

---

## How to Run Locally (without Docker)

**1. Clone the repo:**
```bash
<<<<<<< HEAD
git clone https://github.com/YOUR_USERNAME/parcast.git
cd parcast
=======
git clone https://github.com/YOUR_USERNAME/parkcast.git
cd parkcast
>>>>>>> refs/remotes/origin/main
```

**2. Create and activate virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Start the API:**
```bash
python -m uvicorn app.main:app --reload --port 8000
```

**5. Open the interactive docs:**
```
http://localhost:8000/docs
```

---

## How to Run with Docker

**Option 1 — Pull from Docker Hub:**
```bash
docker pull YOUR_DOCKERHUB_USERNAME/parkcast-api:latest
docker run -d -p 8000:8000 YOUR_DOCKERHUB_USERNAME/parkcast-api:latest
```

**Option 2 — Build locally:**
```bash
docker build -t parkcast-api .
docker run -d -p 8000:8000 parkcast-api
```

**Then open:**
```
http://localhost:8000/docs
```

---

## Example /predict Request

**Input:**
```json
{
  "hour": 19,
  "day_of_week": 4,
  "month": 4,
  "neighborhood": "mission",
  "total_spaces": 45,
  "is_raining": 0,
  "has_nearby_event": 1,
  "is_holiday": 0,
  "is_school_day": 1,
  "temperature": 62.5
}
```

**Output:**
```json
{
  "neighborhood": "mission",
  "hour": 19,
  "day_of_week": 4,
  "predicted_occupancy_pct": 85.83,
  "available_spaces_estimate": 6,
  "demand_level": "Very High",
  "recommendation": "Very hard to park — consider public transit or a garage.",
  "model_version": "RandomForest-v1"
}
```

---

## Supported Neighborhoods

| Neighborhood | Input value |
|-------------|-------------|
| Castro | `castro` |
| Haight | `haight` |
| Marina | `marina` |
| Mission | `mission` |
| Noe Valley | `noe valley` |
| Richmond | `richmond` |
| SoMa | `soma` |
| Sunset | `sunset` |
| Tenderloin | `tenderloin` |

---

## MLflow Tracking

All experiments are tracked on a shared GCP MLflow server.

- **Tracking URI:** `http://34.133.160.231:5000`
- **Experiments:** `parkcast-data-ingestion`, `parkcast-training`
- **Model Registry:** `parkcast-occupancy-model`

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| ML Model | scikit-learn RandomForest |
| Experiment Tracking | MLflow on GCP |
| Containerization | Docker |
| Image Registry | Docker Hub |
| Cloud VM | Google Compute Engine |
