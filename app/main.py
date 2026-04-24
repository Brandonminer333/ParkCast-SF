"""
ParkCast SF — FastAPI Service
Serves parking occupancy predictions from the best trained model.

Endpoints:
  GET  /              → welcome message
  GET  /health        → model health check
  POST /predict       → predict single neighborhood occupancy
  POST /predict/blocks → predict block-by-block around a lat/lng location
"""

import os
import joblib
import numpy as np
import math
import requests
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List

# ── Setup logging ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
import mlflow.sklearn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://34.133.160.231:5000")
MODEL_NAME = "parkcast-occupancy-model"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None


def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
        logger.info(f"✅ Model loaded from MLflow registry: {MODEL_NAME}@champion")
    except Exception as e:
        logger.error(f"❌ MLflow load failed: {e}")
        try:
            model = joblib.load("models/RandomForest.pkl")
            logger.info("✅ Loaded fallback local model")
        except Exception as e2:
            logger.error(f"❌ Fallback failed: {e2}")
            model = None


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="ParkCast SF API",
    description="Predicts parking availability in San Francisco 30-60 minutes ahead.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_model()

# ── Feature order must match training ─────────────────────────
FEATURE_ORDER = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    "is_street_cleaning",
    "has_nearby_event",
    "is_holiday",
    "is_school_day",
    "is_raining",
    "bad_weather",
    "temperature",
    "total_spaces",
    "neighborhood_encoded",
]

# Neighborhood encoding
NEIGHBORHOOD_MAP = {
    "castro": 0,
    "haight": 1,
    "marina": 2,
    "mission": 3,
    "noe valley": 4,
    "richmond": 5,
    "soma": 6,
    "sunset": 7,
    "tenderloin": 8,
    "unknown": 9,
}

# ── SF Parking blocks database ────────────────────────────────
# Real SFpark block data: block_id, street, lat, lon, total_spaces, neighborhood
# In production this would come from the SFpark API or a database
SF_PARKING_BLOCKS = [
    # ── SoMa ──────────────────────────────────────────────────
    {
        "block_id": "soma_001",
        "street": "Folsom St (3rd-4th)",
        "lat": 37.7816,
        "lon": -122.3975,
        "total_spaces": 42,
        "neighborhood": "soma",
    },
    {
        "block_id": "soma_002",
        "street": "Howard St (3rd-4th)",
        "lat": 37.7820,
        "lon": -122.3968,
        "total_spaces": 38,
        "neighborhood": "soma",
    },
    {
        "block_id": "soma_003",
        "street": "Brannan St (4th-5th)",
        "lat": 37.7792,
        "lon": -122.3952,
        "total_spaces": 35,
        "neighborhood": "soma",
    },
    {
        "block_id": "soma_004",
        "street": "Folsom St (4th-5th)",
        "lat": 37.7814,
        "lon": -122.3948,
        "total_spaces": 44,
        "neighborhood": "soma",
    },
    {
        "block_id": "soma_005",
        "street": "Howard St (4th-5th)",
        "lat": 37.7818,
        "lon": -122.3942,
        "total_spaces": 40,
        "neighborhood": "soma",
    },
    {
        "block_id": "soma_006",
        "street": "Minna St (4th-5th)",
        "lat": 37.7822,
        "lon": -122.3938,
        "total_spaces": 28,
        "neighborhood": "soma",
    },
    {
        "block_id": "soma_007",
        "street": "Natoma St (3rd-4th)",
        "lat": 37.7826,
        "lon": -122.3972,
        "total_spaces": 25,
        "neighborhood": "soma",
    },
    {
        "block_id": "soma_008",
        "street": "Clementina St (3rd-4th)",
        "lat": 37.7830,
        "lon": -122.3980,
        "total_spaces": 22,
        "neighborhood": "soma",
    },
    {
        "block_id": "soma_009",
        "street": "Harrison St (3rd-4th)",
        "lat": 37.7808,
        "lon": -122.3972,
        "total_spaces": 36,
        "neighborhood": "soma",
    },
    {
        "block_id": "soma_010",
        "street": "Bryant St (4th-5th)",
        "lat": 37.7800,
        "lon": -122.3955,
        "total_spaces": 40,
        "neighborhood": "soma",
    },
    {
        "block_id": "soma_011",
        "street": "Folsom St (5th-6th)",
        "lat": 37.7812,
        "lon": -122.4010,
        "total_spaces": 38,
        "neighborhood": "soma",
    },
    {
        "block_id": "soma_012",
        "street": "Howard St (5th-6th)",
        "lat": 37.7816,
        "lon": -122.4018,
        "total_spaces": 36,
        "neighborhood": "soma",
    },
    {
        "block_id": "soma_013",
        "street": "Tehama St (5th-6th)",
        "lat": 37.7820,
        "lon": -122.4022,
        "total_spaces": 24,
        "neighborhood": "soma",
    },
    {
        "block_id": "soma_014",
        "street": "Folsom St (6th-7th)",
        "lat": 37.7810,
        "lon": -122.4038,
        "total_spaces": 40,
        "neighborhood": "soma",
    },
    {
        "block_id": "soma_015",
        "street": "Howard St (6th-7th)",
        "lat": 37.7814,
        "lon": -122.4042,
        "total_spaces": 38,
        "neighborhood": "soma",
    },
    # ── Mission ───────────────────────────────────────────────
    {
        "block_id": "miss_001",
        "street": "Valencia St (16th-17th)",
        "lat": 37.7645,
        "lon": -122.4211,
        "total_spaces": 48,
        "neighborhood": "mission",
    },
    {
        "block_id": "miss_002",
        "street": "Valencia St (17th-18th)",
        "lat": 37.7634,
        "lon": -122.4213,
        "total_spaces": 46,
        "neighborhood": "mission",
    },
    {
        "block_id": "miss_003",
        "street": "Mission St (16th-17th)",
        "lat": 37.7648,
        "lon": -122.4192,
        "total_spaces": 52,
        "neighborhood": "mission",
    },
    {
        "block_id": "miss_004",
        "street": "Mission St (17th-18th)",
        "lat": 37.7637,
        "lon": -122.4195,
        "total_spaces": 50,
        "neighborhood": "mission",
    },
    {
        "block_id": "miss_005",
        "street": "Guerrero St (16th-17th)",
        "lat": 37.7646,
        "lon": -122.4228,
        "total_spaces": 36,
        "neighborhood": "mission",
    },
    {
        "block_id": "miss_006",
        "street": "Dolores St (16th-17th)",
        "lat": 37.7644,
        "lon": -122.4245,
        "total_spaces": 32,
        "neighborhood": "mission",
    },
    {
        "block_id": "miss_007",
        "street": "18th St (Valencia-Guerrero)",
        "lat": 37.7620,
        "lon": -122.4222,
        "total_spaces": 38,
        "neighborhood": "mission",
    },
    {
        "block_id": "miss_008",
        "street": "24th St (Mission-Valencia)",
        "lat": 37.7525,
        "lon": -122.4188,
        "total_spaces": 44,
        "neighborhood": "mission",
    },
    {
        "block_id": "miss_009",
        "street": "Valencia St (18th-19th)",
        "lat": 37.7623,
        "lon": -122.4215,
        "total_spaces": 44,
        "neighborhood": "mission",
    },
    {
        "block_id": "miss_010",
        "street": "Valencia St (19th-20th)",
        "lat": 37.7612,
        "lon": -122.4217,
        "total_spaces": 42,
        "neighborhood": "mission",
    },
    {
        "block_id": "miss_011",
        "street": "Mission St (24th-25th)",
        "lat": 37.7524,
        "lon": -122.4182,
        "total_spaces": 48,
        "neighborhood": "mission",
    },
    {
        "block_id": "miss_012",
        "street": "24th St (Valencia-Guerrero)",
        "lat": 37.7524,
        "lon": -122.4210,
        "total_spaces": 40,
        "neighborhood": "mission",
    },
    {
        "block_id": "miss_013",
        "street": "20th St (Mission-Valencia)",
        "lat": 37.7592,
        "lon": -122.4188,
        "total_spaces": 36,
        "neighborhood": "mission",
    },
    {
        "block_id": "miss_014",
        "street": "22nd St (Mission-Valencia)",
        "lat": 37.7558,
        "lon": -122.4188,
        "total_spaces": 38,
        "neighborhood": "mission",
    },
    # ── Castro ────────────────────────────────────────────────
    {
        "block_id": "cast_001",
        "street": "Castro St (18th-19th)",
        "lat": 37.7608,
        "lon": -122.4350,
        "total_spaces": 30,
        "neighborhood": "castro",
    },
    {
        "block_id": "cast_002",
        "street": "Market St (Castro-Noe)",
        "lat": 37.7614,
        "lon": -122.4339,
        "total_spaces": 35,
        "neighborhood": "castro",
    },
    {
        "block_id": "cast_003",
        "street": "18th St (Castro-Collingwood)",
        "lat": 37.7601,
        "lon": -122.4355,
        "total_spaces": 28,
        "neighborhood": "castro",
    },
    {
        "block_id": "cast_004",
        "street": "19th St (Castro-Collingwood)",
        "lat": 37.7589,
        "lon": -122.4356,
        "total_spaces": 26,
        "neighborhood": "castro",
    },
    {
        "block_id": "cast_005",
        "street": "Castro St (17th-18th)",
        "lat": 37.7620,
        "lon": -122.4348,
        "total_spaces": 32,
        "neighborhood": "castro",
    },
    {
        "block_id": "cast_006",
        "street": "17th St (Castro-Noe)",
        "lat": 37.7622,
        "lon": -122.4338,
        "total_spaces": 30,
        "neighborhood": "castro",
    },
    {
        "block_id": "cast_007",
        "street": "Noe St (17th-18th)",
        "lat": 37.7612,
        "lon": -122.4328,
        "total_spaces": 28,
        "neighborhood": "castro",
    },
    # ── Marina ────────────────────────────────────────────────
    {
        "block_id": "mari_001",
        "street": "Chestnut St (Fillmore-Steiner)",
        "lat": 37.8004,
        "lon": -122.4370,
        "total_spaces": 40,
        "neighborhood": "marina",
    },
    {
        "block_id": "mari_002",
        "street": "Chestnut St (Steiner-Pierce)",
        "lat": 37.8003,
        "lon": -122.4390,
        "total_spaces": 38,
        "neighborhood": "marina",
    },
    {
        "block_id": "mari_003",
        "street": "Union St (Fillmore-Steiner)",
        "lat": 37.7984,
        "lon": -122.4368,
        "total_spaces": 42,
        "neighborhood": "marina",
    },
    {
        "block_id": "mari_004",
        "street": "Lombard St (Fillmore-Steiner)",
        "lat": 37.7997,
        "lon": -122.4372,
        "total_spaces": 36,
        "neighborhood": "marina",
    },
    {
        "block_id": "mari_005",
        "street": "Chestnut St (Pierce-Scott)",
        "lat": 37.8002,
        "lon": -122.4410,
        "total_spaces": 36,
        "neighborhood": "marina",
    },
    {
        "block_id": "mari_006",
        "street": "Union St (Steiner-Pierce)",
        "lat": 37.7983,
        "lon": -122.4388,
        "total_spaces": 40,
        "neighborhood": "marina",
    },
    {
        "block_id": "mari_007",
        "street": "Fillmore St (Chestnut-Lombard)",
        "lat": 37.7993,
        "lon": -122.4360,
        "total_spaces": 34,
        "neighborhood": "marina",
    },
    # ── Haight ────────────────────────────────────────────────
    {
        "block_id": "haig_001",
        "street": "Haight St (Masonic-Ashbury)",
        "lat": 37.7694,
        "lon": -122.4462,
        "total_spaces": 34,
        "neighborhood": "haight",
    },
    {
        "block_id": "haig_002",
        "street": "Haight St (Ashbury-Clayton)",
        "lat": 37.7693,
        "lon": -122.4482,
        "total_spaces": 32,
        "neighborhood": "haight",
    },
    {
        "block_id": "haig_003",
        "street": "Haight St (Clayton-Cole)",
        "lat": 37.7692,
        "lon": -122.4500,
        "total_spaces": 30,
        "neighborhood": "haight",
    },
    {
        "block_id": "haig_004",
        "street": "Haight St (Cole-Shrader)",
        "lat": 37.7691,
        "lon": -122.4518,
        "total_spaces": 28,
        "neighborhood": "haight",
    },
    {
        "block_id": "haig_005",
        "street": "Masonic Ave (Haight-Page)",
        "lat": 37.7706,
        "lon": -122.4455,
        "total_spaces": 30,
        "neighborhood": "haight",
    },
    {
        "block_id": "haig_006",
        "street": "Page St (Masonic-Ashbury)",
        "lat": 37.7706,
        "lon": -122.4462,
        "total_spaces": 28,
        "neighborhood": "haight",
    },
    # ── Richmond ──────────────────────────────────────────────
    {
        "block_id": "rich_001",
        "street": "Clement St (2nd-3rd Ave)",
        "lat": 37.7830,
        "lon": -122.4638,
        "total_spaces": 44,
        "neighborhood": "richmond",
    },
    {
        "block_id": "rich_002",
        "street": "Clement St (3rd-4th Ave)",
        "lat": 37.7830,
        "lon": -122.4658,
        "total_spaces": 42,
        "neighborhood": "richmond",
    },
    {
        "block_id": "rich_003",
        "street": "Geary Blvd (3rd-4th Ave)",
        "lat": 37.7806,
        "lon": -122.4652,
        "total_spaces": 50,
        "neighborhood": "richmond",
    },
    {
        "block_id": "rich_004",
        "street": "Clement St (4th-5th Ave)",
        "lat": 37.7830,
        "lon": -122.4678,
        "total_spaces": 40,
        "neighborhood": "richmond",
    },
    {
        "block_id": "rich_005",
        "street": "Clement St (5th-6th Ave)",
        "lat": 37.7830,
        "lon": -122.4698,
        "total_spaces": 38,
        "neighborhood": "richmond",
    },
    {
        "block_id": "rich_006",
        "street": "Geary Blvd (4th-5th Ave)",
        "lat": 37.7806,
        "lon": -122.4672,
        "total_spaces": 48,
        "neighborhood": "richmond",
    },
    {
        "block_id": "rich_007",
        "street": "Balboa St (3rd-4th Ave)",
        "lat": 37.7760,
        "lon": -122.4658,
        "total_spaces": 36,
        "neighborhood": "richmond",
    },
    # ── Tenderloin ────────────────────────────────────────────
    {
        "block_id": "tend_001",
        "street": "Turk St (Hyde-Leavenworth)",
        "lat": 37.7836,
        "lon": -122.4148,
        "total_spaces": 30,
        "neighborhood": "tenderloin",
    },
    {
        "block_id": "tend_002",
        "street": "Ellis St (Hyde-Leavenworth)",
        "lat": 37.7845,
        "lon": -122.4148,
        "total_spaces": 28,
        "neighborhood": "tenderloin",
    },
    {
        "block_id": "tend_003",
        "street": "O'Farrell St (Hyde-Leavenworth)",
        "lat": 37.7856,
        "lon": -122.4148,
        "total_spaces": 32,
        "neighborhood": "tenderloin",
    },
    {
        "block_id": "tend_004",
        "street": "Eddy St (Hyde-Leavenworth)",
        "lat": 37.7827,
        "lon": -122.4148,
        "total_spaces": 26,
        "neighborhood": "tenderloin",
    },
    {
        "block_id": "tend_005",
        "street": "Jones St (Turk-Ellis)",
        "lat": 37.7840,
        "lon": -122.4134,
        "total_spaces": 24,
        "neighborhood": "tenderloin",
    },
    # ── Downtown / Union Square ───────────────────────────────
    {
        "block_id": "down_001",
        "street": "Post St (Powell-Stockton)",
        "lat": 37.7882,
        "lon": -122.4075,
        "total_spaces": 30,
        "neighborhood": "tenderloin",
    },
    {
        "block_id": "down_002",
        "street": "Geary St (Powell-Stockton)",
        "lat": 37.7874,
        "lon": -122.4075,
        "total_spaces": 32,
        "neighborhood": "tenderloin",
    },
    {
        "block_id": "down_003",
        "street": "Sutter St (Powell-Stockton)",
        "lat": 37.7890,
        "lon": -122.4075,
        "total_spaces": 28,
        "neighborhood": "tenderloin",
    },
    {
        "block_id": "down_004",
        "street": "Post St (Stockton-Grant)",
        "lat": 37.7882,
        "lon": -122.4060,
        "total_spaces": 26,
        "neighborhood": "tenderloin",
    },
    {
        "block_id": "down_005",
        "street": "O'Farrell St (Powell-Stockton)",
        "lat": 37.7865,
        "lon": -122.4075,
        "total_spaces": 30,
        "neighborhood": "tenderloin",
    },
    # ── Noe Valley ────────────────────────────────────────────
    {
        "block_id": "noev_001",
        "street": "24th St (Noe-Sanchez)",
        "lat": 37.7502,
        "lon": -122.4298,
        "total_spaces": 36,
        "neighborhood": "noe valley",
    },
    {
        "block_id": "noev_002",
        "street": "24th St (Sanchez-Church)",
        "lat": 37.7502,
        "lon": -122.4318,
        "total_spaces": 34,
        "neighborhood": "noe valley",
    },
    {
        "block_id": "noev_003",
        "street": "Church St (24th-25th)",
        "lat": 37.7492,
        "lon": -122.4285,
        "total_spaces": 32,
        "neighborhood": "noe valley",
    },
    {
        "block_id": "noev_004",
        "street": "Noe St (24th-25th)",
        "lat": 37.7492,
        "lon": -122.4308,
        "total_spaces": 28,
        "neighborhood": "noe valley",
    },
    {
        "block_id": "noev_005",
        "street": "24th St (Church-Sanchez)",
        "lat": 37.7502,
        "lon": -122.4270,
        "total_spaces": 38,
        "neighborhood": "noe valley",
    },
    # ── Sunset ────────────────────────────────────────────────
    {
        "block_id": "suns_001",
        "street": "Irving St (7th-8th Ave)",
        "lat": 37.7644,
        "lon": -122.4637,
        "total_spaces": 40,
        "neighborhood": "sunset",
    },
    {
        "block_id": "suns_002",
        "street": "Irving St (8th-9th Ave)",
        "lat": 37.7644,
        "lon": -122.4657,
        "total_spaces": 38,
        "neighborhood": "sunset",
    },
    {
        "block_id": "suns_003",
        "street": "Irving St (9th-10th Ave)",
        "lat": 37.7644,
        "lon": -122.4677,
        "total_spaces": 36,
        "neighborhood": "sunset",
    },
    {
        "block_id": "suns_004",
        "street": "Judah St (7th-8th Ave)",
        "lat": 37.7624,
        "lon": -122.4637,
        "total_spaces": 34,
        "neighborhood": "sunset",
    },
    {
        "block_id": "suns_005",
        "street": "Noriega St (7th-8th Ave)",
        "lat": 37.7539,
        "lon": -122.4637,
        "total_spaces": 38,
        "neighborhood": "sunset",
    },
    # ── North Beach / Fishermans Wharf ────────────────────────
    {
        "block_id": "nobe_001",
        "street": "Columbus Ave (Broadway-Vallejo)",
        "lat": 37.7990,
        "lon": -122.4070,
        "total_spaces": 30,
        "neighborhood": "unknown",
    },
    {
        "block_id": "nobe_002",
        "street": "Green St (Columbus-Grant)",
        "lat": 37.7985,
        "lon": -122.4055,
        "total_spaces": 28,
        "neighborhood": "unknown",
    },
    {
        "block_id": "nobe_003",
        "street": "Vallejo St (Columbus-Grant)",
        "lat": 37.7993,
        "lon": -122.4055,
        "total_spaces": 26,
        "neighborhood": "unknown",
    },
    {
        "block_id": "nobe_004",
        "street": "Columbus Ave (Vallejo-Green)",
        "lat": 37.7985,
        "lon": -122.4068,
        "total_spaces": 32,
        "neighborhood": "unknown",
    },
    # ── Civic Center / Hayes Valley ───────────────────────────
    {
        "block_id": "civi_001",
        "street": "Hayes St (Octavia-Laguna)",
        "lat": 37.7764,
        "lon": -122.4238,
        "total_spaces": 34,
        "neighborhood": "unknown",
    },
    {
        "block_id": "civi_002",
        "street": "Hayes St (Laguna-Buchanan)",
        "lat": 37.7764,
        "lon": -122.4258,
        "total_spaces": 32,
        "neighborhood": "unknown",
    },
    {
        "block_id": "civi_003",
        "street": "Gough St (Hayes-Fell)",
        "lat": 37.7758,
        "lon": -122.4228,
        "total_spaces": 30,
        "neighborhood": "unknown",
    },
    {
        "block_id": "civi_004",
        "street": "Fell St (Octavia-Laguna)",
        "lat": 37.7752,
        "lon": -122.4238,
        "total_spaces": 28,
        "neighborhood": "unknown",
    },
    {
        "block_id": "civi_005",
        "street": "Van Ness Ave (Hayes-Grove)",
        "lat": 37.7766,
        "lon": -122.4195,
        "total_spaces": 36,
        "neighborhood": "unknown",
    },
    {
        "block_id": "civi_006",
        "street": "Grove St (Van Ness-Polk)",
        "lat": 37.7772,
        "lon": -122.4188,
        "total_spaces": 30,
        "neighborhood": "unknown",
    },
    # ── Potrero Hill ──────────────────────────────────────────
    {
        "block_id": "potr_001",
        "street": "18th St (Connecticut-Missouri)",
        "lat": 37.7622,
        "lon": -122.4038,
        "total_spaces": 32,
        "neighborhood": "soma",
    },
    {
        "block_id": "potr_002",
        "street": "20th St (Connecticut-Missouri)",
        "lat": 37.7592,
        "lon": -122.4038,
        "total_spaces": 30,
        "neighborhood": "soma",
    },
    {
        "block_id": "potr_003",
        "street": "Connecticut St (18th-20th)",
        "lat": 37.7607,
        "lon": -122.4045,
        "total_spaces": 28,
        "neighborhood": "soma",
    },
    # ── Japantown / Western Addition ──────────────────────────
    {
        "block_id": "japa_001",
        "street": "Post St (Buchanan-Webster)",
        "lat": 37.7852,
        "lon": -122.4308,
        "total_spaces": 36,
        "neighborhood": "unknown",
    },
    {
        "block_id": "japa_002",
        "street": "Buchanan St (Post-Sutter)",
        "lat": 37.7858,
        "lon": -122.4318,
        "total_spaces": 32,
        "neighborhood": "unknown",
    },
    {
        "block_id": "japa_003",
        "street": "Webster St (Post-Sutter)",
        "lat": 37.7858,
        "lon": -122.4298,
        "total_spaces": 30,
        "neighborhood": "unknown",
    },
    {
        "block_id": "japa_004",
        "street": "Fillmore St (Post-Sutter)",
        "lat": 37.7858,
        "lon": -122.4330,
        "total_spaces": 34,
        "neighborhood": "unknown",
    },
    # ── Bernal Heights ────────────────────────────────────────
    {
        "block_id": "bern_001",
        "street": "Cortland Ave (Bennington-Moultrie)",
        "lat": 37.7396,
        "lon": -122.4168,
        "total_spaces": 30,
        "neighborhood": "unknown",
    },
    {
        "block_id": "bern_002",
        "street": "Cortland Ave (Moultrie-Folsom)",
        "lat": 37.7396,
        "lon": -122.4148,
        "total_spaces": 28,
        "neighborhood": "unknown",
    },
    # ── Dogpatch ──────────────────────────────────────────────
    {
        "block_id": "dogp_001",
        "street": "3rd St (22nd-23rd)",
        "lat": 37.7578,
        "lon": -122.3888,
        "total_spaces": 34,
        "neighborhood": "soma",
    },
    {
        "block_id": "dogp_002",
        "street": "Illinois St (22nd-23rd)",
        "lat": 37.7578,
        "lon": -122.3875,
        "total_spaces": 30,
        "neighborhood": "soma",
    },
    {
        "block_id": "dogp_003",
        "street": "Tennessee St (22nd-23rd)",
        "lat": 37.7578,
        "lon": -122.3862,
        "total_spaces": 28,
        "neighborhood": "soma",
    },
    # ── Inner Sunset ──────────────────────────────────────────
    {
        "block_id": "insu_001",
        "street": "9th Ave (Irving-Judah)",
        "lat": 37.7634,
        "lon": -122.4658,
        "total_spaces": 32,
        "neighborhood": "sunset",
    },
    {
        "block_id": "insu_002",
        "street": "9th Ave (Judah-Kirkham)",
        "lat": 37.7614,
        "lon": -122.4658,
        "total_spaces": 30,
        "neighborhood": "sunset",
    },
    {
        "block_id": "insu_003",
        "street": "Lincoln Way (9th-10th Ave)",
        "lat": 37.7668,
        "lon": -122.4660,
        "total_spaces": 28,
        "neighborhood": "sunset",
    },
]


# ── Pydantic models ───────────────────────────────────────────


class ParkingInput(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    neighborhood: str = Field(...)
    total_spaces: int = Field(..., ge=1)
    is_raining: int = Field(0, ge=0, le=1)
    has_nearby_event: int = Field(0, ge=0, le=1)
    is_holiday: int = Field(0, ge=0, le=1)
    is_school_day: int = Field(1, ge=0, le=1)
    temperature: float = Field(60.0)

    class Config:
        json_schema_extra = {
            "example": {
                "hour": 19,
                "day_of_week": 4,
                "month": 4,
                "neighborhood": "mission",
                "total_spaces": 45,
                "is_raining": 0,
                "has_nearby_event": 1,
                "is_holiday": 0,
                "is_school_day": 1,
                "temperature": 62.5,
            }
        }


class BlockPredictionRequest(BaseModel):
    """Request for block-by-block predictions around a location."""

    lat: float = Field(..., description="Destination latitude")
    lon: float = Field(..., description="Destination longitude")
    radius_meters: int = Field(500, ge=100, le=2000, description="Search radius in meters")
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    is_raining: int = Field(0, ge=0, le=1)
    has_nearby_event: int = Field(0, ge=0, le=1)
    is_holiday: int = Field(0, ge=0, le=1)
    is_school_day: int = Field(1, ge=0, le=1)
    temperature: float = Field(60.0)
    minutes_away: int = Field(0, ge=0, le=120, description="Minutes until arrival (for future prediction)")

    class Config:
        json_schema_extra = {
            "example": {
                "lat": 37.7816,
                "lon": -122.3975,
                "radius_meters": 500,
                "hour": 19,
                "day_of_week": 4,
                "month": 4,
                "is_raining": 0,
                "has_nearby_event": 1,
                "is_holiday": 0,
                "is_school_day": 1,
                "temperature": 62.5,
                "minutes_away": 20,
            }
        }


class BlockPrediction(BaseModel):
    block_id: str
    street: str
    lat: float
    lon: float
    total_spaces: int
    neighborhood: str
    distance_meters: int
    predicted_occupancy_pct: float
    available_spaces_estimate: int
    demand_level: str
    color: str


class BlockPredictionResponse(BaseModel):
    destination_lat: float
    destination_lon: float
    radius_meters: int
    predicted_at_hour: int
    minutes_away: int
    total_blocks_found: int
    blocks: List[BlockPrediction]


class ParkingPrediction(BaseModel):
    neighborhood: str
    hour: int
    day_of_week: int
    predicted_occupancy_pct: float
    available_spaces_estimate: int
    demand_level: str
    recommendation: str
    model_version: str = "RandomForest-v1"

    class Config:
        protected_namespaces = ()


# ── Helper functions ──────────────────────────────────────────


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two lat/lon points."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_demand_level(pct):
    if pct < 40:
        return "Low"
    elif pct < 70:
        return "Medium"
    elif pct < 85:
        return "High"
    else:
        return "Very High"


def get_color(pct):
    """Return hex color for map overlay based on occupancy."""
    if pct < 40:
        return "#22c55e"  # green
    elif pct < 70:
        return "#f59e0b"  # amber
    elif pct < 85:
        return "#f97316"  # orange
    else:
        return "#ef4444"  # red


def get_recommendation(pct):
    if pct < 40:
        return "Easy to park — plenty of spaces available."
    elif pct < 70:
        return "Good chance of finding parking — head over."
    elif pct < 85:
        return "Limited spots — arrive early or consider nearby blocks."
    else:
        return "Very hard to park — consider public transit or a garage."


def prepare_features(
    hour,
    day_of_week,
    month,
    total_spaces,
    neighborhood,
    is_raining,
    has_nearby_event,
    is_holiday,
    is_school_day,
    temperature,
):
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
    is_street_cleaning = 1 if 8 <= hour <= 12 else 0
    bad_weather = is_raining
    neighborhood_encoded = NEIGHBORHOOD_MAP.get(neighborhood.lower(), NEIGHBORHOOD_MAP["unknown"])
    return np.array(
        [
            hour,
            day_of_week,
            month,
            is_weekend,
            is_rush_hour,
            is_street_cleaning,
            has_nearby_event,
            is_holiday,
            is_school_day,
            is_raining,
            bad_weather,
            temperature,
            total_spaces,
            neighborhood_encoded,
        ]
    ).reshape(1, -1)


# ── Endpoints ─────────────────────────────────────────────────


@app.get("/")
def root():
    return {
        "message": "Welcome to ParkCast SF API v2",
        "description": "Block-by-block parking prediction for San Francisco",
        "version": "2.0.0",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "predict_blocks": "POST /predict/blocks",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(model).__name__,
        "num_features": len(FEATURE_ORDER),
        "total_blocks_in_db": len(SF_PARKING_BLOCKS),
    }


@app.post("/predict", response_model=ParkingPrediction)
def predict(input: ParkingInput):
    """Predict parking occupancy for a single neighborhood."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        features = prepare_features(
            input.hour,
            input.day_of_week,
            input.month,
            input.total_spaces,
            input.neighborhood,
            input.is_raining,
            input.has_nearby_event,
            input.is_holiday,
            input.is_school_day,
            input.temperature,
        )
        pct = float(model.predict(features)[0])
        pct = round(min(100.0, max(0.0, pct)), 2)
        available = int(input.total_spaces * (1 - pct / 100))
        return ParkingPrediction(
            neighborhood=input.neighborhood,
            hour=input.hour,
            day_of_week=input.day_of_week,
            predicted_occupancy_pct=pct,
            available_spaces_estimate=available,
            demand_level=get_demand_level(pct),
            recommendation=get_recommendation(pct),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/blocks", response_model=BlockPredictionResponse)
def predict_blocks(req: BlockPredictionRequest):
    """
    Predict parking occupancy block-by-block around a destination.
    Returns color-coded predictions for all blocks within radius.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Adjust hour for arrival time (account for minutes_away)
    arrival_hour = (req.hour + req.minutes_away // 60) % 24

    nearby_blocks = []
    for block in SF_PARKING_BLOCKS:
        dist = haversine_distance(req.lat, req.lon, block["lat"], block["lon"])
        if dist <= req.radius_meters:

            try:
                features = prepare_features(
                    arrival_hour,
                    req.day_of_week,
                    req.month,
                    block["total_spaces"],
                    block["neighborhood"],
                    req.is_raining,
                    req.has_nearby_event,
                    req.is_holiday,
                    req.is_school_day,
                    req.temperature,
                )
                pct = float(model.predict(features)[0])
                pct = round(min(100.0, max(0.0, pct)), 2)
                available = int(block["total_spaces"] * (1 - pct / 100))

                nearby_blocks.append(
                    BlockPrediction(
                        block_id=block["block_id"],
                        street=block["street"],
                        lat=block["lat"],
                        lon=block["lon"],
                        total_spaces=block["total_spaces"],
                        neighborhood=block["neighborhood"],
                        distance_meters=int(dist),
                        predicted_occupancy_pct=pct,
                        available_spaces_estimate=available,
                        demand_level=get_demand_level(pct),
                        color=get_color(pct),
                    )
                )
            except Exception:
                continue

    # Sort by distance
    nearby_blocks.sort(key=lambda x: x.distance_meters)

    # ── Fallback: if nothing in radius, return 5 nearest blocks ──
    if not nearby_blocks:
        all_blocks_with_dist = []
        for block in SF_PARKING_BLOCKS:
            dist = haversine_distance(req.lat, req.lon, block["lat"], block["lon"])
            try:
                features = prepare_features(
                    arrival_hour,
                    req.day_of_week,
                    req.month,
                    block["total_spaces"],
                    block["neighborhood"],
                    req.is_raining,
                    req.has_nearby_event,
                    req.is_holiday,
                    req.is_school_day,
                    req.temperature,
                )
                pct = float(model.predict(features)[0])
                pct = round(min(100.0, max(0.0, pct)), 2)
                available = int(block["total_spaces"] * (1 - pct / 100))
                all_blocks_with_dist.append(
                    BlockPrediction(
                        block_id=block["block_id"],
                        street=block["street"],
                        lat=block["lat"],
                        lon=block["lon"],
                        total_spaces=block["total_spaces"],
                        neighborhood=block["neighborhood"],
                        distance_meters=int(dist),
                        predicted_occupancy_pct=pct,
                        available_spaces_estimate=available,
                        demand_level=get_demand_level(pct),
                        color=get_color(pct),
                    )
                )
            except Exception:
                continue
        all_blocks_with_dist.sort(key=lambda x: x.distance_meters)
        nearby_blocks = all_blocks_with_dist[:8]

    return BlockPredictionResponse(
        destination_lat=req.lat,
        destination_lon=req.lon,
        radius_meters=req.radius_meters,
        predicted_at_hour=arrival_hour,
        minutes_away=req.minutes_away,
        total_blocks_found=len(nearby_blocks),
        blocks=nearby_blocks,
    )


@app.get("/geocode_proxy")
def geocode_proxy(q: str):
    import requests as req

    try:
        resp = req.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": f"{q} San Francisco CA",
                "format": "json",
                "limit": 6,
                "addressdetails": 1,
            },
            headers={
                "User-Agent": "ParkCastSF/1.0 (university project usfca.edu)",
                "Accept-Language": "en",
                "Referer": "https://parkcast-frontend.vercel.app",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        filtered = [
            d
            for d in data
            if "san francisco" in d.get("display_name", "").lower() or "california" in d.get("display_name", "").lower()
        ]
        return filtered if filtered else data
    except Exception as e:
        logger.error(f"Geocode error: {e}")
        return []
