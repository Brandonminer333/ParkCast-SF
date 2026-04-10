"""
ParkCast SF — EDA Data Scraper
Downloads raw datasets from SF Open Data (Socrata) and Open-Meteo APIs
for exploratory data analysis.

Datasets:
  1. Parking Meter Transactions  — SFMTA (50k most recent)
  2. Parking Meter Locations     — SFMTA (all)
  3. Street Sweeping Schedule    — DPW (50k)
  4. Street-Use Permits          — DPW (10k most recent)
  5. Weather (hourly, 90 days)   — Open-Meteo

Output:
  All files are saved to the data/ directory as CSVs.

Usage:
  python eda_data_scraper.py
"""

import os
import json
import urllib.request

import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # go up from dev/ to project root
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# DataSF (Socrata) datasets — each entry: (filename, url, description)
DATASETS = {
    "meter_transactions.csv": (
        "https://data.sfgov.org/resource/imvp-dq3v.csv?$limit=50000&$order=session_start_dt%20DESC",
        "SFMTA Parking Meter Transactions (50k most recent)",
    ),
    "meter_locations.csv": (
        "https://data.sfgov.org/resource/8vzz-qzz9.csv?$limit=50000",
        "All Parking Meter Locations",
    ),
    "street_sweeping.csv": (
        "https://data.sfgov.org/resource/yhqp-riqs.csv?$limit=50000",
        "Street Sweeping Schedule",
    ),
    "street_use_permits.csv": (
        "https://data.sfgov.org/resource/b6tj-gt35.csv?$limit=10000&$order=data_as_of%20DESC",
        "Street-Use Permits (10k most recent)",
    ),
}

# Open-Meteo weather URL (last 90 days, hourly, San Francisco)
WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast?"
    "latitude=37.7749&longitude=-122.4194"
    "&hourly=temperature_2m,relative_humidity_2m,precipitation,rain,weathercode,windspeed_10m,cloudcover"
    "&past_days=90&forecast_days=1&timezone=America/Los_Angeles"
)


def download_dataset(url, dest_path, description):
    """Download a single dataset to dest_path (always re-downloads)."""
    print(f"Downloading {description}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        size_mb = os.path.getsize(dest_path) / 1e6
        print(f"  → {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  ERROR downloading {description}: {e}")
        return False


def download_weather():
    """Download weather data from Open-Meteo and convert JSON → CSV."""
    weather_path = os.path.join(DATA_DIR, "weather.csv")
    raw_json_path = os.path.join(DATA_DIR, "weather_raw.json")

    print("Downloading Open-Meteo weather (last 90 days)...")
    try:
        urllib.request.urlretrieve(WEATHER_URL, raw_json_path)
        with open(raw_json_path) as f:
            wdata = json.load(f)
        hourly = wdata["hourly"]
        pd.DataFrame(hourly).to_csv(weather_path, index=False)
        print(f"  → {os.path.getsize(weather_path) / 1e6:.1f} MB")
        return True
    except Exception as e:
        print(f"  ERROR downloading weather: {e}")
        return False


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("ParkCast SF — EDA Data Scraper")
    print(f"Data directory: {os.path.abspath(DATA_DIR)}")
    print("=" * 60)

    success_count = 0
    total_count = len(DATASETS) + 1  # +1 for weather

    # Download all Socrata datasets
    for filename, (url, desc) in DATASETS.items():
        dest_path = os.path.join(DATA_DIR, filename)
        if download_dataset(url, dest_path, desc):
            success_count += 1

    # Download weather
    if download_weather():
        success_count += 1

    # Summary
    print()
    print("=" * 60)
    if success_count == total_count:
        print(f"All {total_count} datasets ready.")
    else:
        print(f"WARNING: {total_count - success_count}/{total_count} datasets failed to download.")
    print("=" * 60)


if __name__ == "__main__":
    main()
