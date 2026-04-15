"""
ParkCast SF — EDA Data Scraper
Downloads raw datasets from SF Open Data (Socrata) and Open-Meteo APIs
for exploratory data analysis.

Datasets:
  1. Parking Meter Transactions  — SFMTA (incremental downloads)
  2. Parking Meter Locations     — SFMTA (all)
  3. Street Sweeping Schedule    — DPW (incremental downloads)
  4. Street-Use Permits          — DPW (10k most recent)
  5. Parking Regulations         — SFMTA (for unmetered areas)
  6. Weather (hourly, 90 days)   — Open-Meteo

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
        "https://data.sfgov.org/resource/imvp-dq3v.csv?$limit=1000000&$order=session_start_dt%20DESC",
        "SFMTA Parking Meter Transactions",
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
        "Street-Use Permits",
    ),
    "parking_regulations.csv": (
        "https://data.sfgov.org/resource/hi6h-neyh.csv?$limit=50000",
        "SF Parking Regulations (RPP/Color Curbs)",
    ),
    "street_closures.csv": (
        "https://data.sfgov.org/resource/8x25-yybr.csv?$limit=10000",
        "SF Temporary Street Closures (Events Proxy)",
    ),
    # Citations give us 24/7 parking demand signal — meters are free at night,
    # so paid-meter transactions show false zeros overnight. Citations (esp.
    # street-cleaning tickets) are issued around the clock and prove a car
    # was occupying that spot. Window is narrowed to the transaction range to
    # skip the ~year of spurious future-dated rows in the raw dataset.
    "parking_citations.csv": (
        "https://data.sfgov.org/resource/ab4h-6ztd.csv?"
        "$where=citation_issued_datetime%20between%20"
        "'2025-04-13T00:00:00'%20and%20'2026-04-13T00:00:00'"
        "&$limit=3000000",
        "SFMTA Parking Citations (24/7 demand proxy, 12 months)",
    ),
}

# Open-Meteo archive API — 12 months of hourly weather (San Francisco)
WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive?"
    "latitude=37.7749&longitude=-122.4194"
    "&start_date=2025-04-13&end_date=2026-04-13"
    "&hourly=temperature_2m,relative_humidity_2m,precipitation,rain,weathercode,windspeed_10m,cloudcover"
    "&timezone=America/Los_Angeles"
)


def download_dataset(url, dest_path, description):
    """Download a dataset. If it exists, append and de-duplicate."""
    print(f"Syncing {description}...")
    temp_path = dest_path + ".tmp"
    try:
        # Download new data
        urllib.request.urlretrieve(url, temp_path)
        df_new = pd.read_csv(temp_path)

        if os.path.exists(dest_path):
            # Load old data and merge
            df_old = pd.read_csv(dest_path)
            df_combined = pd.concat([df_old, df_new]).drop_duplicates().reset_index(drop=True)
            df_combined.to_csv(dest_path, index=False)
            print(f"  → Merged {len(df_new)} new with {len(df_old)} existing records")
        else:
            # First download
            df_new.to_csv(dest_path, index=False)
            print(f"  → Initial download: {len(df_new)} records")

        if os.path.exists(temp_path):
            os.remove(temp_path)

        size_mb = os.path.getsize(dest_path) / 1e6
        print(f"  → Total file size: {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  ERROR syncing {description}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
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
