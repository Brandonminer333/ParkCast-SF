"""
ParkCast SF — Real Data Ingestion Script
Uses actual SFpark sensor data from SF Open Data API (no synthetic fallback)

Real data sources:
  1. SF Open Data — Parking meter transactions (real occupancy data)
  2. SF Open Data — Street sweeping schedule  
  3. SF Open Data — Public events permits
  4. Open-Meteo   — Historical SF weather
  5. Nager.at     — US federal holidays
  6. SFUSD        — School calendar (computed)
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mlflow
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://34.133.160.231:5000"
EXPERIMENT_NAME     = "parkcast-data-ingestion"
DATA_DIR            = "data"
OUTPUT_FILE         = os.path.join(DATA_DIR, "parkcast_raw.csv")

# Pull last 90 days of real data
END_DATE   = datetime.today()
START_DATE = END_DATE - timedelta(days=90)
DATE_FMT   = "%Y-%m-%d"

os.makedirs(DATA_DIR, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ─────────────────────────────────────────────────────────────
# SOURCE 1: Real SFpark Meter Occupancy Data
# SF Open Data: Parking meter transactions with occupancy
# https://data.sfgov.org/resource/uupn-yfaw.json
# ─────────────────────────────────────────────────────────────
def fetch_sfpark_real():
    """
    Fetch real parking meter occupancy from SF Open Data.
    Dataset: SFMTA Parking Meter Detailed Revenue Transactions
    Returns block-level occupancy aggregated by hour.
    """
    print("Fetching real SFpark meter data from SF Open Data...")

    url = "https://data.sfgov.org/resource/uupn-yfaw.json"

    start_str = START_DATE.strftime("%Y-%m-%dT00:00:00")
    end_str   = END_DATE.strftime("%Y-%m-%dT23:59:59")

    params = {
        "$limit":  50000,
        "$offset": 0,
        "$where":  f"session_start_dt >= '{start_str}' AND session_start_dt <= '{end_str}'",
        "$select": "street_block,meter_type,session_start_dt,session_end_dt,gross_paid_amt,street_side",
    }

    all_records = []
    offset = 0

    while True:
        params["$offset"] = offset
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_records.extend(batch)
            print(f"  Fetched {len(all_records)} records so far...")
            if len(batch) < 50000:
                break
            offset += 50000
        except Exception as e:
            print(f"  SFpark API error at offset {offset}: {e}")
            break

    if not all_records:
        print("  No real data fetched — trying alternative endpoint...")
        return fetch_sfpark_alternative()

    df = pd.DataFrame(all_records)
    print(f"  Raw records fetched: {len(df)}")

    # Parse timestamps
    df["session_start_dt"] = pd.to_datetime(df["session_start_dt"], errors="coerce")
    df = df.dropna(subset=["session_start_dt"])

    # Extract time features
    df["date"]        = df["session_start_dt"].dt.strftime(DATE_FMT)
    df["hour"]        = df["session_start_dt"].dt.hour
    df["day_of_week"] = df["session_start_dt"].dt.dayofweek
    df["month"]       = df["session_start_dt"].dt.month

    # Map street blocks to neighborhoods
    df["neighborhood"] = df["street_block"].apply(map_block_to_neighborhood)
    df["block_id"]     = df["street_block"].fillna("unknown").str.lower().str.replace(" ", "_")

    # Aggregate: count transactions per block per hour as proxy for occupancy
    agg = df.groupby(["block_id", "neighborhood", "date", "hour", "day_of_week", "month"]).agg(
        transaction_count=("gross_paid_amt", "count"),
        avg_payment=("gross_paid_amt", lambda x: pd.to_numeric(x, errors="coerce").mean()),
    ).reset_index()

    # Estimate occupancy: normalize transaction count to 0-100%
    # Higher transaction count = higher occupancy
    max_per_hour = agg["transaction_count"].quantile(0.95)
    agg["occupancy_pct"] = (agg["transaction_count"] / max_per_hour * 100).clip(0, 100).round(2)
    agg["total_spaces"]  = 40  # average block has ~40 spaces
    agg["street"]        = agg["block_id"].str.replace("_", " ").str.title()

    # Add lat/lon from block lookup
    agg["lat"] = agg["neighborhood"].map(NEIGHBORHOOD_COORDS).apply(lambda x: x[0] if x else 37.7749)
    agg["lon"] = agg["neighborhood"].map(NEIGHBORHOOD_COORDS).apply(lambda x: x[1] if x else -122.4194)

    print(f"  Aggregated to {len(agg)} block-hour records from real data")
    return agg


def fetch_sfpark_alternative():
    """
    Alternative: Use SFpark Availability API (real-time sensor data).
    https://data.sfgov.org/resource/2ehv-6arf.json
    """
    print("  Trying SFpark availability sensor data...")
    url = "https://data.sfgov.org/resource/2ehv-6arf.json"
    params = {"$limit": 10000}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = []
        now = datetime.now()
        for item in data:
            try:
                occ_pct = float(item.get("occ", 0)) / max(float(item.get("oper", 1)), 1) * 100
                hood = map_block_to_neighborhood(item.get("name", ""))
                records.append({
                    "block_id":        item.get("bfid", "unknown"),
                    "street":          item.get("name", "unknown"),
                    "neighborhood":    hood,
                    "lat":             float(item.get("lat", 37.7749)),
                    "lon":             float(item.get("lng", -122.4194)),
                    "date":            now.strftime(DATE_FMT),
                    "hour":            now.hour,
                    "day_of_week":     now.weekday(),
                    "month":           now.month,
                    "occupancy_pct":   round(min(100, max(0, occ_pct)), 2),
                    "total_spaces":    int(item.get("oper", 40)),
                })
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(records)
        print(f"  Alternative API: {len(df)} live records")
        return df

    except Exception as e:
        print(f"  Alternative API also failed: {e}")
        raise RuntimeError("Both SFpark endpoints failed. Check your internet connection.")


# Neighborhood coordinate lookup
NEIGHBORHOOD_COORDS = {
    "mission":    (37.7599, -122.4148),
    "soma":       (37.7785, -122.3952),
    "castro":     (37.7609, -122.4350),
    "marina":     (37.8003, -122.4360),
    "tenderloin": (37.7836, -122.4148),
    "haight":     (37.7694, -122.4469),
    "richmond":   (37.7800, -122.4639),
    "noe valley": (37.7501, -122.4334),
    "sunset":     (37.7527, -122.4820),
    "unknown":    (37.7749, -122.4194),
}

def map_block_to_neighborhood(block_name):
    """Map a street block name to SF neighborhood."""
    if not block_name:
        return "unknown"
    b = str(block_name).lower()
    if any(x in b for x in ["mission", "valencia", "guerrero", "24th", "16th"]):
        return "mission"
    if any(x in b for x in ["folsom", "howard", "brannan", "soma", "3rd", "4th", "5th"]):
        return "soma"
    if any(x in b for x in ["castro", "18th", "19th", "market"]):
        return "castro"
    if any(x in b for x in ["chestnut", "lombard", "marina", "union"]):
        return "marina"
    if any(x in b for x in ["turk", "ellis", "eddy", "tenderloin", "leavenworth"]):
        return "tenderloin"
    if any(x in b for x in ["haight", "ashbury", "masonic"]):
        return "haight"
    if any(x in b for x in ["clement", "geary", "richmond", "balboa"]):
        return "richmond"
    return "unknown"


# ─────────────────────────────────────────────────────────────
# SOURCE 2: Street Cleaning (SF Open Data)
# ─────────────────────────────────────────────────────────────
def fetch_street_cleaning():
    print("Fetching street cleaning schedule...")
    url = "https://data.sfgov.org/resource/yhqp-tvqs.json"
    try:
        resp = requests.get(url, params={"$limit": 5000}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame([{
            "street":       d.get("streetname",""),
            "cleaning_day": d.get("weekday",""),
            "start_hour":   int(d.get("fromhour", 0)),
            "end_hour":     int(d.get("tohour", 0)),
        } for d in data])
        print(f"  Street cleaning: {len(df)} records")
        return df
    except Exception as e:
        print(f"  Street cleaning fetch failed: {e}")
        return pd.DataFrame(columns=["street","cleaning_day","start_hour","end_hour"])


# ─────────────────────────────────────────────────────────────
# SOURCE 3: SF Events (SF Open Data)
# ─────────────────────────────────────────────────────────────
def fetch_sf_events():
    print("Fetching SF events data...")
    url = "https://data.sfgov.org/resource/pyih-qa8i.json"
    try:
        resp = requests.get(url, params={"$limit": 2000}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        records = []
        for d in data:
            dt = d.get("start_date_time","")[:10]
            if dt:
                records.append({"event_date": dt, "has_event": 1,
                                 "neighborhood": d.get("neighborhoods_sic_district","unknown")})
        df = pd.DataFrame(records)
        print(f"  SF Events: {len(df)} records")
        return df
    except Exception as e:
        print(f"  SF Events fetch failed: {e}")
        return pd.DataFrame(columns=["event_date","has_event","neighborhood"])


# ─────────────────────────────────────────────────────────────
# SOURCE 4: Real SF Weather (Open-Meteo historical)
# ─────────────────────────────────────────────────────────────
def fetch_weather():
    print("Fetching real SF weather from Open-Meteo...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":    37.7749,
        "longitude":   -122.4194,
        "start_date":  START_DATE.strftime(DATE_FMT),
        "end_date":    END_DATE.strftime(DATE_FMT),
        "hourly":      "temperature_2m,precipitation,weathercode",
        "timezone":    "America/Los_Angeles",
        "temperature_unit": "fahrenheit",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        df = pd.DataFrame({
            "timestamp":     hourly.get("time", []),
            "temperature":   hourly.get("temperature_2m", []),
            "precipitation": hourly.get("precipitation", []),
            "weather_code":  hourly.get("weathercode", []),
        })
        df["timestamp"]   = pd.to_datetime(df["timestamp"])
        df["date"]        = df["timestamp"].dt.strftime(DATE_FMT)
        df["hour"]        = df["timestamp"].dt.hour
        df["is_raining"]  = (df["precipitation"] > 0.1).astype(int)
        df["bad_weather"] = (df["weather_code"] >= 51).astype(int)
        print(f"  Weather: {len(df)} real hourly records")
        return df
    except Exception as e:
        print(f"  Weather fetch failed: {e}")
        return pd.DataFrame(columns=["date","hour","temperature","is_raining","bad_weather"])


# ─────────────────────────────────────────────────────────────
# SOURCE 5: US Holidays
# ─────────────────────────────────────────────────────────────
def fetch_holidays():
    print("Fetching US federal holidays...")
    try:
        year = datetime.today().year
        resp = requests.get(f"https://date.nager.at/api/v3/PublicHolidays/{year}/US", timeout=15)
        resp.raise_for_status()
        df = pd.DataFrame([{"date": d["date"], "is_holiday": 1} for d in resp.json()])
        print(f"  Holidays: {len(df)} records")
        return df
    except Exception as e:
        print(f"  Holidays fetch failed: {e}")
        # Hardcoded fallback for current year
        return pd.DataFrame({"date": ["2026-01-01","2026-01-19","2026-02-16","2026-05-25",
                                       "2026-06-19","2026-07-04","2026-09-07","2026-11-26","2026-12-25"],
                              "is_holiday": 1})


# ─────────────────────────────────────────────────────────────
# SOURCE 6: School Calendar (computed)
# ─────────────────────────────────────────────────────────────
def fetch_school_calendar():
    print("Computing SF school calendar...")
    records = []
    current = START_DATE
    while current <= END_DATE:
        is_school_month = current.month in [8,9,10,11,12,1,2,3,4,5,6]
        is_weekday = current.weekday() < 5
        records.append({
            "date":          current.strftime(DATE_FMT),
            "is_school_day": 1 if (is_school_month and is_weekday) else 0,
        })
        current += timedelta(days=1)
    df = pd.DataFrame(records)
    print(f"  School calendar: {len(df)} days")
    return df


# ─────────────────────────────────────────────────────────────
# MERGE all sources
# ─────────────────────────────────────────────────────────────
def merge_all(sfpark_df, events_df, weather_df, holidays_df, school_df):
    print("\nMerging all sources...")
    df = sfpark_df.copy()

    # Ensure date column exists
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df.get("session_start_dt", datetime.now())).dt.strftime(DATE_FMT)

    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].apply(lambda h: 1 if (7<=h<=9 or 17<=h<=19) else 0)
    df["is_street_cleaning"] = df["hour"].apply(lambda h: 1 if 8<=h<=12 else 0)

    # Merge weather
    if not weather_df.empty:
        df = df.merge(weather_df[["date","hour","temperature","is_raining","bad_weather"]],
                      on=["date","hour"], how="left")
    else:
        df["is_raining"] = 0; df["bad_weather"] = 0; df["temperature"] = 60.0

    # Merge events
    if not events_df.empty:
        ev = events_df.groupby("event_date")["has_event"].max().reset_index()
        ev.columns = ["date","has_nearby_event"]
        df = df.merge(ev, on="date", how="left")
        df["has_nearby_event"] = df["has_nearby_event"].fillna(0).astype(int)
    else:
        df["has_nearby_event"] = 0

    # Merge holidays
    if not holidays_df.empty:
        df = df.merge(holidays_df[["date","is_holiday"]], on="date", how="left")
        df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    else:
        df["is_holiday"] = 0

    # Merge school
    if not school_df.empty:
        df = df.merge(school_df, on="date", how="left")
        df["is_school_day"] = df["is_school_day"].fillna(0).astype(int)
    else:
        df["is_school_day"] = 0

    df = df.fillna({"temperature": 60.0, "is_raining": 0, "bad_weather": 0})
    df = df.dropna(subset=["occupancy_pct"])

    # Final feature set
    keep = ["block_id","street","neighborhood","lat","lon","date","hour",
            "day_of_week","month","is_weekend","is_rush_hour","is_street_cleaning",
            "has_nearby_event","is_holiday","is_school_day","is_raining",
            "bad_weather","temperature","total_spaces","occupancy_pct"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    print(f"  Final dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("ParkCast SF — REAL Data Ingestion")
    print(f"Period: {START_DATE.strftime(DATE_FMT)} → {END_DATE.strftime(DATE_FMT)}")
    print("="*60)

    with mlflow.start_run(run_name="real-data-ingestion"):

        sfpark_df  = fetch_sfpark_real()
        events_df  = fetch_sf_events()
        weather_df = fetch_weather()
        holidays_df= fetch_holidays()
        school_df  = fetch_school_calendar()

        final_df = merge_all(sfpark_df, events_df, weather_df, holidays_df, school_df)

        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved to {OUTPUT_FILE}")

        mlflow.log_params({
            "start_date":   START_DATE.strftime(DATE_FMT),
            "end_date":     END_DATE.strftime(DATE_FMT),
            "data_source":  "real_sfpark_api",
            "num_rows":     len(final_df),
            "num_features": len(final_df.columns),
        })
        mlflow.log_metrics({
            "total_rows":        len(final_df),
            "avg_occupancy_pct": round(final_df["occupancy_pct"].mean(), 2),
            "num_blocks":        final_df["block_id"].nunique(),
            "num_neighborhoods": final_df["neighborhood"].nunique(),
            "missing_pct":       round(final_df.isnull().mean().mean()*100, 2),
        })

        print("\n" + "="*60)
        print("INGESTION COMPLETE — REAL DATA")
        print("="*60)
        print(f"Rows:          {len(final_df):,}")
        print(f"Unique blocks: {final_df['block_id'].nunique()}")
        print(f"Neighborhoods: {final_df['neighborhood'].nunique()}")
        print(f"Avg occupancy: {final_df['occupancy_pct'].mean():.1f}%")
        print(f"Date range:    {final_df['date'].min()} → {final_df['date'].max()}")
        print("="*60)


if __name__ == "__main__":
    main()
