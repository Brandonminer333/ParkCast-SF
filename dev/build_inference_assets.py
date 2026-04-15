"""
ParkCast SF — Build Inference Assets

One-time build of the lookup tables the API needs to serve predictions:

  blocks.parquet                 → catalog of every known block (lat, lon,
                                   neighborhood, total_spaces)
  lag_history.parquet            → last 35 days of real-meter-hour occupancy
                                   per block, used to populate lag_7d/14d/28d
                                   at inference time
  citations_hourly_median.parquet → citations_hourly_median indexed by
                                   (hour, day_of_week), fit on training window

The LightGBM block aggregates (block_mean, block_hour_mean, block_hour_dow_mean)
are already saved by train_lightgbm.py in LightGBM.block_aggs.parquet.
"""

import os
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "processed_training_data.csv")
CIT_PATH = os.path.join(DATA_DIR, "citations_by_block.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "app", "models")
BLOCKS_PATH = os.path.join(MODEL_DIR, "blocks.parquet")
LAG_PATH = os.path.join(MODEL_DIR, "lag_history.parquet")
CIT_LOOKUP_PATH = os.path.join(MODEL_DIR, "citations_hourly_median.parquet")

LAG_DAYS = 35  # 28d lag needs 28 days, +1 week safety margin


def main():
    print("=" * 60)
    print("ParkCast SF — Inference Asset Builder")
    print("=" * 60)

    print("Loading processed training data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df[df["target_is_estimated"] == 0].reset_index(drop=True)
    print(f"  Real meter-hour rows: {len(df):,}")

    # 1. Block catalog
    print("Building block catalog...")
    blocks = (df.groupby(["lat", "lon"], as_index=False)
                .agg(neighborhood=("neighborhood", "first"),
                     total_spaces=("total_spaces", "first")))
    os.makedirs(MODEL_DIR, exist_ok=True)
    blocks.to_parquet(BLOCKS_PATH, index=False)
    print(f"  {len(blocks):,} blocks → {BLOCKS_PATH}")

    # 2. Lag history (keep only columns the API needs)
    print(f"Building lag history (last {LAG_DAYS} days)...")
    latest = df["timestamp"].max()
    cutoff = latest - pd.Timedelta(days=LAG_DAYS)
    lag_hist = (df[df["timestamp"] >= cutoff]
                [["lat", "lon", "timestamp", "occupancy_pct"]]
                .reset_index(drop=True))
    lag_hist.to_parquet(LAG_PATH, index=False)
    print(f"  {len(lag_hist):,} rows, {cutoff} → {latest}")
    print(f"  → {LAG_PATH}")

    # 3. Citations hourly median (by hour, day_of_week) — fit on the same
    # training-window cutoff the LightGBM script used. We don't know the exact
    # split_time from this script, so use the full citations file; the API only
    # needs a seasonality proxy here, not a leakage-safe value.
    if os.path.exists(CIT_PATH):
        print("Building citations_hourly_median lookup...")
        cit = pd.read_csv(CIT_PATH, parse_dates=["timestamp"])
        cit["hour"] = cit["timestamp"].dt.hour
        cit["day_of_week"] = cit["timestamp"].dt.weekday
        med = (cit.groupby(["hour", "day_of_week"])["citation_count"]
                  .median().reset_index(name="citations_hourly_median"))
        med.to_parquet(CIT_LOOKUP_PATH, index=False)
        print(f"  {len(med)} (hour, day_of_week) buckets → {CIT_LOOKUP_PATH}")
    else:
        print(f"  SKIP: {CIT_PATH} missing")

    print("=" * 60)


if __name__ == "__main__":
    main()
