"""One-off script: build per-block inference lookup tables.

Reads `dev/processed_training_data.csv` and emits three parquet files under
`app/models/` that `app/main.py` uses to populate static and hour/dow-
aggregate features at inference time.

Outputs:
    app/models/block_static.parquet
        Columns: lat, lon,
                 poi_dining_200m, poi_retail_200m,
                 poi_transit_200m, poi_attraction_200m,
                 permit_active, permit_count_30d
        One row per unique (lat, lon) block. POI columns are genuinely static
        across the whole training window. `permit_active` / `permit_count_30d`
        are time-varying in the source data; we snapshot the most recent
        training-time value per block as a stopgap until a live feature feed
        is available.

    app/models/complaints_lookup.parquet
        Columns: lat, lon, hour, day_of_week,
                 complaints_311_median, complaints_311_total
        One row per (block, hour, day_of_week) group. The training data has
        exactly one value per such group.

    app/models/sfpark_lookup.parquet
        Columns: lat, lon, hour, is_weekend, sfpark_block_occ
        One row per (block, hour, is_weekend) group. The training data has
        exactly one value per such group.

Run from the repo root:

    python dev/build_block_lookups.py
"""

from __future__ import annotations

import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "dev", "processed_training_data.csv")
MODEL_DIR = os.path.join(REPO_ROOT, "app", "models")

BLOCK_STATIC_PATH = os.path.join(MODEL_DIR, "block_static.parquet")
COMPLAINTS_PATH = os.path.join(MODEL_DIR, "complaints_lookup.parquet")
SFPARK_PATH = os.path.join(MODEL_DIR, "sfpark_lookup.parquet")

POI_COLS = [
    "poi_dining_200m",
    "poi_retail_200m",
    "poi_transit_200m",
    "poi_attraction_200m",
]
PERMIT_COLS = ["permit_active", "permit_count_30d"]
COMPLAINT_COLS = ["complaints_311_median", "complaints_311_total"]


def main() -> None:
    if not os.path.exists(SRC_PATH):
        raise SystemExit(f"Source CSV not found: {SRC_PATH}")

    print(f"Reading {SRC_PATH} ...")
    needed = (
        ["lat", "lon", "timestamp", "hour", "day_of_week", "is_weekend"]
        + POI_COLS
        + PERMIT_COLS
        + COMPLAINT_COLS
        + ["sfpark_block_occ"]
    )
    df = pd.read_csv(SRC_PATH, usecols=needed, parse_dates=["timestamp"])
    print(f"  {len(df):,} rows, {df['lat'].nunique():,} unique lats")

    print(f"Building {BLOCK_STATIC_PATH} ...")
    poi = df.groupby(["lat", "lon"], as_index=False)[POI_COLS].first()
    permits = (
        df.sort_values("timestamp").groupby(["lat", "lon"], as_index=False)[PERMIT_COLS].last()
    )
    block_static = poi.merge(permits, on=["lat", "lon"], how="left")
    block_static.to_parquet(BLOCK_STATIC_PATH, index=False)
    print(f"  wrote {len(block_static):,} rows")

    print(f"Building {COMPLAINTS_PATH} ...")
    complaints = df.groupby(["lat", "lon", "hour", "day_of_week"], as_index=False)[
        COMPLAINT_COLS
    ].first()
    complaints.to_parquet(COMPLAINTS_PATH, index=False)
    print(f"  wrote {len(complaints):,} rows")

    print(f"Building {SFPARK_PATH} ...")
    sfpark = df.groupby(["lat", "lon", "hour", "is_weekend"], as_index=False)[
        "sfpark_block_occ"
    ].first()
    sfpark.to_parquet(SFPARK_PATH, index=False)
    print(f"  wrote {len(sfpark):,} rows")

    print("Done.")


if __name__ == "__main__":
    main()
