"""
ParkCast SF — SFpark Calibration Builder

Our LightGBM predicts occupancy from paid meter transactions, which
systematically undercounts physical curb occupancy (scofflaws, placards,
loading, missed payments). SFpark's 2011-2013 in-ground sensor pilot
measured true occupancy for ~8,000 spaces; we use it as a ground-truth
anchor.

Produces app/models/sfpark_calibration.parquet:

  columns: pm_district, hour, is_weekend, sfpark_occ_pct
  rows:    10 districts × 24 hours × 2 day_types = 480

Also writes app/models/sfpark_calibration.meta.json with the
district → analysis_neighborhood mapping used at inference time.
"""

import json
import os

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "app", "models")

SRC = os.path.join(DATA_DIR, "sfpark_sensor_2011_2013.csv")
OUT = os.path.join(MODEL_DIR, "sfpark_calibration.parquet")
META_OUT = os.path.join(MODEL_DIR, "sfpark_calibration.meta.json")

# SFpark PM_DISTRICT_NAME → SF analysis_neighborhood values we use in training.
# Derived by hand from district geographic descriptions. Districts with no
# clean 1:1 match (Union Square ↔ multiple nhbds) fall back via the average
# across all districts at inference time.
DISTRICT_TO_NEIGHBORHOODS = {
    "Downtown": ["Financial District/South Beach"],
    "Civic Center": ["Tenderloin", "Hayes Valley"],
    "Fillmore": ["Western Addition", "Pacific Heights"],
    "Marina": ["Marina"],
    "Fisherman's Wharf": ["North Beach", "Russian Hill"],
    "Mission": ["Mission"],
    "South Embarcadero": ["Mission Bay", "South of Market"],
    "Union": ["Nob Hill", "Chinatown"],
    "Inner Richmond": ["Inner Richmond"],
    "West Portal": ["West of Twin Peaks"],
}


def main():
    print("=" * 60)
    print("ParkCast SF — SFpark Calibration Builder")
    print("=" * 60)

    usecols = ["PM_DISTRICT_NAME", "TIME_OF_DAY", "DAY_TYPE",
               "TOTAL_TIME", "TOTAL_OCCUPIED_TIME"]
    print(f"Reading {SRC} ...")
    df = pd.read_csv(SRC, usecols=usecols)
    print(f"  Rows: {len(df):,}")

    # Drop rows with zero / missing sensor time or missing time-of-day.
    df = df.dropna(subset=["TIME_OF_DAY", "TOTAL_TIME", "DAY_TYPE"])
    df = df[df["TOTAL_TIME"] > 0].copy()

    # TIME_OF_DAY is HHMM-coded (e.g. 2100 for 9pm). Convert to hour.
    df["hour"] = (df["TIME_OF_DAY"].astype(int) // 100).astype(int)
    df["is_weekend"] = (df["DAY_TYPE"] == "weekend").astype(int)

    # Aggregate occupancy = sum(occupied seconds) / sum(total seconds).
    # This is a sensor-weighted average — longer spans and busier blocks
    # dominate, which is what we want for a ground-truth anchor.
    grp = (df.groupby(["PM_DISTRICT_NAME", "hour", "is_weekend"])
             [["TOTAL_TIME", "TOTAL_OCCUPIED_TIME"]].sum().reset_index())
    grp["sfpark_occ_pct"] = (
        grp["TOTAL_OCCUPIED_TIME"] / grp["TOTAL_TIME"] * 100
    ).round(2)
    grp = grp.rename(columns={"PM_DISTRICT_NAME": "pm_district"})
    grp = grp[["pm_district", "hour", "is_weekend", "sfpark_occ_pct"]]

    os.makedirs(MODEL_DIR, exist_ok=True)
    grp.to_parquet(OUT, index=False)
    print(f"  Saved {len(grp):,} rows → {OUT}")

    with open(META_OUT, "w") as f:
        json.dump({"district_to_neighborhoods": DISTRICT_TO_NEIGHBORHOODS}, f, indent=2)
    print(f"  Saved mapping → {META_OUT}")

    print("\nSample (weekday noon):")
    print(grp[(grp["hour"] == 12) & (grp["is_weekend"] == 0)]
          .sort_values("sfpark_occ_pct", ascending=False).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
