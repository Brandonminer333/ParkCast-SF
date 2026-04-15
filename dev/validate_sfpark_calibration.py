"""
ParkCast SF — SFpark Calibration Validation

We use SFpark district × hour × is_weekend occupancy as an inference-time
anchor for our predictions. This script answers three questions:

  1. Temporal stability — does a calibration fit on early dates predict
     held-out later dates well? If yes, the anchor generalizes.
  2. Spatial stability — does a district-level anchor predict individual
     block occupancy within that district, or is within-district variance
     too high for district means to be useful?
  3. vs. trivial baselines — does the calibration actually beat:
       a) global SFpark mean (one number for all SF)
       b) district mean (no hour/weekend adjustment)
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJECT_DIR, "data", "sfpark_sensor_2011_2013.csv")


def main():
    print("=" * 60)
    print("SFpark Calibration Validation")
    print("=" * 60)

    usecols = ["BLOCK_ID", "PM_DISTRICT_NAME", "TIME_OF_DAY", "DAY_TYPE",
               "CAL_DATE", "TOTAL_TIME", "TOTAL_OCCUPIED_TIME"]
    print("Loading SFpark sensor data...")
    df = pd.read_csv(SRC, usecols=usecols)
    print(f"  Rows: {len(df):,}")

    df = df.dropna(subset=["TIME_OF_DAY", "TOTAL_TIME", "DAY_TYPE", "CAL_DATE"])
    df = df[df["TOTAL_TIME"] > 0].copy()
    df["CAL_DATE"] = pd.to_datetime(df["CAL_DATE"], errors="coerce")
    df = df.dropna(subset=["CAL_DATE"])
    df["hour"] = (df["TIME_OF_DAY"].astype(int) // 100).astype(int)
    df["is_weekend"] = (df["DAY_TYPE"] == "weekend").astype(int)
    # Per-row occupancy fraction, not aggregated
    df["occ_pct"] = df["TOTAL_OCCUPIED_TIME"] / df["TOTAL_TIME"] * 100

    print(f"  Date range: {df['CAL_DATE'].min().date()} → {df['CAL_DATE'].max().date()}")
    print(f"  Blocks: {df['BLOCK_ID'].nunique():,}   Districts: {df['PM_DISTRICT_NAME'].nunique()}")

    # Temporal 80/20 split by date
    cutoff = df["CAL_DATE"].quantile(0.80)
    train = df[df["CAL_DATE"] < cutoff].copy()
    test = df[df["CAL_DATE"] >= cutoff].copy()
    print(f"\nTemporal split at: {cutoff.date()}")
    print(f"  Train rows: {len(train):,}   Test rows: {len(test):,}")

    # ── 1. Calibration built from train, evaluated on test ──────────────────
    cal = (train.groupby(["PM_DISTRICT_NAME", "hour", "is_weekend"])
                .apply(lambda g: (g["TOTAL_OCCUPIED_TIME"].sum() / g["TOTAL_TIME"].sum()) * 100)
                .reset_index(name="cal_pred"))
    test_cal = test.merge(cal, on=["PM_DISTRICT_NAME", "hour", "is_weekend"], how="left")
    coverage = test_cal["cal_pred"].notna().mean()
    test_cal = test_cal.dropna(subset=["cal_pred"])

    # ── 2. Trivial baselines ────────────────────────────────────────────────
    global_mean = (train["TOTAL_OCCUPIED_TIME"].sum() / train["TOTAL_TIME"].sum()) * 100
    district_mean = (train.groupby("PM_DISTRICT_NAME")
                          .apply(lambda g: (g["TOTAL_OCCUPIED_TIME"].sum() / g["TOTAL_TIME"].sum()) * 100)
                          .reset_index(name="district_mean_pred"))
    test_cmp = test_cal.merge(district_mean, on="PM_DISTRICT_NAME", how="left")

    y = test_cmp["occ_pct"].values
    print(f"\nHeld-out rows used for MAE: {len(y):,} (coverage {coverage*100:.1f}%)")
    print(f"  Mean occupancy on test: {y.mean():.2f}%   Stdev: {y.std():.2f}")

    for label, pred in [
        ("Global train mean (1 number)", np.full(len(y), global_mean)),
        ("District mean (no hour)",      test_cmp["district_mean_pred"].values),
        ("District×hour×weekend (ours)", test_cmp["cal_pred"].values),
    ]:
        mae = mean_absolute_error(y, pred)
        print(f"  {label:<35} MAE={mae:6.3f}")

    # ── 3. Spatial (within-district) variance check ─────────────────────────
    print("\nWithin-district variance — can a district mean predict individual blocks?")
    per_block = (test.groupby(["PM_DISTRICT_NAME", "BLOCK_ID"])
                     .apply(lambda g: (g["TOTAL_OCCUPIED_TIME"].sum() / g["TOTAL_TIME"].sum()) * 100)
                     .reset_index(name="block_occ"))
    dist_stats = per_block.groupby("PM_DISTRICT_NAME")["block_occ"].agg(
        mean="mean", std="std", min="min", max="max", n="count"
    ).round(1)
    print(dist_stats.to_string())

    # Overall: how much block-level variance does district-level capture?
    overall_var = per_block["block_occ"].var()
    within_var = per_block.groupby("PM_DISTRICT_NAME")["block_occ"].var().mean()
    print(f"\n  Overall block variance: {overall_var:.1f}")
    print(f"  Mean within-district variance: {within_var:.1f}")
    print(f"  Variance explained by district: {(1 - within_var/overall_var)*100:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
