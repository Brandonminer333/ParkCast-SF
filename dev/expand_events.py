"""
ParkCast SF — Event Catalog Expander

Current data/events.csv is Chase Center + Oracle Park only (~185 rows,
nearly all sports). The Special Event permits in data/street_closures.csv
cover 1,174 additional events (street fairs, festivals, parades, farmers
markets, Bay to Breakers, etc.) that block parking across residential and
commercial corridors.

We parse each permit's LINESTRING, take the midpoint, infer a neighborhood
via nearest-block lookup from master_blocks.parquet, and bucket by hour.

Output: data/events.csv (replacement — venue events + permit events merged).
"""

import os
import re
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "app", "models")

CLOSURES = os.path.join(DATA_DIR, "street_closures.csv")
EVENTS = os.path.join(DATA_DIR, "events.csv")
MASTER = os.path.join(MODEL_DIR, "master_blocks.parquet")

_LS_RE = re.compile(r"-?\d+\.\d+")


def linestring_midpoint(wkt):
    if not isinstance(wkt, str):
        return None, None
    nums = [float(x) for x in _LS_RE.findall(wkt)]
    if len(nums) < 4 or len(nums) % 2 != 0:
        return None, None
    lons = nums[0::2]
    lats = nums[1::2]
    return float(np.mean(lats)), float(np.mean(lons))


def classify(name: str) -> str:
    n = (name or "").lower()
    if any(k in n for k in ("farmers market", "farmer's market", "farmers' market")):
        return "Market"
    if any(k in n for k in ("marathon", "breakers", "run", "5k", "10k")):
        return "Run"
    if any(k in n for k in ("parade", "pride", "lunar")):
        return "Parade"
    if any(k in n for k in ("festival", "fair", "fiesta", "fest")):
        return "Festival"
    if any(k in n for k in ("concert", "music", "symphony")):
        return "Music"
    return "Community"


def main():
    print("=" * 60)
    print("ParkCast SF — Event Catalog Expander")
    print("=" * 60)

    venue = pd.read_csv(EVENTS)
    print(f"Existing venue events: {len(venue):,}")

    closures = pd.read_csv(CLOSURES, low_memory=False)
    ev = closures[closures["type"] == "Special Event"].copy()
    print(f"Special Event permits: {len(ev):,}")

    lats, lons = [], []
    for wkt in ev["shape"]:
        lat, lon = linestring_midpoint(wkt)
        lats.append(lat)
        lons.append(lon)
    ev["lat"] = lats
    ev["lon"] = lons
    ev = ev.dropna(subset=["lat", "lon", "start_dt", "end_dt"]).copy()

    ev["start_ts"] = pd.to_datetime(ev["start_dt"], errors="coerce")
    ev["end_ts"] = pd.to_datetime(ev["end_dt"], errors="coerce")
    ev = ev.dropna(subset=["start_ts", "end_ts"]).copy()
    ev["date"] = ev["start_ts"].dt.date.astype(str)
    ev["start_hour"] = ev["start_ts"].dt.hour.astype(int)
    ev["end_hour"] = ev["end_ts"].dt.hour.clip(lower=1).astype(int)

    master = pd.read_parquet(MASTER)
    tree = cKDTree(master[["lat", "lon"]].values)
    _, idx = tree.query(ev[["lat", "lon"]].values, k=1)
    ev["neighborhood"] = master.iloc[idx]["neighborhood"].values

    permit = pd.DataFrame({
        "date": ev["date"],
        "event_name": ev["case_name"].fillna("Special Event"),
        "type": ev["case_name"].fillna("").apply(classify),
        "neighborhood": ev["neighborhood"].fillna("").astype(str),
        "venue_lat": ev["lat"].round(6),
        "venue_lon": ev["lon"].round(6),
        "start_hour": ev["start_hour"],
        "end_hour": ev["end_hour"],
    })

    combined = pd.concat([venue, permit], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "event_name", "venue_lat", "venue_lon"]
    ).sort_values(["date", "start_hour"]).reset_index(drop=True)

    combined.to_csv(EVENTS, index=False)
    print(f"\nCombined events: {len(combined):,}")
    print("Type distribution:")
    print(combined["type"].value_counts().to_string())
    print(f"\nDate range: {combined['date'].min()} → {combined['date'].max()}")
    print(f"Saved → {EVENTS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
