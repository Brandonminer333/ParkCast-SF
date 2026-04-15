"""
ParkCast SF — Master Block Catalog Builder

Produces app/models/master_blocks.parquet: one row per SF blockface (citywide),
classified by what data signal we have for it:

  metered       → has at least one active paid meter. Predict via LightGBM.
  rpp           → has a Pay-or-Permit / RPP regulation. Permit info returned.
  no_parking    → "No parking" / "No stopping" / color curb regulation.
  time_limited  → posted time-limit (e.g., 2-hour).
  unmetered     → residential / free street parking.

For each block we store:
  cnn, lat, lon (midpoint), corridor, limits, block_class,
  metered_lat, metered_lon (for LightGBM feature lookup on metered blocks),
  rpp_area, reg_summary, hrlimit, sweep_weekdays
"""

import os
import re
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "app", "models")

SWEEPING_PATH = os.path.join(DATA_DIR, "street_sweeping.csv")
REGULATIONS_PATH = os.path.join(DATA_DIR, "parking_regulations.csv")
METERS_PATH = os.path.join(DATA_DIR, "meter_locations.csv")
RPP_PARCELS_PATH = os.path.join(DATA_DIR, "rpp_parcels.json")
METERED_BLOCKS_PATH = os.path.join(MODEL_DIR, "blocks.parquet")  # metered-only catalog
PARKING_CENSUS_PATH = os.path.join(DATA_DIR, "parking_census.json")
BLUE_CURB_PATH = os.path.join(DATA_DIR, "blue_curb.json")
MUNI_STOPS_PATH = os.path.join(DATA_DIR, "muni_stops.json")
SHUTTLE_STOPS_PATH = os.path.join(DATA_DIR, "shuttle_stops.json")

OUT_PATH = os.path.join(MODEL_DIR, "master_blocks.parquet")

# Spatial match radius: ~50m in lat/lon degrees ≈ 0.00045
MATCH_DEG = 0.00055  # slightly more than 50m to catch slight coord drift


# ── LINESTRING parsing ──────────────────────────────────────────────────────
_LS_RE = re.compile(r"-?\d+\.\d+")


def linestring_midpoint(wkt):
    """Parse LINESTRING (...) WKT into midpoint (lat, lon).
    Handles both LINESTRING and MULTILINESTRING."""
    if not isinstance(wkt, str):
        return None, None
    nums = [float(x) for x in _LS_RE.findall(wkt)]
    if len(nums) < 4 or len(nums) % 2 != 0:
        return None, None
    # Pairs are (lon, lat) in WKT
    lons = nums[0::2]
    lats = nums[1::2]
    return float(np.mean(lats)), float(np.mean(lons))


def bulk_linestring_midpoints(series):
    """Vectorized-ish LINESTRING parsing for a pandas Series."""
    lats, lons = [], []
    for wkt in series:
        lat, lon = linestring_midpoint(wkt)
        lats.append(lat)
        lons.append(lon)
    return np.array(lats), np.array(lons)


def main():
    print("=" * 60)
    print("ParkCast SF — Master Block Catalog")
    print("=" * 60)

    # ── 1. Base catalog from street_sweeping (citywide index) ────────────────
    print("Loading street_sweeping.csv...")
    sw = pd.read_csv(SWEEPING_PATH)
    print(f"  Rows: {len(sw):,}")

    print("Extracting block midpoints from LINESTRINGs...")
    lats, lons = bulk_linestring_midpoints(sw["line"])
    sw["lat_mid"] = lats
    sw["lon_mid"] = lons

    # One row per cnn (collapse all sides/schedules)
    print("Collapsing to one row per cnn...")
    block_df = (sw.dropna(subset=["lat_mid", "lon_mid"])
                   .groupby("cnn", as_index=False)
                   .agg(lat=("lat_mid", "mean"),
                        lon=("lon_mid", "mean"),
                        corridor=("corridor", "first"),
                        limits=("limits", "first"),
                        sweep_weekdays=("weekday", lambda s: ",".join(sorted(set(s.dropna()))))))
    print(f"  Unique blocks: {len(block_df):,}")

    # ── 2. Mark metered blocks ──────────────────────────────────────────────
    print("Tagging metered blocks...")
    metered = pd.read_parquet(METERED_BLOCKS_PATH)  # from build_inference_assets.py
    # Spatial nearest-neighbor: for each metered block, find nearest catalog block
    # Both have lat/lon; use a simple KDTree.
    from scipy.spatial import cKDTree
    tree = cKDTree(block_df[["lat", "lon"]].values)
    dists, idx = tree.query(metered[["lat", "lon"]].values, k=1)
    close = dists < MATCH_DEG
    metered_cnns = block_df.iloc[idx[close]]["cnn"].values

    # Store the metered block's (lat, lon) key so API can look up the LightGBM features
    metered_lat_map = dict(zip(metered_cnns, metered.loc[close, "lat"].values))
    metered_lon_map = dict(zip(metered_cnns, metered.loc[close, "lon"].values))
    block_df["metered_lat"] = block_df["cnn"].map(metered_lat_map)
    block_df["metered_lon"] = block_df["cnn"].map(metered_lon_map)
    block_df["is_metered"] = block_df["cnn"].isin(metered_cnns)
    print(f"  Metered blocks matched: {block_df['is_metered'].sum():,}")

    # Assign every block the neighborhood of its nearest metered block. The
    # metered catalog covers all neighborhoods with paid parking and most
    # adjacent residential zones; the nearest-neighbor label is a good proxy
    # for non-metered blocks so downstream code can look up SFpark district
    # priors by neighborhood.
    nbh_tree = cKDTree(metered[["lat", "lon"]].values)
    _, nbh_idx = nbh_tree.query(block_df[["lat", "lon"]].values, k=1)
    block_df["neighborhood"] = metered.iloc[nbh_idx]["neighborhood"].values

    # ── 3. Attach regulation info (RPP / color curb / time-limited) ─────────
    print("Parsing regulations...")
    reg = pd.read_csv(REGULATIONS_PATH, low_memory=False)
    reg_lats, reg_lons = bulk_linestring_midpoints(reg["shape"])
    reg["lat"] = reg_lats
    reg["lon"] = reg_lons
    reg = reg.dropna(subset=["lat", "lon"])

    # For each regulation, find nearest block
    reg_tree_q = cKDTree(block_df[["lat", "lon"]].values)
    dists, idx = reg_tree_q.query(reg[["lat", "lon"]].values, k=1)
    close = dists < MATCH_DEG
    reg["cnn"] = np.where(close, block_df.iloc[idx]["cnn"].values, np.nan)
    reg_matched = reg[reg["cnn"].notna()].copy()
    print(f"  Regulations matched to blocks: {len(reg_matched):,}")

    def classify_regulations(group):
        types = set(group["regulation"].str.lower().str.strip())
        if any("no parking" in t or "no stopping" in t for t in types if isinstance(t, str)):
            cls = "no_parking"
        elif any("pay or permit" in t or "government permit" in t for t in types if isinstance(t, str)):
            cls = "rpp"
        elif any("time limited" in t for t in types if isinstance(t, str)):
            cls = "time_limited"
        else:
            cls = "other"
        rpp_area = next((a for a in group["rpparea1"].dropna().astype(str).unique()), "")
        summary = "; ".join(sorted(set(group["regulation"].dropna().astype(str))))
        hrlimit = group["hrlimit"].dropna().min()
        return pd.Series({
            "reg_class": cls,
            "rpp_area": rpp_area,
            "reg_summary": summary,
            "hrlimit": float(hrlimit) if pd.notna(hrlimit) else None,
        })

    reg_by_cnn = reg_matched.groupby("cnn").apply(classify_regulations, include_groups=False).reset_index()
    block_df = block_df.merge(reg_by_cnn, on="cnn", how="left")

    # ── 3b. RPP zone tagging via parcel spatial join ────────────────────────
    # `parking_regulations.csv` only lists posted signs, which misses whole
    # RPP zones with just a few boundary signs. The per-parcel eligibility
    # dataset tags every parcel with its RPP area letter (A, D, Q, …), so
    # we assign each block the RPP area of its nearest parcel, then flip
    # currently-unmetered blocks to `rpp` if the parcel has a zone code.
    rpp_area_from_parcels = None
    if os.path.exists(RPP_PARCELS_PATH):
        import json
        print("Tagging RPP zones from parcel polygons...")
        with open(RPP_PARCELS_PATH) as f:
            parcels = json.load(f)

        # Use first polygon vertex as a cheap centroid proxy — parcels are
        # small enough (~20-50m) that exact centroids don't change the
        # nearest-neighbor result for block midpoints.
        p_lat, p_lon, p_rpp = [], [], []
        for p in parcels:
            shp = p.get("shape")
            if not shp or shp.get("type") != "MultiPolygon":
                continue
            coords = shp["coordinates"]
            if not coords or not coords[0] or not coords[0][0]:
                continue
            first = coords[0][0][0]  # [lon, lat]
            p_lon.append(first[0])
            p_lat.append(first[1])
            p_rpp.append(p.get("rppeligib") or "")
        p_lat = np.array(p_lat)
        p_lon = np.array(p_lon)
        p_rpp = np.array(p_rpp)
        print(f"  Parcels with geometry: {len(p_lat):,}")

        parcel_tree = cKDTree(np.column_stack([p_lat, p_lon]))
        # MATCH_DEG ≈ 50m; use 2x for block→parcel. SF parcels are small
        # relative to blocks, so the nearest parcel is almost always on or
        # adjacent to the same block.
        dists, idx = parcel_tree.query(block_df[["lat", "lon"]].values, k=1)
        tagged = p_rpp[idx]
        # Only keep tags for blocks close enough to a parcel
        tagged = np.where(dists < MATCH_DEG * 3, tagged, "")
        block_df["rpp_parcel_area"] = tagged
        print(f"  Blocks in an RPP zone: {(block_df['rpp_parcel_area'] != '').sum():,}")
        rpp_area_from_parcels = True

    # ── 3c. Parking census: authoritative space counts (cnn-keyed) ──────────
    if os.path.exists(PARKING_CENSUS_PATH):
        import json
        print("Joining parking census (prkg_sply)...")
        with open(PARKING_CENSUS_PATH) as f:
            census = json.load(f)
        cen_df = pd.DataFrame([
            {"cnn": str(int(c["cnn"])).zfill(7) if c.get("cnn") and str(c["cnn"]).isdigit() else c.get("cnn"),
             "total_spaces": float(c["prkg_sply"]) if c.get("prkg_sply") else None}
            for c in census
            if c.get("cnn") and c.get("prkg_sply") is not None
        ]).dropna(subset=["total_spaces"])
        # Our master cnn is stored as int-like; normalize both sides
        cen_df["cnn"] = cen_df["cnn"].astype(str).str.lstrip("0")
        block_df["cnn_key"] = block_df["cnn"].astype(str).str.lstrip("0")
        cen_by = cen_df.groupby("cnn", as_index=False)["total_spaces"].sum()
        block_df = block_df.merge(
            cen_by.rename(columns={"cnn": "cnn_key"}), on="cnn_key", how="left"
        ).drop(columns=["cnn_key"])
        print(f"  Blocks with census space counts: {block_df['total_spaces'].notna().sum():,}")
    else:
        block_df["total_spaces"] = np.nan

    # ── 3d. Blue curb (disabled) signs — cnn-keyed ──────────────────────────
    if os.path.exists(BLUE_CURB_PATH):
        import json
        print("Tagging blue-curb (disabled) blocks...")
        with open(BLUE_CURB_PATH) as f:
            blue = json.load(f)
        blue_cnns = {
            str(b.get("cnn", "")).lstrip("0") for b in blue
            if b.get("cnn") and str(b.get("status_code", "")).upper() in ("", "ACTIVE", "INSTALLED") or b.get("cnn")
        }
        blue_cnns.discard("")
        block_df["has_blue_zone"] = (
            block_df["cnn"].astype(str).str.lstrip("0").isin(blue_cnns)
        )
        print(f"  Blocks with blue (disabled) curb: {block_df['has_blue_zone'].sum():,}")
    else:
        block_df["has_blue_zone"] = False

    # ── 3e. Muni stops — nearest-block spatial join ─────────────────────────
    if os.path.exists(MUNI_STOPS_PATH):
        import json
        print("Tagging bus-zone blocks (Muni stops)...")
        with open(MUNI_STOPS_PATH) as f:
            muni = json.load(f)
        mlat, mlon = [], []
        for m in muni:
            try:
                mlat.append(float(m["latitude"]))
                mlon.append(float(m["longitude"]))
            except (KeyError, TypeError, ValueError):
                continue
        if mlat:
            muni_tree = cKDTree(np.column_stack([mlat, mlon]))
            dists, _ = muni_tree.query(block_df[["lat", "lon"]].values, k=1)
            block_df["has_bus_zone"] = dists < MATCH_DEG
        else:
            block_df["has_bus_zone"] = False
        print(f"  Blocks near a Muni stop: {block_df['has_bus_zone'].sum():,}")
    else:
        block_df["has_bus_zone"] = False

    # ── 3f. Commuter shuttle stops — nearest-block spatial join ─────────────
    if os.path.exists(SHUTTLE_STOPS_PATH):
        import json
        print("Tagging shuttle-stop blocks...")
        with open(SHUTTLE_STOPS_PATH) as f:
            shuttle = json.load(f)
        slat, slon = [], []
        for s in shuttle:
            try:
                slat.append(float(s["latitude"]))
                slon.append(float(s["longitude"]))
            except (KeyError, TypeError, ValueError):
                continue
        if slat:
            sh_tree = cKDTree(np.column_stack([slat, slon]))
            dists, _ = sh_tree.query(block_df[["lat", "lon"]].values, k=1)
            block_df["has_shuttle_stop"] = dists < MATCH_DEG
        else:
            block_df["has_shuttle_stop"] = False
        print(f"  Blocks near a commuter shuttle stop: {block_df['has_shuttle_stop'].sum():,}")
    else:
        block_df["has_shuttle_stop"] = False

    # ── 4. Final classification ─────────────────────────────────────────────
    def final_class(row):
        if row["is_metered"]:
            return "metered"
        rc = row.get("reg_class")
        if rc == "no_parking":
            return "no_parking"
        if rc == "rpp":
            return "rpp"
        # Zonal RPP (residential permit): tag from parcel eligibility.
        # Applies to blocks with no specific posted regulation.
        if rpp_area_from_parcels and row.get("rpp_parcel_area"):
            return "rpp"
        if rc == "time_limited":
            return "time_limited"
        return "unmetered"

    block_df["block_class"] = block_df.apply(final_class, axis=1)

    # Prefer the parcel-derived RPP area code over the signed-regulation one
    # when we flagged the block from parcels; the parcel code is zone-level
    # (A, D, Q…) while the regulation "rpparea1" is often missing.
    if rpp_area_from_parcels:
        parcel_rpp = block_df["rpp_parcel_area"].replace("", np.nan)
        block_df["rpp_area"] = block_df["rpp_area"].where(
            block_df["rpp_area"].astype(str) != "", parcel_rpp
        )
        block_df["rpp_area"] = block_df["rpp_area"].fillna(parcel_rpp)
    print(f"\nClass distribution:")
    print(block_df["block_class"].value_counts().to_string())

    # ── 5. Save master catalog ──────────────────────────────────────────────
    keep_cols = ["cnn", "lat", "lon", "corridor", "limits", "block_class",
                 "is_metered", "metered_lat", "metered_lon", "neighborhood",
                 "rpp_area", "reg_summary", "hrlimit", "sweep_weekdays",
                 "total_spaces", "has_blue_zone", "has_bus_zone",
                 "has_shuttle_stop"]
    block_df = block_df[keep_cols]
    block_df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved master catalog: {OUT_PATH}  ({len(block_df):,} blocks)")

    # Note: parking_citations.csv has lat/lon populated on only ~2/1.58M rows,
    # so no usable citywide citation-rate proxy. The metered citations_by_block
    # dataset (keyed by blockface_id) is already wired into the LightGBM model.
    print("=" * 60)


if __name__ == "__main__":
    main()
