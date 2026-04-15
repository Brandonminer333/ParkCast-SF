"""
ParkCast SF — Data Preprocessing Script
Aggregates raw transaction logs into hourly occupancy percentages (%)
Enriches the data with weather, street sweeping, and event features.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "processed_training_data.csv")

# 2026 Holidays (Partial list for labelling)
HOLIDAYS_2026 = [
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-05", # Added Easter Sunday
    "2026-05-25", "2026-06-19", "2026-07-04", "2026-09-07", 
    "2026-10-12", "2026-11-11", "2026-11-26", "2026-12-25"
]

def is_school_day(dt):
    """Simple heuristic for SF school days (Mon-Fri, excluding summer/winter)."""
    if dt.weekday() >= 5: return 0
    # SFUSD Spring Break 2026: March 27 - April 3
    date_only = dt.date()
    if date_only >= datetime(2026, 3, 27).date() and date_only <= datetime(2026, 4, 3).date():
        return 0
        
    return 1

# ── Step 1: Load Data ────────────────────────────────────────────────────────
def load_data():
    print("Loading raw datasets...")
    df_trans = pd.read_csv(os.path.join(DATA_DIR, "meter_transactions.csv"))
    df_locs = pd.read_csv(os.path.join(DATA_DIR, "meter_locations.csv"))
    df_weather = pd.read_csv(os.path.join(DATA_DIR, "weather.csv"))
    df_sweeping = pd.read_csv(os.path.join(DATA_DIR, "street_sweeping.csv"))
    
    # Optional datasets
    try:
        df_closures = pd.read_csv(os.path.join(DATA_DIR, "street_closures.csv"))
    except:
        df_closures = None
        
    return df_trans, df_locs, df_weather, df_sweeping, df_closures

# ── Step 2: Aggregate Occupancy ──────────────────────────────────────────────
def calculate_occupancy(df_trans, df_locs):
    """
    Build a full (active_blockface × hour) grid and count distinct meters
    with an active paid session in each cell. Empty cells become 0, not missing —
    this is the fix for the prior selection bias.
    """
    print("Calculating hourly occupancy from transactions...")

    loc_cols = ['post_id', 'blockface_id', 'analysis_neighborhood', 'latitude', 'longitude']
    df_locs = df_locs.dropna(subset=['blockface_id', 'post_id'])[loc_cols]

    df = df_trans.merge(df_locs, on='post_id', how='inner').reset_index(drop=True)
    df['start'] = pd.to_datetime(df['session_start_dt'])
    df['end'] = pd.to_datetime(df['session_end_dt'])

    # Drop degenerate rows: negative or absurdly long sessions
    dur_hrs = (df['end'] - df['start']).dt.total_seconds() / 3600
    df = df[(dur_hrs > 0) & (dur_hrs <= 24)].reset_index(drop=True)

    # Hour grid spans the whole transaction window
    start_hour = df['start'].min().floor('h')
    end_hour = df['end'].max().floor('h')
    hours = pd.date_range(start=start_hour, end=end_hour, freq='h')

    # Blockface capacity: only count meter posts that actually generate
    # transactions. The raw locations file lists ~38k posts but only ~16k
    # ever bill a session — dead meters, reserved zones, and legacy hardware
    # inflate the denominator and pushed apparent occupancy ~2-3x too low.
    active_posts = set(df['post_id'].dropna().unique())
    df_locs_active = df_locs[df_locs['post_id'].isin(active_posts)]
    block_capacity = df_locs_active.groupby('blockface_id')['post_id'].nunique()
    block_meta = df_locs_active.groupby('blockface_id').agg(
        neighborhood=('analysis_neighborhood', 'first'),
        lat=('latitude', 'mean'),
        lon=('longitude', 'mean'),
    )

    # Drop blockfaces with fewer than MIN_ACTIVE_POSTS active meters. These
    # tiny blockfaces produce binary-looking occupancy_pct (0/100 or 0/50/100),
    # dominate the variance, and aren't useful routing destinations anyway.
    MIN_ACTIVE_POSTS = 4
    valid_bfs = block_capacity[block_capacity >= MIN_ACTIVE_POSTS].index
    block_capacity = block_capacity.loc[valid_bfs]
    block_meta = block_meta.loc[valid_bfs]
    df = df[df['blockface_id'].isin(valid_bfs)].reset_index(drop=True)
    print(f"  Blockfaces with >= {MIN_ACTIVE_POSTS} active meters: {len(valid_bfs):,}")

    # Only keep blockfaces that ever had a paid session (meter-active)
    active_blocks = df['blockface_id'].dropna().unique()
    print(f"  Active blockfaces: {len(active_blocks)} | Hours: {len(hours)} "
          f"| Grid: {len(active_blocks) * len(hours):,} rows")

    # Explode each transaction into the hours it overlaps
    df['start_h'] = df['start'].dt.floor('h')
    df['n_hours'] = ((df['end'].dt.ceil('h') - df['start_h']).dt.total_seconds() / 3600).astype(int)
    df = df[df['n_hours'] > 0].reset_index(drop=True)

    rep = df.loc[df.index.repeat(df['n_hours']), ['blockface_id', 'post_id', 'start_h']].copy()
    rep['offset'] = rep.groupby(rep.index).cumcount()
    rep['hour'] = rep['start_h'] + pd.to_timedelta(rep['offset'], unit='h')

    # Distinct meters paid per (blockface, hour)
    occ = (rep.groupby(['blockface_id', 'hour'])['post_id']
              .nunique()
              .reset_index(name='occupied'))

    # Full cross-product grid, left-joined to counts (missing = 0)
    grid = (pd.MultiIndex
              .from_product([active_blocks, hours], names=['blockface_id', 'timestamp'])
              .to_frame(index=False))
    grid = grid.merge(occ, left_on=['blockface_id', 'timestamp'],
                      right_on=['blockface_id', 'hour'], how='left')
    grid['occupied'] = grid['occupied'].fillna(0)
    grid = grid.drop(columns=['hour'])

    # Attach metadata + capacity
    grid = grid.merge(block_meta, on='blockface_id', how='left')
    grid['total_spaces'] = grid['blockface_id'].map(block_capacity).astype(int)
    grid['occupancy_pct'] = (grid['occupied'] / grid['total_spaces'] * 100).clip(upper=100).round(2)

    return grid[['timestamp', 'blockface_id', 'neighborhood',
                 'lat', 'lon', 'occupancy_pct', 'total_spaces']]

# ── Step 3: Add Features ─────────────────────────────────────────────────────
def enrich_features(df, df_weather, df_sweeping, df_closures):
    print("Enriching with features (Weather, School, Holiday, Events)...")
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.weekday
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Labels
    df['is_holiday'] = df['timestamp'].dt.strftime('%Y-%m-%d').isin(HOLIDAYS_2026).astype(int)
    df['is_school_day'] = df['timestamp'].apply(is_school_day)
    
    # Weather
    print("  Merging weather...")
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df = df.merge(df_weather[['time', 'temperature_2m', 'precipitation']], left_on='timestamp', right_on='time', how='left')
    df.rename(columns={'temperature_2m': 'temperature', 'precipitation': 'is_raining'}, inplace=True)
    df['is_raining'] = (df['is_raining'] > 0).astype(int)
    
    # Street Sweeping (Simplified check: if neighborhood matches and day matches)
    # Real SF sweeping is complex; we'll add a placeholder binary feature based on neighborhood patterns
    df['is_street_cleaning'] = 0 # would be calculated by joining df_sweeping CNNs
    
    # Events — continuous intensity = max over active events of
    # exp(-dist/300) within 800m, during event hours. Captures both
    # distance falloff (~0.07 at 800m, 1.0 at venue) and presence/absence.
    df['event_intensity'] = 0.0
    events_path = os.path.join(DATA_DIR, "events.csv")
    if os.path.exists(events_path):
        print("  Merging events (continuous intensity)...")
        df_ev = pd.read_csv(events_path)
        df_ev['date'] = pd.to_datetime(df_ev['date']).dt.date
        df['date_only'] = df['timestamp'].dt.date

        R = 6_371_000
        lat_rad = np.radians(df['lat'].values)
        lon_rad = np.radians(df['lon'].values)
        intensity = np.zeros(len(df), dtype=np.float32)

        for _, ev in df_ev.iterrows():
            v_lat = np.radians(ev['venue_lat'])
            v_lon = np.radians(ev['venue_lon'])
            dlat = lat_rad - v_lat
            dlon = lon_rad - v_lon
            a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad) * np.cos(v_lat) * np.sin(dlon / 2) ** 2
            dist = 2 * R * np.arcsin(np.sqrt(a))

            time_mask = (
                (df['date_only'].values == ev['date'])
                & (df['hour'].values >= ev['start_hour'])
                & (df['hour'].values <= ev['end_hour'])
            )
            close = time_mask & (dist <= 800)
            if close.any():
                contrib = np.exp(-dist[close] / 300.0).astype(np.float32)
                idx = np.where(close)[0]
                np.maximum.at(intensity, idx, contrib)

        df['event_intensity'] = intensity
        df.drop(columns=['date_only'], inplace=True)

    return df

# ── Citation signal (24/7 demand proxy) ──────────────────────────────────────
def merge_citations(df):
    """
    Join hourly citation counts per blockface. Also compute a robust
    per-(block, hour-of-day, weekday) median so hours with zero citations
    still carry the block's typical signal for that slot.
    """
    cit_path = os.path.join(DATA_DIR, "citations_by_block.csv")
    if not os.path.exists(cit_path):
        print("  No citations_by_block.csv found — skipping citation features.")
        df['citation_count'] = 0.0
        df['citations_hourly_median'] = 0.0
        return df

    print("  Merging citation signal...")
    cit = pd.read_csv(cit_path, parse_dates=['timestamp'])

    # Raw per-hour count
    df = df.merge(cit, on=['blockface_id', 'timestamp'], how='left')
    df['citation_count'] = df['citation_count'].fillna(0.0)

    # Historical median citations per (blockface, hour-of-day, weekday)
    cit['hour_of_day'] = cit['timestamp'].dt.hour
    cit['weekday'] = cit['timestamp'].dt.weekday
    median = (cit.groupby(['blockface_id', 'hour_of_day', 'weekday'])
                 ['citation_count'].median()
                 .reset_index(name='citations_hourly_median'))
    df = df.merge(median,
                  left_on=['blockface_id', 'hour', 'day_of_week'],
                  right_on=['blockface_id', 'hour_of_day', 'weekday'],
                  how='left')
    df['citations_hourly_median'] = df['citations_hourly_median'].fillna(0.0)
    df = df.drop(columns=['hour_of_day', 'weekday'], errors='ignore')
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ParkCast SF — Preprocessing Real Data")
    print("=" * 60)
    
    trans, locs, weather, sweeping, closures = load_data()
    
    # 1. Occupancy
    df_processed = calculate_occupancy(trans, locs)
    
    # 2. Features
    df_final = enrich_features(df_processed, weather, sweeping, closures)
    
    # 3. Merge citation signal (available 24/7, unlike paid meters)
    df_final = merge_citations(df_final)

    # Flag whether meter enforcement is active — lets the model learn that
    # occupancy_pct is only meaningful during enforcement and that citation
    # rate is the dominant demand signal outside it.
    df_final['is_meter_hours'] = (
        (df_final['day_of_week'] < 6)
        & (df_final['hour'].between(9, 17))
        & (df_final['is_holiday'] == 0)
    ).astype(int)

    # 4. Off-hours target. Paid-meter occupancy is ~0 when meters are free,
    #    which would teach the model to predict zero every night. No public
    #    SF dataset measures 24/7 curb occupancy, so off-hour values are an
    #    estimate anchored to real signals: each block's own daytime baseline,
    #    a residential-style diurnal curve, the block's historical citation
    #    demand at that hour, and any nearby event. Rows with estimated
    #    targets are flagged via `target_is_estimated` so downstream work can
    #    weight, filter, or report metrics separately.
    baseline = (df_final[df_final['is_meter_hours'] == 1]
                .groupby(['lat', 'lon'])['occupancy_pct'].mean()
                .rename('blockface_baseline').reset_index())
    df_final = df_final.merge(baseline, on=['lat', 'lon'], how='left')
    df_final['blockface_baseline'] = df_final['blockface_baseline'].fillna(0.0)

    # Diurnal multiplier for off-hours. Residential curb parking is typically
    # fullest overnight (residents home) and emptiest mid-morning (commuters
    # gone, meters just started). Values are relative to the daytime baseline.
    DIURNAL = {
        0: 1.55, 1: 1.60, 2: 1.60, 3: 1.60, 4: 1.55, 5: 1.45,
        6: 1.25, 7: 1.05, 8: 0.95,
        18: 1.15, 19: 1.30, 20: 1.40, 21: 1.45, 22: 1.50, 23: 1.55,
    }
    diurnal_mult = df_final['hour'].map(DIURNAL).fillna(1.0).astype(float)

    synthetic = (
        df_final['blockface_baseline'] * diurnal_mult
        + np.minimum(25.0, 12.0 * df_final['citations_hourly_median'])
        + 15.0 * df_final['event_intensity']
    ).clip(lower=0, upper=100)

    off_mask = df_final['is_meter_hours'] == 0
    df_final['target_is_estimated'] = off_mask.astype(int)
    df_final.loc[off_mask, 'occupancy_pct'] = synthetic[off_mask].round(2)

    print(f"  Off-hours target: {off_mask.sum():,} rows estimated. "
          f"Off-hours mean occ now {df_final.loc[off_mask, 'occupancy_pct'].mean():.2f}")
    print("  By hour (off-hours only):")
    print(df_final[off_mask].groupby('hour')['occupancy_pct'].mean().round(2).to_string())

    final_cols = [
        'timestamp',
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
        'is_school_day', 'is_raining', 'temperature', 'event_intensity',
        'is_meter_hours', 'target_is_estimated',
        'citation_count', 'citations_hourly_median',
        'neighborhood', 'lat', 'lon', 'total_spaces', 'occupancy_pct'
    ]
    df_final = df_final[final_cols].fillna(0)

    print(f"\nFinal dataset shape: {df_final.shape}")
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"SUCCESS: Saved processed data to {OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
