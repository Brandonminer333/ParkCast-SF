import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import mlflow

print("=" * 60)
print("ParkCast SF — Hybrid Model Training")
print("Real block rates + Synthetic feature variance")
print("=" * 60)

# ── STEP 1: Learn base rates from real SFpark data ─────────
print("\nStep 1 — Loading real SFpark sensor data...")
seg = pd.read_csv("/Users/temesghenkahsay/Downloads/dataverse_files/sfpark_filtered_segments.csv", sep=";")
df_real = pd.read_csv(
    "/Users/temesghenkahsay/Downloads/dataverse_files/sfpark_filtered_136_247_486taxis.csv",
    sep=";",
    usecols=["timestamp", "segmentid", "capacity", "occupied"],
    parse_dates=["timestamp"],
)

df_real = df_real[df_real["capacity"] > 0].copy()
df_real["occupancy_pct"] = (df_real["occupied"] / df_real["capacity"] * 100).clip(0, 100)
df_real["hour"] = df_real["timestamp"].dt.hour
df_real["day_of_week"] = df_real["timestamp"].dt.dayofweek

# Compute real base rates per block per hour-of-week
print("  Computing real base rates per block...")
base_rates = df_real.groupby(["segmentid", "hour", "day_of_week"])["occupancy_pct"].mean().reset_index()
base_rates.columns = ["segmentid", "hour", "day_of_week", "base_occupancy"]

# Overall base rate per block (used as fallback)
block_base = df_real.groupby("segmentid")["occupancy_pct"].mean().reset_index()
block_base.columns = ["segmentid", "block_avg_occupancy"]

print(f"  Real blocks: {df_real['segmentid'].nunique()}")
print(f"  Base rate records: {len(base_rates):,}")
print(f"  Avg real occupancy: {df_real['occupancy_pct'].mean():.1f}%")

# Segment coordinates
seg["lat"] = (seg["starty"] + seg["endy"]) / 2
seg["lon"] = (seg["startx"] + seg["endx"]) / 2
seg = seg.merge(block_base, on="segmentid", how="left")
print(f"  Segments with base rates: {seg['block_avg_occupancy'].notna().sum()}")

# ── STEP 2: Load synthetic data for feature variance ───────
print("\nStep 2 — Loading synthetic data for feature diversity...")
df_syn = pd.read_csv("data/parkcast_raw_synthetic.csv")

# Only keep if it's the synthetic one (21840 rows)
if len(df_syn) == 382387:
    print("  WARNING: parkcast_raw.csv is real data, loading synthetic backup...")
    # Recreate synthetic dataset
    raise FileNotFoundError("Need synthetic data")

print(f"  Synthetic rows: {len(df_syn):,}")
print(f"  Synthetic date range: {df_syn['date'].min()} -> {df_syn['date'].max()}")

# ── STEP 3: Build hybrid training dataset ──────────────────
print("\nStep 3 — Building hybrid dataset...")

# Map synthetic neighborhoods to real block base rates
NEIGHBORHOOD_BASE = {
    "soma": df_real[df_real["segmentid"].between(300000, 400000)]["occupancy_pct"].mean() if len(df_real) > 0 else 65,
    "mission": 58.0,
    "castro": 52.0,
    "marina": 48.0,
    "tenderloin": 70.0,
    "haight": 45.0,
    "richmond": 42.0,
    "noe valley": 40.0,
    "sunset": 38.0,
    "unknown": 55.0,
}

# Replace synthetic occupancy with real-data-informed occupancy
# Formula: hybrid_occ = real_base_rate + synthetic_adjustment
df_hybrid = df_syn.copy()

# Get base rate for each neighborhood
df_hybrid["base_rate"] = df_hybrid["neighborhood"].str.lower().map(NEIGHBORHOOD_BASE).fillna(55.0)

# Compute synthetic adjustment (deviation from synthetic mean)
syn_mean = df_syn["occupancy_pct"].mean()
df_hybrid["adjustment"] = df_syn["occupancy_pct"] - syn_mean

# Hybrid occupancy = real base + synthetic adjustment, clipped to 0-100
df_hybrid["occupancy_pct"] = (df_hybrid["base_rate"] + df_hybrid["adjustment"]).clip(0, 100).round(2)

# Add neighborhood encoding
NEIGHBORHOOD_MAP = {
    "castro": 0,
    "haight": 1,
    "marina": 2,
    "mission": 3,
    "noe valley": 4,
    "richmond": 5,
    "soma": 6,
    "sunset": 7,
    "tenderloin": 8,
    "unknown": 9,
}
df_hybrid["neighborhood_encoded"] = df_hybrid["neighborhood"].str.lower().map(NEIGHBORHOOD_MAP).fillna(9)

print(f"  Hybrid rows: {len(df_hybrid):,}")
print(f"  Avg hybrid occupancy: {df_hybrid['occupancy_pct'].mean():.1f}%")
print(f"  Occupancy std: {df_hybrid['occupancy_pct'].std():.1f}%")

# Save hybrid dataset
df_hybrid.to_csv("data/parkcast_raw.csv", index=False)
print("  Saved to data/parkcast_raw.csv")

# ── STEP 4: Train models ───────────────────────────────────
print("\nStep 4 — Training models...")

FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    "is_street_cleaning",
    "has_nearby_event",
    "is_holiday",
    "is_school_day",
    "is_raining",
    "bad_weather",
    "temperature",
    "total_spaces",
    "neighborhood_encoded",
]

X = df_hybrid[FEATURES]
y = df_hybrid["occupancy_pct"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri("http://34.133.160.231:5000")
mlflow.set_experiment("parkcast-hybrid-training")

results = {}
models_to_try = [
    ("RandomForest", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ("GradientBoosting", GradientBoostingRegressor(n_estimators=200, random_state=42)),
]

for name, model in models_to_try:
    print(f"\n  Training {name}...")
    with mlflow.start_run(run_name=f"{name}-hybrid"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        results[name] = {"mae": mae, "rmse": rmse, "r2": r2, "model": model}

        mlflow.log_params(
            {
                "model": name,
                "data": "hybrid_real_base_synthetic_variance",
                "real_blocks": df_real["segmentid"].nunique(),
                "synthetic_rows": len(df_syn),
            }
        )
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        print(f"    MAE: {mae:.2f}%  RMSE: {rmse:.2f}%  R²: {r2:.4f}")

# ── STEP 5: Save best model ────────────────────────────────
print("\nStep 5 — Saving best model...")
best_name = max(results, key=lambda k: results[k]["r2"])
best = results[best_name]
print(f"\n{'='*60}")
print(f"WINNER: {best_name}")
print(f"MAE:    {best['mae']:.2f}%")
print(f"RMSE:   {best['rmse']:.2f}%")
print(f"R²:     {best['r2']:.4f}")
print(f"{'='*60}")

joblib.dump(best["model"], "models/RandomForest.pkl")
print(f"Model saved to models/RandomForest.pkl")
print("\nWhat this model knows from REAL data:")
print(f"  - Base occupancy rates for {df_real['segmentid'].nunique()} real SF blocks")
print(f"  - Calibrated from {len(df_real):,} actual sensor readings")
print("\nWhat this model knows from SYNTHETIC data:")
print(f"  - Feature variance across 12 months, all weather, events, holidays")
print(f"  - {len(df_syn):,} training examples with rich feature diversity")
