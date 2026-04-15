"""
ParkCast SF — LightGBM Training Script (Accuracy-Optimized)

Design choices tuned for lowest real-data MAE:
  * Train ONLY on real paid-meter rows (target_is_estimated == 0).
    Synthetic off-hour targets are excluded — they diluted signal and
    produced inflated R² in earlier runs.
  * Temporal 80/20 split on `timestamp` (no shuffle).
  * Block-level historical aggregates added as features, computed from
    training rows only so there's no lookahead:
        - block_mean:          per (lat, lon)
        - block_hour_mean:     per (lat, lon, hour)
        - block_hour_dow_mean: per (lat, lon, hour, day_of_week)
    Unseen blocks in test fall back to global means.
  * `citations_hourly_median` refit on train window only.
  * LightGBM with more trees, lower LR, early stopping on val MAE.
  * No feature derives from the target at its own timestamp.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "processed_training_data.csv")
CIT_PATH = os.path.join(DATA_DIR, "citations_by_block.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "app", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "LightGBM.pkl")
META_PATH = os.path.join(MODEL_DIR, "LightGBM.meta.json")
BLOCK_AGG_PATH = os.path.join(MODEL_DIR, "LightGBM.block_aggs.parquet")

TARGET = "occupancy_pct"
# Baseline = block_hour_dow_mean. The LightGBM model predicts the RESIDUAL
# (y - baseline), so it can only add to the baseline, not replace it. This is
# the only reliable way to beat the raw lookup given the heavy April-vs-earlier
# distribution shift in the 90-day data window.
BASELINE_COL = "block_hour_dow_mean"
FEATURES_NUMERIC = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "is_school_day", "is_raining", "temperature", "event_intensity",
    "citation_count", "citations_hourly_median",
    "lat", "lon", "total_spaces",
    "block_mean", "block_hour_mean",
    "lag_7d", "lag_14d", "lag_28d",
]
FEATURES_CATEGORICAL = ["neighborhood"]
FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL
LAG_HOURS = [7 * 24, 14 * 24, 28 * 24]


def temporal_split(df, frac=0.80):
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_time = df["timestamp"].quantile(frac)
    return df[df["timestamp"] < split_time].copy(), df[df["timestamp"] >= split_time].copy(), split_time


def refit_citations_median(train, test, split_time):
    if not os.path.exists(CIT_PATH):
        return train, test
    cit = pd.read_csv(CIT_PATH, parse_dates=["timestamp"])
    cit = cit[cit["timestamp"] < split_time].copy()
    cit["hour_of_day"] = cit["timestamp"].dt.hour
    cit["weekday"] = cit["timestamp"].dt.weekday
    med = (cit.groupby(["hour_of_day", "weekday"])["citation_count"]
              .median().reset_index(name="cit_median_train"))
    out = []
    for part in (train, test):
        m = part.merge(med, left_on=["hour", "day_of_week"],
                       right_on=["hour_of_day", "weekday"], how="left")
        m["citations_hourly_median"] = m["cit_median_train"].fillna(0.0)
        m = m.drop(columns=["hour_of_day", "weekday", "cit_median_train"],
                   errors="ignore")
        out.append(m)
    return out[0], out[1]


def build_block_aggregates(train):
    """
    Compute per-block historical means from training rows only.
    Holidays are excluded from the baseline means — observed-holiday Mondays
    (MLK, Presidents Day, etc.) were pulling weekday averages down and
    inflating error on regular Mondays by ~1.7 MAE.
    """
    clean = train[train["is_holiday"] == 0]
    global_mean = float(clean[TARGET].mean())

    block_mean = (clean.groupby(["lat", "lon"])[TARGET]
                       .mean().reset_index(name="block_mean"))
    block_hour_mean = (clean.groupby(["lat", "lon", "hour"])[TARGET]
                            .mean().reset_index(name="block_hour_mean"))
    block_hour_dow_mean = (clean.groupby(["lat", "lon", "hour", "day_of_week"])[TARGET]
                                .mean().reset_index(name="block_hour_dow_mean"))
    return {
        "global_mean": global_mean,
        "block_mean": block_mean,
        "block_hour_mean": block_hour_mean,
        "block_hour_dow_mean": block_hour_dow_mean,
    }


def add_lag_features(df):
    """
    For each row at timestamp t, add occupancy_pct at same (lat, lon) at
    t - 7d / 14d / 28d. All lags point into the past — no leakage.
    """
    keys = ["lat", "lon", "timestamp"]
    for lag_h in LAG_HOURS:
        lag_name = f"lag_{lag_h // 24}d"
        shifted = df[["lat", "lon", "timestamp", TARGET]].copy()
        shifted["timestamp"] = shifted["timestamp"] + pd.Timedelta(hours=lag_h)
        shifted = shifted.rename(columns={TARGET: lag_name})
        df = df.merge(shifted, on=keys, how="left")

    return df


def apply_block_aggregates(df, aggs):
    df = df.merge(aggs["block_mean"], on=["lat", "lon"], how="left")
    df = df.merge(aggs["block_hour_mean"], on=["lat", "lon", "hour"], how="left")
    df = df.merge(aggs["block_hour_dow_mean"],
                  on=["lat", "lon", "hour", "day_of_week"], how="left")
    gm = aggs["global_mean"]
    df["block_mean"] = df["block_mean"].fillna(gm)
    df["block_hour_mean"] = df["block_hour_mean"].fillna(df["block_mean"])
    df["block_hour_dow_mean"] = df["block_hour_dow_mean"].fillna(df["block_hour_mean"])
    return df


def report(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    print(f"  {label:<28} n={len(y_true):>8,}  MAE={mae:6.3f}  R²={r2:6.3f}")
    return {"n": int(len(y_true)), "mae": float(mae), "r2": float(r2)}


def main():
    print("=" * 60)
    print("ParkCast SF — LightGBM Training (Accuracy-Optimized)")
    print("=" * 60)

    print("Loading processed data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"  Rows: {len(df):,}  Cols: {df.shape[1]}")

    print("Filtering to real meter-hour rows (target_is_estimated == 0)...")
    df = df[df["target_is_estimated"] == 0].reset_index(drop=True)
    print(f"  Real rows: {len(df):,}")

    print("Adding lag features (7d / 14d / 28d)...")
    df = add_lag_features(df)
    lag_cov = {f"lag_{h//24}d": df[f"lag_{h//24}d"].notna().mean() for h in LAG_HOURS}
    print(f"  Lag coverage: {lag_cov}")

    print("Temporal 80/20 split...")
    train, test, split_time = temporal_split(df, frac=0.80)
    print(f"  Split at: {split_time}")
    print(f"  Train: {len(train):,}   Test: {len(test):,}")

    print("Refitting citations_hourly_median on train window only...")
    train, test = refit_citations_median(train, test, split_time)

    # Inner train/val split BEFORE building block aggregates so the aggregates
    # don't leak inner-val information into the features.
    train = train.sort_values("timestamp").reset_index(drop=True)
    val_cut = int(len(train) * 0.9)
    tr = train.iloc[:val_cut].copy()
    val_df = train.iloc[val_cut:].copy()
    print(f"  Inner split: train={len(tr):,}  val={len(val_df):,}")

    print("Computing block-level historical aggregates from inner-train only...")
    aggs = build_block_aggregates(tr)
    tr = apply_block_aggregates(tr, aggs)
    val_df = apply_block_aggregates(val_df, aggs)
    test = apply_block_aggregates(test, aggs)

    for part in (tr, val_df, test):
        part["neighborhood"] = part["neighborhood"].astype("category")

    # Residual target: y - baseline. Model learns only the deviation.
    X_tr = tr[FEATURES]
    y_tr = tr[TARGET].values - tr[BASELINE_COL].values
    X_val = val_df[FEATURES]
    y_val = val_df[TARGET].values - val_df[BASELINE_COL].values
    X_test = test[FEATURES]
    y_test = test[TARGET].values

    # Upweight sparse-but-high-signal rows so LightGBM actually splits on
    # event_intensity (otherwise 0.1%-positive features get zero splits
    # against dense features like day_of_week). 20× was chosen to make
    # event rows count like ~2% of training, enough to compete.
    EVENT_WEIGHT = 20.0
    w_tr = np.where(tr["event_intensity"].values > 0, EVENT_WEIGHT, 1.0)
    w_val = np.where(val_df["event_intensity"].values > 0, EVENT_WEIGHT, 1.0)
    print(f"  Event rows (train): {(w_tr > 1).sum():,} weighted {EVENT_WEIGHT}×")

    print("Training LightGBM residual model...")
    # Tight regularization: small leaves, high min_child_samples, low LR.
    # Constrains the model to only make small corrections to the baseline.
    model = lgb.LGBMRegressor(
        n_estimators=6000,
        learning_rate=0.008,
        num_leaves=63,
        min_child_samples=100,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective="regression_l1",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        eval_metric="mae",
        categorical_feature=["neighborhood"],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
    )

    # Final prediction = baseline + residual, clipped to valid range.
    residual_pred = model.predict(X_test)
    preds = np.clip(test[BASELINE_COL].values + residual_pred, 0, 100)

    print("\nEvaluation on held-out temporal TEST set (real meter-hour rows):")
    metrics = {}
    baseline_preds = np.clip(test[BASELINE_COL].values, 0, 100)
    metrics["baseline"] = report(y_test, baseline_preds, "baseline: block×hr×dow mean")
    metrics["residual_model"] = report(y_test, preds, "baseline + residual model")

    # Event-row slice so we can see if the sample-weighted training
    # actually moved predictions where it matters.
    ev_mask = test["event_intensity"].values > 0
    if ev_mask.any():
        print("\n  Event-row slice (event_intensity > 0):")
        metrics["baseline_events"] = report(
            y_test[ev_mask], baseline_preds[ev_mask], "baseline (events only)"
        )
        metrics["residual_events"] = report(
            y_test[ev_mask], preds[ev_mask], "residual model (events only)"
        )

    if metrics["baseline"]["mae"] <= metrics["residual_model"]["mae"]:
        print("\n  → Baseline wins.")
        metrics["winner"] = "baseline"
    else:
        print("\n  → Residual model wins.")
        metrics["winner"] = "residual_model"

    print("\nTop feature importances:")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    for name, v in imp.head(15).items():
        print(f"  {name:<28} {int(v):>6}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Save block aggregates so inference can reproduce features
    combined = (aggs["block_hour_dow_mean"]
                .merge(aggs["block_hour_mean"], on=["lat", "lon", "hour"], how="left")
                .merge(aggs["block_mean"], on=["lat", "lon"], how="left"))
    combined.attrs["global_mean"] = aggs["global_mean"]
    combined.to_parquet(BLOCK_AGG_PATH, index=False)

    with open(META_PATH, "w") as f:
        json.dump({
            "features": FEATURES,
            "target": TARGET,
            "split_time": str(split_time),
            "train_rows": int(len(tr)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test)),
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
            "global_mean": aggs["global_mean"],
            "metrics": metrics,
            "note": "Trained only on real meter-hour rows. Off-hour predictions "
                    "are out of scope for this model.",
        }, f, indent=2)
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved aggs:  {BLOCK_AGG_PATH}")
    print(f"Saved meta:  {META_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
