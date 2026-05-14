"""Re-score the GCS incumbent model on the challenger's held-out test set.

The promotion gate used to compare the challenger's freshly-computed MAE
against the incumbent's *stored* MAE — but each model is scored on its own
temporal 80/20 split, so the test windows differ and the comparison drifts
as new data arrives. This script re-runs the incumbent on the same rows the
challenger was just scored on, yielding an apples-to-apples MAE.

Inputs:
    - challenger test set parquet (saved by train_lightgbm.ipynb pre-block_aggs)
    - incumbent .pkl, meta.json, block_aggs.parquet (downloaded from GCS)

Output:
    JSON file shaped like LightGBM.meta.json with the rescored MAE under
    metrics.residual_model.mae — drop-in replacement for promote_decision.py.

If any incumbent artifact is missing, exits 0 without writing the output.
The caller (refresh_all.sh) treats a missing rescored meta as "no incumbent
to compare against" and falls through to the existing lenient logic in
promote_decision.py (incumbent = +inf → challenger uploads).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

TARGET = "occupancy_pct"
BASELINE_COL = "block_hour_dow_mean"


def _apply_block_aggregates(df: pd.DataFrame, combined: pd.DataFrame,
                            global_mean: float) -> pd.DataFrame:
    """Re-apply block aggregates with the same 3-merge + fillna chain the
    training notebook uses, so the incumbent sees the same feature shape it
    was trained with."""
    block_mean = combined[["lat", "lon", "block_mean"]].drop_duplicates(
        subset=["lat", "lon"])
    block_hour_mean = combined[["lat", "lon", "hour", "block_hour_mean"]].drop_duplicates(
        subset=["lat", "lon", "hour"])
    block_hour_dow_mean = combined[
        ["lat", "lon", "hour", "day_of_week", "block_hour_dow_mean"]]

    df = df.merge(block_mean, on=["lat", "lon"], how="left")
    df = df.merge(block_hour_mean, on=["lat", "lon", "hour"], how="left")
    df = df.merge(block_hour_dow_mean,
                  on=["lat", "lon", "hour", "day_of_week"], how="left")
    df["block_mean"] = df["block_mean"].fillna(global_mean)
    df["block_hour_mean"] = df["block_hour_mean"].fillna(df["block_mean"])
    df["block_hour_dow_mean"] = df["block_hour_dow_mean"].fillna(df["block_hour_mean"])
    return df


def rescore(test_path: str, pkl_path: str, meta_path: str,
            aggs_path: str) -> float:
    test = pd.read_parquet(test_path)
    with open(meta_path) as f:
        meta = json.load(f)
    features = meta["features"]
    global_mean = float(meta["global_mean"])

    combined = pd.read_parquet(aggs_path)
    test = _apply_block_aggregates(test, combined, global_mean)
    test["neighborhood"] = test["neighborhood"].astype("category")

    model = joblib.load(pkl_path)
    residual = model.predict(test[features])
    preds = np.clip(test[BASELINE_COL].values + residual, 0, 100)
    return float(mean_absolute_error(test[TARGET].values, preds))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--challenger-test", required=True,
                   help="parquet of test rows pre-block_aggs (from train_lightgbm.ipynb)")
    p.add_argument("--incumbent-pkl", required=True)
    p.add_argument("--incumbent-meta", required=True)
    p.add_argument("--incumbent-aggs", required=True)
    p.add_argument("--out", required=True,
                   help="meta-shaped JSON to write the rescored MAE into")
    args = p.parse_args(argv)

    for path in (args.challenger_test, args.incumbent_pkl,
                 args.incumbent_meta, args.incumbent_aggs):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"score_incumbent: missing {path} — skipping rescore",
                  file=sys.stderr)
            return 0

    mae = rescore(args.challenger_test, args.incumbent_pkl,
                  args.incumbent_meta, args.incumbent_aggs)
    out = {
        "metrics": {"residual_model": {"mae": mae}},
        "rescored_on_challenger_test_set": True,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  rescored incumbent MAE on challenger test set: {mae:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
