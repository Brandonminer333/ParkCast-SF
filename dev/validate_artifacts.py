"""
Pre-upload artifact validation.

Run from project root after dev/refresh_all.sh produces artifacts. Asserts
each expected parquet exists, has plausible row counts, the columns the API
needs, and acceptable NaN rates on the join keys. Exits non-zero on any
failure so the GitHub Actions workflow halts before uploading garbage to GCS.

This is the gate that catches "the notebook ran but produced an empty
parquet" or "schema drifted because someone renamed a column."
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "models")

# Per-artifact spec: minimum rows, columns the API merges on, max NaN rate
# on those keys (anything higher means the join would silently miss).
EXPECTED: dict[str, dict[str, Any]] = {
    "blocks.parquet": {
        "min_rows": 12_000,
        "required_cols": ["lat", "lon", "neighborhood", "total_spaces", "coverage", "cnn"],
        "max_nan": {"lat": 0.0, "lon": 0.0, "cnn": 0.05, "neighborhood": 0.0},
    },
    "LightGBM.pkl": {"file_only": True, "min_size_kb": 100},
    "LightGBM.meta.json": {"file_only": True, "min_size_kb": 0},
    "LightGBM.block_aggs.parquet": {
        "min_rows": 50_000,
        "required_cols": ["lat", "lon", "hour", "day_of_week", "block_hour_dow_mean"],
        "max_nan": {"lat": 0.0, "lon": 0.0, "block_hour_dow_mean": 0.0},
    },
    # Held-out test set saved pre-block_aggs by train_lightgbm.ipynb so the
    # promotion gate can rescore the incumbent on the same rows. Local-only;
    # not uploaded to GCS.
    "LightGBM.test_set.parquet": {
        "min_rows": 100_000,
        "required_cols": ["lat", "lon", "hour", "day_of_week", "neighborhood",
                          "occupancy_pct"],
        "max_nan": {"lat": 0.0, "lon": 0.0, "occupancy_pct": 0.0},
    },
    "lag_history.parquet": {
        "min_rows": 50_000,
        "required_cols": ["lat", "lon", "timestamp", "occupancy_pct"],
        "max_nan": {"lat": 0.0, "lon": 0.0, "timestamp": 0.0},
    },
    "citations_hourly_median.parquet": {
        "min_rows": 100,
        "required_cols": ["hour", "day_of_week", "citations_hourly_median"],
        "max_nan": {},
    },
    "block_static_features.parquet": {
        "min_rows": 1_000,
        "required_cols": ["lat", "lon"],
        "max_nan": {"lat": 0.0, "lon": 0.0},
    },
    "inferred_block_aggs.parquet": {
        "min_rows": 500_000,
        "required_cols": ["cnn", "hour", "day_of_week", "block_hour_dow_mean", "block_hour_mean", "block_mean"],
        "max_nan": {"cnn": 0.0, "block_hour_dow_mean": 0.0},
    },
}


def check(path: str, spec: dict[str, Any]) -> list[str]:
    fails: list[str] = []
    full = os.path.join(MODEL_DIR, path)
    if not os.path.exists(full):
        return [f"{path}: MISSING"]
    if spec.get("file_only"):
        size_kb = os.path.getsize(full) / 1024
        if size_kb < spec.get("min_size_kb", 0):
            fails.append(f"{path}: size {size_kb:.1f}KB < {spec['min_size_kb']}KB expected")
        return fails
    df = pd.read_parquet(full)
    if len(df) < spec["min_rows"]:
        fails.append(f"{path}: {len(df):,} rows < {spec['min_rows']:,} expected")
    missing = set(spec["required_cols"]) - set(df.columns)
    if missing:
        fails.append(f"{path}: missing columns {sorted(missing)}")
    for col, max_rate in spec.get("max_nan", {}).items():
        if col in df.columns:
            rate = df[col].isna().mean()
            if rate > max_rate:
                fails.append(f"{path}: {col} NaN rate {rate*100:.2f}% > {max_rate*100:.2f}% allowed")
    return fails


def main() -> int:
    failures: list[str] = []
    for name, spec in EXPECTED.items():
        result = check(name, spec)
        status = "FAIL" if result else "OK"
        print(f"  [{status}] {name}")
        for f in result:
            print(f"         - {f}")
        failures.extend(result)
    print()
    if failures:
        print(f"Validation FAILED ({len(failures)} issue(s)).")
        return 1
    print(f"Validation OK ({len(EXPECTED)} artifacts).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
