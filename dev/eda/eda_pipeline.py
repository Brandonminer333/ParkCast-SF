"""
Streamlined EDA for SFpark-style SODA JSON endpoints.

Workflow:
1. Fetch data (with local caching + pagination)
2. Coerce types (numeric + datetime inference)
3. Generate summary tables and plots
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


@dataclass(frozen=True)
class EdaConfig:
    # SFpark public API example:
    # https://dev.socrata.com/foundry/data.sfgov.org/8vzz-qzz9
    url: str = "https://data.sfgov.org/resource/8vzz-qzz9.json"
    dataset_name: str = "sfpark_sensors"

    # Pagination controls for SODA-style endpoints.
    # SODA typically supports $limit and $offset.
    limit: int = 5000
    max_pages: int = 5

    # Query params besides $limit/$offset.
    # Example: {"$where": "avail > 0"}
    extra_params: Optional[Dict[str, Any]] = None

    # Where outputs go.
    output_dir: str = "outputs/eda"
    cache_dir: str = "outputs/cache"

    # If True, prefer cached raw JSON when present.
    use_cache: bool = True


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _cache_key(url: str, params: Dict[str, Any]) -> str:
    key_material = {"url": url, "params": params}
    return hashlib.sha256(_stable_json_dumps(key_material).encode("utf-8")).hexdigest()[:16]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _coerce_numeric(df: pd.DataFrame, min_fraction: float = 0.7) -> Tuple[pd.DataFrame, List[str]]:
    """
    Try to convert object/string columns to numeric when coercion succeeds for a large share.
    """
    df = df.copy()
    converted: List[str] = []
    for col in df.columns:
        s = df[col]
        if s.dtype == "object":
            numeric = pd.to_numeric(s, errors="coerce")
            fraction = numeric.notna().mean() if len(numeric) else 0.0
            if fraction >= min_fraction:
                df[col] = numeric
                converted.append(col)
    return df, converted


def _coerce_datetime(df: pd.DataFrame, min_fraction: float = 0.7) -> Tuple[pd.DataFrame, List[str]]:
    """
    Try to convert likely timestamp-like columns to datetime.
    """
    df = df.copy()
    dt_cols: List[str] = []
    candidates: List[str] = [
        c for c in df.columns if any(k in c.lower() for k in ("time", "date", "timestamp", "updated", "created"))
    ]
    for col in candidates:
        s = df[col]
        if s.dtype != "object":
            continue
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        fraction = dt.notna().mean() if len(dt) else 0.0
        if fraction >= min_fraction:
            df[col] = dt
            dt_cols.append(col)
    return df, dt_cols


def _infer_availability_columns(df: pd.DataFrame) -> List[str]:
    # Heuristics: try common keywords.
    patterns = ("avail", "occup", "space", "spaces",
                "capacity", "capacity_", "count")
    cols: List[str] = []
    for c in df.columns:
        cl = c.lower()
        if any(p in cl for p in patterns):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    # Prefer tighter matches if available.
    avail_like = [c for c in cols if any(
        k in c.lower() for k in ("avail", "occup"))]
    return avail_like if avail_like else cols


def fetch_sfpark_json(cfg: EdaConfig, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    Fetch data from a SODA endpoint with basic pagination and local JSON caching.
    """
    _ensure_dir(cfg.cache_dir)
    session = session or requests.Session()

    extra_params = dict(cfg.extra_params or {})
    params_base: Dict[str, Any] = dict(extra_params)
    params_base["$limit"] = cfg.limit

    cache_params = {**params_base}  # Note: offset handled per-page.
    cache_params_key = dict(cache_params)
    cache_path = os.path.join(
        cfg.cache_dir,
        f"{cfg.dataset_name}_{_cache_key(cfg.url, cache_params_key)}_limit{cfg.limit}_pages{cfg.max_pages}.json",
    )

    if cfg.use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return pd.DataFrame(raw)

    all_rows: List[Dict[str, Any]] = []
    offset = 0
    for page in range(cfg.max_pages):
        params = dict(params_base)
        params["$offset"] = offset
        resp = session.get(cfg.url, params=params, timeout=60)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        # Ensure dict rows (SODA should return list of objects)
        all_rows.extend(rows)
        offset += cfg.limit

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, default=str)

    return pd.DataFrame(all_rows)


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Coerce numeric + datetime, then lightly normalize column names.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    before_cols = list(df.columns)
    # Coerce numeric first so we can use dtype-based heuristics later.
    df, numeric_cols = _coerce_numeric(df)
    df, dt_cols = _coerce_datetime(df)

    # Drop columns that are entirely missing.
    df = df.dropna(axis=1, how="all")

    meta = {
        "before_columns": before_cols,
        "numeric_columns": numeric_cols,
        "datetime_columns": dt_cols,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
    }
    return df, meta


def _save_table(df: pd.DataFrame, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def _save_series(s: pd.Series, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    s.to_csv(path)


def _plot_hist(df: pd.DataFrame, col: str, path: str, bins: int = 30) -> None:
    plt.figure(figsize=(8, 5))
    data = df[col].dropna().astype(float)
    plt.hist(data, bins=bins, color="#4C78A8", alpha=0.9)
    plt.title(f"Distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    _ensure_dir(os.path.dirname(path))
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_time_buckets(
    df: pd.DataFrame,
    time_col: str,
    y_cols: Iterable[str],
    path_prefix: str,
    freq: str = "30min",
) -> None:
    """
    Aggregate y_cols by time buckets and plot lines for quick drift checks.
    """
    df = df.copy()
    df = df.dropna(subset=[time_col])
    if df.empty:
        return

    df["__bucket__"] = df[time_col].dt.floor(freq)
    for y in y_cols:
        if y not in df.columns or not pd.api.types.is_numeric_dtype(df[y]):
            continue
        g = df.groupby("__bucket__")[y].mean().sort_index()
        if g.empty:
            continue

        plt.figure(figsize=(9, 5))
        plt.plot(g.index.to_pydatetime(), g.values, color="#F58518")
        plt.title(f"Mean {y} by {freq} buckets")
        plt.xlabel("Time")
        plt.ylabel(f"Mean {y}")
        plt.tight_layout()
        _ensure_dir(os.path.dirname(path_prefix))
        plt.savefig(f"{path_prefix}_{y}_by_{freq}.png", dpi=150)
        plt.close()


def run_eda(cfg: EdaConfig) -> Dict[str, Any]:
    """
    Fetch -> clean -> generate summary outputs.
    """
    start = datetime.utcnow()
    _ensure_dir(cfg.output_dir)

    df_raw = fetch_sfpark_json(cfg)
    df, meta = clean_dataframe(df_raw)

    # Save data snapshots for reproducibility.
    raw_path = os.path.join(cfg.output_dir, "raw_sample.parquet")
    clean_path = os.path.join(cfg.output_dir, "clean_sample.parquet")
    # We only attempt parquet if pandas has a backend; otherwise fall back to CSV.
    try:
        df_raw.head(20000).to_parquet(raw_path, index=False)
        df.head(20000).to_parquet(clean_path, index=False)
    except Exception:
        raw_path = os.path.join(cfg.output_dir, "raw_sample.csv")
        clean_path = os.path.join(cfg.output_dir, "clean_sample.csv")
        df_raw.head(20000).to_csv(raw_path, index=False)
        df.head(20000).to_csv(clean_path, index=False)

    # Missingness + types.
    missing = (df.isna().mean().sort_values(
        ascending=False)).rename("missing_fraction")
    _save_series(missing, os.path.join(cfg.output_dir, "missingness.csv"))
    _save_table(
        pd.DataFrame({"column": df.columns, "dtype": [
                     str(t) for t in df.dtypes.values]}),
        os.path.join(cfg.output_dir, "dtypes.csv"),
    )

    numeric_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        desc = df[numeric_cols].describe().T.reset_index().rename(
            columns={"index": "column"})
        # Round to keep diffs stable across platforms.
        for col in desc.columns:
            if col != "column":
                desc[col] = desc[col].astype(float).round(6)
        _save_table(desc, os.path.join(cfg.output_dir, "numeric_describe.csv"))

    # Availability/occupancy distributions (if any).
    availability_cols = _infer_availability_columns(df)
    plots_dir = os.path.join(cfg.output_dir, "plots")
    for c in availability_cols[:6]:
        try:
            _plot_hist(df, c, os.path.join(plots_dir, f"hist_{c}.png"))
        except Exception:
            pass

    # Time series drift plots (if any).
    dt_cols = meta["datetime_columns"]
    # Choose the datetime column with the most non-null values.
    time_col = None
    if dt_cols:
        time_col = max(dt_cols, key=lambda c: df[c].notna().mean())

    if time_col and availability_cols and len(availability_cols) > 0:
        _plot_time_buckets(
            df,
            time_col=time_col,
            y_cols=availability_cols[:6],
            path_prefix=os.path.join(plots_dir, "time_mean"),
        )

    # Correlations for numeric subset.
    if len(numeric_cols) >= 2:
        use_cols = numeric_cols[:20]
        corr = df[use_cols].corr(numeric_only=True)
        _save_table(corr.reset_index().rename(
            columns={"index": "column"}), os.path.join(cfg.output_dir, "correlation.csv"))

    end = datetime.utcnow()
    meta.update(
        {
            "started_utc": start.isoformat(),
            "finished_utc": end.isoformat(),
            "output_dir": cfg.output_dir,
            "availability_columns": availability_cols,
            "time_col_used": time_col,
        }
    )
    return meta


def main() -> None:
    cfg = EdaConfig()
    meta = run_eda(cfg)
    print("EDA complete.")
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
