"""ModelBundle: loads the LightGBM model + lookup parquets from disk/GCS.

The monolithic ``__init__`` from the original main.py is decomposed into
focused private methods so each responsibility is isolated and testable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

# ── Artifact location ──────────────────────────────────────────────
GCS_BUCKET = os.getenv("GCS_BUCKET", "").strip()
GCS_PREFIX = os.getenv("GCS_PREFIX", "").strip()
if GCS_PREFIX and not GCS_PREFIX.endswith("/"):
    GCS_PREFIX += "/"

LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CACHE_DIR = os.getenv("PARKCAST_CACHE_DIR", "/tmp/parkcast_models")

ARTIFACT_FILES = [
    "LightGBM.pkl",
    "LightGBM.block_aggs.parquet",
    "LightGBM.meta.json",
    "blocks.parquet",
    "lag_history.parquet",
    "citations_hourly_median.parquet",
    "sfpark_calibration.parquet",
    "block_static_features.parquet",
]

# Optional: enriches blocks with corridor/limits and provides per-cnn
# fallback baselines for inferred (non-metered) blocks.
OPTIONAL_ARTIFACT_FILES = [
    "master_blocks.parquet",
    "inferred_block_aggs.parquet",
]


# ── GCS helpers ────────────────────────────────────────────────────
def _download_from_gcs(
    bucket: str,
    prefix: str,
    files: List[str],
    dest: str,
    optional: bool = False,
) -> None:
    from google.cloud import storage  # lazy — local dev doesn't need it

    os.makedirs(dest, exist_ok=True)
    client = storage.Client()
    b = client.bucket(bucket)
    for f in files:
        key = f"{prefix}{f}"
        try:
            b.blob(key).download_to_filename(os.path.join(dest, f))
            logger.info(f"  downloaded gs://{bucket}/{key}")
        except Exception as e:
            if optional:
                logger.info(f"  skipped optional gs://{bucket}/{key}: {e}")
            else:
                raise


def resolve_model_dir() -> str:
    """Return the directory containing model artifacts (GCS or local)."""
    if GCS_BUCKET:
        try:
            logger.info(f"Fetching artifacts from gs://{GCS_BUCKET}/{GCS_PREFIX} …")
            _download_from_gcs(GCS_BUCKET, GCS_PREFIX, ARTIFACT_FILES, CACHE_DIR)
            _download_from_gcs(
                GCS_BUCKET, GCS_PREFIX, OPTIONAL_ARTIFACT_FILES, CACHE_DIR, optional=True
            )
            return CACHE_DIR
        except Exception as e:  # noqa: BLE001
            logger.warning(f"GCS download failed ({e}); falling back to {LOCAL_MODEL_DIR}")
    else:
        logger.info(f"GCS_BUCKET unset; loading from {LOCAL_MODEL_DIR}")
    return LOCAL_MODEL_DIR


# ── ModelBundle ────────────────────────────────────────────────────
class ModelBundle:
    """Model + every lookup parquet inference needs, loaded once at startup.

    The constructor is decomposed into focused private methods so each
    responsibility (model loading, parquet loading, street enrichment,
    category dtype) is isolated.
    """

    def __init__(self, src: str):
        self.src = src
        self._load_model_and_meta(src)
        self._load_parquets(src)
        self._backfill_coverage()
        self._enrich_street_labels(src)
        self._drop_unlabeled_blocks()
        self._build_neighborhood_dtype()
        logger.info(
            f"ModelBundle loaded from {src}: "
            f"{len(self.blocks):,} blocks · {len(self.features)} features"
        )

    # ── Private init steps ─────────────────────────────────────────

    def _load_model_and_meta(self, src: str) -> None:
        """Load the serialized model and its metadata JSON."""
        self.model = joblib.load(os.path.join(src, "LightGBM.pkl"))
        with open(os.path.join(src, "LightGBM.meta.json")) as f:
            self.meta = json.load(f)
        self.features: List[str] = self.meta["features"]
        self.global_mean: float = float(self.meta.get("global_mean", 45.0))

    def _load_parquets(self, src: str) -> None:
        """Load all lookup parquets into DataFrames."""
        self.blocks = pd.read_parquet(os.path.join(src, "blocks.parquet"))
        self.block_aggs = pd.read_parquet(os.path.join(src, "LightGBM.block_aggs.parquet"))
        self.static = pd.read_parquet(os.path.join(src, "block_static_features.parquet"))
        self.cit_med = pd.read_parquet(os.path.join(src, "citations_hourly_median.parquet"))

        # Per-cnn baselines for inferred blocks (optional).
        inferred_path = os.path.join(src, "inferred_block_aggs.parquet")
        self.inferred_aggs: Optional[pd.DataFrame] = (
            pd.read_parquet(inferred_path) if os.path.exists(inferred_path) else None
        )

        # Per-(block, hour, dow) lag values precomputed from lag_history.
        # The feature builder previously set lag columns to NaN at inference,
        # which made the residual head add noise instead of signal — served
        # MAE ended up worse than the bare block×hour×dow baseline.
        self.lag_lookup: Optional[pd.DataFrame] = self._build_lag_lookup(src)

    @staticmethod
    def _build_lag_lookup(src: str) -> Optional[pd.DataFrame]:
        """Precompute every lag feature the model uses, per (lat, lon, hour, dow).

        Point lags (lag_1d, lag_2d, lag_7d, lag_14d, lag_28d) and rolling
        means (lag_3d_mean = mean of lag_{1..3}d, lag_7d_mean = mean of
        lag_{1..7}d) are all derived. Filling the rolling means is critical:
        if the model trained with them and inference leaves them NaN while
        the point lags are real, the mixed missingness sends the model down
        unfamiliar surrogate paths and predictions collapse toward the mean.

        For each slot, the anchor is the most-recent matching timestamp in
        ``lag_history.parquet``; lag_kd is the block's occupancy at
        ``anchor - k days``. Anchoring on (hour, dow) — rather than on a true
        target timestamp the request schema doesn't carry — keeps the lag
        semantics close to training (point-in-time block occupancy) without
        requiring API changes.
        """
        path = os.path.join(src, "lag_history.parquet")
        if not os.path.exists(path):
            return None
        lh = pd.read_parquet(path)
        lh["hour"] = lh["timestamp"].dt.hour
        lh["day_of_week"] = lh["timestamp"].dt.weekday

        anchor = (
            lh.sort_values("timestamp")
            .groupby(["lat", "lon", "hour", "day_of_week"], as_index=False)
            .tail(1)
            .rename(columns={"timestamp": "anchor_ts"})[
                ["lat", "lon", "hour", "day_of_week", "anchor_ts"]
            ]
        )

        # Compute days 1..7 (needed for lag_7d_mean), plus 14 and 28.
        all_days = list(range(1, 8)) + [14, 28]
        out = anchor
        for d in all_days:
            shifted = lh[["lat", "lon", "timestamp", "occupancy_pct"]].copy()
            shifted["anchor_ts"] = shifted["timestamp"] + pd.Timedelta(days=d)
            shifted = shifted.rename(columns={"occupancy_pct": f"lag_{d}d"})[
                ["lat", "lon", "anchor_ts", f"lag_{d}d"]
            ]
            out = out.merge(shifted, on=["lat", "lon", "anchor_ts"], how="left")

        out["lag_3d_mean"] = out[[f"lag_{d}d" for d in (1, 2, 3)]].mean(axis=1)
        out["lag_7d_mean"] = out[[f"lag_{d}d" for d in range(1, 8)]].mean(axis=1)

        keep = [
            "lat",
            "lon",
            "hour",
            "day_of_week",
            "lag_1d",
            "lag_2d",
            "lag_7d",
            "lag_14d",
            "lag_28d",
            "lag_3d_mean",
            "lag_7d_mean",
        ]
        return out[keep]

    def _backfill_coverage(self) -> None:
        """Older metered-only bundles lack the ``coverage`` column."""
        if "coverage" not in self.blocks.columns:
            self.blocks["coverage"] = "metered"

    def _enrich_street_labels(self, src: str) -> None:
        """Merge corridor/limits from master_blocks to build street labels."""
        self.blocks["street"] = None
        master_path = os.path.join(src, "master_blocks.parquet")
        if not (os.path.exists(master_path) and "cnn" in self.blocks.columns):
            return
        try:
            master = pd.read_parquet(master_path)[["cnn", "corridor", "limits"]].dropna(
                subset=["cnn"]
            )
            master["cnn"] = master["cnn"].astype("Int64")
            blocks_cnn = self.blocks["cnn"].astype("Int64")
            lookup = master.set_index("cnn")
            corridors = blocks_cnn.map(lookup["corridor"])
            limits_s = blocks_cnn.map(lookup["limits"])
            self.blocks["street"] = [
                self._format_street(c, lim) for c, lim in zip(corridors, limits_s)
            ]
        except Exception as e:
            logger.warning(f"master_blocks.parquet unreadable ({e}); all blocks will be dropped")

    def _drop_unlabeled_blocks(self) -> None:
        """Remove blocks without a real street label from the served catalog."""
        before = len(self.blocks)
        self.blocks = self.blocks[self.blocks["street"].notna()].reset_index(drop=True)
        dropped = before - len(self.blocks)
        logger.info(
            f"  street enrichment: kept {len(self.blocks):,}/{before:,} blocks, "
            f"dropped {dropped} without a real street label"
        )

    def _build_neighborhood_dtype(self) -> None:
        """Freeze the neighborhood category set to what training saw."""
        neighborhoods = sorted(self.blocks["neighborhood"].dropna().unique().tolist())
        self.neighborhood_dtype = pd.CategoricalDtype(categories=neighborhoods)

    @staticmethod
    def _format_street(corridor, limits) -> Optional[str]:
        """Build a human-readable street label from corridor + limits."""
        if pd.notna(corridor) and pd.notna(limits):
            return f"{corridor} ({limits})"
        if pd.notna(corridor):
            return str(corridor)
        return None
