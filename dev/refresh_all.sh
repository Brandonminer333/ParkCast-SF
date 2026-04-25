#!/bin/bash
# Full refresh: ingest today's data, rebuild training CSV, rebuild inference
# parquets, retrain LightGBM, upload artifacts to GCS, force Cloud Run
# cold-start.
#
# Model guard: the new .pkl/meta/block_aggs only get uploaded if the new
# test MAE is at least as good as the currently-deployed model. Data files
# (lag_history, blocks, citations, static features) always upload — they're
# feature lookups, not the trained model itself.
#
# Usage:  bash dev/refresh_all.sh [gs://bucket/prefix/]
# Default: gs://parkcast-bucket/Data/
#
# Runtime: roughly 30-45 min end-to-end (download dominates).

set -euo pipefail

GCS_TARGET="${1:-gs://parkcast-bucket/Data/}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { echo ""; echo "── $* ──"; }

echo "=========================================="
echo "ParkCast SF — Full Refresh"
echo "  Project: $PROJECT_DIR"
echo "  GCS:     $GCS_TARGET"
echo "=========================================="

cd "$PROJECT_DIR/dev"

log "1/8 Fetching SFMTA meter transactions (download_12mo_transactions.ipynb)"
jupyter nbconvert --to notebook --execute --inplace download_12mo_transactions.ipynb \
  --ExecutePreprocessor.timeout=3600

log "2/8 Fetching 311 parking complaints (fetch_311.ipynb)"
jupyter nbconvert --to notebook --execute --inplace fetch_311.ipynb \
  --ExecutePreprocessor.timeout=1200

log "3/8 Fetching events calendar (fetch_events_ics.ipynb)"
jupyter nbconvert --to notebook --execute --inplace fetch_events_ics.ipynb \
  --ExecutePreprocessor.timeout=600

log "4/8 Rebuilding training CSV (preprocess_real_data.ipynb)"
jupyter nbconvert --to notebook --execute --inplace preprocess_real_data.ipynb \
  --ExecutePreprocessor.timeout=3600

log "5/8 Rebuilding inference parquets (build_inference_assets.ipynb)"
jupyter nbconvert --to notebook --execute --inplace build_inference_assets.ipynb \
  --ExecutePreprocessor.timeout=600

log "6/8 Rebuilding block_static_features.parquet (build_inference_parquets.ipynb)"
jupyter nbconvert --to notebook --execute --inplace build_inference_parquets.ipynb \
  --ExecutePreprocessor.timeout=600

log "7/8 Retraining LightGBM (train_lightgbm.ipynb)"
jupyter nbconvert --to notebook --execute --inplace train_lightgbm.ipynb \
  --ExecutePreprocessor.timeout=5400

cd "$PROJECT_DIR"

log "8/8 Uploading artifacts"

# Always upload data/feature files — they're lookups, not a trained model.
echo "  [data] lag_history.parquet, blocks.parquet, citations_hourly_median.parquet, block_static_features.parquet, sfpark_calibration.parquet"
gsutil -m cp \
  app/models/lag_history.parquet \
  app/models/blocks.parquet \
  app/models/citations_hourly_median.parquet \
  app/models/block_static_features.parquet \
  app/models/sfpark_calibration.parquet \
  "$GCS_TARGET"

# Guard the model upload — only overwrite if the new run is at least as
# good as the one currently in GCS.
INCUMBENT_META=$(mktemp)
trap 'rm -f "$INCUMBENT_META"' EXIT
gsutil cp "${GCS_TARGET}LightGBM.meta.json" "$INCUMBENT_META" 2>/dev/null || true

CHALLENGER_MAE=$(python3 -c "
import json
d = json.load(open('app/models/LightGBM.meta.json'))
print(d['metrics']['residual_model']['mae'])
")

INCUMBENT_MAE=$(python3 -c "
import json, os, sys
p = '$INCUMBENT_META'
if not os.path.exists(p) or os.path.getsize(p) == 0:
    print('inf')
    sys.exit(0)
try:
    d = json.load(open(p))
    print(d['metrics']['residual_model']['mae'])
except Exception:
    print('inf')
")

echo ""
echo "  incumbent (GCS) MAE: $INCUMBENT_MAE"
echo "  challenger     MAE: $CHALLENGER_MAE"

SHOULD_UPLOAD=$(python3 -c "
incumbent = float('$INCUMBENT_MAE')
challenger = float('$CHALLENGER_MAE')
# Allow a tiny tolerance so ties upload (keeps MLflow run_id fresh in meta).
print('1' if challenger <= incumbent + 1e-4 else '0')
")

if [ "$SHOULD_UPLOAD" = "1" ]; then
  echo "  ✓ model improved (or matched) — uploading .pkl, .meta.json, .block_aggs.parquet"
  gsutil -m cp \
    app/models/LightGBM.pkl \
    app/models/LightGBM.meta.json \
    app/models/LightGBM.block_aggs.parquet \
    "$GCS_TARGET"
else
  echo "  ⚠ model REGRESSED (challenger > incumbent) — keeping existing GCS model"
  echo "    Data files were still refreshed, so lags and static features are current."
fi

log "Forcing Cloud Run cold-start so new files get re-downloaded"
gcloud run services update parkcast-api \
  --region=us-central1 \
  --update-env-vars="REFRESH_TS=$(date +%s)" \
  --project=parkcast >/dev/null
echo "  new revision deployed"

echo ""
echo "=========================================="
echo "✓ FULL REFRESH COMPLETE"
echo "=========================================="
echo ""
echo "Verify:"
echo "  curl -s https://parkcast-api-904706413856.us-central1.run.app/health | python3 -m json.tool"
