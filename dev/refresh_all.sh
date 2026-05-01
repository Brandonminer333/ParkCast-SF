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

log "0/12 Fetching every dataset fresh from DataSF (single source of truth)"
# No bootstrap fallback: if a SODA fetch fails, the pipeline halts. We'd
# rather skip a refresh cycle than ship stale or partial data downstream.
SODA_BASE="https://data.sfgov.org/resource"
mkdir -p "$PROJECT_DIR/data"
fetch() {
  # fetch <filename> <socrata-id> <limit>
  local out="$PROJECT_DIR/data/$1"
  local id="$2"
  local limit="${3:-100000}"
  local ext="${1##*.}"
  local url="$SODA_BASE/$id.$ext?\$limit=$limit"
  echo "  $1 ($id, limit=$limit)"
  if ! curl -fsSL --max-time 1200 -o "$out.tmp" "$url"; then
    rm -f "$out.tmp"
    echo "ERROR: SODA fetch failed for $1 ($url)" >&2
    exit 1
  fi
  if [ ! -s "$out.tmp" ]; then
    rm -f "$out.tmp"
    echo "ERROR: SODA returned empty body for $1" >&2
    exit 1
  fi
  mv "$out.tmp" "$out"
}
fetch parking_census.json       9ivs-nf5y    50000
fetch street_sweeping.csv       yhqp-riqs    50000
fetch parking_regulations.csv   hi6h-neyh    50000
fetch rpp_parcels.json          i886-hxz9   100000
fetch sf_centerlines.json       3psu-pn9h    50000
fetch meter_locations.csv       8vzz-qzz9   100000
fetch parking_citations.csv     ab4h-6ztd  3000000

cd "$PROJECT_DIR/dev"

log "1/12 Fetching SFMTA meter transactions (download_12mo_transactions.ipynb)"
jupyter nbconvert --to notebook --execute --inplace download_12mo_transactions.ipynb \
  --ExecutePreprocessor.timeout=3600

log "2/12 Fetching 311 parking complaints (fetch_311.ipynb)"
jupyter nbconvert --to notebook --execute --inplace fetch_311.ipynb \
  --ExecutePreprocessor.timeout=1200

log "3/12 Fetching events calendar (fetch_events_ics.ipynb)"
jupyter nbconvert --to notebook --execute --inplace fetch_events_ics.ipynb \
  --ExecutePreprocessor.timeout=600

log "4/12 Rebuilding master_blocks (build_master_blocks.ipynb)"
# Static-data drives this — must rerun whenever the SODA fetches above
# updated the underlying files.
jupyter nbconvert --to notebook --execute --inplace build_master_blocks.ipynb \
  --ExecutePreprocessor.timeout=900

log "5/12 Rebuilding training CSV (preprocess_real_data.ipynb)"
jupyter nbconvert --to notebook --execute --inplace preprocess_real_data.ipynb \
  --ExecutePreprocessor.timeout=3600

log "6/12 Rebuilding metered inference parquets (build_inference_assets.ipynb)"
jupyter nbconvert --to notebook --execute --inplace build_inference_assets.ipynb \
  --ExecutePreprocessor.timeout=600

log "7/12 Rebuilding block_static_features.parquet (build_inference_parquets.ipynb)"
jupyter nbconvert --to notebook --execute --inplace build_inference_parquets.ipynb \
  --ExecutePreprocessor.timeout=600

# build_full_blocks.ipynb reads the metered catalog from a .bak so it knows
# which blocks were metered before it overwrites blocks.parquet with the
# citywide version.
cp "$PROJECT_DIR/app/models/blocks.parquet" "$PROJECT_DIR/app/models/blocks.parquet.bak"

log "8/12 Building citywide blocks + KNN inferred_block_aggs (build_full_blocks.ipynb)"
jupyter nbconvert --to notebook --execute --inplace build_full_blocks.ipynb \
  --ExecutePreprocessor.timeout=600

log "9/12 Retraining LightGBM (train_lightgbm.ipynb)"
jupyter nbconvert --to notebook --execute --inplace train_lightgbm.ipynb \
  --ExecutePreprocessor.timeout=14400

cd "$PROJECT_DIR"

log "10/12 Validating artifacts"
python3 dev/validate_artifacts.py

log "11/12 Uploading artifacts"

# Always upload data/feature files — they're lookups, not a trained model.
echo "  [data] lag_history.parquet, blocks.parquet, citations_hourly_median.parquet,"
echo "         block_static_features.parquet, sfpark_calibration.parquet, inferred_block_aggs.parquet"
gsutil -m cp \
  app/models/lag_history.parquet \
  app/models/blocks.parquet \
  app/models/citations_hourly_median.parquet \
  app/models/block_static_features.parquet \
  app/models/sfpark_calibration.parquet \
  app/models/inferred_block_aggs.parquet \
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

CHALLENGER_RUN_ID=$(python3 -c "
import json
d = json.load(open('app/models/LightGBM.meta.json'))
print(d.get('mlflow_run_id', ''))
")

if [ "$SHOULD_UPLOAD" = "1" ]; then
  echo "  ✓ model improved (or matched) — uploading .pkl, .meta.json, .block_aggs.parquet"
  gsutil -m cp \
    app/models/LightGBM.pkl \
    app/models/LightGBM.meta.json \
    app/models/LightGBM.block_aggs.parquet \
    "$GCS_TARGET"
  echo "  MLflow run $CHALLENGER_RUN_ID stays (challenger wins)"
else
  echo "  ⚠ model REGRESSED (challenger > incumbent) — keeping existing GCS model"
  echo "    Data files were still refreshed, so lags and static features are current."
  if [ -n "$CHALLENGER_RUN_ID" ]; then
    echo "  Deleting losing MLflow run $CHALLENGER_RUN_ID so the registry stays clean..."
    python3 -c "
import os, mlflow, sys
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
try:
    mlflow.delete_run('$CHALLENGER_RUN_ID')
    print('    deleted')
except Exception as e:
    print(f'    failed (non-fatal): {e}', file=sys.stderr)
" || true
  fi
fi

log "12/12 Forcing Cloud Run cold-start so new files get re-downloaded"
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
