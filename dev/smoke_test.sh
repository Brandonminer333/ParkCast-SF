#!/bin/bash
# Post-deploy smoke: hit /health and a representative /predict/blocks query,
# fail loudly if the response shape is wrong. Run from the GitHub Actions
# workflow after every deploy or data refresh.
#
# Usage:  bash dev/smoke_test.sh [SERVICE_URL]
#         (defaults to the prod Cloud Run URL)

set -euo pipefail

URL="${1:-https://parkcast-api-904706413856.us-central1.run.app}"
echo "── /health ──"
HEALTH=$(curl -s --max-time 60 --fail "$URL/health")
echo "$HEALTH"

TOTAL_BLOCKS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_blocks', 0))")
if [ "$TOTAL_BLOCKS" -lt 12000 ]; then
  echo "FAIL: total_blocks=$TOTAL_BLOCKS < 12000 — citywide catalog regressed"
  exit 1
fi

echo ""
echo "── /predict/blocks (USF Tue 2pm, 500m) ──"
RESP=$(curl -s --max-time 60 --fail -X POST "$URL/predict/blocks" \
  -H 'content-type: application/json' \
  -d '{"lat":37.7766,"lon":-122.4505,"radius_meters":500,"hour":14,"day_of_week":2,"month":4}')

# Use python for parsing — avoids brittle jq/grep
read -r BLOCKS NO_STREET NO_CNN OCC_STD <<<"$(echo "$RESP" | python3 -c "
import json, sys, statistics
d = json.load(sys.stdin)
bs = d['blocks']
no_street = sum(1 for b in bs if not b.get('street'))
no_cnn    = sum(1 for b in bs if b.get('coverage') == 'inferred' and not b.get('cnn'))
occs = [b['predicted_occupancy_pct'] for b in bs]
std = statistics.stdev(occs) if len(occs) > 1 else 0.0
print(d['total_blocks_found'], no_street, no_cnn, f'{std:.1f}')
")"

echo "  blocks_found    : $BLOCKS"
echo "  no street label : $NO_STREET"
echo "  inferred no cnn : $NO_CNN"
echo "  occ pct std     : $OCC_STD"

if [ "$BLOCKS" -lt 50 ]; then
  echo "FAIL: only $BLOCKS blocks at USF — expected ≥50 (citywide blocks not loading)"
  exit 1
fi
if [ "$NO_STREET" -gt 0 ]; then
  echo "FAIL: $NO_STREET blocks missing street label — cnn join broken"
  exit 1
fi
# Std≈0 means every block predicts the same number → KNN baseline isn't firing
if python3 -c "import sys; sys.exit(0 if float('$OCC_STD') >= 1.0 else 1)"; then
  :
else
  echo "FAIL: occ std=$OCC_STD too low — predictions are uniform (KNN baseline not loading)"
  exit 1
fi

echo ""
echo "✓ Smoke test passed: $BLOCKS blocks, all labeled, real per-block variation"
