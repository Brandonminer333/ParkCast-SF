"""Decide whether a freshly-trained challenger model should overwrite the
incumbent in GCS.

Single source of truth for the promotion gate used by dev/refresh_all.sh.
Extracted so the comparison logic, the JSON key path, and the
missing-incumbent fallback are unit-testable.

CLI:
    python3 -m dev.promote_decision \\
        --challenger app/models/LightGBM.meta.json \\
        --incumbent  /tmp/incumbent.meta.json
    # prints "1" (upload) or "0" (skip) on stdout, exit 0
    # exits non-zero with a message on stderr if the challenger meta is
    # unreadable or missing the MAE field — we never want to silently
    # treat a broken challenger as "ok to upload".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

# Tolerance lets ties upload, which keeps the MLflow run_id in the deployed
# meta fresh even when MAE didn't move.
TIE_TOLERANCE = 1e-4

MAE_KEY_PATH = ("metrics", "residual_model", "mae")


def _read_mae_strict(path: str) -> float:
    """Read challenger MAE. Raises on any problem — a broken challenger
    must never silently pass the gate."""
    with open(path) as f:
        d = json.load(f)
    cur = d
    for k in MAE_KEY_PATH:
        cur = cur[k]
    return float(cur)


def _read_mae_lenient(path: str) -> float:
    """Read incumbent MAE. Returns +inf if the file is missing, empty, or
    malformed — treating "no incumbent" as "infinitely bad" means a cold
    start (first ever upload) correctly promotes the challenger."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return math.inf
    try:
        with open(path) as f:
            d = json.load(f)
        cur = d
        for k in MAE_KEY_PATH:
            cur = cur[k]
        return float(cur)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return math.inf


def should_upload(challenger_path: str, incumbent_path: str) -> bool:
    challenger = _read_mae_strict(challenger_path)
    incumbent = _read_mae_lenient(incumbent_path)
    return challenger <= incumbent + TIE_TOLERANCE


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--challenger", required=True)
    p.add_argument("--incumbent", required=True)
    args = p.parse_args(argv)
    try:
        decision = should_upload(args.challenger, args.incumbent)
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"promote_decision: cannot read challenger meta: {e}", file=sys.stderr)
        return 2
    print("1" if decision else "0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
