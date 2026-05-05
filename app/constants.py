"""Shared constants and demand-level classification for ParkCast SF.

Single source of truth for the occupancy thresholds used by the API,
helpers, and (conceptually) the frontend.
"""

from __future__ import annotations

from typing import NamedTuple

# ── Occupancy thresholds ───────────────────────────────────────────
THRESHOLD_LOW = 40
THRESHOLD_MEDIUM = 70
THRESHOLD_HIGH = 85


class DemandInfo(NamedTuple):
    """Demand level with its display properties."""

    label: str
    color: str
    recommendation: str


_BANDS: list[tuple[float, DemandInfo]] = [
    (THRESHOLD_LOW, DemandInfo("Low", "#22c55e", "Easy to park — plenty of spaces.")),
    (THRESHOLD_MEDIUM, DemandInfo("Medium", "#f59e0b", "Good chance of parking — head over.")),
    (
        THRESHOLD_HIGH,
        DemandInfo("High", "#f97316", "Limited spots — arrive early or check nearby blocks."),
    ),
    (
        float("inf"),
        DemandInfo("Very High", "#ef4444", "Very hard to park — consider transit or a garage."),
    ),
]


def classify_occupancy(pct: float) -> DemandInfo:
    """Return demand label, color, and recommendation for an occupancy %."""
    for threshold, info in _BANDS:
        if pct < threshold:
            return info
    return _BANDS[-1][1]


def demand_level(pct: float) -> str:
    """Demand label string for an occupancy percentage."""
    return classify_occupancy(pct).label


def color_for(pct: float) -> str:
    """Hex color for an occupancy percentage."""
    return classify_occupancy(pct).color


def recommendation_for(pct: float) -> str:
    """User-facing recommendation for an occupancy percentage."""
    return classify_occupancy(pct).recommendation


# ── Feature constants ──────────────────────────────────────────────
LAG_COLS = [
    "lag_1d",
    "lag_2d",
    "lag_7d",
    "lag_14d",
    "lag_28d",
    "lag_3d_mean",
    "lag_7d_mean",
]

# Cap on rows returned by /predict/blocks.
MAX_BLOCKS_RETURNED = 200
