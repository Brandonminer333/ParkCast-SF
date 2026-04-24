"""Unit tests for pure helper functions in `app.main`.

These tests exercise single functions with no FastAPI app, no TestClient, and
no module-global swapping. Importing `app.main` is cheap because asset loading
is deferred to FastAPI's lifespan hook, not module import time.
"""

from __future__ import annotations

import pathlib
import sys
from datetime import date

import pytest

_repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.main import color_for, demand_level, is_school_day  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "occ,expected",
    [
        (0, "Low"),
        (39.9, "Low"),
        (40, "Medium"),
        (69.9, "Medium"),
        (70, "High"),
        (84.9, "High"),
        (85, "Very High"),
        (100, "Very High"),
    ],
)
def test_demand_level_boundaries(occ, expected):
    assert demand_level(occ) == expected


@pytest.mark.parametrize(
    "occ,expected_hex",
    [
        (10, "#22c55e"),
        (55, "#f59e0b"),
        (77, "#f97316"),
        (95, "#ef4444"),
    ],
)
def test_color_matches_legend(occ, expected_hex):
    assert color_for(occ) == expected_hex


def test_is_school_day_heuristic():
    assert is_school_day(date(2026, 3, 4)) == 1
    assert is_school_day(date(2026, 3, 7)) == 0
    assert is_school_day(date(2026, 7, 15)) == 0
    assert is_school_day(date(2026, 1, 1)) == 0
