"""Unit tests for pure helper functions in `app.main`.

These tests exercise single functions with no FastAPI app, no TestClient, and
no module-global swapping. Importing `app.main` is cheap because the
ModelBundle constructor's failure is caught and logged, not re-raised.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.main import _color, _demand_level, _recommendation  # noqa: E402

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
    assert _demand_level(occ) == expected


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
    assert _color(occ) == expected_hex


@pytest.mark.parametrize(
    "occ,expected_prefix",
    [
        (10, "Easy to park"),
        (55, "Good chance"),
        (77, "Limited spots"),
        (95, "Very hard to park"),
    ],
)
def test_recommendation_text(occ, expected_prefix):
    assert _recommendation(occ).startswith(expected_prefix)
