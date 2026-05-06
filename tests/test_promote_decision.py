"""Tests for the model promotion gate in dev/promote_decision.py.

This is the only thing standing between a regressed model and prod, so the
comparison, the JSON key path, and the missing-incumbent fallback all need
to be pinned down.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

_repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dev.promote_decision import TIE_TOLERANCE, main, should_upload


def _write_meta(path: pathlib.Path, mae: float | None) -> pathlib.Path:
    """Write a meta.json shaped like app/models/LightGBM.meta.json. Pass
    mae=None to write a meta missing the MAE key entirely."""
    body: dict = {"mlflow_run_id": "abc123", "metrics": {"residual_model": {}}}
    if mae is not None:
        body["metrics"]["residual_model"]["mae"] = mae
    path.write_text(json.dumps(body))
    return path


def test_better_challenger_uploads(tmp_path):
    challenger = _write_meta(tmp_path / "ch.json", mae=4.0)
    incumbent = _write_meta(tmp_path / "in.json", mae=5.0)
    assert should_upload(str(challenger), str(incumbent)) is True


def test_worse_challenger_skips(tmp_path):
    challenger = _write_meta(tmp_path / "ch.json", mae=6.0)
    incumbent = _write_meta(tmp_path / "in.json", mae=5.0)
    assert should_upload(str(challenger), str(incumbent)) is False


def test_tie_within_tolerance_uploads(tmp_path):
    """Equal MAE must promote so the deployed meta's mlflow_run_id stays
    fresh — the inline comment in refresh_all.sh calls this out explicitly."""
    incumbent_mae = 5.0
    challenger = _write_meta(tmp_path / "ch.json", mae=incumbent_mae + TIE_TOLERANCE / 2)
    incumbent = _write_meta(tmp_path / "in.json", mae=incumbent_mae)
    assert should_upload(str(challenger), str(incumbent)) is True


def test_just_outside_tolerance_skips(tmp_path):
    challenger = _write_meta(tmp_path / "ch.json", mae=5.0 + TIE_TOLERANCE * 10)
    incumbent = _write_meta(tmp_path / "in.json", mae=5.0)
    assert should_upload(str(challenger), str(incumbent)) is False


def test_cold_start_missing_incumbent_uploads(tmp_path):
    """No incumbent in GCS yet (gsutil cp produced no file) → first model
    must upload. Lenient reader returns +inf for missing incumbent."""
    challenger = _write_meta(tmp_path / "ch.json", mae=99.0)
    missing = tmp_path / "does_not_exist.json"
    assert should_upload(str(challenger), str(missing)) is True


def test_empty_incumbent_file_uploads(tmp_path):
    """gsutil sometimes leaves a 0-byte file behind on a 404; treat the
    same as missing."""
    challenger = _write_meta(tmp_path / "ch.json", mae=99.0)
    empty = tmp_path / "empty.json"
    empty.write_text("")
    assert should_upload(str(challenger), str(empty)) is True


def test_corrupt_incumbent_uploads(tmp_path):
    challenger = _write_meta(tmp_path / "ch.json", mae=99.0)
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json")
    assert should_upload(str(challenger), str(corrupt)) is True


def test_incumbent_missing_mae_key_uploads(tmp_path):
    """Old-format meta without the residual_model.mae path → treat as
    no incumbent rather than crashing the whole pipeline."""
    challenger = _write_meta(tmp_path / "ch.json", mae=99.0)
    incumbent = _write_meta(tmp_path / "in.json", mae=None)
    assert should_upload(str(challenger), str(incumbent)) is True


def test_challenger_missing_mae_key_raises(tmp_path):
    """A broken challenger must NEVER silently pass the gate."""
    challenger = _write_meta(tmp_path / "ch.json", mae=None)
    incumbent = _write_meta(tmp_path / "in.json", mae=5.0)
    with pytest.raises(KeyError):
        should_upload(str(challenger), str(incumbent))


def test_challenger_missing_file_raises(tmp_path):
    incumbent = _write_meta(tmp_path / "in.json", mae=5.0)
    with pytest.raises(FileNotFoundError):
        should_upload(str(tmp_path / "nope.json"), str(incumbent))


def test_cli_prints_one_when_uploading(tmp_path, capsys):
    challenger = _write_meta(tmp_path / "ch.json", mae=4.0)
    incumbent = _write_meta(tmp_path / "in.json", mae=5.0)
    rc = main(["--challenger", str(challenger), "--incumbent", str(incumbent)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "1"


def test_cli_prints_zero_when_skipping(tmp_path, capsys):
    challenger = _write_meta(tmp_path / "ch.json", mae=6.0)
    incumbent = _write_meta(tmp_path / "in.json", mae=5.0)
    rc = main(["--challenger", str(challenger), "--incumbent", str(incumbent)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "0"


def test_cli_nonzero_exit_on_broken_challenger(tmp_path, capsys):
    """Broken challenger must surface as a non-zero exit so the bash
    script's `SHOULD_UPLOAD=$(...)` ends up empty and the `if = "1"` branch
    won't fire — fail-closed."""
    challenger = _write_meta(tmp_path / "ch.json", mae=None)
    incumbent = _write_meta(tmp_path / "in.json", mae=5.0)
    rc = main(["--challenger", str(challenger), "--incumbent", str(incumbent)])
    assert rc != 0
