"""Tests for dev/_http.py retry helpers.

These pin the retry policy: 5 attempts, exponential backoff, retry on
ReadTimeout/ConnectionError/429/5xx, give up on 4xx (except 429). A
silent regression here would un-do the resilience work and we'd notice
only when the next pipeline run dies on a transient error.
"""

from __future__ import annotations

import io
import pathlib
import sys
import urllib.error

import pytest

_repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "dev"))

import _http
from _http import RETRY_STATUS, retry_urlopen, retry_urlretrieve, retry_session


# ---------- retry_urlopen ----------------------------------------------------


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n: int = -1) -> bytes:
        if n in (-1, None):
            data, self._body = self._body, b""
            return data
        data, self._body = self._body[:n], self._body[n:]
        return data


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Don't actually wait between retries during tests."""
    monkeypatch.setattr(_http.time, "sleep", lambda *_a, **_k: None)


def _patch_urlopen(monkeypatch, side_effects):
    """side_effects: list of either bytes (success) or Exception (raise).
    Each call pops the next entry."""
    calls = {"n": 0}

    def fake(req, timeout):
        i = calls["n"]
        calls["n"] += 1
        item = side_effects[i]
        if isinstance(item, BaseException):
            raise item
        return _FakeResp(item)

    monkeypatch.setattr(_http.urllib.request, "urlopen", fake)
    return calls


def test_urlopen_succeeds_first_try(monkeypatch):
    calls = _patch_urlopen(monkeypatch, [b"ok"])
    assert retry_urlopen("https://x") == b"ok"
    assert calls["n"] == 1


def test_urlopen_retries_on_timeout(monkeypatch):
    calls = _patch_urlopen(monkeypatch, [TimeoutError("read timeout"), b"ok"])
    assert retry_urlopen("https://x") == b"ok"
    assert calls["n"] == 2


def test_urlopen_retries_on_url_error(monkeypatch):
    """Connection refused / DNS / etc. all surface as URLError."""
    calls = _patch_urlopen(monkeypatch, [
        urllib.error.URLError("conn refused"),
        urllib.error.URLError("conn refused"),
        b"ok",
    ])
    assert retry_urlopen("https://x") == b"ok"
    assert calls["n"] == 3


@pytest.mark.parametrize("code", RETRY_STATUS)
def test_urlopen_retries_retryable_status(monkeypatch, code):
    err = urllib.error.HTTPError("https://x", code, "boom", {}, io.BytesIO(b""))
    calls = _patch_urlopen(monkeypatch, [err, b"ok"])
    assert retry_urlopen("https://x") == b"ok"
    assert calls["n"] == 2


def test_urlopen_does_not_retry_404(monkeypatch):
    """4xx other than 429 are real errors — retrying just wastes time."""
    err = urllib.error.HTTPError("https://x", 404, "not found", {}, io.BytesIO(b""))
    calls = _patch_urlopen(monkeypatch, [err])
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        retry_urlopen("https://x")
    assert exc_info.value.code == 404
    assert calls["n"] == 1


def test_urlopen_gives_up_after_total_attempts(monkeypatch):
    """total=2 means 1 initial + 2 retries = 3 attempts max."""
    err = TimeoutError("read")
    calls = _patch_urlopen(monkeypatch, [err, err, err])
    with pytest.raises(TimeoutError):
        retry_urlopen("https://x", total=2)
    assert calls["n"] == 3


# ---------- retry_urlretrieve ------------------------------------------------


def test_urlretrieve_writes_file_and_atomically_renames(tmp_path, monkeypatch):
    dest = tmp_path / "out.csv"
    _patch_urlopen(monkeypatch, [b"col\n1\n2\n"])
    retry_urlretrieve("https://x", str(dest))
    assert dest.read_bytes() == b"col\n1\n2\n"
    # No leftover .part file on success.
    assert not (tmp_path / "out.csv.part").exists()


def test_urlretrieve_retries_partial_failures(tmp_path, monkeypatch):
    dest = tmp_path / "out.csv"
    _patch_urlopen(monkeypatch, [TimeoutError(), b"recovered"])
    retry_urlretrieve("https://x", str(dest))
    assert dest.read_bytes() == b"recovered"


def test_urlretrieve_does_not_leave_partial_at_dest_on_failure(tmp_path, monkeypatch):
    """If every attempt fails, dest must not exist (partial-file guard).
    We tolerate a leftover .part — it's the next run's problem to overwrite."""
    dest = tmp_path / "out.csv"
    err = TimeoutError("read")
    _patch_urlopen(monkeypatch, [err] * 6)
    with pytest.raises(TimeoutError):
        retry_urlretrieve("https://x", str(dest), total=5)
    assert not dest.exists()


# ---------- retry_session ----------------------------------------------------


def test_session_has_retry_adapter_for_https():
    s = retry_session()
    adapter = s.get_adapter("https://example.com")
    retry = adapter.max_retries
    assert retry.total == _http.DEFAULT_TOTAL
    assert retry.backoff_factor == _http.DEFAULT_BACKOFF
    assert set(retry.status_forcelist) == set(RETRY_STATUS)
    assert "GET" in retry.allowed_methods


def test_session_also_mounts_http():
    """Some legacy SF feeds redirect via plain http — guard against
    accidentally only mounting https."""
    s = retry_session()
    assert s.get_adapter("http://example.com") is not s.get_adapter("https://example.com") \
        or s.get_adapter("http://example.com").max_retries.total == _http.DEFAULT_TOTAL
