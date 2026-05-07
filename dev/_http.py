"""Shared HTTP helpers with retry/backoff for the pipeline's external calls.

Every external endpoint we hit (DataSF Socrata, Open-Meteo, public ICS feeds,
SFMTA download URLs) drops requests under load. Without retries, a single
ReadTimeout kills the whole 4-hour refresh pipeline and we burn a CI run.

Two helpers, one per HTTP library used in the pipeline:
- retry_session()  → requests.Session (paginated DataSF pulls)
- retry_urlopen()  → urllib.request shim (single-shot fetches in notebooks
                     and inline-python in refresh_all.sh)

Both apply the same policy: 5 attempts, exponential backoff (2/4/8/16/32s),
retry on connect errors, read timeouts, and 429/5xx.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Mapping

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TOTAL = 5
DEFAULT_BACKOFF = 2.0
RETRY_STATUS = (429, 500, 502, 503, 504)


def retry_session(
    total: int = DEFAULT_TOTAL,
    backoff_factor: float = DEFAULT_BACKOFF,
) -> requests.Session:
    """requests.Session that retries on ReadTimeout / ConnectionError /
    429 / 5xx with exponential backoff."""
    s = requests.Session()
    retry = Retry(
        total=total,
        connect=total,
        read=total,
        status=total,
        backoff_factor=backoff_factor,
        status_forcelist=RETRY_STATUS,
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


def retry_urlretrieve(
    url: str,
    dest: str,
    *,
    timeout: float = 1200.0,
    headers: Mapping[str, str] | None = None,
    total: int = DEFAULT_TOTAL,
    backoff_factor: float = DEFAULT_BACKOFF,
    chunk: int = 1 << 20,                           # 1 MiB
) -> str:
    """Stream `url` to `dest` (no in-memory buffering — for large CSV
    downloads). Same retry policy as retry_urlopen. Writes to a `.tmp`
    sidecar and renames on success so partial files never appear at
    `dest`."""
    req = urllib.request.Request(url, headers=dict(headers or {}))
    tmp = dest + ".part"
    last_exc: BaseException | None = None
    for attempt in range(total + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
            import os as _os
            _os.replace(tmp, dest)
            return dest
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code not in RETRY_STATUS or attempt == total:
                raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_exc = e
            if attempt == total:
                raise
        time.sleep(backoff_factor * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


def retry_urlopen(
    url: str,
    *,
    timeout: float = 120.0,
    headers: Mapping[str, str] | None = None,
    total: int = DEFAULT_TOTAL,
    backoff_factor: float = DEFAULT_BACKOFF,
) -> bytes:
    """Read `url` and return the response body as bytes. Retries
    URLError / HTTPError (429, 5xx) and socket timeouts with exponential
    backoff. Use when stdlib urllib is preferred over requests (smaller
    dep, e.g. the inline-python in refresh_all.sh).

    Final failure re-raises the underlying exception so the caller's
    error handling still sees a normal urllib exception type.
    """
    req = urllib.request.Request(url, headers=dict(headers or {}))
    last_exc: BaseException | None = None
    for attempt in range(total + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code not in RETRY_STATUS or attempt == total:
                raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_exc = e
            if attempt == total:
                raise
        time.sleep(backoff_factor * (2 ** attempt))
    # Defensive — loop always returns or raises above.
    assert last_exc is not None
    raise last_exc
