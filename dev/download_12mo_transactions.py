"""
ParkCast SF — 12-Month Meter Transactions Downloader

Fetches the last 12 months of SFMTA meter transactions month-by-month
(single-query Socrata responses choke above ~2M rows). Overwrites
data/meter_transactions.csv with the full-year superset.
"""

import os
import urllib.request
from datetime import datetime, timedelta

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUT_PATH = os.path.join(DATA_DIR, "meter_transactions.csv")
TMP_PATH = OUT_PATH + ".tmp"

END = datetime(2026, 4, 13)
START = END - timedelta(days=365)


def month_windows(start, end):
    cur = start.replace(day=1)
    while cur < end:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield cur, min(nxt, end)
        cur = nxt


def fetch_month(window_start, window_end):
    url = (
        "https://data.sfgov.org/resource/imvp-dq3v.csv?"
        "$where=session_start_dt%20between%20"
        f"'{window_start.strftime('%Y-%m-%dT00:00:00')}'%20and%20"
        f"'{window_end.strftime('%Y-%m-%dT00:00:00')}'"
        "&$limit=2000000"
    )
    urllib.request.urlretrieve(url, TMP_PATH)
    df = pd.read_csv(TMP_PATH, low_memory=False)
    return df


def main():
    print("=" * 60)
    print("ParkCast SF — 12-Month Transactions Downloader")
    print(f"Window: {START.date()}  →  {END.date()}")
    print("=" * 60)

    dfs = []
    total = 0
    for w_start, w_end in month_windows(START, END):
        print(f"  {w_start.date()}  →  {w_end.date()} ...", end=" ", flush=True)
        try:
            df = fetch_month(w_start, w_end)
            print(f"{len(df):,} rows")
            dfs.append(df)
            total += len(df)
        except Exception as e:
            print(f"ERROR: {e}")

    if not dfs:
        print("No data fetched, aborting.")
        return

    print(f"\nConcatenating {total:,} rows...")
    final = pd.concat(dfs, ignore_index=True)
    before = len(final)
    final = final.drop_duplicates()
    print(f"  After dedupe: {len(final):,} (removed {before - len(final):,})")

    final.to_csv(OUT_PATH, index=False)
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"\nSaved: {OUT_PATH}  ({size_mb:.1f} MB)")
    if os.path.exists(TMP_PATH):
        os.remove(TMP_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()
