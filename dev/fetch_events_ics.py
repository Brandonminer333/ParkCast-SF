"""
ParkCast SF — Venue Events ICS Fetcher

Pulls public iCalendar feeds for Chase Center and Oracle Park and appends
them to data/events.csv in the existing schema so preprocess_real_data.py
picks them up automatically. Existing hand-curated rows are preserved and
deduped on (date, venue_lat, venue_lon, event_name).
"""

import os
import re
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUT_PATH = os.path.join(DATA_DIR, "events.csv")

PT = ZoneInfo("America/Los_Angeles")

# Each feed: (url, venue_label, neighborhood, venue_lat, venue_lon, event_type)
FEEDS = [
    ("https://www.chasecentercalendar.com/chasecenter.ics",
     "Chase Center", "Mission Bay", 37.7680, -122.3877, "Sports"),
    ("https://www.chasecentercalendar.com/oraclepark.ics",
     "Oracle Park", "South of Market", 37.7786, -122.3893, "Sports"),
]


def parse_ics(text):
    """Yield (summary, dtstart_utc, dtend_utc) for each VEVENT."""
    blocks = re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", text, re.DOTALL)
    for b in blocks:
        summary = re.search(r"\nSUMMARY:([^\r\n]+)", b)
        dtstart = re.search(r"\nDTSTART[^:]*:([0-9TZ]+)", b)
        dtend = re.search(r"\nDTEND[^:]*:([0-9TZ]+)", b)
        if not (summary and dtstart and dtend):
            continue
        try:
            start = datetime.strptime(dtstart.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            end = datetime.strptime(dtend.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        yield summary.group(1).strip(), start, end


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ParkCast/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def main():
    print("=" * 60)
    print("ParkCast SF — Venue Events ICS Fetcher")
    print("=" * 60)

    new_rows = []
    for url, venue, nbh, lat, lon, etype in FEEDS:
        print(f"Fetching {venue}...")
        try:
            text = fetch(url)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        count = 0
        for summary, start_utc, end_utc in parse_ics(text):
            start_pt = start_utc.astimezone(PT)
            end_pt = end_utc.astimezone(PT)
            # Skip zero- or negative-duration garbage
            if end_pt <= start_pt:
                continue
            new_rows.append({
                "date": start_pt.date().isoformat(),
                "event_name": summary,
                "type": etype,
                "neighborhood": nbh,
                "venue_lat": lat,
                "venue_lon": lon,
                "start_hour": start_pt.hour,
                "end_hour": max(start_pt.hour, min(23, end_pt.hour)),
            })
            count += 1
        print(f"  Parsed {count} events")

    df_new = pd.DataFrame(new_rows)
    print(f"\nNew rows: {len(df_new)}")

    if os.path.exists(OUT_PATH):
        df_old = pd.read_csv(OUT_PATH)
        print(f"Existing rows: {len(df_old)}")
        combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        combined = df_new

    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["date", "venue_lat", "venue_lon", "event_name"]
    ).reset_index(drop=True)
    print(f"After dedupe: {len(combined)} (removed {before - len(combined)} duplicates)")

    combined = combined.sort_values(["date", "start_hour"]).reset_index(drop=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
