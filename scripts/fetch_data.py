"""
WeatherOrNot - real data fetcher.

Pulls two independently-sourced, real datasets and caches them locally so
the Streamlit app never has to hit the network:

1. DGCA-sourced national daily on-time-performance by carrier, from the
   Vonter/india-aviation-traffic GitHub mirror (ODbL 1.0 - attribution to
   DGCA / Ministry of Civil Aviation required):
   https://github.com/Vonter/india-aviation-traffic

2. Real historical daily weather (rain, wind) per route endpoint, from the
   Open-Meteo Historical Weather API (free, no key, ERA5 reanalysis):
   https://open-meteo.com/en/docs/historical-weather-api

Run:
    python scripts/fetch_data.py

Re-run any time to refresh the cache. After fetching, this also rebuilds
data_cache/route_probabilities.json via data_pipeline.py.
"""

import csv
import io
import os
import sys
import time
from datetime import date

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from route_definitions import unique_locations  # noqa: E402
import data_pipeline  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache")

DGCA_URL = "https://raw.githubusercontent.com/Vonter/india-aviation-traffic/main/aggregated/daily.csv"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_START = "2019-01-01"
WEATHER_END = "2024-12-31"

# Primary-source PDF for weather_evidence.py's transcribed tables (Rajya
# Sabha Unstarred Question No. 192, Ministry of Civil Aviation, 5 Feb 2024).
# Cached here for auditability; the numbers themselves are transcribed in
# weather_evidence.py since they're a static historical parliamentary
# answer, not a live feed.
WEATHER_EVIDENCE_PDF_URL = "https://sansad.in/getFile/annex/263/AU192.pdf?source=pqars"


def fetch_dgca_otp():
    print(f"Fetching DGCA national daily OTP from {DGCA_URL} ...")
    resp = requests.get(DGCA_URL, timeout=30)
    resp.raise_for_status()
    out_path = os.path.join(CACHE_DIR, "dgca_daily_otp.csv")
    with open(out_path, "wb") as f:
        f.write(resp.content)
    rows = sum(1 for _ in io.StringIO(resp.text)) - 1
    print(f"  saved {out_path} ({rows} rows, {len(resp.content):,} bytes)")


def fetch_weather_for_location(code, lat, lon):
    out_path = os.path.join(CACHE_DIR, f"weather_{code}.csv")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": WEATHER_START,
        "end_date": WEATHER_END,
        "daily": "precipitation_sum,windspeed_10m_max",
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(WEATHER_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    daily = payload["daily"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "precipitation_sum", "windspeed_10m_max"])
        for d, p, w in zip(daily["time"], daily["precipitation_sum"], daily["windspeed_10m_max"]):
            writer.writerow([d, p, w])
    print(f"  saved {out_path} ({len(daily['time'])} days)")


def fetch_weather_evidence_pdf():
    print(f"Fetching primary-source weather-delay-reasons PDF from {WEATHER_EVIDENCE_PDF_URL} ...")
    try:
        resp = requests.get(WEATHER_EVIDENCE_PDF_URL, timeout=30)
        resp.raise_for_status()
        out_path = os.path.join(CACHE_DIR, "AU192_rajya_sabha_weather_delays.pdf")
        with open(out_path, "wb") as f:
            f.write(resp.content)
        print(f"  saved {out_path} ({len(resp.content):,} bytes)")
    except requests.RequestException as e:
        print(f"  WARNING: could not fetch source PDF ({e}). "
              f"weather_evidence.py's transcribed numbers are unaffected.")


def fetch_all_weather():
    locations = unique_locations()
    print(f"Fetching Open-Meteo historical weather for {len(locations)} locations "
          f"({WEATHER_START} to {WEATHER_END}) ...")
    for code, (lat, lon) in locations.items():
        fetch_weather_for_location(code, lat, lon)
        time.sleep(0.5)  # be polite to the free public API


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fetch_dgca_otp()
    fetch_weather_evidence_pdf()
    fetch_all_weather()
    print("\nRebuilding route_probabilities.json from fetched data ...")
    data_pipeline.build()
    print(f"\nDone. Cache refreshed {date.today().isoformat()}.")


if __name__ == "__main__":
    main()
