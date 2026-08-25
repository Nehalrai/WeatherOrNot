"""
WeatherOrNot - real data fetcher.

Pulls three independently-sourced, real datasets and caches them locally so
the Streamlit app never has to hit the network:

1. DGCA-sourced national daily on-time-performance by carrier, from the
   Vonter/india-aviation-traffic GitHub mirror (ODbL 1.0 - attribution to
   DGCA / Ministry of Civil Aviation required):
   https://github.com/Vonter/india-aviation-traffic

2. Real historical daily weather (rain, wind) per route endpoint, from the
   Open-Meteo Historical Weather API (free, no key, ERA5 reanalysis):
   https://open-meteo.com/en/docs/historical-weather-api

3. The Indian Railways delay dataset (station-level arrival delays for ~7,000
   trains over a full year), used for zone-level rail disruption rates:
   https://www.kaggle.com/datasets/rxydenxd/indian-railways-delay-dataset
   Kaggle requires authentication to download, so this step is manual unless
   a Kaggle API token is present - instructions are printed at runtime.

Run:
    python scripts/fetch_data.py

Re-run any time to refresh the cache. After fetching, this also rebuilds
data_cache/route_probabilities.json via data_pipeline.py.
"""

import csv
import io
import os
import shutil
import subprocess
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


RAIL_DATASET = "rxydenxd/indian-railways-delay-dataset"
RAIL_FILES = ["combined_delay.csv", "station_full_names.csv", "train_details.csv"]


def ensure_rail_delay_dataset():
    """Make sure the Indian Railways delay dataset is in data_cache/.

    Tries the Kaggle CLI if it's installed and a token is configured; otherwise
    prints manual download instructions. Returns True if the files are present.
    """
    missing = [f for f in RAIL_FILES if not os.path.exists(os.path.join(CACHE_DIR, f))]
    if not missing:
        size_gb = os.path.getsize(os.path.join(CACHE_DIR, "combined_delay.csv")) / 1e9
        print(f"Indian Railways delay dataset already in data_cache "
              f"(combined_delay.csv {size_gb:.2f} GB) - skipping download.")
        return True

    print(f"Indian Railways delay dataset missing from data_cache: {', '.join(missing)}")
    token = os.path.expanduser("~/.kaggle/kaggle.json")
    if shutil.which("kaggle") and os.path.exists(token):
        print(f"  Kaggle CLI + token found, downloading {RAIL_DATASET} (~1GB) ...")
        rc = subprocess.call([
            "kaggle", "datasets", "download", "-d", RAIL_DATASET,
            "-p", CACHE_DIR, "--unzip",
        ])
        if rc == 0 and not [f for f in RAIL_FILES if not os.path.exists(os.path.join(CACHE_DIR, f))]:
            print("  done.")
            return True
        print(f"  Kaggle CLI exited {rc} and the files still aren't there.")

    print(f"""
  ---------------------------------------------------------------------
  MANUAL STEP NEEDED - Kaggle requires a login to download.

  Option A (browser, quickest):
    1. Open https://www.kaggle.com/datasets/{RAIL_DATASET.split('/')[1]}
       (full URL: https://www.kaggle.com/datasets/{RAIL_DATASET})
    2. Sign in, click Download (~1GB zip).
    3. Unzip it into:
         {CACHE_DIR}
       so that these sit directly in that folder:
         {', '.join(RAIL_FILES)}

  Option B (CLI, if you want re-runs to be automatic):
    1. Kaggle -> Settings -> API -> "Create New Token" (saves kaggle.json)
    2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
    3. pip install kaggle
    4. Re-run this script - it will download the dataset for you.

  NOTE: do NOT use a plain `curl` on the Kaggle download API - without auth
  it returns HTTP 404 and curl will happily save that error page as a .zip.

  Rail probabilities will fall back to the weather proxy until this is done.
  ---------------------------------------------------------------------
""")
    return False


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fetch_dgca_otp()
    fetch_weather_evidence_pdf()
    fetch_all_weather()
    have_rail = ensure_rail_delay_dataset()

    if have_rail:
        print("\nTraining the rail disruption classifier ...")
        try:
            from ml_model import train_rail_model
            train_rail_model()
        except Exception as e:  # training is optional - never break the fetch
            print(f"  WARNING: model training failed ({type(e).__name__}: {e}).")
            print("  Rail probabilities will use the observed zone rates alone.")

    print("\nRebuilding route_probabilities.json from fetched data ...")
    data_pipeline.build()
    print(f"\nDone. Cache refreshed {date.today().isoformat()}.")


if __name__ == "__main__":
    main()
