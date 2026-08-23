"""
WeatherOrNot - route/month disruption data (real-data-backed).

Location identity/coordinates live in route_definitions.py. The actual
disruption probabilities are computed by data_pipeline.py from real fetched
data (DGCA national on-time-performance + Open-Meteo historical weather,
see scripts/fetch_data.py) and cached in data_cache/route_probabilities.json,
for every origin-destination pair within each mode. This module just loads
that cache and serves it to the Streamlit app.
"""

import json
import os

from route_definitions import (  # noqa: F401
    MONTHS,
    MODES,
    FLIGHT_LOCATIONS,
    TRAIN_LOCATIONS,
    DEFAULT_TICKET_INSURANCE,
    get_locations,
    location_name,
    route_id,
)

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache", "route_probabilities.json")

_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        if not os.path.exists(CACHE_PATH):
            raise FileNotFoundError(
                f"{CACHE_PATH} not found. Run `python scripts/fetch_data.py` once "
                "to fetch real DGCA + Open-Meteo data and build the probability cache."
            )
        with open(CACHE_PATH, encoding="utf-8") as f:
            _cache = json.load(f)
    return _cache


def monthly_curve(mode_key, origin_code, dest_code):
    """Return {month_number: probability} for an origin-destination pair, from the real-data cache."""
    cache = _load_cache()
    entry = cache["routes"][route_id(mode_key, origin_code, dest_code)]
    return {int(m): p for m, p in entry["monthly_probability"].items()}


def route_provenance(mode_key, origin_code, dest_code):
    """Human-readable string describing what real data this pair's numbers came from."""
    cache = _load_cache()
    return cache["routes"][route_id(mode_key, origin_code, dest_code)]["provenance"]


def cache_metadata():
    cache = _load_cache()
    return {
        "generated_at": cache["generated_at"],
        "dgca_date_range": cache["dgca_date_range"],
        "weather_date_range": cache["weather_date_range"],
    }


def weather_evidence_summary():
    cache = _load_cache()
    return cache["weather_evidence"]
