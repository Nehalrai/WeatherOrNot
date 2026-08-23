"""
WeatherOrNot - turns the raw cached data (scripts/fetch_data.py) into
per-route, per-month disruption probabilities.

Method (deliberately simple and auditable - no hidden ML step):

  route-month probability =
      national disruption baseline for that calendar month (real, from
      DGCA-sourced daily on-time-performance data for flights, or the
      official FY2024-25 all-India Mail/Express punctuality figure for
      rail)
      x (this route's weather severity that month / the network-average
         weather severity for that month, within the same mode)

"Weather severity" = a blend of how often heavy rain and high wind show up
at the route's two endpoints, computed from real Open-Meteo historical
daily data. This is NOT a claim that we have true per-route DGCA
cancellation records (that data isn't publicly downloadable - see
README.md) - it's an honest, real-data-driven proxy: a real national
baseline, scaled up or down by how much worse or better the weather at
this specific route's endpoints is relative to the network average.

Run standalone:
    python data_pipeline.py
or call build() from scripts/fetch_data.py after a fresh fetch.
"""

import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

from itertools import permutations

from route_definitions import FLIGHT_LOCATIONS, TRAIN_LOCATIONS, route_id
import weather_evidence

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
DGCA_PATH = os.path.join(CACHE_DIR, "dgca_daily_otp.csv")
OUTPUT_PATH = os.path.join(CACHE_DIR, "route_probabilities.json")

HEAVY_RAIN_MM = 20.0          # a day with >20mm rain counts as "heavy rain"
HIGH_WIND_KMH = 40.0          # a day with >40km/h max gust counts as "high wind"

# Official FY2024-25 all-India Mail/Express punctuality, per Railway Minister's
# Lok Sabha statement (reported via PIB / national press, 2026): 77.12%.
RAIL_NATIONAL_BASELINE_DISRUPTION = 1 - 0.7712  # 0.2288

FLIGHT_PROB_BOUNDS = (0.02, 0.45)
RAIL_PROB_BOUNDS = (0.05, 0.55)


def _load_weather_severity_by_month(code):
    """Return {month: severity} for one location code, from its cached weather CSV."""
    path = os.path.join(CACHE_DIR, f"weather_{code}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/fetch_data.py` first."
        )
    by_month_total = defaultdict(int)
    by_month_heavy_rain = defaultdict(int)
    by_month_high_wind = defaultdict(int)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            month = int(row["date"][5:7])
            precip = float(row["precipitation_sum"]) if row["precipitation_sum"] else 0.0
            wind = float(row["windspeed_10m_max"]) if row["windspeed_10m_max"] else 0.0
            by_month_total[month] += 1
            if precip > HEAVY_RAIN_MM:
                by_month_heavy_rain[month] += 1
            if wind > HIGH_WIND_KMH:
                by_month_high_wind[month] += 1

    severity = {}
    for month in range(1, 13):
        total = by_month_total.get(month, 0)
        if total == 0:
            severity[month] = 0.0
            continue
        heavy_rain_frac = by_month_heavy_rain.get(month, 0) / total
        high_wind_frac = by_month_high_wind.get(month, 0) / total
        severity[month] = 0.6 * heavy_rain_frac + 0.4 * high_wind_frac
    return severity


def _load_flight_baseline_by_month():
    """1 - mean(OTP across all reporting carriers) grouped by calendar month,
    from real DGCA-sourced daily data. Returns ({month: baseline}, date_range_str)."""
    if not os.path.exists(DGCA_PATH):
        raise FileNotFoundError(f"Missing {DGCA_PATH}. Run `python scripts/fetch_data.py` first.")

    by_month_values = defaultdict(list)
    dates_seen = []
    with open(DGCA_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        otp_cols = [c for c in reader.fieldnames if c.startswith("On Time Performance")]
        for row in reader:
            date_str = row["Date"]
            try:
                month = int(date_str[5:7])
            except (ValueError, TypeError):
                continue
            dates_seen.append(date_str)
            for col in otp_cols:
                raw = (row[col] or "").strip()
                if not raw:
                    continue
                match = re.match(r"(\d+\.?\d*)", raw)
                if not match:
                    continue
                otp_fraction = float(match.group(1)) / 100.0
                if not (0.0 <= otp_fraction <= 1.0):
                    continue  # guard against malformed source cells (e.g. "62.77.0%")
                by_month_values[month].append(otp_fraction)

    baseline = {}
    for month in range(1, 13):
        values = by_month_values.get(month, [])
        if not values:
            baseline[month] = 1 - RAIL_NATIONAL_BASELINE_DISRUPTION  # fallback, shouldn't happen
            continue
        mean_otp = sum(values) / len(values)
        baseline[month] = round(1 - mean_otp, 4)

    date_range = f"{min(dates_seen)} to {max(dates_seen)}" if dates_seen else "unknown"
    return baseline, date_range


def _pair_severity_by_month(origin_code, dest_code, severity_by_location):
    origin_sev = severity_by_location[origin_code]
    dest_sev = severity_by_location[dest_code]
    return {m: (origin_sev[m] + dest_sev[m]) / 2 for m in range(1, 13)}


def _network_average_severity(route_severities):
    """Average severity-by-month across a list of per-route severity dicts."""
    avg = {}
    for m in range(1, 13):
        values = [sev[m] for sev in route_severities]
        avg[m] = sum(values) / len(values) if values else 0.0
    return avg


def _clip(value, bounds):
    lo, hi = bounds
    return max(lo, min(hi, value))


def _build_mode(mode_key, locations, baseline_by_month, baseline_label, severity_by_location, bounds):
    codes = [loc["code"] for loc in locations]
    pairs = list(permutations(codes, 2))  # every ordered origin != destination pair

    pair_severities = {
        (o, d): _pair_severity_by_month(o, d, severity_by_location) for o, d in pairs
    }
    network_avg = _network_average_severity(list(pair_severities.values()))

    routes_out = {}
    for (origin_code, dest_code), sev in pair_severities.items():
        monthly = {}
        for m in range(1, 13):
            ratio = sev[m] / network_avg[m] if network_avg[m] > 0 else 1.0
            p = _clip(baseline_by_month[m] * ratio, bounds)
            monthly[str(m)] = round(p, 4)
        routes_out[route_id(mode_key, origin_code, dest_code)] = {
            "monthly_probability": monthly,
            "provenance": (
                f"{baseline_label} x Open-Meteo weather severity at "
                f"{origin_code}/{dest_code} (2019-01-01 to 2024-12-31) "
                f"relative to the {mode_key}-network average."
            ),
        }
    return routes_out


def build():
    os.makedirs(CACHE_DIR, exist_ok=True)

    all_codes = {loc["code"] for loc in FLIGHT_LOCATIONS + TRAIN_LOCATIONS}
    severity_by_location = {code: _load_weather_severity_by_month(code) for code in all_codes}

    flight_baseline, dgca_date_range = _load_flight_baseline_by_month()
    weather_date_range = "2019-01-01 to 2024-12-31"

    routes_out = {}

    routes_out.update(_build_mode(
        "flight", FLIGHT_LOCATIONS, flight_baseline,
        f"DGCA national daily on-time-performance baseline ({dgca_date_range}, "
        f"Vonter/india-aviation-traffic, ODbL)",
        severity_by_location, FLIGHT_PROB_BOUNDS,
    ))

    rail_baseline = {m: RAIL_NATIONAL_BASELINE_DISRUPTION for m in range(1, 13)}
    routes_out.update(_build_mode(
        "train", TRAIN_LOCATIONS, rail_baseline,
        "All-India Mail/Express punctuality baseline (77.12%, FY2024-25, "
        "Railway Minister's Lok Sabha statement / PIB) - no public per-route or "
        "month-level DGCA/NTES-equivalent rail dataset exists, so the seasonal "
        "shape here comes entirely from real weather data",
        severity_by_location, RAIL_PROB_BOUNDS,
    ))

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dgca_date_range": dgca_date_range,
        "weather_date_range": weather_date_range,
        "routes": routes_out,
        "weather_evidence": weather_evidence.summarize(),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {OUTPUT_PATH} ({len(routes_out)} routes)")
    return output


if __name__ == "__main__":
    build()
