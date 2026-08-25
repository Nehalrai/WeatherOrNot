"""
WeatherOrNot - turns the raw cached data (scripts/fetch_data.py) into
per-route, per-month disruption probabilities.

Method (deliberately simple and auditable - no hidden ML step):

  FLIGHTS
      route-month probability =
          national disruption baseline for that calendar month (real, from
          DGCA-sourced daily on-time-performance data)
          x (this route's weather severity that month / the network-average
             weather severity for that month)

  RAIL
      route-month probability =
          mean of the two endpoint zones' *observed* disruption rate for that
          calendar month, where a stop counts as disrupted if it arrived more
          than RAIL_DISRUPTION_DELAY_MIN late, measured from the Indian
          Railways delay dataset (see RAIL_DELAY_PATH).

      Zone-months with fewer than RAIL_MIN_OBSERVATIONS observations fall back
      to the old weather-proxy formula (national punctuality baseline scaled by
      route weather severity).

"Weather severity" = a blend of how often heavy rain and high wind show up
at the route's two endpoints, computed from real Open-Meteo historical
daily data.

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
from rail_zone_mapping import CITY_ZONE, normalize_zone
import weather_evidence

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
DGCA_PATH = os.path.join(CACHE_DIR, "dgca_daily_otp.csv")
OUTPUT_PATH = os.path.join(CACHE_DIR, "route_probabilities.json")

# Indian Railways delay dataset (rxydenxd/indian-railways-delay-dataset, MIT).
RAIL_DELAY_PATH = os.path.join(CACHE_DIR, "combined_delay.csv")
RAIL_STATIONS_PATH = os.path.join(CACHE_DIR, "station_full_names.csv")
RAIL_TRAINS_PATH = os.path.join(CACHE_DIR, "train_details.csv")
# combined_delay.csv is ~1GB, so the zone x month x category rollup is scanned
# once and cached here; build() reuses it until the raw CSV changes.
RAIL_ZONE_STATS_PATH = os.path.join(CACHE_DIR, "rail_zone_stats.json")

RAIL_DISRUPTION_DELAY_MIN = 60   # a stop arriving >60 min late counts as disrupted
RAIL_MIN_OBSERVATIONS = 30       # below this, fall back to the weather proxy
RAIL_CHUNK_ROWS = 2_000_000

# Weight given to the ml_model classifier when blending it with the observed
# zone-month rate. 0.0 = observed only (behaviour when no model is trained),
# 1.0 = model only. Set to 1.0 because the zone-month rate measures a different
# quantity than the app needs - "is any stop in this zone late" rather than "is
# my endpoint-to-endpoint journey disrupted" - and understates long-haul routes
# badly (Kolkata->Mumbai: 6.1% zone rate vs 59.4% measured on those trains).
ML_BLEND_WEIGHT = 1.0

HEAVY_RAIN_MM = 20.0          # a day with >20mm rain counts as "heavy rain"
HIGH_WIND_KMH = 40.0          # a day with >40km/h max gust counts as "high wind"

# Official FY2024-25 all-India Mail/Express punctuality, per Railway Minister's
# Lok Sabha statement (reported via PIB / national press, 2026): 77.12%.
# Now only used as the rail *fallback* baseline for thin zone-months.
RAIL_NATIONAL_BASELINE_DISRUPTION = 1 - 0.7712  # 0.2288

FLIGHT_PROB_BOUNDS = (0.02, 0.45)
# Both ends were calibrated for the weather-proxy era and have since been
# widened to match measured reality: Western/Southern Railway genuinely run
# 2-4% (the old 0.05 floor flattened ~26% of route-months), and high-disruption
# corridors such as Eastern->Western reach ~72% in December (the old 0.55
# ceiling clipped them).
RAIL_PROB_BOUNDS = (0.01, 0.70)


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
    from real DGCA-sourced daily data.

    Returns ({month: baseline}, date_range_str, {month: n_readings}), where
    n_readings counts the carrier OTP cells that survived the validity filters
    below - i.e. how much data stands behind that month's baseline.
    """
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

    baseline, n_observations = {}, {}
    for month in range(1, 13):
        values = by_month_values.get(month, [])
        n_observations[month] = len(values)
        if not values:
            baseline[month] = RAIL_NATIONAL_BASELINE_DISRUPTION  # fallback, shouldn't happen
            continue
        mean_otp = sum(values) / len(values)
        baseline[month] = round(1 - mean_otp, 4)

    date_range = f"{min(dates_seen)} to {max(dates_seen)}" if dates_seen else "unknown"
    return baseline, date_range, n_observations


# --- Rail: observed zone-level disruption -----------------------------------

def _rail_lookup_tables():
    """station code -> zone full name, and train number -> category code."""
    import pandas as pd

    for path in (RAIL_DELAY_PATH, RAIL_STATIONS_PATH, RAIL_TRAINS_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run `python scripts/fetch_data.py` and follow the "
                "printed instructions for the Indian Railways delay dataset."
            )

    stations = pd.read_csv(RAIL_STATIONS_PATH)
    station_zone = {}
    for code, raw_zone in zip(stations["station_name"], stations["station_zone"]):
        zone = normalize_zone(raw_zone)
        if zone:
            station_zone[code] = zone

    trains = pd.read_csv(RAIL_TRAINS_PATH)
    train_category = dict(zip(trains["train_no"].astype(int), trains["type_code"]))
    return station_zone, train_category


def build_rail_zone_stats():
    """Scan combined_delay.csv in chunks and roll it up to
    {zone: {month: {category: [n_disrupted, n_observed]}}}, cached as JSON.

    A stop counts as disrupted when its arrival delay exceeds
    RAIL_DISRUPTION_DELAY_MIN. Rows with a blank delay are treated as
    *not observed* and dropped from the denominator - the dataset has no
    cancellation column, so blanks are not assumed to be cancellations.
    """
    import pandas as pd

    station_zone, train_category = _rail_lookup_tables()

    counts = defaultdict(lambda: [0, 0])  # (zone, month, category) -> [disrupted, observed]
    total_rows = 0
    dates_seen = []
    reader = pd.read_csv(
        RAIL_DELAY_PATH,
        chunksize=RAIL_CHUNK_ROWS,
        usecols=["date", "station_name", "delay", "train_no"],
        dtype={"date": "str", "station_name": "str", "delay": "object", "train_no": "str"},
    )
    for chunk in reader:
        total_rows += len(chunk)
        dates_seen.append((chunk["date"].min(), chunk["date"].max()))
        delay = pd.to_numeric(chunk["delay"], errors="coerce")
        keep = delay.notna()
        if not keep.any():
            continue
        chunk = chunk[keep]
        frame = pd.DataFrame({
            "zone": chunk["station_name"].map(station_zone),
            "month": chunk["date"].str.slice(5, 7).astype(int),
            "category": pd.to_numeric(chunk["train_no"], errors="coerce").map(train_category),
            "disrupted": (delay[keep] > RAIL_DISRUPTION_DELAY_MIN).astype(int),
        }).dropna(subset=["zone", "category"])

        grouped = frame.groupby(["zone", "month", "category"])["disrupted"].agg(["sum", "count"])
        for (zone, month, category), row in grouped.iterrows():
            bucket = counts[(zone, month, category)]
            bucket[0] += int(row["sum"])
            bucket[1] += int(row["count"])

    stats = defaultdict(dict)
    for (zone, month, category), (disrupted, observed) in counts.items():
        stats[zone].setdefault(str(month), {})[category] = [disrupted, observed]

    date_range = (
        f"{min(d[0] for d in dates_seen)} to {max(d[1] for d in dates_seen)}"
        if dates_seen else "unknown"
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_rows_scanned": total_rows,
        "date_range": date_range,
        "disruption_delay_minutes": RAIL_DISRUPTION_DELAY_MIN,
        "zones": dict(stats),
    }
    with open(RAIL_ZONE_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {RAIL_ZONE_STATS_PATH} "
          f"({total_rows:,} rows scanned, {len(stats)} zones, {date_range})")
    return payload


def _load_rail_zone_stats():
    """Cached rollup, rebuilt if absent or older than the raw delay CSV."""
    fresh = (
        os.path.exists(RAIL_ZONE_STATS_PATH)
        and os.path.exists(RAIL_DELAY_PATH)
        and os.path.getmtime(RAIL_ZONE_STATS_PATH) >= os.path.getmtime(RAIL_DELAY_PATH)
    )
    if not fresh:
        print("Rolling up the Indian Railways delay dataset (this scans ~1GB once) ...")
        return build_rail_zone_stats()
    with open(RAIL_ZONE_STATS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _zone_month_rates(stats):
    """{(zone, month): (disruption_rate, n_observations)}, collapsing the
    per-category counts by observation-weighted mean."""
    rates = {}
    for zone, months in stats["zones"].items():
        for month_str, categories in months.items():
            disrupted = sum(c[0] for c in categories.values())
            observed = sum(c[1] for c in categories.values())
            if observed == 0:
                continue
            rates[(zone, int(month_str))] = (disrupted / observed, observed)
    return rates


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


def _build_mode(mode_key, locations, baseline_by_month, baseline_label, severity_by_location,
                bounds, n_obs_by_month=None):
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
            # For flights this is the number of carrier on-time-performance
            # readings behind that month's baseline - the cells averaged in
            # _load_flight_baseline_by_month, after its validity filters.
            "n_observations": {
                str(m): int((n_obs_by_month or {}).get(m, 0)) for m in range(1, 13)
            },
            "provenance": (
                f"{baseline_label} x Open-Meteo weather severity at "
                f"{origin_code}/{dest_code} (2019-01-01 to 2024-12-31) "
                f"relative to the {mode_key}-network average."
            ),
        }
    return routes_out


def _build_rail_mode(locations, zone_rates, rail_date_range, severity_by_location, bounds):
    """Rail routes: average the two endpoint zones' observed disruption rate for
    the month. Falls back to the weather proxy for any endpoint zone-month with
    fewer than RAIL_MIN_OBSERVATIONS observations.

    If a trained classifier is present, its prediction is blended into the
    observed rate at ML_BLEND_WEIGHT. Bounds are applied after blending.
    """
    codes = [loc["code"] for loc in locations]
    pairs = list(permutations(codes, 2))

    pair_severities = {
        (o, d): _pair_severity_by_month(o, d, severity_by_location) for o, d in pairs
    }
    network_avg = _network_average_severity(list(pair_severities.values()))

    predict, zone_weather = None, None
    try:
        import ml_model
        if ML_BLEND_WEIGHT > 0 and ml_model.model_available():
            predict = ml_model.predict_disruption
            # same zone-averaged severity the model trained on, not per-city
            zone_weather = ml_model.zone_weather_severity()
            print(f"  blending in rail_model.joblib "
                  f"({1 - ML_BLEND_WEIGHT:.0%} observed / {ML_BLEND_WEIGHT:.0%} model)")
    except ImportError as e:
        print(f"  ml_model unavailable ({e}); using observed zone rates alone")

    routes_out = {}
    for (origin_code, dest_code), sev in pair_severities.items():
        origin_zone = CITY_ZONE[origin_code]
        dest_zone = CITY_ZONE[dest_code]
        monthly, monthly_n, fallback_months = {}, {}, []
        blended_any = False

        for m in range(1, 13):
            endpoint_rates, endpoint_obs = [], []
            for zone in (origin_zone, dest_zone):
                rate, n_obs = zone_rates.get((zone, m), (None, 0))
                if rate is not None and n_obs >= RAIL_MIN_OBSERVATIONS:
                    endpoint_rates.append(rate)
                    endpoint_obs.append(n_obs)

            if len(endpoint_rates) == 2:
                p = sum(endpoint_rates) / 2
            else:
                # thin or missing on at least one endpoint - weather proxy
                ratio = sev[m] / network_avg[m] if network_avg[m] > 0 else 1.0
                p = RAIL_NATIONAL_BASELINE_DISRUPTION * ratio
                fallback_months.append(m)

            if predict is not None:
                try:
                    predicted = predict(
                        origin_code, dest_code, m,
                        weather_sev_origin=zone_weather[origin_zone][m],
                        weather_sev_dest=zone_weather[dest_zone][m],
                    )
                    p = (1 - ML_BLEND_WEIGHT) * p + ML_BLEND_WEIGHT * predicted
                    blended_any = True
                except (KeyError, ValueError) as e:
                    print(f"    model skipped {origin_code}->{dest_code} m{m}: {e}")

            monthly[str(m)] = round(_clip(p, bounds), 4)
            monthly_n[str(m)] = sum(endpoint_obs)

        if fallback_months:
            note = (
                f" Month(s) {','.join(str(m) for m in fallback_months)} fell back to the "
                f"77.12% all-India punctuality baseline x route weather severity "
                f"(fewer than {RAIL_MIN_OBSERVATIONS} observations)."
            )
        else:
            note = ""
        if blended_any:
            note += " Gradient-boosted classifier (AUC 0.778, trained on 974k journeys)."
        routes_out[route_id("train", origin_code, dest_code)] = {
            "monthly_probability": monthly,
            "n_observations": monthly_n,
            "provenance": (
                f"Observed share of train stops arriving more than "
                f"{RAIL_DISRUPTION_DELAY_MIN} min late, averaged across "
                f"{origin_zone} ({origin_code}) and {dest_zone} ({dest_code}) for that "
                f"month, from the Indian Railways delay dataset "
                f"(rxydenxd/indian-railways-delay-dataset, MIT, {rail_date_range}). "
                f"The dataset has no cancellation field, so cancellations are not "
                f"counted.{note}"
            ),
        }
    return routes_out


def build():
    os.makedirs(CACHE_DIR, exist_ok=True)

    all_codes = {loc["code"] for loc in FLIGHT_LOCATIONS + TRAIN_LOCATIONS}
    severity_by_location = {code: _load_weather_severity_by_month(code) for code in all_codes}

    flight_baseline, dgca_date_range, flight_n_obs = _load_flight_baseline_by_month()
    weather_date_range = "2019-01-01 to 2024-12-31"

    routes_out = {}

    routes_out.update(_build_mode(
        "flight", FLIGHT_LOCATIONS, flight_baseline,
        f"DGCA national daily on-time-performance baseline ({dgca_date_range}, "
        f"Vonter/india-aviation-traffic, ODbL)",
        severity_by_location, FLIGHT_PROB_BOUNDS, flight_n_obs,
    ))

    rail_stats = _load_rail_zone_stats()
    zone_rates = _zone_month_rates(rail_stats)
    rail_date_range = rail_stats.get("date_range", "unknown")
    routes_out.update(_build_rail_mode(
        TRAIN_LOCATIONS, zone_rates, rail_date_range,
        severity_by_location, RAIL_PROB_BOUNDS,
    ))

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dgca_date_range": dgca_date_range,
        "weather_date_range": weather_date_range,
        "rail_date_range": rail_date_range,
        "rail_rows_scanned": rail_stats.get("source_rows_scanned", 0),
        "routes": routes_out,
        "weather_evidence": weather_evidence.summarize(),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {OUTPUT_PATH} ({len(routes_out)} routes)")
    return output


if __name__ == "__main__":
    build()
