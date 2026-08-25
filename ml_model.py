"""
WeatherOrNot - gradient-boosted rail disruption classifier.

Trains on the raw Indian Railways delay dataset that Phase 1 downloaded
(data_cache/combined_delay.csv, ~38.4M station-level rows). One row = one
train's stop at one station; the target is whether that stop arrived more than
RAIL_DISRUPTION_DELAY_MIN late.

    python ml_model.py          # train and write the model + metadata

Features (6):
    origin_zone            label-encoded railway zone of the train's first stop
    dest_zone              label-encoded railway zone of the train's last stop
    month                  1-12
    train_category         label-encoded, bucketed to premium/express/local/other
    weather_severity_origin  Open-Meteo rain+wind severity for that zone-month
    weather_severity_dest    same, for the destination zone

Two things worth knowing about the numbers this prints:

  * The target is heavily imbalanced (~12% positive), so ACCURACY IS A WEAK
    SIGNAL - always predicting "on time" already scores ~88%. F1 and ROC AUC
    are the honest measures, and both are reported.
  * The classifier is deliberately NOT class-balanced. Balancing would improve
    F1 but decorrelate predict_proba from real-world frequency, and
    data_pipeline.py blends that probability into a money decision, so
    calibration matters more than a flattering F1.
"""

import json
import os
from datetime import datetime, timezone

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
DELAY_PATH = os.path.join(CACHE_DIR, "combined_delay.csv")
SCHEDULE_PATH = os.path.join(CACHE_DIR, "combined_schedule.csv")
STATIONS_PATH = os.path.join(CACHE_DIR, "station_full_names.csv")
TRAINS_PATH = os.path.join(CACHE_DIR, "train_details.csv")
MODEL_PATH = os.path.join(CACHE_DIR, "rail_model.joblib")
META_PATH = os.path.join(CACHE_DIR, "model_meta.json")

DISRUPTION_DELAY_MIN = 60
RANDOM_STATE = 42

# sklearn's GradientBoostingClassifier is exact and single-threaded, so it does
# not scale to tens of millions of rows. We train on a random subsample instead.
# ponytail: 1M rows is far past the point of diminishing returns for 6 features
# whose combinations number only ~14k; raise if you switch to
# HistGradientBoostingClassifier, which would handle the full set.
SAMPLE_ROWS = 1_000_000

FEATURES = [
    "origin_zone", "dest_zone", "month", "train_category",
    "weather_severity_origin", "weather_severity_dest",
]

# Dataset type_code -> coarse bucket. Kept as a visible dict because the
# boundary is a judgement call: T18 (Vande Bharat), GRB (Garib Rath) and PRM
# (Premium) are arguably "premium" too, but the brief scopes that bucket to
# Rajdhani/Shatabdi, so they land in "other". One edit here moves them.
CATEGORY_BUCKETS = {
    "RAJ-TRAINS": "premium",
    "SHT-TRAINS": "premium",
    "EXP-TRAINS": "express",
    "SF-TRAINS": "express",    # Superfast is part of the Mail/Express family
    "PASS-TRAINS": "local",    # includes the DEMU/MEMU stopping services
    "T18-TRAINS": "other",
    "GRB-TRAINS": "other",
    "PRM-TRAINS": "other",
}
DEFAULT_CATEGORY = "express"


def _bucket_category(type_code):
    return CATEGORY_BUCKETS.get(type_code, "other")


def _zone_weather_severity():
    """{zone: {month: severity}} - the Open-Meteo severity data_pipeline already
    computes, averaged across whichever mapped cities sit in each zone.

    Only the zones in CITY_ZONE have cached weather, which is also the only set
    predict_disruption is ever asked about, so training is restricted to them.
    """
    import data_pipeline
    from rail_zone_mapping import CITY_ZONE

    by_zone = {}
    for city_code, zone in CITY_ZONE.items():
        severity = data_pipeline._load_weather_severity_by_month(city_code)
        by_zone.setdefault(zone, []).append(severity)
    return {
        zone: {m: sum(s[m] for s in severities) / len(severities) for m in range(1, 13)}
        for zone, severities in by_zone.items()
    }


def zone_weather_severity():
    """Public alias - data_pipeline needs the same zone-averaged severity the
    model was trained on, so inference features match training features."""
    return _zone_weather_severity()


def _train_endpoints():
    """{train_no: (origin_zone, dest_zone, category_bucket)} for every train
    whose first and last stop both fall in a weather-backed zone."""
    import pandas as pd
    from rail_zone_mapping import normalize_zone

    stations = pd.read_csv(STATIONS_PATH)
    station_zone = {
        code: normalize_zone(raw) for code, raw in
        zip(stations["station_name"], stations["station_zone"])
    }

    schedule = pd.read_csv(
        SCHEDULE_PATH, usecols=["station_no", "station_name", "train_no"],
        dtype={"station_name": "str", "train_no": "str"},
    )
    seq = schedule.groupby("train_no")["station_no"]
    first = schedule.loc[seq.idxmin().values].set_index("train_no")["station_name"]
    last = schedule.loc[seq.idxmax().values].set_index("train_no")["station_name"]

    trains = pd.read_csv(TRAINS_PATH)
    category = {
        str(no): _bucket_category(code)
        for no, code in zip(trains["train_no"], trains["type_code"])
    }

    weather_zones = set(_zone_weather_severity())
    endpoints = {}
    for train_no in first.index:
        origin_zone = station_zone.get(first[train_no])
        dest_zone = station_zone.get(last[train_no])
        if origin_zone in weather_zones and dest_zone in weather_zones:
            # train numbers are zero-padded in the delay table, bare in the
            # details table - normalise on the integer value
            key = str(int(train_no))
            endpoints[key] = (origin_zone, dest_zone,
                              category.get(key, DEFAULT_CATEGORY))
    return endpoints


def _build_training_frame():
    """Stream the 1GB delay CSV, keep rows for weather-backed trains, and
    randomly subsample down to SAMPLE_ROWS."""
    import numpy as np
    import pandas as pd

    if not os.path.exists(DELAY_PATH):
        raise FileNotFoundError(
            f"Missing {DELAY_PATH}. Run `python scripts/fetch_data.py` first."
        )

    endpoints = _train_endpoints()
    zone_weather = _zone_weather_severity()
    print(f"  {len(endpoints):,} trains have both endpoints in weather-backed zones")

    rng = np.random.default_rng(RANDOM_STATE)
    kept, total_eligible, scanned = [], 0, 0
    reader = pd.read_csv(
        DELAY_PATH, chunksize=2_000_000,
        usecols=["date", "delay", "train_no"],
        dtype={"date": "str", "delay": "object", "train_no": "str"},
    )
    for chunk in reader:
        scanned += len(chunk)
        delay = pd.to_numeric(chunk["delay"], errors="coerce")
        chunk = chunk[delay.notna()].copy()          # blank delay = not observed
        chunk["delay"] = delay[delay.notna()]
        chunk["key"] = chunk["train_no"].astype(int).astype(str)
        chunk = chunk[chunk["key"].isin(endpoints)]
        if chunk.empty:
            continue
        total_eligible += len(chunk)
        # ~SAMPLE_ROWS out of a ~22M eligible pool; sample per chunk to stay in RAM
        keep_frac = min(1.0, SAMPLE_ROWS / 22_000_000)
        mask = rng.random(len(chunk)) < keep_frac
        if mask.any():
            kept.append(chunk[mask])

    frame = pd.concat(kept, ignore_index=True)
    if len(frame) > SAMPLE_ROWS:
        frame = frame.sample(n=SAMPLE_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  scanned {scanned:,} rows -> {total_eligible:,} eligible -> {len(frame):,} sampled")

    meta = frame["key"].map(endpoints)
    frame["origin_zone"] = [m[0] for m in meta]
    frame["dest_zone"] = [m[1] for m in meta]
    frame["train_category"] = [m[2] for m in meta]
    frame["month"] = frame["date"].str.slice(5, 7).astype(int)
    frame["weather_severity_origin"] = [
        zone_weather[z][m] for z, m in zip(frame["origin_zone"], frame["month"])
    ]
    frame["weather_severity_dest"] = [
        zone_weather[z][m] for z, m in zip(frame["dest_zone"], frame["month"])
    ]
    frame["target"] = (frame["delay"] > DISRUPTION_DELAY_MIN).astype(int)
    return frame


def train_rail_model():
    """Train the classifier, report metrics, and serialise model + encoders."""
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    print("Training rail disruption classifier ...")
    frame = _build_training_frame()

    encoders = {}
    for column in ("origin_zone", "dest_zone", "train_category"):
        encoder = LabelEncoder()
        frame[column] = encoder.fit_transform(frame[column])
        encoders[column] = encoder

    X = frame[FEATURES]
    y = frame["target"]
    base_rate = y.mean()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    print(f"  fitting on {len(X_train):,} rows x {len(FEATURES)} features "
          f"(this takes a few minutes) ...")
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predicted)
    f1 = f1_score(y_test, predicted, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)

    importances = sorted(
        zip(FEATURES, (float(v) for v in model.feature_importances_)),
        key=lambda pair: pair[1], reverse=True,
    )

    print(f"\n  base rate (always-predict-on-time accuracy): {1 - base_rate:.1%}")
    print(f"  accuracy: {accuracy:.1%}   F1: {f1:.3f}   ROC AUC: {auc:.3f}")
    print("  feature importances:")
    for name, value in importances:
        print(f"    {value:7.4f}  {name}")
    if accuracy <= 1 - base_rate:
        print("  NOTE: accuracy is at or below the majority-class baseline - "
              "the ranking (AUC) is what the blend actually uses.")

    joblib.dump({"model": model, "encoders": encoders, "features": FEATURES}, MODEL_PATH)

    meta = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(frame)),
        "n_features": len(FEATURES),
        "accuracy": round(float(accuracy), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "base_rate_positive": round(float(base_rate), 4),
        "feature_importances": {name: round(value, 6) for name, value in importances},
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  wrote {MODEL_PATH}")
    print(f"  wrote {META_PATH}")
    return meta


_loaded = None


def _load():
    """Model bundle, cached - data_pipeline calls predict ~1.6k times per build."""
    global _loaded
    if _loaded is None:
        import joblib
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Missing {MODEL_PATH}. Run `python ml_model.py` to train it."
            )
        _loaded = joblib.load(MODEL_PATH)
    return _loaded


def predict_disruption(origin_city, dest_city, month, train_category=DEFAULT_CATEGORY,
                       weather_sev_origin=0.0, weather_sev_dest=0.0):
    """Probability that a stop on this route in this month is >60 min late.

    origin_city / dest_city may be a TRAIN_LOCATIONS station code, a zone code
    ("WR") or a zone name ("Western Railway") - all resolve to a zone.
    """
    import pandas as pd
    from rail_zone_mapping import CITY_ZONE, normalize_zone

    bundle = _load()
    encoders = bundle["encoders"]

    def to_zone(value):
        if value in CITY_ZONE:
            return CITY_ZONE[value]
        zone = normalize_zone(value)
        if zone is None:
            raise KeyError(f"Cannot resolve {value!r} to a railway zone")
        return zone

    def encode(column, value):
        encoder = encoders[column]
        known = list(encoder.classes_)
        if value not in known:
            raise KeyError(f"{value!r} unseen for {column}; known: {known}")
        return int(encoder.transform([value])[0])

    category = train_category if train_category in encoders["train_category"].classes_ \
        else DEFAULT_CATEGORY
    row = pd.DataFrame([{
        "origin_zone": encode("origin_zone", to_zone(origin_city)),
        "dest_zone": encode("dest_zone", to_zone(dest_city)),
        "month": int(month),
        "train_category": encode("train_category", category),
        "weather_severity_origin": float(weather_sev_origin),
        "weather_severity_dest": float(weather_sev_dest),
    }])[bundle["features"]]
    return float(bundle["model"].predict_proba(row)[0, 1])


def model_available():
    return os.path.exists(MODEL_PATH)


if __name__ == "__main__":
    train_rail_model()
