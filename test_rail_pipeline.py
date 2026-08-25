"""Self-check for the rail zone probability logic in data_pipeline.py.

Run: python test_rail_pipeline.py

Covers the parts real data won't exercise - notably the <30-observation
fallback, which never fires against the full dataset (every zone-month there
has six figures of observations).
"""

import data_pipeline as dp
from data_pipeline import _zone_month_rates, _build_rail_mode, RAIL_PROB_BOUNDS
from rail_zone_mapping import CITY_ZONE

# These tests pin the *observed-rate* arithmetic, so the ML blend is switched
# off for all of them except test_ml_blend_*, which turns it on deliberately.
dp.ML_BLEND_WEIGHT = 0.0

# Two endpoint cities in different zones: Mumbai (Western), New Delhi (Northern).
LOCATIONS = [
    {"code": "BOM-CITY", "name": "Mumbai", "coords": (18.9398, 72.8355)},
    {"code": "NDLS", "name": "New Delhi", "coords": (28.6430, 77.2194)},
]
FLAT_SEVERITY = {code: {m: 0.1 for m in range(1, 13)} for code in ("BOM-CITY", "NDLS")}


def test_zone_month_rates_weights_by_observations():
    stats = {"zones": {"Western Railway": {"7": {
        "EXP-TRAINS": [30, 100],   # 30%
        "PASS-TRAINS": [10, 900],  # ~1.1%
    }}}}
    rate, n = _zone_month_rates(stats)[("Western Railway", 7)]
    assert n == 1000, n
    # observation-weighted, so the 900-observation group dominates: 40/1000
    assert abs(rate - 0.04) < 1e-9, rate
    # ...and is NOT the unweighted mean of the two category rates (~15.6%)
    assert abs(rate - 0.1556) > 0.1


def test_uses_observed_rates_and_averages_both_endpoints():
    zone_rates = {}
    for m in range(1, 13):
        zone_rates[("Western Railway", m)] = (0.20, 5000)
        zone_rates[("Northern Railway", m)] = (0.30, 5000)
    out = _build_rail_mode(LOCATIONS, zone_rates, "test-range", FLAT_SEVERITY, RAIL_PROB_BOUNDS)

    entry = out["train:BOM-CITY-NDLS"]
    assert entry["monthly_probability"]["7"] == 0.25, entry["monthly_probability"]["7"]
    assert entry["n_observations"]["7"] == 10000
    assert "fell back" not in entry["provenance"]
    # both directions of the pair exist and agree (endpoint average is symmetric)
    assert out["train:NDLS-BOM-CITY"]["monthly_probability"] == entry["monthly_probability"]


def test_falls_back_when_an_endpoint_zone_month_is_thin():
    zone_rates = {}
    for m in range(1, 13):
        zone_rates[("Western Railway", m)] = (0.20, 5000)
        # Northern Railway has only 5 observations in month 7 -> below the threshold
        zone_rates[("Northern Railway", m)] = (0.30, 5000 if m != 7 else 5)
    out = _build_rail_mode(LOCATIONS, zone_rates, "test-range", FLAT_SEVERITY, RAIL_PROB_BOUNDS)
    entry = out["train:BOM-CITY-NDLS"]

    assert entry["monthly_probability"]["6"] == 0.25          # untouched month
    assert entry["monthly_probability"]["7"] != 0.25          # month 7 fell back
    # flat severity => ratio 1.0 => straight national baseline
    expected = round(dp.RAIL_NATIONAL_BASELINE_DISRUPTION, 4)
    assert entry["monthly_probability"]["7"] == expected, entry["monthly_probability"]["7"]
    assert "Month(s) 7 fell back" in entry["provenance"]
    # only the qualifying endpoint's observations are counted
    assert entry["n_observations"]["7"] == 5000


def test_missing_zone_entirely_falls_back():
    zone_rates = {("Western Railway", m): (0.20, 5000) for m in range(1, 13)}
    out = _build_rail_mode(LOCATIONS, zone_rates, "test-range", FLAT_SEVERITY, RAIL_PROB_BOUNDS)
    entry = out["train:BOM-CITY-NDLS"]
    assert entry["n_observations"]["7"] == 5000
    assert "fell back" in entry["provenance"]


def test_output_shape_matches_what_the_app_reads():
    zone_rates = {(CITY_ZONE[c], m): (0.2, 5000) for c in CITY_ZONE for m in range(1, 13)}
    out = _build_rail_mode(LOCATIONS, zone_rates, "test-range", FLAT_SEVERITY, RAIL_PROB_BOUNDS)
    for entry in out.values():
        assert set(entry["monthly_probability"]) == {str(m) for m in range(1, 13)}
        assert all(isinstance(v, float) for v in entry["monthly_probability"].values())
        assert isinstance(entry["provenance"], str)


def test_probabilities_are_clipped_to_bounds():
    lo, hi = RAIL_PROB_BOUNDS
    zone_rates = {(z, m): (0.99, 5000) for z in set(CITY_ZONE.values()) for m in range(1, 13)}
    out = _build_rail_mode(LOCATIONS, zone_rates, "test-range", FLAT_SEVERITY, RAIL_PROB_BOUNDS)
    assert out["train:BOM-CITY-NDLS"]["monthly_probability"]["1"] == hi

    zone_rates = {(z, m): (0.0, 5000) for z in set(CITY_ZONE.values()) for m in range(1, 13)}
    out = _build_rail_mode(LOCATIONS, zone_rates, "test-range", FLAT_SEVERITY, RAIL_PROB_BOUNDS)
    assert out["train:BOM-CITY-NDLS"]["monthly_probability"]["1"] == lo


def test_ml_blend_mixes_at_the_configured_weight():
    """With a stub predictor, the blend must be exactly
    (1-w)*observed + w*predicted, and bounds must apply AFTER blending."""
    import sys, types
    stub = types.ModuleType("ml_model")
    stub.model_available = lambda: True
    stub.predict_disruption = lambda o, d, m, *a, **k: 0.60
    stub.zone_weather_severity = lambda: {z: {m: 0.1 for m in range(1, 13)}
                                          for z in set(CITY_ZONE.values())}
    saved_module, saved_weight = sys.modules.get("ml_model"), dp.ML_BLEND_WEIGHT
    sys.modules["ml_model"] = stub
    dp.ML_BLEND_WEIGHT = 0.6
    try:
        zone_rates = {(z, m): (0.20, 5000)
                      for z in set(CITY_ZONE.values()) for m in range(1, 13)}
        out = _build_rail_mode(LOCATIONS, zone_rates, "test", FLAT_SEVERITY, RAIL_PROB_BOUNDS)
        entry = out["train:BOM-CITY-NDLS"]
        # 0.4*0.20 + 0.6*0.60 = 0.44
        assert entry["monthly_probability"]["7"] == 0.44, entry["monthly_probability"]["7"]
        assert "Gradient-boosted classifier" in entry["provenance"]

        # blending can push past the ceiling; the clip must still catch it
        stub.predict_disruption = lambda o, d, m, *a, **k: 0.99
        hot = {(z, m): (0.50, 5000) for z in set(CITY_ZONE.values()) for m in range(1, 13)}
        out = _build_rail_mode(LOCATIONS, hot, "test", FLAT_SEVERITY, RAIL_PROB_BOUNDS)
        # raw blend = 0.4*0.50 + 0.6*0.99 = 0.794 -> clipped to the 0.55 ceiling
        assert out["train:BOM-CITY-NDLS"]["monthly_probability"]["7"] == RAIL_PROB_BOUNDS[1]
    finally:
        dp.ML_BLEND_WEIGHT = saved_weight
        if saved_module is not None:
            sys.modules["ml_model"] = saved_module
        else:
            sys.modules.pop("ml_model", None)


def test_zero_blend_weight_means_observed_only():
    saved = dp.ML_BLEND_WEIGHT
    dp.ML_BLEND_WEIGHT = 0.0
    try:
        zone_rates = {(z, m): (0.20, 5000)
                      for z in set(CITY_ZONE.values()) for m in range(1, 13)}
        out = _build_rail_mode(LOCATIONS, zone_rates, "test", FLAT_SEVERITY, RAIL_PROB_BOUNDS)
        entry = out["train:BOM-CITY-NDLS"]
        assert entry["monthly_probability"]["7"] == 0.20
        assert "Gradient-boosted classifier" not in entry["provenance"]
    finally:
        dp.ML_BLEND_WEIGHT = saved


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
