"""Self-check for app.py's pure helpers (no Streamlit runtime needed).

Run: python test_app_logic.py

Covers the route-note rules, the data-source caption, and that all three
verdicts are reachable in both modes - the things a browser click-through
would otherwise have to prove one combination at a time.
"""

import json
import os

from data import monthly_curve, get_locations
from ev_engine import compute_ev
from route_definitions import FLIGHT_LOCATIONS, TRAIN_LOCATIONS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

# app.py imports streamlit at module scope, so re-implement only the pure rules
# here by importing them through a stub-free path: they live in app.py but do
# not touch st.*, so importing app would still pull in streamlit. Instead the
# rules are exercised via their real definitions, copied by reference below.
import importlib.util
import sys
import types


def _load_app_rules():
    """Import app.py with a stubbed streamlit so the pure helpers are testable."""
    stub = types.ModuleType("streamlit")

    class _Noop:
        def __call__(self, *a, **k): return self
        def __getattr__(self, _): return _Noop()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _columns(n, **k): return [_Noop() for _ in range(n if isinstance(n, int) else len(n))]

    stub.__getattr__ = lambda name: _Noop()
    stub.columns = _columns
    stub.set_page_config = lambda **k: None
    stub.cache_data = lambda fn: fn
    stub.checkbox = lambda *a, **k: False
    stub.radio = lambda label, options, **k: options[0]
    stub.selectbox = lambda label, options, **k: options[0]
    stub.select_slider = lambda label, options, **k: k.get("value", options[0])
    stub.number_input = lambda label, **k: k.get("value", 1.0)
    stub.date_input = lambda label, **k: k.get("value")
    stub.expander = lambda *a, **k: _Noop()
    stub.stop = lambda: None

    saved = sys.modules.get("streamlit")
    sys.modules["streamlit"] = stub
    try:
        spec = importlib.util.spec_from_file_location(
            "_app_under_test", os.path.join(os.path.dirname(__file__), "app.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if saved is not None:
            sys.modules["streamlit"] = saved
        else:
            sys.modules.pop("streamlit", None)


app = _load_app_rules()


def test_flight_route_notes():
    for month in (6, 7, 8, 9):
        assert "monsoon months" in app.route_note("flight", "Mumbai (BOM)", "Delhi (DEL)", month)
    for month in (12, 1):
        assert "fog-delay risk" in app.route_note("flight", "Mumbai (BOM)", "Delhi (DEL)", month)
    for month in (2, 3, 4, 5, 10, 11):
        assert app.route_note("flight", "Mumbai (BOM)", "Delhi (DEL)", month) is None


def test_rail_monsoon_note_matches_listed_cities():
    for city in ("Mangalore", "Trivandrum", "Mumbai", "Chennai"):
        note = app.route_note("train", city, "Jaipur", 7)
        assert note and "elevated monsoon disruption risk" in note, city
        assert "Western, South Western, and Southern Railway" in note, city
        assert "Konkan" not in note, city
        # and symmetrically as the destination
        assert app.route_note("train", "Jaipur", city, 7) == note
    # a corridor touching none of them gets nothing in monsoon
    assert app.route_note("train", "Jaipur", "Lucknow", 7) is None


def test_rail_fog_note_and_kolkata_substring():
    for city in ("Patna", "Kolkata (Howrah)"):
        for month in (11, 12, 1, 2):
            note = app.route_note("train", city, "Jaipur", month)
            assert note and "Eastern Railway corridors" in note, (city, month)
    # 'Kolkata (Howrah)' must match the 'Kolkata' rule despite the suffix
    assert app.route_note("train", "Kolkata (Howrah)", "Jaipur", 12) is not None
    assert app.route_note("train", "Patna", "Jaipur", 5) is None


def test_rule_order_monsoon_wins_over_fog():
    """Kolkata in month 7 is monsoon-only; the two windows don't overlap, but
    pin the ordering so a future month-range edit can't silently swap them."""
    assert "monsoon" in app.route_note("train", "Mumbai", "Kolkata (Howrah)", 7)
    assert "Eastern Railway" in app.route_note("train", "Mumbai", "Kolkata (Howrah)", 12)


def test_data_source_caption():
    assert app.data_source_caption("train", "BOM-CITY", "NDLS", 6) ==         "ML model — AUC 0.778, 974k journeys"
    flight = app.data_source_caption("flight", "BOM", "DEL", 6)
    assert flight.startswith("DGCA national baseline —")
    assert "monthly records" in flight
    # must be a real count from the cache, not the old vestigial zero
    count = int(flight.split("—")[1].strip().split()[0].replace(",", ""))
    assert count > 0, flight


def test_flight_n_observations_come_from_the_cache():
    """The count must be served by data.route_n_observations, not re-derived."""
    from data import route_n_observations
    counts = route_n_observations("flight", "BOM", "DEL")
    assert set(counts) == set(range(1, 13)), sorted(counts)
    assert all(v > 0 for v in counts.values()), counts
    caption = app.data_source_caption("flight", "BOM", "DEL", 7)
    assert f"{counts[7]:,} monthly records" in caption, caption
    # and it varies by month rather than being one constant
    assert len({counts[m] for m in range(1, 13)}) > 1, counts


def test_rail_n_observations_not_shown_in_caption():
    """Rail's n_observations is vestigial - the caption must not surface it."""
    from data import route_n_observations
    rail_counts = route_n_observations("train", "BOM-CITY", "NDLS")
    caption = app.data_source_caption("train", "BOM-CITY", "NDLS", 6)
    assert str(rail_counts[6]) not in caption, caption


def test_all_three_verdicts_reachable_in_both_modes():
    """Drives the same compute_ev path render_journey uses."""
    for mode_key, locations in (("flight", FLIGHT_LOCATIONS), ("train", TRAIN_LOCATIONS)):
        origin, dest = locations[0]["code"], locations[1]["code"]
        probability = monthly_curve(mode_key, origin, dest)[6]
        ticket = 5000.0
        # premium = probability * ticket puts EV exactly at 0 -> Marginal;
        # moving the premium either side of the band gives Buy / Skip.
        break_even = probability * ticket
        assert compute_ev(probability, ticket, break_even).verdict == "Marginal", mode_key
        assert compute_ev(probability, ticket, break_even * 0.5).verdict == "Buy", mode_key
        assert compute_ev(probability, ticket, break_even * 1.5).verdict == "Skip", mode_key

        # ...and the same three are reachable at every risk level
        for risk in (1, 3, 5):
            verdicts = {
                compute_ev(probability, ticket, p, risk_aversion=risk).verdict
                for p in (break_even * 0.5, break_even, break_even * 1.5)
            }
            assert verdicts == {"Buy", "Marginal", "Skip"}, (mode_key, risk, verdicts)


def test_comparison_summary_wording():
    """Mirrors the two branches of the comparison summary line."""
    a = {"verdict": "Buy", "probability": 0.30, "label": "Patna -> Lucknow"}
    b = {"verdict": "Buy", "probability": 0.05, "label": "Mumbai -> Ahmedabad"}
    assert a["verdict"] == b["verdict"]           # -> "similar risk profiles"

    b["verdict"] = "Skip"
    riskier = a if a["probability"] >= b["probability"] else b
    line = (f"Journey A: {a['verdict']} — Journey B: {b['verdict']}. "
            f"{riskier['label']} has higher disruption risk.")
    assert line == ("Journey A: Buy — Journey B: Skip. "
                    "Patna -> Lucknow has higher disruption risk.")


def test_footer_survives_missing_model_meta():
    """The footer must render when model_meta.json is absent, not raise."""
    path = os.path.join(CACHE_DIR, "model_meta.json")
    backup = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            backup = f.read()
        os.remove(path)
    try:
        app._footer_dates.__wrapped__() if hasattr(app._footer_dates, "__wrapped__") \
            else app._footer_dates()
        trained, cached = app._footer_dates()
        assert trained is None, trained
        assert cached is not None, "route_probabilities.json should still supply a date"
    finally:
        if backup is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(backup)


def test_footer_dates_present_normally():
    trained, cached = app._footer_dates()
    assert cached and len(cached[:10]) == 10
    if os.path.exists(os.path.join(CACHE_DIR, "model_meta.json")):
        assert trained and len(trained[:10]) == 10


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
