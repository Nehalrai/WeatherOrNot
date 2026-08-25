import datetime
import json
import os

import pandas as pd
import streamlit as st

from data import (
    MONTHS,
    MODES,
    DEFAULT_TICKET_INSURANCE,
    get_locations,
    monthly_curve,
    route_provenance,
    route_n_observations,
    cache_metadata,
    rail_model_meta,
    weather_evidence_summary,
)
from ev_engine import compute_ev

st.set_page_config(
    page_title="WeatherOrNot",
    page_icon="🛡️",
    layout="wide",
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

st.title("WeatherOrNot")
st.caption("A multi-modal expected-value engine for travel insurance decisions")

try:
    meta = cache_metadata()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.expander("About this prototype", expanded=False):
    st.write(
        "This is a working demo of the WeatherOrNot concept from the project report: "
        "it estimates the chance your trip gets disrupted, applies the expected-value "
        "formula `EV = (chance of disruption x ticket price refunded) - insurance cost`, "
        "and returns a Buy / Marginal / Skip recommendation.\n\n"
        "**Data note:** disruption probabilities are computed, not hand-typed. Flight "
        f"routes blend a real national on-time-performance baseline "
        f"(DGCA-sourced, {meta['dgca_date_range']}, via the Vonter/india-aviation-traffic "
        "ODbL mirror) with each route's own weather severity, from real Open-Meteo "
        f"historical daily rain/wind data ({meta['weather_date_range']}) at both endpoints. "
        "Rail routes don't use a weather proxy at all: they're computed from *observed* "
        f"zone-level disruption rates across {meta['rail_rows_scanned']:,} real journey "
        f"records ({meta['rail_date_range']}), grouped by railway zone, calendar month and "
        "train category. A trip is counted as disrupted when a stop arrives more than 60 "
        "minutes late, and each route takes the average of its two endpoint zones' rate for "
        "that month. True per-route DGCA cancellation rates still aren't publicly "
        "downloadable, and the rail dataset carries no cancellation field either, so "
        "cancellations are not counted - see README.md for exactly what's real vs. "
        "estimated here."
    )
    st.caption(f"Data cache last refreshed: {meta['generated_at']}")

    model_meta = rail_model_meta()
    if model_meta:
        st.caption(
            f"Rail model: trained on {model_meta['n_samples']:,} journeys, "
            f"accuracy {model_meta['accuracy']:.1%}"
        )

    ev = weather_evidence_summary()
    avg_share = ev["avg_weather_share_of_reason_coded_delays"]
    avg_rate = ev["avg_weather_delay_rate_vs_total_departures"]
    st.markdown("---")
    st.write(
        f"**Weather really is a documented cause of disruption, not just an assumption:** "
        f"across {ev['years_covered'][0]}-{ev['years_covered'][-1]}, major Indian carriers "
        f"officially attributed weather as the reason for {avg_share:.1%} of all "
        f"reason-coded flight delays ({avg_rate:.2%} of all scheduled departures), per "
        f"government data. *Source: {ev['source']}.*"
    )
    st.caption(
        "Note: this figure alone understates true weather-linked disruption - most "
        "cascading ('Reactionary') delays are downstream of an earlier weather delay "
        "but get recoded once they cascade, which is why this app's probability model "
        "uses the fuller all-cause DGCA on-time-performance baseline above, not this "
        "narrower weather-only figure, as the actual multiplier."
    )


@st.cache_data
def _backtest():
    """(summary_stats, by_month_frame) from the cached back-test, or None."""
    path = os.path.join(CACHE_DIR, "backtest_results.csv")
    if not os.path.exists(path):
        return None
    results = pd.read_csv(path)
    if results.empty:
        return None

    error = results["predicted_prob"] - results["actual_rate"]
    actual = results["actual_rate"]
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    stats = {
        "mae": float(error.abs().mean()),
        "r2": (1 - float((error ** 2).sum()) / ss_tot) if ss_tot > 0 else float("nan"),
        "n_zone_pairs": int(results.groupby(["origin_zone", "dest_zone"]).ngroups),
        "total_n_journeys": int(results["n_journeys"].sum()),
    }
    by_month = (
        results.groupby("month")[["actual_rate", "predicted_prob"]].mean()
        .reindex(range(1, 13))
        .rename(columns={"actual_rate": "Actual", "predicted_prob": "Predicted"})
    )
    by_month.index = [MONTHS[m - 1] for m in by_month.index]
    return stats, by_month


_bt = _backtest()
if _bt:
    _stats, _by_month = _bt
    with st.expander("Model accuracy", expanded=False):
        a1, a2 = st.columns(2)
        a1.metric("Mean absolute error", f"{_stats['mae'] * 100:.1f} pp")
        a2.metric("R²", f"{_stats['r2']:.2f}")
        st.write(
            f"Back-tested against {_stats['total_n_journeys']:,} historical journey "
            f"records across {_stats['n_zone_pairs']} corridor pairs."
        )
        st.bar_chart(_by_month)
        st.caption(
            "Flight model uses national DGCA baselines — per-route flight back-test "
            "not available."
        )


# --- Data-source labelling -------------------------------------------------

def data_source_caption(mode_key, origin_code, dest_code, month):
    """One line of small grey text describing what backs this probability."""
    if mode_key == "train":
        return "ML model — AUC 0.778, 974k journeys"
    n_obs = route_n_observations(mode_key, origin_code, dest_code).get(month, 0)
    return f"DGCA national baseline — {n_obs:,} monthly records"


# --- Route-aware verdict note ----------------------------------------------

MONSOON_MONTHS = {6, 7, 8, 9}
FLIGHT_FOG_MONTHS = {12, 1}
RAIL_MONSOON_CITIES = ("Mangalore", "Trivandrum", "Mumbai", "Chennai")
RAIL_FOG_CITIES = ("Patna", "Kolkata")
RAIL_FOG_MONTHS = {11, 12, 1, 2}


def _touches(origin_name, dest_name, cities):
    """True if either endpoint is one of `cities`. Substring match so
    'Kolkata (Howrah)' matches 'Kolkata'."""
    return any(city in origin_name or city in dest_name for city in cities)


def route_note(mode_key, origin_name, dest_name, month):
    """Route-specific caveat, or None. First matching rule wins."""
    if mode_key == "flight":
        if month in MONSOON_MONTHS:
            return ("Mumbai, Chennai and Kolkata airports see above-average "
                    "cancellations in monsoon months.")
        if month in FLIGHT_FOG_MONTHS:
            return ("Delhi and Kolkata airports face heightened fog-delay risk "
                    "in December–January.")
        return None

    if month in MONSOON_MONTHS and _touches(origin_name, dest_name, RAIL_MONSOON_CITIES):
        return ("⚠ This corridor has elevated monsoon disruption risk — Western, "
                "South Western, and Southern Railway zones are among the most "
                "weather-affected in June–September.")
    if month in RAIL_FOG_MONTHS and _touches(origin_name, dest_name, RAIL_FOG_CITIES):
        return ("⚠ Eastern Railway corridors see significant fog-related delays "
                "in winter months.")
    return None


# --- Reusable journey input / output blocks --------------------------------

def journey_inputs(key_prefix, side_by_side=True):
    """Render one journey's input widgets and return the collected values."""
    mode = st.radio(
        "Travel mode", ["Flight", "Train"], horizontal=True, key=f"{key_prefix}_mode"
    )
    mode_key = MODES[mode]
    locations = get_locations(mode)
    names = [loc["name"] for loc in locations]

    if side_by_side:
        origin_name = st.selectbox(
            "Where are you travelling from?", names, index=0, key=f"{key_prefix}_origin")
        dest_options = [n for n in names if n != origin_name]
        dest_name = st.selectbox(
            "Where are you travelling to?", dest_options, index=0, key=f"{key_prefix}_dest")
    else:
        c1, c2 = st.columns(2)
        with c1:
            origin_name = st.selectbox(
                "Where are you travelling from?", names, index=0, key=f"{key_prefix}_origin")
        with c2:
            dest_options = [n for n in names if n != origin_name]
            dest_name = st.selectbox(
                "Where are you travelling to?", dest_options, index=0, key=f"{key_prefix}_dest")

    origin = next(loc for loc in locations if loc["name"] == origin_name)
    dest = next(loc for loc in locations if loc["name"] == dest_name)

    travel_date = st.date_input(
        "What date are you travelling?",
        value=datetime.date.today() + datetime.timedelta(days=30),
        min_value=datetime.date.today(),
        max_value=datetime.date.today() + datetime.timedelta(days=365),
        key=f"{key_prefix}_date",
    )

    default_ticket_price, default_insurance_cost = DEFAULT_TICKET_INSURANCE[mode_key]
    if side_by_side:
        ticket_price = st.number_input(
            "How much did your ticket cost? (Rs.)", min_value=1.0,
            value=float(default_ticket_price), step=50.0, key=f"{key_prefix}_ticket")
        insurance_cost = st.number_input(
            "How much does the insurance cost? (Rs.)", min_value=1.0,
            value=float(default_insurance_cost), step=5.0, key=f"{key_prefix}_premium")
    else:
        c3, c4 = st.columns(2)
        with c3:
            ticket_price = st.number_input(
                "How much did your ticket cost? (Rs.)", min_value=1.0,
                value=float(default_ticket_price), step=50.0, key=f"{key_prefix}_ticket")
        with c4:
            insurance_cost = st.number_input(
                "How much does the insurance cost? (Rs.)", min_value=1.0,
                value=float(default_insurance_cost), step=5.0, key=f"{key_prefix}_premium")

    return {
        "mode_key": mode_key,
        "origin": origin,
        "dest": dest,
        "travel_date": travel_date,
        "ticket_price": ticket_price,
        "insurance_cost": insurance_cost,
    }


VERDICT_COLORS = {"Buy": "green", "Marginal": "orange", "Skip": "red"}


def render_journey(inputs, risk_aversion, show_risk_caption=True):
    """Compute and render one journey's verdict block. Returns a summary dict."""
    mode_key = inputs["mode_key"]
    origin, dest = inputs["origin"], inputs["dest"]
    month = inputs["travel_date"].month

    curve = monthly_curve(mode_key, origin["code"], dest["code"])
    probability = curve[month]
    payout = inputs["ticket_price"]
    result = compute_ev(probability, payout, inputs["insurance_cost"],
                        risk_aversion=risk_aversion)

    st.markdown(f"### Verdict: :{VERDICT_COLORS[result.verdict]}[{result.verdict}]")
    st.write(result.explanation)
    st.caption(f"Based on typical {MONTHS[month - 1]} disruption patterns for this route.")

    note = route_note(mode_key, origin["name"], dest["name"], month)
    if note:
        st.info(note)

    m1, m2, m3 = st.columns(3)
    m1.metric("Disruption probability", f"{probability:.1%}")
    m1.caption(data_source_caption(mode_key, origin["code"], dest["code"], month))
    m2.metric("Payout if disrupted", f"Rs.{payout:,.0f}")
    m3.metric("Expected value", f"Rs.{result.ev:,.0f}")

    if show_risk_caption:
        st.caption(f"Risk sensitivity: {risk_aversion}/5")
    st.caption(
        f"EV = ({probability:.1%} x Rs.{payout:,.0f} ticket refund) - "
        f"Rs.{inputs['insurance_cost']:,.0f} insurance cost = Rs.{result.ev:,.0f}"
    )

    return {
        "verdict": result.verdict,
        "probability": probability,
        "curve": curve,
        "label": f"{origin['name']} -> {dest['name']}",
        "month": month,
    }


def risk_slider(key):
    """The disruption-sensitivity slider. Rendered once per run - inside the
    calculation setup for a single journey, above both columns when comparing
    (where it is shared rather than per-journey)."""
    choice = st.select_slider(
        "How much would a disruption affect your trip?",
        options=[
            "Barely at all (1)",
            "Minor inconvenience (2)",
            "Moderately (3)",
            "Very disruptive (4)",
            "Would seriously impact me (5)",
        ],
        value="Moderately (3)",
        key=key,
    )
    return int(choice.split("(")[1].rstrip(")"))


# --- Inputs ----------------------------------------------------------------
compare = st.checkbox("Compare two journeys", key="compare_mode")

if compare:
    risk_aversion = risk_slider("risk_choice_compare")
    st.caption(f"Risk sensitivity: {risk_aversion}/5 — applied to both journeys.")
    left, right = st.columns(2)
    with left:
        st.subheader("Journey A")
        inputs_a = journey_inputs("a")
    with right:
        st.subheader("Journey B")
        inputs_b = journey_inputs("b")

    st.divider()

    left, right = st.columns(2)
    with left:
        st.caption(f"**Journey A** — {inputs_a['origin']['name']} -> {inputs_a['dest']['name']}")
        summary_a = render_journey(inputs_a, risk_aversion, show_risk_caption=False)
    with right:
        st.caption(f"**Journey B** — {inputs_b['origin']['name']} -> {inputs_b['dest']['name']}")
        summary_b = render_journey(inputs_b, risk_aversion, show_risk_caption=False)

    st.divider()
    if summary_a["verdict"] == summary_b["verdict"]:
        st.write("Both journeys have similar risk profiles.")
    else:
        riskier = (summary_a if summary_a["probability"] >= summary_b["probability"]
                   else summary_b)
        st.write(
            f"Journey A: {summary_a['verdict']} — Journey B: {summary_b['verdict']}. "
            f"{riskier['label']} has higher disruption risk."
        )
else:
    inputs = journey_inputs("single", side_by_side=False)
    st.caption(
        f":gray[Data provenance:] "
        f"{route_provenance(inputs['mode_key'], inputs['origin']['code'], inputs['dest']['code'])}"
    )
    st.caption(
        "Assumes the insurance refunds your full ticket price if your trip is delayed "
        "or cancelled - the common structure for this kind of travel cover."
    )

    risk_aversion = risk_slider("risk_choice_single")

    st.divider()

    summary = render_journey(inputs, risk_aversion)

    # --- Seasonal chart ----------------------------------------------------
    st.subheader(f"Disruption probability by month — {summary['label']}")
    chart_df = pd.DataFrame({
        "Month": MONTHS,
        "Disruption probability": [summary["curve"][m] for m in range(1, 13)],
    })
    st.bar_chart(chart_df.set_index("Month")["Disruption probability"])
    st.caption(
        f"Your selected date falls in {MONTHS[summary['month'] - 1]}; bars show the "
        f"full seasonal pattern for this route."
    )

st.divider()
st.caption(
    "Prototype built for demo purposes — WeatherOrNot, Fintech Services Minor Project, MIT Manipal."
)


# --- Footer ----------------------------------------------------------------

@st.cache_data
def _footer_dates():
    """(model_training_date, cache_timestamp) - either may be None."""
    trained = None
    model_path = os.path.join(CACHE_DIR, "model_meta.json")
    if os.path.exists(model_path):
        with open(model_path, encoding="utf-8") as f:
            trained = json.load(f).get("training_date")

    cached = None
    routes_path = os.path.join(CACHE_DIR, "route_probabilities.json")
    if os.path.exists(routes_path):
        with open(routes_path, encoding="utf-8") as f:
            cached = json.load(f).get("generated_at")
    return trained, cached


st.divider()
_trained, _cached = _footer_dates()
f1, f2, f3 = st.columns(3)
f1.caption("Data: DGCA · Open-Meteo · Indian Railways (Kaggle)")
if _trained:
    f2.caption(f"Model trained: {_trained[:10]}")
if _cached:
    f3.caption(f"Cache: {_cached[:10]}")
