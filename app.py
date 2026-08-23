import datetime

import pandas as pd
import streamlit as st

from data import (
    MONTHS,
    MODES,
    DEFAULT_TICKET_INSURANCE,
    get_locations,
    monthly_curve,
    route_provenance,
    cache_metadata,
    weather_evidence_summary,
)
from ev_engine import compute_ev

st.set_page_config(page_title="WeatherOrNot", page_icon="[W]", layout="centered")

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
        "Rail routes use the official FY2024-25 all-India Mail/Express punctuality figure "
        "(77.12%, Railway Minister's Lok Sabha statement) scaled the same way by route "
        "weather. True per-route DGCA cancellation rates and per-route rail delay history "
        "aren't publicly downloadable (no clean bulk API for either) - see README.md for "
        "exactly what's real vs. estimated here."
    )
    st.caption(f"Data cache last refreshed: {meta['generated_at']}")

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

# --- Inputs --------------------------------------------------------------
mode = st.radio("Travel mode", ["Flight", "Train"], horizontal=True)
mode_key = MODES[mode]
locations = get_locations(mode)
names = [loc["name"] for loc in locations]

col1, col2 = st.columns(2)
with col1:
    origin_name = st.selectbox("Where are you travelling from?", names, index=0)
with col2:
    dest_options = [n for n in names if n != origin_name]
    dest_name = st.selectbox("Where are you travelling to?", dest_options, index=0)

origin = next(loc for loc in locations if loc["name"] == origin_name)
dest = next(loc for loc in locations if loc["name"] == dest_name)

travel_date = st.date_input(
    "What date are you travelling?",
    value=datetime.date.today() + datetime.timedelta(days=30),
    min_value=datetime.date.today(),
    max_value=datetime.date.today() + datetime.timedelta(days=365),
)

st.caption(f":gray[Data provenance:] {route_provenance(mode_key, origin['code'], dest['code'])}")

default_ticket_price, default_insurance_cost = DEFAULT_TICKET_INSURANCE[mode_key]
col3, col4 = st.columns(2)
with col3:
    ticket_price = st.number_input(
        "How much did your ticket cost? (Rs.)", min_value=1.0,
        value=float(default_ticket_price), step=50.0,
    )
with col4:
    insurance_cost = st.number_input(
        "How much does the insurance cost? (Rs.)", min_value=1.0,
        value=float(default_insurance_cost), step=5.0,
    )
st.caption(
    "Assumes the insurance refunds your full ticket price if your trip is delayed "
    "or cancelled - the common structure for this kind of travel cover."
)

st.divider()

# --- Compute ---------------------------------------------------------------
curve = monthly_curve(mode_key, origin["code"], dest["code"])
probability = curve[travel_date.month]

payout = ticket_price
result = compute_ev(probability, payout, insurance_cost)

verdict_colors = {"Buy": "green", "Marginal": "orange", "Skip": "red"}
color = verdict_colors[result.verdict]

st.markdown(f"### Verdict: :{color}[{result.verdict}]")
st.write(result.explanation)
st.caption(f"Based on typical {MONTHS[travel_date.month - 1]} disruption patterns for this route.")

m1, m2, m3 = st.columns(3)
m1.metric("Disruption probability", f"{probability:.1%}")
m2.metric("Payout if disrupted", f"Rs.{payout:,.0f}")
m3.metric("Expected value", f"Rs.{result.ev:,.0f}")

st.caption(
    f"EV = ({probability:.1%} x Rs.{payout:,.0f} ticket refund) - Rs.{insurance_cost:,.0f} insurance cost "
    f"= Rs.{result.ev:,.0f}"
)

# --- Seasonal chart --------------------------------------------------------
route_title = f"{origin['name']} -> {dest['name']}"
st.subheader(f"Disruption probability by month — {route_title}")
chart_df = pd.DataFrame(
    {"Month": MONTHS, "Disruption probability": [curve[m] for m in range(1, 13)]}
)
st.bar_chart(chart_df.set_index("Month")["Disruption probability"])
st.caption(f"Your selected date falls in {MONTHS[travel_date.month - 1]}; bars show the full seasonal pattern for this route.")

st.divider()
st.caption(
    "Prototype built for demo purposes — WeatherOrNot, Fintech Services Minor Project, MIT Manipal."
)
