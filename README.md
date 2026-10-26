# WeatherOrNot — Working Prototype

A working demo of the expected-value engine from the project report: pick
where you're flying or training from and to, the date you're travelling,
and how much the insurance costs vs. pays out, and get a
**Buy / Marginal / Skip** recommendation with the underlying probability
and EV math shown.

Disruption probabilities are computed from real fetched data, not hand-typed.

## Run it

You need Python 3.9+ installed. Then, in a terminal, from this folder:

```bash
pip install -r requirements.txt
python scripts/fetch_data.py   # one-time: pulls real data, builds the probability cache
streamlit run app.py
```

It will open automatically in your browser at `http://localhost:8501`.
If it doesn't open automatically, copy that URL into your browser.

`fetch_data.py` only needs to be re-run when you want to refresh the cache
(the app itself never hits the network — it reads
`data_cache/route_probabilities.json`, which is checked into neither git
nor required to exist until you run the fetcher once).

## Where the numbers actually come from

| Layer | Source | Real? |
|---|---|---|
| Flight national baseline (by month) | DGCA-sourced daily on-time-performance by carrier, via the [Vonter/india-aviation-traffic](https://github.com/Vonter/india-aviation-traffic) GitHub mirror (ODbL 1.0, attribution to DGCA/MoCA) | Yes — real data, fetched and cached, `1 - mean(OTP)` computed per calendar month |
| Route-specific weather severity | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) — real daily precipitation/wind at each route's two endpoint coordinates, 2019–2024 | Yes — real ERA5-reanalysis data, fetched and cached |
| Rail national baseline | Official FY2024-25 all-India Mail/Express punctuality: **77.12%**, per the Railway Minister's Lok Sabha statement (reported via PIB/national press) | Yes — a real, independently-sourced figure (not from the project report) |
| Flight/rail route-month probability | `national baseline x (this route's weather severity / network-average weather severity)` for that month | Computed from the two real inputs above — see `data_pipeline.py` |
| Weather-as-a-cause evidence (supplementary, not a model input) | [Rajya Sabha Unstarred Question No. 192](https://sansad.in/getFile/annex/263/AU192.pdf?source=pqars), Ministry of Civil Aviation, 5 Feb 2024 — airline-wise delay/cancellation reasons, 2021–2023 | Yes — real, primary-source, cached PDF + transcribed in `weather_evidence.py`. Shown in-app as cited evidence that weather is a DGCA-documented delay cause (~4.3% of reason-coded delays, 2021–2023 average). **Not used as a probability multiplier** — see below. |
| Ticket price / insurance cost | User-editable, pre-filled with illustrative defaults. Payout is derived as the full ticket price (common structure for this kind of cover), not asked separately. | **Not** scraped from live insurer filings — out of scope for this version |

**What isn't real, and why:** true per-route DGCA cancellation-rate data and
per-route rail delay history aren't publicly downloadable. We checked:
the DGCA-derived GitHub mirror only has *national* daily OTP by carrier and
*city-pair passenger volume* (no delay/cancellation figures); NTES has no
bulk historical API; the Kaggle Indian Railway Delays dataset requires an
account/API key; Dataful's Train Punctuality Index is a JS app with no
plain CSV endpoint. So route-level numbers here are a **blend** of two real,
independently-verifiable signals (a real national baseline x real
route-specific weather), not fabricated per-route DGCA statistics. Every
route in the app shows its own "Data provenance" caption so this is never
hidden.

**Why the Rajya Sabha weather-delay-reasons data isn't used as a probability
input:** we checked (see `weather_evidence.py`) - the raw "Weather" reason
code only explains ~0.5-0.75% of total departures and ~3.5-5.7% of
reason-coded delays nationally, because "Reactionary" (knock-on delay from
an earlier flight) dominates at ~70% of reason-coded delays and is
frequently a downstream effect of an earlier weather delay that gets
recoded once it cascades. Feeding the raw "Weather" figure into the EV
model would badly *understate* real weather-linked risk, so it's shown
in-app as cited evidence instead, while the actual probability multiplier
still uses the fuller all-cause DGCA on-time-performance baseline (which
does capture cascading effects).

## Demo script

Three scenarios to show the panel, so it's clear the tool reasons
correctly rather than just rendering a form:

1. **Clear Buy** — Flight, Mumbai (BOM) → Delhi (DEL), travel date in
   **July** (monsoon peak, high weather severity relative to the network
   average).
2. **Clear Skip** — Flight, Mumbai (BOM) → Delhi (DEL), travel date in
   **February** (low weather severity, low disruption probability).
3. **Marginal** — Flight, Chennai (MAA) → Bangalore (BLR), travel date in
   **September**, insurance cost set to **Rs.350** — EV sits at
   break-even here (at the default Rs.200 premium it tips into Buy).

Also worth showing: switch the **Travel mode** toggle to Train. Rail
probabilities come from the trained classifier on real journey records,
not a weather proxy, so they're flatter across the year than flights.
Origin and destination are two independent dropdowns (12 airports for
Flight, 12 rail-hub cities for Train), so any pair can be checked, not
just a curated list.

Exact verdicts now depend on the freshly-fetched data cache rather than
fixed numbers — run `python scripts/fetch_data.py` and check the bar chart
for each pair to pick your live demo dates.

## What's real vs. simplified here

- **Real**: the EV formula (`EV = P_disruption x Payout - Premium`), the
  Buy/Marginal/Skip decision logic, the three-layer architecture (data
  pipeline → EV engine → presentation), the DGCA-sourced national OTP
  data, and the Open-Meteo weather data — all fetched live and cached,
  not hardcoded.
- **Simplified for time**: route-level probabilities are a real-baseline x
  real-weather blend rather than true per-route DGCA/NTES cancellation
  records (which aren't publicly downloadable — see above). Rail uses a
  gradient boosting classifier (`ml_model.py`, ROC-AUC 0.78 on ~975k
  journeys); flights have no ML model yet, as no per-flight delay data is
  publicly available.
  Ticket price / insurance cost are still user-editable illustrative
  defaults, not live insurer product data; payout is derived as a full
  ticket-price refund rather than pulled from real tiered policy wordings. The app takes a full travel date, but the
  underlying model still resolves to a calendar-month probability
  internally (no daily-level historical disruption data exists publicly),
  shown transparently via the "Based on typical `<Month>` disruption
  patterns" caption.

## Files

- `app.py` — Streamlit UI
- `route_definitions.py` — airport/city identity + coordinates (no probabilities)
- `scripts/fetch_data.py` — fetches real DGCA OTP + Open-Meteo weather data, caches it, rebuilds the probability cache
- `data_pipeline.py` — turns cached raw data into per-pair, per-month probabilities (every origin x destination combination within each mode)
- `data.py` — loads the probability cache for the app
- `weather_evidence.py` — real, cited DGCA weather-delay-reason data (Rajya Sabha, Feb 2024), used as supplementary evidence in the UI, not as a model input
- `ev_engine.py` — the EV formula and verdict logic
- `data_cache/` — fetched raw data + computed `route_probabilities.json` (generated by `fetch_data.py`, not checked in)
- `requirements.txt` — dependencies
