"""
WeatherOrNot - back-test the rail classifier against actual outcomes.

For every origin-zone -> destination-zone corridor with enough journey history,
compares what ml_model.predict_disruption() says against what actually happened
in the raw dataset, month by month.

Run:
    python scripts/backtest.py

Writes data_cache/backtest_results.csv and prints a summary. app.py surfaces
the summary in its "Model accuracy" expander.

IMPORTANT - this is an IN-SAMPLE check. The classifier was trained on a ~975k
row sample of the same 38.4M-row dataset measured here, so these numbers
describe how well the model fits observed history, NOT how it generalises to
unseen data. A genuine hold-out would need a date-based split (train on
Feb-Oct, test on Nov-Feb), which the current ml_model.py does not do.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_model  # noqa: E402
from rail_zone_mapping import CITY_ZONE  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache")
DELAY_PATH = os.path.join(CACHE_DIR, "combined_delay.csv")
OUTPUT_PATH = os.path.join(CACHE_DIR, "backtest_results.csv")

DISRUPTION_DELAY_MIN = 60
MIN_JOURNEYS_PER_PAIR = 5_000
REFERENCE_CATEGORY = "express"
CHUNK_ROWS = 4_000_000


def _actual_rates():
    """Scan the raw delay CSV once.

    Returns two dicts keyed by (origin_zone, dest_zone, month):
        all categories      -> [n_disrupted, n_observed]
        express trains only -> [n_disrupted, n_observed]
    """
    if not os.path.exists(DELAY_PATH):
        raise FileNotFoundError(
            f"Missing {DELAY_PATH}. Run `python scripts/fetch_data.py` first."
        )

    endpoints = ml_model._train_endpoints()   # train_no -> (o_zone, d_zone, category)
    pair_of = {k: (o, d) for k, (o, d, _) in endpoints.items()}
    category_of = {k: c for k, (_, _, c) in endpoints.items()}

    all_counts, express_counts = {}, {}
    scanned = 0
    reader = pd.read_csv(
        DELAY_PATH, chunksize=CHUNK_ROWS,
        usecols=["date", "delay", "train_no"],
        dtype={"date": "str", "delay": "object", "train_no": "str"},
    )
    for chunk in reader:
        scanned += len(chunk)
        delay = pd.to_numeric(chunk["delay"], errors="coerce")
        keep = delay.notna()                       # blank delay = not observed
        chunk = chunk[keep].copy()
        if chunk.empty:
            continue
        chunk["key"] = chunk["train_no"].astype(int).astype(str)
        chunk = chunk[chunk["key"].isin(pair_of)]
        if chunk.empty:
            continue

        chunk["disrupted"] = (delay[keep][chunk.index] > DISRUPTION_DELAY_MIN).astype(int)
        chunk["month"] = chunk["date"].str.slice(5, 7).astype(int)
        pairs = chunk["key"].map(pair_of)
        chunk["origin_zone"] = [p[0] for p in pairs]
        chunk["dest_zone"] = [p[1] for p in pairs]

        for target, frame in (
            (all_counts, chunk),
            (express_counts, chunk[chunk["key"].map(category_of) == REFERENCE_CATEGORY]),
        ):
            if frame.empty:
                continue
            grouped = frame.groupby(
                ["origin_zone", "dest_zone", "month"])["disrupted"].agg(["sum", "count"])
            for key, row in grouped.iterrows():
                bucket = target.setdefault(key, [0, 0])
                bucket[0] += int(row["sum"])
                bucket[1] += int(row["count"])

    print(f"  scanned {scanned:,} rows")
    return all_counts, express_counts


def run_backtest():
    print("Back-testing the rail classifier against actual outcomes ...")
    if not ml_model.model_available():
        raise FileNotFoundError(
            "No trained model. Run `python ml_model.py` (or scripts/fetch_data.py) first."
        )

    all_counts, express_counts = _actual_rates()
    zone_weather = ml_model.zone_weather_severity()

    # Qualify corridors on their full-year volume, per the >=5,000 threshold.
    pair_totals = {}
    for (origin, dest, _month), (_dis, obs) in all_counts.items():
        pair_totals[(origin, dest)] = pair_totals.get((origin, dest), 0) + obs
    qualifying = {p for p, n in pair_totals.items() if n >= MIN_JOURNEYS_PER_PAIR}
    print(f"  {len(qualifying)} corridors with >={MIN_JOURNEYS_PER_PAIR:,} journeys "
          f"(of {len(pair_totals)} total)")

    rows = []
    for (origin, dest, month), (disrupted, observed) in sorted(all_counts.items()):
        if (origin, dest) not in qualifying or observed == 0:
            continue
        predicted = ml_model.predict_disruption(
            origin, dest, month, REFERENCE_CATEGORY,
            weather_sev_origin=zone_weather[origin][month],
            weather_sev_dest=zone_weather[dest][month],
        )
        actual = disrupted / observed
        express_disrupted, express_observed = express_counts.get((origin, dest, month), (0, 0))
        rows.append({
            "origin_zone": origin,
            "dest_zone": dest,
            "month": month,
            "n_journeys": observed,
            "predicted_prob": round(predicted, 6),
            "actual_rate": round(actual, 6),
            "abs_error": round(abs(predicted - actual), 6),
            "direction": "over" if predicted > actual else "under",
            # diagnostic: predictions are made for "express", so this is the
            # like-for-like actual. Not part of the headline metrics.
            "actual_rate_express": (round(express_disrupted / express_observed, 6)
                                    if express_observed else ""),
            "n_journeys_express": express_observed,
        })

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No qualifying corridors - nothing to back-test.")
    results.to_csv(OUTPUT_PATH, index=False)

    _print_summary(results)
    print(f"\n  wrote {OUTPUT_PATH} ({len(results)} corridor-months)")
    return results


def summarise(results):
    """Headline stats. Shared with app.py's Model accuracy expander."""
    error = results["predicted_prob"] - results["actual_rate"]
    mae = float(error.abs().mean())
    rmse = float((error ** 2).mean() ** 0.5)
    actual = results["actual_rate"]
    ss_res = float((error ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    within_10pp = float((error.abs() <= 0.10).mean())
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "within_10pp": within_10pp,
        "n_rows": int(len(results)),
        "n_zone_pairs": int(results.groupby(["origin_zone", "dest_zone"]).ngroups),
        "total_n_journeys": int(results["n_journeys"].sum()),
    }


def _print_summary(results):
    stats = summarise(results)

    print("\n  Top 5 worst predictions (by absolute error):")
    print(f"    {'corridor':<46}{'mth':>4}{'pred':>8}{'actual':>8}{'err':>8}{'n':>10}")
    worst = results.sort_values("abs_error", ascending=False).head(5)
    for _, row in worst.iterrows():
        corridor = f"{row['origin_zone'][:20]} -> {row['dest_zone'][:20]}"
        print(f"    {corridor:<46}{row['month']:>4}"
              f"{row['predicted_prob'] * 100:>7.1f}%{row['actual_rate'] * 100:>7.1f}%"
              f"{(row['predicted_prob'] - row['actual_rate']) * 100:>+7.1f}"
              f"{row['n_journeys']:>10,}")

    print(f"\n  MAE           {stats['mae'] * 100:6.2f} pp")
    print(f"  RMSE          {stats['rmse'] * 100:6.2f} pp")
    print(f"  R^2           {stats['r2']:6.3f}")
    print(f"  within +/-10pp{stats['within_10pp'] * 100:6.1f}%")
    print(f"  corridor-months {stats['n_rows']}, corridors {stats['n_zone_pairs']}, "
          f"journeys {stats['total_n_journeys']:,}")

    over = (results["direction"] == "over").mean()
    print(f"  bias: {over * 100:.0f}% of predictions over-estimate, "
          f"{(1 - over) * 100:.0f}% under-estimate")

    # Predictions are for "express"; the headline actual_rate mixes all
    # categories. Quantify how much that mismatch costs.
    matched = results[results["actual_rate_express"] != ""].copy()
    if len(matched):
        matched["actual_rate_express"] = matched["actual_rate_express"].astype(float)
        mae_matched = float(
            (matched["predicted_prob"] - matched["actual_rate_express"]).abs().mean())
        print(f"\n  like-for-like check: comparing against express-only actuals instead "
              f"of\n  all categories gives MAE {mae_matched * 100:.2f} pp "
              f"(vs {stats['mae'] * 100:.2f} pp headline) over {len(matched)} rows.")

    thin = (results["n_journeys"] < 500).sum()
    if thin:
        print(f"\n  NOTE: {thin} of {len(results)} corridor-months have <500 journeys; "
              f"their\n  actual_rate is noisy and inflates MAE.")
    print("\n  NOTE: in-sample - the model was trained on this same dataset.")


if __name__ == "__main__":
    run_backtest()
