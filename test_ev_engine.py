"""Self-check for risk-aware verdicts in ev_engine.py.

Run: python test_ev_engine.py
"""

from ev_engine import (
    compute_ev, EVResult, MARGINAL_BAND_FRACTION, RISK_AVERSION_DEFAULT,
    _band_multiplier,
)

PAYOUT = 1200.0
PREMIUM = 35.0
BASE_BAND = MARGINAL_BAND_FRACTION * PREMIUM   # 7.0


def _prob_for_ev(target_ev):
    """Probability that yields exactly target_ev at PAYOUT/PREMIUM."""
    return (target_ev + PREMIUM) / PAYOUT


def test_band_multiplier_matches_the_spec():
    assert _band_multiplier(1) == 1.4          # widest  - hardest to reach Buy
    assert _band_multiplier(2) == 1.2
    assert _band_multiplier(3) == 1.0          # unchanged from pre-risk behaviour
    assert abs(_band_multiplier(4) - 0.8) < 1e-9
    assert abs(_band_multiplier(5) - 0.6) < 1e-9   # narrowest - easiest to reach Buy


def test_all_three_verdicts_at_every_risk_level():
    for risk in (1, 3, 5):
        band = BASE_BAND * _band_multiplier(risk)
        buy = compute_ev(_prob_for_ev(band + 1), PAYOUT, PREMIUM, risk_aversion=risk)
        marginal = compute_ev(_prob_for_ev(0.0), PAYOUT, PREMIUM, risk_aversion=risk)
        skip = compute_ev(_prob_for_ev(-band - 1), PAYOUT, PREMIUM, risk_aversion=risk)
        assert buy.verdict == "Buy", (risk, buy.verdict)
        assert marginal.verdict == "Marginal", (risk, marginal.verdict)
        assert skip.verdict == "Skip", (risk, skip.verdict)


def test_same_ev_flips_verdict_as_risk_changes():
    """One EV sitting between the risk=5 and risk=1 bands: Buy for the sensitive
    traveller, Marginal for the tolerant one. This is the whole feature."""
    ev = BASE_BAND * 0.8                       # 5.6: inside risk=1's 9.8 band,
    probability = _prob_for_ev(ev)             #      outside risk=5's 4.2 band
    assert compute_ev(probability, PAYOUT, PREMIUM, risk_aversion=5).verdict == "Buy"
    assert compute_ev(probability, PAYOUT, PREMIUM, risk_aversion=3).verdict == "Marginal"
    assert compute_ev(probability, PAYOUT, PREMIUM, risk_aversion=1).verdict == "Marginal"

    # ...and symmetrically on the Skip side
    probability = _prob_for_ev(-ev)
    assert compute_ev(probability, PAYOUT, PREMIUM, risk_aversion=5).verdict == "Skip"
    assert compute_ev(probability, PAYOUT, PREMIUM, risk_aversion=1).verdict == "Marginal"


def test_marginal_explanation_is_risk_aware():
    marginal = _prob_for_ev(0.0)
    assert "leaning toward buying" in compute_ev(
        marginal, PAYOUT, PREMIUM, risk_aversion=4).explanation
    assert "leaning toward buying" in compute_ev(
        marginal, PAYOUT, PREMIUM, risk_aversion=5).explanation
    assert "skipping is defensible" in compute_ev(
        marginal, PAYOUT, PREMIUM, risk_aversion=1).explanation
    assert "skipping is defensible" in compute_ev(
        marginal, PAYOUT, PREMIUM, risk_aversion=2).explanation
    assert "close to break-even" in compute_ev(
        marginal, PAYOUT, PREMIUM, risk_aversion=3).explanation


def test_buy_and_skip_explanations_are_unchanged_by_risk():
    for risk in (1, 3, 5):
        band = BASE_BAND * _band_multiplier(risk)
        buy = compute_ev(_prob_for_ev(band + 1), PAYOUT, PREMIUM, risk_aversion=risk)
        skip = compute_ev(_prob_for_ev(-band - 1), PAYOUT, PREMIUM, risk_aversion=risk)
        assert "rational choice for this itinerary" in buy.explanation
        assert "Skipping is the rational choice" in skip.explanation


def test_backward_compatible_without_risk_aversion():
    """The pre-risk call signature must behave exactly as risk_aversion=3."""
    for target in (-20.0, -5.0, 0.0, 5.0, 20.0):
        probability = _prob_for_ev(target)
        old = compute_ev(probability, PAYOUT, PREMIUM)
        new = compute_ev(probability, PAYOUT, PREMIUM,
                         risk_aversion=RISK_AVERSION_DEFAULT)
        assert old == new, target
        assert isinstance(old, EVResult)


def test_out_of_range_risk_is_clamped_not_inverted():
    """Unclamped, risk_aversion=8 gives a negative band, which would make
    everything Buy - including a clearly-negative EV."""
    bad = compute_ev(_prob_for_ev(-50.0), PAYOUT, PREMIUM, risk_aversion=8)
    assert bad.verdict == "Skip", bad.verdict
    assert compute_ev(_prob_for_ev(-50.0), PAYOUT, PREMIUM, risk_aversion=99) == \
        compute_ev(_prob_for_ev(-50.0), PAYOUT, PREMIUM, risk_aversion=5)
    assert compute_ev(_prob_for_ev(0.0), PAYOUT, PREMIUM, risk_aversion=-3) == \
        compute_ev(_prob_for_ev(0.0), PAYOUT, PREMIUM, risk_aversion=1)


def test_ev_itself_is_untouched_by_risk():
    """Risk aversion moves the verdict boundary, never the arithmetic."""
    probability = _prob_for_ev(3.0)
    evs = {compute_ev(probability, PAYOUT, PREMIUM, risk_aversion=r).ev
           for r in (1, 2, 3, 4, 5)}
    assert len(evs) == 1, evs


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")

    print(f"\n{len(tests)} passed\n")
    print("  band and verdict at EV = +Rs.5.60 (fixed itinerary, risk varying):")
    probability = _prob_for_ev(BASE_BAND * 0.8)
    for risk in (1, 2, 3, 4, 5):
        r = compute_ev(probability, PAYOUT, PREMIUM, risk_aversion=risk)
        print(f"    risk {risk}/5  band +/-Rs.{BASE_BAND * _band_multiplier(risk):5.2f}"
              f"   EV Rs.{r.ev:+6.2f}   -> {r.verdict}")
