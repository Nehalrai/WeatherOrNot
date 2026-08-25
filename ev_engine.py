"""
WeatherOrNot - expected value engine.

Implements the core formula from the project report (Ch.2.5 / Ch.4.2):

    EV = (P_disruption x Payout) - Premium

If EV > 0, insurance purchase is economically rational (Rothschild &
Stiglitz, 1976).
"""

from dataclasses import dataclass


@dataclass
class EVResult:
    probability: float
    payout: float
    premium: float
    ev: float
    verdict: str
    explanation: str


# Marginal band as a fraction of the premium: EV within +/- this band is
# reported as "Marginal" rather than a hard Buy/Skip.
MARGINAL_BAND_FRACTION = 0.20

# How much a traveller's disruption sensitivity stretches or shrinks that band.
RISK_AVERSION_DEFAULT = 3
RISK_AVERSION_RANGE = (1, 5)
RISK_AVERSION_BAND_STEP = 0.2


def _band_multiplier(risk_aversion: int) -> float:
    """Low tolerance for disruption (5) narrows the marginal band so borderline
    cases resolve to Buy; high tolerance (1) widens it so they resolve to Skip.

    risk_aversion is clamped to RISK_AVERSION_RANGE: out-of-range values would
    otherwise drive the multiplier to zero or negative, which silently inverts
    the Buy/Skip comparison rather than failing.
    """
    low, high = RISK_AVERSION_RANGE
    clamped = max(low, min(high, int(risk_aversion)))
    return 1 + RISK_AVERSION_BAND_STEP * (RISK_AVERSION_DEFAULT - clamped)


def compute_ev(probability: float, payout: float, premium: float,
               risk_aversion: int = RISK_AVERSION_DEFAULT) -> EVResult:
    ev = probability * payout - premium
    band = MARGINAL_BAND_FRACTION * premium * _band_multiplier(risk_aversion)

    if ev > band:
        verdict = "Buy"
        explanation = (
            f"Expected payout ({probability:.1%} chance x Rs.{payout:,.0f}) "
            f"clearly exceeds what the insurance costs. Buying is the rational choice for this itinerary."
        )
    elif ev < -band:
        verdict = "Skip"
        explanation = (
            f"Disruption risk ({probability:.1%}) is low enough that the expected payout "
            f"doesn't justify the Rs.{premium:,.0f} insurance cost. Skipping is the rational choice."
        )
    else:
        verdict = "Marginal"
        low, high = RISK_AVERSION_RANGE
        clamped = max(low, min(high, int(risk_aversion)))
        if clamped >= 4:
            explanation = (
                "Given your low tolerance for disruption, leaning toward buying is "
                "reasonable even at break-even."
            )
        elif clamped <= 2:
            explanation = (
                "At break-even and with your higher risk tolerance, skipping is "
                "defensible."
            )
        else:
            explanation = (
                "The expected value is close to break-even. It's a personal call, not a "
                "clear-cut financial one."
            )

    return EVResult(
        probability=probability,
        payout=payout,
        premium=premium,
        ev=ev,
        verdict=verdict,
        explanation=explanation,
    )
