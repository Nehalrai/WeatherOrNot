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


def compute_ev(probability: float, payout: float, premium: float) -> EVResult:
    ev = probability * payout - premium
    band = MARGINAL_BAND_FRACTION * premium

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
