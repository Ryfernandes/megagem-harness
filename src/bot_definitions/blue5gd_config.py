"""Numbers Blue5GD uses when it bids. The overnight search changes these.

A coin is worth 1 point at the end of the game. Earlier, a coin is worth more
than 1, because you can still spend it. Falling charts (B and D) start more
expensive than rising charts (A and C). Chart E peaks in the middle.

The prior ratios are a starting guess for "bid / public prize" before this
match has seen any bids. They are taken from the five bots we care about:
ValueLiquidityBot bids about 0.55 of its signal, MicahBot's treasure prior is
0.78, YaleBot starts at 0.30 / 0.45 / 0.60, and AggressiveBot spends harder
on rising charts than on falling ones.
"""

from __future__ import annotations

from dataclasses import dataclass

from lib import ValueChart


@dataclass(frozen=True, slots=True)
class Blue5Config:
    """Every knob the tuner is allowed to change. Nothing else is free."""

    # Added on top of 1.0. 0 means a coin is worth exactly 1 point.
    coin_k_rising: float = 0.50
    coin_k_falling: float = 1.00
    coin_k_peak: float = 1.25
    # Once this many gems are left, the extra coin value fades toward 0.
    endgame_face_gems: float = 6.0

    # 1.0 counts a mission we finish right now at full points.
    mission_now_weight: float = 1.0
    # Fraction of a mission's points we count for moving closer without finishing.
    mission_progress_weight: float = 0.0

    # Treasure priors: bid divided by the public value of the gems for sale.
    prior_treasure_early_rising: float = 0.50
    prior_treasure_early_falling: float = 0.50
    prior_treasure_early_peak: float = 0.45
    prior_treasure_late_rising: float = 0.80
    prior_treasure_late_falling: float = 0.65
    prior_treasure_late_peak: float = 0.70

    # Invest priors: bid divided by the investment bonus.
    prior_invest_early_rising: float = 0.50
    prior_invest_early_falling: float = 0.40
    prior_invest_early_peak: float = 0.45
    prior_invest_late_rising: float = 0.80
    prior_invest_late_falling: float = 0.40
    prior_invest_late_peak: float = 0.20

    # Loan priors: bid divided by the loan size.
    prior_loan_early_rising: float = 0.45
    prior_loan_early_falling: float = 0.40
    prior_loan_early_peak: float = 0.22
    prior_loan_late_rising: float = 0.00
    prior_loan_late_falling: float = 0.15
    prior_loan_late_peak: float = 0.18

    # How many imaginary past bids the prior counts as. Bigger trusts the prior longer.
    learn_strength: float = 12.0
    # How much other seats' bids count, compared with this seat's own bids.
    pool_weight: float = 0.50
    # Width of the guessed bid, as a fraction of the prize. Wider means we shade down.
    spread_ratio: float = 0.12

    # Extra prize for taking a lot that would hand a rival a mission.
    denial_weight: float = 0.40
    # Most extra coins that denial is allowed to add to a bid.
    denial_max_coins: float = 3.0

    # 0 reveals whatever makes the whole remaining market look cheaper.
    # Higher cares more about colors this bot still wants.
    reveal_weight: float = 0.0
    # Points subtracted for winning, because the winner bids last on the next tie.
    tie_penalty: float = 0.70

    def coin_k(self, family: str) -> float:
        """Return the coin premium for a rising, falling, or peak chart."""
        if family == "falling":
            return self.coin_k_falling
        if family == "peak":
            return self.coin_k_peak
        return self.coin_k_rising

    def prior_ratio(self, kind: str, late: bool, family: str) -> float:
        """Return the starting bid/prize guess for this auction, phase, and chart."""
        phase = "late" if late else "early"
        return float(getattr(self, f"prior_{kind}_{phase}_{family}"))


# Taken from the 2026-11-20 search (40 deals, 3 passes). On new seeds the
# score gap moved from -11.87 to -6.18, and sole wins from 22.3% to 31.6%.
# The cell guard failed: 3-player chart D against MicahBot fell by 3.59, and
# 3-player chart E against MicahBot fell by 4.55. These numbers are used anyway.
DEFAULT_BLUE5_CONFIG = Blue5Config()


def chart_family(chart: ValueChart) -> str:
    """Return "falling", "peak", or "rising" from the printed chart values.

    B and D pay the most for the first copies, so they are falling.
    E goes up and then back down, so it is peak.
    A and C pay more as more copies are revealed, so they are rising.
    """
    low = chart.get_value(0)
    high = chart.get_value(5)
    if low > high:
        return "falling"
    middle = max(chart.get_value(count) for count in range(1, 5))
    if middle > low and middle > high:
        return "peak"
    return "rising"
