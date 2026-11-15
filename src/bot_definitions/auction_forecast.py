"""Exact finite-deck auction horizons and optional liquidity planning.

Contributors: Grace Guo (85375651b), OpenAI Codex (implementation assistance).

Treasure stopping times integrate the remaining one/two-gem card counts, without
replacement. Financial cards are exchangeable among the gaps between treasures.
The game stops at the treasure that exhausts the gems; later finances never
count. Horizon bounds/means are exact under a uniformly shuffled auction deck.
Loan ownership, net proceeds and investment lockups are conservative heuristics,
used only for k, never as actual cash, free score, or additional bidding budget.
"""

from dataclasses import dataclass
from functools import lru_cache

from bots import BotInfo
from constants import AuctionType


@lru_cache(maxsize=8192)
def treasure_stopping_pmf(gems: int, singles: int, doubles: int) -> tuple[float, ...]:
    """P(number of future treasure draws needed), including a truncated last lot."""
    if min(gems, singles, doubles) < 0:
        raise ValueError("Gem and card counts must be nonnegative")
    if gems == 0:
        return (1.0,)
    if singles + 2 * doubles < gems:
        raise ValueError("Remaining treasure cards cannot exhaust the gems")
    result = [0.0] * (singles + doubles + 1)
    for size, copies in ((1, singles), (2, doubles)):
        if not copies:
            continue
        tail = treasure_stopping_pmf(
            max(0, gems - size), singles - int(size == 1), doubles - int(size == 2)
        )
        for draws, probability in enumerate(tail):
            result[draws + 1] += copies / (singles + doubles) * probability
    return tuple(result)


@dataclass(frozen=True)
class AuctionForecast:
    min_treasure: int
    max_treasure: int
    expected_treasure: float
    min_rounds: int
    max_rounds: int
    expected_rounds: float
    expected_loans: float
    expected_investments: float
    expected_loan_coins: float
    # Weight a finance card by the fraction of treasure opportunities after it.
    # Funds available early matter more than funds arriving before the last lot.
    timed_loan_coins: float
    timed_investment_coins: float
    timed_investments: float


@lru_cache(maxsize=4096)
def _forecast(
    gems: int,
    singles: int,
    doubles: int,
    loan10: int,
    loan20: int,
    invest5: int,
    invest10: int,
) -> AuctionForecast:
    if min(gems, singles, doubles, loan10, loan20, invest5, invest10) < 0:
        raise ValueError("Gem and card counts must be nonnegative")
    if not gems:
        return AuctionForecast(0, 0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pmf = treasure_stopping_pmf(gems, singles, doubles)
    possible = [draws for draws, probability in enumerate(pmf) if probability > 0]
    expected = sum(draws * probability for draws, probability in enumerate(pmf))
    treasure = singles + doubles
    finance = loan10 + loan20 + invest5 + invest10
    # A particular financial card lies in one of treasure+1 equiprobable gaps.
    # Conditional on stopping at T, exactly T gaps precede the final treasure.
    seen = expected / (treasure + 1)
    # Average fraction of opportunities following the card:
    # sum_{j=0}^{T-1} ((T-j)/T) / (treasure+1) = (T+1)/(2*(treasure+1)).
    timed = (expected + 1) / (2 * (treasure + 1))
    loan_coins = 10 * loan10 + 20 * loan20
    invest_coins = 5 * invest5 + 10 * invest10
    return AuctionForecast(
        min(possible),
        max(possible),
        expected,
        min(possible),
        max(possible) + finance,
        expected + finance * seen,
        (loan10 + loan20) * seen,
        (invest5 + invest10) * seen,
        loan_coins * seen,
        loan_coins * timed,
        invest_coins * timed,
        (invest5 + invest10) * timed,
    )


def forecast_auctions(info: BotInfo) -> AuctionForecast:
    """Forecast rounds AFTER the current auction from public counts only."""
    card = info.current_auction
    bought = (
        min(card.value, len(info.auction_gems))
        if card is not None and card.type is AuctionType.AUCTION
        else 0
    )
    counts = {
        (card.type, card.value): n for card, n in info.auction_cards_remaining.items()
    }
    return _forecast(
        max(0, info.gems_remaining - bought),
        counts.get((AuctionType.AUCTION, 1), 0),
        counts.get((AuctionType.AUCTION, 2), 0),
        counts.get((AuctionType.LOAN, 10), 0),
        counts.get((AuctionType.LOAN, 20), 0),
        counts.get((AuctionType.INVEST, 5), 0),
        counts.get((AuctionType.INVEST, 10), 0),
    )


def finance_opportunity_cost(
    info: BotInfo,
    liquidity: float,
    future_points: float,
    scale: float,
    floor: float,
    bid_fraction: float,
) -> float:
    """Experimental k: exact auction horizon plus approximate future funding.

    Conservatively retain half of anticipated loan principal after acquisition
    costs and split it equally across players. Forecast investment bids using
    the existing finance policy, capped by a reserve-preserving average budget.
    These predictions alter only k, not valuations, reserves or legal cash.
    """
    forecast = forecast_auctions(info)
    if not forecast.expected_treasure:
        return 0.0
    players = info.num_players
    cash = info.self_player.coins
    total_cash = cash + sum(p.coins for p in info.other_players)
    net_loans = 0.5 * forecast.timed_loan_coins
    investment_bids = min(
        bid_fraction * forecast.timed_investment_coins / max(0.70, liquidity - 1),
        0.60 * (total_cash + net_loans) / players * forecast.timed_investments,
    )
    projected_total = max(0.0, total_cash + net_loans - investment_bids)
    projected_own = max(0.0, cash + (net_loans - investment_bids) / players)
    target_budget = 0.62 * future_points / players
    shortage = max(0.0, 1 - projected_own / max(1.0, target_budget))
    premium = max(
        0.0,
        min(3.0, 0.85 * future_points / max(1.0, projected_total) - 1),
        0.70 * shortage,
    )
    # Keep treasure opportunities and total remaining auction duration visible.
    # Financial rounds affect this heuristic horizon but create no free points.
    treasure_horizon = forecast.expected_treasure / (forecast.expected_treasure + 1.5)
    total_horizon = forecast.expected_rounds / (forecast.expected_rounds + 1.5)
    horizon = 0.5 * (treasure_horizon + total_horizon)
    return (scale * premium + floor) * horizon
