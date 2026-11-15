"""Expected-utility arithmetic and a future-cash opportunity-cost heuristic.

Contributors: Grace Guo (85375651b), OpenAI Codex (implementation assistance).
This module supplies helpers, not a separately submitted bot. Valuation and
hidden-hand inference are unchanged when these helpers choose treasure bids.
"""

from collections import Counter
from collections.abc import Mapping
from math import ceil, exp, isfinite

from bots import BotInfo, PlayerInfo
from constants import COINS_PER_PLAYER, AuctionType, GemType


def expected_utility(probability: float, value: float, cost: int, k: float) -> float:
    """Expected surplus p(win) * (E[value] - (1+k) * cost)."""
    if (
        not all(isfinite(x) for x in (probability, value, cost, k))
        or not 0 <= probability <= 1 + 1e-12
        or cost < 0
        or k < 0
    ):
        raise ValueError("Expected utility needs valid probability, cost and penalty")
    return min(1.0, probability) * (value - (1 + k) * cost)


def opportunity_cost(
    info: BotInfo, liquidity: float, scale: float, floor: float
) -> float:
    """Scale the existing premium and add a reserve premium that expires.

    A positive floor values cash for future purchases even when the existing
    scarcity estimate is zero. It tapers to zero at the final treasure lot.
    """
    if (
        not isfinite(liquidity)
        or liquidity < 1
        or not all(isfinite(x) and x >= 0 for x in (scale, floor))
    ):
        raise ValueError("Opportunity-cost parameters must be finite and nonnegative")
    card = info.current_auction
    bought = (
        min(card.value, len(info.auction_gems))
        if card is not None and card.type is AuctionType.AUCTION
        else 0
    )
    remaining = max(0, info.gems_remaining - bought)
    cards = [
        (c.value, n)
        for c, n in info.auction_cards_remaining.items()
        if c.type is AuctionType.AUCTION
    ]
    average = (
        sum(value * n for value, n in cards) / max(1, sum(n for _, n in cards)) or 1.0
    )
    lots = remaining / average
    horizon = lots / (lots + 1.5)
    return scale * max(0.0, liquidity - 1) + floor * horizon


def completion_bonus(info: BotInfo, player: PlayerInfo) -> float:
    card = info.current_auction
    if card is None or card.type is not AuctionType.AUCTION:
        return 0.0
    after = Counter(player.won_gems)
    after.update(info.auction_gems[: card.value])
    return float(
        sum(
            m.value
            for m in info.available_missions
            if m.is_satisfied(after) and not m.is_satisfied(player.won_gems)
        )
    )


def bid_distribution(
    info: BotInfo,
    player: PlayerInfo,
    prices: Mapping[GemType, float],
    bid_rates: Mapping[int, float],
    bid_fraction: float,
    *,
    cash_limits: bool,
) -> list[float]:
    card = info.current_auction
    if card is None:
        return [1.0] + [0.0] * player.coins
    value = sum(prices[g] for g in info.auction_gems[: card.value])
    value += completion_bonus(info, player)
    center = bid_rates.get(player.id, bid_fraction) * value
    if cash_limits:
        center = min(player.coins, center)
        spread = max(1.5, 0.30 * center + 0.10 * player.coins)
        support = player.coins
    else:
        # Deliberately budget-unaware prediction for ablation only. Actual
        # bids still obey every affordability rule in the harness.
        spread = max(1.5, 0.30 * center + 0.10 * COINS_PER_PLAYER[info.num_players])
        support = max(1, ceil(center + 8 * spread))
    masses = [exp(-0.5 * ((bid - center) / spread) ** 2) for bid in range(support + 1)]
    total = sum(masses)
    return [mass / total for mass in masses]


def win_probability(
    info: BotInfo, bid: int, distributions: Mapping[int, list[float]]
) -> float:
    previous = info.previous_winner_id
    if previous is None:
        # Integrate prod(P(lower) + t * P(tie)) over t in [0, 1].
        # k tied opponents give us exactly 1/(k+1) chance in round one.
        coefficients = [1.0]
        for player in info.other_players:
            pmf = distributions[player.id]
            lower = sum(pmf[:bid])
            tie = pmf[bid] if bid < len(pmf) else 0.0
            updated = [0.0] * (len(coefficients) + 1)
            for i, coefficient in enumerate(coefficients):
                updated[i] += coefficient * lower
                updated[i + 1] += coefficient * tie
            coefficients = updated
        return sum(c / (i + 1) for i, c in enumerate(coefficients))
    our_rank = (info.self_player.id - previous - 1) % info.num_players
    probability = 1.0
    for player in info.other_players:
        rival_rank = (player.id - previous - 1) % info.num_players
        allowed = bid + int(our_rank < rival_rank)
        probability *= sum(distributions[player.id][:allowed])
    return probability


def update_bid_rate(
    rates: dict[int, float],
    player: PlayerInfo,
    item_value: float,
    bid: int,
    bid_fraction: float,
) -> None:
    """Learn softly from affordable public bids; no hidden-hand inference."""
    expected = max(1.0, bid_fraction * item_value)
    if player.coins >= expected:
        rate = min(1.2, bid / max(1.0, item_value))
        old = rates.get(player.id, bid_fraction)
        rates[player.id] = 0.82 * old + 0.18 * rate
