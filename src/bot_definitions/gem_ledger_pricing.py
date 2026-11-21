"""Stateless gem, mission, coin, and reveal valuation for GemLedgerBot."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb, exp, prod

from bots import BotInfo, PlayerInfo
from constants import STANDARD_GEMS, AuctionType, GemType, WildCardRequirement
from lib import AuctionCard, MissionCard, ValueChart


@dataclass(frozen=True)
class PricingConfig:
    """Tuning parameters for coin scarcity, mission races, and reveal weighting."""

    coin_scarcity_gain: float = 4.0
    endgame_gem_threshold: int = 8
    endgame_premium_floor: float = 0.35
    max_coin_value: float = 2.0
    rival_mission_contest: float = 0.5
    mission_urgency_divisor: float = 10.0


DEFAULT_PRICING = PricingConfig()


def _env_float(name: str, default: float) -> float:
    """Read a float setting, or keep the default when the variable is unset."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    """Read an integer setting, or keep the default when the variable is unset."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def pricing_from_env() -> PricingConfig:
    """Build pricing settings from GL_CFG_* variables.

    The harness calls GemLedgerBot() with no arguments, so a search script
    passes candidate numbers through the environment. Unset variables keep
    today's defaults.
    """
    defaults = DEFAULT_PRICING
    return PricingConfig(
        coin_scarcity_gain=_env_float(
            "GL_CFG_COIN_SCARCITY", defaults.coin_scarcity_gain
        ),
        endgame_gem_threshold=_env_int(
            "GL_CFG_ENDGAME_GEMS", defaults.endgame_gem_threshold
        ),
        endgame_premium_floor=_env_float(
            "GL_CFG_ENDGAME_FLOOR", defaults.endgame_premium_floor
        ),
        max_coin_value=_env_float("GL_CFG_MAX_COIN", defaults.max_coin_value),
        rival_mission_contest=_env_float(
            "GL_CFG_RIVAL_CONTEST", defaults.rival_mission_contest
        ),
        mission_urgency_divisor=_env_float(
            "GL_CFG_MISSION_DIVISOR", defaults.mission_urgency_divisor
        ),
    )


@dataclass(frozen=True)
class MarketContext:
    """Current expected final values and remaining auction supply by gem color."""

    gem_values: dict[GemType, float]
    expected_supply: dict[GemType, float]


@dataclass(frozen=True)
class AuctionTerms:
    """Normalized payoff and budget parameters for the current auction round."""

    gross_value: float
    marginal_coin_cost: float
    reference_scale: float
    max_legal_bid: int


def expected_chart_payout(
    chart: ValueChart,
    base_revealed: int,
    copies_in_pool: int,
    pool_size: int,
    hidden_draws: int,
) -> float:
    """Compute exact expected final chart value under a hypergeometric draw."""
    low = max(0, hidden_draws - (pool_size - copies_in_pool))
    high = min(copies_in_pool, hidden_draws)
    weighted_sum = 0
    for drawn_copies in range(low, high + 1):
        ways = comb(copies_in_pool, drawn_copies) * comb(
            pool_size - copies_in_pool, hidden_draws - drawn_copies
        )
        weighted_sum += ways * chart.get_value(base_revealed + drawn_copies)
    return weighted_sum / comb(pool_size, hidden_draws)


def expected_remaining_supply(info: BotInfo) -> dict[GemType, float]:
    """Estimate how many copies of each gem color will still be auctioned."""
    pool_size = sum(info.unknown_gems.values())
    deck_share = info.gem_deck_count / pool_size if pool_size else 0.0
    exposed_counts = Counter(info.auction_gems)
    return {
        gem: exposed_counts[gem] + info.unknown_gems.get(gem, 0) * deck_share
        for gem in STANDARD_GEMS
    }


def build_market_context(info: BotInfo) -> MarketContext:
    """Compute each gem color's expected final value and remaining market supply."""
    hand_counts = Counter(info.self_player.hidden_gems)
    pool_size = sum(info.unknown_gems.values())
    hidden_draws = pool_size - info.gem_deck_count
    gem_values = {
        gem: expected_chart_payout(
            info.value_chart,
            info.value_display.get(gem, 0) + hand_counts[gem],
            info.unknown_gems.get(gem, 0),
            pool_size,
            hidden_draws,
        )
        for gem in STANDARD_GEMS
    }
    return MarketContext(
        gem_values=gem_values,
        expected_supply=expected_remaining_supply(info),
    )


def marginal_coin_value(
    info: BotInfo,
    card: AuctionCard,
    market: MarketContext,
    current_lot_value: float,
    cfg: PricingConfig = DEFAULT_PRICING,
) -> float:
    """Price one coin in points by comparing future prizes to table liquidity."""
    gems_after_round = info.gems_remaining
    if card.type is AuctionType.AUCTION:
        gems_after_round -= min(card.value, len(info.auction_gems))

    undrawn_cards = info.auction_cards_remaining
    future_deck_gems = sum(
        future.value * count
        for future, count in undrawn_cards.items()
        if future.type is AuctionType.AUCTION
    )
    deck_exposure = (
        min(1.0, gems_after_round / future_deck_gems) if future_deck_gems else 0.0
    )

    future_prizes = (
        sum(market.gem_values[g] * market.expected_supply[g] for g in STANDARD_GEMS)
        - current_lot_value
        + sum(mission.value for mission in info.available_missions) / 2
    )
    table_coins = float(
        info.self_player.coins + sum(p.coins for p in info.other_players)
    )
    for future, count in undrawn_cards.items():
        if future.type is AuctionType.INVEST:
            future_prizes += future.value * count * deck_exposure
        elif future.type is AuctionType.LOAN:
            table_coins += future.value * count * deck_exposure

    if table_coins <= 0.0:
        raw_value = 1.0
    else:
        scarcity_ratio = max(0.0, future_prizes / table_coins - 1.0)
        raw_value = 1.0 + cfg.coin_scarcity_gain * scarcity_ratio

    premium = raw_value - 1.0
    if premium > 0.0 and gems_after_round < cfg.endgame_gem_threshold:
        progress = max(gems_after_round, 0) / cfg.endgame_gem_threshold
        taper = cfg.endgame_premium_floor + (1.0 - cfg.endgame_premium_floor) * progress
        raw_value = 1.0 + premium * taper

    return min(raw_value, cfg.max_coin_value)


def poisson_at_least(target: int, expected_supply: float, win_share: float) -> float:
    """Return P(X >= target) for a Poisson approximation of future gem wins."""
    if target <= 0:
        return 1.0
    if expected_supply < target:
        return 0.0
    rate = expected_supply * win_share
    term = exp(-rate)
    cdf_below = 0.0
    for k in range(target):
        cdf_below += term
        term *= rate / (k + 1)
    return max(0.0, 1.0 - cdf_below)


def _wildcard_completion_prob(
    mission: MissionCard,
    leftover: Mapping[GemType, int],
    supply: Mapping[GemType, float],
    win_share: float,
) -> float:
    """Estimate completion probability for the wildcard portion of a mission."""
    wild_count = mission.gem_requirements.get(GemType.WILD_CARD, 0)
    if not wild_count:
        return 1.0

    req = mission.wild_card_requirement
    if req is WildCardRequirement.SAME:
        return max(
            poisson_at_least(wild_count - leftover[g], supply[g], win_share)
            for g in STANDARD_GEMS
        )
    if req is WildCardRequirement.DIFFERENT:
        ranked = sorted(
            (
                poisson_at_least(1 - leftover[g], supply[g], win_share)
                for g in STANDARD_GEMS
            ),
            reverse=True,
        )
        return prod(ranked[:wild_count])
    if req is WildCardRequirement.PAIRS:
        existing_pairs = sum(leftover[g] // 2 for g in STANDARD_GEMS)
        pairs_remaining = max(0, wild_count // 2 - existing_pairs)
        ranked = sorted(
            (
                poisson_at_least(2 - leftover[g] % 2, supply[g], win_share)
                for g in STANDARD_GEMS
            ),
            reverse=True,
        )
        return prod(ranked[:pairs_remaining])
    return 0.0


def mission_completion_prob(
    mission: MissionCard,
    owned: Mapping[GemType, int],
    supply: Mapping[GemType, float],
    win_share: float,
) -> float:
    """Estimate the probability that a player completes a mission."""
    leftover = dict(owned)
    fixed_prob = 1.0
    for gem, required in mission.gem_requirements.items():
        if gem is GemType.WILD_CARD:
            continue
        fixed_prob *= poisson_at_least(required - leftover[gem], supply[gem], win_share)
        leftover[gem] = max(0, leftover[gem] - required)

    return fixed_prob * _wildcard_completion_prob(mission, leftover, supply, win_share)


def mission_incremental_value(
    info: BotInfo,
    lot: Sequence[GemType],
    supply: Mapping[GemType, float],
    cfg: PricingConfig = DEFAULT_PRICING,
) -> float:
    """Compute expected mission points gained from acquiring the current gem lot."""
    post_lot_supply = dict(supply)
    for gem in lot:
        post_lot_supply[gem] = max(0.0, post_lot_supply[gem] - 1.0)

    me = info.self_player
    lot_counts = Counter(lot)
    owned_before = {g: me.won_gems.get(g, 0) for g in STANDARD_GEMS}
    owned_after = {g: owned_before[g] + lot_counts[g] for g in STANDARD_GEMS}
    table_coins = me.coins + sum(p.coins for p in info.other_players)
    self_share = me.coins / table_coins if table_coins else 0.0

    rival_states = [
        (
            {g: opp.won_gems.get(g, 0) for g in STANDARD_GEMS},
            opp.coins / table_coins if table_coins else 0.0,
        )
        for opp in info.other_players
    ]

    expected_gain = 0.0
    for mission in info.available_missions:
        unclaimed_factor = 1.0
        for rival_owned, rival_share in rival_states:
            rival_prob = mission_completion_prob(
                mission, rival_owned, supply, rival_share
            )
            unclaimed_factor *= 1.0 - cfg.rival_mission_contest * rival_prob

        prob_before = unclaimed_factor * mission_completion_prob(
            mission, owned_before, supply, self_share
        )
        if mission.is_satisfied(owned_after):
            prob_after = 1.0
        else:
            prob_after = unclaimed_factor * mission_completion_prob(
                mission, owned_after, post_lot_supply, self_share
            )
        expected_gain += mission.value * (prob_after - prob_before)
    return expected_gain


def evaluate_auction_terms(
    info: BotInfo,
    card: AuctionCard,
    market: MarketContext,
    cfg: PricingConfig = DEFAULT_PRICING,
) -> AuctionTerms:
    """Translate the active auction card into prize value, coin cost, and cap."""
    is_gem_lot = card.type is AuctionType.AUCTION
    lot = info.auction_gems[: card.value] if is_gem_lot else ()
    lot_ev = sum(market.gem_values[gem] for gem in lot) if is_gem_lot else 0.0
    coin_val = marginal_coin_value(info, card, market, lot_ev, cfg)
    reference_scale = max(lot_ev if is_gem_lot else float(card.value), 1.0)

    if card.type is AuctionType.INVEST:
        gross_value = float(card.value)
        coin_cost = coin_val - 1.0
    elif card.type is AuctionType.LOAN:
        gross_value = (coin_val - 1.0) * card.value
        coin_cost = coin_val
    else:
        gross_value = lot_ev + mission_incremental_value(
            info, lot, market.expected_supply, cfg
        )
        coin_cost = coin_val

    max_legal_bid = info.self_player.coins + (
        card.value if card.type is AuctionType.LOAN else 0
    )
    return AuctionTerms(
        gross_value=gross_value,
        marginal_coin_cost=coin_cost,
        reference_scale=reference_scale,
        max_legal_bid=max_legal_bid,
    )


def mission_color_pressure(
    player: PlayerInfo,
    missions: Sequence[MissionCard],
    divisor: float = 10.0,
) -> dict[GemType, float]:
    """Score how badly a player still needs each non-wildcard color for missions."""
    pressure = dict.fromkeys(STANDARD_GEMS, 0.0)
    owned = {gem: player.won_gems.get(gem, 0) for gem in STANDARD_GEMS}
    for mission in missions:
        if mission.is_satisfied(owned):
            continue
        for gem, required in mission.gem_requirements.items():
            if gem is GemType.WILD_CARD or gem not in pressure:
                continue
            shortfall = required - owned[gem]
            if shortfall > 0:
                pressure[gem] += shortfall * mission.value / divisor
    return pressure


def _richest_opponent(info: BotInfo) -> PlayerInfo:
    """Return the opponent with the largest coin stack (tie-breaking on leader)."""
    leader = info.previous_winner_id
    return max(
        info.other_players,
        key=lambda opp: (opp.coins, opp.id == leader),
    )


def select_reveal_index(
    info: BotInfo, cfg: PricingConfig = DEFAULT_PRICING
) -> int:
    """Choose the hidden gem reveal that shifts future prices in our favor."""
    hand = info.self_player.hidden_gems
    if not hand:
        return 0

    supply = expected_remaining_supply(info)
    hand_counts = Counter(hand)
    pool_size = sum(info.unknown_gems.values()) + len(hand)
    hidden_draws = pool_size - info.gem_deck_count

    own_pressure = mission_color_pressure(
        info.self_player, info.available_missions, cfg.mission_urgency_divisor
    )
    rival_pressure = mission_color_pressure(
        _richest_opponent(info), info.available_missions, cfg.mission_urgency_divisor
    )

    color_scores: dict[GemType, float] = {}
    for gem in hand_counts:
        shown = info.value_display.get(gem, 0)
        copies = info.unknown_gems.get(gem, 0) + hand_counts[gem]
        before = expected_chart_payout(
            info.value_chart, shown, copies, pool_size, hidden_draws
        )
        after = expected_chart_payout(
            info.value_chart,
            shown + 1,
            copies - 1,
            pool_size - 1,
            hidden_draws - 1,
        )
        price_shift = (after - before) * supply.get(gem, 0.0)
        alignment = 1.0 + own_pressure[gem] - rival_pressure[gem]
        color_scores[gem] = price_shift * alignment

    return min(range(len(hand)), key=lambda idx: (color_scores[hand[idx]], idx))
