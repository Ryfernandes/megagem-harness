"""What a lot is worth to Blue5GD, and what one coin costs.

Gem values come from MicahBot's exact formula. The new part is the coin price
and how missions, investments, and loans use that price.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from bot_definitions.blue5gd_config import Blue5Config, chart_family
from bot_definitions.micah_strategy import immediate_mission_bonus
from bots import BotInfo
from constants import STANDARD_GEMS, AuctionType, GemType
from lib import MissionCard

# Few gems left: switch from the "early" bid prior to the "late" one.
LATE_GEMS = 4


def is_late(info: BotInfo) -> bool:
    """Return True when the game is close enough to the end to use late priors."""
    return info.gems_remaining <= LATE_GEMS


def coin_price(info: BotInfo, config: Blue5Config) -> float:
    """Return how many points one coin is worth right now. Never below 1."""
    family = chart_family(info.value_chart)
    premium = config.coin_k(family)
    face_at = config.endgame_face_gems
    if face_at <= 0.0 or info.gems_remaining <= 0:
        scale = 0.0
    elif info.gems_remaining >= face_at:
        scale = 1.0
    else:
        scale = info.gems_remaining / face_at
    return max(1.0, 1.0 + premium * scale)


def _filled_fraction(mission: MissionCard, collection: Mapping[GemType, int]) -> float:
    """Share of a mission's gem slots that this collection already covers.

    Fixed colors are exact. Extra gems of any color fill wildcard slots one
    for one. Pair and same-color wildcards are only approximate. The mission
    weight knob can turn this credit down if that approximation bids too much.
    """
    needed = 0
    filled = 0
    leftover = 0
    for gem in STANDARD_GEMS:
        required = int(mission.gem_requirements.get(gem, 0))
        owned = int(collection.get(gem, 0))
        needed += required
        filled += min(owned, required)
        leftover += max(0, owned - required)
    wild = int(mission.gem_requirements.get(GemType.WILD_CARD, 0))
    needed += wild
    if needed <= 0:
        return 1.0
    filled += min(wild, leftover)
    return min(1.0, filled / float(needed))


def mission_progress_points(
    won_gems: Mapping[GemType, int],
    added_gems: Sequence[GemType],
    available_missions: Sequence[MissionCard],
) -> float:
    """Points of progress on missions this lot moves forward but does not finish."""
    if not added_gems or not available_missions:
        return 0.0
    after = dict(won_gems)
    for gem in added_gems:
        after[gem] = after.get(gem, 0) + 1
    total = 0.0
    for mission in available_missions:
        if mission.is_satisfied(won_gems) or mission.is_satisfied(after):
            continue
        before = _filled_fraction(mission, won_gems)
        after_fraction = _filled_fraction(mission, after)
        total += float(mission.value) * max(0.0, after_fraction - before)
    return total


def lot_gems(info: BotInfo) -> tuple[GemType, ...]:
    """Gems a treasure auction would award. Empty for invest and loan cards."""
    card = info.current_auction
    if card is None or card.type is not AuctionType.AUCTION:
        return ()
    return tuple(info.auction_gems[: card.value])


def own_prize(
    info: BotInfo,
    config: Blue5Config,
    gem_evs: Mapping[GemType, float],
) -> tuple[float, float]:
    """Return (prize in points, points given up per coin of the bid).

    Treasure: gem value plus missions, and you pay the full coin price.
    Invest: the bonus is the prize. The bid comes back at the end, so you only
    pay the extra value of locking the coins up (coin price minus 1).
    Loan: you must repay the loan at face value, so the prize is only the extra
    value of holding those coins until then. The bid itself costs the full coin price.
    """
    card = info.current_auction
    if card is None:
        return 0.0, 1.0
    price = coin_price(info, config)
    if card.type is AuctionType.INVEST:
        return float(card.value), max(0.0, price - 1.0)
    if card.type is AuctionType.LOAN:
        return (price - 1.0) * float(card.value), price

    lot = lot_gems(info)
    gem_value = sum(gem_evs.get(gem, 0.0) for gem in lot)
    finished = float(
        immediate_mission_bonus(
            info.self_player.won_gems, lot, info.available_missions
        )
    )
    progress = mission_progress_points(
        info.self_player.won_gems, lot, info.available_missions
    )
    prize = (
        gem_value
        + config.mission_now_weight * finished
        + config.mission_progress_weight * progress
    )
    return prize, price


def denial_points(info: BotInfo, config: Blue5Config, coin_cost: float) -> float:
    """Extra prize for blocking a rival's mission, capped in coins.

    The cap is `denial_max_coins` times what one coin costs, so denial cannot
    talk us into a huge bid. YaleBot does not do this. The search decides
    whether we should.
    """
    if config.denial_weight <= 0.0 or coin_cost <= 0.0:
        return 0.0
    lot = lot_gems(info)
    if not lot:
        return 0.0
    rival = 0.0
    for opponent in info.other_players:
        rival += float(
            immediate_mission_bonus(
                opponent.won_gems, lot, info.available_missions
            )
        )
        rival += mission_progress_points(
            opponent.won_gems, lot, info.available_missions
        )
    if rival <= 0.0:
        return 0.0
    return min(config.denial_max_coins * coin_cost, config.denial_weight * rival)


def public_scale(info: BotInfo, public_evs: Mapping[GemType, float]) -> float:
    """Prize size used to read an opponent's bid as a fraction. At least 1."""
    card = info.current_auction
    if card is None:
        return 1.0
    if card.type is not AuctionType.AUCTION:
        return max(1.0, float(card.value))
    lot = info.auction_gems[: card.value]
    return max(1.0, sum(public_evs.get(gem, 0.0) for gem in lot))
