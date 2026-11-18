"""Pure, typed strategy helpers for MicahBot (Milestone 2 analytic core)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

from bots import BotInfo, PlayerInfo
from constants import STANDARD_GEMS, AuctionType, GemType, WildCardRequirement
from lib import AuctionCard, MissionCard


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Tunable parameters for MicahBot's analytic and rollout valuation."""

    liquidity_k0: float = 1.05
    spend_per_round: float = 5.5
    scarcity_opp_boost: float = 0.35
    tie_penalty_base: float = 0.55
    mission_progress_discount: float = 0.25
    opp_treasure_bid_ratio: float = 0.78
    opp_invest_bid_ratio: float = 0.75
    opp_loan_bid_ratio: float = 0.32
    win_prob_spread_ratio: float = 0.24
    bid_prior_strength: float = 12.0
    denial_weight: float = 1.0
    bid_inference_strength: float = 0.40
    reveal_inference_strength: float = 0.0
    valuation_inference_blend: float = 0.40
    reveal_leak_penalty: float = 0.40
    reveal_directional_weight: float = 1.0


DEFAULT_STRATEGY_CONFIG = StrategyConfig()


def legal_bid_cap(info: BotInfo) -> int:
    """Return the maximum legal bid in coins for the current auction."""
    coins = int(info.self_player.coins)
    auction = info.current_auction
    if auction is not None and auction.type is AuctionType.LOAN:
        return max(0, int(coins + auction.value))
    return max(0, coins)


def clamp_bid(bid: int, max_bid: int) -> int:
    """Clamp a bid to [0, max_bid] and guarantee a plain Python int."""
    if max_bid <= 0 or bid <= 0:
        return 0
    if bid >= max_bid:
        return int(max_bid)
    return int(bid)


def tie_priority_rank(
    player_id: int,
    previous_winner_id: int | None,
    num_players: int,
) -> int | None:
    """Return tie priority rank (0 = first, num_players - 1 = last), or None in R1."""
    if previous_winner_id is None or num_players <= 0:
        return None
    return (player_id - previous_winner_id - 1) % num_players


def beats_on_tie(
    player_id: int,
    other_id: int,
    previous_winner_id: int | None,
    num_players: int,
) -> bool | None:
    """Return True if player_id beats other_id on ties, False if not, None if R1."""
    rank_self = tie_priority_rank(player_id, previous_winner_id, num_players)
    rank_other = tie_priority_rank(other_id, previous_winner_id, num_players)
    if rank_self is None or rank_other is None:
        return None
    return rank_self < rank_other


def tie_order_penalty(
    self_id: int,
    previous_winner_id: int | None,
    num_players: int,
    rounds_left_after: float,
    base_penalty: float = DEFAULT_STRATEGY_CONFIG.tie_penalty_base,
) -> float:
    """Penalty for becoming previous_winner_id (moving to last in tie priority)."""
    if rounds_left_after <= 0.0 or num_players <= 1 or base_penalty <= 0.0:
        return 0.0
    rank = tie_priority_rank(self_id, previous_winner_id, num_players)
    if rank is None:
        drop_fraction = 0.5
    else:
        drop_fraction = float(num_players - 1 - rank) / float(num_players - 1)
    horizon_scale = min(1.0, rounds_left_after / 4.0)
    return base_penalty * drop_fraction * horizon_scale


@lru_cache(maxsize=8192)
def hypergeom_count_probs(
    known_display: int,
    pool_color: int,
    pool_total: int,
    hidden_draws: int,
) -> tuple[float, ...]:
    """Return P(final display count == x) for x in 0..6 via exact hypergeometric."""
    base = max(0, min(6, known_display))
    probs = [0.0] * 7
    if (
        pool_total <= 0
        or hidden_draws <= 0
        or hidden_draws > pool_total
        or pool_color <= 0
    ):
        probs[base] = 1.0
        return tuple(probs)

    total_ways = math.comb(pool_total, hidden_draws)
    non_color = pool_total - pool_color
    max_dx = min(pool_color, hidden_draws)
    for dx in range(max_dx + 1):
        if non_color >= hidden_draws - dx:
            prob = (
                math.comb(pool_color, dx) * math.comb(non_color, hidden_draws - dx)
            ) / total_ways
            probs[min(6, base + dx)] += prob
    return tuple(probs)


@lru_cache(maxsize=8192)
def expected_chart_value(
    chart_values: tuple[int, ...],
    known_display: int,
    pool_color: int,
    pool_total: int,
    hidden_draws: int,
) -> float:
    """Compute exact expected coin value of one gem given display/pool counts."""
    probs = hypergeom_count_probs(known_display, pool_color, pool_total, hidden_draws)
    return sum(p * float(v) for p, v in zip(probs, chart_values))


def gem_expected_values(info: BotInfo) -> dict[GemType, float]:
    """Return exact hypergeometric expected value per gem type using private hand."""
    chart_values = info.value_chart.values
    pool_total = sum(info.unknown_gems.get(gem, 0) for gem in STANDARD_GEMS)
    hidden_draws = sum(player.hidden_gem_count for player in info.other_players)
    return {
        gem: expected_chart_value(
            chart_values,
            info.value_display.get(gem, 0) + info.self_player.hidden_gems.count(gem),
            info.unknown_gems.get(gem, 0),
            pool_total,
            hidden_draws,
        )
        for gem in STANDARD_GEMS
    }


def public_gem_expected_values(info: BotInfo) -> dict[GemType, float]:
    """Return hypergeometric expected value per gem using public display only."""
    chart_values = info.value_chart.values
    self_hidden = info.self_player.hidden_gems
    self_hidden_count = len(self_hidden)
    pool_total = (
        sum(info.unknown_gems.get(gem, 0) for gem in STANDARD_GEMS) + self_hidden_count
    )
    hidden_draws = (
        sum(player.hidden_gem_count for player in info.other_players)
        + self_hidden_count
    )
    return {
        gem: expected_chart_value(
            chart_values,
            info.value_display.get(gem, 0),
            info.unknown_gems.get(gem, 0) + self_hidden.count(gem),
            pool_total,
            hidden_draws,
        )
        for gem in STANDARD_GEMS
    }


def public_value_shift_from_counts(
    chart_values: tuple[int, ...],
    public_display_count: int,
    public_pool_color: int,
    public_pool_total: int,
    total_hidden_hands: int,
) -> float:
    """Change in a public observer's EV of a color if one hidden copy is revealed."""
    if public_pool_color <= 0 or public_pool_total <= 0 or total_hidden_hands <= 0:
        return 0.0
    ev_before = expected_chart_value(
        chart_values,
        public_display_count,
        public_pool_color,
        public_pool_total,
        total_hidden_hands,
    )
    ev_after = expected_chart_value(
        chart_values,
        min(6, public_display_count + 1),
        public_pool_color - 1,
        public_pool_total - 1,
        total_hidden_hands - 1,
    )
    return ev_after - ev_before


def public_value_shift(info: BotInfo, color: GemType) -> float:
    """Change in a public observer's EV of color if one hidden copy is revealed."""
    self_hidden = info.self_player.hidden_gems
    self_hidden_count = len(self_hidden)
    public_pool_total = (
        sum(info.unknown_gems.get(gem, 0) for gem in STANDARD_GEMS) + self_hidden_count
    )
    public_pool_color = info.unknown_gems.get(color, 0) + self_hidden.count(color)
    total_hidden_hands = (
        sum(player.hidden_gem_count for player in info.other_players)
        + self_hidden_count
    )
    return public_value_shift_from_counts(
        info.value_chart.values,
        info.value_display.get(color, 0),
        public_pool_color,
        public_pool_total,
        total_hidden_hands,
    )


def expected_future_supply(info: BotInfo, color: GemType) -> float:
    """Expected remaining auction supply of color (face-up + expected deck share)."""
    face_up = float(info.auction_gems.count(color))
    total_unknown = sum(info.unknown_gems.get(gem, 0) for gem in STANDARD_GEMS)
    if total_unknown <= 0:
        return face_up
    unknown_color = info.unknown_gems.get(color, 0)
    return face_up + (float(unknown_color) * float(info.gem_deck_count)) / float(
        total_unknown
    )


def expected_future_supplies(info: BotInfo) -> dict[GemType, float]:
    """Expected remaining auction supply for every standard gem color."""
    return {gem: expected_future_supply(info, gem) for gem in STANDARD_GEMS}


def expected_future_supplies_after_current(info: BotInfo) -> dict[GemType, float]:
    """Expected remaining auction supply of each color after current lot resolves."""
    supplies = expected_future_supplies(info)
    card = info.current_auction
    if card is not None and card.type is AuctionType.AUCTION:
        for gem in info.auction_gems[: min(card.value, len(info.auction_gems))]:
            supplies[gem] = max(0.0, supplies[gem] - 1.0)
    return supplies


def estimate_rounds_left_after(info: BotInfo) -> float:
    """Estimate expected number of auction rounds remaining after current_auction."""
    card = info.current_auction
    consumed_now = (
        min(card.value, info.gems_remaining)
        if (card is not None and card.type is AuctionType.AUCTION)
        else 0
    )
    gems_after = max(0, info.gems_remaining - consumed_now)
    if gems_after == 0:
        return 0.0

    total_cards = sum(info.auction_cards_remaining.values())
    future_treasure_gems = sum(
        c.value * count
        for c, count in info.auction_cards_remaining.items()
        if c.type is AuctionType.AUCTION
    )
    if total_cards <= 0 or future_treasure_gems <= 0:
        return max(1.0, float(gems_after) / 1.25)

    avg_gems_per_card = float(future_treasure_gems) / float(total_cards)
    estimated = float(gems_after) / avg_gems_per_card
    return max(1.0, min(float(total_cards), estimated))


def player_known_score(player: PlayerInfo, gem_evs: Mapping[GemType, float]) -> float:
    """Compute a player's expected terminal score from current known holdings."""
    gem_score = sum(
        float(player.won_gems.get(gem, 0)) * gem_evs.get(gem, 0.0)
        for gem in STANDARD_GEMS
    )
    mission_score = sum(float(m.value) for m in player.mission_cards)
    finance_score = sum(float(c.end_game_value) for c in player.auction_cards)
    return (
        float(player.coins)
        + float(player.invested_coins)
        + gem_score
        + mission_score
        + finance_score
    )


def all_known_scores(
    info: BotInfo, gem_evs: Mapping[GemType, float]
) -> dict[int, float]:
    """Return expected current known terminal score keyed by player ID."""
    scores = {info.self_player.id: player_known_score(info.self_player, gem_evs)}
    for opp in info.other_players:
        scores[opp.id] = player_known_score(opp, gem_evs)
    return scores


def liquidity_surplus(
    coins: float,
    rounds_left_after: float,
    opponent_coins: Sequence[int],
    avg_lot_ev: float = 10.0,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> float:
    """Concave expected point surplus above face value from holding spendable coins."""
    if coins <= 0.0 or rounds_left_after <= 0.0:
        return 0.0

    opp_mean = (
        sum(opponent_coins) / float(len(opponent_coins)) if opponent_coins else 0.0
    )
    # Coins buy slightly more surplus when opponents are poorer than normal.
    scarcity_adj = 1.0 + config.scarcity_opp_boost * max(
        -0.5, min(1.0, (12.0 - opp_mean) / 12.0)
    )
    k_eff = max(0.0, config.liquidity_k0 * scarcity_adj)

    if config.spend_per_round <= 0.0:
        return k_eff * coins * min(1.0, rounds_left_after / 6.0)

    lot_scale = max(0.75, min(1.35, avg_lot_ev / 10.0))
    scale = max(4.0, rounds_left_after * config.spend_per_round * lot_scale)
    max_surplus = k_eff * scale
    return max_surplus * (1.0 - math.exp(-coins / scale))


def future_coin_value(
    coins: float,
    rounds_left_after: float,
    opponent_coins: Sequence[int],
    avg_lot_ev: float = 10.0,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> float:
    """Total expected value (face value + concave liquidity surplus) of coins."""
    return float(coins) + liquidity_surplus(
        coins, rounds_left_after, opponent_coins, avg_lot_ev, config
    )


def projected_terminal_scores(
    info: BotInfo,
    gem_evs: Mapping[GemType, float],
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> dict[int, float]:
    """Project each player's terminal score including remaining coin option value."""
    rounds_left = estimate_rounds_left_after(info)
    avg_ev = sum(gem_evs.values()) / float(len(STANDARD_GEMS)) if gem_evs else 10.0
    all_players: tuple[PlayerInfo, ...] = (info.self_player, *info.other_players)
    projected: dict[int, float] = {}
    for player in all_players:
        other_coins = [p.coins for p in all_players if p.id != player.id]
        base = player_known_score(player, gem_evs)
        surplus = liquidity_surplus(
            float(player.coins), rounds_left, other_coins, avg_ev, config
        )
        projected[player.id] = base + surplus
    return projected


def immediate_mission_bonus(
    won_gems: Mapping[GemType, int],
    added_gems: Sequence[GemType],
    available_missions: Sequence[MissionCard],
) -> int:
    """Total bonus of all unclaimed missions newly satisfied by adding added_gems."""
    if not available_missions or not added_gems:
        return 0
    after = dict(won_gems)
    for gem in added_gems:
        after[gem] = after.get(gem, 0) + 1
    return sum(
        mission.value
        for mission in available_missions
        if not mission.is_satisfied(won_gems) and mission.is_satisfied(after)
    )


def _mission_progress_and_missing(
    mission: MissionCard, collection: Mapping[GemType, int]
) -> tuple[float, int, dict[GemType, int]]:
    """Return (progress_fraction, missing_gems_count, missing_fixed_gems)."""
    fixed_req = 0
    fixed_met = 0
    missing_fixed: dict[GemType, int] = {}
    remaining: list[int] = []
    for gem in STANDARD_GEMS:
        req = mission.gem_requirements.get(gem, 0)
        owned = collection.get(gem, 0)
        fixed_req += req
        fixed_met += min(owned, req)
        if owned < req:
            missing_fixed[gem] = req - owned
        remaining.append(max(0, owned - req))

    wc_req = mission.gem_requirements.get(GemType.WILD_CARD, 0)
    wc_met = 0.0
    wc_missing = 0
    if wc_req > 0:
        match mission.wild_card_requirement:
            case None:
                got = min(sum(remaining), wc_req)
                wc_met = float(got)
                wc_missing = wc_req - got
            case WildCardRequirement.SAME:
                got = min(max(remaining, default=0), wc_req)
                wc_met = float(got)
                wc_missing = wc_req - got
            case WildCardRequirement.DIFFERENT:
                got = min(sum(1 for count in remaining if count > 0), wc_req)
                wc_met = float(got)
                wc_missing = wc_req - got
            case WildCardRequirement.PAIRS:
                target_pairs = wc_req // 2
                pairs = sum(count // 2 for count in remaining)
                singles = sum(count % 2 for count in remaining)
                used_pairs = min(pairs, target_pairs)
                used_singles = min(singles, max(0, target_pairs - used_pairs))
                wc_met = float(used_pairs * 2) + 0.5 * float(used_singles)
                wc_missing = max(0, wc_req - used_pairs * 2 - used_singles)

    total_req = fixed_req + wc_req
    if total_req <= 0:
        return (1.0 if mission.is_satisfied(collection) else 0.0), 0, {}
    frac = min(1.0, (float(fixed_met) + wc_met) / float(total_req))
    missing_total = sum(missing_fixed.values()) + wc_missing
    return frac, missing_total, missing_fixed


def mission_progress_value(
    won_gems: Mapping[GemType, int],
    added_gems: Sequence[GemType],
    available_missions: Sequence[MissionCard],
    future_supplies_after: Mapping[GemType, float] | None = None,
    rounds_left_after: float = 0.0,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> float:
    """Discounted progress value toward unclaimed missions not yet completed."""
    if (
        rounds_left_after <= 0.0
        or config.mission_progress_discount <= 0.0
        or not available_missions
        or not added_gems
    ):
        return 0.0

    after = dict(won_gems)
    for gem in added_gems:
        after[gem] = after.get(gem, 0) + 1

    total = 0.0
    for mission in available_missions:
        if mission.is_satisfied(won_gems) or mission.is_satisfied(after):
            continue
        frac_before, _, _ = _mission_progress_and_missing(mission, won_gems)
        frac_after, missing_after, missing_fixed = _mission_progress_and_missing(
            mission, after
        )
        delta = max(0.0, frac_after - frac_before)
        if delta <= 0.0:
            continue

        # A mission requiring more copies of a sold-out gem is dead.
        if future_supplies_after is not None:
            dead = any(
                future_supplies_after.get(gem, 0.0) + 1e-9 < float(need)
                for gem, need in missing_fixed.items()
            )
            if dead:
                continue

        horizon_feasibility = min(
            1.0, (rounds_left_after * 1.25) / float(max(1, missing_after))
        )
        total += (
            config.mission_progress_discount
            * horizon_feasibility
            * float(mission.value)
            * delta
        )
    return total


def _average_remaining_gem_ev(gem_evs: Mapping[GemType, float]) -> float:
    if not gem_evs:
        return 10.0
    return sum(gem_evs.values()) / float(len(gem_evs))


def auction_win_net_gain(
    info: BotInfo,
    card: AuctionCard,
    bid: int,
    gem_evs: Mapping[GemType, float],
    future_supplies_after: Mapping[GemType, float],
    rounds_left_after: float,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> float:
    """Net utility of winning current auction at `bid` vs losing at 0 cost."""
    coins = float(info.self_player.coins)
    opp_coins = [player.coins for player in info.other_players]
    avg_ev = _average_remaining_gem_ev(gem_evs)
    base_cash_val = future_coin_value(
        coins, rounds_left_after, opp_coins, avg_ev, config
    )
    tie_pen = tie_order_penalty(
        info.self_player.id,
        info.previous_winner_id,
        info.num_players,
        rounds_left_after,
        config.tie_penalty_base,
    )

    if card.type is AuctionType.AUCTION:
        gems = info.auction_gems[: min(card.value, len(info.auction_gems))]
        gem_val = sum(gem_evs.get(gem, 0.0) for gem in gems)
        imm_mission = float(
            immediate_mission_bonus(
                info.self_player.won_gems, gems, info.available_missions
            )
        )
        prog_mission = mission_progress_value(
            info.self_player.won_gems,
            gems,
            info.available_missions,
            future_supplies_after,
            rounds_left_after,
            config,
        )
        after_cash_val = future_coin_value(
            coins - float(bid), rounds_left_after, opp_coins, avg_ev, config
        )
        cash_cost = base_cash_val - after_cash_val
        return gem_val + imm_mission + prog_mission - cash_cost - tie_pen

    if card.type is AuctionType.INVEST:
        after_cash_val = future_coin_value(
            coins - float(bid), rounds_left_after, opp_coins, avg_ev, config
        )
        liquidity_lost = (base_cash_val - after_cash_val) - float(bid)
        return float(card.value) - liquidity_lost - tie_pen

    # Loan: winning gives L - bid cash now and -L at scoring -> net cash + L - b.
    if bid >= card.value:
        return -1e9
    after_cash_val = future_coin_value(
        coins + float(card.value - bid), rounds_left_after, opp_coins, avg_ev, config
    )
    return (after_cash_val - float(card.value)) - base_cash_val - tie_pen


def _opponent_reference_bid(
    info: BotInfo,
    opp: PlayerInfo,
    card: AuctionCard,
    public_evs: Mapping[GemType, float],
    config: StrategyConfig,
) -> tuple[float, float]:
    """Return (expected_opp_bid, spread_scale) for a single opponent."""
    opp_cap = opp.coins + (card.value if card.type is AuctionType.LOAN else 0)
    if card.type is AuctionType.AUCTION:
        gems = info.auction_gems[: min(card.value, len(info.auction_gems))]
        pub_ev = sum(public_evs.get(gem, 0.0) for gem in gems)
        opp_mission = float(
            immediate_mission_bonus(opp.won_gems, gems, info.available_missions)
        )
        ref = pub_ev + 0.55 * opp_mission
        target = min(float(opp_cap), max(0.0, config.opp_treasure_bid_ratio * ref))
        return target, max(4.0, ref)

    if card.type is AuctionType.INVEST:
        ref = float(card.value)
        target = min(float(opp_cap), config.opp_invest_bid_ratio * ref)
        return target, max(4.0, ref)

    scarcity = max(0.0, 1.0 - float(opp.coins) / 20.0)
    loan_frac = config.opp_loan_bid_ratio * (0.65 + 0.75 * scarcity)
    target = min(float(opp_cap), float(card.value) * loan_frac)
    return target, max(4.0, float(card.value) * 0.5)


def estimate_win_probability(
    info: BotInfo,
    card: AuctionCard,
    bid: int,
    public_evs: Mapping[GemType, float],
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> float:
    """Estimate P(win | bid) using legal caps, tie order, and public EV."""
    if not info.other_players:
        return 1.0

    num_players = info.num_players
    prev_winner = info.previous_winner_id
    self_id = info.self_player.id

    prob = 1.0
    for opp in info.other_players:
        opp_cap = opp.coins + (card.value if card.type is AuctionType.LOAN else 0)
        if bid > opp_cap:
            continue

        beats = beats_on_tie(self_id, opp.id, prev_winner, num_players)
        if bid == opp_cap and beats is True:
            continue
        if opp_cap <= 0:
            if bid > 0:
                continue
            if beats is False:
                return 0.0
            prob *= 0.5
            continue

        opp_target, ref_scale = _opponent_reference_bid(
            info, opp, card, public_evs, config
        )
        tie_shift = 0.0 if beats is None else (0.5 if beats else -0.5)
        spread = max(1.25, config.win_prob_spread_ratio * ref_scale)
        z = max(-30.0, min(30.0, (float(bid) + tie_shift - opp_target) / spread))
        prob *= 1.0 / (1.0 + math.exp(-z))
    return prob


def choose_bid(
    info: BotInfo,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> int:
    """Select the integer bid maximizing expected utility within legal limits."""
    card = info.current_auction
    if card is None:
        return 0
    max_legal = legal_bid_cap(info)
    if max_legal <= 0:
        return 0

    max_candidate = (
        min(max_legal, card.value - 1) if card.type is AuctionType.LOAN else max_legal
    )
    if max_candidate <= 0:
        return 0

    gem_evs = gem_expected_values(info)
    public_evs = public_gem_expected_values(info)
    future_supplies_after = expected_future_supplies_after_current(info)
    rounds_left_after = estimate_rounds_left_after(info)

    best_bid = 0
    best_utility = -1e18
    for bid in range(max_candidate + 1):
        net_gain = auction_win_net_gain(
            info, card, bid, gem_evs, future_supplies_after, rounds_left_after, config
        )
        if net_gain <= 0.0 and bid > 0:
            break
        p_win = estimate_win_probability(info, card, bid, public_evs, config)
        utility = p_win * net_gain
        if utility > best_utility + 1e-12:
            best_utility = utility
            best_bid = bid

    if best_utility <= 0.0:
        return 0
    return clamp_bid(best_bid, max_legal)


def _color_mission_interest(
    player: PlayerInfo,
    color: GemType,
    available_missions: Sequence[MissionCard],
    future_supplies: Mapping[GemType, float],
) -> float:
    """Estimate how strongly a player wants `color` to claim available missions."""
    if not available_missions:
        return 0.0
    after = dict(player.won_gems)
    after[color] = after.get(color, 0) + 1

    interest = 0.0
    for mission in available_missions:
        if mission.is_satisfied(player.won_gems):
            continue
        if mission.is_satisfied(after):
            interest += float(mission.value) / 10.0
            continue
        frac_before, _, _ = _mission_progress_and_missing(mission, player.won_gems)
        frac_after, _, missing_fixed = _mission_progress_and_missing(mission, after)
        delta = max(0.0, frac_after - frac_before)
        if delta > 0.0 and all(
            future_supplies.get(gem, 0.0) + 1e-9 >= float(need)
            for gem, need in missing_fixed.items()
        ):
            interest += 0.35 * (float(mission.value) / 10.0) * delta
    return interest


def choose_reveal_index(
    info: BotInfo,
    bids: Mapping[int, int] | None = None,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> int:
    """Choose a hidden-hand gem index to reveal using future supply and value shift."""
    del bids  # Unused in Milestone 2; reserved for Milestone 4 inference.
    hidden = info.self_player.hidden_gems
    if not hidden or info.gems_remaining <= 0:
        return 0

    supplies = expected_future_supplies(info)
    next_gem = info.auction_gems[0] if info.auction_gems else None
    opp_max_coins = max((p.coins for p in info.other_players), default=0)
    self_buying_power = 1.0 if info.self_player.coins >= opp_max_coins else 0.7

    best_idx = 0
    best_score = -1e18
    cached_scores: dict[GemType, float] = {}

    for idx, gem in enumerate(hidden):
        if gem not in cached_scores:
            supply = supplies.get(gem, 0.0)
            if supply <= 1e-9:
                # Sold-out colors move nobody's valuation; reveal them first.
                cached_scores[gem] = 0.0
            else:
                shift = public_value_shift(info, gem)
                self_interest = self_buying_power + _color_mission_interest(
                    info.self_player, gem, info.available_missions, supplies
                )
                opp_interest = 0.55 + max(
                    (
                        (0.4 + 0.6 * min(1.0, float(opp.coins) / 10.0))
                        * _color_mission_interest(
                            opp, gem, info.available_missions, supplies
                        )
                        for opp in info.other_players
                    ),
                    default=0.0,
                )
                # Slightly raise weight when the color is immediately up next.
                immediacy = 1.25 if gem is next_gem else 1.0
                # If self_interest > opp_interest: prefer negative shift (cheaper);
                # if opp_interest > self_interest: prefer positive shift (rivals pay).
                directional = (
                    config.reveal_directional_weight
                    * (opp_interest - self_interest)
                    * shift
                    * min(1.5, supply / 1.5)
                    * immediacy
                )
                supply_leak_penalty = config.reveal_leak_penalty * supply
                cached_scores[gem] = directional - supply_leak_penalty

        score = cached_scores[gem]
        if score > best_score + 1e-12:
            best_score = score
            best_idx = idx

    return int(best_idx)
