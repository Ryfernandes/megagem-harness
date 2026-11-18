"""Milestone 3/5 bid choice for MicahBot.

Turns censored opponent bid memory, scaled late blocking, terminal-round
first-place odds, and deadline-bounded Monte Carlo rollouts into one integer
bid on top of the Milestone 2/4 analytic core.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping

from bot_definitions.micah_bid_model import (
    OpponentBidModel,
    lot_reference_value,
    phase_is_late,
    probability_we_win,
)
from bot_definitions.micah_endgame import denial_value, endgame_first_place_probs
from bot_definitions.micah_inference import OpponentHandBeliefs
from bot_definitions.micah_rollout import (
    compute_rollout_deadline,
    evaluate_candidates_rollout,
    select_rollout_candidates,
    should_run_rollout,
)
from bot_definitions.micah_strategy import (
    DEFAULT_STRATEGY_CONFIG,
    StrategyConfig,
    auction_win_net_gain,
    clamp_bid,
    estimate_rounds_left_after,
    expected_future_supplies_after_current,
    future_coin_value,
    gem_expected_values,
    legal_bid_cap,
    projected_terminal_scores,
    public_gem_expected_values,
    tie_order_penalty,
)
from bots import BotInfo
from constants import AuctionType, GemType
from lib import AuctionCard

# Hidden-hand draws used on the terminal treasure auction.
_ENDGAME_SAMPLES = 48
# Weight for terminal first-place probability blended with expected net gain.
_ENDGAME_WIN_WEIGHT = 25.0
# Only pay to deny rivals when we are within contention of first place.
_SELF_CONTENTION_GAP = 10.0
_UTILITY_TIE = 1e-9
_ROLLOUT_MAX_WORLDS = 24
_ROLLOUT_MIN_WORLDS = 12


def choose_bid_m3(
    info: BotInfo,
    model: OpponentBidModel,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
    rng: random.Random | None = None,
    hand_beliefs: OpponentHandBeliefs | None = None,
    *,
    use_rollouts: bool = False,
    timeout_seconds: float | None = None,
) -> int:
    """Return the best integer bid in 0..legal_cap using analytic + rollout."""
    start_time = time.perf_counter()
    card = info.current_auction
    our_cap = legal_bid_cap(info)
    if card is None or our_cap == 0:
        return 0

    if card.type is AuctionType.LOAN:
        max_candidate = min(our_cap, card.value - 1)
    else:
        max_candidate = our_cap
    if max_candidate <= 0:
        return 0

    if hand_beliefs is not None and config.valuation_inference_blend > 0.0:
        gem_evs = hand_beliefs.gem_expected_values_with_beliefs(
            info, inference_strength=config.valuation_inference_blend
        )
    else:
        gem_evs = gem_expected_values(info)
    public_evs = public_gem_expected_values(info)
    future_supplies_after = expected_future_supplies_after_current(info)
    rounds_left_after = estimate_rounds_left_after(info)
    late = phase_is_late(info)

    distributions: dict[int, tuple[tuple[int, float], ...]] = {}
    for opponent in info.other_players:
        opponent_cap = opponent.coins + (
            card.value if card.type is AuctionType.LOAN else 0
        )
        opponent_cap = max(0, opponent_cap)
        reference = lot_reference_value(info, opponent, public_evs)
        distributions[opponent.id] = model.bid_distribution(
            opponent.id,
            card.type,
            late,
            opponent.coins,
            reference,
            opponent_cap,
            config,
        )

    candidates = list(range(max_candidate + 1))
    draw = rng if rng is not None else random.Random(0)
    first_place: dict[int, float] | None = None
    if rounds_left_after == 0.0:
        hand_sampler = (
            hand_beliefs.sample_opponent_hands_with_beliefs
            if hand_beliefs is not None
            else None
        )
        first_place = endgame_first_place_probs(
            info,
            candidates,
            distributions,
            draw,
            samples=_ENDGAME_SAMPLES,
            hand_sampler=hand_sampler,
        )

    apply_denial = False
    if late and first_place is None:
        projected = projected_terminal_scores(info, gem_evs, config)
        best_proj = max(projected.values())
        if best_proj - projected[info.self_player.id] <= _SELF_CONTENTION_GAP:
            apply_denial = True

    our_id = info.self_player.id
    previous_winner_id = info.previous_winner_id
    num_players = info.num_players
    best_bid = 0
    best_utility = -math.inf
    candidate_utilities: dict[int, float] = {}

    for bid in candidates:
        net = _net_gain(
            info,
            card,
            bid,
            gem_evs,
            future_supplies_after,
            rounds_left_after,
            config,
        )
        if first_place is None and not apply_denial and net <= 0.0 and bid > 0:
            break
        win_chance = probability_we_win(
            bid,
            our_id,
            distributions,
            previous_winner_id,
            num_players,
        )
        if first_place is not None:
            utility = win_chance * net + _ENDGAME_WIN_WEIGHT * first_place[bid]
        elif apply_denial and bid > 0:
            utility = win_chance * net + denial_value(
                info, bid, distributions, public_evs, config, gem_evs=gem_evs
            )
        else:
            utility = win_chance * net
        candidate_utilities[bid] = utility
        if utility > best_utility + _UTILITY_TIE:
            best_utility = utility
            best_bid = bid

    analytic_bid = (
        0
        if (first_place is None and best_utility <= 0.0)
        else clamp_bid(best_bid, max_candidate)
    )

    if (
        use_rollouts
        and rounds_left_after > 0.0
        and should_run_rollout(info, rounds_left_after, candidate_utilities)
    ):
        rollout_cands = select_rollout_candidates(
            info, candidate_utilities, max_candidates=5
        )
        if len(rollout_cands) >= 2:
            deadline = compute_rollout_deadline(
                start_time,
                timeout_seconds,
                reserve_seconds=0.45,
                max_budget_seconds=0.18,
            )
            rollout_res = evaluate_candidates_rollout(
                info,
                rollout_cands,
                model=model,
                hand_beliefs=hand_beliefs,
                candidate_utilities=candidate_utilities,
                config=config,
                rng=draw,
                max_worlds=_ROLLOUT_MAX_WORLDS,
                min_worlds=_ROLLOUT_MIN_WORLDS,
                deadline=deadline,
            )
            if rollout_res.best_bid is not None:
                return clamp_bid(rollout_res.best_bid, max_candidate)

    return analytic_bid


def pending_observation_rows(
    info: BotInfo,
    public_evs: Mapping[GemType, float],
) -> tuple[tuple[int, AuctionType, bool, int, float, int], ...]:
    """One row per other player: (id, type, late, coins, reference, legal_cap)."""
    card = info.current_auction
    if card is None:
        return ()
    late = phase_is_late(info)
    loan_add = card.value if card.type is AuctionType.LOAN else 0
    return tuple(
        (
            opponent.id,
            card.type,
            late,
            opponent.coins,
            lot_reference_value(info, opponent, public_evs),
            max(0, opponent.coins + loan_add),
        )
        for opponent in info.other_players
    )


def _net_gain(
    info: BotInfo,
    card: AuctionCard,
    bid: int,
    gem_evs: Mapping[GemType, float],
    future_supplies_after: Mapping[GemType, float],
    rounds_left_after: float,
    config: StrategyConfig,
) -> float:
    """Value of winning at ``bid`` compared with losing for free."""
    if card.type is not AuctionType.LOAN or bid < card.value:
        return auction_win_net_gain(
            info,
            card,
            bid,
            gem_evs,
            future_supplies_after,
            rounds_left_after,
            config,
        )

    coins = float(info.self_player.coins)
    opponent_coins = [player.coins for player in info.other_players]
    average_ev = sum(gem_evs.values()) / float(len(gem_evs)) if gem_evs else 10.0
    cash_now = future_coin_value(
        coins, rounds_left_after, opponent_coins, average_ev, config
    )
    cash_after = future_coin_value(
        coins + float(card.value - bid),
        rounds_left_after,
        opponent_coins,
        average_ev,
        config,
    )
    penalty = tie_order_penalty(
        info.self_player.id,
        info.previous_winner_id,
        info.num_players,
        rounds_left_after,
        config.tie_penalty_base,
    )
    return (cash_after - float(card.value)) - cash_now - penalty
