"""Lightweight CRN Monte Carlo rollout engine for MicahBot (Milestone 5)."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random

from bot_definitions.micah_bid_model import OpponentBidModel, covering_bid
from bot_definitions.micah_endgame import sample_opponent_hidden_hands
from bot_definitions.micah_inference import OpponentHandBeliefs
from bot_definitions.micah_strategy import (
    DEFAULT_STRATEGY_CONFIG,
    StrategyConfig,
    beats_on_tie,
    expected_chart_value,
    future_coin_value,
    legal_bid_cap,
    public_value_shift_from_counts,
    tie_order_penalty,
)
from bots import BotInfo, PlayerInfo
from constants import (
    GEMS_ON_AUCTION,
    STANDARD_GEMS,
    AuctionType,
    GemType,
    WildCardRequirement,
)
from lib import AuctionCard, MissionCard

_GEM_TO_IDX: dict[GemType, int] = {gem: idx for idx, gem in enumerate(STANDARD_GEMS)}


@dataclass(frozen=True, slots=True)
class RolloutWorld:
    """Immutable sampled hidden state + CRN rolls for a single rollout world."""

    opp_hands: dict[int, tuple[GemType, ...]]
    gem_deck: tuple[GemType, ...]
    auction_deck: tuple[AuctionCard, ...]
    round_bid_rolls: tuple[dict[int, float], ...]
    round_tie_seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Aggregated rollout evaluation across completed CRN worlds."""

    stats: dict[int, tuple[float, float, float]]
    worlds_completed: int
    best_bid: int | None


@dataclass(frozen=True, slots=True)
class _CompiledMission:
    value: int
    req: tuple[int, int, int, int, int]
    wc_req: int
    wc_type: WildCardRequirement | None
    total_req: int


@dataclass(slots=True)
class _SimPlayer:
    pid: int
    coins: int
    invested_coins: int
    end_game_bonus: int
    won_gems: list[int]
    hidden_gems: list[int]


def _compile_mission(mission: MissionCard) -> _CompiledMission:
    req = (
        int(mission.gem_requirements.get(STANDARD_GEMS[0], 0)),
        int(mission.gem_requirements.get(STANDARD_GEMS[1], 0)),
        int(mission.gem_requirements.get(STANDARD_GEMS[2], 0)),
        int(mission.gem_requirements.get(STANDARD_GEMS[3], 0)),
        int(mission.gem_requirements.get(STANDARD_GEMS[4], 0)),
    )
    wc_req = int(mission.gem_requirements.get(GemType.WILD_CARD, 0))
    total_req = sum(req) + wc_req
    return _CompiledMission(
        value=int(mission.value),
        req=req,
        wc_req=wc_req,
        wc_type=mission.wild_card_requirement,
        total_req=total_req,
    )


def _mission_satisfied(mission: _CompiledMission, won: Sequence[int]) -> bool:
    r0 = won[0] - mission.req[0]
    r1 = won[1] - mission.req[1]
    r2 = won[2] - mission.req[2]
    r3 = won[3] - mission.req[3]
    r4 = won[4] - mission.req[4]
    if r0 < 0 or r1 < 0 or r2 < 0 or r3 < 0 or r4 < 0:
        return False
    wc = mission.wc_req
    if wc <= 0:
        return True
    wt = mission.wc_type
    if wt is None:
        return (r0 + r1 + r2 + r3 + r4) >= wc
    if wt is WildCardRequirement.SAME:
        return max(r0, r1, r2, r3, r4) >= wc
    if wt is WildCardRequirement.DIFFERENT:
        return ((r0 > 0) + (r1 > 0) + (r2 > 0) + (r3 > 0) + (r4 > 0)) >= wc
    return ((r0 // 2) + (r1 // 2) + (r2 // 2) + (r3 // 2) + (r4 // 2)) * 2 >= wc


def _mission_progress_missing(
    mission: _CompiledMission, won: Sequence[int]
) -> tuple[float, int, tuple[int, int, int, int, int]]:
    req = mission.req
    fixed_met = 0
    m0 = req[0] - won[0] if won[0] < req[0] else 0
    m1 = req[1] - won[1] if won[1] < req[1] else 0
    m2 = req[2] - won[2] if won[2] < req[2] else 0
    m3 = req[3] - won[3] if won[3] < req[3] else 0
    m4 = req[4] - won[4] if won[4] < req[4] else 0

    fixed_met += won[0] if won[0] < req[0] else req[0]
    fixed_met += won[1] if won[1] < req[1] else req[1]
    fixed_met += won[2] if won[2] < req[2] else req[2]
    fixed_met += won[3] if won[3] < req[3] else req[3]
    fixed_met += won[4] if won[4] < req[4] else req[4]

    r0 = won[0] - req[0] if won[0] > req[0] else 0
    r1 = won[1] - req[1] if won[1] > req[1] else 0
    r2 = won[2] - req[2] if won[2] > req[2] else 0
    r3 = won[3] - req[3] if won[3] > req[3] else 0
    r4 = won[4] - req[4] if won[4] > req[4] else 0

    wc_req = mission.wc_req
    wc_met = 0.0
    wc_missing = 0
    if wc_req > 0:
        wt = mission.wc_type
        if wt is None:
            got = min(r0 + r1 + r2 + r3 + r4, wc_req)
            wc_met = float(got)
            wc_missing = wc_req - got
        elif wt is WildCardRequirement.SAME:
            got = min(max(r0, r1, r2, r3, r4), wc_req)
            wc_met = float(got)
            wc_missing = wc_req - got
        elif wt is WildCardRequirement.DIFFERENT:
            got = min((r0 > 0) + (r1 > 0) + (r2 > 0) + (r3 > 0) + (r4 > 0), wc_req)
            wc_met = float(got)
            wc_missing = wc_req - got
        else:
            target_pairs = wc_req // 2
            pairs = (r0 // 2) + (r1 // 2) + (r2 // 2) + (r3 // 2) + (r4 // 2)
            singles = (r0 % 2) + (r1 % 2) + (r2 % 2) + (r3 % 2) + (r4 % 2)
            used_pairs = min(pairs, target_pairs)
            used_singles = min(singles, max(0, target_pairs - used_pairs))
            wc_met = float(used_pairs * 2) + 0.5 * float(used_singles)
            wc_missing = max(0, wc_req - used_pairs * 2 - used_singles)

    missing_total = m0 + m1 + m2 + m3 + m4 + wc_missing
    if mission.total_req <= 0:
        frac = 1.0 if _mission_satisfied(mission, won) else 0.0
        return frac, 0, (0, 0, 0, 0, 0)
    frac = min(1.0, (float(fixed_met) + wc_met) / float(mission.total_req))
    return frac, missing_total, (m0, m1, m2, m3, m4)


def compute_rollout_deadline(
    start_time: float,
    timeout_seconds: float | None,
    *,
    reserve_seconds: float = 0.40,
    max_budget_seconds: float = 0.20,
) -> float:
    """Compute monotonic perf_counter deadline for bounded rollout search."""
    if timeout_seconds is None:
        return start_time + max_budget_seconds
    now = time.perf_counter()
    usable = max(0.0, float(timeout_seconds) - reserve_seconds - (now - start_time))
    return now + min(max_budget_seconds, usable)


def should_run_rollout(
    info: BotInfo,
    rounds_left_after: float,
    candidate_utilities: Mapping[int, float],
) -> bool:
    """Return True only when rollout simulation adds value on high-leverage rounds."""
    card = info.current_auction
    if card is None or rounds_left_after <= 0.0 or len(candidate_utilities) < 2:
        return False
    if rounds_left_after <= 3.5:
        return True
    if card.type is AuctionType.LOAN and rounds_left_after <= 6.0:
        return True
    if rounds_left_after <= 5.0:
        sorted_u = sorted(candidate_utilities.values(), reverse=True)
        if sorted_u[0] - sorted_u[1] <= 1.25:
            return True
        all_players: tuple[PlayerInfo, ...] = (info.self_player, *info.other_players)
        for mission in info.available_missions:
            compiled = _compile_mission(mission)
            for player in all_players:
                won = [int(player.won_gems.get(g, 0)) for g in STANDARD_GEMS]
                if not _mission_satisfied(compiled, won):
                    _, missing, _ = _mission_progress_missing(compiled, won)
                    if missing <= 2:
                        return True
    return False


def select_rollout_candidates(
    info: BotInfo,
    candidate_utilities: Mapping[int, float],
    max_candidates: int = 5,
) -> list[int]:
    """Select top analytic bids plus 0 and tight covering bid, deduplicated."""
    if max_candidates <= 0 or not candidate_utilities:
        return []

    sorted_bids = sorted(
        candidate_utilities.keys(),
        key=lambda bid: (-float(candidate_utilities[bid]), int(bid)),
    )
    max_legal = legal_bid_cap(info)
    card = info.current_auction
    if card is not None and card.type is AuctionType.LOAN:
        cov_cap = min(max_legal, max(0, card.value - 1))
    else:
        cov_cap = max_legal

    loan_add = card.value if (card is not None and card.type is AuctionType.LOAN) else 0
    opp_caps = {opp.id: max(0, opp.coins + loan_add) for opp in info.other_players}
    cov = covering_bid(
        info.self_player.id,
        opp_caps,
        info.previous_winner_id,
        info.num_players,
        cov_cap,
    )

    extras: list[int] = []
    if 0 <= max_legal and 0 in candidate_utilities:
        extras.append(0)
    if cov is not None and cov not in extras:
        extras.append(cov)

    selected: list[int] = [int(sorted_bids[0])]
    remaining_extras = [b for b in extras if b != selected[0]]

    for bid in sorted_bids[1:]:
        b_int = int(bid)
        if b_int in remaining_extras:
            selected.append(b_int)
            remaining_extras.remove(b_int)
            if len(selected) >= max_candidates:
                break
        elif len(selected) + len(remaining_extras) < max_candidates:
            selected.append(b_int)

    for b_int in remaining_extras:
        if len(selected) >= max_candidates:
            break
        if b_int not in selected:
            selected.append(b_int)

    return selected[:max_candidates]


def sample_rollout_world(
    info: BotInfo,
    rng: Random,
    hand_beliefs: OpponentHandBeliefs | None = None,
) -> RolloutWorld:
    """Sample one consistent hidden world and its per-round CRN rolls."""
    if hand_beliefs is not None:
        opp_hands = hand_beliefs.sample_opponent_hands_with_beliefs(info, rng)
    else:
        opp_hands = sample_opponent_hidden_hands(info, rng)

    rem = {gem: int(info.unknown_gems.get(gem, 0)) for gem in STANDARD_GEMS}
    for hand in opp_hands.values():
        for gem in hand:
            if rem.get(gem, 0) > 0:
                rem[gem] -= 1

    gem_deck: list[GemType] = [
        gem for gem in STANDARD_GEMS for _ in range(max(0, rem[gem]))
    ]
    rng.shuffle(gem_deck)

    auction_deck: list[AuctionCard] = []
    for card, count in info.auction_cards_remaining.items():
        if count > 0:
            auction_deck.extend([card] * int(count))
    rng.shuffle(auction_deck)

    total_rounds = 1 + len(auction_deck)
    opp_ids = tuple(sorted(opp.id for opp in info.other_players))
    round_bid_rolls = tuple(
        {opp_id: rng.random() for opp_id in opp_ids} for _ in range(total_rounds)
    )
    round_tie_seeds = tuple(rng.getrandbits(64) for _ in range(total_rounds))

    return RolloutWorld(
        opp_hands=opp_hands,
        gem_deck=tuple(gem_deck),
        auction_deck=tuple(auction_deck),
        round_bid_rolls=round_bid_rolls,
        round_tie_seeds=round_tie_seeds,
    )


def choose_rollout_winner(
    stats: Mapping[int, tuple[float, float, float]],
    worlds_completed: int,
    *,
    min_worlds: int = 12,
    tie_tol: float = 0.02,
    candidate_utilities: Mapping[int, float] | None = None,
) -> int | None:
    """Select winning candidate bid from rollout stats, or None if too few worlds."""
    if worlds_completed < min_worlds or not stats:
        return None
    max_fp = max(fp for fp, _, _ in stats.values())
    contenders = [bid for bid, (fp, _, _) in stats.items() if max_fp - fp <= tie_tol]
    return max(
        contenders,
        key=lambda bid: (
            stats[bid][2],
            stats[bid][1],
            float(candidate_utilities.get(bid, 0.0))
            if candidate_utilities is not None
            else 0.0,
            stats[bid][0],
            -bid,
        ),
    )


def _draw_from_dist(dist: Sequence[tuple[int, float]], roll: float) -> int:
    if not dist:
        return 0
    cumulative = 0.0
    last_bid = dist[-1][0]
    for bid, prob in dist:
        cumulative += prob
        if roll < cumulative:
            return bid
        last_bid = bid
    return last_bid


def _select_winner_idx(
    bids: Sequence[int],
    player_ids: Sequence[int],
    previous_winner_id: int | None,
    tie_seed: int,
) -> int:
    highest = max(bids)
    tied = [i for i, b in enumerate(bids) if b == highest]
    if len(tied) == 1:
        return tied[0]
    if previous_winner_id is None:
        tied.sort(key=lambda i: player_ids[i])
        return tied[tie_seed % len(tied)]
    n = len(bids)
    return min(
        tied,
        key=lambda i: (player_ids[i] - previous_winner_id - 1) % n,
    )


def _sim_immediate_mission_bonus(
    won: Sequence[int],
    lot: Sequence[int],
    missions: Sequence[_CompiledMission],
) -> int:
    if not missions or not lot:
        return 0
    after = list(won)
    for gi in lot:
        after[gi] += 1
    total = 0
    for mission in missions:
        if not _mission_satisfied(mission, won) and _mission_satisfied(mission, after):
            total += mission.value
    return total


def _sim_mission_is_feasible(
    mission: _CompiledMission,
    after: Sequence[int],
    missing_fixed: Sequence[int],
    future_supplies: Sequence[float],
) -> bool:
    for gi in range(5):
        if missing_fixed[gi] > 0 and future_supplies[gi] <= 1e-9:
            return False

    wc_req = mission.wc_req
    if wc_req <= 0:
        return True

    rem = [max(0, after[gi] - mission.req[gi]) for gi in range(5)]
    sup = [max(0.0, future_supplies[gi] - float(missing_fixed[gi])) for gi in range(5)]
    wt = mission.wc_type
    if wt is None:
        return sum(rem) + sum(sup) + 1e-9 >= float(wc_req)
    if wt is WildCardRequirement.SAME:
        return max((rem[gi] + sup[gi] for gi in range(5)), default=0.0) + 1e-9 >= float(
            wc_req
        )
    if wt is WildCardRequirement.DIFFERENT:
        return sum(1 for gi in range(5) if rem[gi] > 0 or sup[gi] > 1e-9) >= wc_req
    target_pairs = wc_req // 2
    have_pairs = sum(v // 2 for v in rem)
    rem_pairs_needed = max(0, target_pairs - have_pairs)
    poss_pairs = sum(
        int(math.floor((rem[gi] % 2) + sup[gi] + 1e-9)) // 2 for gi in range(5)
    )
    return poss_pairs >= rem_pairs_needed


def _sim_mission_values(
    won: Sequence[int],
    lot: Sequence[int],
    missions: Sequence[_CompiledMission],
    future_supplies_after: Sequence[float],
    rounds_left_after: float,
    config: StrategyConfig,
) -> tuple[int, float]:
    if not missions or not lot:
        return 0, 0.0
    after = list(won)
    for gi in lot:
        after[gi] += 1
    imm = 0
    prog = 0.0
    do_prog = rounds_left_after > 0.0 and config.mission_progress_discount > 0.0
    for mission in missions:
        if _mission_satisfied(mission, won):
            continue
        if _mission_satisfied(mission, after):
            imm += mission.value
            continue
        if not do_prog:
            continue
        frac_before, _, _ = _mission_progress_missing(mission, won)
        frac_after, missing_after, missing_fixed = _mission_progress_missing(
            mission, after
        )
        delta = frac_after - frac_before
        if delta <= 0.0:
            continue
        if not _sim_mission_is_feasible(
            mission, after, missing_fixed, future_supplies_after
        ):
            continue
        feasibility = min(
            1.0, (rounds_left_after * 1.25) / float(max(1, missing_after))
        )
        prog += (
            config.mission_progress_discount
            * feasibility
            * float(mission.value)
            * delta
        )
    return imm, prog


def _sim_opp_distributions(
    players: Sequence[_SimPlayer],
    self_idx: int,
    card: AuctionCard,
    lot: Sequence[int],
    missions: Sequence[_CompiledMission],
    public_evs: Sequence[float],
    rounds_left_after: float,
    model: OpponentBidModel,
    config: StrategyConfig,
) -> dict[int, tuple[tuple[int, float], ...]]:
    late = rounds_left_after <= 3.0
    is_treasure = card.type is AuctionType.AUCTION
    lot_ev = sum(public_evs[gi] for gi in lot) if is_treasure else 0.0
    loan_add = card.value if card.type is AuctionType.LOAN else 0
    card_val = float(card.value)
    dists: dict[int, tuple[tuple[int, float], ...]] = {}
    for idx, player in enumerate(players):
        if idx == self_idx:
            continue
        opp_cap = max(0, player.coins + loan_add)
        if opp_cap <= 0:
            dists[player.pid] = ((0, 1.0),)
            continue
        if is_treasure:
            bonus = _sim_immediate_mission_bonus(player.won_gems, lot, missions)
            ref = max(0.0, lot_ev + 0.55 * float(bonus))
        else:
            ref = card_val
        dists[player.pid] = model.bid_distribution(
            player.pid,
            card.type,
            late,
            player.coins,
            ref,
            opp_cap,
            config,
        )
    return dists


def _sim_rounds_left_after(
    card: AuctionCard,
    auction_gems: Sequence[int],
    gem_deck_len: int,
    auction_deck: Sequence[AuctionCard],
) -> float:
    gems_rem = gem_deck_len + len(auction_gems)
    consumed_now = min(card.value, gems_rem) if card.type is AuctionType.AUCTION else 0
    gems_after = max(0, gems_rem - consumed_now)
    if gems_after == 0:
        return 0.0
    total_cards = len(auction_deck)
    if total_cards <= 0:
        return max(1.0, float(gems_after) / 1.25)
    future_treasure_gems = sum(
        c.value for c in auction_deck if c.type is AuctionType.AUCTION
    )
    if future_treasure_gems <= 0:
        return max(1.0, float(gems_after) / 1.25)
    avg_gems_per_card = float(future_treasure_gems) / float(total_cards)
    estimated = float(gems_after) / avg_gems_per_card
    return max(1.0, min(float(total_cards), estimated))


def _sim_supplies(
    auction_gems: Sequence[int],
    gem_deck_len: int,
    unknown_counts: Sequence[int],
) -> list[float]:
    total_unknown = sum(unknown_counts)
    supplies = [0.0] * 5
    for gi in auction_gems:
        supplies[gi] += 1.0
    if total_unknown > 0 and gem_deck_len > 0:
        scale = float(gem_deck_len) / float(total_unknown)
        for gi in range(5):
            supplies[gi] += float(unknown_counts[gi]) * scale
    return supplies


def _sim_color_mission_interest(
    won: Sequence[int],
    gi: int,
    missions: Sequence[_CompiledMission],
    supplies: Sequence[float],
) -> float:
    if not missions:
        return 0.0
    after = list(won)
    after[gi] += 1
    interest = 0.0
    for mission in missions:
        if _mission_satisfied(mission, won):
            continue
        if _mission_satisfied(mission, after):
            interest += float(mission.value) / 10.0
            continue
        frac_before, _, _ = _mission_progress_missing(mission, won)
        frac_after, _, missing_fixed = _mission_progress_missing(mission, after)
        delta = frac_after - frac_before
        if delta > 0.0 and _sim_mission_is_feasible(
            mission, after, missing_fixed, supplies
        ):
            interest += 0.35 * (float(mission.value) / 10.0) * delta
    return interest


def _pop_reveal_gem(
    winner: _SimPlayer,
    winner_idx: int,
    self_idx: int,
    players: Sequence[_SimPlayer],
    auction_gems: Sequence[int],
    gem_deck_len: int,
    value_display: list[int],
    self_unknown: list[int],
    self_hidden_counts: list[int],
    missions: Sequence[_CompiledMission],
    chart_values: tuple[int, ...],
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> int | None:
    hidden = winner.hidden_gems
    if not hidden:
        return None
    if len(hidden) == 1:
        gi = hidden.pop()
        value_display[gi] += 1
        if winner_idx == self_idx:
            self_hidden_counts[gi] -= 1
        elif self_unknown[gi] > 0:
            self_unknown[gi] -= 1
        return gi

    public_pool = [self_unknown[i] + self_hidden_counts[i] for i in range(5)]
    public_total = sum(public_pool)
    total_hidden = sum(len(p.hidden_gems) for p in players)

    if winner_idx == self_idx:
        supplies = _sim_supplies(auction_gems, gem_deck_len, self_unknown)
        next_gi = auction_gems[0] if auction_gems else -1
        opp_max_coins = max(
            (p.coins for i, p in enumerate(players) if i != self_idx), default=0
        )
        self_power = 1.0 if winner.coins >= opp_max_coins else 0.7
        best_pos = 0
        best_score = -1e18
        cached: dict[int, float] = {}
        for pos, gi in enumerate(hidden):
            if gi not in cached:
                sup = supplies[gi]
                if sup <= 1e-9:
                    cached[gi] = 0.0
                else:
                    shift = public_value_shift_from_counts(
                        chart_values,
                        value_display[gi],
                        public_pool[gi],
                        public_total,
                        total_hidden,
                    )
                    self_int = self_power + _sim_color_mission_interest(
                        winner.won_gems, gi, missions, supplies
                    )
                    opp_int = 0.55 + max(
                        (
                            (0.4 + 0.6 * min(1.0, float(p.coins) / 10.0))
                            * _sim_color_mission_interest(
                                p.won_gems, gi, missions, supplies
                            )
                            for i, p in enumerate(players)
                            if i != self_idx
                        ),
                        default=0.0,
                    )
                    imm = 1.25 if gi == next_gi else 1.0
                    directional = (
                        config.reveal_directional_weight
                        * (opp_int - self_int)
                        * shift
                        * min(1.5, sup / 1.5)
                        * imm
                    )
                    cached[gi] = directional - config.reveal_leak_penalty * sup
            if cached[gi] > best_score + 1e-12:
                best_score = cached[gi]
                best_pos = pos
        gi = hidden.pop(best_pos)
        value_display[gi] += 1
        self_hidden_counts[gi] -= 1
        return gi

    # Opponent reveal: lowest public-leak heuristic matching micah_inference.
    supplies_pub = _sim_supplies(auction_gems, gem_deck_len, public_pool)
    best_pos = 0
    best_cost = 1e18
    cached_cost: dict[int, float] = {}
    for pos, gi in enumerate(hidden):
        if gi not in cached_cost:
            sup = supplies_pub[gi]
            if sup <= 0.0:
                cached_cost[gi] = 0.0
            else:
                shift = public_value_shift_from_counts(
                    chart_values,
                    value_display[gi],
                    public_pool[gi],
                    public_total,
                    total_hidden,
                )
                cached_cost[gi] = shift * (sup - 0.5 * float(winner.won_gems[gi]))
        if cached_cost[gi] < best_cost - 1e-12:
            best_cost = cached_cost[gi]
            best_pos = pos
    gi = hidden.pop(best_pos)
    value_display[gi] += 1
    if self_unknown[gi] > 0:
        self_unknown[gi] -= 1
    return gi


def _choose_sim_self_bid(
    players: Sequence[_SimPlayer],
    self_idx: int,
    card: AuctionCard,
    auction_gems: Sequence[int],
    missions: Sequence[_CompiledMission],
    previous_winner_id: int | None,
    gem_evs: Sequence[float],
    opp_dists: Mapping[int, tuple[tuple[int, float], ...]],
    rounds_left_after: float,
    future_supplies_after: Sequence[float],
    config: StrategyConfig,
) -> int:
    self_p = players[self_idx]
    coins = self_p.coins
    max_legal = max(0, coins + (card.value if card.type is AuctionType.LOAN else 0))
    max_cand = (
        min(max_legal, card.value - 1) if card.type is AuctionType.LOAN else max_legal
    )
    if max_cand <= 0:
        return 0

    opp_coins = [p.coins for i, p in enumerate(players) if i != self_idx]
    avg_ev = sum(gem_evs) / 5.0
    base_cash = future_coin_value(
        float(coins), rounds_left_after, opp_coins, avg_ev, config
    )
    tie_pen = tie_order_penalty(
        self_p.pid,
        previous_winner_id,
        len(players),
        rounds_left_after,
        config.tie_penalty_base,
    )

    if card.type is AuctionType.AUCTION:
        lot = auction_gems[: min(card.value, len(auction_gems))]
        lot_val = sum(gem_evs[gi] for gi in lot)
        imm_miss, prog_miss = _sim_mission_values(
            self_p.won_gems,
            lot,
            missions,
            future_supplies_after,
            rounds_left_after,
            config,
        )
        const_term = lot_val + float(imm_miss) + prog_miss - base_cash - tie_pen
    elif card.type is AuctionType.INVEST:
        const_term = float(card.value) - base_cash - tie_pen
    else:
        const_term = -float(card.value) - base_cash - tie_pen

    # Precompute O(1)-lookup win probability arrays per opponent.
    n_players = len(players)
    beat_arrays: list[list[float]] = []
    for idx, opp in enumerate(players):
        if idx == self_idx:
            continue
        dist = opp_dists.get(opp.pid, ((0, 1.0),))
        tie_res = beats_on_tie(self_p.pid, opp.pid, previous_winner_id, n_players)
        tie_w = 1.0 if tie_res is True else (0.5 if tie_res is None else 0.0)
        pmf = [0.0] * (max_cand + 1)
        for their_bid, prob in dist:
            if their_bid <= max_cand:
                pmf[their_bid] += prob
        beat_arr = [0.0] * (max_cand + 1)
        acc = 0.0
        for b in range(max_cand + 1):
            p_eq = pmf[b]
            beat_arr[b] = acc + tie_w * p_eq
            acc += p_eq
        beat_arrays.append(beat_arr)

    def eval_bid(bid: int) -> tuple[float, float]:
        if card.type is AuctionType.AUCTION:
            after_cash = future_coin_value(
                float(coins - bid), rounds_left_after, opp_coins, avg_ev, config
            )
            net = const_term + after_cash
        elif card.type is AuctionType.INVEST:
            after_cash = future_coin_value(
                float(coins - bid), rounds_left_after, opp_coins, avg_ev, config
            )
            net = const_term + after_cash + float(bid)
        else:
            after_cash = future_coin_value(
                float(coins + card.value - bid),
                rounds_left_after,
                opp_coins,
                avg_ev,
                config,
            )
            net = const_term + after_cash
        if net <= 0.0 and bid > 0:
            return net, -1e18
        p_win = 1.0
        for arr in beat_arrays:
            p_win *= arr[bid]
        return net, p_win * net

    best_bid = 0
    best_u = -1e18

    if max_cand <= 25:
        for b in range(max_cand + 1):
            net, u = eval_bid(b)
            if net <= 0.0 and b > 0:
                break
            if u > best_u + 1e-12:
                best_u = u
                best_bid = b
    else:
        step = max(2, max_cand // 20)
        coarse_bids = list(range(0, max_cand + 1, step))
        if coarse_bids[-1] != max_cand:
            coarse_bids.append(max_cand)
        best_coarse = 0
        best_coarse_u = -1e18
        for b in coarse_bids:
            net, u = eval_bid(b)
            if u > best_coarse_u + 1e-12:
                best_coarse_u = u
                best_coarse = b
            if net <= 0.0 and b > 0:
                break

        lo = max(0, best_coarse - step)
        hi = min(max_cand, best_coarse + step)
        check_bids = [0, 1] if lo > 1 else []
        check_bids.extend(range(lo, hi + 1))
        for b in check_bids:
            net, u = eval_bid(b)
            if u > best_u + 1e-12:
                best_u = u
                best_bid = b

    if best_u <= 0.0:
        return 0
    return best_bid


def _simulate_candidate_on_world(
    info: BotInfo,
    world: RolloutWorld,
    cand_bid: int,
    round0_opp_bids: Mapping[int, int],
    compiled_missions: Sequence[_CompiledMission],
    model: OpponentBidModel,
    config: StrategyConfig,
) -> tuple[float, float, float]:
    """Run one complete game simulation from round 0 for a single candidate bid."""
    card0 = info.current_auction
    assert card0 is not None

    chart_values = info.value_chart.values
    all_info_players = (info.self_player, *info.other_players)
    players: list[_SimPlayer] = []
    for idx, info_p in enumerate(all_info_players):
        if idx == 0:
            hidden = [
                _GEM_TO_IDX[g] for g in info.self_player.hidden_gems if g in _GEM_TO_IDX
            ]
        else:
            hidden = [
                _GEM_TO_IDX[g]
                for g in world.opp_hands.get(info_p.id, ())
                if g in _GEM_TO_IDX
            ]
        bonus = sum(m.value for m in info_p.mission_cards) + sum(
            c.end_game_value for c in info_p.auction_cards
        )
        players.append(
            _SimPlayer(
                pid=info_p.id,
                coins=int(info_p.coins),
                invested_coins=int(info_p.invested_coins),
                end_game_bonus=bonus,
                won_gems=[int(info_p.won_gems.get(g, 0)) for g in STANDARD_GEMS],
                hidden_gems=hidden,
            )
        )

    self_idx = 0
    player_ids = [p.pid for p in players]
    gem_deck = [_GEM_TO_IDX[g] for g in world.gem_deck if g in _GEM_TO_IDX]
    auction_deck = list(world.auction_deck)
    auction_gems: deque[int] = deque(
        _GEM_TO_IDX[g] for g in info.auction_gems if g in _GEM_TO_IDX
    )
    value_display = [int(info.value_display.get(g, 0)) for g in STANDARD_GEMS]
    self_unknown = [int(info.unknown_gems.get(g, 0)) for g in STANDARD_GEMS]
    self_hidden_counts = [0] * 5
    for gi in players[self_idx].hidden_gems:
        self_hidden_counts[gi] += 1
    available_missions = list(compiled_missions)
    previous_winner_id = info.previous_winner_id

    def apply_round(round_idx: int, card: AuctionCard, bids: Sequence[int]) -> None:
        nonlocal previous_winner_id
        tie_seed = (
            world.round_tie_seeds[round_idx]
            if round_idx < len(world.round_tie_seeds)
            else 0
        )
        winner_idx = _select_winner_idx(bids, player_ids, previous_winner_id, tie_seed)
        winner = players[winner_idx]
        win_bid = bids[winner_idx]

        if card.type is AuctionType.AUCTION:
            take = min(card.value, len(auction_gems))
            for _ in range(take):
                winner.won_gems[auction_gems.popleft()] += 1
        elif card.type is AuctionType.LOAN:
            winner.coins += card.value
            winner.end_game_bonus -= card.value
        else:
            winner.invested_coins += win_bid
            winner.end_game_bonus += card.value
        winner.coins -= win_bid
        previous_winner_id = winner.pid

        _pop_reveal_gem(
            winner,
            winner_idx,
            self_idx,
            players,
            list(auction_gems),
            len(gem_deck),
            value_display,
            self_unknown,
            self_hidden_counts,
            available_missions,
            chart_values,
            config,
        )

        while len(auction_gems) < GEMS_ON_AUCTION and gem_deck:
            drawn = gem_deck.pop()
            auction_gems.append(drawn)
            if self_unknown[drawn] > 0:
                self_unknown[drawn] -= 1

        if available_missions:
            remaining_missions: list[_CompiledMission] = []
            for m in available_missions:
                if _mission_satisfied(m, winner.won_gems):
                    winner.end_game_bonus += m.value
                else:
                    remaining_missions.append(m)
            available_missions[:] = remaining_missions

    # Round 0: candidate bid vs world's CRN opponent bids.
    round0_bids = [
        cand_bid if idx == self_idx else round0_opp_bids.get(p.pid, 0)
        for idx, p in enumerate(players)
    ]
    apply_round(0, card0, round0_bids)

    # Future rounds until auction_gems empty (or auction_deck exhausted).
    round_idx = 1
    while auction_gems and auction_deck:
        card = auction_deck.pop()
        auction_list = list(auction_gems)
        rounds_left_after = _sim_rounds_left_after(
            card, auction_list, len(gem_deck), auction_deck
        )

        self_unknown_total = sum(self_unknown)
        self_hidden_total = sum(self_hidden_counts)
        opp_hidden_total = sum(
            len(p.hidden_gems) for i, p in enumerate(players) if i != self_idx
        )
        public_total = self_unknown_total + self_hidden_total
        total_hidden = opp_hidden_total + self_hidden_total

        self_evs = [
            expected_chart_value(
                chart_values,
                min(6, value_display[gi] + self_hidden_counts[gi]),
                self_unknown[gi],
                self_unknown_total,
                opp_hidden_total,
            )
            for gi in range(5)
        ]
        public_evs = [
            expected_chart_value(
                chart_values,
                min(6, value_display[gi]),
                self_unknown[gi] + self_hidden_counts[gi],
                public_total,
                total_hidden,
            )
            for gi in range(5)
        ]

        lot = (
            auction_list[: min(card.value, len(auction_list))]
            if card.type is AuctionType.AUCTION
            else []
        )
        opp_dists = _sim_opp_distributions(
            players,
            self_idx,
            card,
            lot,
            available_missions,
            public_evs,
            rounds_left_after,
            model,
            config,
        )

        supplies_after = _sim_supplies(auction_list, len(gem_deck), self_unknown)
        if card.type is AuctionType.AUCTION:
            for gi in lot:
                if supplies_after[gi] > 1.0:
                    supplies_after[gi] -= 1.0
                else:
                    supplies_after[gi] = 0.0

        self_bid = _choose_sim_self_bid(
            players,
            self_idx,
            card,
            auction_list,
            available_missions,
            previous_winner_id,
            self_evs,
            opp_dists,
            rounds_left_after,
            supplies_after,
            config,
        )

        rolls = (
            world.round_bid_rolls[round_idx]
            if round_idx < len(world.round_bid_rolls)
            else {}
        )
        bids = [0] * len(players)
        for idx, p in enumerate(players):
            if idx == self_idx:
                bids[idx] = self_bid
            else:
                bids[idx] = _draw_from_dist(opp_dists[p.pid], rolls.get(p.pid, 0.5))

        apply_round(round_idx, card, bids)
        round_idx += 1

    # End of game: reveal all remaining hidden hands into value_display.
    for p in players:
        for gi in p.hidden_gems:
            value_display[gi] += 1

    gem_values = [chart_values[min(6, value_display[gi])] for gi in range(5)]
    scores = [
        p.coins
        + p.invested_coins
        + p.end_game_bonus
        + sum(p.won_gems[gi] * gem_values[gi] for gi in range(5))
        for p in players
    ]

    self_score = scores[self_idx]
    max_other = max(s for idx, s in enumerate(scores) if idx != self_idx)
    max_score = max(scores)
    if self_score == max_score:
        tied_count = sum(1 for s in scores if s == max_score)
        first_place = 1.0
        frac_win = 1.0 / float(tied_count)
    else:
        first_place = 0.0
        frac_win = 0.0
    margin = float(self_score - max_other)
    return first_place, frac_win, margin


def evaluate_candidates_rollout(
    info: BotInfo,
    candidates: Sequence[int],
    model: OpponentBidModel | None = None,
    hand_beliefs: OpponentHandBeliefs | None = None,
    candidate_utilities: Mapping[int, float] | None = None,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
    rng: Random | None = None,
    max_worlds: int = 32,
    min_worlds: int = 12,
    deadline: float | None = None,
) -> RolloutResult:
    """Evaluate candidate bids over CRN rollout worlds within optional deadline."""
    ordered_candidates: list[int] = []
    seen: set[int] = set()
    for bid in candidates:
        b = int(bid)
        if b not in seen:
            seen.add(b)
            ordered_candidates.append(b)

    card0 = info.current_auction
    if card0 is None or not ordered_candidates or max_worlds <= 0:
        return RolloutResult(stats={}, worlds_completed=0, best_bid=None)

    if deadline is not None and time.perf_counter() >= deadline:
        return RolloutResult(stats={}, worlds_completed=0, best_bid=None)

    active_model = model if model is not None else OpponentBidModel()
    draw_rng = rng if rng is not None else Random()
    compiled_missions = tuple(_compile_mission(m) for m in info.available_missions)

    # Precompute round 0 opponent distributions once across all worlds/candidates.
    chart_values = info.value_chart.values
    value_display0 = [int(info.value_display.get(g, 0)) for g in STANDARD_GEMS]
    self_hidden0 = [info.self_player.hidden_gems.count(g) for g in STANDARD_GEMS]
    self_unknown0 = [int(info.unknown_gems.get(g, 0)) for g in STANDARD_GEMS]
    public_pool0 = [self_unknown0[i] + self_hidden0[i] for i in range(5)]
    public_total0 = sum(public_pool0)
    total_hidden0 = sum(opp.hidden_gem_count for opp in info.other_players) + sum(
        self_hidden0
    )
    public_evs0 = [
        expected_chart_value(
            chart_values,
            min(6, value_display0[gi]),
            public_pool0[gi],
            public_total0,
            total_hidden0,
        )
        for gi in range(5)
    ]

    auction_list0 = [_GEM_TO_IDX[g] for g in info.auction_gems if g in _GEM_TO_IDX]
    lot0 = (
        auction_list0[: min(card0.value, len(auction_list0))]
        if card0.type is AuctionType.AUCTION
        else []
    )
    auction_cards0 = [
        c for c, cnt in info.auction_cards_remaining.items() for _ in range(max(0, cnt))
    ]
    rounds_left0 = _sim_rounds_left_after(
        card0, auction_list0, info.gem_deck_count, auction_cards0
    )

    sim_players0 = [
        _SimPlayer(
            pid=p.id,
            coins=int(p.coins),
            invested_coins=int(p.invested_coins),
            end_game_bonus=0,
            won_gems=[int(p.won_gems.get(g, 0)) for g in STANDARD_GEMS],
            hidden_gems=[],
        )
        for p in (info.self_player, *info.other_players)
    ]
    round0_opp_dists = _sim_opp_distributions(
        sim_players0,
        0,
        card0,
        lot0,
        compiled_missions,
        public_evs0,
        rounds_left0,
        active_model,
        config,
    )

    fp_acc = dict.fromkeys(ordered_candidates, 0.0)
    fw_acc = dict.fromkeys(ordered_candidates, 0.0)
    margin_acc = dict.fromkeys(ordered_candidates, 0.0)
    completed = 0

    for world_idx in range(max_worlds):
        if deadline is not None and world_idx > 0 and time.perf_counter() >= deadline:
            break
        if deadline is not None and world_idx == 0 and time.perf_counter() >= deadline:
            break

        world = sample_rollout_world(info, draw_rng, hand_beliefs=hand_beliefs)
        rolls0 = world.round_bid_rolls[0] if world.round_bid_rolls else {}
        round0_opp_bids = {
            opp_id: _draw_from_dist(dist, rolls0.get(opp_id, 0.5))
            for opp_id, dist in round0_opp_dists.items()
        }

        world_outcomes: list[tuple[int, float, float, float]] = []
        for cand_bid in ordered_candidates:
            fp, fw, margin = _simulate_candidate_on_world(
                info,
                world,
                cand_bid,
                round0_opp_bids,
                compiled_missions,
                active_model,
                config,
            )
            world_outcomes.append((cand_bid, fp, fw, margin))

        for cand_bid, fp, fw, margin in world_outcomes:
            fp_acc[cand_bid] += fp
            fw_acc[cand_bid] += fw
            margin_acc[cand_bid] += margin
        completed += 1

    if completed == 0:
        return RolloutResult(stats={}, worlds_completed=0, best_bid=None)

    scale = float(completed)
    stats = {
        bid: (fp_acc[bid] / scale, fw_acc[bid] / scale, margin_acc[bid] / scale)
        for bid in ordered_candidates
    }
    best_bid = choose_rollout_winner(
        stats,
        completed,
        min_worlds=min_worlds,
        candidate_utilities=candidate_utilities,
    )
    return RolloutResult(
        stats=stats,
        worlds_completed=completed,
        best_bid=best_bid,
    )
