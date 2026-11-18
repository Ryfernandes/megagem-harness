"""Bayesian opponent hidden-hand inference for MicahBot (Milestone 4)."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

from bot_definitions.micah_strategy import (
    DEFAULT_STRATEGY_CONFIG,
    StrategyConfig,
    estimate_rounds_left_after,
    expected_chart_value,
    expected_future_supplies_after_current,
    expected_future_supply,
    immediate_mission_bonus,
    mission_progress_value,
    public_value_shift,
)
from bots import BotInfo
from constants import STANDARD_GEMS, AuctionType, GemType
from lib import MissionCard

HandTuple = tuple[int, int, int, int, int]

_GEM_INDEX: dict[GemType, int] = {gem: idx for idx, gem in enumerate(STANDARD_GEMS)}

_MIN_LIKELIHOOD = 0.15
_MAX_LIKELIHOOD = 6.0


@lru_cache(maxsize=16)
def hand_tuples(hand_size: int) -> tuple[HandTuple, ...]:
    """Enumerate all 5-tuples of non-negative ints summing to ``hand_size``."""
    if hand_size <= 0:
        return ((0, 0, 0, 0, 0),)
    result: list[HandTuple] = []
    for c0 in range(hand_size + 1):
        rem1 = hand_size - c0
        for c1 in range(rem1 + 1):
            rem2 = rem1 - c1
            for c2 in range(rem2 + 1):
                rem3 = rem2 - c2
                for c3 in range(rem3 + 1):
                    c4 = rem3 - c3
                    result.append((c0, c1, c2, c3, c4))
    return tuple(result)


def tuple_prior_weight(
    hand: HandTuple,
    unknown_counts: tuple[int, ...],
) -> float:
    """Hypergeometric combinatorial weight of ``hand`` drawn from ``unknown_counts``."""
    if len(unknown_counts) != 5:
        raise ValueError("unknown_counts must have 5 entries matching STANDARD_GEMS")
    weight = 1
    for u, c in zip(unknown_counts, hand):
        if c < 0 or u < 0 or c > u:
            return 0.0
        weight *= math.comb(u, c)
    return float(weight)


@dataclass(frozen=True, slots=True)
class InferenceSnapshot:
    """Exact pre-round state snapshot used to update beliefs after resolution."""

    chart_values: tuple[int, ...]
    value_display: tuple[int, ...]
    public_unknown: tuple[int, ...]
    unknown_counts: tuple[int, ...]
    total_hidden_hands: int
    lot: tuple[GemType, ...]
    auction_type: AuctionType
    opp_snapshots: dict[int, tuple[int, int, int, float, float]]
    future_supplies: tuple[float, ...]
    public_shifts: tuple[float, ...]


def build_inference_snapshot(
    info: BotInfo,
    opp_bid_ratios: Mapping[int, float] | None = None,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> InferenceSnapshot | None:
    """Capture pre-round state from ``info`` for post-round Bayesian hand updates."""
    card = info.current_auction
    if card is None:
        return None

    self_hidden = info.self_player.hidden_gems
    value_display = tuple(int(info.value_display.get(g, 0)) for g in STANDARD_GEMS)
    unknown_counts = tuple(int(info.unknown_gems.get(g, 0)) for g in STANDARD_GEMS)
    public_unknown = tuple(
        unknown_counts[i] + self_hidden.count(g) for i, g in enumerate(STANDARD_GEMS)
    )
    total_hidden_hands = sum(opp.hidden_gem_count for opp in info.other_players) + len(
        self_hidden
    )

    lot: tuple[GemType, ...]
    if card.type is AuctionType.AUCTION:
        lot = info.auction_gems[: min(card.value, len(info.auction_gems))]
    else:
        lot = ()

    future_supplies_after = expected_future_supplies_after_current(info)
    rounds_left_after = estimate_rounds_left_after(info)
    loan_add = card.value if card.type is AuctionType.LOAN else 0

    opp_snapshots: dict[int, tuple[int, int, int, float, float]] = {}
    for opp in info.other_players:
        legal_cap = max(0, opp.coins + loan_add)
        if card.type is AuctionType.AUCTION and lot:
            imm = float(
                immediate_mission_bonus(opp.won_gems, lot, info.available_missions)
            )
            prog = mission_progress_value(
                opp.won_gems,
                lot,
                info.available_missions,
                future_supplies_after,
                rounds_left_after,
                config,
            )
            opp_mission_val = imm + 0.5 * prog
        else:
            opp_mission_val = 0.0
        opp_bid_ratio = (
            float(opp_bid_ratios[opp.id])
            if opp_bid_ratios is not None and opp.id in opp_bid_ratios
            else config.opp_treasure_bid_ratio
        )
        opp_snapshots[opp.id] = (
            int(opp.coins),
            int(legal_cap),
            int(opp.hidden_gem_count),
            float(opp_mission_val),
            float(opp_bid_ratio),
        )

    future_supplies = tuple(expected_future_supply(info, g) for g in STANDARD_GEMS)
    public_shifts = tuple(public_value_shift(info, g) for g in STANDARD_GEMS)

    return InferenceSnapshot(
        chart_values=info.value_chart.values,
        value_display=value_display,
        public_unknown=public_unknown,
        unknown_counts=unknown_counts,
        total_hidden_hands=total_hidden_hands,
        lot=lot,
        auction_type=card.type,
        opp_snapshots=opp_snapshots,
        future_supplies=future_supplies,
        public_shifts=public_shifts,
    )


def _normalize_bounded(raw: dict[HandTuple, float]) -> dict[HandTuple, float]:
    """Scale and softly clamp feasible likelihoods to [0.15, 6.0] with mean == 1.0."""
    if not raw:
        return {}
    n = len(raw)
    vals = [max(1e-12, float(v)) for v in raw.values()]
    max_v = max(vals)
    min_v = min(vals)
    if max_v <= 0.0:
        return {k: 1.0 for k in raw}

    mean_v = sum(vals) / float(n)
    scaled = [v / mean_v for v in vals]
    if all(_MIN_LIKELIHOOD <= v <= _MAX_LIKELIHOOD for v in scaled):
        return dict(zip(raw.keys(), scaled))

    # Binary search scale factor c > 0 so mean(clamp(c * v, lo, hi)) == 1.0.
    lo_c = _MIN_LIKELIHOOD / max_v
    hi_c = _MAX_LIKELIHOOD / min_v
    for _ in range(60):
        mid_c = 0.5 * (lo_c + hi_c)
        mid_mean = sum(
            _MIN_LIKELIHOOD
            if (cv := mid_c * v) < _MIN_LIKELIHOOD
            else (_MAX_LIKELIHOOD if cv > _MAX_LIKELIHOOD else cv)
            for v in vals
        ) / float(n)
        if mid_mean < 1.0:
            lo_c = mid_c
        else:
            hi_c = mid_c

    c = 0.5 * (lo_c + hi_c)
    bounded = [
        _MIN_LIKELIHOOD
        if (cv := c * v) < _MIN_LIKELIHOOD
        else (_MAX_LIKELIHOOD if cv > _MAX_LIKELIHOOD else cv)
        for v in vals
    ]
    return dict(zip(raw.keys(), bounded))


def _is_feasible(hand: HandTuple, unknown_counts: tuple[int, ...]) -> bool:
    return all(hand[i] <= unknown_counts[i] for i in range(5))


class OpponentHandBeliefs:
    """Compact per-game Bayesian likelihood multipliers over opponent hand tuples."""

    def __init__(self) -> None:
        self.likelihoods: dict[int, dict[HandTuple, float]] = {}

    def clear(self) -> None:
        """Reset all stored opponent hand beliefs."""
        self.likelihoods.clear()

    def ensure_opponent(self, opp_id: int, hand_size: int) -> dict[HandTuple, float]:
        """Return beliefs if keys sum to ``hand_size``, else initialize uniform."""
        target = max(0, int(hand_size))
        existing = self.likelihoods.get(opp_id)
        if (
            existing is not None
            and existing
            and all(sum(k) == target for k in existing)
        ):
            return existing
        fresh = {t: 1.0 for t in hand_tuples(target)}
        self.likelihoods[opp_id] = fresh
        return fresh

    def _valid_beliefs(
        self, opp_id: int, hand_size: int
    ) -> dict[HandTuple, float] | None:
        """Return existing belief map if keys sum to ``hand_size``, else None."""
        target = max(0, int(hand_size))
        existing = self.likelihoods.get(opp_id)
        if (
            existing is not None
            and existing
            and all(sum(k) == target for k in existing)
        ):
            return existing
        return None

    def expected_hand_counts(
        self,
        opp_id: int,
        hand_size: int,
        unknown_counts: tuple[int, ...],
    ) -> tuple[float, float, float, float, float]:
        """Return posterior expected gem counts E[hand] for ``opp_id``."""
        target = max(0, int(hand_size))
        if target <= 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        beliefs = self._valid_beliefs(opp_id, target)
        totals = [0.0, 0.0, 0.0, 0.0, 0.0]
        total_w = 0.0
        for t in hand_tuples(target):
            prior = tuple_prior_weight(t, unknown_counts)
            if prior <= 0.0:
                continue
            mult = beliefs.get(t, 0.0) if beliefs is not None else 1.0
            w = prior * mult
            if w <= 0.0:
                continue
            total_w += w
            for i in range(5):
                totals[i] += w * float(t[i])
        if total_w <= 0.0:
            total_u = sum(unknown_counts)
            if total_u <= 0:
                return (0.0, 0.0, 0.0, 0.0, 0.0)
            scale = float(target) / float(total_u)
            return (
                float(unknown_counts[0]) * scale,
                float(unknown_counts[1]) * scale,
                float(unknown_counts[2]) * scale,
                float(unknown_counts[3]) * scale,
                float(unknown_counts[4]) * scale,
            )
        return (
            totals[0] / total_w,
            totals[1] / total_w,
            totals[2] / total_w,
            totals[3] / total_w,
            totals[4] / total_w,
        )

    def update_from_bid(
        self,
        opp_id: int,
        snapshot: InferenceSnapshot,
        bid: int,
        strength: float = 0.30,
        spread_ratio: float = 0.24,
    ) -> None:
        """Update opponent hand likelihoods from an observed treasure auction bid."""
        if (
            snapshot.auction_type is not AuctionType.AUCTION
            or not snapshot.lot
            or strength <= 0.0
            or opp_id not in snapshot.opp_snapshots
        ):
            return

        coins, legal_cap, hand_size, opp_mission_val, opp_bid_ratio = (
            snapshot.opp_snapshots[opp_id]
        )
        if hand_size <= 0 or legal_cap <= 0:
            return
        if bid == 0 and coins < 6:
            return

        old_L = self.ensure_opponent(opp_id, hand_size)
        lot_indices = tuple(_GEM_INDEX[gem] for gem in snapshot.lot)
        unique_indices = set(lot_indices)
        pool_total_opp = sum(snapshot.public_unknown) - hand_size
        other_hidden = snapshot.total_hidden_hands - hand_size

        ev_opp: dict[int, list[float]] = {}
        for gi in unique_indices:
            evs: list[float] = []
            for count in range(hand_size + 1):
                rem_color = snapshot.public_unknown[gi] - count
                if rem_color < 0 or pool_total_opp < 0 or other_hidden < 0:
                    evs.append(0.0)
                else:
                    evs.append(
                        expected_chart_value(
                            snapshot.chart_values,
                            min(6, snapshot.value_display[gi] + count),
                            rem_color,
                            pool_total_opp,
                            other_hidden,
                        )
                    )
            ev_opp[gi] = evs

        new_L: dict[HandTuple, float] = {}
        tempered_strength = min(1.0, strength)
        for t, old_val in old_L.items():
            if old_val <= 0.0 or not _is_feasible(t, snapshot.unknown_counts):
                continue
            v_opp = sum(ev_opp[gi][t[gi]] for gi in lot_indices) + opp_mission_val
            center = opp_bid_ratio * v_opp
            spread = max(2.0, spread_ratio * max(4.0, v_opp))
            if bid >= legal_cap:
                z_cap = max(
                    -30.0,
                    min(30.0, ((float(legal_cap) - 0.5) - center) / spread),
                )
                like = 1.0 / (1.0 + math.exp(z_cap))
                factor = like**tempered_strength
            else:
                z = (float(bid) - center) / spread
                factor = math.exp(-0.5 * strength * min(9.0, z * z))
            new_L[t] = max(1e-12, old_val * factor)

        if not new_L:
            return
        self.likelihoods[opp_id] = _normalize_bounded(new_L)

    def update_from_reveal(
        self,
        opp_id: int,
        revealed_gem: GemType,
        snapshot: InferenceSnapshot,
        new_unknown_counts: tuple[int, ...],
        opp_won_gems: Mapping[GemType, int],
        available_missions: Sequence[MissionCard],
        reveal_strength: float = 0.08,
    ) -> None:
        """Condition beliefs on ``opp_id`` revealing ``revealed_gem`` (size - 1)."""
        del available_missions
        r_idx = _GEM_INDEX.get(revealed_gem)
        if r_idx is None or opp_id not in snapshot.opp_snapshots:
            return
        _, _, hand_size, _, _ = snapshot.opp_snapshots[opp_id]
        if hand_size <= 0:
            return

        old_L = self.ensure_opponent(opp_id, hand_size)
        new_hand_size = hand_size - 1
        if new_hand_size == 0:
            self.likelihoods[opp_id] = {(0, 0, 0, 0, 0): 1.0}
            return

        opp_won = tuple(int(opp_won_gems.get(g, 0)) for g in STANDARD_GEMS)
        supplies = snapshot.future_supplies
        shifts = snapshot.public_shifts

        num_acc: dict[HandTuple, float] = {}
        den_acc: dict[HandTuple, float] = {}

        for t_after in hand_tuples(new_hand_size):
            if not _is_feasible(t_after, new_unknown_counts):
                continue
            t_pre_list = list(t_after)
            t_pre_list[r_idx] += 1
            t_pre: HandTuple = (
                t_pre_list[0],
                t_pre_list[1],
                t_pre_list[2],
                t_pre_list[3],
                t_pre_list[4],
            )
            if not _is_feasible(t_pre, snapshot.unknown_counts):
                continue
            prior_w = tuple_prior_weight(t_pre, snapshot.unknown_counts)
            if prior_w <= 0.0:
                continue
            old_val = old_L.get(t_pre, 0.0)
            if old_val <= 0.0:
                continue

            if reveal_strength <= 0.0:
                reveal_like = 1.0
            else:
                leak_costs: list[float] = []
                leak_r = 0.0
                for g in range(5):
                    if t_pre[g] < 1:
                        continue
                    cost = (
                        shifts[g] * (supplies[g] - 0.5 * float(opp_won[g]))
                        if supplies[g] > 0.0
                        else 0.0
                    )
                    leak_costs.append(cost)
                    if g == r_idx:
                        leak_r = cost
                min_leak = min(leak_costs) if leak_costs else 0.0
                if leak_r <= min_leak:
                    reveal_like = 1.0
                else:
                    reveal_like = math.exp(
                        -reveal_strength * min(3.0, max(0.0, leak_r - min_leak))
                    )

            num_acc[t_after] = num_acc.get(t_after, 0.0) + (
                prior_w * old_val * reveal_like
            )
            den_acc[t_after] = den_acc.get(t_after, 0.0) + prior_w

        if not num_acc:
            return
        raw_after = {
            t_after: max(1e-12, num_acc[t_after] / den_acc[t_after])
            for t_after in num_acc
            if den_acc[t_after] > 0.0
        }
        if not raw_after:
            return
        self.likelihoods[opp_id] = _normalize_bounded(raw_after)

    def gem_expected_values_with_beliefs(
        self,
        info: BotInfo,
        inference_strength: float = 1.0,
    ) -> dict[GemType, float]:
        """Compute per-gem EVs using Vandermonde-weighted DP over hand beliefs."""
        chart_values = info.value_chart.values
        unknown_counts = tuple(int(info.unknown_gems.get(g, 0)) for g in STANDARD_GEMS)
        u = sum(unknown_counts)
        total_h = sum(opp.hidden_gem_count for opp in info.other_players)
        deck = u - total_h
        alpha = min(1.0, max(0.0, float(inference_strength)))

        active_opps = [
            (opp, self._valid_beliefs(opp.id, opp.hidden_gem_count))
            for opp in info.other_players
            if opp.hidden_gem_count > 0
        ]
        no_beliefs = all(beliefs is None for _, beliefs in active_opps)

        result: dict[GemType, float] = {}
        for gem_idx, gem in enumerate(STANDARD_GEMS):
            known_disp = max(
                0,
                min(
                    6,
                    info.value_display.get(gem, 0)
                    + info.self_player.hidden_gems.count(gem),
                ),
            )
            w = unknown_counts[gem_idx]
            if (
                u <= 0
                or total_h <= 0
                or w <= 0
                or deck < 0
                or alpha <= 0.0
                or no_beliefs
            ):
                result[gem] = expected_chart_value(
                    chart_values, known_disp, w, u, total_h
                )
                continue

            max_d = min(w, total_h)
            dp = [1.0] + [0.0] * max_d
            cur_max = 0

            for opp, beliefs in active_opps:
                h_i = opp.hidden_gem_count
                k_cap = min(h_i, w)
                if beliefs is None:
                    r_vals = [1.0] * (k_cap + 1)
                else:
                    num = [0.0] * (k_cap + 1)
                    den = [0.0] * (k_cap + 1)
                    for t in hand_tuples(h_i):
                        k = t[gem_idx]
                        if k > k_cap:
                            continue
                        prior = tuple_prior_weight(t, unknown_counts)
                        if prior <= 0.0:
                            continue
                        l_val = beliefs.get(t, 0.0)
                        eff = (1.0 - alpha) + alpha * l_val
                        num[k] += prior * eff
                        den[k] += prior
                    r_vals = [
                        num[k] / den[k] if den[k] > 0.0 else 1.0
                        for k in range(k_cap + 1)
                    ]

                kernel = [
                    float(math.comb(h_i, k)) * r_vals[k] for k in range(k_cap + 1)
                ]
                next_max = min(max_d, cur_max + k_cap)
                next_dp = [0.0] * (max_d + 1)
                for d in range(cur_max + 1):
                    dp_d = dp[d]
                    if dp_d == 0.0:
                        continue
                    limit = min(k_cap, next_max - d)
                    for k in range(limit + 1):
                        next_dp[d + k] += dp_d * kernel[k]
                dp = next_dp
                cur_max = next_max

            weights = [0.0] * (max_d + 1)
            for d in range(max_d + 1):
                rem_w = w - d
                if 0 <= rem_w <= deck:
                    weights[d] = dp[d] * float(math.comb(deck, rem_w))

            total_weight = sum(weights)
            if total_weight <= 0.0:
                result[gem] = expected_chart_value(
                    chart_values, known_disp, w, u, total_h
                )
            else:
                result[gem] = sum(
                    (weights[d] / total_weight)
                    * float(chart_values[min(6, known_disp + d)])
                    for d in range(max_d + 1)
                )

        return result

    def sample_opponent_hands_with_beliefs(
        self,
        info: BotInfo,
        rng: random.Random,
    ) -> dict[int, tuple[GemType, ...]]:
        """Sequentially sample opponent hidden hands weighted by current beliefs."""
        rem = [int(info.unknown_gems.get(g, 0)) for g in STANDARD_GEMS]
        hands: dict[int, tuple[GemType, ...]] = {}

        for opp in sorted(info.other_players, key=lambda p: p.id):
            need = opp.hidden_gem_count
            rem_total = sum(rem)
            if need <= 0 or rem_total <= 0:
                hands[opp.id] = ()
                continue

            rem_tuple = (rem[0], rem[1], rem[2], rem[3], rem[4])
            beliefs = self._valid_beliefs(opp.id, need)

            candidates: list[HandTuple] = []
            cum_weights: list[float] = []
            running = 0.0
            if need <= rem_total:
                for t in hand_tuples(need):
                    prior = tuple_prior_weight(t, rem_tuple)
                    if prior <= 0.0:
                        continue
                    mult = beliefs.get(t, 0.0) if beliefs is not None else 1.0
                    w = prior * mult
                    if w <= 0.0:
                        continue
                    running += w
                    candidates.append(t)
                    cum_weights.append(running)

            if running > 0.0 and candidates:
                roll = rng.random() * running
                chosen = candidates[-1]
                for t, cw in zip(candidates, cum_weights):
                    if roll < cw:
                        chosen = t
                        break
                for i in range(5):
                    rem[i] -= chosen[i]
                hands[opp.id] = tuple(
                    gem
                    for gem, count in zip(STANDARD_GEMS, chosen)
                    for _ in range(count)
                )
            else:
                bag = [
                    gem for gem, count in zip(STANDARD_GEMS, rem) for _ in range(count)
                ]
                rng.shuffle(bag)
                take = min(need, len(bag))
                drawn = tuple(bag[:take])
                hands[opp.id] = drawn
                for gem in drawn:
                    rem[_GEM_INDEX[gem]] -= 1

        return hands
