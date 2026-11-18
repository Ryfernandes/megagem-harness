"""Per-opponent bid model and auction win probabilities for MicahBot."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from bot_definitions.micah_strategy import (
    DEFAULT_STRATEGY_CONFIG,
    StrategyConfig,
    beats_on_tie,
    estimate_rounds_left_after,
    immediate_mission_bonus,
    tie_priority_rank,
)
from bots import BotInfo, PlayerInfo
from constants import AuctionType, GemType

AUCTION_CODE = 0  # AuctionType.AUCTION
INVEST_CODE = 1
LOAN_CODE = 2

_AUCTION_CODES: dict[AuctionType, int] = {
    AuctionType.AUCTION: AUCTION_CODE,
    AuctionType.INVEST: INVEST_CODE,
    AuctionType.LOAN: LOAN_CODE,
}

# Six live observations of weight 1 match this prior so 1-2 noisy/saving
# rounds do not erase the analytic prior, while steady habits shift it.
_PRIOR_STRENGTH = 6.0
_EXACT_CELL_WEIGHT = 1.0
_LOOSE_CELL_WEIGHT = 0.35
_ZERO_BID_WEIGHT = 0.15
_MIN_RECORD_REFERENCE = 2.0
_MAX_RATIO_MILLI = 5000


def auction_code(auction_type: AuctionType) -> int:
    """Return the compact code for an auction type (0 treasure, 1 invest, 2 loan)."""
    return _AUCTION_CODES[auction_type]


def phase_is_late(info: BotInfo) -> bool:
    """Return True when estimate_rounds_left_after(info) <= 3.0."""
    return estimate_rounds_left_after(info) <= 3.0


def coin_bucket(coins: int) -> int:
    """Return 0 if coins <= 8, 1 if coins <= 20, else 2."""
    if coins <= 8:
        return 0
    if coins <= 20:
        return 1
    return 2


def lot_reference_value(
    info: BotInfo,
    opponent: PlayerInfo,
    public_evs: Mapping[GemType, float],
) -> float:
    """Denominator for a bid-to-value ratio.

    Treasure: public EV of the current lot (the first ``card.value`` auction
    gems) plus 0.55 times that opponent's immediate mission bonus. Invest or
    loan: ``float(card.value)``. If there is no current auction, return 0.0.
    Never negative.
    """
    card = info.current_auction
    if card is None:
        return 0.0
    if card.type is AuctionType.AUCTION:
        gems = info.auction_gems[: card.value]
        lot_ev = sum(float(public_evs.get(gem, 0.0)) for gem in gems)
        bonus = immediate_mission_bonus(
            opponent.won_gems, gems, info.available_missions
        )
        value = lot_ev + 0.55 * float(bonus)
    else:
        value = float(card.value)
    if value < 0.0:
        return 0.0
    return value


def _logistic_cdf(x: float, center: float, spread: float) -> float:
    z = max(-30.0, min(30.0, (x - center) / spread))
    return 1.0 / (1.0 + math.exp(-z))


def _baseline_center(
    auction_type: AuctionType,
    coins: int,
    reference: float,
    legal_cap: int,
    config: StrategyConfig,
) -> tuple[float, float]:
    """Return (typical bid, spread input) using the Milestone 2 opponent ratios."""
    if auction_type is AuctionType.AUCTION:
        center = config.opp_treasure_bid_ratio * reference
        scale = max(4.0, reference)
    elif auction_type is AuctionType.INVEST:
        center = config.opp_invest_bid_ratio * reference
        scale = max(4.0, reference)
    else:
        scarcity = max(0.0, 1.0 - float(coins) / 20.0)
        loan_frac = config.opp_loan_bid_ratio * (0.65 + 0.75 * scarcity)
        center = loan_frac * reference
        scale = max(4.0, reference * 0.5)
    center = min(float(legal_cap), max(0.0, center))
    return center, scale


def _smooth_logistic_pmf(
    center: float, spread: float, legal_cap: int
) -> tuple[tuple[int, float], ...]:
    """Smooth discrete PMF over 0..legal_cap without blind-spot gaps."""
    if legal_cap <= 0:
        return ((0, 1.0),)

    entries: list[tuple[int, float]] = []
    prev_cdf = 0.0
    for bid in range(legal_cap):
        cdf = _logistic_cdf(float(bid) + 0.5, center, spread)
        prob = max(0.0, cdf - prev_cdf)
        if prob > 1e-12:
            entries.append((bid, prob))
        prev_cdf = cdf

    cap_prob = max(0.0, 1.0 - prev_cdf)
    if cap_prob > 1e-12:
        entries.append((legal_cap, cap_prob))

    if not entries:
        return ((legal_cap, 1.0),)

    total = sum(prob for _, prob in entries)
    if abs(total - 1.0) > 1e-9:
        entries = [(bid, prob / total) for bid, prob in entries]
    return tuple(entries)


class OpponentBidModel:
    """Compact per-game observations. Pickle-small. No history across clear()."""

    def __init__(self) -> None:
        # (opponent_id, auction_code, phase_late 0/1, coin_bucket, ratio_milli)
        self.observations: list[tuple[int, int, int, int, int]] = []
        # Parallel censoring metadata: (coins_at_bid, reference_milli, is_all_in 0/1)
        self._obs_meta: list[tuple[int, int, int]] = []

    def clear(self) -> None:
        """Drop every stored observation."""
        self.observations.clear()
        self._obs_meta.clear()

    def record(
        self,
        opponent_id: int,
        auction_type: AuctionType,
        phase_late: bool,
        coins: int,
        reference: float,
        bid: int,
        legal_cap: int | None = None,
    ) -> None:
        """Store one bid/value ratio, or skip an unusable sample.

        Skips when ``reference < 2.0`` or ``bid < 0``. Otherwise stores
        ``ratio_milli = round(1000 * bid / reference)``, clamped to 0..5000.
        """
        if reference < _MIN_RECORD_REFERENCE or bid < 0:
            return
        ratio_milli = round(1000.0 * bid / reference)
        if ratio_milli < 0:
            ratio_milli = 0
        elif ratio_milli > _MAX_RATIO_MILLI:
            ratio_milli = _MAX_RATIO_MILLI
        self.observations.append(
            (
                opponent_id,
                auction_code(auction_type),
                1 if phase_late else 0,
                coin_bucket(coins),
                ratio_milli,
            )
        )
        if legal_cap is not None:
            is_all_in = 1 if bid >= legal_cap else 0
        else:
            is_all_in = (
                1 if (auction_type is not AuctionType.LOAN and bid >= coins) else 0
            )
        ref_milli = round(1000.0 * reference)
        self._obs_meta.append((int(coins), ref_milli, is_all_in))

    def bid_distribution(
        self,
        opponent_id: int,
        auction_type: AuctionType,
        phase_late: bool,
        coins: int,
        reference: float,
        legal_cap: int,
        config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
    ) -> tuple[tuple[int, float], ...]:
        """Smooth discrete distribution of this opponent's bid in 0..legal_cap.

        Coin-capped all-in bids are treated as lower bounds (never pulling the
        expected ratio downward), and zero bids by cash-poor opponents are
        ignored so passing due to lack of cash is not mistaken for low bids.
        """
        if legal_cap <= 0:
            return ((0, 1.0),)

        center, scale = _baseline_center(
            auction_type, coins, reference, legal_cap, config
        )
        strength = max(0.5, float(config.bid_prior_strength))
        if reference >= _MIN_RECORD_REFERENCE:
            base_ratio = center / reference
            ratio_mass = strength * base_ratio
            code = auction_code(auction_type)
            late_flag = 1 if phase_late else 0
            bucket = coin_bucket(coins)
            has_meta = len(self._obs_meta) == len(self.observations)
            for idx, (
                obs_id,
                obs_code,
                obs_late,
                obs_bucket,
                ratio_milli,
            ) in enumerate(self.observations):
                if obs_id != opponent_id or obs_code != code:
                    continue
                ratio = ratio_milli / 1000.0
                obs_coins = coins
                obs_ref = reference
                is_all_in = 0
                if has_meta:
                    obs_coins, ref_milli, is_all_in = self._obs_meta[idx]
                    obs_ref = max(_MIN_RECORD_REFERENCE, ref_milli / 1000.0)

                if ratio_milli == 0:
                    if float(obs_coins) < 0.75 * obs_ref:
                        continue
                    weight = _ZERO_BID_WEIGHT
                elif is_all_in == 1 and ratio < base_ratio:
                    # Coin-capped all-in gives a lower bound, not a lower ratio.
                    continue
                elif obs_late == late_flag and obs_bucket == bucket:
                    weight = _EXACT_CELL_WEIGHT
                else:
                    weight = _LOOSE_CELL_WEIGHT
                strength += weight
                ratio_mass += weight * min(3.0, ratio)
            center = min(
                float(legal_cap), max(0.0, (ratio_mass / strength) * reference)
            )

        spread = max(1.25, config.win_prob_spread_ratio * scale)
        return _smooth_logistic_pmf(center, spread, legal_cap)

    def estimate_bid_ratio(
        self,
        opponent_id: int,
        auction_type: AuctionType,
        phase_late: bool,
        coins: int,
        reference: float,
        config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
    ) -> float:
        """Return the opponent's learned bid-to-reference ratio for treasure."""
        if auction_type is not AuctionType.AUCTION or reference < _MIN_RECORD_REFERENCE:
            return config.opp_treasure_bid_ratio
        base_ratio = config.opp_treasure_bid_ratio
        strength = max(0.5, float(config.bid_prior_strength))
        ratio_mass = strength * base_ratio
        code = auction_code(auction_type)
        late_flag = 1 if phase_late else 0
        bucket = coin_bucket(coins)
        has_meta = len(self._obs_meta) == len(self.observations)
        for idx, (
            obs_id,
            obs_code,
            obs_late,
            obs_bucket,
            ratio_milli,
        ) in enumerate(self.observations):
            if obs_id != opponent_id or obs_code != code:
                continue
            ratio = ratio_milli / 1000.0
            obs_coins = coins
            obs_ref = reference
            is_all_in = 0
            if has_meta:
                obs_coins, ref_milli, is_all_in = self._obs_meta[idx]
                obs_ref = max(_MIN_RECORD_REFERENCE, ref_milli / 1000.0)
            if ratio_milli == 0:
                if float(obs_coins) < 0.75 * obs_ref:
                    continue
                weight = _ZERO_BID_WEIGHT
            elif is_all_in == 1 and ratio < base_ratio:
                continue
            elif obs_late == late_flag and obs_bucket == bucket:
                weight = _EXACT_CELL_WEIGHT
            else:
                weight = _LOOSE_CELL_WEIGHT
            strength += weight
            ratio_mass += weight * min(3.0, ratio)
        return max(0.25, min(1.8, ratio_mass / strength))


def probability_we_win(
    our_bid: int,
    our_id: int,
    opponent_distributions: Mapping[int, Sequence[tuple[int, float]]],
    previous_winner_id: int | None,
    num_players: int,
) -> float:
    """Return the product, over opponents, of P(our bid beats that opponent)."""
    chance = 1.0
    for opp_id, dist in opponent_distributions.items():
        if opp_id == our_id:
            continue
        tie_result = beats_on_tie(our_id, opp_id, previous_winner_id, num_players)
        beat = 0.0
        for their_bid, mass in dist:
            if our_bid > their_bid:
                beat += mass
            elif our_bid == their_bid:
                if tie_result is True:
                    beat += mass
                elif tie_result is None:
                    beat += 0.5 * mass
        chance *= beat
    return chance


def winner_probabilities(
    our_id: int,
    our_bid: int,
    opponent_distributions: Mapping[int, Sequence[tuple[int, float]]],
    previous_winner_id: int | None,
    num_players: int,
) -> dict[int, float]:
    """Return exact P(each player, including us, wins the auction) in O(N^2 * B)."""
    all_dists: dict[int, tuple[tuple[int, float], ...]] = {
        our_id: ((int(our_bid), 1.0),)
    }
    for opp_id, dist in opponent_distributions.items():
        if opp_id == our_id:
            continue
        if not dist:
            all_dists[opp_id] = ((0, 1.0),)
        else:
            all_dists[opp_id] = tuple(
                (max(0, int(bid)), float(prob)) for bid, prob in dist if prob > 0.0
            ) or ((0, 1.0),)

    pids = list(all_dists.keys())
    max_b = max(max(bid for bid, _ in dist) for dist in all_dists.values())

    pmf: dict[int, list[float]] = {}
    cdf: dict[int, list[float]] = {}
    for pid, dist in all_dists.items():
        arr = [0.0] * (max_b + 1)
        for bid, prob in dist:
            arr[bid] += prob
        pmf[pid] = arr
        cum = [0.0] * (max_b + 1)
        acc = 0.0
        for b in range(max_b + 1):
            acc += arr[b]
            cum[b] = acc
        cdf[pid] = cum

    totals = {pid: 0.0 for pid in pids}
    if previous_winner_id is not None:
        ordered = sorted(
            pids,
            key=lambda pid: (
                tie_priority_rank(pid, previous_winner_id, num_players) or 0
            ),
        )
        n = len(ordered)
        for b in range(max_b + 1):
            suffix = [1.0] * (n + 1)
            for idx in range(n - 1, -1, -1):
                suffix[idx] = suffix[idx + 1] * cdf[ordered[idx]][b]
            prefix_lt = 1.0
            for idx, pid in enumerate(ordered):
                p_eq = pmf[pid][b]
                if p_eq > 0.0:
                    totals[pid] += p_eq * prefix_lt * suffix[idx + 1]
                prefix_lt *= cdf[pid][b - 1] if b > 0 else 0.0
    else:
        n = len(pids)
        for b in range(max_b + 1):
            for i, pid in enumerate(pids):
                p_eq = pmf[pid][b]
                if p_eq <= 0.0:
                    continue
                dp = [0.0] * n
                dp[0] = 1.0
                cur_max = 0
                for j, qid in enumerate(pids):
                    if j == i:
                        continue
                    q_eq = pmf[qid][b]
                    q_lt = cdf[qid][b - 1] if b > 0 else 0.0
                    cur_max += 1
                    for k in range(cur_max, 0, -1):
                        dp[k] = dp[k] * q_lt + dp[k - 1] * q_eq
                    dp[0] *= q_lt
                totals[pid] += p_eq * sum(dp[k] / float(k + 1) for k in range(n))
    return totals


def covering_bid(
    our_id: int,
    opponent_caps: Mapping[int, int],
    previous_winner_id: int | None,
    num_players: int,
    our_cap: int,
) -> int | None:
    """Return the smallest bid that beats every opponent for sure."""
    required_total: int | None = None
    for opp_id, cap in opponent_caps.items():
        if beats_on_tie(our_id, opp_id, previous_winner_id, num_players) is True:
            required = cap
        else:
            required = cap + 1
        if required_total is None or required > required_total:
            required_total = required
    if required_total is None:
        required_total = 0
    if required_total > our_cap:
        return None
    return required_total
