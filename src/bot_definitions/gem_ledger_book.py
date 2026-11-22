"""Empirical opponent bid book and best-response bid selector for GemLedgerBot."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from bot_definitions.gem_ledger_pricing import AuctionTerms
from bots import BotInfo, PlayerInfo
from constants import AuctionType
from lib import AuctionCard


@dataclass(frozen=True)
class BookConfig:
    """Weighting and memory settings for opponent bid forecasting."""

    opening_ratios: tuple[float, ...] = (0.3, 0.45, 0.6)
    opening_weight: float = 0.25
    opening_drop_after: int = 2
    max_records: int = 12
    table_weight: float = 1.0
    seat_weight: float = 3.0


DEFAULT_BOOK_CONFIG = BookConfig()


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


def _env_ratios(default: tuple[float, ...]) -> tuple[float, ...]:
    """Read comma-separated opening bid ratios, or keep the default."""
    raw = os.environ.get("GL_CFG_OPENING_RATIOS")
    if raw is None or raw.strip() == "":
        return default
    return tuple(float(part) for part in raw.split(","))


def book_from_env() -> BookConfig:
    """Build the bid-book settings from GL_CFG_* variables.

    Unset variables keep today's defaults. See pricing_from_env for why the
    search passes numbers through the environment.
    """
    defaults = DEFAULT_BOOK_CONFIG
    return BookConfig(
        opening_ratios=_env_ratios(defaults.opening_ratios),
        opening_weight=_env_float("GL_CFG_OPENING_WEIGHT", defaults.opening_weight),
        opening_drop_after=_env_int("GL_CFG_OPENING_DROP", defaults.opening_drop_after),
        max_records=_env_int("GL_CFG_MAX_RECORDS", defaults.max_records),
        table_weight=_env_float("GL_CFG_TABLE_WEIGHT", defaults.table_weight),
        seat_weight=_env_float("GL_CFG_SEAT_WEIGHT", defaults.seat_weight),
    )


@dataclass(frozen=True)
class BidObservation:
    """One observed non-bankrupt opponent bid in normalized and raw coin terms."""

    ratio: float
    coins: int


@dataclass(frozen=True)
class OpenRoundSnapshot:
    """Pre-resolution auction scale and opponent coin stacks awaiting bids."""

    reference_scale: float
    opponent_coins: dict[int, int]


@dataclass(frozen=True)
class BidHistogram:
    """Weighted bid distribution over integer bids 0..cap with prefix sums."""

    player_id: int
    weights: tuple[float, ...]
    prefix_below: tuple[float, ...]
    total_weight: float

    def win_probability(self, bid: int, tie_win_prob: float) -> float:
        """Probability that ``bid`` beats this opponent given tie priority."""
        if bid >= len(self.weights):
            return 1.0
        favorable = self.prefix_below[bid] + tie_win_prob * self.weights[bid]
        return favorable / self.total_weight


class OpponentBidBook:
    """Track per-seat and cross-game table bid histories by auction type."""

    def __init__(self, config: BookConfig = DEFAULT_BOOK_CONFIG) -> None:
        self.config = config
        self._seat_history: dict[tuple[int, AuctionType], list[BidObservation]] = {}
        self._table_history: dict[AuctionType, list[BidObservation]] = {}
        self._open_round: OpenRoundSnapshot | None = None

    def start_new_game(self) -> None:
        """Reset seat-specific records for a new table while keeping table priors."""
        self._seat_history = {}
        self._open_round = None

    def mark_round_open(
        self, reference_scale: float, opponents: Sequence[PlayerInfo]
    ) -> None:
        """Remember this round's prize scale and each opponent's starting coins."""
        self._open_round = OpenRoundSnapshot(
            reference_scale=reference_scale,
            opponent_coins={opp.id: opp.coins for opp in opponents},
        )

    def commit_round(
        self, auction_type: AuctionType, bids: Mapping[int, int]
    ) -> None:
        """Record bids from opponents who had coins to spend this round."""
        if self._open_round is None:
            return
        snapshot = self._open_round
        self._open_round = None

        cap = self.config.max_records
        table_list = self._table_history.setdefault(auction_type, [])
        for player_id, bid in bids.items():
            if snapshot.opponent_coins.get(player_id, 0) <= 0:
                continue
            entry = BidObservation(ratio=bid / snapshot.reference_scale, coins=bid)
            table_list.append(entry)
            if len(table_list) > cap:
                del table_list[:-cap]

            seat_list = self._seat_history.setdefault((player_id, auction_type), [])
            seat_list.append(entry)
            if len(seat_list) > cap:
                del seat_list[:-cap]

    def opponent_histogram(
        self, player: PlayerInfo, card: AuctionCard, reference_scale: float
    ) -> BidHistogram:
        """Build a recency-weighted bid histogram capped at the player's budget."""
        cfg = self.config
        budget = player.coins + (card.value if card.type is AuctionType.LOAN else 0)
        weights = [0.0] * (budget + 1)

        seat_records = self._seat_history.get((player.id, card.type), ())
        if len(seat_records) < cfg.opening_drop_after:
            for ratio in cfg.opening_ratios:
                level = min(budget, round(ratio * reference_scale))
                weights[level] += cfg.opening_weight

        table_records = self._table_history.get(card.type, ())
        self._accumulate_records(
            weights, budget, reference_scale, table_records, cfg.table_weight
        )
        self._accumulate_records(
            weights, budget, reference_scale, seat_records, cfg.seat_weight
        )

        prefix: list[float] = [0.0] * (budget + 2)
        running = 0.0
        for level, weight in enumerate(weights):
            prefix[level] = running
            running += weight
        prefix[budget + 1] = running

        return BidHistogram(
            player_id=player.id,
            weights=tuple(weights),
            prefix_below=tuple(prefix),
            total_weight=running,
        )

    @staticmethod
    def _accumulate_records(
        weights: list[float],
        budget: int,
        reference_scale: float,
        records: Sequence[BidObservation],
        base_weight: float,
    ) -> None:
        count = len(records)
        if count == 0:
            return
        for idx, entry in enumerate(records):
            recency = 0.5 + 0.5 * ((idx + 1) / count)
            weight = base_weight * recency
            scaled_bid = min(budget, round(entry.ratio * reference_scale))
            raw_bid = min(budget, entry.coins)
            weights[scaled_bid] += weight
            weights[raw_bid] += weight


def tie_win_probability(
    self_id: int,
    opponent_id: int,
    num_players: int,
    previous_winner_id: int | None,
) -> float:
    """Return probability of beating ``opponent_id`` on a tied bid."""
    if previous_winner_id is None:
        return 0.5
    self_turn = (self_id - previous_winner_id - 1) % num_players
    opp_turn = (opponent_id - previous_winner_id - 1) % num_players
    return 1.0 if self_turn < opp_turn else 0.0


def choose_optimal_bid(
    info: BotInfo,
    terms: AuctionTerms,
    histograms: Sequence[BidHistogram],
) -> int:
    """Select the affordable integer bid maximizing expected net point gain."""
    opponents = [
        (
            hist,
            tie_win_probability(
                info.self_player.id,
                hist.player_id,
                info.num_players,
                info.previous_winner_id,
            ),
        )
        for hist in histograms
    ]

    best_bid = 0
    best_payoff = float("-inf")
    for bid in range(terms.max_legal_bid + 1):
        margin = terms.gross_value - terms.marginal_coin_cost * bid
        if margin < 0.0:
            break
        win_prob = 1.0
        for hist, tie_prob in opponents:
            win_prob *= hist.win_probability(bid, tie_prob)
        payoff = margin * win_prob
        if payoff > best_payoff:
            best_bid = bid
            best_payoff = payoff
    return best_bid
