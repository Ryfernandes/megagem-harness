"""Guess what each opponent will bid, then update that guess from bids we see.

The game does not tell us the other players' class names. We start from the
prior ratios in Blue5Config, then mix in bids from this seat and, more weakly,
bids from the rest of the table. Pooled bids are kept across games in a match.
Per-seat bids are cleared every game, because seat numbers are only meaningful
inside one game. That matches YaleBot, which remembers a shared history but
forgets per-seat history when a new game starts.
"""

from __future__ import annotations

import math

from bot_definitions.blue5gd_config import Blue5Config
from constants import AuctionType

# One huge bid must not drag the average forever.
RATIO_CAP = 2.0

RatioStats = tuple[float, float]
PersonalKey = tuple[int, AuctionType]


class OpponentMemory:
    """Bid/prize averages. Safe to pickle as part of the bot."""

    def __init__(self) -> None:
        self.personal: dict[PersonalKey, RatioStats] = {}
        self.pooled: dict[AuctionType, RatioStats] = {}

    def begin_game(self) -> None:
        """Forget per-seat bids. Keep the shared history from earlier games."""
        self.personal.clear()

    def record(
        self,
        opp_id: int,
        kind: AuctionType,
        bid: int,
        scale: float,
        coins: int,
    ) -> None:
        """Store one observed bid. A broke opponent bidding 0 teaches us nothing."""
        if coins <= 0 and bid <= 0:
            return
        safe_scale = scale if scale > 1.0 else 1.0
        ratio = min(RATIO_CAP, max(0.0, bid / safe_scale))
        self.personal[(opp_id, kind)] = _add(
            self.personal.get((opp_id, kind), (0.0, 0.0)), ratio
        )
        self.pooled[kind] = _add(self.pooled.get(kind, (0.0, 0.0)), ratio)

    def mean_ratio(
        self,
        opp_id: int,
        kind: AuctionType,
        prior: float,
        config: Blue5Config,
    ) -> float:
        """Prior, this seat's bids, and other seats' bids, as one ratio."""
        personal_sum, personal_n = self.personal.get((opp_id, kind), (0.0, 0.0))
        pooled_sum, pooled_n = self.pooled.get(kind, (0.0, 0.0))
        # This seat's bids are inside the pool. Take them back out so they
        # are not counted twice.
        others_sum = pooled_sum - personal_sum
        others_n = pooled_n - personal_n
        if others_n < 0.0:
            others_sum, others_n = 0.0, 0.0
        weight = config.pool_weight
        numerator = (
            prior * config.learn_strength
            + personal_sum
            + weight * others_sum
        )
        denominator = (
            config.learn_strength + personal_n + weight * others_n
        )
        if denominator <= 0.0:
            return prior
        return numerator / denominator

    def bid_weights(
        self,
        opp_id: int,
        kind: AuctionType,
        prior: float,
        scale: float,
        cap: int,
        config: Blue5Config,
    ) -> list[float]:
        """Relative chance this opponent bids 0, 1, ..., cap."""
        if cap <= 0:
            return [1.0]
        mean_bid = self.mean_ratio(opp_id, kind, prior, config) * max(scale, 1.0)
        spread = max(0.75, config.spread_ratio * max(scale, 1.0))
        return [
            math.exp(-abs(bid - mean_bid) / spread) for bid in range(cap + 1)
        ]


def _add(stats: RatioStats, ratio: float) -> RatioStats:
    total, count = stats
    return (total + ratio, count + 1.0)
