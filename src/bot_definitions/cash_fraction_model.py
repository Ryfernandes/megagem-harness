"""A conservative, type-specific empirical opponent bid predictor.

Contributors: Grace Guo (85375651b), OpenAI Codex (implementation assistance).

Log validated public bids against PRE-auction available cash, separately for
each opponent and auction type. Loan budgets include the loan proceeds. Keep
zero-budget observations for audit, without treating forced zero bids as style.
Treasure samples have one-quarter weight because lot values and missions vary.
Two pseudo-observations from a broad normal prior prevent brittle predictions;
the remaining mass is an empirical distribution of historical cash fractions.
No opponent private hand is accessed or inferred by this model.
"""

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

from bots import BotInfo, PlayerInfo
from constants import AuctionType

PRIOR_WEIGHT = 2.0
PRIOR_FRACTIONS = {
    AuctionType.AUCTION: (0.20, 0.14),
    AuctionType.INVEST: (0.20, 0.16),
    AuctionType.LOAN: (0.06, 0.10),
}


def available_budget(info: BotInfo, player: PlayerInfo) -> int:
    card = info.current_auction
    return player.coins + (
        card.value if card is not None and card.type is AuctionType.LOAN else 0
    )


@dataclass(frozen=True)
class BidObservation:
    round_number: int
    auction_type: AuctionType
    auction_value: int
    coins_before: int
    budget_before: int
    bid: int
    fraction: float | None
    weight: float


class CashFractionModel:
    """Small pickleable per-game history; an opponent never shares another's data."""

    def __init__(self) -> None:
        self.history: dict[int, dict[AuctionType, list[BidObservation]]] = {}

    def observe(self, info: BotInfo, player: PlayerInfo, bid: int) -> None:
        card = info.current_auction
        if card is None:
            return
        budget = available_budget(info, player)
        if not 0 <= bid <= budget:
            raise ValueError("Only validated, affordable bids may train the model")
        fraction = bid / budget if budget else None
        record = BidObservation(
            info.round_number,
            card.type,
            card.value,
            player.coins,
            budget,
            bid,
            fraction,
            (0.25 if card.type is AuctionType.AUCTION else 1.0) if budget else 0.0,
        )
        records = self.history.setdefault(player.id, {}).setdefault(card.type, [])
        records.append(record)
        del records[:-32]

    def fraction_moments(
        self, player_id: int, kind: AuctionType
    ) -> tuple[float, float]:
        """Shrunk fraction moments before prior clipping and integer rounding."""
        mean, spread = PRIOR_FRACTIONS[kind]
        records = self.history.get(player_id, {}).get(kind, [])
        weight = PRIOR_WEIGHT + sum(r.weight for r in records)
        first = PRIOR_WEIGHT * mean + sum(
            r.weight * r.fraction for r in records if r.fraction is not None
        )
        second = PRIOR_WEIGHT * (spread**2 + mean**2) + sum(
            r.weight * r.fraction**2 for r in records if r.fraction is not None
        )
        posterior_mean = first / weight
        variance = max(0.0, second / weight - posterior_mean**2)
        return posterior_mean, sqrt(variance)

    def distribution(self, info: BotInfo, player: PlayerInfo) -> list[float]:
        """PMF over legal integer bids: prior + rescaled empirical observations.

        A sample at fraction f predicts round(f * CURRENT available budget).
        Rounded normal prior tails are clipped to 0/budget, preserving pass and
        all-in mass. The persistent broad prior represents noisy changing lots;
        this deliberately avoids a precise point forecast from a few samples.
        """
        card = info.current_auction
        budget = available_budget(info, player)
        if not budget:
            return [1.0]
        if card is None:
            return [1.0] + [0.0] * budget
        mean, spread = PRIOR_FRACTIONS[card.type]
        normal = NormalDist(mean * budget, max(0.75, spread * budget))
        cdfs = [0.0] + [normal.cdf(bid + 0.5) for bid in range(budget)] + [1.0]
        masses = [PRIOR_WEIGHT * max(0.0, hi - lo) for lo, hi in zip(cdfs, cdfs[1:])]
        for record in self.history.get(player.id, {}).get(card.type, []):
            if record.fraction is not None:
                bid = min(budget, max(0, round(record.fraction * budget)))
                masses[bid] += record.weight
        total = sum(masses)
        return [mass / total for mass in masses]
