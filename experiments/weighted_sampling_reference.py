"""Private previous weighted L4 for reproducible ablations; not a public bot.

Shares the unchanged rollout/search engine. Only this experimental control owns
an L3 tracker and applies its likelihoods to uniformly proposed hidden worlds.
The default/public SamplingBot contains neither tracker nor weighting logic.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from math import exp, isfinite
from random import Random

from bot_definitions.sampled_rollout import SampledWorld
from bot_definitions.sampling_bot import SamplingBot
from bot_definitions.soft_tracking_bot import SoftTrackingBot
from bots import BotInfo, GameSummary, RoundResult
from constants import GemType

Likelihoods = Mapping[int, Mapping[GemType, Sequence[float]]]


def weighted_means(
    log_weights: list[float],
    outcomes: list[list[int]],
) -> tuple[list[float], float]:
    """Self-normalized importance means and effective sample size, stably scaled."""
    if not log_weights or len(log_weights) != len(outcomes):
        raise ValueError("One nonempty outcome vector is required per world")
    width = len(outcomes[0])
    if not width or any(len(row) != width for row in outcomes):
        raise ValueError("Every world must evaluate the same candidate set")
    maximum = max(log_weights)
    weights = [exp(log_weight - maximum) for log_weight in log_weights]
    total = sum(weights)
    means = [
        sum(w * row[i] for w, row in zip(weights, outcomes)) / total
        for i in range(width)
    ]
    effective = total**2 / sum(w * w for w in weights)
    return means, effective


def world_log_weight(
    info: BotInfo, world: SampledWorld, likelihoods: Likelihoods, evidence_weight: float
) -> float:
    total = 0.0
    for player in info.other_players:
        counts = Counter(world.hands[player.id])
        for gem, logs in likelihoods.get(player.id, {}).items():
            total += evidence_weight * logs[counts[gem]]
    return total


class WeightedSamplingReference(SamplingBot):
    """Previous 0.1-weight policy, kept outside the submitted bot definitions."""

    use_bid_evidence = True

    def __init__(
        self,
        *,
        decision_seconds: float = 0.85,
        max_worlds: int = 16,
        max_candidates: int = 7,
        evidence_weight: float = 0.1,
    ) -> None:
        if not isfinite(evidence_weight) or not 0 <= evidence_weight <= 1:
            raise ValueError("Evidence weight must be between zero and one")
        self.evidence_weight = evidence_weight
        super().__init__(
            decision_seconds=decision_seconds,
            max_worlds=max_worlds,
            max_candidates=max_candidates,
        )

    def on_game_start(self) -> None:
        super().on_game_start()
        self.tracker = SoftTrackingBot(
            use_bid_evidence=True,
            evidence_weight=self.evidence_weight,
            use_endgame_rules=False,
            use_finance_horizon=False,
            use_cash_fraction_model=False,
            use_strategic_reveal=False,
            use_bid_model=True,
            use_cash_limits=True,
        )
        self._world_weights: list[float] = []
        self._effective_worlds = 0.0

    def get_bid(self, info: BotInfo) -> int:
        self.tracker.observe_auction(info)
        self._world_weights = []
        self._effective_worlds = 0.0
        bid = super().get_bid(info)
        self.last_search["effective_worlds"] = self._effective_worlds
        return bid

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        super().on_round_end(info, result)
        self.tracker.on_round_end(info, result)

    def on_game_end(self, info: BotInfo, result: GameSummary) -> None:
        super().on_game_end(info, result)
        self.tracker.on_game_end(info, result)

    def _sample_world(self, info: BotInfo, rng: Random) -> SampledWorld:
        world = super()._sample_world(info, rng)
        self._world_weights.append(
            world_log_weight(
                info, world, self.tracker.likelihoods, self.evidence_weight
            )
        )
        return world

    def _aggregate_worlds(self, outcomes: list[list[int]]) -> list[float]:
        # Proposals are evaluated in order; discard any incomplete final batch.
        scores, effective = weighted_means(
            self._world_weights[: len(outcomes)], outcomes
        )
        self._effective_worlds = effective
        return scores
