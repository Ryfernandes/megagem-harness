"""MicahBot: analytic-core MegaGem tournament bot.

Strategy overview:
    1. Exact gem valuation: computes each gem color's exact terminal display
       distribution via hypergeometric sampling (combining revealed gems,
       this bot's private hidden hand, and unknown gems across the draw pile
       and opponents' hands) and maps counts through the active ValueChart
       (A-E).
    2. Concave liquidity & multi-round horizon: estimates remaining auction
       rounds from remaining gems and undrawn auction cards, and values cash
       with a diminishing-returns curve so early scarce coins carry a higher
       option value while final-round cash reduces to face value.
    3. Auction-specific utility:
       - Treasure: expected gem value + immediate multi-mission completions +
         feasibility-discounted mission progress - liquidity-adjusted cost -
         tie-priority penalty.
       - Investment: bonus minus lost liquidity and tie-priority penalty
         (accounting for bid return at scoring).
       - Loan: concave liquidity value of (L - bid) minus principal-repayment
         cost bid and tie-priority penalty, never bidding >= L without denial.
    4. Bid selection & reveals: scores every legal integer bid against a
       tie-order-aware win-probability estimate (hard-capped at opponents'
       legal coin limits). Reveals hidden gems with zero/minimal future supply
       first, and otherwise uses chart-aware public_value_shift to depress
       prices of colors MicahBot wants and inflate prices rivals contest.
    5. Opponent bid modeling & endgame (Milestone 3): tracks each opponent's
       bid-to-public-value ratio per game with censoring awareness for cash-
       capped all-ins and cash-poor passes, adds scaled late blocking against
       contenders, and evaluates terminal-round bids by first-place probability.
    6. Opponent hand inference (Milestone 4): maintains weak Bayesian beliefs
       over each opponent's hidden-hand color counts using chart-aware bid
       likelihoods (explaining away mission-driven bids) and reveal choices,
       feeding those beliefs into gem valuation and endgame sampling.
    7. Bounded lookahead (Milestone 5): for close or high-impact decisions
       (late treasure, loans, mission races, near-tied bids), evaluates top
       candidates on Common-Random-Numbers rollout worlds using the remaining
       decks and opponent models, strictly bounded by ``timeout_seconds``.

Contributors:
    Micah
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping

from bot_definitions.micah_bid_model import (
    OpponentBidModel,
    lot_reference_value,
    phase_is_late,
)
from bot_definitions.micah_decide import choose_bid_m3, pending_observation_rows
from bot_definitions.micah_inference import (
    InferenceSnapshot,
    OpponentHandBeliefs,
    build_inference_snapshot,
)
from bot_definitions.micah_strategy import (
    DEFAULT_STRATEGY_CONFIG,
    StrategyConfig,
    choose_bid,
    choose_reveal_index,
    public_gem_expected_values,
)
from bots import Bot, BotInfo, GameSummary, RoundResult
from constants import STANDARD_GEMS, AuctionType, GemType


class MicahBot(Bot):
    """Tournament bot for MegaGem."""

    def __init__(self) -> None:
        self.games_played: int = 0
        self.rounds_seen: int = 0
        self.config: StrategyConfig = DEFAULT_STRATEGY_CONFIG
        self.model = OpponentBidModel()
        self.hand_beliefs = OpponentHandBeliefs()
        self.pending: list[tuple[int, AuctionType, bool, int, float, int]] | None = None
        self.pending_inference: InferenceSnapshot | None = None
        self.use_milestone3 = os.environ.get("MICAH_MILESTONE2_ONLY") != "1"
        self.use_hand_inference = os.environ.get("MICAH_NO_INFERENCE") != "1"
        self.use_rollouts = os.environ.get("MICAH_NO_ROLLOUT") != "1"

    def on_game_start(self) -> None:
        self.games_played += 1
        self.rounds_seen = 0
        self.model.clear()
        self.hand_beliefs.clear()
        self.pending = None
        self.pending_inference = None

    def get_bid(self, info: BotInfo) -> int:
        if not self.use_milestone3:
            return int(choose_bid(info, self.config))
        public_evs: Mapping[GemType, float]
        if info.current_auction is None:
            public_evs = {}
        else:
            public_evs = public_gem_expected_values(info)
        self.pending = list(pending_observation_rows(info, public_evs))

        hand_beliefs: OpponentHandBeliefs | None = None
        if self.use_hand_inference and info.current_auction is not None:
            late = phase_is_late(info)
            opp_ratios = {
                opp.id: self.model.estimate_bid_ratio(
                    opp.id,
                    info.current_auction.type,
                    late,
                    opp.coins,
                    lot_reference_value(info, opp, public_evs),
                    self.config,
                )
                for opp in info.other_players
            }
            self.pending_inference = build_inference_snapshot(
                info, opp_ratios, self.config
            )
            hand_beliefs = self.hand_beliefs
        else:
            self.pending_inference = None

        rng = random.Random(random.randrange(2**30))
        return int(
            choose_bid_m3(
                info,
                self.model,
                self.config,
                rng,
                hand_beliefs=hand_beliefs,
                use_rollouts=self.use_rollouts,
                timeout_seconds=self.timeout_seconds,
            )
        )

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        return int(choose_reveal_index(info, bids, self.config))

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        if self.use_milestone3 and self.pending is not None:
            for (
                opp_id,
                auction_type,
                phase_late,
                coins,
                reference,
                legal_cap,
            ) in self.pending:
                self.model.record(
                    opp_id,
                    auction_type,
                    phase_late,
                    coins,
                    reference,
                    int(result.bids.get(opp_id, 0)),
                    legal_cap,
                )
            self.pending = None

        if (
            self.use_milestone3
            and self.use_hand_inference
            and self.pending_inference is not None
        ):
            snap = self.pending_inference
            for opp in info.other_players:
                bid = int(result.bids.get(opp.id, 0))
                self.hand_beliefs.update_from_bid(
                    opp.id,
                    snap,
                    bid,
                    strength=self.config.bid_inference_strength,
                    spread_ratio=self.config.win_prob_spread_ratio,
                )
            if (
                result.revealed_gem is not None
                and result.winner_id != info.self_player.id
            ):
                winner_opp = next(
                    (o for o in info.other_players if o.id == result.winner_id),
                    None,
                )
                if winner_opp is not None:
                    new_unknown = tuple(
                        int(info.unknown_gems.get(g, 0)) for g in STANDARD_GEMS
                    )
                    self.hand_beliefs.update_from_reveal(
                        result.winner_id,
                        result.revealed_gem,
                        snap,
                        new_unknown,
                        winner_opp.won_gems,
                        info.available_missions,
                        reveal_strength=self.config.reveal_inference_strength,
                    )
            self.pending_inference = None

        self.rounds_seen += 1

    def on_game_end(self, info: BotInfo, result: GameSummary) -> None:
        pass
