"""MicahBot: price every gem exactly, then bid only when the points are worth it.

Strategy:
    1. Gem values. Each color's final chart value is an exact average over
       the gems still hidden. The average uses the value display, this bot's
       own hidden hand, and the public unknown pool (opponents' hands and
       the draw pile counted together). It works for charts A through E.
       Hidden deck order and other players' hidden gems are never read.
    2. Coin price. Remaining auctions are estimated from the gems and auction
       cards left. Early coins are worth more than 1 point, with diminishing
       returns. At the end a coin is worth its face value.
    3. What a win is worth.
       - Treasure: gem value, plus missions finished now, plus discounted
         progress on missions that are still possible, minus the coin cost,
         minus a penalty for winning the tie-break and bidding last next time.
       - Investment: the bonus, minus the cost of locking the coins up. The
         bid itself comes back at scoring.
       - Loan: the extra value of holding the borrowed coins, minus the bid.
         The bot does not bid the whole loan unless blocking a rival pays for it.
    4. Choosing a bid and a reveal. Every legal integer bid is scored by the
       chance of winning times the points gained. That chance uses the tie
       order and never assumes an opponent can bid more coins than they have.
       The reveal prefers a color that will not be sold again. Otherwise it
       reveals a gem that makes colors this bot still wants look cheaper, and
       colors rivals want look more expensive.
    5. Opponent bids and the endgame. Each opponent's bid-to-value ratio is
       learned during the game. An all-in that hit their coin cap, and a zero
       from a broke player, are not treated as their real strategy. Late in
       the game the bot will spend extra to block a contender, and the last
       bids are scored by the chance of finishing in first place.
    6. Hidden hands. A weak probability is kept for each opponent's color
       counts. It is updated from their bids and from gems they reveal, and
       mission-driven bids are not treated as pure color preference. Those
       beliefs feed the gem values and the endgame.
    7. Short lookahead. On close or expensive decisions (late treasure, loans,
       mission races, near-tied bids) the top bids are tried on a few possible
       futures. The same random worlds are used for every candidate. The
       search stops inside ``timeout_seconds``.

UltimateBot is the tournament entry. This class is one of the specialists
it calls. It is not a separate submission.

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
