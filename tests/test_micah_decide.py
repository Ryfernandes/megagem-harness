"""Unit tests for MicahBot Milestone 3 bid choice."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

from test_micah_bot import _make_info  # noqa: E402

from bot_definitions.micah_bid_model import (  # noqa: E402
    OpponentBidModel,
    lot_reference_value,
    phase_is_late,
    probability_we_win,
)
from bot_definitions.micah_bot import MicahBot  # noqa: E402
from bot_definitions.micah_decide import choose_bid_m3  # noqa: E402
from bot_definitions.micah_strategy import (  # noqa: E402
    choose_bid,
    legal_bid_cap,
    public_gem_expected_values,
)
from bots import BotInfo, PlayerInfo, RoundResult  # noqa: E402
from constants import STANDARD_GEMS, AuctionType, GemType  # noqa: E402
from immutable import FrozenMap  # noqa: E402
from lib import AuctionCard  # noqa: E402

BLUE = GemType.BLUE_CIRCLE


def _one_opponent(coins: int) -> PlayerInfo:
    return PlayerInfo(
        id=1,
        coins=coins,
        hidden_gem_count=0,
        revealed_gems=(),
        won_gems=FrozenMap(dict.fromkeys(STANDARD_GEMS, 0)),
        auction_cards=(),
        mission_cards=(),
        invested_coins=0,
    )


def _late_treasure_info(opponent_coins: int = 15) -> BotInfo:
    """Late treasure where a 20-point gem is still worth a modest bid.

    Three treasure cards remain after this one, so this is late (blocking can
    turn on) but not the final-round sampler. Chart B prices a missing color
    at 20, and the only rival has 15 coins.
    """
    return _make_info(
        coins=50,
        hidden_gems=(),
        others=(_one_opponent(opponent_coins),),
        current_auction=AuctionCard(AuctionType.AUCTION, 1),
        gem_deck_count=3,
        unknown_gems=dict.fromkeys(STANDARD_GEMS, 0),
        auction_gems=(BLUE,),
        value_display=dict.fromkeys(STANDARD_GEMS, 0),
        previous_winner_id=1,
        chart_name="B",
        auction_cards_remaining={AuctionCard(AuctionType.AUCTION, 1): 3},
    )


class ChooseBidM3Tests(unittest.TestCase):
    def test_returns_legal_int_for_treasure_invest_and_early_loan(self) -> None:
        cases = (
            ("treasure", AuctionCard(AuctionType.AUCTION, 1)),
            ("invest", AuctionCard(AuctionType.INVEST, 10)),
            ("loan", AuctionCard(AuctionType.LOAN, 20)),
        )
        for name, card in cases:
            with self.subTest(name=name):
                info = _make_info(current_auction=card)
                bid = choose_bid_m3(info, OpponentBidModel())
                self.assertIs(type(bid), int)
                self.assertGreaterEqual(bid, 0)
                self.assertLessEqual(bid, legal_bid_cap(info))
                if card.type is AuctionType.LOAN:
                    self.assertFalse(phase_is_late(info))
                    self.assertLess(bid, card.value)

    def test_remembered_high_bids_change_our_bid(self) -> None:
        info = _late_treasure_info(opponent_coins=40)
        self.assertTrue(phase_is_late(info))
        opponent = info.other_players[0]
        public_evs = public_gem_expected_values(info)
        reference = lot_reference_value(info, opponent, public_evs)
        self.assertGreaterEqual(reference, 2.0)

        empty = OpponentBidModel()
        trained = OpponentBidModel()
        for _ in range(40):
            trained.record(
                opponent.id,
                AuctionType.AUCTION,
                True,
                opponent.coins,
                reference,
                100,
            )

        def chance(model: OpponentBidModel) -> float:
            distribution = model.bid_distribution(
                opponent.id,
                AuctionType.AUCTION,
                True,
                opponent.coins,
                reference,
                opponent.coins,
            )
            return probability_we_win(
                12,
                info.self_player.id,
                {opponent.id: distribution},
                info.previous_winner_id,
                info.num_players,
            )

        # A bid of 12 beats the prior's low guesses and loses to a rival who
        # has been bidding far above the lot.
        self.assertLess(chance(trained), chance(empty))
        for model in (empty, trained):
            bid = choose_bid_m3(info, model)
            self.assertIs(type(bid), int)
            self.assertGreaterEqual(bid, 0)
            self.assertLessEqual(bid, legal_bid_cap(info))

    def test_bot_records_bids_then_forgets_on_game_start(self) -> None:
        info = _late_treasure_info()
        card = info.current_auction
        self.assertIsNotNone(card)
        assert card is not None

        bot = MicahBot()
        bot.on_game_start()
        bid = bot.get_bid(info)
        result = RoundResult(
            auction_card=card,
            bids=FrozenMap({info.self_player.id: bid, info.other_players[0].id: 4}),
            winner_id=info.other_players[0].id,
            winning_bid=4,
            revealed_gem=None,
        )
        bot.on_round_end(info, result)
        self.assertTrue(bot.model.observations)

        bot.on_game_start()
        self.assertEqual(bot.model.observations, [])

    def test_milestone2_only_matches_choose_bid(self) -> None:
        key = "MICAH_MILESTONE2_ONLY"
        previous = os.environ.get(key)
        os.environ[key] = "1"
        try:
            bot = MicahBot()
            info = _make_info()
            self.assertEqual(bot.get_bid(info), int(choose_bid(info, bot.config)))
            self.assertEqual(bot.model.observations, [])
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


if __name__ == "__main__":
    unittest.main()
