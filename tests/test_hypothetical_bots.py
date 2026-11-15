"""Opponent-style legality, ordering and experiment configuration coverage."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from bot_definitions.hypothetical_bots import (
    AggressiveBidder,
    PassiveBidder,
    RandomBidder,
)
from bot_definitions.soft_tracking_bot import SoftTrackingBot
from constants import AuctionType
from decks import VALUE_CHARTS
from experiments.mixed_policies import CANDIDATES, POLICIES
from lib import AuctionCard
from simulation import bot_catalog
from tests.test_valuation_bot import view


class HypotheticalTests(unittest.TestCase):
    def test_styles_are_legal_across_charts_cash_levels_and_auction_types(self):
        cards = [
            AuctionCard(t, v)
            for t, values in (
                (AuctionType.AUCTION, (1, 2)),
                (AuctionType.INVEST, (5, 10)),
                (AuctionType.LOAN, (10, 20)),
            )
            for v in values
        ]
        for chart in VALUE_CHARTS:
            for card in cards:
                for cash in (0, 1, 10, 25, 80):
                    info = replace(
                        view(),
                        value_chart=chart,
                        current_auction=card,
                        self_player=replace(view().self_player, coins=cash),
                    )
                    bids = []
                    for cls in (AggressiveBidder, PassiveBidder, RandomBidder):
                        bid = cls().get_bid(info)
                        self.assertIs(type(bid), int)
                        self.assertGreaterEqual(bid, 0)
                        self.assertLessEqual(
                            bid,
                            cash + (card.value if card.type is AuctionType.LOAN else 0),
                        )
                        bids.append(bid)
                    self.assertGreaterEqual(bids[0], bids[1])

    def test_random_style_can_pass_or_bid_and_is_selectable(self):
        bot = RandomBidder()
        with patch("bot_definitions.hypothetical_bots.uniform", return_value=0):
            self.assertEqual(bot.get_bid(view()), 0)
        with patch("bot_definitions.hypothetical_bots.uniform", return_value=1.4):
            self.assertGreater(bot.get_bid(view()), 0)
        for name in ("RandomBidder", "AggressiveBidder", "PassiveBidder"):
            self.assertIn(name, bot_catalog())

    def test_all_inconclusive_feature_combinations_and_no_harmful_extras(self):
        flags = set()
        for name in CANDIDATES:
            cls = POLICIES[name]
            if issubclass(cls, SoftTrackingBot):
                self.assertFalse(cls.use_endgame_rules)
                self.assertFalse(cls.use_cash_fraction_model)
                self.assertTrue(cls.use_bid_model)
                self.assertTrue(cls.use_cash_limits)
                flags.add(
                    (
                        cls.use_bid_evidence,
                        cls.use_finance_horizon,
                        cls.use_strategic_reveal,
                    )
                )
        self.assertEqual(len(flags), 7)
        self.assertEqual(len(CANDIDATES), 9)
        self.assertFalse(
            any(name in CANDIDATES for name in ("random", "aggressive", "passive"))
        )


if __name__ == "__main__":
    unittest.main()
