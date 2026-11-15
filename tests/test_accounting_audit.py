"""Audit snapshots against host-side truth, and caps against actual auction rules.

Tests may inspect the hidden draw pile; this information is never sent to bots.
"""

import unittest
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from random import Random
from unittest.mock import patch

from bot_definitions.soft_tracking_bot import SoftTrackingBot, infer_locations
from bot_definitions.valuation_bot import AuctionValuation, ValuationBot
from bot_info import build_bot_info
from bot_runner import BotRunner
from constants import STANDARD_GEMS, AuctionType
from custom_bots import FixedRuleBot
from decks import GEM_CARDS, VALUE_CHARTS
from harness import create_game, finish_game, play_round
from lib import AuctionCard
from player import Player
from tests.test_valuation_bot import view


class AccountingAuditTests(unittest.TestCase):
    def test_all_round_snapshots_match_actual_hidden_hands_and_deck(self):
        two_gem_rounds = 0
        with ExitStack() as stack:
            runners = [
                stack.enter_context(BotRunner(FixedRuleBot, timeout=2, seed=i))
                for i in range(5)
            ]
            for seats in (3, 4, 5):
                for chart in VALUE_CHARTS:
                    with self.subTest(seats=seats, chart=chart.name):
                        with create_game(
                            seats,
                            bots=runners[:seats],
                            rng=Random(42),
                            value_chart=chart,
                        ) as game:
                            starting_hands = Counter(
                                g for p in game.players for g in p.hidden_gem_hand
                            )

                            def check_snapshots():
                                revealed = Counter(
                                    g for p in game.players for g in p.public_gem_hand
                                )
                                self.assertEqual(game.value_display, revealed)
                                remaining_hands = Counter(
                                    g for p in game.players for g in p.hidden_gem_hand
                                )
                                self.assertEqual(
                                    revealed + remaining_hands, starting_hands
                                )
                                sold = Counter()
                                for p in game.players:
                                    sold.update(p.gem_collection)
                                self.assertEqual(
                                    revealed
                                    + remaining_hands
                                    + sold
                                    + Counter(game.auction_gems)
                                    + Counter(game.gem_deck._cards),
                                    Counter(GEM_CARDS),
                                )
                                for p in game.players:
                                    info = build_bot_info(game, p)
                                    opponent_hands = Counter(
                                        g
                                        for other in game.players
                                        if other.id != p.id
                                        for g in other.hidden_gem_hand
                                    )
                                    actual_unknown = opponent_hands + Counter(
                                        game.gem_deck._cards
                                    )
                                    self.assertEqual(
                                        Counter(info.unknown_gems), actual_unknown
                                    )
                                    self.assertEqual(
                                        info.self_player.hidden_gems,
                                        tuple(p.hidden_gem_hand),
                                    )
                                    known_display = Counter(info.value_display)
                                    known_display.update(info.self_player.hidden_gems)
                                    self.assertEqual(
                                        known_display + opponent_hands, starting_hands
                                    )
                                    if not game.gem_deck:
                                        actual_prices = {
                                            g: chart.get_value(starting_hands[g])
                                            for g in STANDARD_GEMS
                                        }
                                        prior_prices = ValuationBot.expected_gem_values(
                                            info
                                        )
                                        # Bias one opponent's location likelihoods; with
                                        # no deck, display supply is fixed.
                                        logs = {
                                            info.other_players[0].id: {
                                                g: [n / 10 for n in range(7)]
                                                for g in STANDARD_GEMS
                                            }
                                        }
                                        tracked_prices = infer_locations(
                                            info, logs
                                        ).gem_values
                                        for gem in STANDARD_GEMS:
                                            self.assertAlmostEqual(
                                                prior_prices[gem], actual_prices[gem]
                                            )
                                            self.assertAlmostEqual(
                                                tracked_prices[gem], actual_prices[gem]
                                            )

                            check_snapshots()
                            while game.auction_gems:
                                before = sum(game.value_display.values())
                                result = play_round(game)
                                if result.auction_card.type is AuctionType.AUCTION:
                                    two_gem_rounds += int(
                                        result.auction_card.value == 2
                                    )
                                self.assertEqual(
                                    sum(game.value_display.values()) - before,
                                    int(result.revealed_gem is not None),
                                )
                                check_snapshots()
                            finish_game(game)
                            check_snapshots()
                            self.assertEqual(game.value_display, starting_hands)
        self.assertGreater(two_gem_rounds, 0)

    def test_broke_opponents_do_not_hide_a_rich_opponent(self):
        bot = SoftTrackingBot(use_cash_limits=True, use_bid_model=True)
        base = view()
        for rich_id in (1, 2, 3):
            info = replace(
                base,
                other_players=tuple(
                    replace(p, coins=10 if p.id == rich_id else 0)
                    for p in base.other_players
                ),
            )
            # One rival bids ten with certainty. Zero-cap rivals cannot erase it.
            pmfs = {
                p.id: ([0.0] * 10 + [1.0] if p.id == rich_id else [1.0])
                for p in info.other_players
            }
            for previous in (None, 0, 1, 2, 3):
                current = replace(info, previous_winner_id=previous)
                self.assertEqual(bot.win_probability(current, 1, pmfs), 0)
                self.assertEqual(bot.win_probability(current, 9, pmfs), 0)
                self.assertEqual(bot.win_probability(current, 11, pmfs), 1)
            self.assertEqual(bot.win_probability(info, 10, pmfs), 0.5)
            self.assertEqual(
                bot.win_probability(replace(info, previous_winner_id=0), 10, pmfs), 0
            )
            self.assertEqual(
                bot.win_probability(
                    replace(info, previous_winner_id=rich_id), 10, pmfs
                ),
                1,
            )
            forecast = {
                p.id: bot._bid_distribution(info, p) for p in info.other_players
            }
            self.assertLess(bot.win_probability(info, 1, forecast), 1)
            self.assertTrue(
                all(len(forecast[p.id]) == p.coins + 1 for p in info.other_players)
            )

    def test_valuation_uses_richest_cap_and_still_respects_own_cash(self):
        bot = ValuationBot()
        base = view()
        estimate = AuctionValuation(100, 0, 0, 1, 0, 0, 0)
        with (
            patch.object(bot, "evaluate", return_value=estimate),
            # Fix rivals at their maximum bids to isolate the affordability
            # cap from the learned model's deliberate bid shading.
            patch.object(
                bot,
                "_bid_distribution",
                side_effect=lambda info, player: [0.0] * player.coins + [1.0],
            ),
        ):
            for rich_id in (1, 2, 3):
                info = replace(
                    base,
                    other_players=tuple(
                        replace(p, coins=10 if p.id == rich_id else 0)
                        for p in base.other_players
                    ),
                )
                self.assertEqual(bot.get_bid(info), 11)
                poor = replace(info, self_player=replace(info.self_player, coins=5))
                # With a known rival bid of ten, no affordable bid can win.
                self.assertEqual(bot.get_bid(poor), 0)
            broke = replace(
                base,
                other_players=tuple(replace(p, coins=0) for p in base.other_players),
            )
            self.assertEqual(bot.get_bid(broke), 1)

    def test_zero_cash_loan_bids_include_proceeds_but_other_auctions_do_not(self):
        player = Player(0, 0, [])
        self.addCleanup(player.bot.close)
        base = view()
        loan = replace(base, current_auction=AuctionCard(AuctionType.LOAN, 20))
        treasure = replace(base, current_auction=AuctionCard(AuctionType.AUCTION, 1))
        invest = replace(base, current_auction=AuctionCard(AuctionType.INVEST, 10))
        with patch.object(player.bot, "call", return_value=20):
            self.assertEqual(player.get_bid(loan), 20)
            self.assertEqual(player.get_bid(treasure), 0)
            self.assertEqual(player.get_bid(invest), 0)
        with patch.object(player.bot, "call", return_value=21):
            self.assertEqual(player.get_bid(loan), 0)


if __name__ == "__main__":
    unittest.main()
