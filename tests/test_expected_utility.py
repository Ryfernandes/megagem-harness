"""EU arithmetic, action-set optimization, cash safety and financial isolation."""

import pickle
import unittest
from dataclasses import replace
from unittest.mock import patch

from bot_definitions.expected_utility import expected_utility, opportunity_cost
from bot_definitions.soft_tracking_bot import UTILITY_BOTS, SoftTrackingBot
from bot_definitions.valuation_bot import AuctionValuation
from bot_definitions.valuation_bot import FractionalValuationBot as ValuationBot
from bot_definitions.valuation_bot import ValuationBot as CurrentLevelTwo
from constants import AuctionType
from lib import AuctionCard
from tests.test_valuation_bot import view


class ExpectedUtilityTests(unittest.TestCase):
    def test_formula_prices_win_probability_and_opportunity_cost(self):
        self.assertAlmostEqual(expected_utility(0.9, 10, 3, 1), 3.6)
        self.assertAlmostEqual(expected_utility(0.2, 10, 2, 1), 1.2)
        self.assertEqual(expected_utility(0, 10, 20, 4), 0)
        self.assertEqual(expected_utility(1, 10, 5, 1), 0)
        self.assertLess(
            expected_utility(0.9, 10, 3, 2), expected_utility(0.9, 10, 3, 1)
        )

    def _fixture(self):
        info = view()
        info = replace(
            info,
            previous_winner_id=3,
            other_players=tuple(
                replace(p, coins=5 if p.id == 1 else 0) for p in info.other_players
            ),
        )
        distributions = {1: [0.2, 0, 0, 0.7, 0.1, 0], 2: [1.0], 3: [1.0]}
        estimate = AuctionValuation(10, 0, 0, 2, 0, 0, 0)
        return info, distributions, estimate

    def test_optimizer_picks_higher_expected_profit_not_fixed_fraction(self):
        info, pmfs, estimate = self._fixture()
        bot = UTILITY_BOTS["heuristic"]()
        with (
            patch.object(bot, "evaluate", return_value=estimate),
            patch.object(
                bot, "_bid_distribution", side_effect=lambda info, p: pmfs[p.id]
            ),
        ):
            self.assertEqual(bot.get_bid(info), 3)
        base = ValuationBot()
        with patch.object(base, "evaluate", return_value=estimate):
            self.assertEqual(base.get_bid(info), 2)
        # Bid 3 wins 90% with four points of surplus; bid 2 wins 20% with six.
        self.assertGreater(0.9 * 4, 0.2 * 6)

    def test_all_affordable_bids_are_considered_without_old_value_discount(self):
        info = view()
        info = replace(
            info,
            previous_winner_id=3,
            other_players=tuple(
                replace(p, coins=10 if p.id == 1 else 0) for p in info.other_players
            ),
        )
        pmfs = {1: [0.0] * 10 + [1.0], 2: [1.0], 3: [1.0]}
        estimate = AuctionValuation(10.1, 0, 0, 1, 0, 0, 0)
        for name, expected in [("old_eu", 0), ("heuristic", 10)]:
            bot = UTILITY_BOTS[name]()
            with (
                patch.object(bot, "evaluate", return_value=estimate),
                patch.object(
                    bot, "_bid_distribution", side_effect=lambda info, p: pmfs[p.id]
                ),
            ):
                self.assertEqual(bot.get_bid(info), expected)

    def test_larger_cash_penalty_shades_bids_and_expires_at_final_lot(self):
        info, pmfs, estimate = self._fixture()
        bids = []
        for scale in (1, 4):
            bot = UTILITY_BOTS["heuristic"](utility_k_scale=scale)
            with (
                patch.object(bot, "evaluate", return_value=estimate),
                patch.object(
                    bot, "_bid_distribution", side_effect=lambda info, p: pmfs[p.id]
                ),
            ):
                bids.append(bot.get_bid(info))
        self.assertLess(bids[1], bids[0])
        early = opportunity_cost(info, 1, 1, 2)
        self.assertGreater(early, 0)
        last = replace(
            info, gems_remaining=1, auction_gems=info.auction_gems[:1], gem_deck_count=0
        )
        self.assertEqual(opportunity_cost(last, 1, 4, 2), 0)

    def test_reserve_and_financial_policy_are_unchanged_for_all_variants(self):
        for cls in UTILITY_BOTS.values():
            bot = cls()
            for cash in (0, 1, 25):
                info = replace(
                    view(), self_player=replace(view().self_player, coins=cash)
                )
                bid = bot.get_bid(info)
                reserve = ValuationBot().evaluate(info).reserve
                self.assertTrue(0 <= bid <= cash - reserve)
                for kind, amount in ((AuctionType.INVEST, 10), (AuctionType.LOAN, 20)):
                    financial = replace(info, current_auction=AuctionCard(kind, amount))
                    self.assertEqual(
                        bot.get_bid(financial), ValuationBot().get_bid(financial)
                    )

    def test_penalty_configuration_survives_checkpoint_and_new_game(self):
        bot = SoftTrackingBot(
            utility_k_scale=2, utility_k_floor=1, utility_player_counts=(3, 4, 5)
        )
        clone = SoftTrackingBot.__new__(SoftTrackingBot)
        clone.load_state(pickle.loads(pickle.dumps(bot.save_state())))
        clone.on_game_start()
        self.assertEqual((clone.utility_k_scale, clone.utility_k_floor), (2, 1))
        self.assertEqual(clone.utility_player_counts, (3, 4, 5))

    def test_default_policy_matches_current_level_two_at_every_seat_count(self):
        bot = SoftTrackingBot()
        self.assertEqual(bot.utility_player_counts, (3, 4, 5))
        self.assertTrue(bot.use_bid_model and bot.use_cash_limits)
        self.assertFalse(
            bot.use_bid_evidence
            or bot.use_endgame_rules
            or bot.use_finance_horizon
            or bot.use_cash_fraction_model
            or bot.use_strategic_reveal
        )
        self.assertEqual((bot.utility_k_scale, bot.utility_k_floor), (1, 0.5))
        for count in (3, 4, 5):
            info = replace(view(), num_players=count)
            self.assertEqual(bot.get_bid(info), CurrentLevelTwo().get_bid(info))
        info = replace(view(), num_players=3)
        self.assertEqual(bot.get_bid(info), UTILITY_BOTS["floor0.5"]().get_bid(info))

    def test_invalid_parameters_are_rejected(self):
        for probability in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                expected_utility(probability, 10, 1, 0)
        for penalty in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                SoftTrackingBot(utility_k_scale=penalty)
            with self.assertRaises(ValueError):
                opportunity_cost(view(), 1, 1, penalty)


if __name__ == "__main__":
    unittest.main()
