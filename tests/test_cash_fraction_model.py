"""Type/cash conditioning, pre-bid timing, EU optimization and financial costs."""

import pickle
import unittest
from dataclasses import replace
from unittest.mock import patch

from bot_definitions.cash_fraction_model import CashFractionModel, available_budget
from bot_definitions.soft_tracking_bot import SoftTrackingBot
from bot_definitions.valuation_bot import AuctionValuation, ValuationBot
from constants import AuctionType
from decks import VALUE_CHARTS
from lib import AuctionCard
from tests.test_soft_tracking_bot import after_round
from tests.test_valuation_bot import view

TREASURE, INVEST, LOAN = AuctionType.AUCTION, AuctionType.INVEST, AuctionType.LOAN


class CashFractionTests(unittest.TestCase):
    def test_pmf_is_normalized_capped_by_current_cash_and_loan_proceeds(self):
        model = CashFractionModel()
        for kind, amount in ((TREASURE, 1), (INVEST, 10), (LOAN, 20)):
            for cash in (0, 1, 6, 25):
                info = replace(view(), current_auction=AuctionCard(kind, amount))
                player = replace(info.other_players[0], coins=cash)
                pmf = model.distribution(info, player)
                budget = cash + (amount if kind is LOAN else 0)
                self.assertEqual(len(pmf), budget + 1)
                self.assertAlmostEqual(sum(pmf), 1)
                self.assertTrue(all(p >= 0 for p in pmf))
                self.assertAlmostEqual(sum(pmf[: budget + 1]), 1)
                self.assertEqual(available_budget(info, player), budget)
        with self.assertRaises(ValueError):
            model.observe(view(), replace(view().other_players[0], coins=6), 7)

    def test_history_is_separate_by_opponent_and_auction_type(self):
        model = CashFractionModel()
        info = view()
        player = replace(info.other_players[0], coins=20)
        initial = model.fraction_moments(player.id, TREASURE)
        model.observe(info, player, 16)
        self.assertGreater(model.fraction_moments(player.id, TREASURE)[0], initial[0])
        self.assertEqual(model.fraction_moments(2, TREASURE), initial)
        self.assertEqual(model.fraction_moments(player.id, INVEST), (0.20, 0.16))
        investment = replace(info, current_auction=AuctionCard(INVEST, 10))
        model.observe(investment, player, 16)
        treasure_shift = model.fraction_moments(player.id, TREASURE)[0] - 0.20
        invest_shift = model.fraction_moments(player.id, INVEST)[0] - 0.20
        self.assertGreater(invest_shift, treasure_shift)
        self.assertEqual(model.history[player.id][TREASURE][0].weight, 0.25)
        self.assertEqual(model.history[player.id][INVEST][0].weight, 1)

    def test_empirical_samples_scale_with_current_cash_and_keep_uncertainty(self):
        model = CashFractionModel()
        info = view()
        player = replace(info.other_players[0], coins=20)
        for _ in range(20):
            model.observe(info, player, 4)  # consistently 20% of the old bankroll
        rich = replace(player, coins=40)
        pmf = model.distribution(info, rich)
        self.assertEqual(max(range(len(pmf)), key=pmf.__getitem__), 8)
        self.assertGreater(pmf[8], 0.7)
        self.assertGreater(sum(pmf[9:]), 0)  # broad prior never collapses away
        self.assertGreater(model.fraction_moments(player.id, TREASURE)[1], 0)

    def test_zero_cash_is_logged_without_learning_a_forced_pass(self):
        model = CashFractionModel()
        info = view()
        player = replace(info.other_players[0], coins=0)
        before = model.fraction_moments(player.id, TREASURE)
        model.observe(info, player, 0)
        record = model.history[player.id][TREASURE][0]
        self.assertIsNone(record.fraction)
        self.assertEqual(record.weight, 0)
        self.assertEqual(model.fraction_moments(player.id, TREASURE), before)
        loan = replace(info, current_auction=AuctionCard(LOAN, 20))
        model.observe(loan, player, 5)
        self.assertEqual(model.history[player.id][LOAN][0].fraction, 0.25)

    def test_learning_uses_pre_auction_cash_all_validated_bids_and_types(self):
        for kind, amount in ((TREASURE, 1), (INVEST, 10), (LOAN, 20)):
            bot = SoftTrackingBot(use_cash_fraction_model=True)
            info = replace(view(), current_auction=AuctionCard(kind, amount))
            bot.get_bid(info)
            after, result = after_round(info, {0: 0, 1: 6, 2: 2, 3: 0})
            after = replace(
                after,
                other_players=tuple(replace(p, coins=999) for p in after.other_players),
            )
            self.assertEqual(bot.cash_fraction_model.history, {})
            bot.on_round_end(after, result)
            for player in info.other_players:
                record = bot.cash_fraction_model.history[player.id][kind][0]
                budget = player.coins + (amount if kind is LOAN else 0)
                self.assertEqual(record.coins_before, player.coins)
                self.assertEqual(record.fraction, result.bids[player.id] / budget)
            bot.on_round_end(after, result)
            self.assertEqual(len(bot.cash_fraction_model.history[1][kind]), 1)

    def test_bid_probability_changes_with_bid_and_with_tie_priority(self):
        bot = SoftTrackingBot(use_cash_fraction_model=True)
        info = replace(
            view(),
            other_players=tuple(
                replace(p, coins=6 if p.id == 1 else 0) for p in view().other_players
            ),
        )
        distributions = {
            p.id: bot._bid_distribution(info, p) for p in info.other_players
        }
        probs = [bot.win_probability(info, c, distributions) for c in range(8)]
        self.assertEqual(probs, sorted(probs))
        self.assertLess(probs[0], probs[3])
        self.assertAlmostEqual(probs[7], 1)
        tied = {1: [0.0] * 6 + [1.0], 2: [1.0], 3: [1.0]}
        estimate = AuctionValuation(30, 0, 0, 1, 0, 0, 0)
        with (
            patch.object(bot, "evaluate", return_value=estimate),
            patch.object(
                bot, "_bid_distribution", side_effect=lambda info, p: tied[p.id]
            ),
        ):
            self.assertEqual(bot.get_bid(replace(info, previous_winner_id=3)), 6)
            self.assertEqual(bot.get_bid(replace(info, previous_winner_id=0)), 7)

    def test_financial_utility_returns_invested_bid_and_offsets_loan_principal(self):
        bot = SoftTrackingBot(use_cash_fraction_model=True)
        info = replace(view(), previous_winner_id=3)
        # Rival surely bids four; we win the tie. Investing four for five bonus
        # coins is profitable at k=.7 because the principal is returned.
        invest = replace(info, current_auction=AuctionCard(INVEST, 5))
        pmfs = {1: [0.0] * 4 + [1.0], 2: [1.0], 3: [1.0]}
        estimate = AuctionValuation(0, 0, 0, 1, 0, 20, 100)
        with (
            patch.object(bot, "evaluate", return_value=estimate),
            patch.object(
                bot, "_bid_distribution", side_effect=lambda info, p: pmfs[p.id]
            ),
        ):
            self.assertEqual(bot.get_bid(invest), 4)
            loan = replace(info, current_auction=AuctionCard(LOAN, 20))
            self.assertEqual(bot.get_bid(loan), 0)  # no premium => no free loan profit
        productive = replace(estimate, liquidity=3)
        loan = replace(
            info,
            current_auction=AuctionCard(LOAN, 20),
            self_player=replace(info.self_player, coins=0),
        )
        with (
            patch.object(bot, "evaluate", return_value=productive),
            patch.object(
                bot, "_bid_distribution", side_effect=lambda info, p: pmfs[p.id]
            ),
        ):
            self.assertEqual(bot.get_bid(loan), 4)
            self.assertTrue(0 < bot.get_bid(loan) < 20)

    def test_off_default_checkpoint_reset_and_bounded_history(self):
        self.assertFalse(SoftTrackingBot().use_cash_fraction_model)
        bot = SoftTrackingBot(use_cash_fraction_model=True)
        info = view()
        bot.get_bid(info)
        after, result = after_round(info, {0: 1, 1: 4, 2: 2, 3: 0})
        bot.on_round_end(after, result)
        clone = SoftTrackingBot.__new__(SoftTrackingBot)
        clone.load_state(pickle.loads(pickle.dumps(bot.save_state())))
        self.assertEqual(
            clone.cash_fraction_model.history, bot.cash_fraction_model.history
        )
        self.assertEqual(clone.get_bid(after), bot.get_bid(after))
        clone.on_game_start()
        self.assertEqual(clone.cash_fraction_model.history, {})
        self.assertTrue(clone.use_cash_fraction_model)
        for _ in range(50):
            bot.cash_fraction_model.observe(info, info.other_players[0], 2)
        self.assertEqual(len(bot.cash_fraction_model.history[1][TREASURE]), 32)
        with self.assertRaises(ValueError):
            SoftTrackingBot(use_cash_fraction_model=True, use_cash_limits=False)
        with self.assertRaises(ValueError):
            SoftTrackingBot(use_cash_fraction_model=True, use_bid_model=False)

    def test_default_still_matches_level_two_and_other_extras_stay_off(self):
        on = SoftTrackingBot(use_cash_fraction_model=True)
        self.assertFalse(
            on.use_bid_evidence or on.use_endgame_rules or on.use_finance_horizon
        )
        for chart in VALUE_CHARTS:
            info = replace(view(), value_chart=chart)
            self.assertEqual(
                SoftTrackingBot().get_bid(info), ValuationBot().get_bid(info)
            )
            self.assertEqual(on.evaluate(info), ValuationBot().evaluate(info))


if __name__ == "__main__":
    unittest.main()
