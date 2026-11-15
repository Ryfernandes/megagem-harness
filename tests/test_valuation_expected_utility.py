"""Public valuation bidding, policy isolation, and learned-state checkpoints."""

import pickle
import unittest
from random import Random

from bot_definitions.soft_tracking_bot import SoftTrackingBot
from bot_definitions.valuation_bot import FractionalValuationBot, ValuationBot
from bot_info import build_bot_info
from decks import VALUE_CHARTS
from harness import create_game, finish_game, play_round
from tests.test_soft_tracking_bot import after_round
from tests.test_valuation_bot import view


class ValuationExpectedUtilityTests(unittest.TestCase):
    def test_shared_bidder_and_old_policy_switch_match_on_real_rounds(self):
        with self.assertNoLogs("bot_runner", level="WARNING"):
            for seats in (3, 4, 5):
                for chart in VALUE_CHARTS:
                    new = ValuationBot()
                    disabled = ValuationBot(use_expected_utility=False)
                    before = FractionalValuationBot()
                    soft = SoftTrackingBot()
                    self.assertFalse(soft.use_finance_horizon)
                    self.assertFalse(soft.use_cash_fraction_model)
                    self.assertFalse(soft.use_strategic_reveal)
                    with create_game(
                        seats,
                        bots=[FractionalValuationBot] * seats,
                        rng=Random(2100),
                        value_chart=chart,
                        bot_timeout=2.0,
                    ) as game:

                        def observe(info):
                            self.assertEqual(new.evaluate(info), before.evaluate(info))
                            self.assertEqual(new.get_bid(info), soft.get_bid(info))
                            self.assertEqual(
                                disabled.get_bid(info), before.get_bid(info)
                            )
                            self.assertEqual(
                                new.choose_hidden_gem(info, {}),
                                before.choose_hidden_gem(info, {}),
                            )

                        while game.auction_gems:
                            result = play_round(game, before_bid=observe)
                            after = build_bot_info(game, game.players[0])
                            new.on_round_end(after, result)
                            soft.on_round_end(after, result)
                            disabled.on_round_end(after, result)
                            self.assertEqual(new.bid_rates, soft.bid_rates)
                        finish_game(game)
                        self.assertTrue(new.bid_rates)
                        self.assertFalse(disabled.bid_rates)

    def test_learned_rates_checkpoint_and_reset_without_resetting_switch(self):
        info = view()
        bot = ValuationBot()
        bot.get_bid(info)
        after, result = after_round(info, {0: 1, 1: 8, 2: 2, 3: 0})
        bot.on_round_end(after, result)
        self.assertTrue(bot.bid_rates)
        restored = ValuationBot.__new__(ValuationBot)
        restored.load_state(pickle.loads(pickle.dumps(bot.save_state())))
        self.assertEqual(bot.bid_rates, restored.bid_rates)
        self.assertEqual(bot.get_bid(after), restored.get_bid(after))
        restored.on_game_start()
        self.assertEqual(restored.bid_rates, {})
        self.assertIsNone(restored._pending)
        self.assertTrue(restored.use_expected_utility)
        disabled = ValuationBot(use_expected_utility=False)
        disabled.on_game_start()
        self.assertFalse(disabled.use_expected_utility)


if __name__ == "__main__":
    unittest.main()
