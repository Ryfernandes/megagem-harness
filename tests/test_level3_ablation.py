"""Feature isolation and legal actions for the level-3 ablation switches."""

import pickle
import unittest
from dataclasses import replace
from random import Random
from unittest.mock import patch

from bot_definitions.soft_tracking_bot import (
    ABLATION_BOTS,
    FEATURE_NAMES,
)
from bot_definitions.valuation_bot import FractionalValuationBot as ValuationBot
from constants import AuctionType
from decks import AUCTION_CARDS, VALUE_CHARTS
from harness import create_game, finish_game, play_round
from immutable import FrozenMap
from lib import AuctionCard, MissionCard
from tests.test_soft_tracking_bot import after_round, end_view
from tests.test_valuation_bot import BLUE, PURPLE, view


class AblationTests(unittest.TestCase):
    def test_all_sixteen_configs_pin_flags_and_survive_checkpoints(self):
        self.assertEqual(len(ABLATION_BOTS), 16)
        for name, cls in ABLATION_BOTS.items():
            enabled = set() if name == "baseline" else set(name.split("+"))
            bot = cls()
            actual = dict(
                zip(
                    FEATURE_NAMES,
                    (
                        bot.use_bid_evidence,
                        bot.use_endgame_rules,
                        bot.use_cash_limits,
                        bot.use_bid_model,
                    ),
                )
            )
            self.assertEqual({k for k, v in actual.items() if v}, enabled)
            clone = cls.__new__(cls)
            clone.load_state(pickle.loads(pickle.dumps(bot.save_state())))
            clone.on_game_start()
            self.assertEqual(
                (
                    clone.use_bid_evidence,
                    clone.use_endgame_rules,
                    clone.use_cash_limits,
                    clone.use_bid_model,
                ),
                tuple(actual.values()),
            )

    def test_all_off_matches_level_two_decisions_for_every_chart_and_auction(self):
        bot = ABLATION_BOTS["baseline"]()
        reference = ValuationBot()
        for chart in VALUE_CHARTS:
            for cash in (0, 1, 5, 25, 50):
                for card in set(AUCTION_CARDS):
                    info = replace(
                        view(),
                        value_chart=chart,
                        current_auction=card,
                        self_player=replace(view().self_player, coins=cash),
                    )
                    self.assertEqual(bot.get_bid(info), reference.get_bid(info))

    def test_all_off_full_games_match_three_to_five_level_two_players(self):
        def run(seats, seed, chart, cls):
            with create_game(
                seats,
                bots=[cls] + [ValuationBot] * (seats - 1),
                rng=Random(seed),
                bot_seed=seed,
                value_chart=chart,
                bot_timeout=2.0,
            ) as game:
                rounds = []
                while game.auction_gems:
                    rounds.append(play_round(game))
                return rounds, finish_game(game).scores

        with self.assertNoLogs("bot_runner", level="WARNING"):
            for seats in (3, 4, 5):
                self.assertEqual(
                    run(seats, 360, VALUE_CHARTS[seats - 3], ABLATION_BOTS["baseline"]),
                    run(seats, 360, VALUE_CHARTS[seats - 3], ValuationBot),
                )

    def test_inference_alone_updates_beliefs_without_bid_optimization(self):
        bot = ABLATION_BOTS["inference"]()
        info = view()
        with patch.object(
            bot, "_bid_distribution", side_effect=AssertionError("unwanted model")
        ):
            bot.get_bid(info)
            after, result = after_round(info, {0: 1, 1: 25, 2: 1, 3: 1})
            bot.on_round_end(after, result)
            self.assertNotAlmostEqual(
                bot.gem_values(after)[PURPLE],
                ValuationBot.expected_gem_values(after)[PURPLE],
            )
            self.assertEqual(bot.bid_rates, {})
            self.assertEqual(bot.get_bid(after), ValuationBot.get_bid(bot, after))

    def test_endgame_rules_act_only_when_enabled(self):
        info = end_view()
        self.assertGreater(ABLATION_BOTS["baseline"]().get_bid(info), 0)
        self.assertEqual(ABLATION_BOTS["endgame"]().get_bid(info), 0)
        mission = MissionCard(10, {BLUE: 1, PURPLE: 1})
        completing = replace(
            info,
            self_player=replace(info.self_player, won_gems=FrozenMap({BLUE: 1})),
            available_missions=(mission,),
        )
        self.assertGreater(ABLATION_BOTS["endgame"]().get_bid(completing), 0)

    def test_cash_conditioning_is_independent_of_the_bid_model(self):
        info = replace(
            view(),
            other_players=tuple(replace(p, coins=0) for p in view().other_players),
        )
        uncapped = ABLATION_BOTS["bid_model"]()
        capped = ABLATION_BOTS["cash_limits+bid_model"]()
        for p in info.other_players:
            self.assertGreater(len(uncapped._bid_distribution(info, p)), 1)
            self.assertEqual(capped._bid_distribution(info, p), [1.0])
        self.assertLess(
            uncapped.win_probability(
                info,
                1,
                {p.id: uncapped._bid_distribution(info, p) for p in info.other_players},
            ),
            1,
        )
        self.assertEqual(
            capped.win_probability(info, 1, {p.id: [1.0] for p in info.other_players}),
            1,
        )
        # Cash conditioning has nothing to condition without a probabilistic
        # predictor. It does not disable level 2's existing affordability caps.
        self.assertEqual(
            ABLATION_BOTS["cash_limits"]().get_bid(info),
            ABLATION_BOTS["baseline"]().get_bid(info),
        )

    def test_every_configuration_returns_affordable_integer_bids(self):
        for cls in ABLATION_BOTS.values():
            for cash in (0, 1, 25):
                for kind, value in (
                    (AuctionType.AUCTION, 1),
                    (AuctionType.INVEST, 10),
                    (AuctionType.LOAN, 20),
                ):
                    info = replace(
                        view(),
                        current_auction=AuctionCard(kind, value),
                        self_player=replace(view().self_player, coins=cash),
                    )
                    bid = cls().get_bid(info)
                    limit = cash + (value if kind is AuctionType.LOAN else 0)
                    self.assertIs(type(bid), int)
                    self.assertTrue(0 <= bid <= limit)


if __name__ == "__main__":
    unittest.main()
