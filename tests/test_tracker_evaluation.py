"""Forecast timing, no intervention/leakage, proper losses, and clustered scoring."""

import copy
import unittest
from dataclasses import replace
from random import Random
from unittest.mock import patch

from bot_definitions.soft_tracking_bot import SoftTrackingBot
from bot_definitions.valuation_bot import ValuationBot
from bot_info import build_bot_info
from constants import STANDARD_GEMS
from decks import VALUE_CHARTS
from experiments.evaluate_tracker import (
    METRICS,
    ShadowForecasts,
    aggregate,
    losses,
    moments,
    prediction_gate,
    prior_final_pmf,
)
from harness import create_game, finish_game, play_round
from tests.test_soft_tracking_bot import after_round, end_view
from tests.test_valuation_bot import PURPLE, view


class TrackerEvaluationTests(unittest.TestCase):
    def test_prior_pmf_matches_level_two_prices_and_initial_tracker(self):
        for chart in VALUE_CHARTS:
            info = replace(view(), value_chart=chart)
            shadow = ShadowForecasts()
            with patch.object(
                shadow.tracker,
                "get_bid",
                side_effect=AssertionError("tracker cannot bid"),
            ):
                shadow.before_bid(info)
            prices = ValuationBot.expected_gem_values(info)
            for gem, row in zip(STANDARD_GEMS, shadow.rows):
                pmf = prior_final_pmf(info, gem)
                self.assertAlmostEqual(sum(pmf), 1)
                self.assertAlmostEqual(
                    sum(chart.get_value(n) * p for n, p in enumerate(pmf)), prices[gem]
                )
                for actual, expected in zip(row["tracker_pmf"], pmf):
                    self.assertAlmostEqual(actual, expected)
                self.assertNotIn("actual_count", row)
            with self.assertRaises(ValueError):
                shadow.before_bid(info)
            with self.assertRaises(ValueError):
                shadow.finalize(info)

    def test_forecast_is_frozen_before_current_bids_are_learned(self):
        info = view()
        shadow = ShadowForecasts()
        shadow.before_bid(info)
        recorded = copy.deepcopy(shadow.rows)
        after, result = after_round(info, {0: 1, 1: 25, 2: 1, 3: 1})
        shadow.after_round(after, result)
        self.assertEqual(shadow.rows, recorded)
        self.assertTrue(shadow.tracker.likelihoods)
        next_round = replace(after, round_number=info.round_number + 1)
        shadow.before_bid(next_round)
        purple = next(
            r for r in shadow.rows if r["round"] == 2 and r["color"] == PURPLE.name
        )
        # Several opponents' bids contribute; the aggregate need not rise
        # merely because one rival bid heavily. It must change only afterward.
        self.assertNotAlmostEqual(purple["tracker_mean"], purple["prior_mean"])
        self.assertTrue(all("actual_count" not in r for r in shadow.rows))

    def test_known_counts_and_perfect_forecasts_have_zero_loss(self):
        info = end_view()
        for gem in STANDARD_GEMS:
            pmf = prior_final_pmf(info, gem)
            mean, variance = moments(pmf)
            self.assertEqual(variance, 0)
            self.assertEqual(losses(pmf, int(mean)), dict.fromkeys(METRICS, 0.0))
        with self.assertRaises(ValueError):
            moments((0.5,) * 7)

    def test_shadow_observer_cannot_change_bids_or_see_other_hands(self):
        def run(observe):
            shadow = ShadowForecasts()
            with create_game(
                3,
                bots=[ValuationBot] * 3,
                value_chart=VALUE_CHARTS[1],
                rng=Random(840),
                bot_seed=840,
                bot_timeout=2.0,
            ) as game:
                observer = 1

                def hook(info):
                    self.assertEqual(info.self_player.id, observer)
                    self.assertTrue(
                        all(not hasattr(p, "hidden_gems") for p in info.other_players)
                    )
                    self.assertFalse(hasattr(info, "gem_deck"))
                    shadow.before_bid(info)
                    # Even a non-None observer return cannot become an actual bid.
                    return 1000000

                rounds = []
                while game.auction_gems:
                    result = play_round(
                        game, before_bid=hook if observe else None, observer_id=observer
                    )
                    rounds.append(result)
                    if observe:
                        shadow.after_round(
                            build_bot_info(game, game.players[observer]), result
                        )
                        self.assertTrue(
                            all("actual_count" not in r for r in shadow.rows)
                        )
                result = finish_game(game)
                if observe:
                    rows = shadow.finalize(build_bot_info(game, game.players[observer]))
                    self.assertEqual(len(rows), len(rounds) * 5)
                    for row in rows:
                        gem = next(g for g in STANDARD_GEMS if g.name == row["color"])
                        self.assertEqual(row["actual_count"], game.value_display[gem])
                    self.assertFalse(shadow.tracker.use_bid_model)
                    with self.assertRaises(ValueError):
                        shadow.finalize(build_bot_info(game, game.players[observer]))
                return rounds, result.scores

        with self.assertNoLogs("bot_runner", level="WARNING"):
            self.assertEqual(run(False), run(True))

    def test_invalid_observer_does_not_advance_the_game(self):
        with create_game(3, rng=Random(3)) as game:
            auctions = len(game.auction_deck)
            random_state = game.rng.getstate()
            with self.assertRaises(ValueError):
                play_round(game, before_bid=lambda info: None, observer_id=3)
            self.assertEqual(game.round_number, 0)
            self.assertEqual(len(game.auction_deck), auctions)
            self.assertEqual(game.rng.getstate(), random_state)

    def test_scoring_clusters_correlated_profiles_by_deal_seed(self):
        def game(seed, gain):
            return {
                "seed": seed,
                "rows": 100,
                "informative_rows": 100,
                "known_rows": 0,
                "metrics": {
                    m: {"prior": 1.0, "tracker": 1.0 - gain, "gain": gain}
                    for m in METRICS
                },
            }

        games = [game(1, 0.2), game(1, -0.2), game(2, 0.1), game(2, 0.3)]
        report = aggregate(games)
        self.assertEqual(report["independent_seed_clusters"], 2)
        self.assertEqual(report["games"], 4)
        self.assertAlmostEqual(report["mse"]["gain"], 0.1)
        self.assertAlmostEqual(report["mse"]["gain_ci_low"], -0.096)
        duplicate = aggregate(games + games)
        self.assertEqual(duplicate["independent_seed_clusters"], 2)
        self.assertAlmostEqual(
            duplicate["mse"]["gain_ci_low"], report["mse"]["gain_ci_low"]
        )

    def test_gate_requires_clear_accuracy_gain_and_no_profile_harm(self):
        good = {
            m: {"gain": 0.1, "gain_ci_low": 0.05, "gain_ci_high": 0.15} for m in METRICS
        }
        self.assertTrue(prediction_gate(good, {"valuation": good}))
        uncertain = copy.deepcopy(good)
        uncertain["mse"]["gain_ci_low"] = -0.01
        self.assertFalse(prediction_gate(uncertain, {"valuation": good}))
        uncertain = copy.deepcopy(good)
        uncertain["brier"]["gain_ci_low"] = 0
        self.assertFalse(prediction_gate(uncertain, {"valuation": good}))
        bad = copy.deepcopy(good)
        bad["mse"]["gain_ci_high"] = -0.01
        self.assertFalse(prediction_gate(good, {"fixed": bad}))
        # Diagnostics must not silently re-enable the deployed policy.
        self.assertFalse(SoftTrackingBot().use_bid_evidence)


if __name__ == "__main__":
    unittest.main()
