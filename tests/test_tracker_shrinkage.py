"""Evidence tempering, prior recovery, cache/checkpoint safety, and paired tuning."""

import math
import pickle
import unittest
from dataclasses import replace
from itertools import permutations

from bot_definitions.soft_tracking_bot import SoftTrackingBot, infer_locations
from bot_definitions.valuation_bot import FractionalValuationBot as ValuationBot
from constants import STANDARD_GEMS
from experiments.evaluate_tracker import METRICS, ShadowForecasts
from experiments.tune_tracker import compare_weights
from immutable import FrozenMap
from tests.test_soft_tracking_bot import after_round
from tests.test_valuation_bot import BLUE, GREEN, PURPLE, view


class ShrinkageTests(unittest.TestCase):
    def test_tempered_posterior_matches_enumerated_worlds(self):
        info = view()
        info = replace(
            info,
            self_player=replace(info.self_player, hidden_gems=()),
            other_players=tuple(
                replace(p, hidden_gem_count=int(p.id < 3)) for p in info.other_players
            ),
            unknown_gems=FrozenMap(
                {g: 2 if g in (BLUE, GREEN) else 0 for g in STANDARD_GEMS}
            ),
            gem_deck_count=2,
        )
        colors = (BLUE, BLUE, GREEN, GREEN)
        raw = {1: {BLUE: [0, 2, 0, 0, 0, 0, 0]}, 2: {GREEN: [0, -1, 0, 0, 0, 0, 0]}}
        for weight in (0, 0.1, 0.4, 1):
            partition = 0.0
            hand_blue = 0.0
            for one, two in permutations(range(4), 2):
                mass = math.exp(
                    weight * (2 * int(colors[one] is BLUE) - int(colors[two] is GREEN))
                )
                partition += mass
                hand_blue += mass * int(colors[one] is BLUE)
            actual = infer_locations(info, raw, evidence_weight=weight)
            self.assertAlmostEqual(
                actual.expected_counts[1][BLUE], hand_blue / partition
            )
            self.assertAlmostEqual(sum(actual.expected_counts[-1].values()), 2)

    def test_zero_weight_preserves_prior_and_level_two_bids(self):
        bot = SoftTrackingBot(
            use_bid_evidence=True,
            evidence_weight=0,
            use_bid_model=False,
            use_cash_limits=False,
            use_endgame_rules=False,
        )
        bot.likelihoods = {1: {PURPLE: [-3, -2, -1, 0, 1, 2, 3]}}
        for chart in ("A", "B", "E"):
            info = view()
            from decks import VALUE_CHARTS

            info = replace(
                info, value_chart=next(c for c in VALUE_CHARTS if c.name == chart)
            )
            self.assertEqual(
                bot.gem_values(info), ValuationBot.expected_gem_values(info)
            )
            self.assertEqual(bot.get_bid(info), ValuationBot().get_bid(info))

    def test_legacy_strength_matches_previous_effective_likelihood(self):
        info = view()
        previous = {1: {PURPLE: [-1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2]}}
        raw = {
            p: {g: [n / 0.4 for n in table] for g, table in colors.items()}
            for p, colors in previous.items()
        }
        old = infer_locations(info, previous)
        legacy = infer_locations(info, raw, evidence_weight=0.4)
        for loc in old.expected_counts:
            for gem in STANDARD_GEMS:
                self.assertAlmostEqual(
                    old.expected_counts[loc][gem], legacy.expected_counts[loc][gem]
                )
                self.assertAlmostEqual(old.gem_values[gem], legacy.gem_values[gem])

    def test_smaller_weight_shrinks_evidence_and_invalidates_cache(self):
        info = view()
        bot = SoftTrackingBot(use_bid_evidence=True, evidence_weight=0.4)
        bot.likelihoods = {1: {BLUE: [float(n) for n in range(7)]}}
        high = bot.beliefs(info)
        bot.evidence_weight = 0.1
        small = bot.beliefs(info)
        prior = infer_locations(info, {})
        self.assertIsNot(high, small)
        self.assertLess(small.expected_counts[1][BLUE], high.expected_counts[1][BLUE])
        self.assertGreater(
            small.expected_counts[1][BLUE], prior.expected_counts[1][BLUE]
        )
        restored = SoftTrackingBot.__new__(SoftTrackingBot)
        restored.load_state(pickle.loads(pickle.dumps(bot.save_state())))
        self.assertEqual(restored.evidence_weight, 0.1)
        restored.on_game_start()
        self.assertEqual(restored.evidence_weight, 0.1)
        bot.evidence_weight = 0
        self.assertEqual(bot.gem_values(info), ValuationBot.expected_gem_values(info))

    def test_weights_must_be_finite_probabilistic_shrinkage_values(self):
        for weight in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    SoftTrackingBot(evidence_weight=weight)
                with self.assertRaises(ValueError):
                    infer_locations(view(), {}, evidence_weight=weight)

    def test_parallel_shadows_do_not_share_or_retroactively_change_evidence(self):
        info = view()
        after, result = after_round(info, {0: 1, 1: 25, 2: 1, 3: 1})
        next_info = replace(after, round_number=2)

        def run(weights):
            shadows = {w: ShadowForecasts(evidence_weight=w) for w in weights}
            for shadow in shadows.values():
                shadow.before_bid(info)
            initial = {
                w: tuple(tuple(r["tracker_pmf"]) for r in s.rows)
                for w, s in shadows.items()
            }
            for shadow in shadows.values():
                shadow.after_round(after, result)
                shadow.before_bid(next_info)
            for w, shadow in shadows.items():
                self.assertEqual(
                    tuple(tuple(r["tracker_pmf"]) for r in shadow.rows[:5]), initial[w]
                )
            return {w: shadows[w].rows for w in weights}

        combined = run([0, 0.1, 0.4])
        self.assertEqual(combined[0.1], run([0.1])[0.1])
        self.assertEqual(combined, run([0.4, 0.1, 0]))
        self.assertTrue(all(r["tracker_pmf"] == r["prior_pmf"] for r in combined[0]))

    def test_pairwise_weight_comparison_uses_matched_seed_losses(self):
        def game(seed, loss):
            return {
                "seed": seed,
                "profile": "valuation",
                "seats": 3,
                "rows": 10,
                "informative_rows": 10,
                "known_rows": 0,
                "metrics": {m: {"tracker": loss} for m in METRICS},
            }

        candidate = [game(1, 0.5), game(2, 0.6)]
        reference = [game(1, 0.7), game(2, 0.8)]
        comparison = compare_weights(candidate, reference)
        self.assertAlmostEqual(comparison["mse"]["gain"], 0.2)
        self.assertAlmostEqual(comparison["mse"]["gain_ci_low"], 0.2)
        self.assertAlmostEqual(comparison["mse"]["candidate"], 0.55)
        with self.assertRaises(AssertionError):
            compare_weights(candidate, list(reversed(reference)))


if __name__ == "__main__":
    unittest.main()
