"""Hidden-world conservation, no private leakage, real-engine parity and search."""

import pickle
import signal
import unittest
from collections import Counter
from dataclasses import replace
from math import log
from random import Random
from time import perf_counter, sleep
from unittest.mock import patch

from bot_definitions.sampled_rollout import (
    RolloutDeadline,
    SampledWorld,
    finish_rollout,
    historical_rates,
    interruptible_deadline,
    public_players,
    reframe,
    root_bids,
    sample_world,
    world_game,
)
from bot_definitions.sampling_bot import SamplingBot, mean_scores
from bot_definitions.soft_tracking_bot import SoftTrackingBot, infer_locations
from bot_definitions.valuation_bot import ValuationBot
from bot_info import build_bot_info
from constants import STANDARD_GEMS, AuctionType
from decks import GEM_CARDS, VALUE_CHARTS
from experiments.weighted_sampling_reference import (
    WeightedSamplingReference,
    weighted_means,
    world_log_weight,
)
from harness import create_game, finish_game, play_round, resolve_auction
from immutable import FrozenMap
from simulation import bot_catalog
from tests.test_soft_tracking_bot import after_round
from tests.test_valuation_bot import BLUE, view


class SamplingTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(signal, "setitimer"), "Unix worker alarm")
    def test_slow_rollouts_are_interrupted_and_return_level_two(self):
        info = view()
        expected = ValuationBot().get_bid(info)
        for completed in (0, 1):
            with self.subTest(completed_worlds=completed):
                bot = SamplingBot(decision_seconds=0.2, max_worlds=2, max_candidates=2)
                calls = 0

                def slow_continuation(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    # Pilot + one other candidate complete the first world.
                    if completed and calls <= 2:
                        return (10, 0, 0, 0)
                    sleep(10)
                    raise AssertionError("Search did not interrupt a slow rollout")

                started = perf_counter()
                with patch(
                    "bot_definitions.sampling_bot.finish_rollout",
                    side_effect=slow_continuation,
                ) as rollout:
                    self.assertEqual(bot.get_bid(info), expected)
                self.assertTrue(rollout.called)
                self.assertLess(perf_counter() - started, 2.0)
                self.assertEqual(bot.last_search["worlds"], completed)
                self.assertTrue(bot.last_search["baseline_ready"])
                self.assertIsNotNone(bot.last_search["fallback"])
                self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    @unittest.skipUnless(hasattr(signal, "setitimer"), "Unix worker alarm")
    def test_deadline_interrupts_noncooperative_work_and_restores_alarm(self):
        previous = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(RolloutDeadline):
            with interruptible_deadline(0.01) as enabled:
                self.assertTrue(enabled)
                sleep(0.1)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        bot = SamplingBot(decision_seconds=0.02)
        with patch.object(bot.baseline, "get_bid", side_effect=lambda _: sleep(0.1)):
            self.assertEqual(bot.get_bid(view()), 0)
        self.assertEqual(bot.last_search["fallback"], "baseline_deadline_zero_bid")
        self.assertLess(bot.last_search["seconds"], 0.2)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)

    def test_importance_samples_recover_the_tracker_weighted_marginal(self):
        info = view()
        logs = {1: {BLUE: [min(3.0, 0.7 * n) for n in range(7)]}}
        exact = infer_locations(info, logs, evidence_weight=0.4).expected_counts[1][
            BLUE
        ]
        rng = Random(17)
        worlds = [sample_world(info, rng) for _ in range(5000)]
        counts = [[world.hands[1].count(BLUE)] for world in worlds]
        actual, _ = weighted_means(
            [world_log_weight(info, world, logs, 0.4) for world in worlds], counts
        )
        self.assertAlmostEqual(actual[0], exact, delta=0.05)

    def test_worlds_preserve_every_public_fact_and_every_location_capacity(self):
        info = view()
        rng = Random(7)
        for _ in range(100):
            world = sample_world(info, rng)
            self.assertEqual(
                world.hands[info.self_player.id], info.self_player.hidden_gems
            )
            self.assertEqual(len(world.gem_deck), info.gem_deck_count)
            self.assertEqual(
                Counter(world.auction_deck), Counter(info.auction_cards_remaining)
            )
            unseen = Counter(world.gem_deck)
            for player in info.other_players:
                self.assertEqual(len(world.hands[player.id]), player.hidden_gem_count)
                unseen.update(world.hands[player.id])
            self.assertEqual(unseen, Counter(info.unknown_gems))
            all_cards = Counter(info.value_display)
            for hand in world.hands:
                all_cards.update(hand)
            all_cards.update(world.gem_deck)
            all_cards.update(info.auction_gems)
            for player in public_players(info):
                all_cards.update(player.won_gems)
            self.assertEqual(all_cards, Counter(GEM_CARDS))

    def test_uniform_prior_and_private_weighted_control(self):
        info = view()
        logs = {
            p.id: {g: [n / 3 for n in range(7)] for g in STANDARD_GEMS}
            for p in info.other_players
        }
        world = sample_world(info, Random(3))
        self.assertFalse(hasattr(world, "log_weight"))
        self.assertAlmostEqual(
            world_log_weight(info, world, logs, 0.1),
            0.1 * sum(p.hidden_gem_count for p in info.other_players) / 3,
        )
        self.assertEqual(world_log_weight(info, world, logs, 0), 0)
        inconsistent = replace(
            info, unknown_gems=FrozenMap({g: 5 for g in STANDARD_GEMS})
        )
        with self.assertRaises(ValueError):
            sample_world(inconsistent, Random(1))
        exact = infer_locations(info, {}, evidence_weight=0).expected_counts[1][BLUE]
        rng = Random(71)
        average = (
            sum(sample_world(info, rng).hands[1].count(BLUE) for _ in range(4000))
            / 4000
        )
        self.assertAlmostEqual(average, exact, delta=0.05)

    def test_reframed_views_expose_only_each_policy_own_private_hand(self):
        info = view()
        world = sample_world(info, Random(4))
        for player_id in range(info.num_players):
            frame = reframe(info, player_id, world.hands[player_id])
            self.assertEqual(frame.self_player.hidden_gems, world.hands[player_id])
            self.assertTrue(
                all(not hasattr(p, "hidden_gems") for p in frame.other_players)
            )
            actual_unknown = Counter(world.gem_deck)
            for other_id, hand in enumerate(world.hands):
                if other_id != player_id:
                    actual_unknown.update(hand)
            self.assertEqual(Counter(frame.unknown_gems), actual_unknown)

    def test_no_nested_workers_and_clones_do_not_mutate_root_or_other_actions(self):
        info = view()
        world = sample_world(info, Random(4))
        with patch(
            "bot_runner.BotRunner.__init__", side_effect=AssertionError("nested worker")
        ):
            root = world_game(info, world, [{}] * 4, float("inf"))
            bids = root_bids(root)
            before = build_bot_info(root, root.players[0])
            original_rates = [dict(p.policy.bid_rates) for p in root.players]
            score1 = finish_rollout(root, bids, 0, 0, float("inf"))
            score2 = finish_rollout(root, bids, 0, 0, float("inf"))
            self.assertEqual(score1, score2)
            self.assertEqual(build_bot_info(root, root.players[0]), before)
            self.assertEqual([p.policy.bid_rates for p in root.players], original_rates)

    def test_full_continuations_match_isolated_real_harness(self):
        with self.assertNoLogs("bot_runner", level="WARNING"):
            for seats in (3, 4, 5):
                for chart in VALUE_CHARTS:
                    expected = []
                    with create_game(
                        seats,
                        bots=[ValuationBot] * seats,
                        rng=Random(3300),
                        value_chart=chart,
                        bot_timeout=2.0,
                    ) as game:

                        def capture(info):
                            world = SampledWorld(
                                tuple(tuple(p.hidden_gem_hand) for p in game.players),
                                tuple(game.gem_deck._cards),
                                tuple(game.auction_deck._cards),
                                0,
                            )
                            root = world_game(info, world, [{}] * seats, float("inf"))
                            root.rng.setstate(game.rng.getstate())
                            bids = root_bids(root)
                            expected.append(
                                finish_rollout(root, bids, 0, bids[0], float("inf"))
                            )

                        play_round(game, before_bid=capture)
                        while game.auction_gems:
                            play_round(game)
                        self.assertEqual(finish_game(game).scores, expected[0])

    def test_history_replay_reconstructs_each_world_level_two_learning(self):
        info = view()
        world = sample_world(info, Random(5))
        game = world_game(info, world, [{}] * 4, float("inf"))
        history = []
        for _ in range(5):
            before = build_bot_info(game, game.players[0])
            bids = root_bids(game)
            result = resolve_auction(game, bids)
            history.append((before, result))
            if not game.auction_gems:
                break
            # Draw the next auction exactly as play_round does, before bidding.
            game.current_auction = game.auction_deck.draw()
            game.auction_history.append(game.current_auction)
            game.round_number += 1
            current = build_bot_info(game, game.players[0])
            truth = SampledWorld(
                tuple(tuple(p.hidden_gem_hand) for p in game.players),
                tuple(game.gem_deck._cards),
                tuple(game.auction_deck._cards),
                0,
            )
            for player in game.players:
                reconstructed = historical_rates(current, truth, player.id, history)
                self.assertEqual(reconstructed, player.policy.bid_rates)

    def test_weights_average_final_coins_not_win_counts(self):
        means, effective = weighted_means([0, log(3)], [[100, 0], [0, 50]])
        self.assertAlmostEqual(means[0], 25)
        self.assertAlmostEqual(means[1], 37.5)
        self.assertAlmostEqual(effective, 1.6)
        scaled, _ = weighted_means([1000, 1000 + log(3)], [[100, 0], [0, 50]])
        self.assertAlmostEqual(scaled[1], 37.5)
        with self.assertRaises(ValueError):
            weighted_means([], [])

    def test_search_finishes_worlds_respects_legal_cash_and_has_a_deadline(self):
        bot = SamplingBot(decision_seconds=0.45, max_worlds=4, max_candidates=5)
        started = perf_counter()
        bid = bot.get_bid(view())
        self.assertTrue(0 <= bid <= view().self_player.coins)
        self.assertLess(perf_counter() - started, 2)
        self.assertGreaterEqual(bot.last_search["worlds"], 2)
        self.assertIsNone(bot.last_search["fallback"])
        self.assertEqual(
            bot.last_search["rollouts"],
            bot.last_search["worlds"] * len(bot.last_search["candidates"]),
        )
        no_time = SamplingBot(decision_seconds=0)
        self.assertEqual(no_time.get_bid(view()), ValuationBot().get_bid(view()))
        self.assertEqual(no_time.last_search["fallback"], "no_search_time")
        # Partial candidate batches must never leave unequal sample sets.
        partial = SamplingBot(decision_seconds=0.5, max_worlds=2)
        with (
            patch(
                "bot_definitions.sampling_bot.finish_rollout",
                side_effect=[(10, 0, 0, 0), (9, 0, 0, 0), RolloutDeadline()],
            ),
            patch(
                "bot_definitions.sampling_bot.mean_scores",
                side_effect=AssertionError("incomplete world was used"),
            ),
        ):
            self.assertEqual(partial.get_bid(view()), ValuationBot().get_bid(view()))
        self.assertEqual(partial.last_search["worlds"], 0)
        self.assertEqual(partial.last_search["abandoned_worlds"], 1)

    def test_terminal_search_uses_exact_final_coin_score_and_tie_priority(self):
        info = view()
        world = sample_world(info, Random(1))
        game = world_game(info, world, [{}] * 4, float("inf"))
        while True:
            if (
                game.current_auction.type is AuctionType.AUCTION
                and len(game.gem_deck) + len(game.auction_gems)
                <= game.current_auction.value
            ):
                break
            bids = root_bids(game)
            resolve_auction(game, bids)
            game.current_auction = game.auction_deck.draw()
            game.auction_history.append(game.current_auction)
            game.round_number += 1
        root = build_bot_info(game, game.players[0])
        # Force an affordable one-coin winning bid against bankrupt rivals.
        root = replace(
            root,
            self_player=replace(root.self_player, coins=3),
            other_players=tuple(replace(p, coins=0) for p in root.other_players),
            previous_winner_id=0,
        )
        color = root.auction_gems[0]
        price = ValuationBot.expected_gem_values(root)[color]
        self.assertGreater(price, 1)
        bot = SamplingBot(decision_seconds=1, max_worlds=4)
        self.assertEqual(bot.get_bid(root), 1)
        self.assertEqual(bot.last_search["worlds"], 4)

    def test_uniform_state_and_history_survive_restart_without_a_tracker(self):
        bot = SamplingBot(decision_seconds=0, max_worlds=4)
        self.assertNotIsInstance(bot, SoftTrackingBot)
        info = view()
        with patch(
            "bot_definitions.soft_tracking_bot.infer_locations",
            side_effect=AssertionError("L4 invoked hidden-hand inference"),
        ):
            bot.get_bid(info)
            after, result = after_round(info, {0: 0, 1: 9, 2: 3, 3: 0})
            bot.on_round_end(after, result)
        for name in ("tracker", "likelihoods", "evidence_weight"):
            self.assertNotIn(name, vars(bot))
        self.assertEqual(bot.choose_hidden_gem(info, result.bids), 0)
        clone = SamplingBot.__new__(SamplingBot)
        clone.load_state(pickle.loads(pickle.dumps(bot.save_state())))
        self.assertEqual(clone.history, bot.history)
        self.assertEqual(clone.baseline.bid_rates, bot.baseline.bid_rates)
        clone.on_game_start()
        self.assertEqual(clone.history, [])
        self.assertEqual(clone.search_log, [])
        self.assertEqual(clone.max_worlds, 4)
        self.assertIn("SamplingBot", bot_catalog())
        self.assertNotIn("WeightedSamplingReference", bot_catalog())

    def test_uniform_means_optimize_own_score_and_require_equal_batches(self):
        self.assertEqual(mean_scores([[100, 0], [0, 60]]), [50, 30])
        for invalid in ([], [[]], [[1, 2], [3]]):
            with self.assertRaises(ValueError):
                mean_scores(invalid)

    def test_simplified_uniform_matches_zero_weight_control_on_fixed_worlds(self):
        info = view()
        world = sample_world(info, Random(5))
        game = world_game(info, world, [{}] * 4, float("inf"))
        kwargs = dict(decision_seconds=1.35, max_worlds=2, max_candidates=2)
        uniform = SamplingBot(**kwargs)
        reference = WeightedSamplingReference(**kwargs, evidence_weight=0)
        for round_index in range(8):
            before = build_bot_info(game, game.players[0])
            with patch(
                "bot_definitions.sampling_bot.getrandbits",
                return_value=71 + round_index,
            ):
                expected = reference.get_bid(before)
                actual = uniform.get_bid(before)
            self.assertEqual(actual, expected)
            self.assertIsNone(uniform.last_search["fallback"])
            self.assertIsNone(reference.last_search["fallback"])
            self.assertEqual(
                uniform.last_search["worlds"], reference.last_search["worlds"]
            )
            if uniform.last_search["worlds"]:
                self.assertEqual(uniform.last_search["worlds"], 2)
                self.assertEqual(
                    uniform.last_search["mean_final_scores"],
                    reference.last_search["mean_final_scores"],
                )
            bids = root_bids(game)
            bids[0] = actual
            result = resolve_auction(game, bids)
            after = build_bot_info(game, game.players[0])
            uniform.on_round_end(after, result)
            reference.on_round_end(after, result)
            self.assertEqual(uniform.history, reference.history)
            self.assertEqual(uniform.baseline.bid_rates, reference.baseline.bid_rates)
            if not game.auction_gems:
                break
            game.current_auction = game.auction_deck.draw()
            game.auction_history.append(game.current_auction)
            game.round_number += 1


if __name__ == "__main__":
    unittest.main()
