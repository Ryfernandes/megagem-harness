"""Constrained inference, behavioral updates, bidding, and worker integration."""

import pickle
import unittest
from collections import Counter
from dataclasses import replace
from itertools import permutations, product
from math import exp
from time import perf_counter

from bot_definitions.soft_tracking_bot import DECK, SoftTrackingBot, infer_locations
from bot_definitions.valuation_bot import ValuationBot
from bots import GameSummary, RoundResult
from constants import STANDARD_GEMS, AuctionType
from decks import MISSION_CARDS, VALUE_CHARTS
from immutable import FrozenMap
from lib import AuctionCard, MissionCard
from simulation import simulate
from tests.test_valuation_bot import BLUE, GREEN, PINK, PURPLE, view


def after_round(info, bids, winner=1, reveal=GREEN):
    """Resolve public award/reveal facts without depending on any hidden deal."""
    players = []
    lot = info.auction_gems[: info.current_auction.value]
    for p in info.other_players:
        if p.id == winner:
            collection = Counter(p.won_gems)
            collection.update(lot)
            p = replace(
                p,
                coins=p.coins - bids[p.id],
                hidden_gem_count=p.hidden_gem_count - int(reveal is not None),
                revealed_gems=p.revealed_gems + ((reveal,) if reveal else ()),
                won_gems=FrozenMap(collection),
            )
        players.append(p)
    unknown = Counter(info.unknown_gems)
    display = Counter(info.value_display)
    if reveal is not None:
        unknown[reveal] -= 1
        display[reveal] += 1
    after = replace(
        info,
        other_players=tuple(players),
        unknown_gems=FrozenMap(unknown),
        value_display=FrozenMap(display),
        auction_gems=info.auction_gems[len(lot) :],
        gems_remaining=info.gems_remaining - len(lot),
        previous_winner_id=winner,
    )
    result = RoundResult(
        info.current_auction, FrozenMap(bids), winner, bids[winner], reveal
    )
    return after, result


def end_view():
    info = view()
    return replace(
        info,
        auction_gems=(PURPLE,),
        gem_deck_count=0,
        gems_remaining=1,
        unknown_gems=FrozenMap(dict(zip(STANDARD_GEMS, (3, 3, 2, 2, 2)))),
    )


class InferenceTests(unittest.TestCase):
    def test_uniform_prior_matches_hypergeometric_and_location_capacities(self):
        for chart in VALUE_CHARTS:
            info = replace(view(), value_chart=chart)
            belief = infer_locations(info, {})
            for gem, expected in ValuationBot.expected_gem_values(info).items():
                self.assertAlmostEqual(belief.gem_values[gem], expected)
                self.assertAlmostEqual(sum(belief.hidden_count_probabilities[gem]), 1)
                self.assertAlmostEqual(
                    sum(p[gem] for p in belief.location_probabilities.values()), 1
                )
                self.assertAlmostEqual(
                    sum(p[gem] for p in belief.expected_counts.values()),
                    info.unknown_gems[gem],
                )
                in_hand = belief.opponent_hand_probabilities[gem]
                self.assertAlmostEqual(
                    in_hand + belief.location_probabilities[DECK][gem], 1
                )
                expected_hidden = sum(
                    n * probability
                    for n, probability in enumerate(
                        belief.hidden_count_probabilities[gem]
                    )
                )
                self.assertAlmostEqual(
                    expected_hidden, info.unknown_gems[gem] * in_hand
                )
            for p in info.other_players:
                self.assertAlmostEqual(
                    sum(belief.expected_counts[p.id].values()), p.hidden_gem_count
                )
            self.assertAlmostEqual(
                sum(belief.expected_counts[DECK].values()), info.gem_deck_count
            )

    def test_weighted_marginals_equal_brute_force_over_labeled_gems(self):
        info = replace(
            view(),
            self_player=replace(view().self_player, hidden_gems=(BLUE,)),
            other_players=tuple(
                replace(p, hidden_gem_count=int(p.id < 3)) for p in view().other_players
            ),
            gem_deck_count=2,
            value_display=FrozenMap({BLUE: 1}),
            unknown_gems=FrozenMap(
                {
                    g: (2 if g is BLUE else int(g in (GREEN, PURPLE)))
                    for g in STANDARD_GEMS
                }
            ),
        )
        logs = {
            1: {BLUE: [0, 0.8, 0, 0, 0, 0, 0]},
            2: {GREEN: [0.2, -0.5, 0, 0, 0, 0, 0]},
        }
        colors = (BLUE, BLUE, GREEN, PURPLE)
        for chart in VALUE_CHARTS:
            current = replace(info, value_chart=chart)
            belief = infer_locations(current, logs)
            totals = {loc: dict.fromkeys(STANDARD_GEMS, 0.0) for loc in (DECK, 1, 2, 3)}
            values = dict.fromkeys(STANDARD_GEMS, 0.0)
            partition = 0.0
            for one, two in permutations(range(4), 2):
                weight = exp(
                    0.8 * (colors[one] is BLUE)
                    + (-0.5 if colors[two] is GREEN else 0.2)
                )
                partition += weight
                for gem in STANDARD_GEMS:
                    counts = {
                        1: int(colors[one] is gem),
                        2: int(colors[two] is gem),
                        3: 0,
                    }
                    counts[DECK] = colors.count(gem) - counts[1] - counts[2]
                    for loc in totals:
                        totals[loc][gem] += weight * counts[loc]
                    values[gem] += weight * chart.get_value(
                        (2 if gem is BLUE else 0) + counts[1] + counts[2]
                    )
            for loc in totals:
                for gem in STANDARD_GEMS:
                    self.assertAlmostEqual(
                        belief.expected_counts[loc][gem], totals[loc][gem] / partition
                    )
                    self.assertAlmostEqual(
                        belief.gem_values[gem], values[gem] / partition
                    )

    def test_no_deck_means_behavior_cannot_change_terminal_prices(self):
        info = end_view()
        baseline = infer_locations(info, {})
        logs = {1: {PURPLE: [-1, -0.4, 0.2, 0.6, 1, 1, 1]}}
        belief = infer_locations(info, logs)
        for gem in STANDARD_GEMS:
            self.assertAlmostEqual(belief.gem_values[gem], baseline.gem_values[gem])
            self.assertEqual(belief.expected_counts[DECK][gem], 0)
        empty = replace(
            info,
            gem_deck_count=0,
            unknown_gems=FrozenMap(dict.fromkeys(STANDARD_GEMS, 0)),
            other_players=tuple(
                replace(p, hidden_gem_count=0) for p in info.other_players
            ),
        )
        exact = infer_locations(empty, {})
        self.assertEqual(exact.gem_values, ValuationBot.expected_gem_values(empty))

    def test_auction_award_and_own_reveal_do_not_add_display_supply(self):
        logs = {1: {PURPLE: [-1, -0.4, 0.2, 0.6, 1, 1, 1]}}
        for chart in VALUE_CHARTS:
            info = replace(view(), value_chart=chart)
            before = infer_locations(info, logs)
            collection = Counter(info.self_player.won_gems)
            collection[PURPLE] += 1
            awarded = replace(
                info,
                self_player=replace(info.self_player, won_gems=FrozenMap(collection)),
                auction_gems=info.auction_gems[1:],
                gems_remaining=info.gems_remaining - 1,
            )
            self.assertEqual(
                infer_locations(awarded, logs).gem_values, before.gem_values
            )
            display = Counter(info.value_display)
            revealed = awarded.self_player.hidden_gems[0]
            display[revealed] += 1
            revealed_info = replace(
                awarded,
                self_player=replace(
                    awarded.self_player,
                    hidden_gems=awarded.self_player.hidden_gems[1:],
                    hidden_gem_count=awarded.self_player.hidden_gem_count - 1,
                ),
                value_display=FrozenMap(display),
            )
            self.assertEqual(
                infer_locations(revealed_info, logs).gem_values, before.gem_values
            )

    def test_exposed_deck_card_conditions_hand_probability_not_display_count(self):
        # One previously visible yellow gem is now owned. Filling the row with
        # green establishes that one green copy was in the deck, not a hand.
        info = view()
        yellow = STANDARD_GEMS[3]
        info = replace(
            info,
            self_player=replace(info.self_player, won_gems=FrozenMap({yellow: 1})),
            auction_gems=(PURPLE,),
            gems_remaining=13,
        )
        for chart in VALUE_CHARTS:
            for logs in ({}, {1: {GREEN: [-1, -0.4, 0.2, 0.6, 1, 1, 1]}}):
                current = replace(info, value_chart=chart)
                before = infer_locations(current, logs)
                unseen = current.unknown_gems[GREEN]
                # Condition on a uniformly exposed deck card being green:
                # likelihood of a world is proportional to its deck greens.
                conditional = [
                    probability * (unseen - hands)
                    for hands, probability in enumerate(
                        before.hidden_count_probabilities[GREEN]
                    )
                ]
                normalizer = sum(conditional)
                known = current.value_display.get(GREEN, 0)
                known += current.self_player.hidden_gems.count(GREEN)
                expected_price = sum(
                    mass / normalizer * chart.get_value(known + hands)
                    for hands, mass in enumerate(conditional)
                )
                unknown = dict(current.unknown_gems)
                unknown[GREEN] -= 1
                exposed = replace(
                    current,
                    unknown_gems=FrozenMap(unknown),
                    gem_deck_count=current.gem_deck_count - 1,
                    auction_gems=(*current.auction_gems, GREEN),
                )
                after = infer_locations(exposed, logs)
                self.assertAlmostEqual(after.gem_values[GREEN], expected_price)
                self.assertEqual(exposed.value_display, current.value_display)
                self.assertAlmostEqual(
                    sum(after.expected_counts[DECK].values()), exposed.gem_deck_count
                )


class LearningTests(unittest.TestCase):
    def observe(self, info, bids, winner=1, *, use_bid_evidence=True):
        bot = SoftTrackingBot(
            use_bid_evidence=use_bid_evidence,
            use_bid_model=True,
            use_cash_limits=True,
            use_endgame_rules=False,
        )
        bot.get_bid(info)
        after, result = after_round(info, bids, winner)
        bot.on_round_end(after, result)
        return bot, after

    def test_heavy_bids_are_soft_and_chart_aware(self):
        for chart in VALUE_CHARTS:
            info = replace(view(), value_chart=chart)
            heavy, after = self.observe(info, {0: 1, 1: 25, 2: 1, 3: 1})
            neutral, _ = self.observe(info, {0: 1, 1: 5, 2: 1, 3: 1})
            self.assertGreaterEqual(
                heavy.beliefs(after).gem_values[PURPLE],
                neutral.beliefs(after).gem_values[PURPLE] - 1e-10,
            )
            if chart.name == "A":
                self.assertGreater(
                    heavy.beliefs(after).expected_counts[1][PURPLE],
                    neutral.beliefs(after).expected_counts[1][PURPLE],
                )
            if chart.name == "B":
                self.assertLess(
                    heavy.beliefs(after).expected_counts[1][PURPLE],
                    neutral.beliefs(after).expected_counts[1][PURPLE],
                )
            for location in heavy.beliefs(after).location_probabilities.values():
                self.assertTrue(all(0 < location[g] < 1 for g in STANDARD_GEMS))
            self.assertEqual(heavy.opponent_coins[1], 0)
            self.assertEqual(heavy.revealed_counts[GREEN], 1)
            self.assertEqual(heavy.rounds_observed, 1)

    def test_zero_cash_pass_is_not_private_hand_evidence(self):
        info = replace(
            view(),
            other_players=(
                replace(view().other_players[0], coins=0),
                *view().other_players[1:],
            ),
        )
        bot, _ = self.observe(info, {0: 0, 1: 0, 2: 1, 3: 0}, winner=2)
        self.assertEqual(bot.likelihoods[1][PURPLE], [0.0] * 7)
        self.assertNotIn(1, bot.bid_rates)

    def test_mission_pass_is_weak_evidence_and_does_not_change_public_collection(self):
        info = view()
        rival = replace(info.other_players[0], won_gems=FrozenMap({BLUE: 1}))
        info = replace(
            info,
            other_players=(rival, *info.other_players[1:]),
            available_missions=(MissionCard(5, {BLUE: 1, PURPLE: 1}),),
        )
        bot, after = self.observe(info, {0: 0, 1: 0, 2: 1, 3: 0}, winner=2)
        self.assertLess(bot.bid_rates[1], bot.bid_fraction)
        self.assertTrue(any(v != 0 for v in bot.likelihoods[1][PURPLE]))
        self.assertEqual(after.other_players[0].won_gems, rival.won_gems)
        self.assertTrue(all(abs(v) <= 1.2 for v in bot.likelihoods[1][PURPLE]))

    def test_decay_reveal_transport_and_checkpoint_reset(self):
        bot = SoftTrackingBot(use_bid_evidence=True)
        bot.likelihoods = {1: {PURPLE: [i / 10 for i in range(7)]}}
        info = view()
        after, result = after_round(info, {0: 0, 1: 0, 2: 0, 3: 0}, reveal=PURPLE)
        # Missing pre-bid memory must not discard known reveal facts.
        bot.on_round_end(after, result)
        self.assertEqual(
            bot.likelihoods[1][PURPLE], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6]
        )
        financial = replace(after, current_auction=AuctionCard(AuctionType.INVEST, 5))
        bot.get_bid(financial)
        result = RoundResult(
            financial.current_auction, FrozenMap(dict.fromkeys(range(4), 0)), 0, 0, None
        )
        bot.on_round_end(financial, result)
        self.assertAlmostEqual(bot.likelihoods[1][PURPLE][0], 0.096)
        restored = SoftTrackingBot.__new__(SoftTrackingBot)
        restored.load_state(pickle.loads(pickle.dumps(bot.save_state())))
        self.assertTrue(restored.use_bid_evidence)
        self.assertEqual(restored.get_bid(after), bot.get_bid(after))
        self.assertEqual(restored.likelihoods, bot.likelihoods)
        terminal = replace(
            after,
            value_display=FrozenMap(dict(zip(STANDARD_GEMS, (4, 3, 3, 3, 3)))),
        )
        summary = GameSummary(FrozenMap({0: 10, 1: 5, 2: 4, 3: 3}), (0,), 10)
        restored.on_game_end(terminal, summary)
        self.assertEqual(restored.revealed_counts, dict(terminal.value_display))
        self.assertIsNone(restored._pending)
        restored.on_game_start()
        self.assertTrue(restored.use_bid_evidence)
        self.assertEqual(restored.likelihoods, {})
        self.assertEqual(restored.opponent_coins, {})
        self.assertEqual(restored.rounds_observed, 0)

    def test_opt_out_ignores_bid_evidence_but_tracks_cash_and_bid_rates(self):
        self.assertTrue(SoftTrackingBot(use_bid_evidence=True).use_bid_evidence)
        self.assertFalse(SoftTrackingBot(use_bid_evidence=False).use_bid_evidence)
        for chart in VALUE_CHARTS:
            info = replace(view(), value_chart=chart)
            heavy, after = self.observe(
                info, {0: 1, 1: 25, 2: 1, 3: 1}, use_bid_evidence=False
            )
            neutral, neutral_after = self.observe(
                info, {0: 1, 1: 5, 2: 1, 3: 1}, use_bid_evidence=False
            )
            prior = ValuationBot.expected_gem_values(after)
            for gem in STANDARD_GEMS:
                self.assertAlmostEqual(heavy.gem_values(after)[gem], prior[gem])
                self.assertAlmostEqual(
                    neutral.gem_values(neutral_after)[gem], prior[gem]
                )
            self.assertEqual(heavy.likelihoods, {})
            self.assertEqual(neutral.likelihoods, {})
            self.assertGreater(heavy.bid_rates[1], neutral.bid_rates[1])
            self.assertEqual(heavy.opponent_coins[1], 0)
            self.assertEqual(heavy.revealed_counts[GREEN], 1)

    def test_switch_invalidates_cached_evidence_and_survives_restarts(self):
        info = view()
        bot = SoftTrackingBot(use_bid_evidence=True)
        bot.likelihoods = {1: {PURPLE: [-1, -0.4, 0.2, 0.6, 1, 1, 1]}}
        weighted = bot.gem_values(info)[PURPLE]
        prior = ValuationBot.expected_gem_values(info)[PURPLE]
        self.assertNotAlmostEqual(weighted, prior)
        bot.use_bid_evidence = False
        self.assertAlmostEqual(bot.gem_values(info)[PURPLE], prior)
        restored = SoftTrackingBot.__new__(SoftTrackingBot)
        restored.load_state(pickle.loads(pickle.dumps(bot.save_state())))
        self.assertFalse(restored.use_bid_evidence)
        self.assertAlmostEqual(restored.gem_values(info)[PURPLE], prior)
        restored.on_game_start()
        self.assertFalse(restored.use_bid_evidence)


class TrackingBidTests(unittest.TestCase):
    def test_tie_probabilities_match_exhaustive_simultaneous_bids(self):
        info = view()
        distributions = {p.id: [0.2, 0.5, 0.3] for p in info.other_players}
        for previous in (None, 0, 1, 3):
            current = replace(info, previous_winner_id=previous)
            for bid in range(4):
                expected = 0.0
                for rival_bids in product(range(3), repeat=3):
                    mass = 1.0
                    for p, b in zip(info.other_players, rival_bids):
                        mass *= distributions[p.id][b]
                    bids = (bid, *rival_bids)
                    if bid != max(bids):
                        continue
                    tied = [i for i, b in enumerate(bids) if b == bid]
                    if previous is None:
                        expected += mass / len(tied)
                    elif min(tied, key=lambda i: (i - previous - 1) % 4) == 0:
                        expected += mass
                self.assertAlmostEqual(
                    SoftTrackingBot.win_probability(current, bid, distributions),
                    expected,
                )

    def test_opponent_bid_support_respects_cash(self):
        info = view()
        bot = SoftTrackingBot(use_cash_limits=True, use_bid_model=True)
        for cash in (0, 1, 20):
            p = replace(info.other_players[0], coins=cash)
            pmf = bot._bid_distribution(info, p)
            self.assertEqual(len(pmf), cash + 1)
            self.assertAlmostEqual(sum(pmf), 1)
        all_broke = replace(
            info, other_players=tuple(replace(p, coins=0) for p in info.other_players)
        )
        distributions = {p.id: [1.0] for p in all_broke.other_players}
        self.assertEqual(bot.win_probability(all_broke, 1, distributions), 1)
        self.assertAlmostEqual(bot.win_probability(all_broke, 0, distributions), 0.25)

    def test_endgame_buys_gems_without_missions_and_values_finishes_or_denial(self):
        info = end_view()
        bot = SoftTrackingBot()
        self.assertGreater(ValuationBot().get_bid(info), 0)
        self.assertGreater(bot.get_bid(info), 0)
        mission = MissionCard(10, {BLUE: 1, PURPLE: 1})
        completing = replace(
            info,
            self_player=replace(info.self_player, won_gems=FrozenMap({BLUE: 1})),
            available_missions=(mission,),
        )
        bid = SoftTrackingBot().get_bid(completing)
        self.assertGreater(bid, 0)
        irrelevant = replace(info, available_missions=(MissionCard(5, {GREEN: 2}),))
        self.assertGreater(SoftTrackingBot().get_bid(irrelevant), 0)
        rival = replace(info.other_players[0], won_gems=FrozenMap({BLUE: 1}))
        denying = replace(
            info,
            other_players=(rival, *info.other_players[1:]),
            available_missions=(mission,),
        )
        self.assertGreater(SoftTrackingBot().get_bid(denying), 0)

    def test_near_end_mission_finish_keeps_the_normal_cash_reserve(self):
        info = replace(
            end_view(),
            gem_deck_count=2,
            gems_remaining=3,
            unknown_gems=FrozenMap(dict(zip(STANDARD_GEMS, (3, 3, 3, 3, 2)))),
            available_missions=(MissionCard(10, {BLUE: 1, PURPLE: 1}),),
        )
        info = replace(
            info, self_player=replace(info.self_player, won_gems=FrozenMap({BLUE: 1}))
        )
        bot = SoftTrackingBot()
        estimate = bot.evaluate(info)
        self.assertGreater(estimate.reserve, 1)
        self.assertGreater(bot.get_bid(info), 0)
        self.assertLessEqual(
            bot.get_bid(info), info.self_player.coins - estimate.reserve
        )

    def test_worker_games_all_seats_and_charts_and_selfplay_are_legal(self):
        with self.assertNoLogs("bot_runner", level="WARNING"):
            for seats in (3, 4, 5):
                for chart in VALUE_CHARTS:
                    names = [
                        "SoftTrackingBot",
                        "ValuationBot",
                        "FixedRuleBot",
                        "SoftTrackingBot",
                        "SmallBidBot",
                    ][:seats]
                    replay = simulate({"bots": names, "chart": chart.name, "seed": 13})
                    self.assertEqual(replay["frames"][-1]["event"]["type"], "FINAL")
                    for frame in replay["frames"]:
                        self.assertTrue(
                            all(p["coins"] >= 0 for p in frame["state"]["players"])
                        )
            config = {"bots": ["SoftTrackingBot"] * 5, "chart": "E", "seed": 11}
            self.assertEqual(simulate(config), simulate(config))

    def test_worst_initial_hand_capacities_fit_callback_budget(self):
        info = view()
        info = replace(
            info,
            num_players=5,
            gem_deck_count=13,
            gems_remaining=15,
            self_player=replace(
                info.self_player, hidden_gems=(BLUE, GREEN, PINK), hidden_gem_count=3
            ),
            other_players=tuple(
                replace(info.other_players[0], id=i, hidden_gem_count=3)
                for i in range(1, 5)
            ),
            unknown_gems=FrozenMap(dict(zip(STANDARD_GEMS, (5, 5, 5, 5, 5)))),
            available_missions=MISSION_CARDS[:4],
        )
        bot = SoftTrackingBot(
            use_bid_evidence=True, use_bid_model=True, use_cash_limits=True
        )
        start = perf_counter()
        bid = bot.get_bid(info)
        self.assertIs(type(bid), int)
        self.assertTrue(0 <= bid <= info.self_player.coins)
        self.assertLess(perf_counter() - start, 2.0)


if __name__ == "__main__":
    unittest.main()
