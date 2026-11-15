"""Exact ascending priorities, descending perception, E's peak and legal reveals."""

import pickle
import unittest
from collections import Counter
from dataclasses import replace
from time import perf_counter
from unittest.mock import patch

from bot_definitions.reveal_policy import assess_reveals, choose_reveal
from bot_definitions.soft_tracking_bot import LEVEL3_FEATURES, SoftTrackingBot
from bot_definitions.valuation_bot import ValuationBot
from constants import STANDARD_GEMS, GemType, WildCardRequirement
from decks import MISSION_CARDS, VALUE_CHARTS
from immutable import FrozenMap
from lib import MissionCard
from tests.test_valuation_bot import BLUE, GREEN, PINK, view


def reveal_view(
    chart="A", hand=(BLUE, GREEN), display=None, missions=(), won=None, row=None
):
    base = view()
    row = base.auction_gems if row is None else row
    unknown = Counter(dict.fromkeys(STANDARD_GEMS, 6))
    unknown.subtract(hand)
    unknown.subtract(display or {})
    unknown.subtract(won or {})
    unknown.subtract(row)
    deck = sum(unknown.values()) - sum(p.hidden_gem_count for p in base.other_players)
    return replace(
        base,
        value_chart=next(c for c in VALUE_CHARTS if c.name == chart),
        self_player=replace(
            base.self_player,
            hidden_gems=hand,
            hidden_gem_count=len(hand),
            won_gems=FrozenMap(won or {}),
        ),
        value_display=FrozenMap(display or {}),
        available_missions=missions,
        auction_gems=row,
        unknown_gems=FrozenMap(unknown),
        gem_deck_count=deck,
        gems_remaining=deck + len(row),
    )


def inputs(info):
    bot = ValuationBot()
    return bot.gem_values(info), {g: bot.deck_fraction(info, g) for g in STANDARD_GEMS}


class RevealPolicyTests(unittest.TestCase):
    def test_a_public_count_dominates_missions_and_private_prices(self):
        info = reveal_view(
            display={BLUE: 0, GREEN: 1},
            missions=(MissionCard(5, {BLUE: 2}), MissionCard(10, {BLUE: 1, PINK: 1})),
        )
        values = dict.fromkeys(STANDARD_GEMS, 1.0)
        values[BLUE] = 100.0  # Private value cannot override the requested A rule.
        fractions = dict.fromkeys(STANDARD_GEMS, 0.5)
        self.assertEqual(choose_reveal(info, values, fractions), 0)

    def test_a_mission_count_breaks_equal_public_counts(self):
        info = reveal_view(missions=(MissionCard(5, {BLUE: 2}),))
        self.assertEqual(choose_reveal(info, *inputs(info)), 1)
        # A pair uses two copies, but is ONE mission, not two mission votes.
        assessment = assess_reveals(info, *inputs(info))
        self.assertEqual(assessment[BLUE].mission_count, 1)

    def test_a_randomizes_only_final_ties_including_duplicate_card_indices(self):
        info = reveal_view(hand=(BLUE, BLUE, GREEN))
        values, fractions = inputs(info)
        values[BLUE], values[GREEN] = 1.0, 20.0
        with patch("bot_definitions.reveal_policy.choice", return_value=2) as pick:
            self.assertEqual(choose_reveal(info, values, fractions), 2)
            pick.assert_called_once_with([0, 1, 2])

    def test_public_count_excludes_owned_gems_row_and_private_hand(self):
        info = reveal_view(
            hand=(BLUE, BLUE, GREEN),
            display={BLUE: 0, GREEN: 1},
            won={BLUE: 3},
            row=(BLUE, PINK),
        )
        assessments = assess_reveals(info, *inputs(info))
        self.assertEqual(assessments[BLUE].public_count, 0)
        self.assertEqual(assessments[BLUE].priority[0], 0)
        with patch("bot_definitions.reveal_policy.choice", return_value=0) as pick:
            self.assertEqual(choose_reveal(info, *inputs(info)), 0)
            pick.assert_called_once_with([0, 1])

    def test_satisfied_missions_are_ignored_and_wildcards_count_once(self):
        info = reveal_view(
            missions=(
                MissionCard(5, {BLUE: 2}),
                MissionCard(5, {GemType.WILD_CARD: 2}, WildCardRequirement.SAME),
            ),
            won={BLUE: 2},
        )
        # Both missions are already completed by the newly awarded collection.
        self.assertTrue(
            all(
                a.mission_count == 0
                for a in assess_reveals(info, *inputs(info)).values()
            )
        )
        info = replace(
            info, self_player=replace(info.self_player, won_gems=FrozenMap())
        )
        assessment = assess_reveals(info, *inputs(info))
        self.assertEqual(assessment[BLUE].mission_count, 2)
        self.assertEqual(assessment[GREEN].mission_count, 1)

    def test_c_preserves_ascending_priority(self):
        info = reveal_view(
            "C", display={BLUE: 2, GREEN: 1}, missions=(MissionCard(5, {GREEN: 2}),)
        )
        self.assertEqual(choose_reveal(info, *inputs(info)), 1)

    def test_b_and_d_cool_valuable_future_targets_and_prefer_more_missions(self):
        for chart in ("B", "D"):
            info = reveal_view(chart, missions=(MissionCard(5, {BLUE: 2}),))
            values = dict.fromkeys(STANDARD_GEMS, 10.0)
            fractions = dict.fromkeys(STANDARD_GEMS, 0.5)
            self.assertEqual(choose_reveal(info, values, fractions), 0)
            values[GREEN] = 15.0  # Intrinsic count-derived value outranks missions.
            self.assertEqual(choose_reveal(info, values, fractions), 1)
            # A color that cannot appear in future auctions cannot be protected.
            info = replace(info, auction_gems=(BLUE,))
            fractions[GREEN] = 0.0
            self.assertEqual(choose_reveal(info, values, fractions), 0)

    def test_descending_mission_progress_uses_collection_without_consuming_gems(self):
        info = reveal_view(
            "B",
            hand=(BLUE, GREEN),
            won={PINK: 1},
            missions=(MissionCard(5, {BLUE: 1, PINK: 1}), MissionCard(5, {GREEN: 2})),
        )
        values = dict.fromkeys(STANDARD_GEMS, 10.0)
        fractions = dict.fromkeys(STANDARD_GEMS, 0.5)
        self.assertEqual(choose_reveal(info, values, fractions), 0)
        self.assertEqual(info.self_player.won_gems, {PINK: 1})

    def test_e_heats_less_valuable_colors_before_the_peak(self):
        info = reveal_view("E", display={BLUE: 0, GREEN: 2})
        values = dict.fromkeys(STANDARD_GEMS, 6.0)
        values[BLUE], values[GREEN] = 3.0, 14.0
        fractions = dict.fromkeys(STANDARD_GEMS, 0.5)
        assessments = assess_reveals(info, values, fractions)
        self.assertEqual(assessments[BLUE].display_price_change, 4)
        self.assertEqual(assessments[GREEN].display_price_change, 8)
        self.assertEqual(choose_reveal(info, values, fractions), 0)

    def test_e_can_cool_a_preferred_color_after_the_peak(self):
        info = reveal_view("E", display={BLUE: 3, GREEN: 4})
        values = dict.fromkeys(STANDARD_GEMS, 1.0)
        values[BLUE], values[GREEN] = 6.0, 0.0
        fractions = dict.fromkeys(STANDARD_GEMS, 0.5)
        assessments = assess_reveals(info, values, fractions)
        self.assertEqual(assessments[BLUE].display_price_change, -12)
        self.assertEqual(choose_reveal(info, values, fractions), 0)

    def test_e_can_choose_neutral_instead_of_advertising_its_best_target(self):
        info = reveal_view("E", display={BLUE: 2, GREEN: 5})
        values = dict.fromkeys(STANDARD_GEMS, 3.0)
        values[BLUE], values[GREEN] = 18.0, 0.0
        fractions = dict.fromkeys(STANDARD_GEMS, 0.5)
        fractions[GREEN] = 0.0
        self.assertEqual(choose_reveal(info, values, fractions), 1)

    def test_revealing_does_not_change_our_terminal_prices_or_scoring_collection(self):
        for chart in VALUE_CHARTS:
            info = replace(view(), value_chart=chart)
            bot = SoftTrackingBot(use_strategic_reveal=True)
            before = bot.gem_values(info)
            index = bot.choose_hidden_gem(info, {})
            hand = list(info.self_player.hidden_gems)
            color = hand.pop(index)
            display = Counter(info.value_display)
            display[color] += 1
            after = replace(
                info,
                value_display=FrozenMap(display),
                self_player=replace(
                    info.self_player,
                    hidden_gems=tuple(hand),
                    hidden_gem_count=len(hand),
                ),
            )
            self.assertEqual(before, bot.gem_values(after))
            self.assertEqual(info.self_player.won_gems, after.self_player.won_gems)

    def test_switch_checkpoint_and_no_bid_policy_changes(self):
        self.assertIn("use_strategic_reveal", LEVEL3_FEATURES)
        self.assertFalse(SoftTrackingBot().use_strategic_reveal)
        bot = SoftTrackingBot(use_strategic_reveal=True)
        self.assertFalse(
            bot.use_bid_evidence
            or bot.use_endgame_rules
            or bot.use_finance_horizon
            or bot.use_cash_fraction_model
        )
        clone = SoftTrackingBot.__new__(SoftTrackingBot)
        clone.load_state(pickle.loads(pickle.dumps(bot.save_state())))
        clone.on_game_start()
        self.assertTrue(clone.use_strategic_reveal)
        for chart in VALUE_CHARTS:
            info = replace(view(), value_chart=chart)
            self.assertEqual(bot.get_bid(info), ValuationBot().get_bid(info))
        self.assertEqual(SoftTrackingBot().choose_hidden_gem(view(), {}), 0)

    def test_empty_hand_and_callback_runtime(self):
        info = reveal_view(hand=())
        with self.assertRaises(ValueError):
            choose_reveal(info, *inputs(info))
        info = replace(view(), available_missions=MISSION_CARDS[:4])
        start = perf_counter()
        index = SoftTrackingBot(use_strategic_reveal=True).choose_hidden_gem(info, {})
        self.assertTrue(0 <= index < len(info.self_player.hidden_gems))
        self.assertLess(perf_counter() - start, 2)


if __name__ == "__main__":
    unittest.main()
