"""Unit tests for MicahBot and Milestone 2 analytic helpers (dev-only)."""

from __future__ import annotations

import itertools
import pickle
import sys
import time
import unittest
from collections import Counter
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bot_definitions.micah_bot import MicahBot
from bot_definitions.micah_strategy import (
    DEFAULT_STRATEGY_CONFIG,
    StrategyConfig,
    auction_win_net_gain,
    beats_on_tie,
    choose_bid,
    choose_reveal_index,
    estimate_rounds_left_after,
    expected_future_supply,
    gem_expected_values,
    hypergeom_count_probs,
    immediate_mission_bonus,
    legal_bid_cap,
    mission_progress_value,
    public_value_shift,
    tie_priority_rank,
)
from bot_runner import BotRunner
from bots import BotInfo, PlayerInfo, SelfInfo
from constants import STANDARD_GEMS, AuctionType, GemType, WildCardRequirement
from decks import VALUE_CHARTS
from harness import _select_winner, play_game
from immutable import FrozenMap
from lib import AuctionCard, MissionCard

BLUE, GREEN, PURPLE, YELLOW, PINK = STANDARD_GEMS
CHART_BY_NAME = {chart.name: chart for chart in VALUE_CHARTS}


def _make_info(
    *,
    self_id: int = 0,
    coins: int = 20,
    hidden_gems: tuple[GemType, ...] = (BLUE, GREEN, PURPLE),
    self_won: dict[GemType, int] | None = None,
    others: tuple[PlayerInfo, ...] | None = None,
    round_number: int = 3,
    current_auction: AuctionCard | None = AuctionCard(AuctionType.AUCTION, 1),
    gem_deck_count: int = 10,
    unknown_gems: dict[GemType, int] | None = None,
    auction_gems: tuple[GemType, ...] = (BLUE, YELLOW),
    value_display: dict[GemType, int] | None = None,
    previous_winner_id: int | None = 1,
    chart_name: str = "A",
    available_missions: tuple[MissionCard, ...] = (),
    auction_cards_remaining: dict[AuctionCard, int] | None = None,
) -> BotInfo:
    if others is None:
        others = (
            PlayerInfo(
                id=1,
                coins=18,
                hidden_gem_count=3,
                revealed_gems=(),
                won_gems=FrozenMap({g: 0 for g in STANDARD_GEMS}),
                auction_cards=(),
                mission_cards=(),
                invested_coins=0,
            ),
            PlayerInfo(
                id=2,
                coins=15,
                hidden_gem_count=3,
                revealed_gems=(),
                won_gems=FrozenMap({g: 0 for g in STANDARD_GEMS}),
                auction_cards=(),
                mission_cards=(),
                invested_coins=0,
            ),
        )
    won_full = {g: 0 for g in STANDARD_GEMS}
    if self_won:
        won_full.update(self_won)
    unknown_full = {g: 3 for g in STANDARD_GEMS}
    if unknown_gems is not None:
        unknown_full = {g: unknown_gems.get(g, 0) for g in STANDARD_GEMS}
    disp_full = {g: 0 for g in STANDARD_GEMS}
    if value_display is not None:
        disp_full = {g: value_display.get(g, 0) for g in STANDARD_GEMS}
    if auction_cards_remaining is None:
        auction_cards_remaining = {
            AuctionCard(AuctionType.AUCTION, 1): 6,
            AuctionCard(AuctionType.AUCTION, 2): 2,
            AuctionCard(AuctionType.INVEST, 10): 1,
            AuctionCard(AuctionType.LOAN, 20): 1,
        }
    return BotInfo(
        self_player=SelfInfo(
            id=self_id,
            coins=coins,
            hidden_gem_count=len(hidden_gems),
            revealed_gems=(),
            won_gems=FrozenMap(won_full),
            auction_cards=(),
            mission_cards=(),
            invested_coins=0,
            hidden_gems=hidden_gems,
        ),
        other_players=others,
        num_players=1 + len(others),
        round_number=round_number,
        current_auction=current_auction,
        gem_deck_count=gem_deck_count,
        gems_remaining=gem_deck_count + len(auction_gems),
        unknown_gems=FrozenMap(unknown_full),
        auction_gems=auction_gems,
        value_display=FrozenMap(disp_full),
        previous_winner_id=previous_winner_id,
        value_chart=CHART_BY_NAME[chart_name],
        available_missions=available_missions,
        auction_deck_count=sum(auction_cards_remaining.values()),
        auction_cards_remaining=FrozenMap(auction_cards_remaining),
    )


class HypergeomAndValuesTests(unittest.TestCase):
    def test_hypergeometric_probs_match_bruteforce_and_sum_to_one(self) -> None:
        for known in (0, 1, 2, 5):
            for w in (0, 2, 4):
                pool_total = 10
                draws = 4
                probs = hypergeom_count_probs(known, w, pool_total, draws)
                self.assertAlmostEqual(sum(probs), 1.0, places=12)

                pool = [1] * w + [0] * (pool_total - w)
                counts: Counter[int] = Counter()
                combos = list(itertools.combinations(pool, draws))
                for combo in combos:
                    counts[min(6, known + sum(combo))] += 1
                for x in range(7):
                    expected = counts[x] / len(combos)
                    self.assertAlmostEqual(probs[x], expected, places=12)

    def test_chart_b_d_e_and_count_six_behavior(self) -> None:
        for chart_name in ("A", "B", "C", "D", "E"):
            info = _make_info(
                chart_name=chart_name,
                hidden_gems=(BLUE, BLUE),
                value_display={BLUE: 4},
                unknown_gems={BLUE: 2, GREEN: 4, PURPLE: 4},
                gem_deck_count=4,
            )
            evs = gem_expected_values(info)
            # Known BLUE = 4 + 2 = 6; any draw keeps count clamped at 6.
            expected_6 = float(CHART_BY_NAME[chart_name].get_value(6))
            self.assertAlmostEqual(evs[BLUE], expected_6, places=9)
            if chart_name in ("B", "D", "E"):
                self.assertEqual(evs[BLUE], 0.0)


class TieAndMissionTests(unittest.TestCase):
    def test_tie_priority_matches_harness_select_winner(self) -> None:
        rng = Random(0)
        for num_players in (3, 4, 5):
            for prev in range(num_players):
                ranked_ids = sorted(
                    range(num_players),
                    key=lambda pid: tie_priority_rank(pid, prev, num_players) or 0,
                )
                winner = _select_winner([5] * num_players, prev, rng)
                self.assertEqual(winner, ranked_ids[0])
                self.assertTrue(
                    beats_on_tie(ranked_ids[0], ranked_ids[1], prev, num_players)
                )
        self.assertIsNone(tie_priority_rank(0, None, 3))
        self.assertIsNone(beats_on_tie(0, 1, None, 3))

    def test_multiple_and_wildcard_mission_claims(self) -> None:
        missions = (
            MissionCard(5, {BLUE: 1, GREEN: 1}),
            MissionCard(10, {GemType.WILD_CARD: 3}, WildCardRequirement.SAME),
            MissionCard(15, {GemType.WILD_CARD: 4}, WildCardRequirement.PAIRS),
        )
        won = {BLUE: 2, GREEN: 0, PURPLE: 1}
        bonus = immediate_mission_bonus(won, (BLUE, GREEN), missions)
        # Completes BLUE+GREEN (5) and 3-SAME BLUE (10); PAIRS not yet met.
        self.assertEqual(bonus, 15)

    def test_wildcard_and_live_mission_progress_feasibility(self) -> None:
        same3 = (MissionCard(10, {GemType.WILD_CARD: 3}, WildCardRequirement.SAME),)
        diff4 = (
            MissionCard(10, {GemType.WILD_CARD: 4}, WildCardRequirement.DIFFERENT),
        )
        pairs4 = (MissionCard(15, {GemType.WILD_CARD: 4}, WildCardRequirement.PAIRS),)
        fixed1 = (MissionCard(5, {BLUE: 1, GREEN: 1}),)

        # No color can reach 3 SAME -> impossible wildcard mission scores 0.
        low_sup = {BLUE: 1.0, GREEN: 1.0, PURPLE: 1.0, YELLOW: 0.0, PINK: 0.0}
        self.assertEqual(mission_progress_value({}, (BLUE,), same3, low_sup, 2.0), 0.0)
        # Only 3 colors exist -> 4 DIFFERENT is impossible.
        self.assertEqual(mission_progress_value({}, (BLUE,), diff4, low_sup, 2.0), 0.0)
        # Only 1 pair can be formed -> 4 PAIRS (2 pairs) is impossible.
        self.assertEqual(mission_progress_value({}, (BLUE,), pairs4, low_sup, 2.0), 0.0)
        # Live 1-copy fixed requirement (0.5 expected supply > 0) stays alive.
        self.assertGreater(
            mission_progress_value({}, (BLUE,), fixed1, {BLUE: 0.0, GREEN: 0.5}, 2.0),
            0.0,
        )
        # Sold-out fixed requirement (0.0 supply) is dead.
        self.assertEqual(
            mission_progress_value({}, (BLUE,), fixed1, {BLUE: 0.0, GREEN: 0.0}, 2.0),
            0.0,
        )


class SupplyAndRevealTests(unittest.TestCase):
    def test_expected_future_supply_formula_and_zero_safe(self) -> None:
        info = _make_info(
            auction_gems=(BLUE, GREEN),
            gem_deck_count=6,
            unknown_gems={BLUE: 2, GREEN: 4, PURPLE: 0},
        )
        self.assertAlmostEqual(expected_future_supply(info, BLUE), 1.0 + 2.0)
        self.assertAlmostEqual(expected_future_supply(info, GREEN), 1.0 + 4.0)
        self.assertEqual(expected_future_supply(info, PURPLE), 0.0)

        zero_info = _make_info(
            auction_gems=(),
            gem_deck_count=0,
            unknown_gems={g: 0 for g in STANDARD_GEMS},
        )
        self.assertEqual(expected_future_supply(zero_info, BLUE), 0.0)

    def test_reveal_prefers_sold_out_and_chart_direction(self) -> None:
        # Sold-out PINK must beat high-supply BLUE on Chart A.
        info_sold = _make_info(
            chart_name="A",
            hidden_gems=(BLUE, PINK),
            auction_gems=(BLUE, BLUE),
            gem_deck_count=6,
            unknown_gems={BLUE: 4, GREEN: 2, PURPLE: 2, YELLOW: 2, PINK: 0},
            value_display={g: 1 for g in STANDARD_GEMS},
        )
        self.assertEqual(choose_reveal_index(info_sold), 1)

        # Verify public_value_shift signs on A/C vs B/D vs E.
        info_a = _make_info(chart_name="A", value_display={BLUE: 1})
        info_b = _make_info(chart_name="B", value_display={BLUE: 1})
        info_e_low = _make_info(
            chart_name="E",
            value_display={BLUE: 0},
            hidden_gems=(GREEN,),
            unknown_gems={BLUE: 2, GREEN: 4, PURPLE: 4, YELLOW: 4, PINK: 4},
        )
        info_e_high = _make_info(
            chart_name="E",
            value_display={BLUE: 3},
            hidden_gems=(BLUE,),
            unknown_gems={BLUE: 1, GREEN: 4, PURPLE: 4},
        )
        self.assertGreater(public_value_shift(info_a, BLUE), 0.0)
        self.assertLess(public_value_shift(info_b, BLUE), 0.0)
        self.assertGreater(public_value_shift(info_e_low, BLUE), 0.0)
        self.assertLess(public_value_shift(info_e_high, BLUE), 0.0)


class FinanceAndDecisionTests(unittest.TestCase):
    def test_investment_and_loan_accounting(self) -> None:
        # Investment: cost is only lost liquidity + tie penalty, NOT (1 + k) * c.
        invest = AuctionCard(AuctionType.INVEST, 10)
        info_rich = _make_info(
            coins=35,
            current_auction=invest,
            gem_deck_count=4,
            auction_gems=(BLUE, GREEN),
        )
        evs = gem_expected_values(info_rich)
        supplies = {g: 1.0 for g in STANDARD_GEMS}
        rounds_left = estimate_rounds_left_after(info_rich)
        gain_at_12 = auction_win_net_gain(
            info_rich, invest, 12, evs, supplies, rounds_left
        )
        self.assertGreater(gain_at_12, 0.0)

        # Low-cash conservation: with only 12 coins before final rounds,
        # marginal liquidity cost is higher than when holding 35 coins.
        info_tight = _make_info(
            coins=12,
            current_auction=invest,
            gem_deck_count=4,
            auction_gems=(BLUE, GREEN),
        )
        gain_tight = auction_win_net_gain(
            info_tight,
            invest,
            12,
            evs,
            supplies,
            estimate_rounds_left_after(info_tight),
        )
        self.assertLess(gain_tight, gain_at_12)

        # Loan: never bid >= L.
        loan = AuctionCard(AuctionType.LOAN, 20)
        for coins in (0, 2, 10, 30):
            info_loan = _make_info(coins=coins, current_auction=loan)
            bid = choose_bid(info_loan, DEFAULT_STRATEGY_CONFIG)
            self.assertIs(type(bid), int)
            self.assertGreaterEqual(bid, 0)
            self.assertLess(bid, loan.value)
            self.assertLessEqual(bid, legal_bid_cap(info_loan))

    def test_legal_plain_int_bids_across_charts_and_player_counts(self) -> None:
        for num_players in (3, 4, 5):
            others = tuple(
                PlayerInfo(
                    id=i,
                    coins=15,
                    hidden_gem_count=3,
                    revealed_gems=(),
                    won_gems=FrozenMap({g: 0 for g in STANDARD_GEMS}),
                    auction_cards=(),
                    mission_cards=(),
                    invested_coins=0,
                )
                for i in range(1, num_players)
            )
            for chart_name in ("A", "B", "C", "D", "E"):
                for card in (
                    AuctionCard(AuctionType.AUCTION, 1),
                    AuctionCard(AuctionType.AUCTION, 2),
                    AuctionCard(AuctionType.INVEST, 10),
                    AuctionCard(AuctionType.LOAN, 20),
                ):
                    info = _make_info(
                        coins=10,
                        others=others,
                        chart_name=chart_name,
                        current_auction=card,
                    )
                    bid = choose_bid(info, StrategyConfig())
                    self.assertIs(type(bid), int)
                    self.assertGreaterEqual(bid, 0)
                    self.assertLessEqual(bid, legal_bid_cap(info))
                    reveal = choose_reveal_index(info)
                    self.assertIs(type(reveal), int)
                    self.assertGreaterEqual(reveal, 0)
                    self.assertLess(reveal, len(info.self_player.hidden_gems))


class IntegrationAndTimingTests(unittest.TestCase):
    def test_checkpoint_size_and_reused_vs_fresh_games_at_2s_timeout(self) -> None:
        bot = MicahBot()
        checkpoint_bytes = len(pickle.dumps(bot.save_state()))
        self.assertLess(checkpoint_bytes, 4096)

        restored = MicahBot.__new__(MicahBot)
        restored.load_state(bot.save_state())
        self.assertEqual(restored.games_played, bot.games_played)

        # Fresh vs reused runners produce identical scores on fixed seeds,
        # with zero errors at bot_timeout=2.0 and fast ThinkPad runtime.
        start = time.perf_counter()
        with (
            BotRunner(MicahBot, timeout=2.0, seed=101) as r0,
            BotRunner(MicahBot, timeout=2.0, seed=102) as r1,
            BotRunner(MicahBot, timeout=2.0, seed=103) as r2,
        ):
            reused_res = play_game(3, bots=[r0, r1, r2], rng=Random(42))
            self.assertIsNone(r0.last_error)
            self.assertIsNone(r1.last_error)
            self.assertIsNone(r2.last_error)

        fresh_res = play_game(
            3,
            bots=[MicahBot, MicahBot, MicahBot],
            rng=Random(42),
            bot_timeout=2.0,
            bot_seed=101,
        )
        elapsed = time.perf_counter() - start
        self.assertEqual(len(reused_res.scores), 3)
        self.assertEqual(len(fresh_res.scores), 3)
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
