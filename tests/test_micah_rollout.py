"""Unit tests for MicahBot Milestone 5 CRN Monte Carlo rollout engine."""

from __future__ import annotations

import ast
import sys
import time
import unittest
from collections import Counter
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bot_definitions.micah_bid_model import OpponentBidModel  # noqa: E402
from bot_definitions.micah_inference import (  # noqa: E402
    OpponentHandBeliefs,
    build_inference_snapshot,
)
from bot_definitions.micah_rollout import (  # noqa: E402
    choose_rollout_winner,
    compute_rollout_deadline,
    evaluate_candidates_rollout,
    sample_rollout_world,
    select_rollout_candidates,
    should_run_rollout,
)
from bots import BotInfo, PlayerInfo, SelfInfo  # noqa: E402
from constants import (  # noqa: E402
    STANDARD_GEMS,
    AuctionType,
    GemType,
    WildCardRequirement,
)
from decks import VALUE_CHARTS  # noqa: E402
from immutable import FrozenMap  # noqa: E402
from lib import AuctionCard, MissionCard  # noqa: E402

BLUE, GREEN, PURPLE, YELLOW, PINK = STANDARD_GEMS
CHART_BY_NAME = {chart.name: chart for chart in VALUE_CHARTS}


def _gem_map(counts: dict[GemType, int] | None = None) -> FrozenMap[GemType, int]:
    full = dict.fromkeys(STANDARD_GEMS, 0)
    if counts is not None:
        full.update(counts)
    return FrozenMap(full)


def _opponent(
    player_id: int,
    *,
    coins: int = 20,
    hidden_gem_count: int = 0,
    won: dict[GemType, int] | None = None,
    missions: tuple[MissionCard, ...] = (),
    auction_cards: tuple[AuctionCard, ...] = (),
    invested_coins: int = 0,
) -> PlayerInfo:
    return PlayerInfo(
        id=player_id,
        coins=coins,
        hidden_gem_count=hidden_gem_count,
        revealed_gems=(),
        won_gems=_gem_map(won),
        auction_cards=auction_cards,
        mission_cards=missions,
        invested_coins=invested_coins,
    )


def _make_info(
    *,
    self_id: int = 0,
    coins: int = 20,
    hidden_gems: tuple[GemType, ...] = (BLUE, GREEN, PURPLE),
    self_won: dict[GemType, int] | None = None,
    others: tuple[PlayerInfo, ...] | None = None,
    round_number: int = 5,
    current_auction: AuctionCard | None = AuctionCard(AuctionType.AUCTION, 1),
    gem_deck_count: int = 6,
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
            _opponent(1, coins=18, hidden_gem_count=2),
            _opponent(2, coins=15, hidden_gem_count=2),
        )
    if unknown_gems is None:
        unknown_gems = {BLUE: 2, GREEN: 2, PURPLE: 2, YELLOW: 2, PINK: 2}
    if auction_cards_remaining is None:
        auction_cards_remaining = {
            AuctionCard(AuctionType.AUCTION, 1): 4,
            AuctionCard(AuctionType.AUCTION, 2): 1,
            AuctionCard(AuctionType.INVEST, 10): 1,
        }
    return BotInfo(
        self_player=SelfInfo(
            id=self_id,
            coins=coins,
            hidden_gem_count=len(hidden_gems),
            revealed_gems=(),
            won_gems=_gem_map(self_won),
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
        unknown_gems=_gem_map(unknown_gems),
        auction_gems=auction_gems,
        value_display=_gem_map(value_display),
        previous_winner_id=previous_winner_id,
        value_chart=CHART_BY_NAME[chart_name],
        available_missions=available_missions,
        auction_deck_count=sum(auction_cards_remaining.values()),
        auction_cards_remaining=FrozenMap(auction_cards_remaining),
    )


class MicahRolloutTests(unittest.TestCase):
    def test_no_forbidden_imports(self) -> None:
        rollout_path = ROOT / "src" / "bot_definitions" / "micah_rollout.py"
        tree = ast.parse(rollout_path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module.split(".")[0])
        for forbidden in ("harness", "player", "bot_runner", "multiprocessing"):
            self.assertNotIn(forbidden, imported)

    def test_deadline_and_trigger_and_candidate_helpers(self) -> None:
        start = time.perf_counter()
        self.assertAlmostEqual(
            compute_rollout_deadline(start, None, max_budget_seconds=0.20),
            start + 0.20,
            places=6,
        )
        bounded = compute_rollout_deadline(
            start, 1.0, reserve_seconds=0.40, max_budget_seconds=0.20
        )
        self.assertGreater(bounded, start)
        self.assertLessEqual(bounded - start, 0.25)

        tight = compute_rollout_deadline(
            start - 0.80, 1.0, reserve_seconds=0.40, max_budget_seconds=0.20
        )
        self.assertLessEqual(tight - time.perf_counter(), 0.01)

        info_early = _make_info(
            current_auction=AuctionCard(AuctionType.AUCTION, 1),
            available_missions=(),
        )
        # Early round with wide gap -> should not run
        self.assertFalse(should_run_rollout(info_early, 7.0, {0: 0.0, 5: 4.0}))
        # Terminal round -> should not run (handled by endgame_first_place_probs)
        self.assertFalse(should_run_rollout(info_early, 0.0, {0: 0.0, 5: 4.0}))
        # Single candidate -> should not run
        self.assertFalse(should_run_rollout(info_early, 2.0, {5: 4.0}))
        # Late round -> triggers
        self.assertTrue(should_run_rollout(info_early, 3.5, {0: 0.0, 5: 4.0}))
        # Near-tied candidates with rounds_left_after <= 5.0 -> triggers
        self.assertTrue(should_run_rollout(info_early, 4.5, {4: 3.1, 5: 4.0}))

        # Loan with rounds_left_after <= 6.0 -> triggers
        info_loan = _make_info(current_auction=AuctionCard(AuctionType.LOAN, 20))
        self.assertTrue(should_run_rollout(info_loan, 5.5, {0: 0.0, 5: 3.0}))
        self.assertFalse(should_run_rollout(info_loan, 7.0, {0: 0.0, 5: 3.0}))

        # Mission within 1-2 gems with rounds_left_after <= 5.0 -> triggers
        mission = MissionCard(10, {BLUE: 1, GREEN: 1, PURPLE: 1})
        info_mission = _make_info(
            self_won={BLUE: 1},
            available_missions=(mission,),
        )
        self.assertTrue(should_run_rollout(info_mission, 4.8, {0: 0.0, 5: 4.0}))

        # Candidate selection includes top utilities, 0, and tight covering bid.
        info_cov = _make_info(
            coins=15,
            others=(_opponent(1, coins=10), _opponent(2, coins=8)),
            previous_winner_id=1,
        )
        # Self id 0 beats id 1 on tie? (0 - 1 - 1) % 3 = 1; (1 - 1 - 1) % 3 = 2;
        # (2 - 1 - 1) % 3 = 0. Self beats 1 on tie, loses to 2 on tie -> cov = 10.
        utils = {b: float(5 - abs(b - 5)) for b in range(16)}
        selected = select_rollout_candidates(info_cov, utils, max_candidates=5)
        self.assertLessEqual(len(selected), 5)
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(selected[0], 5)
        self.assertIn(0, selected)
        self.assertIn(10, selected)

    def test_world_conservation_with_and_without_beliefs(self) -> None:
        info = _make_info(
            hidden_gems=(BLUE, GREEN),
            others=(
                _opponent(1, coins=20, hidden_gem_count=3),
                _opponent(2, coins=18, hidden_gem_count=3),
            ),
            gem_deck_count=4,
            unknown_gems={BLUE: 3, GREEN: 2, PURPLE: 2, YELLOW: 2, PINK: 1},
            auction_cards_remaining={
                AuctionCard(AuctionType.AUCTION, 1): 3,
                AuctionCard(AuctionType.AUCTION, 2): 1,
                AuctionCard(AuctionType.INVEST, 5): 1,
                AuctionCard(AuctionType.LOAN, 10): 1,
            },
        )
        beliefs = OpponentHandBeliefs()
        snap = build_inference_snapshot(info)
        self.assertIsNotNone(snap)
        assert snap is not None
        beliefs.update_from_bid(1, snap, bid=15)

        expected_unknown = Counter(dict(info.unknown_gems))
        expected_auctions = Counter(dict(info.auction_cards_remaining))

        for active_beliefs in (None, beliefs):
            for seed in range(6):
                world = sample_rollout_world(
                    info, Random(seed), hand_beliefs=active_beliefs
                )
                self.assertEqual(len(world.gem_deck), info.gem_deck_count)
                combined = Counter(world.gem_deck)
                for opp in info.other_players:
                    self.assertEqual(len(world.opp_hands[opp.id]), opp.hidden_gem_count)
                    combined.update(world.opp_hands[opp.id])
                self.assertEqual(combined, expected_unknown)
                self.assertEqual(Counter(world.auction_deck), expected_auctions)

    def test_crn_determinism(self) -> None:
        info = _make_info(
            coins=20,
            hidden_gems=(BLUE, GREEN),
            others=(
                _opponent(1, coins=16, hidden_gem_count=2),
                _opponent(2, coins=14, hidden_gem_count=2),
            ),
            gem_deck_count=4,
            unknown_gems={BLUE: 2, GREEN: 2, PURPLE: 2, YELLOW: 1, PINK: 1},
            auction_gems=(BLUE, YELLOW),
            auction_cards_remaining={
                AuctionCard(AuctionType.AUCTION, 1): 2,
                AuctionCard(AuctionType.AUCTION, 2): 1,
            },
        )
        candidates = [0, 6, 10, 14]
        model = OpponentBidModel()
        res1 = evaluate_candidates_rollout(
            info,
            candidates,
            model=model,
            rng=Random(12345),
            max_worlds=8,
            min_worlds=4,
            deadline=time.perf_counter() + 5.0,
        )
        res2 = evaluate_candidates_rollout(
            info,
            candidates,
            model=model,
            rng=Random(12345),
            max_worlds=8,
            min_worlds=4,
            deadline=time.perf_counter() + 5.0,
        )
        self.assertEqual(res1.worlds_completed, 8)
        self.assertEqual(res1.worlds_completed, res2.worlds_completed)
        self.assertEqual(res1.stats, res2.stats)
        self.assertEqual(res1.best_bid, res2.best_bid)

    def test_expired_deadline_stops_cleanly_under_5ms(self) -> None:
        info = _make_info()
        expired = time.perf_counter() - 0.01
        t0 = time.perf_counter()
        res = evaluate_candidates_rollout(
            info,
            [0, 5, 10],
            rng=Random(0),
            max_worlds=32,
            min_worlds=12,
            deadline=expired,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(res.worlds_completed, 0)
        self.assertIsNone(res.best_bid)
        self.assertEqual(res.stats, {})
        self.assertLess(elapsed_ms, 5.0)

    def test_choose_rollout_winner_tie_breaking(self) -> None:
        self.assertIsNone(choose_rollout_winner({0: (1.0, 1.0, 5.0)}, 8, min_worlds=12))
        # Within tie_tol=0.02 on first_place_prob, higher mean_margin wins.
        stats = {
            0: (0.76, 0.76, 1.0),
            8: (0.75, 0.75, 4.5),
            12: (0.50, 0.50, 8.0),
        }
        self.assertEqual(
            choose_rollout_winner(stats, 100, min_worlds=12, tie_tol=0.02),
            8,
        )
        # Exact tie on fp, fw, and margin breaks by analytic utility.
        tied = {
            4: (0.50, 0.50, 2.0),
            6: (0.50, 0.50, 2.0),
        }
        self.assertEqual(
            choose_rollout_winner(
                tied, 20, min_worlds=12, candidate_utilities={4: 3.5, 6: 2.0}
            ),
            4,
        )

    def test_late_game_covering_bid_and_loan_for_final_lot(self) -> None:
        # Scenario 1: 2 rounds left, covering bid beats 0 bid and overbidding.
        # Chart B gives 20 per blue (0 in display, 0 hidden). We have 18 coins,
        # opponent has 10 coins and bids ~8-10, 2 auctions of 1 blue each remain.
        # Bidding 11 guarantees winning the first blue (net +9) while leaving 7
        # coins; bidding 0 lets opponent buy blue for <= 10 and still contest
        # the second blue!
        info_cover = _make_info(
            coins=18,
            hidden_gems=(),
            others=(_opponent(1, coins=10, hidden_gem_count=0),),
            current_auction=AuctionCard(AuctionType.AUCTION, 1),
            gem_deck_count=0,
            unknown_gems=dict.fromkeys(STANDARD_GEMS, 0),
            auction_gems=(BLUE, BLUE),
            value_display=dict.fromkeys(STANDARD_GEMS, 0),
            previous_winner_id=0,  # Opponent 1 beats us on ties!
            chart_name="B",
            auction_cards_remaining={AuctionCard(AuctionType.AUCTION, 1): 1},
        )
        res_cov = evaluate_candidates_rollout(
            info_cover,
            [0, 11, 18],
            rng=Random(7),
            max_worlds=24,
            min_worlds=12,
        )
        self.assertEqual(res_cov.worlds_completed, 24)
        self.assertEqual(res_cov.best_bid, 11)
        self.assertGreater(res_cov.stats[11][0], res_cov.stats[0][0])

        # Scenario 2: Loan needed to buy final 2-gem lot!
        # Round 0 is LOAN 20. Next round is AUCTION 2 for (BLUE, GREEN) worth 40
        # on Chart B (0 in display). Self has 0 coins, opponent has 12 coins.
        # Winning Loan 20 at bid 5 gives 15 coins -> outbids opponent's 12 coins
        # on the 40-pt lot (winning +40 - 13 - 5 = +22 vs opponent +28 if we pass).
        info_loan = _make_info(
            coins=0,
            hidden_gems=(),
            others=(_opponent(1, coins=12, hidden_gem_count=0),),
            current_auction=AuctionCard(AuctionType.LOAN, 20),
            gem_deck_count=0,
            unknown_gems=dict.fromkeys(STANDARD_GEMS, 0),
            auction_gems=(BLUE, GREEN),
            value_display=dict.fromkeys(STANDARD_GEMS, 0),
            previous_winner_id=1,
            chart_name="B",
            auction_cards_remaining={AuctionCard(AuctionType.AUCTION, 2): 1},
        )
        # Opponent with 12 coins can bid up to ~5-6 on LOAN 20; bidding 8 or 10
        # secures the loan and guarantees winning the 40-pt final lot!
        res_loan = evaluate_candidates_rollout(
            info_loan,
            [0, 10],
            rng=Random(9),
            max_worlds=24,
            min_worlds=12,
        )
        self.assertEqual(res_loan.worlds_completed, 24)
        self.assertEqual(res_loan.best_bid, 10)
        self.assertGreater(res_loan.stats[10][0], res_loan.stats[0][0])

    def test_multi_mission_claim_and_chart_b_e_scoring(self) -> None:
        # Chart E: values (0, 4, 10, 18, 6, 0).
        # Self already has 1 BLUE won; display has 2 BLUE and 1 GREEN.
        # Hidden hands: self has 1 BLUE, opp has 0 hidden -> final BLUE display = 3
        # (Chart E value 18!), final GREEN display = 1 (Chart E value 4!).
        # Available missions:
        #   M1: {BLUE: 2} -> value 5
        #   M2: {BLUE: 1, GREEN: 1} -> value 5
        #   M3: {WILD_CARD: 3, SAME} -> value 10 (needs 3 same -> NOT satisfied)
        # Round 0 is AUCTION 2 for (BLUE, GREEN).
        # If self wins at bid 12:
        #   won_gems becomes {BLUE: 2, GREEN: 1} -> satisfies BOTH M1 (+5) and M2 (+5)!
        #   Wait: opponent has 5 coins and 0 gems; self has 20 coins.
        #   Self score at bid 12:
        #     coins = 20 - 12 = 8
        #     gems = 2 * 18 (BLUE) + 1 * 4 (GREEN) = 40
        #     missions = 5 + 5 = 10
        #     total = 8 + 40 + 10 = 58!
        #   Opp score: coins = 5, gems = 0, missions = 0 -> total = 5!
        #   Exact margin = 58 - 5 = 53.0!
        m1 = MissionCard(5, {BLUE: 2})
        m2 = MissionCard(5, {BLUE: 1, GREEN: 1})
        m3 = MissionCard(10, {GemType.WILD_CARD: 3}, WildCardRequirement.SAME)
        info_e = _make_info(
            coins=20,
            hidden_gems=(BLUE,),
            self_won={BLUE: 1},
            others=(_opponent(1, coins=5, hidden_gem_count=0),),
            current_auction=AuctionCard(AuctionType.AUCTION, 2),
            gem_deck_count=0,
            unknown_gems=dict.fromkeys(STANDARD_GEMS, 0),
            auction_gems=(BLUE, GREEN),
            value_display={BLUE: 2, GREEN: 1},
            previous_winner_id=1,
            chart_name="E",
            available_missions=(m1, m2, m3),
            auction_cards_remaining={},
        )
        res_e = evaluate_candidates_rollout(
            info_e,
            [12],
            rng=Random(0),
            max_worlds=16,
            min_worlds=12,
        )
        fp, fw, margin = res_e.stats[12]
        self.assertEqual(fp, 1.0)
        self.assertEqual(fw, 1.0)
        self.assertAlmostEqual(margin, 53.0, places=6)

        # Chart B with 6 gems in display (clamped to index 6 -> value 0):
        # Display starts with 4 BLUE; self has 1 hidden BLUE, opp has 1 hidden BLUE.
        # Total display BLUE = 6 -> Chart B gives 0 per BLUE.
        info_b = _make_info(
            coins=10,
            hidden_gems=(BLUE,),
            self_won={BLUE: 1},
            others=(_opponent(1, coins=5, hidden_gem_count=1),),
            current_auction=AuctionCard(AuctionType.AUCTION, 1),
            gem_deck_count=0,
            unknown_gems={BLUE: 1},
            auction_gems=(BLUE,),
            value_display={BLUE: 4},
            previous_winner_id=1,
            chart_name="B",
            available_missions=(),
            auction_cards_remaining={},
        )
        res_b = evaluate_candidates_rollout(
            info_b,
            [6],
            rng=Random(0),
            max_worlds=16,
            min_worlds=12,
        )
        # Self coins: 10 - 6 = 4, BLUE value = 0 -> score = 4; opp = 5 -> margin -1.0
        self.assertAlmostEqual(res_b.stats[6][2], -1.0, places=6)

    def test_rollout_pop_reveal_respects_config_leak_penalty(self) -> None:
        from bot_definitions.micah_rollout import _pop_reveal_gem, _SimPlayer
        from bot_definitions.micah_strategy import StrategyConfig

        # Choose between BLUE (gi=0, sup=1, shift>0) and GREEN (gi=1, sup=2, shift=0).
        # With coins=20 vs opp=10, self wants BLUE cheaper (negative directional),
        # so at leak_penalty=0.0 GREEN wins, and at high leak_penalty BLUE wins.
        p0 = _SimPlayer(0, 20, 0, 0, [0, 0, 0, 0, 0], [0, 1])
        p1 = _SimPlayer(1, 10, 0, 0, [0, 0, 0, 0, 0], [2])
        popped_low = _pop_reveal_gem(
            _SimPlayer(0, 20, 0, 0, [0, 0, 0, 0, 0], [0, 1]),
            0,
            0,
            [p0, p1],
            [0, 1, 1],
            1,
            [1, 5, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [1, 1, 0, 0, 0],
            (),
            VALUE_CHARTS[0].values,
            StrategyConfig(reveal_leak_penalty=0.0, reveal_directional_weight=1.0),
        )
        popped_high = _pop_reveal_gem(
            _SimPlayer(0, 20, 0, 0, [0, 0, 0, 0, 0], [0, 1]),
            0,
            0,
            [p0, p1],
            [0, 1, 1],
            1,
            [1, 5, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [1, 1, 0, 0, 0],
            (),
            VALUE_CHARTS[0].values,
            StrategyConfig(reveal_leak_penalty=2.0, reveal_directional_weight=1.0),
        )
        self.assertEqual(popped_low, 1)  # GREEN (shift=0, sup=2) beats negative BLUE
        self.assertEqual(
            popped_high, 0
        )  # BLUE (sup=1) beats high leak penalty on GREEN

    def test_benchmark_32_worlds(self) -> None:
        info = _make_info(
            coins=24,
            hidden_gems=(BLUE, GREEN, PURPLE),
            self_won={BLUE: 1},
            others=(
                _opponent(1, coins=20, hidden_gem_count=2, won={YELLOW: 1}),
                _opponent(2, coins=16, hidden_gem_count=2, won={PINK: 1}),
                _opponent(3, coins=18, hidden_gem_count=2),
            ),
            current_auction=AuctionCard(AuctionType.AUCTION, 2),
            gem_deck_count=6,
            unknown_gems={BLUE: 2, GREEN: 3, PURPLE: 2, YELLOW: 3, PINK: 2},
            auction_gems=(BLUE, GREEN),
            value_display={BLUE: 1, YELLOW: 1},
            available_missions=(
                MissionCard(10, {BLUE: 1, GREEN: 1, PURPLE: 1}),
                MissionCard(5, {YELLOW: 2}),
            ),
            auction_cards_remaining={
                AuctionCard(AuctionType.AUCTION, 1): 4,
                AuctionCard(AuctionType.AUCTION, 2): 1,
                AuctionCard(AuctionType.INVEST, 10): 1,
                AuctionCard(AuctionType.LOAN, 10): 1,
            },
        )
        t0 = time.perf_counter()
        res = evaluate_candidates_rollout(
            info,
            [0, 8, 12, 16, 21],
            rng=Random(42),
            max_worlds=32,
            min_worlds=12,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(res.worlds_completed, 32)
        self.assertIsNotNone(res.best_bid)
        self.assertLess(elapsed_ms, 500.0)


if __name__ == "__main__":
    unittest.main()
