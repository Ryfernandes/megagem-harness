"""Unit tests for Bayesian opponent hidden-hand inference (Milestone 4)."""

from __future__ import annotations

import pickle
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bot_definitions.micah_inference import (  # noqa: E402
    InferenceSnapshot,
    OpponentHandBeliefs,
    build_inference_snapshot,
    hand_tuples,
    tuple_prior_weight,
)
from bot_definitions.micah_strategy import gem_expected_values  # noqa: E402
from bots import BotInfo, PlayerInfo, SelfInfo  # noqa: E402
from constants import STANDARD_GEMS, AuctionType, GemType  # noqa: E402
from decks import VALUE_CHARTS  # noqa: E402
from immutable import FrozenMap  # noqa: E402
from lib import AuctionCard, MissionCard, ValueChart  # noqa: E402

BLUE = GemType.BLUE_CIRCLE
GREEN = GemType.GREEN_OVAL
PURPLE = GemType.PURPLE_TRIANGLE
YELLOW = GemType.YELLOW_SQUARE
PINK = GemType.PINK_PENTAGON


def _gem_map(counts: dict[GemType, int] | None = None) -> FrozenMap[GemType, int]:
    full = dict.fromkeys(STANDARD_GEMS, 0)
    if counts is not None:
        full.update(counts)
    return FrozenMap(full)


def _opponent(
    player_id: int,
    *,
    coins: int = 20,
    hidden_gem_count: int = 3,
    won: dict[GemType, int] | None = None,
) -> PlayerInfo:
    return PlayerInfo(
        id=player_id,
        coins=coins,
        hidden_gem_count=hidden_gem_count,
        revealed_gems=(),
        won_gems=_gem_map(won),
        auction_cards=(),
        mission_cards=(),
        invested_coins=0,
    )


def _bot_info(
    *,
    chart: ValueChart = VALUE_CHARTS[0],
    self_hidden: tuple[GemType, ...] = (GREEN, YELLOW, PURPLE),
    opponents: tuple[PlayerInfo, ...] | None = None,
    unknown_gems: dict[GemType, int] | None = None,
    value_display: dict[GemType, int] | None = None,
    current_auction: AuctionCard | None = None,
    auction_gems: tuple[GemType, ...] = (BLUE, GREEN),
    available_missions: tuple[MissionCard, ...] = (),
    gem_deck_count: int = 10,
) -> BotInfo:
    opps = opponents if opponents is not None else (_opponent(1),)
    unk = _gem_map(
        unknown_gems
        if unknown_gems is not None
        else {BLUE: 4, GREEN: 3, PURPLE: 3, YELLOW: 3, PINK: 4}
    )
    disp = _gem_map(value_display)
    card = (
        current_auction
        if current_auction is not None
        else AuctionCard(AuctionType.AUCTION, 1)
    )
    return BotInfo(
        self_player=SelfInfo(
            id=0,
            coins=20,
            hidden_gem_count=len(self_hidden),
            revealed_gems=(),
            won_gems=_gem_map(),
            auction_cards=(),
            mission_cards=(),
            invested_coins=0,
            hidden_gems=self_hidden,
        ),
        other_players=opps,
        num_players=1 + len(opps),
        round_number=1,
        current_auction=card,
        gem_deck_count=gem_deck_count,
        gems_remaining=gem_deck_count + len(auction_gems),
        unknown_gems=unk,
        auction_gems=auction_gems,
        value_display=disp,
        previous_winner_id=None,
        value_chart=chart,
        available_missions=available_missions,
        auction_deck_count=10,
        auction_cards_remaining=FrozenMap({AuctionCard(AuctionType.AUCTION, 1): 10}),
    )


class HandTuplesAndPriorTests(unittest.TestCase):
    def test_hand_tuples_counts_and_sums(self) -> None:
        expected_counts = {0: 1, 1: 5, 2: 15, 3: 35, 4: 70, 5: 126}
        for size, expected in expected_counts.items():
            tuples = hand_tuples(size)
            self.assertEqual(len(tuples), expected)
            self.assertEqual(len(set(tuples)), expected)
            for t in tuples:
                self.assertEqual(len(t), 5)
                self.assertEqual(sum(t), size)
                self.assertTrue(all(c >= 0 for c in t))

        self.assertEqual(hand_tuples(-2), ((0, 0, 0, 0, 0),))

    def test_tuple_prior_weight(self) -> None:
        unknown = (4, 3, 2, 1, 0)
        self.assertEqual(tuple_prior_weight((2, 1, 0, 0, 0), unknown), 18.0)
        self.assertEqual(tuple_prior_weight((0, 0, 0, 0, 1), unknown), 0.0)
        self.assertEqual(tuple_prior_weight((5, 0, 0, 0, 0), unknown), 0.0)


class UniformBeliefsExactnessTests(unittest.TestCase):
    def test_uniform_beliefs_match_exact_hypergeom_across_all_charts(self) -> None:
        opps = (
            _opponent(1, hidden_gem_count=3),
            _opponent(2, hidden_gem_count=2),
            _opponent(3, hidden_gem_count=4),
        )
        unknown = {BLUE: 4, GREEN: 2, PURPLE: 3, YELLOW: 5, PINK: 1}
        display = {BLUE: 1, GREEN: 2, PURPLE: 0, YELLOW: 1, PINK: 3}
        for chart in VALUE_CHARTS:
            info = _bot_info(
                chart=chart,
                self_hidden=(BLUE, GREEN, GREEN),
                opponents=opps,
                unknown_gems=unknown,
                value_display=display,
                gem_deck_count=6,
            )
            exact = gem_expected_values(info)

            # Empty beliefs (no entries yet)
            fresh = OpponentHandBeliefs()
            ev_fresh = fresh.gem_expected_values_with_beliefs(info)
            for gem in STANDARD_GEMS:
                self.assertAlmostEqual(ev_fresh[gem], exact[gem], delta=1e-12)

            # Explicitly populated uniform beliefs (exercises Vandermonde DP)
            uniform = OpponentHandBeliefs()
            for opp in opps:
                uniform.ensure_opponent(opp.id, opp.hidden_gem_count)
            ev_dp = uniform.gem_expected_values_with_beliefs(info)
            for gem in STANDARD_GEMS:
                self.assertAlmostEqual(
                    ev_dp[gem],
                    exact[gem],
                    delta=1e-12,
                    msg=f"Chart {chart.name}, gem {gem}",
                )


class BidInferenceDirectionTests(unittest.TestCase):
    def _posterior_blue(
        self,
        beliefs: OpponentHandBeliefs,
        info: BotInfo,
        opp_id: int = 1,
    ) -> float:
        unk = tuple(info.unknown_gems.get(g, 0) for g in STANDARD_GEMS)
        opp = next(p for p in info.other_players if p.id == opp_id)
        return beliefs.expected_hand_counts(opp_id, opp.hidden_gem_count, unk)[0]

    def test_chart_a_high_bid_raises_expected_blue(self) -> None:
        info = _bot_info(
            chart=VALUE_CHARTS[0],  # Chart A: (0, 4, 8, 12, 16, 20)
            self_hidden=(GREEN, PURPLE, YELLOW),
            opponents=(_opponent(1, coins=25, hidden_gem_count=3),),
            unknown_gems={BLUE: 4, GREEN: 3, PURPLE: 3, YELLOW: 3, PINK: 4},
            value_display={BLUE: 0},
            auction_gems=(BLUE, GREEN),
        )
        snap = build_inference_snapshot(info)
        assert snap is not None
        beliefs = OpponentHandBeliefs()
        prior_blue = self._posterior_blue(beliefs, info)

        beliefs.update_from_bid(1, snap, bid=10)
        post_blue = self._posterior_blue(beliefs, info)
        self.assertGreater(post_blue, prior_blue + 0.1)

        # EV with beliefs increases for BLUE under Chart A and scales with strength.
        exact_ev = gem_expected_values(info)[BLUE]
        half_ev = beliefs.gem_expected_values_with_beliefs(
            info, inference_strength=0.5
        )[BLUE]
        full_ev = beliefs.gem_expected_values_with_beliefs(
            info, inference_strength=1.0
        )[BLUE]
        self.assertGreater(half_ev, exact_ev)
        self.assertGreater(full_ev, half_ev)

        # Bounded & mean == 1.0
        opp_l = beliefs.likelihoods[1]
        self.assertAlmostEqual(sum(opp_l.values()) / len(opp_l), 1.0, places=9)
        self.assertTrue(all(0.15 - 1e-12 <= v <= 6.0 + 1e-12 for v in opp_l.values()))

    def test_censored_all_in_bid_raises_expected_blue_on_chart_a(self) -> None:
        info = _bot_info(
            chart=VALUE_CHARTS[0],
            self_hidden=(GREEN, PURPLE, YELLOW),
            opponents=(_opponent(1, coins=9, hidden_gem_count=3),),
            unknown_gems={BLUE: 4, GREEN: 3, PURPLE: 3, YELLOW: 3, PINK: 4},
            value_display={BLUE: 0},
            auction_gems=(BLUE, GREEN),
        )
        snap = build_inference_snapshot(info)
        assert snap is not None
        beliefs = OpponentHandBeliefs()
        prior_blue = self._posterior_blue(beliefs, info)
        beliefs.update_from_bid(1, snap, bid=9)
        self.assertGreater(self._posterior_blue(beliefs, info), prior_blue + 0.05)

    def test_chart_b_high_bid_lowers_expected_blue(self) -> None:
        info = _bot_info(
            chart=VALUE_CHARTS[1],  # Chart B: (20, 16, 12, 8, 4, 0)
            self_hidden=(GREEN, PURPLE, YELLOW),
            opponents=(_opponent(1, coins=25, hidden_gem_count=3),),
            unknown_gems={BLUE: 4, GREEN: 3, PURPLE: 3, YELLOW: 3, PINK: 4},
            value_display={BLUE: 0},
            auction_gems=(BLUE, GREEN),
        )
        snap = build_inference_snapshot(info)
        assert snap is not None
        beliefs = OpponentHandBeliefs()
        prior_blue = self._posterior_blue(beliefs, info)

        beliefs.update_from_bid(1, snap, bid=15)
        post_blue = self._posterior_blue(beliefs, info)
        self.assertLess(post_blue, prior_blue - 0.1)

    def test_chart_e_non_monotonic_direction_flips_with_display(self) -> None:
        # Chart E: (0, 4, 10, 18, 6, 0)
        # Display=1: extra BLUE moves toward 2..3 (higher EV) -> high bid raises E.
        info_low_disp = _bot_info(
            chart=VALUE_CHARTS[4],
            self_hidden=(GREEN, PURPLE, YELLOW),
            opponents=(_opponent(1, coins=25, hidden_gem_count=3),),
            unknown_gems={BLUE: 4, GREEN: 3, PURPLE: 3, YELLOW: 3, PINK: 4},
            value_display={BLUE: 1},
            auction_gems=(BLUE, GREEN),
        )
        snap_low = build_inference_snapshot(info_low_disp)
        assert snap_low is not None
        beliefs_low = OpponentHandBeliefs()
        prior_low = self._posterior_blue(beliefs_low, info_low_disp)
        beliefs_low.update_from_bid(1, snap_low, bid=14)
        self.assertGreater(
            self._posterior_blue(beliefs_low, info_low_disp), prior_low + 0.05
        )

        # Display=3: at peak (18); extra BLUE pushes to 4..5 (6/0) -> high bid lowers E.
        info_peak_disp = _bot_info(
            chart=VALUE_CHARTS[4],
            self_hidden=(GREEN, PURPLE, YELLOW),
            opponents=(_opponent(1, coins=25, hidden_gem_count=3),),
            unknown_gems={BLUE: 3, GREEN: 4, PURPLE: 3, YELLOW: 3, PINK: 4},
            value_display={BLUE: 3},
            auction_gems=(BLUE, GREEN),
        )
        snap_peak = build_inference_snapshot(info_peak_disp)
        assert snap_peak is not None
        beliefs_peak = OpponentHandBeliefs()
        prior_peak = self._posterior_blue(beliefs_peak, info_peak_disp)
        beliefs_peak.update_from_bid(1, snap_peak, bid=14)
        self.assertLess(
            self._posterior_blue(beliefs_peak, info_peak_disp), prior_peak - 0.05
        )

    def test_mission_explaining_away_reduces_blue_shift(self) -> None:
        mission = MissionCard(15, {BLUE: 5})
        info_no_mission = _bot_info(
            chart=VALUE_CHARTS[0],
            self_hidden=(GREEN, PURPLE, YELLOW),
            opponents=(_opponent(1, coins=30, hidden_gem_count=3),),
            unknown_gems={BLUE: 4, GREEN: 3, PURPLE: 3, YELLOW: 3, PINK: 4},
            value_display={BLUE: 0},
            current_auction=AuctionCard(AuctionType.AUCTION, 2),
            auction_gems=(BLUE, BLUE),
            available_missions=(),
        )
        info_with_mission = _bot_info(
            chart=VALUE_CHARTS[0],
            self_hidden=(GREEN, PURPLE, YELLOW),
            opponents=(_opponent(1, coins=30, hidden_gem_count=3, won={BLUE: 3}),),
            unknown_gems={BLUE: 4, GREEN: 3, PURPLE: 3, YELLOW: 3, PINK: 4},
            value_display={BLUE: 0},
            current_auction=AuctionCard(AuctionType.AUCTION, 2),
            auction_gems=(BLUE, BLUE),
            available_missions=(mission,),
        )

        snap_no = build_inference_snapshot(info_no_mission)
        snap_yes = build_inference_snapshot(info_with_mission)
        assert snap_no is not None and snap_yes is not None

        b_no = OpponentHandBeliefs()
        b_yes = OpponentHandBeliefs()
        prior = self._posterior_blue(b_no, info_no_mission)

        b_no.update_from_bid(1, snap_no, bid=20)
        b_yes.update_from_bid(1, snap_yes, bid=20)

        shift_no_mission = self._posterior_blue(b_no, info_no_mission) - prior
        shift_with_mission = self._posterior_blue(b_yes, info_with_mission) - prior
        self.assertGreater(shift_no_mission, 0.2)
        self.assertLess(abs(shift_with_mission), shift_no_mission * 0.4)

    def test_cash_poor_zero_bid_and_non_auction_are_skipped(self) -> None:
        poor_info = _bot_info(
            opponents=(_opponent(1, coins=4, hidden_gem_count=3),),
        )
        snap_poor = build_inference_snapshot(poor_info)
        assert snap_poor is not None
        beliefs = OpponentHandBeliefs()
        beliefs.update_from_bid(1, snap_poor, bid=0)
        self.assertEqual(beliefs.likelihoods, {})

        invest_info = _bot_info(
            current_auction=AuctionCard(AuctionType.INVEST, 10),
        )
        snap_invest = build_inference_snapshot(invest_info)
        assert snap_invest is not None
        beliefs.update_from_bid(1, snap_invest, bid=8)
        self.assertEqual(beliefs.likelihoods, {})


class RevealConditioningTests(unittest.TestCase):
    def test_reveal_pink_removes_zero_pink_tuples_and_shrinks_hand(self) -> None:
        info = _bot_info(
            chart=VALUE_CHARTS[0],
            opponents=(_opponent(1, coins=20, hidden_gem_count=2),),
            unknown_gems={BLUE: 3, GREEN: 3, PURPLE: 3, YELLOW: 3, PINK: 1},
            value_display={PINK: 0},
        )
        snap = build_inference_snapshot(info)
        assert snap is not None

        beliefs = OpponentHandBeliefs()
        # Bias pre-reveal beliefs strongly toward (BLUE=2, PINK=0) vs (BLUE=1, PINK=1)
        pre = beliefs.ensure_opponent(1, 2)
        for t in list(pre):
            if t[4] == 0:
                pre[t] = 5.0
            else:
                pre[t] = 0.5

        new_unknown = (3, 3, 3, 3, 0)
        beliefs.update_from_reveal(
            opp_id=1,
            revealed_gem=PINK,
            snapshot=snap,
            new_unknown_counts=new_unknown,
            opp_won_gems=_gem_map(),
            available_missions=(),
            reveal_strength=0.0,
        )
        after = beliefs.likelihoods[1]
        self.assertTrue(all(sum(t) == 1 for t in after))
        # Since new_unknown has PINK == 0, no surviving hand tuple can have PINK > 0
        self.assertTrue(all(t[4] == 0 for t in after))
        # If PINK was 2 pre-reveal and 1 post-reveal, uniform pre stays uniform post
        info2 = _bot_info(
            chart=VALUE_CHARTS[0],
            opponents=(_opponent(1, coins=20, hidden_gem_count=3),),
            unknown_gems={BLUE: 3, GREEN: 3, PURPLE: 3, YELLOW: 3, PINK: 2},
        )
        snap2 = build_inference_snapshot(info2)
        assert snap2 is not None
        beliefs2 = OpponentHandBeliefs()
        beliefs2.update_from_reveal(
            opp_id=1,
            revealed_gem=PINK,
            snapshot=snap2,
            new_unknown_counts=(3, 3, 3, 3, 1),
            opp_won_gems=_gem_map(),
            available_missions=(),
            reveal_strength=0.0,
        )
        after2 = beliefs2.likelihoods[1]
        self.assertTrue(all(sum(t) == 2 for t in after2))
        self.assertTrue(all(abs(v - 1.0) < 1e-12 for v in after2.values()))

    def test_reveal_pink_filters_out_pre_reveal_pink_zero_weight(self) -> None:
        info = _bot_info(
            chart=VALUE_CHARTS[0],
            opponents=(_opponent(1, coins=20, hidden_gem_count=2),),
            unknown_gems={BLUE: 3, GREEN: 3, PURPLE: 3, YELLOW: 3, PINK: 2},
        )
        snap = build_inference_snapshot(info)
        assert snap is not None

        beliefs = OpponentHandBeliefs()
        pre = beliefs.ensure_opponent(1, 2)
        # Only (2, 0, 0, 0, 0) and (0, 1, 0, 0, 1) have high weight;
        # (1, 0, 0, 0, 1) has lower weight than (0, 1, 0, 0, 1).
        for t in pre:
            if t == (2, 0, 0, 0, 0):
                pre[t] = 6.0
            elif t == (0, 1, 0, 0, 1):
                pre[t] = 4.0
            else:
                pre[t] = 0.5

        beliefs.update_from_reveal(
            opp_id=1,
            revealed_gem=PINK,
            snapshot=snap,
            new_unknown_counts=(3, 3, 3, 3, 1),
            opp_won_gems=_gem_map(),
            available_missions=(),
            reveal_strength=0.0,
        )
        after = beliefs.likelihoods[1]
        # (2, 0, 0, 0, 0) had PINK == 0, so (1, 0, 0, 0, 0) gets pre[(1,0,0,0,1)]=0.5
        # whereas (0, 1, 0, 0, 0) gets pre[(0, 1, 0, 0, 1)] = 4.0!
        self.assertGreater(after[(0, 1, 0, 0, 0)], after[(1, 0, 0, 0, 0)] * 4.0)


class SamplingAndSerializationTests(unittest.TestCase):
    def test_sampling_conserves_unknown_gems_and_shifts_in_inferred_direction(
        self,
    ) -> None:
        opps = (
            _opponent(1, coins=25, hidden_gem_count=3),
            _opponent(2, coins=25, hidden_gem_count=3),
        )
        unknown = {BLUE: 4, GREEN: 3, PURPLE: 3, YELLOW: 3, PINK: 3}
        info_a = _bot_info(
            chart=VALUE_CHARTS[0],
            opponents=opps,
            unknown_gems=unknown,
            auction_gems=(BLUE,),
        )
        snap_a = build_inference_snapshot(info_a)
        assert snap_a is not None
        beliefs_a = OpponentHandBeliefs()
        beliefs_a.update_from_bid(1, snap_a, bid=16, strength=0.6)

        info_b = _bot_info(
            chart=VALUE_CHARTS[1],
            opponents=opps,
            unknown_gems=unknown,
            auction_gems=(BLUE,),
        )
        snap_b = build_inference_snapshot(info_b)
        assert snap_b is not None
        beliefs_b = OpponentHandBeliefs()
        beliefs_b.update_from_bid(1, snap_b, bid=16, strength=0.6)

        rng_a = random.Random(42)
        rng_b = random.Random(42)
        blue_a = 0
        blue_b = 0
        trials = 400
        for _ in range(trials):
            hands_a = beliefs_a.sample_opponent_hands_with_beliefs(info_a, rng_a)
            hands_b = beliefs_b.sample_opponent_hands_with_beliefs(info_b, rng_b)

            for opp in opps:
                self.assertEqual(len(hands_a[opp.id]), opp.hidden_gem_count)
                self.assertEqual(len(hands_b[opp.id]), opp.hidden_gem_count)

            for gem in STANDARD_GEMS:
                total_a = sum(hand.count(gem) for hand in hands_a.values())
                total_b = sum(hand.count(gem) for hand in hands_b.values())
                self.assertLessEqual(total_a, unknown[gem])
                self.assertLessEqual(total_b, unknown[gem])

            blue_a += hands_a[1].count(BLUE)
            blue_b += hands_b[1].count(BLUE)

        self.assertGreater(blue_a / float(trials), blue_b / float(trials) + 0.15)

    def test_pickle_round_trip(self) -> None:
        info = _bot_info()
        snap = build_inference_snapshot(info)
        assert snap is not None
        beliefs = OpponentHandBeliefs()
        beliefs.update_from_bid(1, snap, bid=12)

        restored_snap: InferenceSnapshot = pickle.loads(pickle.dumps(snap))
        restored_beliefs: OpponentHandBeliefs = pickle.loads(pickle.dumps(beliefs))
        self.assertEqual(restored_snap, snap)
        self.assertEqual(restored_beliefs.likelihoods, beliefs.likelihoods)

        restored_beliefs.clear()
        self.assertEqual(restored_beliefs.likelihoods, {})


if __name__ == "__main__":
    unittest.main()
