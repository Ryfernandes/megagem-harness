"""Unit tests for the MicahBot opponent bid model."""

from __future__ import annotations

import pickle
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bot_definitions.micah_bid_model import (  # noqa: E402
    AUCTION_CODE,
    INVEST_CODE,
    LOAN_CODE,
    OpponentBidModel,
    auction_code,
    coin_bucket,
    covering_bid,
    lot_reference_value,
    phase_is_late,
    probability_we_win,
    winner_probabilities,
)
from bot_definitions.micah_strategy import (  # noqa: E402
    beats_on_tie,
    estimate_rounds_left_after,
    tie_priority_rank,
)
from bots import BotInfo, PlayerInfo, SelfInfo  # noqa: E402
from constants import STANDARD_GEMS, AuctionType, GemType  # noqa: E402
from decks import VALUE_CHARTS  # noqa: E402
from immutable import FrozenMap  # noqa: E402
from lib import AuctionCard, MissionCard  # noqa: E402

BLUE = GemType.BLUE_CIRCLE
YELLOW = GemType.YELLOW_SQUARE
GREEN = GemType.GREEN_OVAL


def _mean(dist: tuple[tuple[int, float], ...]) -> float:
    return sum(bid * prob for bid, prob in dist)


def _player(
    player_id: int,
    coins: int = 10,
    won: dict[GemType, int] | None = None,
) -> PlayerInfo:
    counts = dict.fromkeys(STANDARD_GEMS, 0)
    if won is not None:
        counts.update(won)
    return PlayerInfo(
        id=player_id,
        coins=coins,
        hidden_gem_count=0,
        revealed_gems=(),
        won_gems=FrozenMap(counts),
        auction_cards=(),
        mission_cards=(),
        invested_coins=0,
    )


def _info(
    *,
    opponent: PlayerInfo | None = None,
    current_auction: AuctionCard | None = None,
    auction_gems: tuple[GemType, ...] = (),
    gems_remaining: int = 0,
    available_missions: tuple[MissionCard, ...] = (),
    auction_cards_remaining: dict[AuctionCard, int] | None = None,
    previous_winner_id: int | None = None,
) -> BotInfo:
    other = opponent if opponent is not None else _player(1)
    remaining = FrozenMap(auction_cards_remaining or {})
    empty_counts = FrozenMap(dict.fromkeys(STANDARD_GEMS, 0))
    return BotInfo(
        self_player=SelfInfo(
            id=0,
            coins=20,
            hidden_gem_count=0,
            revealed_gems=(),
            won_gems=empty_counts,
            auction_cards=(),
            mission_cards=(),
            invested_coins=0,
            hidden_gems=(),
        ),
        other_players=(other,),
        num_players=2,
        round_number=1,
        current_auction=current_auction,
        gem_deck_count=gems_remaining,
        gems_remaining=gems_remaining,
        unknown_gems=empty_counts,
        auction_gems=auction_gems,
        value_display=empty_counts,
        previous_winner_id=previous_winner_id,
        value_chart=VALUE_CHARTS[0],
        available_missions=available_missions,
        auction_deck_count=sum(remaining.values()),
        auction_cards_remaining=remaining,
    )


class BidDistributionTests(unittest.TestCase):
    def test_bid_distribution_never_exceeds_legal_cap(self) -> None:
        model = OpponentBidModel()
        piled = model.bid_distribution(
            opponent_id=1,
            auction_type=AuctionType.AUCTION,
            phase_late=False,
            coins=12,
            reference=1_000_000_000.0,
            legal_cap=4,
        )
        self.assertAlmostEqual(sum(prob for _, prob in piled), 1.0, places=9)
        self.assertEqual(max(bid for bid, _ in piled), 4)
        for bid, prob in piled:
            self.assertIs(type(bid), int)
            self.assertIs(type(prob), float)
            self.assertGreaterEqual(bid, 0)
            self.assertLessEqual(bid, 4)

        # Center wants about 0.78*20, which the cap of 15 cuts off.
        partial = model.bid_distribution(1, AuctionType.AUCTION, False, 12, 20.0, 15)
        self.assertEqual(max(bid for bid, _ in partial), 15)
        self.assertLess(min(bid for bid, _ in partial), 15)
        self.assertTrue(all(0 <= bid <= 15 for bid, _ in partial))
        self.assertAlmostEqual(sum(prob for _, prob in partial), 1.0, places=9)
        self.assertEqual(
            model.bid_distribution(1, AuctionType.AUCTION, False, 0, 20.0, 0),
            ((0, 1.0),),
        )
        self.assertEqual(
            model.bid_distribution(1, AuctionType.LOAN, False, 0, 20.0, -3),
            ((0, 1.0),),
        )

    def test_treasure_prior_is_not_a_single_spike(self) -> None:
        model = OpponentBidModel()
        treasure = model.bid_distribution(1, AuctionType.AUCTION, False, 10, 20.0, 30)
        self.assertGreaterEqual(len(treasure), 2)
        self.assertAlmostEqual(_mean(treasure), 0.78 * 20.0, delta=1.5)
        invest = model.bid_distribution(1, AuctionType.INVEST, False, 10, 20.0, 30)
        self.assertGreaterEqual(len(invest), 2)
        self.assertAlmostEqual(_mean(invest), 0.75 * 20.0, delta=1.5)
        loan = model.bid_distribution(1, AuctionType.LOAN, False, 10, 20.0, 30)
        self.assertGreaterEqual(len(loan), 2)
        self.assertGreater(_mean(loan), 0.25 * 20.0)
        self.assertLess(_mean(loan), 0.50 * 20.0)

    def test_low_reference_still_maps_recorded_ratios(self) -> None:
        model = OpponentBidModel()
        model.record(1, AuctionType.AUCTION, False, 10, 4.0, 4)
        dist = model.bid_distribution(1, AuctionType.AUCTION, False, 10, 1.0, 10)
        weights = dict(dist)
        self.assertTrue(all(0 <= bid <= 10 for bid in weights))
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
        empty = OpponentBidModel().bid_distribution(
            1, AuctionType.AUCTION, False, 10, 1.0, 10
        )
        self.assertGreaterEqual(_mean(dist), _mean(empty) - 1e-9)

    def test_same_cell_records_raise_the_mean_bid(self) -> None:
        query = (1, AuctionType.AUCTION, False, 10, 20.0, 50)
        fresh = OpponentBidModel()
        prior = fresh.bid_distribution(*query)
        self.assertAlmostEqual(_mean(prior), 0.78 * 20.0, delta=1.5)

        learned = OpponentBidModel()
        for _ in range(5):
            learned.record(1, AuctionType.AUCTION, False, 10, 20.0, 40)
        after = learned.bid_distribution(*query)
        self.assertGreater(_mean(after), _mean(prior) + 5.0)
        self.assertEqual(
            learned.observations,
            [(1, AUCTION_CODE, 0, coin_bucket(10), 2000)] * 5,
        )

        loose = OpponentBidModel()
        loose.record(1, AuctionType.AUCTION, True, 10, 20.0, 40)
        loose_mean = _mean(loose.bid_distribution(*query))
        self.assertGreater(loose_mean, _mean(prior))
        self.assertGreater(loose_mean, _mean(prior))
        self.assertLess(loose_mean, _mean(after))

        ignored = OpponentBidModel()
        ignored.record(7, AuctionType.AUCTION, False, 10, 20.0, 40)
        ignored.record(1, AuctionType.INVEST, False, 10, 20.0, 40)
        self.assertEqual(ignored.bid_distribution(*query), prior)

    def test_record_skips_reference_below_two(self) -> None:
        model = OpponentBidModel()
        model.record(1, AuctionType.AUCTION, False, 10, 1.99, 5)
        model.record(1, AuctionType.AUCTION, False, 10, 4.0, -1)
        self.assertEqual(model.observations, [])

        model.record(1, AuctionType.AUCTION, False, 10, 2.0, 0)
        self.assertEqual(model.observations, [(1, AUCTION_CODE, 0, 1, 0)])
        model.record(3, AuctionType.LOAN, True, 21, 2.0, 30)
        self.assertEqual(model.observations[-1], (3, LOAN_CODE, 1, 2, 5000))

        model.clear()
        self.assertEqual(model.observations, [])
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.observations, [])
        restored.record(1, AuctionType.INVEST, False, 9, 2.0, 1)
        self.assertEqual(restored.observations, [(1, INVEST_CODE, 0, 1, 500)])
        self.assertEqual(model.observations, [])


class WinProbabilityTests(unittest.TestCase):
    def test_probability_we_win_is_one_above_every_cap(self) -> None:
        distributions = {1: ((6, 1.0),), 2: ((9, 1.0),)}
        self.assertEqual(
            probability_we_win(
                10, 0, distributions, previous_winner_id=1, num_players=3
            ),
            1.0,
        )
        # Independent 50/50 chances multiply; they are not a min or an average.
        split = {
            1: ((0, 0.5), (10, 0.5)),
            2: ((0, 0.5), (10, 0.5)),
        }
        self.assertAlmostEqual(
            probability_we_win(5, 0, split, previous_winner_id=1, num_players=3),
            0.25,
            places=9,
        )
        self.assertEqual(probability_we_win(3, 0, {}, None, 3), 1.0)

    def test_probability_we_win_is_zero_when_we_lose_the_tie(self) -> None:
        num_players = 3
        previous_winner_id = 0
        our_id = 0
        opponent_id = 1
        our_rank = tie_priority_rank(our_id, previous_winner_id, num_players)
        their_rank = tie_priority_rank(opponent_id, previous_winner_id, num_players)
        self.assertIsNotNone(our_rank)
        self.assertIsNotNone(their_rank)
        assert our_rank is not None and their_rank is not None
        self.assertLess(their_rank, our_rank)
        self.assertIs(
            beats_on_tie(our_id, opponent_id, previous_winner_id, num_players),
            False,
        )
        self.assertEqual(
            probability_we_win(
                4,
                our_id,
                {opponent_id: ((4, 1.0),)},
                previous_winner_id,
                num_players,
            ),
            0.0,
        )
        # The seat that wins the tie takes the equal bid for sure.
        self.assertIs(beats_on_tie(1, 0, 0, 3), True)
        self.assertEqual(probability_we_win(4, 1, {0: ((4, 1.0),)}, 0, 3), 1.0)
        # Round 1 has no previous winner, so an equal bid is a coin flip.
        self.assertIsNone(beats_on_tie(0, 1, None, 3))
        self.assertAlmostEqual(
            probability_we_win(4, 0, {1: ((4, 1.0),)}, None, 3),
            0.5,
            places=9,
        )
        # Multiway Round-1 ties split uniformly: 1/(k+1) across k tied rivals.
        self.assertAlmostEqual(
            probability_we_win(5, 0, {1: ((5, 1.0),), 2: ((5, 1.0),)}, None, 3),
            1.0 / 3.0,
            places=9,
        )
        self.assertAlmostEqual(
            probability_we_win(
                5,
                0,
                {1: ((5, 1.0),), 2: ((5, 1.0),), 3: ((5, 1.0),), 4: ((5, 1.0),)},
                None,
                5,
            ),
            0.2,
            places=9,
        )

    def test_winner_probabilities_sum_to_one(self) -> None:
        above = winner_probabilities(
            0,
            9,
            {1: ((4, 1.0),), 2: ((3, 0.25), (8, 0.75))},
            1,
            3,
        )
        self.assertAlmostEqual(sum(above.values()), 1.0, places=9)
        self.assertAlmostEqual(above[0], 1.0, places=9)
        self.assertEqual(above[1], 0.0)
        self.assertEqual(above[2], 0.0)

        split = winner_probabilities(0, 5, {1: ((5, 1.0),)}, None, 3)
        self.assertAlmostEqual(sum(split.values()), 1.0, places=9)
        self.assertAlmostEqual(split[0], 0.5, places=9)
        self.assertAlmostEqual(split[1], 0.5, places=9)

        # Ranks with previous winner 0: seat 1, then seat 2, then seat 0.
        ranked = winner_probabilities(0, 5, {1: ((5, 1.0),), 2: ((5, 1.0),)}, 0, 3)
        self.assertAlmostEqual(sum(ranked.values()), 1.0, places=9)
        self.assertAlmostEqual(ranked[1], 1.0, places=9)
        self.assertEqual(ranked[0], 0.0)
        self.assertEqual(ranked[2], 0.0)

        mixed = winner_probabilities(
            0,
            5,
            {1: ((0, 0.5), (5, 0.5)), 2: ((0, 1.0),)},
            0,
            3,
        )
        self.assertAlmostEqual(sum(mixed.values()), 1.0, places=9)
        self.assertAlmostEqual(mixed[0], 0.5, places=9)
        self.assertAlmostEqual(mixed[1], 0.5, places=9)
        self.assertEqual(mixed[2], 0.0)

        empty = winner_probabilities(0, 1, {1: ()}, 0, 3)
        self.assertAlmostEqual(sum(empty.values()), 1.0, places=9)
        self.assertAlmostEqual(empty[0], 1.0, places=9)
        tied_zero = winner_probabilities(0, 0, {1: ()}, 0, 3)
        self.assertAlmostEqual(sum(tied_zero.values()), 1.0, places=9)
        self.assertAlmostEqual(tied_zero[1], 1.0, places=9)
        self.assertEqual(tied_zero[0], 0.0)


class CoveringBidTests(unittest.TestCase):
    def test_covering_bid_tie_priority_and_cap(self) -> None:
        # Seat 1 beats seat 0 on a tie, so bidding the opponent's cap is enough.
        self.assertIs(beats_on_tie(1, 0, 0, 3), True)
        self.assertEqual(covering_bid(1, {0: 10}, 0, 3, our_cap=10), 10)
        self.assertIsNone(covering_bid(1, {0: 10}, 0, 3, our_cap=9))

        # Seat 0 loses that tie, so the sure bid is one past the cap.
        self.assertIs(beats_on_tie(0, 1, 0, 3), False)
        self.assertEqual(covering_bid(0, {1: 10}, 0, 3, our_cap=11), 11)
        self.assertIsNone(covering_bid(0, {1: 10}, 0, 3, our_cap=10))

        # Round 1 ties are a coin flip, so cap + 1 is required.
        self.assertIsNone(beats_on_tie(0, 1, None, 3))
        self.assertEqual(covering_bid(0, {1: 4}, None, 3, our_cap=5), 5)
        self.assertIsNone(covering_bid(0, {1: 4}, None, 3, our_cap=4))

        # Seat 2 beats seat 0 (need 5) and loses to seat 1 (need 6).
        self.assertEqual(covering_bid(2, {0: 5, 1: 5}, 0, 3, our_cap=6), 6)
        self.assertIsNone(covering_bid(2, {0: 5, 1: 5}, 0, 3, our_cap=5))
        self.assertEqual(covering_bid(0, {}, None, 3, our_cap=0), 0)


class ReferenceAndPhaseTests(unittest.TestCase):
    def test_auction_code_and_coin_bucket(self) -> None:
        self.assertEqual(auction_code(AuctionType.AUCTION), AUCTION_CODE)
        self.assertEqual(auction_code(AuctionType.INVEST), INVEST_CODE)
        self.assertEqual(auction_code(AuctionType.LOAN), LOAN_CODE)
        self.assertEqual((AUCTION_CODE, INVEST_CODE, LOAN_CODE), (0, 1, 2))
        self.assertEqual(coin_bucket(8), 0)
        self.assertEqual(coin_bucket(9), 1)
        self.assertEqual(coin_bucket(20), 1)
        self.assertEqual(coin_bucket(21), 2)

    def test_phase_is_late_follows_rounds_left(self) -> None:
        early = _info(gems_remaining=20)
        finished = _info(gems_remaining=0)
        self.assertGreater(estimate_rounds_left_after(early), 3.0)
        self.assertFalse(phase_is_late(early))
        self.assertLessEqual(estimate_rounds_left_after(finished), 3.0)
        self.assertTrue(phase_is_late(finished))

        exactly_three = _info(
            gems_remaining=3,
            auction_cards_remaining={AuctionCard(AuctionType.AUCTION, 1): 3},
        )
        just_after = _info(
            gems_remaining=4,
            auction_cards_remaining={AuctionCard(AuctionType.AUCTION, 1): 4},
        )
        self.assertAlmostEqual(estimate_rounds_left_after(exactly_three), 3.0, places=9)
        self.assertTrue(phase_is_late(exactly_three))
        self.assertGreater(estimate_rounds_left_after(just_after), 3.0)
        self.assertFalse(phase_is_late(just_after))

    def test_lot_reference_value(self) -> None:
        poor = _player(1)
        both = (MissionCard(8, {BLUE: 1, YELLOW: 1}),)
        treasure = _info(
            opponent=poor,
            current_auction=AuctionCard(AuctionType.AUCTION, 2),
            auction_gems=(BLUE, YELLOW, GREEN),
            available_missions=both,
        )
        evs = {BLUE: 4.0, YELLOW: 6.0, GREEN: 100.0}
        # First two gems only: 4 + 6, plus 0.55 * 8. The green gem is not in the lot.
        self.assertAlmostEqual(lot_reference_value(treasure, poor, evs), 14.4, places=9)

        rich = _player(1, won={BLUE: 1, YELLOW: 1})
        already = _info(
            opponent=rich,
            current_auction=AuctionCard(AuctionType.AUCTION, 2),
            auction_gems=(BLUE, YELLOW, GREEN),
            available_missions=both,
        )
        self.assertAlmostEqual(lot_reference_value(already, rich, evs), 10.0, places=9)

        one_gem = _info(
            current_auction=AuctionCard(AuctionType.AUCTION, 1),
            auction_gems=(BLUE, YELLOW),
            available_missions=(MissionCard(10, {BLUE: 3}),),
        )
        self.assertEqual(
            lot_reference_value(one_gem, poor, {BLUE: 4.0, YELLOW: 9.0}),
            4.0,
        )
        self.assertEqual(
            lot_reference_value(
                _info(
                    current_auction=AuctionCard(AuctionType.AUCTION, 1),
                    auction_gems=(BLUE,),
                ),
                poor,
                {},
            ),
            0.0,
        )
        self.assertEqual(
            lot_reference_value(
                _info(
                    current_auction=AuctionCard(AuctionType.AUCTION, 1),
                    auction_gems=(BLUE,),
                ),
                poor,
                {BLUE: -4.0},
            ),
            0.0,
        )
        self.assertEqual(lot_reference_value(_info(), poor, evs), 0.0)
        loan = _info(
            current_auction=AuctionCard(AuctionType.LOAN, 7),
            auction_gems=(BLUE,),
            available_missions=(MissionCard(8, {BLUE: 1}),),
        )
        self.assertEqual(lot_reference_value(loan, poor, {BLUE: 4.0}), 7.0)
        invest = _info(current_auction=AuctionCard(AuctionType.INVEST, 15))
        self.assertEqual(lot_reference_value(invest, poor, evs), 15.0)


if __name__ == "__main__":
    unittest.main()
