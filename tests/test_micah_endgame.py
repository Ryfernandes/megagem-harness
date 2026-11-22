"""Unit tests for endgame denial and first-place sampling."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bot_definitions.micah_endgame import (
    denial_value,
    endgame_first_place_probs,
    sample_opponent_hidden_hands,
)
from bot_definitions.micah_strategy import estimate_rounds_left_after
from bots import BotInfo, PlayerInfo, SelfInfo
from constants import STANDARD_GEMS, AuctionType, GemType
from decks import VALUE_CHARTS
from immutable import FrozenMap
from lib import AuctionCard, MissionCard

BLUE, GREEN, PURPLE, YELLOW, PINK = STANDARD_GEMS
CHART_BY_NAME = {chart.name: chart for chart in VALUE_CHARTS}


def _opponent(
    player_id: int,
    coins: int,
    hidden_gem_count: int = 0,
) -> PlayerInfo:
    return PlayerInfo(
        id=player_id,
        coins=coins,
        hidden_gem_count=hidden_gem_count,
        revealed_gems=(),
        won_gems=FrozenMap({gem: 0 for gem in STANDARD_GEMS}),
        auction_cards=(),
        mission_cards=(),
        invested_coins=0,
    )


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


def _zero_unknown() -> dict[GemType, int]:
    return {gem: 0 for gem in STANDARD_GEMS}


def _flat_evs(value: float) -> dict[GemType, float]:
    return {gem: value for gem in STANDARD_GEMS}


def _late_treasure(our_coins: int, rival_coins: int) -> BotInfo:
    """One gem left after this treasure, so about one auction remains."""
    return _make_info(
        coins=our_coins,
        hidden_gems=(),
        others=(_opponent(1, rival_coins),),
        current_auction=AuctionCard(AuctionType.AUCTION, 1),
        gem_deck_count=0,
        unknown_gems=_zero_unknown(),
        auction_gems=(BLUE, YELLOW),
        value_display=_zero_unknown(),
        previous_winner_id=1,
        chart_name="A",
        auction_cards_remaining={AuctionCard(AuctionType.AUCTION, 1): 1},
    )


def _terminal_treasure(our_coins: int, rival_coins: int) -> BotInfo:
    """This treasure spends the last gem, so the game ends afterwards.

    Chart B pays 20 for a color that has not appeared on the display. With no
    hidden gems left to reveal, that price does not move between samples.
    """
    return _make_info(
        coins=our_coins,
        hidden_gems=(),
        others=(_opponent(1, rival_coins),),
        current_auction=AuctionCard(AuctionType.AUCTION, 1),
        gem_deck_count=0,
        unknown_gems=_zero_unknown(),
        auction_gems=(BLUE,),
        value_display=_zero_unknown(),
        previous_winner_id=1,
        chart_name="B",
        auction_cards_remaining={},
    )


class MicahEndgameTest(unittest.TestCase):
    def test_denial_is_zero_early_in_the_game(self) -> None:
        # A big deck and a full auction deck keep several rounds ahead of us.
        # The rival is richer, bids 2, and the lot is worth far more than 2.
        # Denial must still be 0 because this helper turns off that early.
        info = _make_info(
            coins=8,
            hidden_gems=(),
            others=(_opponent(1, 40),),
            current_auction=AuctionCard(AuctionType.AUCTION, 1),
            gem_deck_count=12,
            unknown_gems=_zero_unknown(),
            auction_gems=(BLUE, YELLOW),
            previous_winner_id=1,
        )
        self.assertGreater(estimate_rounds_left_after(info), 3.0)
        distributions = {1: ((2, 1.0),)}
        denial = denial_value(info, 50, distributions, _flat_evs(25.0))
        self.assertEqual(denial, 0.0)

    def test_late_denial_is_positive_when_a_cheap_lot_is_blocked(self) -> None:
        info = _late_treasure(our_coins=8, rival_coins=36)
        rounds_left = estimate_rounds_left_after(info)
        self.assertGreater(rounds_left, 0.0)
        self.assertLessEqual(rounds_left, 3.0)

        # They bid 2 for sure. Bidding 0 lets them take a lot worth 20.
        # Bidding 20 takes it away from them.
        distributions = {1: ((2, 1.0),)}
        public_evs = _flat_evs(20.0)
        self.assertEqual(denial_value(info, 0, distributions, public_evs), 0.0)
        self.assertGreater(denial_value(info, 20, distributions, public_evs), 0.0)

    def test_late_denial_is_zero_when_rival_overpays(self) -> None:
        info = _late_treasure(our_coins=8, rival_coins=40)
        rounds_left = estimate_rounds_left_after(info)
        self.assertGreater(rounds_left, 0.0)
        self.assertLessEqual(rounds_left, 3.0)

        # 34 is inside their 40 coins, and it is far above the lot's public
        # value of 1. Outbidding them blocks a purchase that does not help them.
        distributions = {1: ((34, 1.0),)}
        denial = denial_value(info, 39, distributions, _flat_evs(1.0))
        self.assertAlmostEqual(denial, 0.0, places=6)

    def test_first_place_probs_none_when_many_rounds_remain(self) -> None:
        info = _make_info(
            coins=8,
            hidden_gems=(),
            others=(_opponent(1, 40),),
            gem_deck_count=12,
            unknown_gems=_zero_unknown(),
        )
        self.assertGreater(estimate_rounds_left_after(info), 2.0)
        probs = endgame_first_place_probs(
            info,
            [0, 10],
            {1: ((2, 1.0),)},
            Random(0),
        )
        self.assertIsNone(probs)

    def test_terminal_outbid_raises_first_place_probability(self) -> None:
        # We have 12 coins and they have 10. Chart B prices the last blue gem
        # at 20. If they buy it for 4, their score becomes 26 and they pass us.
        # If we buy it for 5, our score becomes 27 and we stay ahead.
        info = _terminal_treasure(our_coins=12, rival_coins=10)
        self.assertEqual(estimate_rounds_left_after(info), 0.0)
        self.assertEqual(info.value_chart.get_value(0), 20)

        probs = endgame_first_place_probs(
            info,
            [0, 5],
            {1: ((4, 1.0),)},
            Random(0),
            samples=8,
        )
        self.assertIsNotNone(probs)
        assert probs is not None
        self.assertEqual(set(probs), {0, 5})
        for probability in probs.values():
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)
        self.assertGreater(probs[5], probs[0])
        self.assertEqual(probs[0], 0.0)
        self.assertEqual(probs[5], 1.0)

        # Id 1 is listed second, but hands are drawn in id order, so id 1
        # takes two of the three unknown blues and id 2 gets the last one.
        crowded = _make_info(
            hidden_gems=(),
            others=(
                _opponent(2, coins=5, hidden_gem_count=2),
                _opponent(1, coins=5, hidden_gem_count=2),
            ),
            unknown_gems={BLUE: 3},
            gem_deck_count=0,
            auction_gems=(YELLOW,),
        )
        for seed in range(12):
            hands = sample_opponent_hidden_hands(crowded, Random(seed))
            counts: Counter[GemType] = Counter()
            for hand in hands.values():
                counts.update(hand)
            self.assertLessEqual(counts[BLUE], 3)
            self.assertEqual(sum(counts.values()), 3)
            for gem in STANDARD_GEMS:
                if gem is not BLUE:
                    self.assertEqual(counts[gem], 0)
            self.assertEqual(len(hands[1]), 2)
            self.assertEqual(len(hands[2]), 1)

    def test_first_place_is_certain_when_lead_exceeds_lot(self) -> None:
        # 50 coins up, and the last gem is worth 20 even if they win it free.
        # Paying 1 for it leaves them at 19, which is still short of 50.
        info = _terminal_treasure(our_coins=50, rival_coins=0)
        self.assertEqual(estimate_rounds_left_after(info), 0.0)
        lot_value = info.value_chart.get_value(0)
        self.assertLess(lot_value, 50)

        probs = endgame_first_place_probs(
            info,
            [0, 6],
            {1: ((1, 1.0),)},
            Random(0),
            samples=6,
        )
        self.assertIsNotNone(probs)
        assert probs is not None
        self.assertEqual(set(probs), {0, 6})
        self.assertEqual(probs[0], 1.0)
        self.assertEqual(probs[6], 1.0)


if __name__ == "__main__":
    unittest.main()
