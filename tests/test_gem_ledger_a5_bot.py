"""Coin pricing for GemLedgerA5Bot on chart A with 5 players."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bot_definitions.gem_ledger_a5_bot import GemLedgerA5Bot
from bot_definitions.gem_ledger_bot import GemLedgerBot
from bot_definitions.gem_ledger_pricing import (
    AuctionTerms,
    build_market_context,
    evaluate_auction_terms,
)
from bots import PlayerInfo
from constants import STANDARD_GEMS, AuctionType
from immutable import FrozenMap
from lib import AuctionCard
from tests.test_micah_bot import _make_info


def _others(count: int) -> tuple[PlayerInfo, ...]:
    return tuple(
        PlayerInfo(
            id=index,
            coins=18,
            hidden_gem_count=3,
            revealed_gems=(),
            won_gems=FrozenMap({gem: 0 for gem in STANDARD_GEMS}),
            auction_cards=(),
            mission_cards=(),
            invested_coins=0,
        )
        for index in range(1, count + 1)
    )


def _info(chart_name: str, players: int, card: AuctionCard, coins: int = 20):
    return _make_info(
        chart_name=chart_name,
        current_auction=card,
        coins=coins,
        others=_others(players - 1),
    )


def _terms(bot: GemLedgerBot, chart_name: str, players: int, card: AuctionCard) -> AuctionTerms:
    info = _info(chart_name, players, card)
    if isinstance(bot, GemLedgerA5Bot):
        bot.pricing = bot._pricing_for(info)
    market = build_market_context(info)
    return evaluate_auction_terms(info, card, market, bot.pricing)


class A5PricingTests(unittest.TestCase):
    def test_other_tables_match_gem_ledger(self) -> None:
        card = AuctionCard(AuctionType.AUCTION, 2)
        bot = GemLedgerA5Bot()
        for chart_name, players in (("A", 3), ("A", 4), ("B", 5), ("D", 5)):
            info = _info(chart_name, players, card)
            pricing = bot._pricing_for(info)
            ledger = GemLedgerBot().pricing
            self.assertEqual(pricing.coin_scarcity_gain, ledger.coin_scarcity_gain)
            self.assertEqual(pricing.max_coin_value, ledger.max_coin_value)
            self.assertEqual(
                pricing.endgame_premium_floor, ledger.endgame_premium_floor
            )
            self.assertAlmostEqual(
                _terms(bot, chart_name, players, card).marginal_coin_cost,
                _terms(GemLedgerBot(), chart_name, players, card).marginal_coin_cost,
            )

    def test_chart_a_five_players_uses_a5_numbers(self) -> None:
        card = AuctionCard(AuctionType.AUCTION, 2)
        bot = GemLedgerA5Bot()
        pricing = bot._pricing_for(_info("A", 5, card))
        self.assertEqual(pricing.coin_scarcity_gain, bot.A5_SCARCITY_GAIN)
        self.assertEqual(pricing.max_coin_value, bot.A5_MAX_COIN_VALUE)
        self.assertEqual(pricing.endgame_premium_floor, bot.A5_ENDGAME_PREMIUM_FLOOR)

    def test_bid_stays_inside_the_coin_stack(self) -> None:
        bot = GemLedgerA5Bot()
        for card in (
            AuctionCard(AuctionType.AUCTION, 2),
            AuctionCard(AuctionType.INVEST, 10),
            AuctionCard(AuctionType.LOAN, 10),
        ):
            info = _info("A", 5, card, coins=12)
            bid = bot.get_bid(info)
            self.assertIsInstance(bid, int)
            budget = 12 + (card.value if card.type is AuctionType.LOAN else 0)
            self.assertGreaterEqual(bid, 0)
            self.assertLessEqual(bid, budget)


if __name__ == "__main__":
    unittest.main()
