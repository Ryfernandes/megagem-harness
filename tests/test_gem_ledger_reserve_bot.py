"""Coin-reserve pricing for GemLedgerReserveBot.

GemLedgerBot treats an investment bid as almost free, because those coins
come back at the end. GemLedgerReserveBot adds the lock-up cost, and on the
steep charts it lets a coin's price rise above GemLedgerBot's usual cap.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bot_definitions.gem_ledger_bot import GemLedgerBot
from bot_definitions.gem_ledger_pricing import (
    AuctionTerms,
    build_market_context,
    evaluate_auction_terms,
)
from bot_definitions.gem_ledger_reserve_bot import GemLedgerReserveBot
from constants import AuctionType
from lib import AuctionCard
from tests.test_micah_bot import _make_info


def _terms(bot: GemLedgerBot, chart_name: str, card: AuctionCard) -> AuctionTerms:
    info = _make_info(chart_name=chart_name, current_auction=card, coins=20)
    if isinstance(bot, GemLedgerReserveBot):
        bot.pricing = bot._pricing_for(info)
    market = build_market_context(info)
    return evaluate_auction_terms(info, card, market, bot.pricing)


class ReservePricingTests(unittest.TestCase):
    def test_gem_ledger_itself_does_not_charge_lockup(self) -> None:
        # The reserve is a third bot. GemLedgerBot's own price stays unchanged.
        self.assertEqual(GemLedgerBot().pricing.invest_lockup_cost, 0.0)

    def test_other_charts_match_gem_ledger_and_steep_charts_cost_more(self) -> None:
        card = AuctionCard(AuctionType.INVEST, 10)
        bot = GemLedgerReserveBot()
        for chart_name in ("A", "B", "C"):
            info = _make_info(chart_name=chart_name, current_auction=card)
            pricing = bot._pricing_for(info)
            ledger = GemLedgerBot().pricing
            self.assertEqual(pricing.max_coin_value, ledger.max_coin_value)
            self.assertEqual(pricing.coin_scarcity_gain, ledger.coin_scarcity_gain)
            self.assertEqual(pricing.invest_lockup_cost, 0.0)
            self.assertAlmostEqual(
                _terms(bot, chart_name, card).marginal_coin_cost,
                _terms(GemLedgerBot(), chart_name, card).marginal_coin_cost,
            )
        for chart_name in ("D", "E"):
            pricing = bot._pricing_for(
                _make_info(chart_name=chart_name, current_auction=card)
            )
            self.assertEqual(pricing.max_coin_value, bot.STEEP_MAX_COIN_VALUE)
            self.assertEqual(pricing.coin_scarcity_gain, bot.STEEP_SCARCITY_GAIN)
            self.assertEqual(pricing.invest_lockup_cost, bot.INVEST_LOCKUP_COST)
            self.assertGreater(
                _terms(bot, chart_name, card).marginal_coin_cost,
                _terms(GemLedgerBot(), chart_name, card).marginal_coin_cost,
            )

    def test_bid_stays_inside_the_coin_stack(self) -> None:
        bot = GemLedgerReserveBot()
        for chart_name in ("A", "D", "E"):
            for card in (
                AuctionCard(AuctionType.AUCTION, 2),
                AuctionCard(AuctionType.INVEST, 10),
                AuctionCard(AuctionType.LOAN, 10),
            ):
                info = _make_info(chart_name=chart_name, current_auction=card, coins=12)
                bid = bot.get_bid(info)
                self.assertIsInstance(bid, int)
                budget = 12 + (card.value if card.type is AuctionType.LOAN else 0)
                self.assertGreaterEqual(bid, 0)
                self.assertLessEqual(bid, budget)


if __name__ == "__main__":
    unittest.main()
