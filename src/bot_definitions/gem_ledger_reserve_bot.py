"""GemLedgerReserveBot: GemLedger that saves coins on charts D and E.

GemLedgerBot already prices gems, missions, and opponent bids well. On charts
D and E it spends those coins too early. Two things cause that:

1. An investment bid is returned as points at the end, so GemLedger treats the
   bid as almost free and locks coins that later gem auctions needed.
2. GemLedger never prices one coin above 2 points. Charts D and E pay 11 to 20
   for a gem while few copies are showing, so a coin saved for the next lot is
   worth more than that cap.

This bot keeps GemLedger's gem values, mission model, bid ledger, and reveal
rule. On charts D and E only, a coin can be worth up to 2.5 points, and each
coin locked in an investment costs one extra point. Charts A, B, and C use
GemLedger's prices unchanged. Raising the cap further, or charging the
reserve on every chart, gave away lots that GemLedger was already winning.

UltimateBot is the tournament entry. This class is one of the specialists
it calls. It is not a separate submission.

Contributors:
    Micah
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from bot_definitions.gem_ledger_bot import GemLedgerBot
from bot_definitions.gem_ledger_pricing import PricingConfig
from bots import BotInfo


class GemLedgerReserveBot(GemLedgerBot):
    """GemLedger with a higher coin price on charts D and E."""

    # Charts where the expensive copies show up before the end of the game.
    STEEP_CHARTS = frozenset({"D", "E"})
    # GemLedger uses 6 and a cap of 2. These reach a slightly higher cap sooner.
    STEEP_SCARCITY_GAIN = 10.0
    STEEP_MAX_COIN_VALUE = 2.5
    # Extra points charged for each coin locked in an investment.
    INVEST_LOCKUP_COST = 1.0

    def __init__(self) -> None:
        super().__init__()
        self._normal_pricing = self.pricing
        self._steep_pricing = replace(
            self.pricing,
            coin_scarcity_gain=self.STEEP_SCARCITY_GAIN,
            max_coin_value=self.STEEP_MAX_COIN_VALUE,
            invest_lockup_cost=self.INVEST_LOCKUP_COST,
        )

    def _pricing_for(self, info: BotInfo) -> PricingConfig:
        """Return the coin price for this game's value chart."""
        if info.value_chart.name in self.STEEP_CHARTS:
            return self._steep_pricing
        return self._normal_pricing

    def get_bid(self, info: BotInfo) -> int:
        self.pricing = self._pricing_for(info)
        return super().get_bid(info)

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        self.pricing = self._pricing_for(info)
        return super().choose_hidden_gem(info, bids)
