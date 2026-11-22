"""GemLedgerA5Bot: GemLedger tuned for chart A with 5 players.

GemLedgerBot already beats most submitted bots on chart A with 5 players.
YaleBot is the holdout. At that table everyone starts with only 20 coins, and
chart A pays ordinary gem values. GemLedger's usual coin price is a bit too
high there: scarcity gain 6 and a hard cap of 2 points make it save coins that
Yale spends on the good lots.

This bot keeps GemLedger's gem values, mission model, bid ledger, and reveal
rule. On chart A with 5 players only, it uses a lower scarcity gain and a
higher coin-value cap, closer to YaleBot's money model. Every other chart and
table size keeps GemLedger's normal prices, so UltimateBot can call this class
only for that one combination.

UltimateBot is the tournament entry. This class is one of the specialists
it calls. It is not a separate submission.

Contributors:
    Micah
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace

from bot_definitions.gem_ledger_bot import GemLedgerBot
from bot_definitions.gem_ledger_pricing import PricingConfig
from bots import BotInfo


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


class GemLedgerA5Bot(GemLedgerBot):
    """GemLedger that spends a little more freely on chart A with 5 players."""

    # Table where YaleBot was beating ordinary GemLedger.
    TARGET_CHART = "A"
    TARGET_PLAYERS = 5
    # Ordinary GemLedger uses 6.0 and a hard cap of 2.0. These numbers were
    # chosen by a local search against the other submitted bots on this table.
    A5_SCARCITY_GAIN = 5.0
    A5_MAX_COIN_VALUE = 2.5
    # 1.0 means the endgame taper leaves the scarcity premium alone.
    A5_ENDGAME_PREMIUM_FLOOR = 1.0

    def __init__(self) -> None:
        super().__init__()
        self._normal_pricing = self.pricing
        # GL_A5_* lets a search script try other numbers without editing this file.
        self._a5_pricing = replace(
            self.pricing,
            coin_scarcity_gain=_env_float("GL_A5_SCARCITY", self.A5_SCARCITY_GAIN),
            max_coin_value=_env_float("GL_A5_MAX_COIN", self.A5_MAX_COIN_VALUE),
            endgame_premium_floor=_env_float(
                "GL_A5_ENDGAME_FLOOR", self.A5_ENDGAME_PREMIUM_FLOOR
            ),
        )

    def _pricing_for(self, info: BotInfo) -> PricingConfig:
        """Return the coin price for this chart and table size."""
        if (
            info.value_chart.name == self.TARGET_CHART
            and info.num_players == self.TARGET_PLAYERS
        ):
            return self._a5_pricing
        return self._normal_pricing

    def get_bid(self, info: BotInfo) -> int:
        self.pricing = self._pricing_for(info)
        return super().get_bid(info)

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        self.pricing = self._pricing_for(info)
        return super().choose_hidden_gem(info, bids)
