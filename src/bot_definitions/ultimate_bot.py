"""UltimateBot: play a different specialist for each chart and table size.

This is the tournament entry. MicahBot, GemLedgerBot, and GemLedgerReserveBot
live in this folder as the specialists it calls. They are not separate entries.

Before the first decision, the bot reads the value chart and the player count
and hands the rest of the game to the specialist that scored the most points
on that kind of table:

- MicahBot prices each gem from the chart, learns how opponents bid, and
  spends extra only late, when blocking someone can change first place.
  It plays chart A with 3 players, charts B and C, and chart E with 3 players.
- GemLedgerBot uses those gem prices, keeps a bid ledger for each seat, and
  chooses which hidden gem to reveal. It plays chart A with 4 or 5 players
  and chart D with 3 players.
- GemLedgerReserveBot is GemLedger on the cheaper charts. On charts D and E
  it saves coins: an investment is no longer treated as free, and one coin
  can be worth up to 2.5 points. It plays chart D with 4 or 5 players and
  chart E with 4 or 5 players.

Only the chosen specialist is kept for that game. GemLedger's table-wide bid
notes can carry into the next game when GemLedger is chosen again. Switching
to a different specialist starts that one fresh.

The table below comes from local games. If a chart shows up with some other
player count, the chart's usual specialist is used, and otherwise MicahBot.

Contributors:
    Micah
"""

from __future__ import annotations

from collections.abc import Mapping

from bot_definitions.gem_ledger_bot import GemLedgerBot
from bot_definitions.gem_ledger_reserve_bot import GemLedgerReserveBot
from bot_definitions.micah_bot import MicahBot
from bots import Bot, BotInfo, GameSummary, RoundResult

MICAH = "micah"
LEDGER = "ledger"
RESERVE = "reserve"

DELEGATE_CLASSES: dict[str, type[Bot]] = {
    MICAH: MicahBot,
    LEDGER: GemLedgerBot,
    RESERVE: GemLedgerReserveBot,
}

# Best specialist per (value chart, player count), measured in local games.
DELEGATE_BY_COMBO: dict[tuple[str, int], str] = {
    ("A", 3): MICAH,
    ("A", 4): LEDGER,
    ("A", 5): LEDGER,
    ("B", 3): MICAH,
    ("B", 4): MICAH,
    ("B", 5): MICAH,
    ("C", 3): MICAH,
    ("C", 4): MICAH,
    ("C", 5): MICAH,
    ("D", 3): LEDGER,
    ("D", 4): RESERVE,
    ("D", 5): RESERVE,
    ("E", 3): MICAH,
    ("E", 4): RESERVE,
    ("E", 5): RESERVE,
}

# Fallback when a chart appears without a player-count entry.
DELEGATE_BY_CHART: dict[str, str] = {
    "A": MICAH,
    "B": MICAH,
    "C": MICAH,
    "D": RESERVE,
    "E": RESERVE,
}

DEFAULT_DELEGATE = MICAH


class UltimateBot(Bot):
    """Hand each game to the specialist that scores the most on that table."""

    def __init__(self) -> None:
        self.delegate: Bot | None = None
        self.delegate_name: str | None = None
        self.needs_start = True

    def _select(self, info: BotInfo) -> Bot:
        chart = info.value_chart.name
        name = DELEGATE_BY_COMBO.get(
            (chart, info.num_players),
            DELEGATE_BY_CHART.get(chart, DEFAULT_DELEGATE),
        )
        if self.delegate is None or self.delegate_name != name:
            self.delegate = DELEGATE_CLASSES[name]()
            self.delegate_name = name
            self.needs_start = True
        if self.needs_start:
            # The first callback is the first point where chart and table size
            # are known, so the delegate's game reset happens here.
            self.delegate.on_game_start()
            self.needs_start = False
        return self.delegate

    def on_game_start(self) -> None:
        # Keep the previous delegate so GemLedger's cross-game table memory
        # survives, but force its per-game reset on the next decision.
        self.needs_start = True

    def get_bid(self, info: BotInfo) -> int:
        return self._select(info).get_bid(info)

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        return self._select(info).choose_hidden_gem(info, bids)

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        self._select(info).on_round_end(info, result)

    def on_game_end(self, info: BotInfo, result: GameSummary) -> None:
        self._select(info).on_game_end(info, result)
