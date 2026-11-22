"""Bots the CLI can load by name.

This submission enters one tournament bot: UltimateBot. It calls MicahBot,
GemLedgerBot, GemLedgerReserveBot, and GemLedgerA5Bot. Those classes stay in
bot_definitions so UltimateBot can use them. They are not separate entries.

SmallBidBot is the example that came with the harness. It is not a
tournament entry.
"""

from bot_definitions.small_bid_bot import SmallBidBot
from bot_definitions.ultimate_bot import UltimateBot

__all__ = [
    "SmallBidBot",
    "UltimateBot",
]
