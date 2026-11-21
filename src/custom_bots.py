"""Bots the CLI can load by name.

This team submits two tournament bots:

- MicahBot
- GemLedgerBot

SmallBidBot is the example that came with the harness. It is not a
tournament entry.
"""

from bot_definitions.gem_ledger_bot import GemLedgerBot
from bot_definitions.micah_bot import MicahBot
from bot_definitions.small_bid_bot import SmallBidBot

__all__ = [
    "GemLedgerBot",
    "MicahBot",
    "SmallBidBot",
]
