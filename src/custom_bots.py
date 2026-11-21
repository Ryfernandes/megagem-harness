"""Bots the CLI can load by name.

This team submits MicahBot as a tournament bot.

SmallBidBot is the example that came with the harness. It is not a
tournament entry. Blue5GD is no longer an entry.
"""

from bot_definitions.micah_bot import MicahBot
from bot_definitions.small_bid_bot import SmallBidBot

__all__ = [
    "MicahBot",
    "SmallBidBot",
]
