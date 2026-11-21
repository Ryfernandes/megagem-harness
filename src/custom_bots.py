"""Bots the CLI can load by name.

This team submits three tournament bots, which is the sprint maximum:

- MicahBot
- Blue5GD

SmallBidBot is the example that came with the harness. It is not a
tournament entry.
"""

from bot_definitions.blue5gd_bot import Blue5GD
from bot_definitions.micah_bot import MicahBot
from bot_definitions.small_bid_bot import SmallBidBot

__all__ = [
    "Blue5GD",
    "MicahBot",
    "SmallBidBot",
]
