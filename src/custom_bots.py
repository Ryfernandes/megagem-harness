"""Sprint submission API: only SamplingBot (L4) and ValuationBot (L2).

The names in __all__ are the two intended tournament submissions. Baselines,
level-3 experiments and hypothetical opponents remain importable for local use
and are listed separately in EXPERIMENTAL_BOTS for the simulation viewer.
"""

from bot_definitions.fixed_rule_bot import FixedRuleBot
from bot_definitions.hypothetical_bots import (
    AggressiveBidder,
    PassiveBidder,
    RandomBidder,
)
from bot_definitions.sampling_bot import SamplingBot
from bot_definitions.small_bid_bot import SmallBidBot
from bot_definitions.soft_tracking_bot import SoftTrackingBot
from bot_definitions.valuation_bot import ValuationBot

__all__ = [
    "SamplingBot",
    "ValuationBot",
]

# Development opponents and controls, deliberately excluded from submissions.
# Keep explicit imports working for existing CLI commands and experiments.
EXPERIMENTAL_BOTS = {
    "FixedRuleBot": FixedRuleBot,
    "SmallBidBot": SmallBidBot,
    "SoftTrackingBot": SoftTrackingBot,
    "RandomBidder": RandomBidder,
    "AggressiveBidder": AggressiveBidder,
    "PassiveBidder": PassiveBidder,
}
