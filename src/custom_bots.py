"""Public bot names for the CLI; add imports and __all__ entries for new bots."""

from bot_definitions.small_bid_bot import SmallBidBot

from bot_definitions.adaptive_finance_bots import (
    ConservativeAdaptiveBot,
    FinanceAwareAdaptiveBot,
    LoanAwareAdaptiveBot,
)

__all__ = [
    "SmallBidBot",
    "ConservativeAdaptiveBot",
    "LoanAwareAdaptiveBot",
    "FinanceAwareAdaptiveBot",
]
