"""Pinned, importable candidate configurations for the mixed-opponent experiment."""

from bot_definitions.hypothetical_bots import (
    AggressiveBidder,
    PassiveBidder,
    RandomBidder,
)
from bot_definitions.sampling_bot import SamplingBot
from bot_definitions.soft_tracking_bot import SoftTrackingBot
from bot_definitions.valuation_bot import ValuationBot
from bots import Bot


class L3I(SoftTrackingBot):
    use_bid_evidence = True


class L3F(SoftTrackingBot):
    use_finance_horizon = True


class L3R(SoftTrackingBot):
    use_strategic_reveal = True


class L3IF(L3I):
    use_finance_horizon = True


class L3IR(L3I):
    use_strategic_reveal = True


class L3FR(L3F):
    use_strategic_reveal = True


class L3IFR(L3IF):
    use_strategic_reveal = True


POLICIES: dict[str, type[Bot]] = {
    "L2": ValuationBot,
    "L3-I": L3I,
    "L3-F": L3F,
    "L3-R": L3R,
    "L3-IF": L3IF,
    "L3-IR": L3IR,
    "L3-FR": L3FR,
    "L3-IFR": L3IFR,
    "L4": SamplingBot,
    "random": RandomBidder,
    "aggressive": AggressiveBidder,
    "passive": PassiveBidder,
}
CANDIDATES = tuple(POLICIES)[:9]
SCENARIOS = {
    "level2": ("L2", "L2", "L2"),
    "level3": ("L3-I", "L3-F", "L3-R"),
    "random": ("random", "random", "random"),
    "aggressive": ("aggressive", "aggressive", "aggressive"),
    "passive": ("passive", "passive", "passive"),
    "mixed_styles": ("random", "aggressive", "passive"),
    "mixed_experts": ("L2", "L3-IFR", "L4"),
}
