"""A deliberately simple, deterministic baseline for Megagem.

Strategy: bid floor(current coins / 5) on every treasure auction, regardless
of the gems, chart, or mission requirements. Bid zero on investments and loans.
Zero bids can still win (including financial cards) under the game's tie rules.
Reveal the first hidden gem; revealing cannot change terminal gem values.
The engine automatically claims every completed mission, without a payment or
consuming gems. This bot makes no additional mission-specific bid adjustment.

Contributors: Grace Guo (85375651b), OpenAI Codex (implementation assistance).
"""

from collections.abc import Mapping

from bots import Bot, BotInfo
from constants import AuctionType


class FixedRuleBot(Bot):
    """Spend a fixed 20% of liquid coins on treasure; zero on financial cards."""

    def get_bid(self, info: BotInfo) -> int:
        card = info.current_auction
        if card is not None and card.type is AuctionType.AUCTION:
            return info.self_player.coins // 5
        return 0

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        return 0
