"""Value-aware stress opponents with different bidding styles.

Contributors: Grace Guo (85375651b), OpenAI Codex (implementation assistance).
These are opponent hypotheses, not claims about human behavior. All use L2's
chart/mission valuations and first-card reveals; only spending appetite changes.
Aggressive bids 110% of liquidity-adjusted value and holds a quarter of L2's
reserve. Passive bids 12% and holds 1.5 times its reserve. Random draws its bid
fraction uniformly from [0, 1.4] each round, retaining half the normal reserve.
Investments charge lost liquidity; loans use productive funding, cap bids below
principal and net them against proceeds. Every bid obeys the legal cash cap.
"""

from random import uniform

from bot_definitions.valuation_bot import FractionalValuationBot
from bots import BotInfo
from constants import AuctionType


class _StyleBidder(FractionalValuationBot):
    fraction = 0.45
    reserve_scale = 1.0

    def spending_fraction(self) -> float:
        return self.fraction

    def get_bid(self, info: BotInfo) -> int:
        card = info.current_auction
        if card is None:
            return 0
        estimate = self.evaluate(info)
        fraction = self.spending_fraction()
        cash = info.self_player.coins
        budget = max(0, cash - int(estimate.reserve * self.reserve_scale))
        premium = estimate.liquidity - 1
        if card.type is AuctionType.AUCTION:
            target = fraction * estimate.treasure / estimate.liquidity
        elif card.type is AuctionType.INVEST:
            target = fraction * card.value / max(self.investment_premium_floor, premium)
        else:
            target = (
                fraction
                * premium
                * min(card.value, estimate.productive_loan)
                / estimate.liquidity
            )
            budget = min(cash + card.value, card.value - 1)
        rival_limit = max(p.coins for p in info.other_players)
        if card.type is AuctionType.LOAN:
            rival_limit += card.value
        return max(0, min(int(target), budget, rival_limit + 1))


class RandomBidder(_StyleBidder):
    """Unpredictable value-bounded bids; not uniform over the entire bankroll."""

    reserve_scale = 0.5

    def spending_fraction(self) -> float:
        return uniform(0.0, 1.4)


class AggressiveBidder(_StyleBidder):
    """Large bids and a small reserve, accepting early overpayment risk."""

    fraction = 1.10
    reserve_scale = 0.25


class PassiveBidder(_StyleBidder):
    """Small positive bids with a substantial reserve; opportunistic purchases."""

    fraction = 0.12
    reserve_scale = 1.5
