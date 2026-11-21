"""GemLedgerBot: market-based gem pricing with an adaptive opponent bid ledger.

Each round follows three steps:

1. Valuation (`gem_ledger_pricing`):
   - Computes exact expected final chart values for each gem color using the
     hypergeometric distribution over unknown cards.
   - Prices coins above face value when remaining prizes outweigh available
     table coins, tapers that premium down as the last gems approach, and caps
     a coin's implied value so profitable late lots are not passed up.
   - Estimates expected mission gains using Poisson completion odds adjusted
     for rivals' chances of claiming the same mission first.
2. Opponent ledger (`gem_ledger_book`):
   - Maintains a recency-weighted bid book per seat (reset each game) and
     across the table (carried across games), dropping the opening prior for a
     seat once enough live bids of that auction type have been observed.
   - Picks the affordable integer bid that maximizes expected net gain under
     the game's tie-break priority.
3. Strategic reveal (`select_reveal_index`):
   - Reveals the hidden gem that best shifts future gem valuations downward
     for colors we still need on public missions and upward for colors the
     wealthiest opponent still needs.
"""

from __future__ import annotations

from collections.abc import Mapping

from bot_definitions.gem_ledger_book import (
    DEFAULT_BOOK_CONFIG,
    OpponentBidBook,
    choose_optimal_bid,
)
from bot_definitions.gem_ledger_pricing import (
    DEFAULT_PRICING,
    PricingConfig,
    build_market_context,
    evaluate_auction_terms,
    select_reveal_index,
)
from bots import Bot, BotInfo, RoundResult


class GemLedgerBot(Bot):
    """Best-response bidder driven by exact gem expectations and a live bid ledger."""

    def __init__(self) -> None:
        self.pricing: PricingConfig = DEFAULT_PRICING
        self.book: OpponentBidBook = OpponentBidBook(DEFAULT_BOOK_CONFIG)

    def on_game_start(self) -> None:
        self.book.start_new_game()

    def get_bid(self, info: BotInfo) -> int:
        card = info.current_auction
        if card is None:
            return 0
        market = build_market_context(info)
        terms = evaluate_auction_terms(info, card, market, self.pricing)
        self.book.mark_round_open(terms.reference_scale, info.other_players)
        histograms = [
            self.book.opponent_histogram(opp, card, terms.reference_scale)
            for opp in info.other_players
        ]
        return choose_optimal_bid(info, terms, histograms)

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        return select_reveal_index(info, self.pricing)

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        self.book.commit_round(result.auction_card.type, result.bids)
