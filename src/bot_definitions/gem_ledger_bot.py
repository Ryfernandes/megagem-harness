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

UltimateBot is the tournament entry. This class is one of the specialists
it calls. It is not a separate submission.

Contributors:
    Micah
"""

from __future__ import annotations

from collections.abc import Mapping

from bot_definitions.gem_ledger_book import (
    DEFAULT_BOOK_CONFIG,
    OpponentBidBook,
    book_from_env,
    choose_optimal_bid,
)
from bot_definitions.gem_ledger_pricing import (
    DEFAULT_PRICING,
    PricingConfig,
    build_market_context,
    evaluate_auction_terms,
    pricing_from_env,
    select_reveal_index,
)
from bots import Bot, BotInfo, RoundResult


def default_settings_environment() -> dict[str, str]:
    """Environment variables that reproduce today's GemLedgerBot numbers.

    A search script copies this dictionary, changes one value, and puts it in
    the environment before starting a game.
    """
    pricing = DEFAULT_PRICING
    book = DEFAULT_BOOK_CONFIG
    return {
        "GL_CFG_COIN_SCARCITY": str(pricing.coin_scarcity_gain),
        "GL_CFG_ENDGAME_GEMS": str(pricing.endgame_gem_threshold),
        "GL_CFG_ENDGAME_FLOOR": str(pricing.endgame_premium_floor),
        "GL_CFG_MAX_COIN": str(pricing.max_coin_value),
        "GL_CFG_RIVAL_CONTEST": str(pricing.rival_mission_contest),
        "GL_CFG_MISSION_DIVISOR": str(pricing.mission_urgency_divisor),
        "GL_CFG_INVEST_LOCKUP": str(pricing.invest_lockup_cost),
        "GL_CFG_OPENING_RATIOS": ",".join(str(ratio) for ratio in book.opening_ratios),
        "GL_CFG_OPENING_WEIGHT": str(book.opening_weight),
        "GL_CFG_OPENING_DROP": str(book.opening_drop_after),
        "GL_CFG_MAX_RECORDS": str(book.max_records),
        "GL_CFG_TABLE_WEIGHT": str(book.table_weight),
        "GL_CFG_SEAT_WEIGHT": str(book.seat_weight),
    }


class GemLedgerBot(Bot):
    """Best-response bidder driven by exact gem expectations and a live bid ledger."""

    def __init__(self) -> None:
        self.pricing: PricingConfig = pricing_from_env()
        self.book: OpponentBidBook = OpponentBidBook(book_from_env())

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
