"""A fast, deterministic valuation-and-liquidity baseline for Megagem.

Submission role: L2 passive-opponent specialist, exported as
custom_bots:ValuationBot. This is the ordinary L2 policy, not PassiveBidder.
Contributors: Grace Guo (85375651b), OpenAI Codex (implementation assistance).

Gem prices integrate the Value Chart over an exact hypergeometric distribution:
public reveals and our hand fix part of the terminal display, while unseen gems
are split between opponents' hands and the auction deck. This matters especially
on nonlinear Chart E. Revealing a gem cannot change terminal prices.

Treasure bids maximize p(win) * (gem EV + mission/denial value - (1+k)*bid),
using the shared expected-utility bidder at every supported player count.
The cash penalty k adds a tapering 0.5 future-auction premium to the existing
liquidity premium. Opponent bid rates learn softly from public treasure bids;
their bid distributions respect cash and the game's tie priority. Unfinished
missions use an approximate completion probability, missing-gem acquisition cost,
and competing-player discount. All wildcard patterns are supported. A mission
completed immediately is worth its actual reward. Gems are reusable under the
rules, so there is no invented consumption or exclusivity penalty.

A reserve holds back 40% of cash while several treasure lots remain, tapering
near the end, with at least one coin retained before the final treasure lot.
Investments return the bid, so only its liquidity premium is charged against the
bonus. Loans' principal cancels against repayment: their benefit is additional
productive liquidity, minus the bid and its lost purchasing power. Borrow only
against remaining opportunities and never bid away all the loan proceeds.

This is an analytic valuation heuristic with a learned public-bid model.
Mission probabilities approximate independent color acquisitions; market prices
and race discounts are deliberately conservative estimates. No opponent hidden
hands, deck order, hand-content inference, external libraries, or randomness are
used. FractionalValuationBot preserves the former policy for matched comparisons;
it is not an additional public submission. Investments/loans and reveals retain
the former policy. Set use_expected_utility=False to restore fractional bids.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations_with_replacement
from math import ceil, comb, prod

from bot_definitions.expected_utility import (
    bid_distribution,
    completion_bonus,
    expected_utility,
    opportunity_cost,
    update_bid_rate,
    win_probability,
)
from bots import Bot, BotInfo, GameSummary, PlayerInfo, RoundResult
from constants import STANDARD_GEMS, AuctionType, GemType, WildCardRequirement
from lib import MissionCard


@lru_cache(maxsize=64)
def _targets(
    fixed: tuple[int, ...], wild: int, pattern: WildCardRequirement | None
) -> tuple[tuple[int, ...], ...]:
    """Expand a wildcard mission into all minimal satisfying collections."""
    targets = []
    for allocation in combinations_with_replacement(range(5), wild):
        counts = tuple(allocation.count(i) for i in range(5))
        if pattern is WildCardRequirement.SAME and max(counts) < wild:
            continue
        if pattern is WildCardRequirement.DIFFERENT and max(counts) > 1:
            continue
        if pattern is WildCardRequirement.PAIRS and any(c % 2 for c in counts):
            continue
        targets.append(tuple(a + b for a, b in zip(fixed, counts)))
    return tuple(targets)


def _mission_targets(mission: MissionCard) -> tuple[tuple[int, ...], ...]:
    return _targets(
        tuple(mission.gem_requirements.get(g, 0) for g in STANDARD_GEMS),
        mission.gem_requirements.get(GemType.WILD_CARD, 0),
        mission.wild_card_requirement,
    )


def _missing(mission: MissionCard, collection: Mapping[GemType, int]) -> int:
    return _missing_counts(mission, tuple(collection.get(g, 0) for g in STANDARD_GEMS))


@lru_cache(maxsize=8192)
def _missing_counts(mission: MissionCard, collection: tuple[int, ...]) -> int:
    return min(
        sum(max(0, n - held) for held, n in zip(collection, target))
        for target in _mission_targets(mission)
    )


@lru_cache(maxsize=8192)
def _acquisition_probability(
    needed: int, visible: int, unknown: int, share: float, deck_fraction: float
) -> float:
    """Tail probability for visible cards and unseen cards acquired in future.

    Unseen cards must both be in the deck and be won. Independence between
    different colors is an approximation, used for missions only.
    """
    if needed <= 0:
        return 1.0
    if needed > visible + unknown:
        return 0.0
    unseen_share = share * deck_fraction
    return sum(
        comb(visible, a)
        * share**a
        * (1 - share) ** (visible - a)
        * comb(unknown, b)
        * unseen_share**b
        * (1 - unseen_share) ** (unknown - b)
        for a in range(visible + 1)
        for b in range(unknown + 1)
        if a + b >= needed
    )


@dataclass(frozen=True)
class AuctionValuation:
    """Inspectable estimates in end-game points, before the bid is discounted."""

    gems: float
    missions: float
    denial: float
    liquidity: float
    reserve: int
    productive_loan: float
    future_points: float

    @property
    def treasure(self) -> float:
        return self.gems + self.missions + self.denial


class FractionalValuationBot(Bot):
    """Frozen former valuation policy, kept out of custom_bots.__all__."""

    bid_fraction = 0.45
    # Even a calm market has unmodelled purchasing opportunities. This floor
    # prevents the cheap-looking early investments from absorbing the bankroll.
    investment_premium_floor = 0.70

    @staticmethod
    def expected_gem_values(info: BotInfo) -> dict[GemType, float]:
        hand = Counter(info.self_player.hidden_gems)
        population = sum(info.unknown_gems.values())
        hidden = sum(p.hidden_gem_count for p in info.other_players)
        denominator = comb(population, hidden)
        values = {}
        for gem in STANDARD_GEMS:
            known = info.value_display.get(gem, 0) + hand[gem]
            unseen = info.unknown_gems.get(gem, 0)
            values[gem] = sum(
                info.value_chart.get_value(known + count)
                * comb(unseen, count)
                * comb(population - unseen, hidden - count)
                / denominator
                for count in range(
                    max(0, hidden - population + unseen), min(unseen, hidden) + 1
                )
            )
        return values

    def gem_values(self, info: BotInfo) -> dict[GemType, float]:
        """Price hook for valuation policies with additional information."""
        return self.expected_gem_values(info)

    def deck_fraction(self, info: BotInfo, gem: GemType) -> float:
        """Chance that an unseen copy of this color belongs to the deck."""
        return info.gem_deck_count / max(1, sum(info.unknown_gems.values()))

    def _mission_potential(
        self,
        info: BotInfo,
        mission: MissionCard,
        player: PlayerInfo,
        collection: Mapping[GemType, int],
        visible: Counter[GemType],
        values: Mapping[GemType, float],
    ) -> float:
        if mission.is_satisfied(collection):
            return float(mission.value)
        players = (info.self_player, *info.other_players)
        # Smoothing permits free auctions without assuming a bankrupt player
        # can buy an equal share of the remaining deck.
        share = (player.coins + 1) / sum(p.coins + 1 for p in players)
        share = min(0.65, share)
        our_distance = _missing(mission, collection)
        rival_distance = min(
            _missing(mission, p.won_gems) for p in players if p.id != player.id
        )
        race = 1 / (1 + 0.45 * our_distance / max(1, rival_distance))
        best = 0.0
        for target in _mission_targets(mission):
            needs = [
                max(0, n - collection.get(g, 0)) for g, n in zip(STANDARD_GEMS, target)
            ]
            probability = prod(
                _acquisition_probability(
                    needed,
                    visible[g],
                    info.unknown_gems[g],
                    share,
                    self.deck_fraction(info, g),
                )
                for g, needed in zip(STANDARD_GEMS, needs)
            )
            # Missing gems retain their intrinsic scoring value, so charge
            # only a modest share of their estimated acquisition cost here.
            cost = sum(n * values[g] * 0.62 for g, n in zip(STANDARD_GEMS, needs))
            affordability = min(1.0, (player.coins + 1) / max(1.0, cost))
            potential = (
                probability
                * affordability
                * race
                * max(0.0, 0.88 * mission.value - 0.18 * cost)
            )
            best = max(best, potential)
        return best

    def evaluate(self, info: BotInfo) -> AuctionValuation:
        """Value the current auction using only the bot's immutable snapshot."""
        card = info.current_auction
        values = self.gem_values(info)
        lot = (
            info.auction_gems[: card.value]
            if card is not None and card.type is AuctionType.AUCTION
            else ()
        )
        visible = Counter(info.auction_gems[len(lot) :])
        future_points = sum(
            values[g]
            * (visible[g] + info.unknown_gems[g] * self.deck_fraction(info, g))
            for g in STANDARD_GEMS
        )
        future_gems = info.gems_remaining - len(lot)
        treasure_cards = [
            (c.value, n)
            for c, n in info.auction_cards_remaining.items()
            if c.type is AuctionType.AUCTION
        ]
        average_lot = (
            sum(v * n for v, n in treasure_cards)
            / max(1, sum(n for _, n in treasure_cards))
        ) or 1.0
        future_lots = future_gems / average_lot
        horizon = future_lots / (future_lots + 1.5)
        cash = info.self_player.coins
        total_cash = cash + sum(p.coins for p in info.other_players)
        target_budget = 0.62 * future_points / info.num_players
        shortage = max(0.0, 1 - cash / max(1.0, target_budget))
        premium = (
            max(
                0.0,
                min(3.0, 0.85 * future_points / max(1, total_cash) - 1),
                0.70 * shortage,
            )
            * horizon
        )
        reserve = min(cash, ceil(cash * 0.40 * min(1.0, future_lots / 4)))
        if future_gems > 0 and cash > 0:
            reserve = max(1, reserve)
        productive_loan = max(0.0, target_budget - cash)

        missions = 0.0
        denial = 0.0
        if lot:
            players = (info.self_player, *info.other_players)
            gains = {}
            for player in players:
                after = Counter(player.won_gems)
                after.update(lot)
                gain = 0.0
                for mission in info.available_missions:
                    before_value = self._mission_potential(
                        info, mission, player, player.won_gems, visible, values
                    )
                    after_value = self._mission_potential(
                        info, mission, player, after, visible, values
                    )
                    marginal = max(0.0, after_value - before_value)
                    if player.id != info.self_player.id:
                        marginal /= 1 + _missing(mission, after)
                    gain += marginal
                gains[player.id] = gain
            missions = gains[info.self_player.id]
            # Only one opponent can acquire this lot; do not sum their gains.
            denial = 0.30 * max(gains[p.id] for p in info.other_players)
        return AuctionValuation(
            sum(values[g] for g in lot),
            missions,
            denial,
            1 + premium,
            reserve,
            productive_loan,
            future_points,
        )

    def get_bid(self, info: BotInfo) -> int:
        card = info.current_auction
        if card is None:
            return 0
        estimate = self.evaluate(info)
        cash = info.self_player.coins
        budget = max(0, cash - estimate.reserve)
        premium = estimate.liquidity - 1
        if card.type is AuctionType.AUCTION:
            bid = int(self.bid_fraction * estimate.treasure / estimate.liquidity)
        elif card.type is AuctionType.INVEST:
            # The bid itself returns at scoring. Only its lost liquidity is
            # a cost, unlike a treasure bid which is actually spent.
            bid = int(
                self.bid_fraction
                * card.value
                / max(self.investment_premium_floor, premium)
            )
        else:
            productive = min(card.value, estimate.productive_loan)
            profit = premium * productive
            bid = int(self.bid_fraction * profit / estimate.liquidity)
            # Net bid against loan proceeds, keeping useful borrowed cash.
            budget = min(cash + card.value, card.value - 1)
        # Never pay more than needed to beat every affordable rival bid.
        rival_limit = max(p.coins for p in info.other_players)
        if card.type is AuctionType.LOAN:
            rival_limit += card.value
        return max(0, min(bid, budget, rival_limit + 1))

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        # Terminal prices are fixed by starting hands, not reveal order.
        return 0


class ValuationBot(FractionalValuationBot):
    """The valuation model with shared expected-utility treasure bidding."""

    utility_k_scale = 1.0
    utility_k_floor = 0.5

    def __init__(self, *, use_expected_utility: bool = True) -> None:
        self.use_expected_utility = use_expected_utility
        self.on_game_start()

    def on_game_start(self) -> None:
        self.bid_rates: dict[int, float] = {}
        self._pending: BotInfo | None = None

    def _bid_distribution(self, info: BotInfo, player: PlayerInfo) -> list[float]:
        return bid_distribution(
            info,
            player,
            self.gem_values(info),
            self.bid_rates,
            self.bid_fraction,
            cash_limits=True,
        )

    win_probability = staticmethod(win_probability)

    def get_bid(self, info: BotInfo) -> int:
        self._pending = info
        card = info.current_auction
        if (
            not self.use_expected_utility
            or card is None
            or card.type is not AuctionType.AUCTION
        ):
            return super().get_bid(info)
        estimate = self.evaluate(info)
        budget = max(0, info.self_player.coins - estimate.reserve)
        ceiling = min(budget, max(p.coins for p in info.other_players) + 1)
        k = opportunity_cost(
            info, estimate.liquidity, self.utility_k_scale, self.utility_k_floor
        )
        distributions = {
            p.id: self._bid_distribution(info, p) for p in info.other_players
        }
        return max(
            range(ceiling + 1),
            key=lambda bid: expected_utility(
                self.win_probability(info, bid, distributions),
                estimate.treasure,
                bid,
                k,
            ),
        )

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        before = self._pending
        if (
            self.use_expected_utility
            and before is not None
            and before.round_number == info.round_number
            and result.auction_card.type is AuctionType.AUCTION
        ):
            prices = self.gem_values(before)
            gem_value = sum(
                prices[g] for g in before.auction_gems[: result.auction_card.value]
            )
            for player in before.other_players:
                update_bid_rate(
                    self.bid_rates,
                    player,
                    gem_value + completion_bonus(before, player),
                    result.bids[player.id],
                    self.bid_fraction,
                )
        self._pending = None

    def on_game_end(self, info: BotInfo, result: GameSummary) -> None:
        self._pending = None
