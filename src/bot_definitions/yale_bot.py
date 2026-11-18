"""An adaptive first-price bidder; run with --bots custom_bots:YaleBot.

Each auction is valued from the rules, then bid on as a best response:

- Gem values: a gem's final count is the number of copies dealt to hands. That is
  fixed at setup, so reveals change only what players know. The unknown copies in
  opponents' hands follow a hypergeometric law, which gives an exact expected value.
- Missions: the change in this bot's estimated chance of claiming them first.
- Coins: worth 1 point at game end, plus a shadow price while the remaining prizes
  exceed the coins available to chase them. Invest bids are refunded, so they cost
  only that shadow price; loans are worth that shadow price on their proceeds.
- Bids: each opponent's bids are learned from past rounds, then the bot picks the
  bid maximizing P(win | bid) * (prize - cost), using coin caps and tie priority.

Bidding extra just to deny a rival lost games in testing, so prizes are not inflated
for it. Exact optimal play is intractable here: the game has hidden hands, sealed
simultaneous bids, and multiple players, so this approximates a best response.
"""

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from math import comb, exp, prod

from bots import Bot, BotInfo, PlayerInfo, RoundResult
from constants import STANDARD_GEMS, AuctionType, GemType, WildCardRequirement
from lib import AuctionCard, MissionCard

Sample = tuple[float, int]  # A bid as (fraction of the prize, coins).


class YaleBot(Bot):
    """Value each auction exactly where possible, then best-respond to observed bids."""

    # Assumed bids, as fractions of the prize, before any bids are observed.
    PRIOR_RATIOS = (0.3, 0.45, 0.6)
    # Weight of each assumed bid; one observed bid counts as 2 for its bidder.
    PRIOR_WEIGHT = 0.25
    # Observed bids kept for each opponent and auction type.
    HISTORY = 12
    # Multiplier on the coin shadow price, tuned by tournament; results were flat
    # from 3 to 5 and clearly worse at 2 or below.
    SCARCITY = 4.0
    # Chance that a rival who can complete a mission gets there first.
    RACE = 0.5

    def __init__(self) -> None:
        self.seen: dict[tuple[int, AuctionType], deque[Sample]] = {}
        self.pooled: dict[AuctionType, deque[Sample]] = {}
        self.pending: tuple[float, dict[int, int]] | None = None

    def on_game_start(self) -> None:
        self.seen = {}  # Seats can change between games; pooled bids carry over.
        self.pending = None

    def get_bid(self, info: BotInfo) -> int:
        card = info.current_auction
        if card is None:
            return 0
        me = info.self_player
        values = self._gem_values(info)
        supply = self._supply(info)
        lot = info.auction_gems[: card.value]
        is_treasure = card.type is AuctionType.AUCTION
        lot_value = sum(values[gem] for gem in lot) if is_treasure else 0.0
        price = self._coin_price(info, card, values, supply, lot_value)
        # A prize worth nothing still gets a positive scale for reading bid ratios.
        scale = max(lot_value if is_treasure else float(card.value), 1.0)
        self.pending = (scale, {p.id: p.coins for p in info.other_players})

        if card.type is AuctionType.INVEST:
            prize = float(card.value)
            coin_cost = price - 1  # The bid is refunded, but stays locked up.
        elif card.type is AuctionType.LOAN:
            prize = (price - 1) * card.value
            coin_cost = price
        else:
            prize = lot_value + self._mission_gain(info, me, lot, supply)
            coin_cost = price

        limit = me.coins + (card.value if card.type is AuctionType.LOAN else 0)
        models = [
            (p.id, self._predict_bids(p, card, scale)) for p in info.other_players
        ]
        best_bid, best_payoff = 0, float("-inf")
        for bid in range(limit + 1):
            margin = prize - coin_cost * bid
            if margin < 0:
                break
            payoff = margin * self._win_chance(info, models, bid)
            if payoff > best_payoff:  # Strict: prefer the cheaper of equal bids.
                best_bid, best_payoff = bid, payoff
        return best_bid

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        """Reveal the gem whose news would most cheapen the gems still for sale."""
        hand = info.self_player.hidden_gems
        supply = self._supply(info)
        unseen = sum(info.unknown_gems.values()) + len(hand)
        hidden = unseen - info.gem_deck_count

        def repricing(gem: GemType) -> float:
            shown = info.value_display.get(gem, 0)
            copies = info.unknown_gems.get(gem, 0) + hand.count(gem)
            before = self._expected_value(info, shown, copies, unseen, hidden)
            after = self._expected_value(
                info, shown + 1, copies - 1, unseen - 1, hidden - 1
            )
            return (after - before) * supply[gem]

        return min(range(len(hand)), key=lambda index: repricing(hand[index]))

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        if self.pending is None:
            return
        scale, coins = self.pending
        self.pending = None
        kind = result.auction_card.type
        pooled = self.pooled.setdefault(kind, deque(maxlen=self.HISTORY))
        for player_id, bid in result.bids.items():
            # A bankrupt opponent's zero says nothing about its strategy.
            if coins.get(player_id, 0) > 0:
                sample = (bid / scale, bid)
                pooled.append(sample)
                self.seen.setdefault(
                    (player_id, kind), deque(maxlen=self.HISTORY)
                ).append(sample)

    # Return each gem's exact expected final value, averaged over opponents' hands.
    def _gem_values(self, info: BotInfo) -> dict[GemType, float]:
        hand = info.self_player.hidden_gems
        unseen = sum(info.unknown_gems.values())
        hidden = unseen - info.gem_deck_count
        return {
            gem: self._expected_value(
                info,
                info.value_display.get(gem, 0) + hand.count(gem),
                info.unknown_gems.get(gem, 0),
                unseen,
                hidden,
            )
            for gem in STANDARD_GEMS
        }

    # Average a gem's chart value when `hidden` of `unseen` cards reach hands.
    @staticmethod
    def _expected_value(
        info: BotInfo, known: int, copies: int, unseen: int, hidden: int
    ) -> float:
        total = 0
        for found in range(min(copies, hidden) + 1):
            ways = comb(copies, found) * comb(unseen - copies, hidden - found)
            total += ways * info.value_chart.get_value(known + found)
        return total / comb(unseen, hidden)

    # Return the expected number of each gem still to be sold, exposed gems included.
    @staticmethod
    def _supply(info: BotInfo) -> dict[GemType, float]:
        unseen = sum(info.unknown_gems.values())
        drawn = info.gem_deck_count / unseen if unseen else 0.0
        return {
            gem: info.auction_gems.count(gem) + info.unknown_gems.get(gem, 0) * drawn
            for gem in STANDARD_GEMS
        }

    # Return what one coin is worth in points, counting its use in later auctions.
    def _coin_price(
        self,
        info: BotInfo,
        card: AuctionCard,
        values: Mapping[GemType, float],
        supply: Mapping[GemType, float],
        lot_value: float,
    ) -> float:
        gems_left = info.gems_remaining
        if card.type is AuctionType.AUCTION:
            gems_left -= min(card.value, len(info.auction_gems))
        cards = info.auction_cards_remaining
        deck_gems = sum(
            future.value * count
            for future, count in cards.items()
            if future.type is AuctionType.AUCTION
        )
        # Later rounds, estimated from the gems sold by the average remaining card.
        seen = min(1.0, gems_left / deck_gems) if deck_gems else 0.0

        prizes = sum(values[gem] * supply[gem] for gem in STANDARD_GEMS) - lot_value
        prizes += sum(mission.value for mission in info.available_missions) / 2
        coins = float(sum(p.coins for p in self._players(info)))
        for future, count in cards.items():
            if future.type is AuctionType.INVEST:
                prizes += future.value * count * seen
            elif future.type is AuctionType.LOAN:
                coins += future.value * count * seen
        if coins <= 0:
            return 1.0
        return 1 + self.SCARCITY * max(0.0, prizes / coins - 1)

    @staticmethod
    def _players(info: BotInfo) -> tuple[PlayerInfo, ...]:
        return (info.self_player, *info.other_players)

    # Return the mission points a player would gain, in expectation, from this lot.
    def _mission_gain(
        self,
        info: BotInfo,
        player: PlayerInfo,
        lot: Sequence[GemType],
        supply: Mapping[GemType, float],
    ) -> float:
        remaining = dict(supply)
        for gem in lot:
            remaining[gem] = max(0.0, remaining[gem] - 1)
        have = {g: player.won_gems.get(g, 0) for g in STANDARD_GEMS}
        after = {g: have[g] + lot.count(g) for g in STANDARD_GEMS}
        total = sum(p.coins for p in self._players(info))
        gain = 0.0
        for mission in info.available_missions:
            rivals = 1.0
            for other in self._players(info):
                if other.id != player.id:
                    rival_have = {g: other.won_gems.get(g, 0) for g in STANDARD_GEMS}
                    share = other.coins / total if total else 0.0
                    rivals *= 1 - self.RACE * self._completion_odds(
                        mission, rival_have, supply, share
                    )
            share = player.coins / total if total else 0.0
            if mission.is_satisfied(after):
                odds_after = 1.0
            else:
                odds_after = rivals * self._completion_odds(
                    mission, after, remaining, share
                )
            odds_before = rivals * self._completion_odds(mission, have, supply, share)
            gain += mission.value * (odds_after - odds_before)
        return gain

    # Estimate the chance of completing a mission if each gem is won with odds `share`.
    @classmethod
    def _completion_odds(
        cls,
        mission: MissionCard,
        have: Mapping[GemType, int],
        supply: Mapping[GemType, float],
        share: float,
    ) -> float:
        owned = dict(have)
        odds = 1.0
        for gem, needed in mission.gem_requirements.items():
            if gem is not GemType.WILD_CARD:
                odds *= cls._odds_of(needed - owned[gem], supply[gem], share)
                owned[gem] = max(0, owned[gem] - needed)
        wild = mission.gem_requirements.get(GemType.WILD_CARD, 0)
        if not wild:
            return odds

        # Return each color's odds of winning `missing(color)` gems, best first.
        def ranked(missing: Callable[[GemType], int]) -> list[float]:
            return sorted(
                (cls._odds_of(missing(g), supply[g], share) for g in STANDARD_GEMS),
                reverse=True,
            )

        match mission.wild_card_requirement:
            case WildCardRequirement.SAME:
                return odds * ranked(lambda g: wild - owned[g])[0]
            case WildCardRequirement.DIFFERENT:
                # Colors already owned rank first, with odds of one.
                return odds * prod(ranked(lambda g: 1 - owned[g])[:wild])
            case WildCardRequirement.PAIRS:
                pairs_left = wild // 2 - sum(owned[g] // 2 for g in STANDARD_GEMS)
                best = ranked(lambda g: 2 - owned[g] % 2)
                return odds * prod(best[: max(0, pairs_left)])
        return 0.0

    # Return the Poisson chance of winning `needed` gems from an expected supply.
    @staticmethod
    def _odds_of(needed: int, supply: float, share: float) -> float:
        if needed <= 0:
            return 1.0
        if supply < needed:
            return 0.0
        mean = supply * share
        term, below = exp(-mean), 0.0
        for count in range(needed):
            below += term
            term *= mean / (count + 1)
        return max(0.0, 1 - below)

    # Return an opponent's likely bids, as counts, for a prize of size `scale`.
    def _predict_bids(
        self, player: PlayerInfo, card: AuctionCard, scale: float
    ) -> dict[int, float]:
        limit = player.coins + (card.value if card.type is AuctionType.LOAN else 0)
        predicted: dict[int, float] = {}

        def add(bid: float, weight: float) -> None:
            capped = min(limit, round(bid))
            predicted[capped] = predicted.get(capped, 0.0) + weight

        for ratio in self.PRIOR_RATIOS:
            add(ratio * scale, self.PRIOR_WEIGHT)
        pooled = self.pooled.get(card.type, ())
        personal = self.seen.get((player.id, card.type), ())
        for samples, weight in ((pooled, 1), (personal, 2)):
            for ratio, coins in samples:
                add(ratio * scale, weight)
                add(coins, weight)
        return predicted

    # Return the chance a bid beats every opponent, applying tie priority.
    @staticmethod
    def _win_chance(
        info: BotInfo, models: Sequence[tuple[int, Mapping[int, float]]], bid: int
    ) -> float:
        me = info.self_player.id
        leader = info.previous_winner_id
        chance = 1.0
        for player_id, model in models:
            if leader is None:
                tie = 0.5
            else:
                seat = (me - leader - 1) % info.num_players
                tie = float(seat < (player_id - leader - 1) % info.num_players)
            total = sum(model.values())
            below = sum(n for value, n in model.items() if value < bid)
            chance *= (below + tie * model.get(bid, 0.0)) / total
        return chance
