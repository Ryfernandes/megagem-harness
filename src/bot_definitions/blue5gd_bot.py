"""Blue5GD: bid the coins that beat the bids we expect, and no more.

Each turn:
1. Price the gems with the same exact chart formula as MicahBot. The inputs
   are the value display, this bot's own hidden hand, and the public
   unknown-gem pool. Hidden deck order is not used.
2. Price a coin from the chart family (rising, falling, or a middle peak)
   and from how many gems are left. Early coins are worth more than 1 point.
   Late coins fall back toward face value.
3. Guess each opponent's bid from a starting ratio plus the bids already
   seen this game. The guess never goes above the coins that seat can legally
   spend, including a loan's proceeds.
4. Score every legal bid by (chance of winning) times (prize minus cost)
   and play the best one. Zero is always allowed.
5. After a win, reveal a hidden gem that makes later public prices worse
   for colors this bot still wants. Revealing does not change the final
   chart values. It only changes what other bots believe.

There is no search over future games. The bot stays fast so a tuner can play
a huge number of matches, and every callback finishes well inside the
2 second limit. The constructor takes no arguments, which is what the
harness requires. A tuner may replace ``config`` after construction.

Contributors:
    Micah
"""

from __future__ import annotations

from collections.abc import Mapping

from bot_definitions.blue5gd_config import (
    DEFAULT_BLUE5_CONFIG,
    Blue5Config,
    chart_family,
)
from bot_definitions.blue5gd_opponents import OpponentMemory
from bot_definitions.blue5gd_value import (
    denial_points,
    is_late,
    own_prize,
    public_scale,
)
from bot_definitions.micah_strategy import (
    beats_on_tie,
    expected_future_supplies,
    gem_expected_values,
    public_gem_expected_values,
    public_value_shift,
)
from bots import Bot, BotInfo, GameSummary, RoundResult
from constants import AuctionType, GemType


def _our_cap(info: BotInfo) -> int:
    """Most coins we are allowed to bid, including a loan's proceeds."""
    card = info.current_auction
    coins = int(info.self_player.coins)
    if card is not None and card.type is AuctionType.LOAN:
        return max(0, coins + int(card.value))
    return max(0, coins)


def _their_cap(coins: int, info: BotInfo) -> int:
    card = info.current_auction
    if card is not None and card.type is AuctionType.LOAN:
        return max(0, coins + int(card.value))
    return max(0, coins)


def _tie_share(info: BotInfo, opp_id: int) -> float:
    """Chance we win if this opponent bids the same amount we do."""
    leader = info.previous_winner_id
    if leader is None:
        return 0.5
    beat = beats_on_tie(info.self_player.id, opp_id, leader, info.num_players)
    if beat is None:
        return 0.5
    return 1.0 if beat else 0.0


def _auction_kind(info: BotInfo) -> str:
    card = info.current_auction
    if card is None or card.type is AuctionType.AUCTION:
        return "treasure"
    if card.type is AuctionType.INVEST:
        return "invest"
    return "loan"


def choose_bid(info: BotInfo, memory: OpponentMemory, config: Blue5Config) -> int:
    """Return the bid with the highest expected points. 0 is always allowed."""
    card = info.current_auction
    cap = _our_cap(info)
    if card is None or cap <= 0:
        return 0

    gem_evs: dict[GemType, float]
    public_evs: dict[GemType, float]
    if card.type is AuctionType.AUCTION:
        gem_evs = gem_expected_values(info)
        public_evs = public_gem_expected_values(info)
    else:
        gem_evs = {}
        public_evs = {}

    prize, coin_cost = own_prize(info, config, gem_evs)
    prize += denial_points(info, config, coin_cost)
    horizon = min(1.0, info.gems_remaining / 8.0)
    prize -= config.tie_penalty * horizon

    scale = public_scale(info, public_evs)
    kind_name = _auction_kind(info)
    kind = card.type
    family = chart_family(info.value_chart)
    prior = config.prior_ratio(kind_name, is_late(info), family)
    prepared: list[tuple[list[float], list[float], float, float]] = []
    for opponent in info.other_players:
        weights = memory.bid_weights(
            opponent.id,
            kind,
            prior,
            scale,
            _their_cap(int(opponent.coins), info),
            config,
        )
        prefix = [0.0]
        for weight in weights:
            prefix.append(prefix[-1] + weight)
        prepared.append(
            (weights, prefix, prefix[-1], _tie_share(info, opponent.id))
        )

    best_bid = 0
    best_utility = 0.0
    for bid in range(cap + 1):
        margin = prize - coin_cost * bid
        if margin < 0.0:
            # A more expensive bid is worse when each coin costs something.
            if coin_cost > 0.0:
                break
            continue
        chance = 1.0
        for weights, prefix, total, tie_share in prepared:
            if total <= 0.0 or bid >= len(weights):
                continue
            chance *= (prefix[bid] + tie_share * weights[bid]) / total
        utility = chance * margin
        if utility > best_utility:
            best_bid = bid
            best_utility = utility
    return int(best_bid)


def choose_reveal_index(info: BotInfo, config: Blue5Config) -> int:
    """Index of the hidden gem whose reveal helps us most on later prices.

    Revealing does not change the final gem values. It only changes what the
    other bots believe. A negative price shift makes a color look cheaper.
    """
    hand = info.self_player.hidden_gems
    if not hand:
        return 0
    supplies = expected_future_supplies(info)
    evs = gem_expected_values(info)

    def score(index: int) -> float:
        gem: GemType = hand[index]
        shift = public_value_shift(info, gem)
        desire = supplies.get(gem, 0.0) + config.reveal_weight * max(
            evs.get(gem, 0.0), 0.0
        )
        return shift * desire

    return min(range(len(hand)), key=score)


class Blue5GD(Bot):
    """Tournament bot. Set `config` before the match if you are tuning."""

    def __init__(self) -> None:
        self.games_played: int = 0
        self.config: Blue5Config = DEFAULT_BLUE5_CONFIG
        self.memory: OpponentMemory = OpponentMemory()
        self.pending: tuple[float, dict[int, int]] | None = None

    def on_game_start(self) -> None:
        self.games_played += 1
        self.memory.begin_game()
        self.pending = None

    def get_bid(self, info: BotInfo) -> int:
        card = info.current_auction
        if card is None:
            self.pending = None
            return 0
        if card.type is AuctionType.AUCTION:
            public_evs = public_gem_expected_values(info)
        else:
            public_evs = {}
        scale = public_scale(info, public_evs)
        self.pending = (
            scale,
            {player.id: int(player.coins) for player in info.other_players},
        )
        return int(choose_bid(info, self.memory, self.config))

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        return int(choose_reveal_index(info, self.config))

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        if self.pending is None:
            return
        scale, coins = self.pending
        self.pending = None
        me = info.self_player.id
        for player_id, bid in result.bids.items():
            if player_id == me:
                continue
            self.memory.record(
                int(player_id),
                result.auction_card.type,
                int(bid),
                scale,
                int(coins.get(player_id, 0)),
            )

    def on_game_end(self, info: BotInfo, result: GameSummary) -> None:
        pass
