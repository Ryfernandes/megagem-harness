"""Legal hidden worlds and in-process level-2 continuations for sampling search.

Contributors: Grace Guo (85375651b), OpenAI Codex (implementation assistance).

Worlds use ONLY BotInfo, sampling the conditional prior uniformly.
Each rollout player gets its own BotInfo through the real snapshot builder;
sampled truth is never passed to its bidding policy. The real harness resolves
auctions, reveals, missions and terminal scores. No nested workers are spawned.
"""

import signal
from collections import Counter, deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from random import Random
from threading import current_thread, main_thread
from time import perf_counter
from types import FrameType

from bot_definitions.expected_utility import completion_bonus, update_bid_rate
from bot_definitions.valuation_bot import ValuationBot
from bot_info import _public_player, build_bot_info
from bots import BotInfo, GameSummary, PlayerInfo, RoundResult, SelfInfo
from constants import STANDARD_GEMS, AuctionType, GemType
from decks import AUCTION_CARDS, GEM_CARDS
from harness import GameState, finish_game, play_round, resolve_auction
from immutable import FrozenMap
from lib import AuctionCard, Deck
from player import Player

History = Sequence[tuple[BotInfo, RoundResult]]


class RolloutDeadline(Exception):
    """A bounded search ran out of time; discard its incomplete comparison batch."""


def _interrupt_search(signum: int, frame: FrameType | None) -> None:
    raise RolloutDeadline()


@contextmanager
def interruptible_deadline(seconds: float) -> Iterator[bool]:
    """Interrupt long Python/CPU steps on Unix; retain cooperative checks elsewhere.

    BotRunner callbacks execute on the worker main thread. Never replace an
    existing caller-owned alarm, and always disarm/restore before returning to
    BotRunner's checkpoint and socket protocol.
    """
    if seconds <= 0:
        raise RolloutDeadline()
    enabled = hasattr(signal, "setitimer") and current_thread() is main_thread()
    if not enabled or signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        yield False
        return
    previous = signal.signal(signal.SIGALRM, _interrupt_search)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield True
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        finally:
            signal.signal(signal.SIGALRM, previous)


class CachedValuationPolicy(ValuationBot):
    """Exact level 2, caching only repeated pure gem-price calls within a view."""

    def gem_values(self, info: BotInfo) -> dict[GemType, float]:
        if getattr(self, "_price_view", None) is not info:
            self._price_view = info
            self._price_values = self.expected_gem_values(info)
        return dict(self._price_values)


def public_players(info: BotInfo) -> tuple[PlayerInfo, ...]:
    """Strip the observer's private SelfInfo fields before reframing a snapshot."""
    own = info.self_player
    public = PlayerInfo(
        own.id,
        own.coins,
        own.hidden_gem_count,
        own.revealed_gems,
        own.won_gems,
        own.auction_cards,
        own.mission_cards,
        own.invested_coins,
    )
    return tuple(sorted((public, *info.other_players), key=lambda p: p.id))


def reframe(info: BotInfo, player_id: int, hand: Sequence[GemType]) -> BotInfo:
    players = public_players(info)
    player = players[player_id]
    if len(hand) != player.hidden_gem_count:
        raise ValueError("A sampled private hand must match the public hand size")
    own = SelfInfo(
        player.id,
        player.coins,
        player.hidden_gem_count,
        player.revealed_gems,
        player.won_gems,
        player.auction_cards,
        player.mission_cards,
        player.invested_coins,
        tuple(hand),
    )
    unseen = Counter(GEM_CARDS)
    unseen.subtract(info.value_display)
    unseen.subtract(hand)
    unseen.subtract(info.auction_gems)
    for other in players:
        unseen.subtract(other.won_gems)
    if any(unseen[g] < 0 for g in STANDARD_GEMS):
        raise ValueError("A sampled hand contradicts public gem counts")
    return replace(
        info,
        self_player=own,
        other_players=tuple(p for p in players if p.id != player_id),
        unknown_gems=FrozenMap({g: unseen[g] for g in STANDARD_GEMS}),
    )


@dataclass(frozen=True)
class SampledWorld:
    hands: tuple[tuple[GemType, ...], ...]
    gem_deck: tuple[GemType, ...]
    auction_deck: tuple[AuctionCard, ...]
    tie_seed: int


def sample_world(
    info: BotInfo,
    rng: Random,
) -> SampledWorld:
    known_unknown = Counter(GEM_CARDS)
    known_unknown.subtract(info.value_display)
    known_unknown.subtract(info.self_player.hidden_gems)
    known_unknown.subtract(info.auction_gems)
    for player in public_players(info):
        known_unknown.subtract(player.won_gems)
    if any(known_unknown[g] != info.unknown_gems.get(g, 0) for g in STANDARD_GEMS):
        raise ValueError("Unknown gems contradict the six-copy catalog or public facts")
    unknown = [g for g in STANDARD_GEMS for _ in range(info.unknown_gems.get(g, 0))]
    expected = info.gem_deck_count + sum(p.hidden_gem_count for p in info.other_players)
    if len(unknown) != expected or any(n < 0 for n in info.unknown_gems.values()):
        raise ValueError("Unknown gem inventory does not match hand/deck capacities")
    rng.shuffle(unknown)
    hands: list[tuple[GemType, ...]] = [()] * info.num_players
    hands[info.self_player.id] = info.self_player.hidden_gems
    offset = 0
    for player in info.other_players:
        hand = tuple(unknown[offset : offset + player.hidden_gem_count])
        offset += player.hidden_gem_count
        hands[player.id] = hand
    auctions = [
        card for card, n in info.auction_cards_remaining.items() for _ in range(n)
    ]
    if len(auctions) != info.auction_deck_count:
        raise ValueError("Remaining auction inventory does not match the deck size")
    rng.shuffle(auctions)
    return SampledWorld(
        tuple(hands),
        tuple(unknown[offset:]),
        tuple(auctions),
        rng.getrandbits(64),
    )


def historical_rates(
    root: BotInfo,
    world: SampledWorld,
    player_id: int,
    history: History,
) -> dict[int, float]:
    """Replay the actual public bid history with that world's sampled private hand.

    Current revealed cards fix the already-revealed portion of each past hand;
    only its still-hidden portion is sampled. Level-2 learning depends on hand
    COUNTS, so a remaining-hand color multiset is sufficient for this replay.
    """
    current = public_players(root)[player_id]
    rates: dict[int, float] = {}
    for before, result in history:
        if result.auction_card.type is not AuctionType.AUCTION:
            continue
        past = public_players(before)[player_id]
        hand = current.revealed_gems[len(past.revealed_gems) :] + world.hands[player_id]
        frame = reframe(before, player_id, hand)
        prices = ValuationBot.expected_gem_values(frame)
        gems = sum(prices[g] for g in frame.auction_gems[: result.auction_card.value])
        for rival in frame.other_players:
            update_bid_rate(
                rates,
                rival,
                gems + completion_bonus(frame, rival),
                result.bids[rival.id],
                ValuationBot.bid_fraction,
            )
    return rates


class RolloutPlayer(Player):
    """Player with a direct level-2 policy, never a BotRunner or process."""

    def __init__(
        self,
        source: PlayerInfo,
        hand: Sequence[GemType],
        deadline: float,
        rates: Mapping[int, float] | None = None,
    ) -> None:
        self.id = source.id
        self.coins = source.coins
        self.hidden_gem_hand = list(hand)
        self.public_gem_hand = list(source.revealed_gems)
        self.gem_collection = Counter(source.won_gems)
        self.auction_cards = list(source.auction_cards)
        self.mission_cards = list(source.mission_cards)
        self.invested_coins = source.invested_coins
        self.policy = CachedValuationPolicy()
        self.policy.bid_rates = dict(rates or {})
        self.deadline = deadline

    def on_game_start(self) -> None:
        self.policy.on_game_start()

    def get_bid(self, info: BotInfo) -> int:
        if perf_counter() >= self.deadline:
            raise RolloutDeadline()
        bid = self.policy.get_bid(info)
        card = info.current_auction
        budget = self.coins + (
            card.value if card is not None and card.type is AuctionType.LOAN else 0
        )
        if type(bid) is not int or not 0 <= bid <= budget:
            raise ValueError("A level-2 rollout policy produced an illegal bid")
        return bid

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        return self.policy.choose_hidden_gem(info, bids)

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        self.policy.on_round_end(info, result)

    def on_game_end(self, info: BotInfo, result: GameSummary) -> None:
        self.policy.on_game_end(info, result)


def world_game(
    info: BotInfo,
    world: SampledWorld,
    rates: Sequence[Mapping[int, float]],
    deadline: float,
) -> GameState:
    players: list[Player] = [
        RolloutPlayer(p, world.hands[p.id], deadline, rates[p.id])
        for p in public_players(info)
    ]
    drawn = Counter(AUCTION_CARDS)
    drawn.subtract(info.auction_cards_remaining)
    return GameState(
        players,
        Deck(world.gem_deck),
        Deck(world.auction_deck),
        list(info.available_missions),
        info.value_chart,
        Random(world.tie_seed),
        deque(info.auction_gems),
        Counter(info.value_display),
        info.previous_winner_id,
        info.round_number,
        info.current_auction,
        [card for card, n in drawn.items() for _ in range(n)],
    )


def clone_game(game: GameState, deadline: float) -> GameState:
    players: list[Player] = []
    for source in game.players:
        assert isinstance(source, RolloutPlayer)
        player = RolloutPlayer(
            _public_player(source),
            source.hidden_gem_hand,
            deadline,
            source.policy.bid_rates,
        )
        player.policy._pending = source.policy._pending
        players.append(player)
    rng = Random(0)
    rng.setstate(game.rng.getstate())
    return GameState(
        players,
        Deck(game.gem_deck._cards),
        Deck(game.auction_deck._cards),
        list(game.available_missions),
        game.value_chart,
        rng,
        deque(game.auction_gems),
        Counter(game.value_display),
        game.previous_winner_id,
        game.round_number,
        game.current_auction,
        list(game.auction_history),
    )


def root_bids(game: GameState) -> list[int]:
    views = [build_bot_info(game, p) for p in game.players]
    return [p.get_bid(info) for p, info in zip(game.players, views)]


def finish_rollout(
    root: GameState,
    bids: Sequence[int],
    player_id: int,
    forced_bid: int,
    deadline: float,
) -> tuple[int, ...]:
    game = clone_game(root, deadline)
    chosen = list(bids)
    chosen[player_id] = forced_bid
    if perf_counter() >= deadline:
        raise RolloutDeadline()
    resolve_auction(game, chosen)
    while game.auction_gems:
        if perf_counter() >= deadline:
            raise RolloutDeadline()
        play_round(game)
    if perf_counter() >= deadline:
        raise RolloutDeadline()
    return finish_game(game).scores
