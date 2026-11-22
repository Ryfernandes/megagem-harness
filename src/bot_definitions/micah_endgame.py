"""Endgame denial value and sampled first-place odds for MicahBot.

Denial prices how much it is worth to stop a rival from winning this auction.
The sampler estimates the chance our terminal score is at least as high as
every other score after this auction resolves.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from random import Random

from bot_definitions.micah_bid_model import winner_probabilities
from bot_definitions.micah_strategy import (
    DEFAULT_STRATEGY_CONFIG,
    StrategyConfig,
    beats_on_tie,
    estimate_rounds_left_after,
    expected_future_supplies_after_current,
    future_coin_value,
    gem_expected_values,
    immediate_mission_bonus,
    player_known_score,
    projected_terminal_scores,
)
from bots import BotInfo, PlayerInfo
from constants import STANDARD_GEMS, AuctionType, GemType
from lib import AuctionCard

# Denial is off while several auctions remain. A rival counts as a contender
# when their projected score is this close to the best score in the game.
_DENIAL_ROUND_LIMIT = 3.0
_CONTENDER_GAP = 8.0
# First-place sampling is only for the last couple of auctions.
_ENDGAME_ROUND_LIMIT = 2.0


def _players(info: BotInfo) -> tuple[PlayerInfo, ...]:
    """Return our seat followed by the other seats."""
    return (info.self_player, *info.other_players)


def _other_coins(info: BotInfo, player_id: int) -> list[int]:
    """Coins of every player except ``player_id``, used for liquidity."""
    return [player.coins for player in _players(info) if player.id != player_id]


def _mean_public_ev(public_evs: Mapping[GemType, float]) -> float:
    """Average of the public per-gem values, or 10 when the map is empty."""
    if not public_evs:
        return 10.0
    return sum(public_evs.values()) / float(len(public_evs))


def _as_distribution(
    distribution: Sequence[tuple[int, float]] | None,
) -> tuple[tuple[int, float], ...]:
    """Return a usable bid distribution. Missing or empty means a sure bid of 0."""
    if not distribution:
        return ((0, 1.0),)
    positive = tuple(
        (int(bid), float(prob)) for bid, prob in distribution if prob > 0.0
    )
    if not positive:
        return ((0, 1.0),)
    return positive


def _mean_bid(distribution: Sequence[tuple[int, float]] | None) -> float:
    """Expected bid of a discrete distribution. Missing distributions bid 0."""
    if not distribution:
        return 0.0
    weighted = 0.0
    total_prob = 0.0
    for bid, prob in distribution:
        weighted += float(bid) * float(prob)
        total_prob += float(prob)
    if total_prob <= 0.0:
        return 0.0
    return weighted / total_prob


def _clamped_payment(mean_bid: float, coins: int) -> float:
    """Coins actually paid when a rival cannot bid more than they hold."""
    if mean_bid <= 0.0 or coins <= 0:
        return 0.0
    return min(mean_bid, float(coins))


def _treasure_lot(info: BotInfo, card: AuctionCard) -> tuple[GemType, ...]:
    """Gems awarded by a treasure card, in auction order."""
    return info.auction_gems[: min(card.value, len(info.auction_gems))]


def _tie_shares(
    tied: Sequence[int],
    previous_winner_id: int | None,
    num_players: int,
) -> dict[int, float]:
    """Probability each tied bidder wins, using ``beats_on_tie``.

    Round 1 has no previous winner, so ``beats_on_tie`` is ``None`` and the
    tied players split the win equally. Later rounds have a total order.
    """
    if len(tied) == 1:
        return {tied[0]: 1.0}
    if beats_on_tie(tied[0], tied[1], previous_winner_id, num_players) is None:
        share = 1.0 / float(len(tied))
        return {player_id: share for player_id in tied}
    winner = tied[0]
    for other in tied[1:]:
        if beats_on_tie(winner, other, previous_winner_id, num_players) is False:
            winner = other
    return {winner: 1.0}


def _win_probabilities(
    info: BotInfo,
    our_bid: int,
    opponent_distributions: Mapping[int, Sequence[tuple[int, float]]],
) -> dict[int, float]:
    """P(each player wins) when our bid is certain and rivals bid independently."""
    return winner_probabilities(
        info.self_player.id,
        our_bid,
        opponent_distributions,
        info.previous_winner_id,
        info.num_players,
    )


def _contender_net_gain(
    info: BotInfo,
    player: PlayerInfo,
    card: AuctionCard,
    public_evs: Mapping[GemType, float],
    mean_bid: float,
    rounds_left: float,
    avg_lot_ev: float,
    config: StrategyConfig,
) -> float:
    """How many points this rival gains by winning, after liquidity. Never below 0."""
    coins = float(player.coins)
    others = _other_coins(info, player.id)
    current_cash = future_coin_value(coins, rounds_left, others, avg_lot_ev, config)

    if card.type is AuctionType.AUCTION:
        lot = _treasure_lot(info, card)
        gem_value = sum(public_evs.get(gem, 0.0) for gem in lot)
        mission = float(
            immediate_mission_bonus(player.won_gems, lot, info.available_missions)
        )
        payment = _clamped_payment(mean_bid, player.coins)
        after_cash = future_coin_value(
            coins - payment, rounds_left, others, avg_lot_ev, config
        )
        liquidity_cost = current_cash - after_cash
        return max(0.0, gem_value + mission - liquidity_cost)

    if card.type is AuctionType.INVEST:
        payment = _clamped_payment(mean_bid, player.coins)
        after_cash = future_coin_value(
            coins - payment, rounds_left, others, avg_lot_ev, config
        )
        # The bid itself comes back as invested coins. Only the lost surplus
        # is a real cost, and that surplus cannot be negative here.
        liquidity_lost = (current_cash - after_cash) - payment
        return max(0.0, float(card.value) - max(0.0, liquidity_lost))

    # Loan: paying at least the principal does not leave them ahead.
    if mean_bid >= float(card.value):
        return 0.0
    after_cash = future_coin_value(
        coins + float(card.value) - mean_bid,
        rounds_left,
        others,
        avg_lot_ev,
        config,
    )
    return max(0.0, (after_cash - float(card.value)) - current_cash)


def denial_value(
    info: BotInfo,
    our_bid: int,
    opponent_distributions: Mapping[int, Sequence[tuple[int, float]]],
    public_evs: Mapping[GemType, float],
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
    gem_evs: Mapping[GemType, float] | None = None,
) -> float:
    """Extra value of blocking rivals. This is not our own lot value.

    Returns 0 when more than three auctions are expected after this one, when
    nothing is up for auction, or when no rival is within 8 points of the best
    projected score. Projected scores use ``gem_expected_values``, not
    ``public_evs``.

    For each contender, compare P(they win if we bid 0) with P(they win if we
    bid ``our_bid``). Their distributions are independent. Our bid is certain.
    A rival missing from the mapping bids 0. Ties use ``beats_on_tie``; a
    round-1 tie (``None``) splits the probability equally.

    The rival's net gain is never the raw lot value:

    * Treasure: public value of the lot, plus missions the lot completes,
      minus the liquidity cost of paying their mean bid. If that mean is
      above their coins, they pay only the coins they have.
    * Invest: the card bonus, minus surplus lost while the bid is tied up.
      The bid's face value comes back, so it is not a scoring cost.
    * Loan: 0 when their mean bid is at least the principal. Otherwise the
      gain in coin value from borrowing, minus the principal they must repay.

    Denial for one rival is ``max(0, P(win at 0) - P(win at our bid)) * net``.
    The result is the sum of those terms, with no extra weight.
    """
    if estimate_rounds_left_after(info) > _DENIAL_ROUND_LIMIT:
        return 0.0
    card = info.current_auction
    if card is None or not info.other_players:
        return 0.0

    resolved_gem_evs = gem_evs if gem_evs is not None else gem_expected_values(info)
    projected = projected_terminal_scores(info, resolved_gem_evs, config)
    best = max(projected.values())
    top_opp = max(projected[player.id] for player in info.other_players)
    contenders = [
        player
        for player in info.other_players
        if best - projected[player.id] <= _CONTENDER_GAP
        and top_opp - projected[player.id] <= 6.0
    ]
    if not contenders:
        return 0.0

    rounds_left = estimate_rounds_left_after(info)
    avg_lot_ev = _mean_public_ev(public_evs)
    probs_at_zero = _win_probabilities(info, 0, opponent_distributions)
    if our_bid == 0:
        probs_at_bid = probs_at_zero
    else:
        probs_at_bid = _win_probabilities(info, our_bid, opponent_distributions)

    denial = 0.0
    for player in contenders:
        mean_bid = _mean_bid(opponent_distributions.get(player.id))
        net = _contender_net_gain(
            info,
            player,
            card,
            public_evs,
            mean_bid,
            rounds_left,
            avg_lot_ev,
            config,
        )
        blocked = probs_at_zero[player.id] - probs_at_bid[player.id]
        denial += max(0.0, blocked) * net
    # In N-player games, paying to block one rival also helps the other N-2 players.
    return config.denial_weight * (denial / float(max(1, info.num_players - 1)))


def sample_opponent_hidden_hands(
    info: BotInfo, rng: Random
) -> dict[int, tuple[GemType, ...]]:
    """Draw each opponent's hidden hand from ``unknown_gems`` without replacement.

    The bag is every gem still hidden in rival hands or in the draw pile.
    Opponents are filled in ascending id order. If the bag runs out, the
    current hand keeps only what is left and every later hand is empty.
    """
    bag: list[GemType] = []
    for gem in STANDARD_GEMS:
        bag.extend([gem] * int(info.unknown_gems.get(gem, 0)))
    rng.shuffle(bag)

    hands: dict[int, tuple[GemType, ...]] = {}
    cursor = 0
    bag_size = len(bag)
    for opponent in sorted(info.other_players, key=lambda player: player.id):
        need = opponent.hidden_gem_count
        if need <= 0 or cursor >= bag_size:
            hands[opponent.id] = ()
            continue
        take = min(need, bag_size - cursor)
        hands[opponent.id] = tuple(bag[cursor : cursor + take])
        cursor += take
    return hands


def _terminal_display_counts(
    info: BotInfo, hands: Mapping[int, Sequence[GemType]]
) -> dict[GemType, int]:
    """Display counts after our hidden gems and the sampled rival hands.

    Auctioned gems stay out of the display. Gems left in the bag are the
    deck, so they stay out too.
    """
    counts = {gem: int(info.value_display.get(gem, 0)) for gem in STANDARD_GEMS}
    for gem in info.self_player.hidden_gems:
        if gem in counts:
            counts[gem] += 1
    for hand in hands.values():
        for gem in hand:
            if gem in counts:
                counts[gem] += 1
    return counts


def _chart_values(
    info: BotInfo, display_counts: Mapping[GemType, int]
) -> dict[GemType, float]:
    """Certain per-gem chart value for one sampled display. Counts clamp at 6."""
    chart = info.value_chart
    return {
        gem: float(chart.get_value(min(6, display_counts.get(gem, 0))))
        for gem in STANDARD_GEMS
    }


def _draw_bid(distribution: Sequence[tuple[int, float]] | None, rng: Random) -> int:
    """Draw one bid. A missing distribution bids 0 and does not touch ``rng``."""
    if not distribution:
        return 0
    roll = rng.random()
    cumulative = 0.0
    last_bid = distribution[-1][0]
    for bid, prob in distribution:
        cumulative += prob
        if roll < cumulative:
            return bid
        last_bid = bid
    return last_bid


def _select_winner_id(
    bids: Mapping[int, int],
    previous_winner_id: int | None,
    num_players: int,
    rng: Random,
) -> int:
    """Highest bid wins. Round-1 ties use ``rng.choice``; later ties use seat order.

    ``rng`` is used only when several players share the high bid and there is
    no previous winner. Tied ids are sorted so that coin flip is stable.
    """
    highest = max(bids.values())
    tied = sorted(player_id for player_id, bid in bids.items() if bid == highest)
    if len(tied) == 1:
        return tied[0]
    if beats_on_tie(tied[0], tied[1], previous_winner_id, num_players) is None:
        return rng.choice(tied)
    winner = tied[0]
    for other in tied[1:]:
        if beats_on_tie(winner, other, previous_winner_id, num_players) is False:
            winner = other
    return winner


def _unique_bids(candidate_bids: Sequence[int]) -> list[int]:
    """Candidate bids in first-seen order, without repeats."""
    ordered: list[int] = []
    seen: set[int] = set()
    for bid in candidate_bids:
        if bid in seen:
            continue
        seen.add(bid)
        ordered.append(bid)
    return ordered


def _scores_after_auction(
    base_scores: Mapping[int, float],
    card: AuctionCard,
    winner_id: int,
    bid: int,
    lot_value: float,
    mission_bonus: Mapping[int, int],
) -> dict[int, float]:
    """Copy base scores and apply this auction to the winner only.

    Treasure adds chart value of the lot, plus any mission just completed,
    minus the bid. Invest adds only the bonus: the bid moves from coins to
    invested coins and comes back at the end. Loan subtracts the bid, because
    the coins gained now are cancelled by repaying the principal.
    """
    scores = dict(base_scores)
    if card.type is AuctionType.AUCTION:
        scores[winner_id] += (
            lot_value + float(mission_bonus.get(winner_id, 0)) - float(bid)
        )
    elif card.type is AuctionType.INVEST:
        scores[winner_id] += float(card.value)
    else:
        scores[winner_id] -= float(bid)
    return scores


def _coins_after_payment(
    base_coins: Mapping[int, int],
    card: AuctionCard,
    winner_id: int,
    bid: int,
) -> dict[int, int]:
    """Coins after the winner pays. Losers keep the coins they have now."""
    coins = dict(base_coins)
    if card.type is AuctionType.LOAN:
        coins[winner_id] = coins[winner_id] + card.value - bid
    else:
        coins[winner_id] = coins[winner_id] - bid
    return coins


def _award_leftover(
    scores: dict[int, float],
    coins_after: Mapping[int, int],
    leftover: float,
) -> None:
    """Give future lot value to the player or players with the most coins."""
    if not coins_after:
        return
    richest_coins = max(coins_after.values())
    richest = [
        player_id for player_id, coins in coins_after.items() if coins == richest_coins
    ]
    share = leftover / float(len(richest))
    for player_id in richest:
        scores[player_id] += share


def _we_are_at_least_tied(scores: Mapping[int, float], self_id: int) -> bool:
    """True when our score is greater than or equal to every other score."""
    ours = scores[self_id]
    return all(
        ours >= score for player_id, score in scores.items() if player_id != self_id
    )


def endgame_first_place_probs(
    info: BotInfo,
    candidate_bids: Sequence[int],
    opponent_distributions: Mapping[int, Sequence[tuple[int, float]]],
    rng: Random,
    samples: int = 24,
    hand_sampler: Callable[[BotInfo, Random], dict[int, tuple[GemType, ...]]]
    | None = None,
) -> dict[int, float] | None:
    """Map each candidate bid to P(our score >= every other score) after this auction.

    Returns ``None`` when there is no current auction, or when more than two
    auctions are expected after this one. A tie for the best score counts as
    success. Every candidate bid appears in the returned dict. ``samples <= 0``
    is treated as one sample.

    Each sample draws rival hidden hands once and each rival bid once. Those
    draws are reused for every candidate bid. Chart values are certain for the
    sampled display. When auctions remain after this one, the value of the
    expected leftover supply is split across the players who hold the most
    coins after the payment. At the true end of the game that leftover is 0.
    """
    card = info.current_auction
    rounds_left = estimate_rounds_left_after(info)
    if card is None or rounds_left > _ENDGAME_ROUND_LIMIT:
        return None

    sample_count = samples if samples > 0 else 1
    ordered_bids = _unique_bids(candidate_bids)
    players = _players(info)
    self_id = info.self_player.id
    lot = _treasure_lot(info, card) if card.type is AuctionType.AUCTION else ()
    mission_bonus: dict[int, int] = {}
    if card.type is AuctionType.AUCTION:
        for player in players:
            mission_bonus[player.id] = immediate_mission_bonus(
                player.won_gems, lot, info.available_missions
            )
    supplies = (
        expected_future_supplies_after_current(info) if rounds_left > 0.0 else None
    )
    base_coins = {player.id: player.coins for player in players}
    opponents = tuple(sorted(info.other_players, key=lambda player: player.id))
    successes = {bid: 0.0 for bid in ordered_bids}
    previous_winner_id = info.previous_winner_id
    num_players = info.num_players

    sampler = hand_sampler if hand_sampler is not None else sample_opponent_hidden_hands
    for _ in range(sample_count):
        hands = sampler(info, rng)
        opponent_bids = {
            opponent.id: _draw_bid(opponent_distributions.get(opponent.id), rng)
            for opponent in opponents
        }
        gem_evs = _chart_values(info, _terminal_display_counts(info, hands))
        base_scores = {
            player.id: player_known_score(player, gem_evs) for player in players
        }
        lot_value = sum(gem_evs.get(gem, 0.0) for gem in lot)
        leftover = 0.0
        if supplies is not None:
            leftover = sum(
                supplies.get(gem, 0.0) * gem_evs.get(gem, 0.0) for gem in STANDARD_GEMS
            )

        for our_bid in ordered_bids:
            full_bids = dict(opponent_bids)
            full_bids[self_id] = our_bid
            winner_id = _select_winner_id(
                full_bids, previous_winner_id, num_players, rng
            )
            winning_bid = full_bids[winner_id]
            scores = _scores_after_auction(
                base_scores,
                card,
                winner_id,
                winning_bid,
                lot_value,
                mission_bonus,
            )
            if supplies is not None:
                _award_leftover(
                    scores,
                    _coins_after_payment(base_coins, card, winner_id, winning_bid),
                    leftover,
                )
            if _we_are_at_least_tied(scores, self_id):
                successes[our_bid] += 1.0

    scale = float(sample_count)
    return {bid: successes[bid] / scale for bid in ordered_bids}
