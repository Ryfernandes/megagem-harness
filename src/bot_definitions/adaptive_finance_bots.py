"""Adaptive Megagem bots built on exact posterior values and surplus bidding.

Contributors: Julian Herbst

Strategy
--------
Every bot here shares one core (EVCore) and differs only in how it prices
financing auctions:

1. Valuation. The final value display equals the current display plus every
   hidden hand (auction-won gems never join it), so the end-of-game count of
   each color follows a hypergeometric distribution over the unseen pool,
   conditioned on our own hidden hand. Summing the value chart over that
   posterior gives an exact expected coin value per gem, hence per lot,
   plus any mission bonuses the lot completes.

2. Bidding. Opponents are modelled as bidding a learned fraction of their
   own estimated value: we watch what each rival actually pays per auction
   and shrink toward a prior, so the bot stays cheap at a quiet table and
   keeps up at a hot one. Bids maximize expected surplus,
   (value - (1 + lambda) * bid) * P(win | bid), where P(win) comes from the
   opponent bid model with the official clockwise tiebreak, and lambda is a
   pacing shadow price that makes coins expensive early and cheaper as the
   deck runs out.

3. Financing. Loans are worth the extra bids their coins unlock, priced at
   the same shadow price, so the bot pays for liquidity exactly when coins
   are scarce and valuable. Investment lock-in is priced the same way
   instead of at a flat rate.

4. Reveals. After a win we reveal the hidden gem whose price impact on
   future lots is smallest.

Everything is closed form; the most expensive call is a 7-point
hypergeometric pmf, so callbacks finish in microseconds, far inside the
2-second tournament budget.

Submitted bots (strongest last):
  ConservativeAdaptiveBot  - the core strategy above with flat-rate financing.
  LoanAwareAdaptiveBot     - + loans priced by the liquidity they unlock.
  FinanceAwareAdaptiveBot  - + investment lock-in priced at the shadow price.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping

from bots import Bot, BotInfo, RoundResult
from constants import (
    CARDS_PER_PLAYER,
    COINS_PER_PLAYER,
    STANDARD_GEMS,
    AuctionType,
    GemType,
)

TOTAL_GEMS = 30  # six copies of each of the five colors


def hypergeom_pmf(population: int, successes: int, draws: int) -> list[float]:
    """P(X=k) for k = 0..min(successes, draws), X ~ Hypergeometric(pop, succ, draws).

    Used for: how many gems of one color sit in the opponents' hidden hands,
    given the counts still unseen (their hands plus the hidden draw pile).
    """
    k_max = min(successes, draws)
    if draws <= 0 or successes <= 0 or population <= 0:
        return [1.0] + [0.0] * k_max
    denom = math.comb(population, draws)
    pmf = []
    for k in range(k_max + 1):
        if draws - k > population - successes:
            pmf.append(0.0)
        else:
            pmf.append(
                math.comb(successes, k)
                * math.comb(population - successes, draws - k)
                / denom
            )
    return pmf


class EVCore(Bot):
    """Shared valuation/bidding machinery; subclasses flip ladder switches."""

    # Ladder switches.
    use_missions = False
    use_surplus = False
    use_pacing = False
    use_financing = False
    smart_reveal = False
    adaptive_opp = False

    # Parameters (class attributes so experiments can subclass and sweep).
    shade = 3  # opponents modelled as bidding uniformly in [value - shade, value]
    lam0 = 0.35  # pacing shadow price: base cost of a coin
    kappa = 1.2  # pacing: extra cost when spending ahead of pace
    invest_lock_cost = 0.15  # per-coin opportunity cost of locking an investment
    loan_eagerness = 3  # max coins to pay for a loan when liquidity is needed
    loan_min_gems_left = 4  # loans stop being useful near the end
    opp_prior_ratio = 0.75  # prior: opponents bid ~75% of their value estimate
    opp_prior_weight = 3.0  # pseudo-observations behind the prior
    opp_ratio_cap = 3.0  # ignore insane single-auction ratios
    invest_v2 = False  # price investment lock-in at the pacing shadow price
    loan_v2 = False  # value loans by the liquidity they unlock

    # Per-game adaptation state (opponent ids are seat indices per game).
    def on_game_start(self) -> None:
        """Reset per-game state; never let a callback crash the runner."""
        try:
            self._on_game_start_impl()
        except Exception:
            pass

    def _on_game_start_impl(self) -> None:
        self._opp_ratio_sum: dict[int, float] = {}
        self._opp_ratio_n: dict[int, int] = {}
        self._last_opp_values: dict[int, float] = {}
        self._my_id = -1

    def _opp_ratio(self, seat: int) -> float:
        """Shrunk estimate of how much of their value an opponent bids."""
        s = self._opp_ratio_sum.get(seat, 0.0)
        n = self._opp_ratio_n.get(seat, 0)
        return (s + self.opp_prior_weight * self.opp_prior_ratio) / (
            n + self.opp_prior_weight
        )

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        """Learn from the round; learning must never crash the runner."""
        try:
            self._on_round_end_impl(info, result)
        except Exception:
            pass

    def _on_round_end_impl(self, info: BotInfo, result: RoundResult) -> None:
        # Learn each opponent's bid-to-value ratio from treasure auctions.
        if not self.adaptive_opp or not self._last_opp_values:
            return
        if result.auction_card.type is not AuctionType.AUCTION:
            return
        for seat, value in self._last_opp_values.items():
            bid = result.bids.get(seat)
            if bid is None or seat == self._my_id or value < 1.0:
                continue
            ratio = min(bid / value, self.opp_ratio_cap)
            self._opp_ratio_sum[seat] = self._opp_ratio_sum.get(seat, 0.0) + ratio
            self._opp_ratio_n[seat] = self._opp_ratio_n.get(seat, 0) + 1

    # ----- exact posterior values -------------------------------------

    def expected_gem_values(self, info: BotInfo) -> dict[GemType, float]:
        """Exact E[chart value per gem] for each color.

        The final display = current display + own hidden hand + opponents'
        hidden hands (auction-won gems never join the display). Opponents'
        hidden gems are a random subset of the unseen pool, so the final
        count of each color is hypergeometric. Mapping the whole count
        distribution through the chart handles non-linear charts correctly
        (E[chart(n)] instead of chart(E[n])).
        """
        hidden_opp = sum(p.hidden_gem_count for p in info.other_players)
        pool = sum(info.unknown_gems.values())
        own = Counter(info.self_player.hidden_gems)
        chart = info.value_chart
        values = {}
        for gem in STANDARD_GEMS:
            base = info.value_display.get(gem, 0) + own.get(gem, 0)
            pmf = hypergeom_pmf(pool, info.unknown_gems.get(gem, 0), hidden_opp)
            values[gem] = sum(
                p * chart.get_value(min(base + k, 6)) for k, p in enumerate(pmf)
            )
        return values

    def lot_gems(self, info: BotInfo) -> Counter:
        """Gems the current treasure auction awards (first gem first)."""
        card = info.current_auction
        if card is None or card.type is not AuctionType.AUCTION:
            return Counter()
        return Counter(info.auction_gems[: card.value])

    @staticmethod
    def mission_gain(missions, have: Counter, gained: Counter) -> float:
        """Exact value of the missions `gained` would newly complete."""
        if not gained:
            return 0.0
        new = have + gained
        return sum(
            m.value
            for m in missions
            if not m.is_satisfied(have) and m.is_satisfied(new)
        )

    def my_lot_value(self, info: BotInfo, values: dict[GemType, float]) -> float:
        gained = self.lot_gems(info)
        v = sum(values[g] * n for g, n in gained.items())
        if self.use_missions:
            v += self.mission_gain(
                info.available_missions, Counter(info.self_player.won_gems), gained
            )
        return v

    def opp_lot_values(self, info: BotInfo, values: dict[GemType, float]) -> list[float]:
        gained = self.lot_gems(info)
        base = sum(values[g] * n for g, n in gained.items())
        out = []
        for p in info.other_players:
            v = base
            if self.use_missions:
                v += self.mission_gain(
                    info.available_missions, Counter(p.won_gems), gained
                )
            out.append(v)
        return out

    # ----- bidding -----------------------------------------------------

    def bid_range(self, value: float, coins: int) -> tuple[int, int]:
        """Model an opponent's bid as uniform over integers [lo, hi]."""
        hi = max(0, min(int(round(value)), coins))
        lo = max(0, min(hi, int(round(value)) - self.shade))
        return lo, hi

    @staticmethod
    def win_probability(
        b: int, ranges: list[tuple[int, int, int]], my_id: int, prev, n: int
    ) -> float:
        """P(b wins) against independent uniform opponents, with tiebreaks.

        Ties resolve clockwise starting left of the previous winner; the first
        auction breaks ties uniformly at random (approximated per opponent).
        """
        p = 1.0
        for lo, hi, seat in ranges:
            width = hi - lo + 1
            if b < lo:
                return 0.0
            if b > hi:
                continue  # certain to beat this opponent
            p_lt = (b - lo) / width
            p_eq = 1.0 / width
            if prev is None:
                tie = 0.5
            else:
                tie = (
                    1.0
                    if ((my_id - prev - 1) % n) < ((seat - prev - 1) % n)
                    else 0.0
                )
            p *= p_lt + tie * p_eq
        return p

    def pacing_lambda(self, info: BotInfo) -> float:
        """Shadow price of a coin: positive early, fading as auctions deplete."""
        total = TOTAL_GEMS - info.num_players * CARDS_PER_PLAYER[info.num_players]
        progress = 1.0 - info.gems_remaining / total if total else 1.0
        start = COINS_PER_PLAYER[info.num_players]
        spend = 1.0 - info.self_player.coins / start
        return max(0.0, self.lam0 + self.kappa * (spend - progress))

    def _surplus_bid(
        self, info: BotInfo, surplus: float, opp_values: list[float], cap: int
    ) -> int:
        """argmax_b (surplus - (1+lam)*b) * P(win|b) over affordable bids."""
        lam = self.pacing_lambda(info) if self.use_pacing else 0.0
        if self.adaptive_opp:
            ranges = []
            for v, p in zip(opp_values, info.other_players):
                pred = self._opp_ratio(p.id) * v
                hi = max(0, min(int(round(pred)) + 1, p.coins))
                lo = max(0, min(hi, int(round(pred)) - 6))
                ranges.append((lo, hi, p.id))
        else:
            ranges = [
                (*self.bid_range(v, p.coins), p.id)
                for v, p in zip(opp_values, info.other_players)
            ]
        best_b, best_score = 0, 0.0
        for b in range(0, max(0, cap) + 1):
            p = self.win_probability(
                b, ranges, info.self_player.id, info.previous_winner_id,
                info.num_players,
            )
            score = (surplus - (1.0 + lam) * b) * p
            if score > best_score + 1e-12:
                best_b, best_score = b, score
        return best_b

    def get_bid(self, info: BotInfo) -> int:
        """Bid; on any unexpected state, pass rather than crash the runner."""
        try:
            return self._get_bid_impl(info)
        except Exception:
            return 0

    def _get_bid_impl(self, info: BotInfo) -> int:
        card = info.current_auction
        if card is None:
            return 0
        coins = info.self_player.coins

        if card.type is AuctionType.AUCTION:
            values = self.expected_gem_values(info)
            v_opps = self.opp_lot_values(info, values)
            self._my_id = info.self_player.id
            self._last_opp_values = {
                p.id: v for v, p in zip(v_opps, info.other_players)
            }
            v_my = self.my_lot_value(info, values)
            if v_my < 1.0:
                return 0
            if not self.use_surplus:
                return min(coins, max(0, int(round(v_my))))
            return self._surplus_bid(
                info, v_my, v_opps, min(coins, int(v_my))
            )

        if card.type is AuctionType.INVEST:
            if not self.use_financing:
                return 0
            # Winning at b locks b until scoring and returns b + bonus: the
            # certain gain is the bonus, the cost is the locked cash. The
            # lock cost per coin is the pacing shadow price: locking cash is
            # expensive exactly when liquidity is scarce.
            lock_cost = (
                self.pacing_lambda(info)
                if self.invest_v2
                else self.invest_lock_cost
            )
            opp_caps = [min(card.value, p.coins) for p in info.other_players]
            ranges = [(0, c, p.id) for c, p in zip(opp_caps, info.other_players)]
            best_b, best_score = 0, 0.0
            for b in range(0, coins + 1):
                p_win = self.win_probability(
                    b, ranges, info.self_player.id, info.previous_winner_id,
                    info.num_players,
                )
                score = (card.value - lock_cost * b) * p_win
                if score > best_score + 1e-12:
                    best_b, best_score = b, score
            return best_b

        if card.type is AuctionType.LOAN:
            if not self.use_financing:
                return 0
            if info.gems_remaining < self.loan_min_gems_left:
                return 0
            if not self.loan_v2:
                # A loan is pure liquidity: the principal cancels at
                # scoring, so only pay for it when we are poorer than every
                # opponent.
                poorest = all(
                    coins < p.coins for p in info.other_players
                )
                if not poorest:
                    return 0
                return min(self.loan_eagerness, coins + card.value)
            # The principal cancels at scoring, so a loan is worth only the
            # profitable spending its liquidity unlocks: the pacing shadow
            # price per coin times the coins provided, zero when cash is
            # already plentiful. The bid itself is a pure scoring cost.
            lam = self.pacing_lambda(info)
            v_loan = lam * card.value
            if v_loan < 1.0:
                return 0
            ranges = [
                (0, min(3, p.coins + card.value), p.id)
                for p in info.other_players
            ]
            cap = min(coins + card.value, int(v_loan))
            best_b, best_score = 0, 0.0
            for b in range(0, max(0, cap) + 1):
                p_win = self.win_probability(
                    b, ranges, info.self_player.id, info.previous_winner_id,
                    info.num_players,
                )
                score = (v_loan - (1.0 + lam) * b) * p_win
                if score > best_score + 1e-12:
                    best_b, best_score = b, score
            return best_b

        return 0

    # ----- reveal ------------------------------------------------------

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        """Reveal; on any unexpected state, reveal the first card instead."""
        try:
            return self._choose_hidden_gem_impl(info, bids)
        except Exception:
            return 0

    def _choose_hidden_gem_impl(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        hand = info.self_player.hidden_gems
        if not self.smart_reveal:
            return random.randrange(len(hand))
        # Revealing raises the display count and hence the market price of
        # future lots of that color. Reveal the color whose price move
        # matters least: few of that color left to auction, and a small
        # marginal chart step at the current display count.
        chart = info.value_chart
        pool = sum(info.unknown_gems.values())
        deck = info.gem_deck_count
        best_i, best_impact = 0, float("inf")
        for i, gem in enumerate(hand):
            disp = info.value_display.get(gem, 0)
            gain = max(
                0.0, chart.get_value(min(disp + 1, 6)) - chart.get_value(disp)
            )
            future_supply = info.auction_gems.count(gem) + (
                info.unknown_gems.get(gem, 0) * deck / pool if pool else 0.0
            )
            impact = future_supply * gain
            if impact < best_impact:
                best_i, best_impact = i, impact
        return best_i



class ConservativeAdaptiveBot(EVCore):
    """Full strategy with conservative pacing; flat-rate financing rules."""

    use_missions = True
    use_surplus = True
    smart_reveal = True
    use_pacing = True
    use_financing = True
    adaptive_opp = True
    lam0 = 0.7
    opp_prior_ratio = 0.7


class LoanAwareAdaptiveBot(ConservativeAdaptiveBot):
    """Adds liquidity-priced loans: pay for coins exactly when scarce."""

    loan_v2 = True


class FinanceAwareAdaptiveBot(ConservativeAdaptiveBot):
    """Adds shadow-priced investment lock-in on top of liquidity loans."""

    invest_v2 = True
    loan_v2 = True
