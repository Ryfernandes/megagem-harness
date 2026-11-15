"""Configurable level-3 ablations on top of the level-2 valuation policy.

Contributors: Grace Guo (85375651b), OpenAI Codex (implementation assistance).

The default matches the CURRENT level-2 ValuationBot at all supported seat
counts. Expected-utility bids and cash-bounded public-bid learning are shared
level-2 behavior. All level-3 extras default OFF: hand inference, mission-only
endgame rules, finance-aware k, cash-fraction bid prediction, and strategic
reveals. Enabling an extra leaves the others off.
Shadow count forecasts did not establish a gameplay benefit from inference.

use_bid_evidence: optional bounded, decaying bid likelihoods reweight a uniform
allocation prior over opponents' remaining hands versus the deck. The small
joint dynamic program enforces every color total and location capacity. Gem
prices integrate only starting-hand counts over the actual Value Chart. Public
reveals and own remaining hands are counted once; deck/auction gems never enter
the final display. Reveal-policy likelihoods remain heuristic.

evidence_weight: posterior is proportional to prior * exp(weight * raw evidence).
The conservative experimental default is 0.1, versus the historical 0.4 strength.
Zero recovers the plain prior. Raw evidence decays and is clipped to +/-3;
hard card facts are never discounted. Cached beliefs include the weight.

use_endgame_rules: the original optional mission-only buying restriction,
mission-finish urgency bonus, and reserve override when the draw pile is small.

use_finance_horizon: exact remaining treasure/total-round horizons and tight
round-count bounds from auction composition, plus conservative loan funding and
investment lockup forecasts. This changes only the treasure bid's k. Future
loans are neither guaranteed to us nor added to actual cash or final score.

use_cash_fraction_model: separate per-opponent/per-auction-type histories of
validated bids divided by PRE-bid legal cash (cash+proceeds for loans). Mix a
broad normal prior with weighted empirical fractions scaled to current cash.
Treasure observations have quarter weight because lot values differ. Predict
p(win) for every bid and use EU on treasure and financial auctions. Investment
bids return; loans add debt. This learns bidding style, not hidden hand contents.

use_strategic_reveal: ascending charts reveal fewest public-display copies,
then fewest remaining missions, then random. Descending charts instead try to
cool competition for high-value future purchases. E uses the direction of its
price curve and expected final values. Owned gems, future supply and mission
progress refine descending/peaked choices. Reveals never change final prices.

use_cash_limits: condition the shared bid predictor on each rival's current
cash. This is ON in level 2 and in the default level-3 bot. All configurations
ALWAYS enforce legal own bids and retain level 2's maximum-rival spending cap;
this switch never disables the game's affordability rules.

use_bid_model: the shared level-2 public-bid learner/EU optimizer (ON by default).
It remains switchable to reproduce historical fractional-policy ablations.
With cash conditioning off,
experimental predictions may extend beyond opponents' actual budgets, while
actual actions remain legal. All modes handle first-round/clockwise ties.
Treasure bids maximize p(win) * (E[value] - (1+k)*bid). The selected k is the
existing liquidity premium plus 0.5 times a tapering future-auction horizon.
Disabling use_bid_model restores fractional bids. Historical ablation classes
retain their original k and value-discounted action cap.

Financial auctions retain the valuation policy when cash-fraction prediction is
off. Memory is pickleable and resets
between games. Private, importable classes pin all 16 flag combinations for
reproducible worker benchmarks; they are absent from the public bot selector.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from math import comb, exp, isfinite

from bot_definitions.auction_forecast import finance_opportunity_cost
from bot_definitions.cash_fraction_model import CashFractionModel, available_budget
from bot_definitions.expected_utility import (
    bid_distribution,
    completion_bonus,
    expected_utility,
    opportunity_cost,
    update_bid_rate,
    win_probability,
)
from bot_definitions.reveal_policy import choose_reveal
from bot_definitions.valuation_bot import FractionalValuationBot as ValuationBot
from bot_definitions.valuation_bot import _missing
from bots import BotInfo, GameSummary, PlayerInfo, RoundResult
from constants import STANDARD_GEMS, AuctionType, GemType

DECK = -1
Counts = tuple[int, ...]
Likelihoods = dict[int, dict[GemType, list[float]]]


@dataclass(frozen=True)
class LocationBelief:
    """Marginals of a single jointly capacity-constrained allocation model."""

    expected_counts: dict[int, dict[GemType, float]]
    location_probabilities: dict[int, dict[GemType, float]]
    hidden_count_probabilities: dict[GemType, tuple[float, ...]]
    conditional_values: dict[int, dict[GemType, tuple[float, ...]]]
    gem_values: dict[GemType, float]

    @property
    def opponent_hand_probabilities(self) -> dict[GemType, float]:
        """P(an unseen copy is in any opponent's hand, rather than the deck).

        Copies of each color are exchangeable. Prices use the full joint-model
        count PMF, not a chart lookup at this probability's expected count.
        """
        return {
            gem: sum(
                counts[gem]
                for loc, counts in self.location_probabilities.items()
                if loc != DECK
            )
            for gem in STANDARD_GEMS
        }


def infer_locations(
    info: BotInfo, likelihoods: Likelihoods, *, evidence_weight: float = 1.0
) -> LocationBelief:
    """Sum weighted allocations exactly; setup uses 36–256 hand-capacity states.

    For each color, weight an assignment by its multinomial multiplicity times
    each opponent's count likelihood. Forward/backward sums enforce the shared
    location capacities across colors, rather than treating colors independently.
    """
    if not isfinite(evidence_weight) or not 0 <= evidence_weight <= 1:
        raise ValueError("Evidence weight must be finite and between zero and one")
    opponents = info.other_players
    capacities = tuple(p.hidden_gem_count for p in opponents)
    unknown = tuple(info.unknown_gems[g] for g in STANDARD_GEMS)
    if sum(unknown) != sum(capacities) + info.gem_deck_count:
        raise ValueError("Unseen gem totals do not match known location capacities")
    zero = (0,) * len(capacities)
    options: list[list[tuple[Counts, float]]] = []
    for gem, total in zip(STANDARD_GEMS, unknown):
        choices = []
        for allocation in product(*(range(min(cap, total) + 1) for cap in capacities)):
            deck = total - sum(allocation)
            if not 0 <= deck <= info.gem_deck_count:
                continue
            multiplicity = 1
            remaining = total
            log_weight = 0.0
            for player, count in zip(opponents, allocation):
                multiplicity *= comb(remaining, count)
                remaining -= count
                table = likelihoods.get(player.id, {}).get(gem)
                if table is not None:
                    log_weight += table[count]
            choices.append(
                (allocation, multiplicity * exp(evidence_weight * log_weight))
            )
        options.append(choices)

    def advance(
        previous: dict[Counts, float],
        choices: list[tuple[Counts, float]],
        processed: int,
    ) -> dict[Counts, float]:
        result: dict[Counts, float] = {}
        for used, mass in previous.items():
            for delta, weight in choices:
                state = tuple(a + b for a, b in zip(used, delta))
                if any(a > b for a, b in zip(state, capacities)):
                    continue
                if processed - sum(state) > info.gem_deck_count:
                    continue
                result[state] = result.get(state, 0.0) + mass * weight
        return result

    forward = [{zero: 1.0}]
    processed = 0
    for total, choices in zip(unknown, options):
        processed += total
        forward.append(advance(forward[-1], choices, processed))
    backward: list[dict[Counts, float]] = [{} for _ in range(6)]
    backward[5] = {zero: 1.0}
    processed = 0
    for i in range(4, -1, -1):
        processed += unknown[i]
        backward[i] = advance(backward[i + 1], options[i], processed)
    partition = forward[-1][capacities]
    locations = (DECK, *(p.id for p in opponents))
    expected = {loc: dict.fromkeys(STANDARD_GEMS, 0.0) for loc in locations}
    probabilities = {loc: dict.fromkeys(STANDARD_GEMS, 0.0) for loc in locations}
    hidden_pmfs: dict[GemType, tuple[float, ...]] = {}
    conditional: dict[int, dict[GemType, tuple[float, ...]]] = {
        p.id: {} for p in opponents
    }
    values: dict[GemType, float] = {}
    own_hand = Counter(info.self_player.hidden_gems)
    for index, (gem, total) in enumerate(zip(STANDARD_GEMS, unknown)):
        pmf = [0.0] * (total + 1)
        masses = [[0.0] * 7 for _ in opponents]
        value_sums = [[0.0] * 7 for _ in opponents]
        # Final display = already revealed starting cards + our remaining hand
        # + opponents' remaining hands. Auction/deck gems never enter it.
        known = info.value_display.get(gem, 0) + own_hand[gem]
        for delta, weight in options[index]:
            mass = 0.0
            for used, prefix in forward[index].items():
                tail = tuple(c - a - b for c, a, b in zip(capacities, used, delta))
                mass += prefix * weight * backward[index + 1].get(tail, 0.0)
            mass /= partition
            count = sum(delta)  # Opponent-hand copies only; exclude deck copies.
            pmf[count] += mass
            price = info.value_chart.get_value(known + count)
            expected[DECK][gem] += mass * (total - count)
            for i, (p, n) in enumerate(zip(opponents, delta)):
                expected[p.id][gem] += mass * n
                masses[i][n] += mass
                value_sums[i][n] += mass * price
        hidden_pmfs[gem] = tuple(pmf)
        values[gem] = sum(
            probability * info.value_chart.get_value(known + n)
            for n, probability in enumerate(pmf)
        )
        for loc in locations:
            probabilities[loc][gem] = expected[loc][gem] / total if total else 0.0
        for i, p in enumerate(opponents):
            conditional[p.id][gem] = tuple(
                value_sums[i][n] / masses[i][n] if masses[i][n] else values[gem]
                for n in range(7)
            )
    return LocationBelief(expected, probabilities, hidden_pmfs, conditional, values)


class SoftTrackingBot(ValuationBot):
    """Expected-utility bids with individually switchable experimental features."""

    use_bid_evidence = False
    use_endgame_rules = False
    use_finance_horizon = False
    use_cash_fraction_model = False
    use_strategic_reveal = False
    use_cash_limits = True
    use_bid_model = True
    # Posterior allocation weight = prior * exp(evidence_weight * raw evidence).
    # 0 is the plain prior; 0.4 reproduces the former update strength.
    evidence_weight = 0.1
    utility_k_scale = 1.0
    utility_k_floor = 0.5
    utility_legacy_cap = False
    utility_player_counts: tuple[int, ...] = (3, 4, 5)
    cash_fraction_auction_types: tuple[AuctionType, ...] = (
        AuctionType.AUCTION,
        AuctionType.INVEST,
        AuctionType.LOAN,
    )

    def __init__(
        self,
        *,
        use_bid_evidence: bool | None = None,
        use_endgame_rules: bool | None = None,
        use_finance_horizon: bool | None = None,
        use_cash_fraction_model: bool | None = None,
        use_strategic_reveal: bool | None = None,
        use_cash_limits: bool | None = None,
        use_bid_model: bool | None = None,
        evidence_weight: float | None = None,
        utility_k_scale: float | None = None,
        utility_k_floor: float | None = None,
        utility_player_counts: tuple[int, ...] | None = None,
    ) -> None:
        self.use_bid_evidence = (
            type(self).use_bid_evidence
            if use_bid_evidence is None
            else use_bid_evidence
        )
        self.use_endgame_rules = (
            type(self).use_endgame_rules
            if use_endgame_rules is None
            else use_endgame_rules
        )
        self.use_finance_horizon = (
            type(self).use_finance_horizon
            if use_finance_horizon is None
            else use_finance_horizon
        )
        self.use_cash_fraction_model = (
            type(self).use_cash_fraction_model
            if use_cash_fraction_model is None
            else use_cash_fraction_model
        )
        self.use_strategic_reveal = (
            type(self).use_strategic_reveal
            if use_strategic_reveal is None
            else use_strategic_reveal
        )
        self.use_cash_limits = (
            type(self).use_cash_limits if use_cash_limits is None else use_cash_limits
        )
        self.use_bid_model = (
            type(self).use_bid_model if use_bid_model is None else use_bid_model
        )
        if self.use_finance_horizon and not self.use_bid_model:
            raise ValueError("Finance horizon requires expected-utility bidding")
        if self.use_cash_fraction_model and not (
            self.use_bid_model and self.use_cash_limits
        ):
            raise ValueError("Cash-fraction model requires cash-bounded EU bidding")
        self.evidence_weight = (
            type(self).evidence_weight if evidence_weight is None else evidence_weight
        )
        if not isfinite(self.evidence_weight) or not 0 <= self.evidence_weight <= 1:
            raise ValueError("Evidence weight must be finite and between zero and one")
        self.utility_k_scale = (
            type(self).utility_k_scale if utility_k_scale is None else utility_k_scale
        )
        self.utility_k_floor = (
            type(self).utility_k_floor if utility_k_floor is None else utility_k_floor
        )
        if not all(
            isfinite(x) and x >= 0 for x in (self.utility_k_scale, self.utility_k_floor)
        ):
            raise ValueError("Utility penalties must be finite and nonnegative")
        self.utility_player_counts = (
            type(self).utility_player_counts
            if utility_player_counts is None
            else utility_player_counts
        )
        if not self.utility_player_counts or any(
            count not in (3, 4, 5) for count in self.utility_player_counts
        ):
            raise ValueError("Utility player counts must be drawn from 3, 4, 5")
        self.on_game_start()

    def on_game_start(self) -> None:
        self.likelihoods: Likelihoods = {}
        self.bid_rates: dict[int, float] = {}
        self.cash_fraction_model = CashFractionModel()
        self.opponent_coins: dict[int, int] = {}
        self.revealed_counts: dict[GemType, int] = {}
        self.rounds_observed = 0
        self._pending: BotInfo | None = None
        self._cached_info: BotInfo | None = None
        self._cached_belief: LocationBelief | None = None
        self._cached_evidence_enabled: bool | None = None
        self._cached_evidence_weight: float | None = None

    def _observe(self, info: BotInfo) -> None:
        self.opponent_coins = {p.id: p.coins for p in info.other_players}
        self.revealed_counts = dict(info.value_display)

    def observe_auction(self, info: BotInfo) -> None:
        """Remember a pre-bid view, also usable by a non-playing shadow tracker."""
        self._pending = info
        self._observe(info)

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        if not self.use_strategic_reveal:
            return super().choose_hidden_gem(info, bids)
        return choose_reveal(
            info,
            self.gem_values(info),
            {gem: self.deck_fraction(info, gem) for gem in STANDARD_GEMS},
        )

    def on_game_end(self, info: BotInfo, result: GameSummary) -> None:
        self._observe(info)
        self._pending = None
        self._cached_info = None
        self._cached_belief = None

    def beliefs(self, info: BotInfo) -> LocationBelief:
        if (
            self._cached_info != info
            or self._cached_belief is None
            or self._cached_evidence_enabled != self.use_bid_evidence
            or self._cached_evidence_weight != self.evidence_weight
        ):
            self._cached_belief = infer_locations(
                info,
                self.likelihoods if self.use_bid_evidence else {},
                evidence_weight=self.evidence_weight,
            )
            self._cached_info = info
            self._cached_evidence_enabled = self.use_bid_evidence
            self._cached_evidence_weight = self.evidence_weight
        return self._cached_belief

    def gem_values(self, info: BotInfo) -> dict[GemType, float]:
        if not self.use_bid_evidence or self.evidence_weight == 0:
            return self.expected_gem_values(info)
        return self.beliefs(info).gem_values

    def deck_fraction(self, info: BotInfo, gem: GemType) -> float:
        if not self.use_bid_evidence or self.evidence_weight == 0:
            return super().deck_fraction(info, gem)
        count = info.unknown_gems[gem]
        return self.beliefs(info).expected_counts[DECK][gem] / count if count else 0.0

    _completion_bonus = staticmethod(completion_bonus)

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        evidence_active = self.use_bid_evidence and self.evidence_weight > 0
        if not evidence_active:
            self.likelihoods.clear()
        before = self._pending
        if before is not None and before.round_number == info.round_number:
            if (
                self.use_cash_fraction_model
                and before.current_auction == result.auction_card
            ):
                for player in before.other_players:
                    self.cash_fraction_model.observe(
                        before, player, result.bids[player.id]
                    )
            belief = self.beliefs(before) if evidence_active else None
            if evidence_active:
                for tables in self.likelihoods.values():
                    for logs in tables.values():
                        logs[:] = [value * 0.96 for value in logs]
            if result.auction_card.type is AuctionType.AUCTION:
                lot = before.auction_gems[: result.auction_card.value]
                prices = self.gem_values(before)
                gem_value = sum(prices[g] for g in lot)
                for player in before.other_players:
                    bid = result.bids[player.id]
                    bonus = self._completion_bonus(before, player)
                    item_value = gem_value + bonus
                    expected_bid = max(1.0, self.bid_fraction * item_value)
                    # An unaffordable normal bid is censored evidence, not a pass.
                    if belief is not None:
                        confidence = min(1.0, player.coins / max(1.0, 2 * expected_bid))
                        surprise = max(
                            -1.0, min(1.0, (bid - expected_bid) / expected_bid)
                        )
                        if bid == 0 and bonus > 0:
                            confidence *= 0.75
                        tables = self.likelihoods.setdefault(player.id, {})
                        for gem in dict.fromkeys(lot):
                            logs = tables.setdefault(gem, [0.0] * 7)
                            for n in range(7):
                                relative_value = (
                                    belief.conditional_values[player.id][gem][n]
                                    - prices[gem]
                                ) / 10.0
                                # Store raw evidence, then temper it in inference.
                                # Scaling the old +/-1.2 bound by 1/0.4 preserves
                                # the legacy model at evidence_weight=0.4.
                                change = confidence * surprise * relative_value
                                logs[n] = max(-3.0, min(3.0, logs[n] + change))
                    if self.use_bid_model:
                        update_bid_rate(
                            self.bid_rates, player, item_value, bid, self.bid_fraction
                        )
            self.rounds_observed += 1
        # A reveal is a hard fact even if the preceding bid callback failed.
        if evidence_active and result.revealed_gem is not None:
            revealed_logs = self.likelihoods.get(result.winner_id, {}).get(
                result.revealed_gem
            )
            if revealed_logs is not None:
                revealed_logs[:] = revealed_logs[1:] + revealed_logs[-1:]
        self._pending = None
        self._cached_info = None
        self._cached_belief = None
        self._observe(info)

    def _bid_distribution(self, info: BotInfo, player: PlayerInfo) -> list[float]:
        if (
            self.use_cash_fraction_model
            and info.current_auction is not None
            and info.current_auction.type in self.cash_fraction_auction_types
        ):
            return self.cash_fraction_model.distribution(info, player)
        return bid_distribution(
            info,
            player,
            self.gem_values(info),
            self.bid_rates,
            self.bid_fraction,
            cash_limits=self.use_cash_limits,
        )

    win_probability = staticmethod(win_probability)

    def _endgame_adjustment(self, info: BotInfo) -> tuple[bool, float]:
        """The original optional mission-only rule and urgency bonus."""
        card = info.current_auction
        if not self.use_endgame_rules or info.gem_deck_count > 2 or card is None:
            return True, 0.0
        lot = info.auction_gems[: card.value]
        after = Counter(info.self_player.won_gems)
        after.update(lot)
        visible = Counter(info.auction_gems[len(lot) :])
        prices = self.gem_values(info)
        needed = any(
            _missing(m, after) < _missing(m, info.self_player.won_gems)
            and self._mission_potential(
                info, m, info.self_player, after, visible, prices
            )
            > 0
            for m in info.available_missions
        )
        denial = False
        for player in info.other_players:
            rival = Counter(player.won_gems)
            rival.update(lot)
            denial |= any(
                _missing(m, rival) <= 1
                and _missing(m, rival) < _missing(m, player.won_gems)
                and self._mission_potential(info, m, player, rival, visible, prices) > 0
                for m in info.available_missions
            )
        completion = float(
            sum(
                m.value
                for m in info.available_missions
                if _missing(m, info.self_player.won_gems) == 1 and m.is_satisfied(after)
            )
        )
        return needed or denial, completion

    def get_bid(self, info: BotInfo) -> int:
        self.observe_auction(info)
        card = info.current_auction
        if card is None:
            return 0
        if (
            self.use_cash_fraction_model
            and card.type is not AuctionType.AUCTION
            and card.type in self.cash_fraction_auction_types
            and info.num_players in self.utility_player_counts
        ):
            return self._cash_fraction_financial_bid(info)
        if (
            card.type is not AuctionType.AUCTION
            or not self.use_bid_model
            or info.num_players not in self.utility_player_counts
        ):
            bid = super().get_bid(info)
            if card.type is AuctionType.AUCTION and self.use_endgame_rules:
                allowed, completion = self._endgame_adjustment(info)
                if not allowed:
                    return 0
                if completion:
                    estimate = self.evaluate(info)
                    reserve = (
                        0
                        if info.gems_remaining <= card.value
                        else min(1, info.self_player.coins)
                    )
                    bid = min(
                        info.self_player.coins - reserve,
                        int(
                            self.bid_fraction
                            * (estimate.treasure + 0.5 * completion)
                            / estimate.liquidity
                        ),
                        int(0.95 * estimate.treasure / estimate.liquidity),
                        max(p.coins for p in info.other_players) + 1,
                    )
            return bid
        estimate = self.evaluate(info)
        cash = info.self_player.coins
        budget = max(0, cash - estimate.reserve)
        value = estimate.treasure
        allowed, completion = self._endgame_adjustment(info)
        if not allowed:
            return 0
        if completion:
            budget = cash - (0 if info.gems_remaining <= card.value else min(1, cash))
            value += 0.5 * completion
        k = opportunity_cost(
            info, estimate.liquidity, self.utility_k_scale, self.utility_k_floor
        )
        if self.use_finance_horizon:
            k = finance_opportunity_cost(
                info,
                estimate.liquidity,
                estimate.future_points,
                self.utility_k_scale,
                self.utility_k_floor,
                self.bid_fraction,
            )
        if self.utility_legacy_cap:
            ceiling = min(budget, int(0.95 * estimate.treasure / estimate.liquidity))
        else:
            # Bids above every rival's current cash cannot improve win odds.
            # Within that exact dominance bound, examine every affordable bid.
            ceiling = min(budget, max(p.coins for p in info.other_players) + 1)
        distributions = {
            p.id: self._bid_distribution(info, p) for p in info.other_players
        }
        return max(
            range(max(0, ceiling) + 1),
            key=lambda bid: expected_utility(
                self.win_probability(info, bid, distributions), value, bid, k
            ),
        )

    def _cash_fraction_financial_bid(self, info: BotInfo) -> int:
        """Use type-specific win odds with the existing financial value estimates.

        Investments return their bid: profit is face - opportunity_cost * bid.
        Loan principal and debt cancel: benefit is productive liquidity minus
        the bid and its opportunity cost. Neither loan proceeds nor locked
        investment bids are mistaken for free final-score profit.
        """
        card = info.current_auction
        assert card is not None and card.type is not AuctionType.AUCTION
        estimate = self.evaluate(info)
        premium = estimate.liquidity - 1
        budget = max(0, info.self_player.coins - estimate.reserve)
        if card.type is AuctionType.INVEST:
            k = max(self.investment_premium_floor, premium)
            value = float(card.value)
        else:
            k = premium
            value = premium * min(card.value, estimate.productive_loan)
            budget = min(available_budget(info, info.self_player), card.value - 1)
        ceiling = min(
            budget, max(available_budget(info, p) for p in info.other_players) + 1
        )
        distributions = {
            p.id: self._bid_distribution(info, p) for p in info.other_players
        }
        return max(
            range(ceiling + 1),
            key=lambda bid: expected_utility(
                self.win_probability(info, bid, distributions),
                value + bid if card.type is AuctionType.INVEST else value,
                bid,
                k,
            ),
        )


# Importable, private worker classes pin ALL settings for reproducible ablations.
# They are deliberately absent from custom_bots.__all__ and the viewer selector.
FEATURE_NAMES = ("inference", "endgame", "cash_limits", "bid_model")
ABLATION_BOTS: dict[str, type[SoftTrackingBot]] = {}
for _flags in product((False, True), repeat=4):
    _label = "+".join(name for name, enabled in zip(FEATURE_NAMES, _flags) if enabled)
    _label = _label or "baseline"
    _name = "_Ablation" + "".join(str(int(value)) for value in _flags)
    _variant: type[SoftTrackingBot] = type(
        _name,
        (SoftTrackingBot,),
        {
            "__module__": __name__,
            "use_bid_evidence": _flags[0],
            "use_endgame_rules": _flags[1],
            "use_finance_horizon": False,
            "use_cash_fraction_model": False,
            "use_strategic_reveal": False,
            "use_cash_limits": _flags[2],
            "use_bid_model": _flags[3],
            "evidence_weight": 0.4,  # Pin historical ablations to legacy strength.
            "utility_legacy_cap": True,
            "utility_k_scale": 1.0,
            "utility_k_floor": 0.0,
            "utility_player_counts": (3, 4, 5),
        },
    )
    globals()[_name] = _variant
    ABLATION_BOTS[_label] = _variant

# Common strengths are importable in spawned gameplay workers as well as usable
# by the host-only shadow evaluator. Historical ABLATION_BOTS stay pinned to 0.4.
EVIDENCE_WEIGHTS = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 1.0)
WEIGHTED_ABLATION_BOTS: dict[tuple[str, float], type[SoftTrackingBot]] = {}
for _label, _base in ABLATION_BOTS.items():
    for _weight in EVIDENCE_WEIGHTS:
        if _weight == 0.4:
            WEIGHTED_ABLATION_BOTS[(_label, _weight)] = _base
            continue
        _name = _base.__name__ + "_W" + str(_weight).replace(".", "_")
        _variant = type(
            _name,
            (_base,),
            {"__module__": __name__, "evidence_weight": _weight},
        )
        globals()[_name] = _variant
        WEIGHTED_ABLATION_BOTS[(_label, _weight)] = _variant


# Matched EU experiments: same valuation, finances, reserve, win model and prior.
# Only the opportunity cost changes. Historical optimizer remains a control.
UTILITY_SETTINGS = {
    "myopic": (0.0, 0.0),
    "heuristic": (1.0, 0.0),
    "scale2": (2.0, 0.0),
    "scale4": (4.0, 0.0),
    "floor0.5": (1.0, 0.5),
    "floor1": (1.0, 1.0),
    "floor2": (1.0, 2.0),
    "scale2_floor1": (2.0, 1.0),
}
UTILITY_BOTS: dict[str, type[ValuationBot]] = {
    "before": ValuationBot,
    "old_eu": ABLATION_BOTS["cash_limits+bid_model"],
}
for _label, (_scale, _floor) in UTILITY_SETTINGS.items():
    _name = "_Utility_" + _label.replace(".", "_")
    _utility_variant: type[SoftTrackingBot] = type(
        _name,
        (SoftTrackingBot,),
        {
            "__module__": __name__,
            "use_bid_evidence": False,
            "use_endgame_rules": False,
            "use_finance_horizon": False,
            "use_cash_fraction_model": False,
            "use_strategic_reveal": False,
            "use_cash_limits": True,
            "use_bid_model": True,
            "utility_k_scale": _scale,
            "utility_k_floor": _floor,
            "utility_legacy_cap": False,
            "utility_player_counts": (3, 4, 5),
        },
    )
    globals()[_name] = _utility_variant
    UTILITY_BOTS[_label] = _utility_variant


# Current level-3 switches, separate from the frozen historical four-flag grid.
# Cash-bounded EU bidding is now part of level 2, not a level-3 extra.
LEVEL3_FEATURES = (
    "use_bid_evidence",
    "use_endgame_rules",
    "use_finance_horizon",
    "use_cash_fraction_model",
    "use_strategic_reveal",
)
LEVEL3_BOTS: dict[tuple[bool, ...], type[SoftTrackingBot]] = {}
for _flags in product((False, True), repeat=len(LEVEL3_FEATURES)):
    # Keep previous three-extra worker import paths stable when the new flag is off.
    _name = "_Level3_" + "".join(str(int(value)) for value in _flags[:3])
    if _flags[3]:
        _name += "_CashFractions"
    if _flags[4]:
        _name += "_Reveal"
    _variant = type(
        _name,
        (SoftTrackingBot,),
        {
            "__module__": __name__,
            **dict(zip(LEVEL3_FEATURES, _flags)),
            "use_cash_limits": True,
            "use_bid_model": True,
            "evidence_weight": 0.1,
            "utility_k_scale": 1.0,
            "utility_k_floor": 0.5,
            "utility_legacy_cap": False,
            "utility_player_counts": (3, 4, 5),
            "cash_fraction_auction_types": (
                AuctionType.AUCTION,
                AuctionType.INVEST,
                AuctionType.LOAN,
            ),
        },
    )
    globals()[_name] = _variant
    LEVEL3_BOTS[_flags] = _variant

WEIGHTED_LEVEL3_BOTS: dict[tuple[tuple[bool, ...], float], type[SoftTrackingBot]] = {}
for _flags, _base in LEVEL3_BOTS.items():
    for _weight in EVIDENCE_WEIGHTS:
        if _weight == 0.1:
            WEIGHTED_LEVEL3_BOTS[(_flags, _weight)] = _base
            continue
        _name = _base.__name__ + "_W" + str(_weight).replace(".", "_")
        _variant = type(
            _name,
            (_base,),
            {"__module__": __name__, "evidence_weight": _weight},
        )
        globals()[_name] = _variant
        WEIGHTED_LEVEL3_BOTS[(_flags, _weight)] = _variant


# Isolate the distribution change from extending EU to financial auctions.
class _CashFractionTreasureOnlyBot(SoftTrackingBot):
    use_bid_evidence = False
    use_endgame_rules = False
    use_finance_horizon = False
    use_cash_fraction_model = True
    use_strategic_reveal = False
    use_bid_model = True
    use_cash_limits = True
    utility_k_scale = 1.0
    utility_k_floor = 0.5
    utility_legacy_cap = False
    utility_player_counts = (3, 4, 5)
    cash_fraction_auction_types = (AuctionType.AUCTION,)
