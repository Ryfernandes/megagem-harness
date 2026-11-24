"""Market Reader: model each opponent's bids, then bid just enough.

Self-contained (standard library and harness modules only). The first half is
the shared Megagem logic (gem-value beliefs, mission value, coin pricing,
reveals); the second half is the bot itself.

Otis Ikpase
"""

from functools import lru_cache
from math import comb, sqrt

from bots import Bot, BotInfo, RoundResult
from constants import (
    CARDS_PER_PLAYER,
    COINS_PER_PLAYER,
    STANDARD_GEMS,
    AuctionType,
    GemType,
    WildCardRequirement,
)

NUM_COLORS = len(STANDARD_GEMS)
_WILD_KIND = {
    None: 0,
    WildCardRequirement.SAME: 1,
    WildCardRequirement.DIFFERENT: 2,
    WildCardRequirement.PAIRS: 3,
}


# ---------------------------------------------------------------- gem values


@lru_cache(maxsize=4096)
def hyper_pmf(pool: int, hits: int, draws: int) -> tuple[float, ...]:
    """P(j of `hits` marked cards land in `draws` cards drawn from `pool`)."""
    if pool <= 0 or draws <= 0 or hits <= 0:
        return (1.0,)
    draws = min(draws, pool)
    hits = min(hits, pool)
    total = comb(pool, draws)
    return tuple(
        comb(hits, j) * comb(pool - hits, draws - j) / total
        for j in range(0, min(hits, draws) + 1)
    )


def _value_stats(chart: tuple[int, ...], base: int, pmf: tuple[float, ...]):
    top = len(chart) - 1
    mean = 0.0
    second = 0.0
    for j, p in enumerate(pmf):
        v = chart[min(base + j, top)]
        mean += p * v
        second += p * v * v
    return mean, max(0.0, second - mean * mean)


def color_values(info) -> list[tuple[float, float]]:
    """(expected final value, variance) per colour, using this bot's own hand."""
    chart = info.value_chart.values
    hidden = info.self_player.hidden_gems
    opp_hidden = sum(p.hidden_gem_count for p in info.other_players)
    unknown = [info.unknown_gems.get(g, 0) for g in STANDARD_GEMS]
    pool = sum(unknown)
    out = []
    for i, gem in enumerate(STANDARD_GEMS):
        base = info.value_display.get(gem, 0) + hidden.count(gem)
        out.append(_value_stats(chart, base, hyper_pmf(pool, unknown[i], opp_hidden)))
    return out


def color_slopes(info) -> list[float]:
    """Change in a colour's expected value if one more of it were in hands."""
    chart = info.value_chart.values
    hidden = info.self_player.hidden_gems
    opp_hidden = sum(p.hidden_gem_count for p in info.other_players)
    unknown = [info.unknown_gems.get(g, 0) for g in STANDARD_GEMS]
    pool = sum(unknown)
    out = []
    for i, gem in enumerate(STANDARD_GEMS):
        base = info.value_display.get(gem, 0) + hidden.count(gem)
        pmf = hyper_pmf(pool, unknown[i], opp_hidden)
        out.append(
            _value_stats(chart, base + 1, pmf)[0] - _value_stats(chart, base, pmf)[0]
        )
    return out


def public_color_values(info, extra_display=None) -> list[float]:
    """Expected final value per colour as seen by someone with no private hand.

    ``extra_display`` is an optional colour index treated as already revealed
    from this bot's hand (used to score reveal choices).
    """
    chart = info.value_chart.values
    hidden = info.self_player.hidden_gems
    all_hidden = sum(p.hidden_gem_count for p in info.other_players) + len(hidden)
    out = []
    unknown = [info.unknown_gems.get(g, 0) + hidden.count(g) for g in STANDARD_GEMS]
    display = [info.value_display.get(g, 0) for g in STANDARD_GEMS]
    if extra_display is not None:
        unknown[extra_display] -= 1
        display[extra_display] += 1
        all_hidden -= 1
    pool = sum(unknown)
    for i in range(NUM_COLORS):
        out.append(
            _value_stats(chart, display[i], hyper_pmf(pool, unknown[i], all_hidden))[0]
        )
    return out


# ------------------------------------------------------------------ missions


def mission_spec(mission) -> tuple:
    req = mission.gem_requirements
    return (
        mission.value,
        tuple(req.get(g, 0) for g in STANDARD_GEMS),
        req.get(GemType.WILD_CARD, 0),
        _WILD_KIND[mission.wild_card_requirement],
    )


@lru_cache(maxsize=200_000)
def mission_need(spec: tuple, coll: tuple[int, ...]) -> int:
    """Fewest extra gems that would complete the mission from ``coll``."""
    _, fixed, wild, kind = spec
    need = 0
    rest = []
    for have, want in zip(coll, fixed):
        if have < want:
            need += want - have
            rest.append(0)
        else:
            rest.append(have - want)
    if wild <= 0:
        return need
    if kind == 0:  # any colours
        return need + max(0, wild - sum(rest))
    if kind == 1:  # all the same colour
        return need + max(0, wild - max(rest))
    if kind == 2:  # all different colours
        return need + max(0, wild - sum(1 for r in rest if r > 0))
    pairs_needed = wild // 2 - sum(r // 2 for r in rest)
    if pairs_needed <= 0:
        return need
    odd = sum(r % 2 for r in rest)
    cheap = min(odd, pairs_needed)
    return need + cheap + 2 * (pairs_needed - cheap)


def _mission_weight(spec, coll, gems_left, rho, avail, opp_need, race_pen):
    need = mission_need(spec, coll)
    if need == 0:
        return 1.0
    if need > gems_left:
        return 0.0
    if avail is not None:
        for have, want, supply in zip(coll, spec[1], avail):
            if want - have > supply:
                return 0.0
    weight = rho**need
    if opp_need is not None and opp_need < need:
        weight *= race_pen
    return weight


def mission_value(
    specs,
    coll,
    lot_idx,
    gems_left_after: int,
    rho: float,
    avail=None,
    opp_needs=None,
    race_pen: float = 1.0,
) -> float:
    """Points gained on missions by adding the lot to a collection.

    A completed mission counts in full. Progress short of completion is worth
    ``rho ** gems_still_needed`` of the reward while enough gems remain.
    ``avail`` (unseen supply per colour after this lot) zeroes missions whose
    named colours have run out. ``opp_needs`` (fewest gems any rival needs, per
    mission) scales progress by ``race_pen`` when a rival is closer.
    """
    if not specs:
        return 0.0
    after = list(coll)
    for i in lot_idx:
        after[i] += 1
    after_t = tuple(after)
    coll_t = tuple(coll)
    avail_before = None
    if avail is not None:
        avail_before = list(avail)
        for i in lot_idx:
            avail_before[i] += 1
    total = 0.0
    for k, spec in enumerate(specs):
        rival = opp_needs[k] if opp_needs is not None else None
        w1 = _mission_weight(
            spec, after_t, gems_left_after, rho, avail, rival, race_pen
        )
        w0 = _mission_weight(
            spec,
            coll_t,
            gems_left_after + len(lot_idx),
            rho,
            avail_before,
            rival,
            race_pen,
        )
        if w1 > w0:
            total += spec[0] * (w1 - w0)
    return total


# ------------------------------------------------------------- auction facts


class View:
    """Everything a bid decision needs, computed once per callback."""

    __slots__ = (
        "info",
        "card",
        "kind",
        "amount",
        "coins",
        "max_bid",
        "n",
        "start_coins",
        "lot_idx",
        "frac_left",
        "final",
        "values",
        "specs",
        "my_coll",
        "gems_left_after",
        "avail",
        "opp_colls",
    )

    def __init__(self, info):
        self.info = info
        card = info.current_auction
        self.card = card
        me = info.self_player
        self.coins = me.coins
        self.n = info.num_players
        self.start_coins = COINS_PER_PLAYER[self.n]
        total_auctioned = 30 - self.n * CARDS_PER_PLAYER[self.n]
        self.lot_idx = ()
        self.amount = card.value if card is not None else 0
        if card is None:
            self.kind = "none"
        elif card.type is AuctionType.AUCTION:
            self.kind = "treasure"
            lot = info.auction_gems[: min(card.value, len(info.auction_gems))]
            self.lot_idx = tuple(STANDARD_GEMS.index(g) for g in lot)
        elif card.type is AuctionType.INVEST:
            self.kind = "invest"
        else:
            self.kind = "loan"
        self.max_bid = self.coins + (self.amount if self.kind == "loan" else 0)
        self.gems_left_after = info.gems_remaining - len(self.lot_idx)
        self.frac_left = max(0.0, self.gems_left_after / total_auctioned)
        self.final = self.kind == "treasure" and self.gems_left_after <= 0
        self.values = color_values(info)
        self.specs = tuple(mission_spec(m) for m in info.available_missions)
        self.my_coll = tuple(me.won_gems.get(g, 0) for g in STANDARD_GEMS)
        row_rest = info.auction_gems[len(self.lot_idx) :]
        self.avail = [
            info.unknown_gems.get(g, 0) + row_rest.count(g) for g in STANDARD_GEMS
        ]
        self.opp_colls = [
            tuple(p.won_gems.get(g, 0) for g in STANDARD_GEMS)
            for p in info.other_players
        ]

    def opp_needs(self) -> list[int]:
        """Fewest gems any rival still needs, for each open mission."""
        return [
            min(mission_need(spec, coll) for coll in self.opp_colls)
            for spec in self.specs
        ]

    def lot_value(
        self, rho: float, supply_check: bool = False, race_pen: float = 1.0
    ) -> tuple[float, float]:
        """(expected points, standard deviation) of winning the lot for me."""
        mean = sum(self.values[i][0] for i in self.lot_idx)
        var = sum(self.values[i][1] for i in self.lot_idx)
        mean += mission_value(
            self.specs,
            self.my_coll,
            self.lot_idx,
            self.gems_left_after,
            rho,
            avail=self.avail if supply_check else None,
            opp_needs=self.opp_needs() if race_pen != 1.0 and self.specs else None,
            race_pen=race_pen,
        )
        return mean, sqrt(var)

    def coin_premium(self, lam0: float, lam_min: float, kappa: float) -> float:
        """Extra worth of a spendable coin above its one point of face value."""
        if self.final:
            return 0.0
        spend = 1.0 - min(1.0, self.coins / self.start_coins)
        lam = lam0 * self.frac_left + kappa * (spend - (1.0 - self.frac_left))
        return min(2.5, max(lam_min, lam))


def clamp_bid(x: float, cap: int) -> int:
    """A legal whole-number bid. The harness turns anything else into zero."""
    if x != x:  # NaN
        return 0
    return int(max(0, min(cap, round(x))))


# -------------------------------------------------------------------- reveal


def choose_reveal(info, sign: float = 1.0) -> int:
    """Index of the hidden gem to reveal.

    Every hidden gem reaches the display by the end, so the choice only changes
    what opponents believe meanwhile. With ``sign`` = 1 this reveals the colour
    that least raises rivals' valuations of gems still to be sold.
    """
    hidden = info.self_player.hidden_gems
    if len(hidden) <= 1 or sign == 0.0:
        return 0
    before = public_color_values(info)
    unknown = [info.unknown_gems.get(g, 0) for g in STANDARD_GEMS]
    pool = sum(unknown)
    deck_share = info.gem_deck_count / pool if pool else 0.0
    best_index, best_score = 0, None
    seen = set()
    for index, gem in enumerate(hidden):
        c = STANDARD_GEMS.index(gem)
        if c in seen:
            continue
        seen.add(c)
        after = public_color_values(info, extra_display=c)
        supply = info.auction_gems.count(gem) + unknown[c] * deck_share
        score = -sign * (after[c] - before[c]) * supply
        if best_score is None or score > best_score + 1e-12:
            best_index, best_score = index, score
    return best_index


# =============================================================== the bot

# Prior guesses for an opponent with no history, as fractions of the "value"
# scale: lot value for gems, coins held for investments, loan size for loans.
_PRIOR = {
    "treasure": (0.0, 0.15, 0.3, 0.45, 0.6),
    "invest": (0.0, 0.1, 0.25, 0.5),
    "loan": (0.0, 0.1, 0.2, 0.3),
}
_HISTORY_LIMIT = 60
_PRICE_EMA = 0.2


class MarketReader(Bot):
    # Core settings tuned over multiple games.
    #   lam0/lam_min/kappa  coin premium schedule      delta  value shading
    #   rho  mission progress credit   prior_weight/decay  opponent model
    # Optional parts, each off at the value shown in brackets:
    #   la_mix [0]       share of the coin premium taken from look-ahead
    #   la_share         share of future gems the look-ahead plans to buy
    #   color_k [0]      strength of inferring rivals' hands from their bids
    #   split_lots [0]   model one-gem and two-gem lots separately
    #   censor [0]       treat all-in bids as "at least this much"
    #   reset_detect [0] forget a seat's history when its behaviour changes
    #   supply_check [0] drop missions whose named colours have run out
    #   race_pen [1]     scale mission progress when a rival is closer
    #   deny_w [0]       credit for stopping a rival completing a mission
    #   win_cost [0]     points charged for any win (forced reveal, tie order)
    #   by_n [None]      overrides per player count, e.g. {3: {"lam0": 1.2}}
    PARAMS = {
        # Fallback settings (tuned over 3, 4 and 5 players together).
        "lam0": 2.275,
        "kappa": 0.143,
        "delta": -0.675,
        "rho": 0.454,
        "prior_weight": 2.348,
        "decay": 0.955,
        "la_mix": 0.069,
        "win_cost": -0.801,
        "race_pen": 0.911,
        "deny_w": 0.086,
        "lam_min": 0.05,
        "reveal_sign": 1.0,
        "la_share": 6.0,
        "reset_detect": 1.0,
        "color_k": 0.0,
        "split_lots": 0.0,
        "censor": 0.0,
        "supply_check": 0.0,
        # Settings tuned separately for each player count.
        "by_n": {
            3: {
                "lam0": 1.758,
                "kappa": 0.254,
                "delta": -0.573,
                "rho": 0.453,
                "prior_weight": 2.318,
                "decay": 0.959,
                "la_mix": 0.058,
                "win_cost": -1.236,
                "race_pen": 0.762,
                "deny_w": 0.18,
            },
            4: {
                "lam0": 2.192,
                "kappa": 0.264,
                "delta": -0.491,
                "rho": 0.491,
                "prior_weight": 2.025,
                "decay": 0.976,
                "la_mix": 0.222,
                "win_cost": -0.688,
                "race_pen": 0.776,
                "deny_w": 0.173,
            },
            5: {
                "lam0": 2.786,
                "kappa": 0.306,
                "delta": 0.059,
                "rho": 0.364,
                "prior_weight": 2.641,
                "decay": 0.981,
                "la_mix": 0.034,
                "win_cost": -0.831,
                "race_pen": 0.839,
                "deny_w": 0.194,
            },
        },
    }

    def __init__(self) -> None:
        # history[(opponent id, kind)] = rows of
        # (bid, abs scale, value scale, cap, bid was all-in)
        self.history: dict = {}
        self.pending = None
        self.price = None  # running winning price per gem
        self.color_sig = [0.0] * len(STANDARD_GEMS)

    def on_game_start(self) -> None:
        self.color_sig = [0.0] * len(STANDARD_GEMS)  # hands are new each game
        self.pending = None

    def _params(self, n: int) -> dict:
        p = self.PARAMS
        extra = p.get("by_n")
        if extra:
            override = extra.get(n) or extra.get(str(n))
            if override:
                return {**p, **override}
        return p

    # ------------------------------------------------------- opponent model

    def _scales(self, v: View, info: BotInfo, p: dict) -> dict:
        """Per-opponent (absolute scale, value scale, legal cap) for this lot."""
        out = {}
        public = public_color_values(info) if v.kind == "treasure" else None
        for opp in info.other_players:
            if v.kind == "treasure":
                coll = tuple(opp.won_gems.get(g, 0) for g in STANDARD_GEMS)
                value = sum(public[i] for i in v.lot_idx) + mission_value(
                    v.specs, coll, v.lot_idx, v.gems_left_after, p["rho"]
                )
                out[opp.id] = (float(len(v.lot_idx)), value, opp.coins)
            elif v.kind == "invest":
                out[opp.id] = (1.0, float(opp.coins), opp.coins)
            else:
                out[opp.id] = (1.0, float(v.amount), opp.coins + v.amount)
        return out

    def _samples(self, opp_id: int, kind: str, lot_size: int, p: dict):
        if kind == "treasure" and p["split_lots"]:
            rows = self.history.get((opp_id, f"treasure{lot_size}"))
            if rows and len(rows) >= 4:
                return rows
        return self.history.get((opp_id, kind))

    def _bid_pmf(self, samples, scale, kind: str, p: dict) -> list[float]:
        """Probability of each bid 0..cap for one opponent on the current lot."""
        abs_now, val_now, cap = scale
        pmf = [0.0] * (cap + 1)
        prior = _PRIOR[kind]
        w = p["prior_weight"] / len(prior)
        for r in prior:
            pmf[clamp_bid(r * val_now, cap)] += w
        if samples:
            if p["censor"]:
                free = [s for s in samples if not s[4]]
                if len(free) >= 3:
                    samples = free
            # Pick the scale that explains this opponent's past bids best:
            # fixed amounts, a share of value, or a share of their budget.
            best, best_err = 1, None
            if len(samples) >= 3:
                for h in (0, 1, 2):
                    ratios = sorted(s[0] / s[1 + h] for s in samples if s[1 + h] > 0)
                    if not ratios:
                        continue
                    med = ratios[len(ratios) // 2]
                    err = sum(
                        abs(s[0] - min(s[3], med * s[1 + h])) for s in samples
                    ) / len(samples)
                    if best_err is None or err < best_err - 1e-9:
                        best, best_err = h, err
            now = (abs_now, val_now, float(cap))[best]
            weight = 1.0
            for s in reversed(samples):
                base = s[1 + best]
                guess = s[0] if base <= 0 else s[0] / base * now
                pmf[clamp_bid(guess, cap)] += weight
                weight *= p["decay"]
        total = sum(pmf)
        return [x / total for x in pmf]

    def _win_probs(self, v: View, info: BotInfo, scales: dict, p: dict):
        """P(my bid b wins) for b = 0..max_bid, and each rival's expected bid."""
        me, n, prev = info.self_player.id, info.num_players, info.previous_winner_id
        probs = [1.0] * (v.max_bid + 1)
        expected = {}
        for opp in info.other_players:
            samples = self._samples(opp.id, v.kind, len(v.lot_idx), p)
            pmf = self._bid_pmf(samples, scales[opp.id], v.kind, p)
            expected[opp.id] = sum(b * q for b, q in enumerate(pmf))
            if prev is None:
                tie = 0.5
            else:
                tie = 1.0 if (me - prev - 1) % n < (opp.id - prev - 1) % n else 0.0
            below = 0.0
            for b in range(v.max_bid + 1):
                at = pmf[b] if b < len(pmf) else 0.0
                probs[b] *= below + tie * at
                below += at
        return probs, expected

    # ----------------------------------------------------------- look-ahead

    def _lookahead(self, v: View, info: BotInfo, p: dict):
        """(premium per coin, spare coins, coins short) from what is still to
        come: the gems left, the going price, and what rivals can afford."""
        pool = sum(v.avail)
        if pool <= 0 or v.gems_left_after <= 0:
            return 0.0, float(v.coins), 0.0
        avg_value = sum(v.values[c][0] * v.avail[c] for c in range(len(v.avail))) / pool
        price = self.price if self.price is not None else 0.35 * avg_value
        richest = max(o.coins for o in info.other_players)
        price = max(0.5, min(price, richest + 1.0))
        lam = min(3.0, max(0.0, avg_value / price - 1.0))
        wanted = v.gems_left_after * price * p["la_share"]
        return lam, max(0.0, v.coins - wanted), max(0.0, wanted - v.coins)

    # ------------------------------------------------------------ callbacks

    def get_bid(self, info: BotInfo) -> int:
        v = View(info)
        if v.kind == "none":
            return 0
        p = self._params(v.n)
        if p["color_k"] and v.kind == "treasure":
            slopes = color_slopes(info)
            v.values = [
                (max(0.0, m + p["color_k"] * self.color_sig[c] * slopes[c]), var)
                for c, (m, var) in enumerate(v.values)
            ]
        scales = self._scales(v, info, p)
        probs, expected = self._win_probs(v, info, scales, p)
        self.pending = (info.round_number, v.kind, scales, v.lot_idx, expected)

        lam = v.coin_premium(p["lam0"], p["lam_min"], p["kappa"])
        mix = 0.0 if v.final else p["la_mix"]
        la_lam, spare, short = self._lookahead(v, info, p) if mix else (0.0, 0.0, 0.0)

        def premium(b):  # extra cost of spending or locking b coins
            return (1.0 - mix) * lam * b + mix * la_lam * max(0.0, b - spare)

        if v.kind == "treasure":
            value = v.lot_value(p["rho"], bool(p["supply_check"]), p["race_pen"])[0]
            value -= p["delta"]
            if p["deny_w"] and v.specs:
                after = [list(c) for c in v.opp_colls]
                for coll in after:
                    for i in v.lot_idx:
                        coll[i] += 1
                for spec in v.specs:
                    if any(mission_need(spec, tuple(c)) == 0 for c in after):
                        value += p["deny_w"] * spec[0]
            gain = lambda b: value - b - premium(b)  # noqa: E731
        elif v.kind == "invest":
            gain = lambda b: v.amount - premium(b)  # noqa: E731
        else:

            def gain(b):
                got = max(0.0, v.amount - b)
                return (1.0 - mix) * lam * got + mix * la_lam * min(got, short) - b

        best_bid, best_u = 0, 0.0
        cost = p["win_cost"]
        for b in range(v.max_bid + 1):
            u = (gain(b) - cost) * probs[b]
            if u > best_u + 1e-9:
                best_bid, best_u = b, u
        return clamp_bid(best_bid, v.max_bid)

    def choose_hidden_gem(self, info: BotInfo, bids) -> int:
        return choose_reveal(info, self.PARAMS["reveal_sign"])

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        pending, self.pending = self.pending, None
        if pending is None or pending[0] != info.round_number:
            return
        _, kind, scales, lot_idx, expected = pending
        p = self._params(info.num_players)
        if kind == "treasure" and lot_idx:
            per_gem = result.winning_bid / len(lot_idx)
            self.price = (
                per_gem
                if self.price is None
                else self.price + _PRICE_EMA * (per_gem - self.price)
            )
        for opp_id, bid in result.bids.items():
            scale = scales.get(opp_id)
            if scale is None:
                continue
            row = (bid, scale[0], scale[1], float(scale[2]), bid >= scale[2] > 0)
            keys = [(opp_id, kind)]
            if kind == "treasure":
                keys.append((opp_id, f"treasure{len(lot_idx)}"))
                # A rival paying over the odds for a colour probably holds it.
                z = (bid - expected.get(opp_id, bid)) / max(2.0, scale[1])
                z = max(-1.0, min(1.0, z)) / (len(scales) * len(lot_idx))
                for c in lot_idx:
                    self.color_sig[c] += z
            for key in keys:
                rows = self.history.setdefault(key, [])
                rows.append(row)
                if len(rows) > _HISTORY_LIMIT:
                    del rows[0]
            if p["reset_detect"]:
                self._maybe_reset(opp_id, kind)

    def _maybe_reset(self, opp_id: int, kind: str) -> None:
        """Drop old history if a seat's recent bids no longer fit it."""
        rows = self.history.get((opp_id, kind))
        if not rows or len(rows) < 12:
            return
        old, recent = rows[:-5], rows[-5:]
        for h in (0, 1):
            ratios = sorted(s[0] / s[1 + h] for s in old if s[1 + h] > 0)
            if not ratios:
                return
            med = ratios[len(ratios) // 2]

            def err(group, med=med, h=h):
                return sum(abs(s[0] - min(s[3], med * s[1 + h])) for s in group) / len(
                    group
                )

            if err(recent) <= 3.0 * err(old) + 2.0:
                return
        del rows[:-5]
        for size in (1, 2):
            extra = self.history.get((opp_id, f"{kind}{size}"))
            if extra:
                del extra[:-3]
