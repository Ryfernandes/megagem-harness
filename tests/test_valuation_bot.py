"""Check probability calculations, economic edge cases, and real worker games."""

import unittest
from collections import Counter
from dataclasses import replace
from itertools import combinations
from random import Random
from time import perf_counter

from bot_definitions.valuation_bot import ValuationBot, _mission_targets
from bots import BotInfo, PlayerInfo, SelfInfo
from constants import STANDARD_GEMS, AuctionType
from decks import AUCTION_CARDS, MISSION_CARDS, VALUE_CHARTS
from immutable import FrozenMap
from lib import AuctionCard, MissionCard
from simulation import simulate

BLUE, GREEN, PURPLE, YELLOW, PINK = STANDARD_GEMS


def view():
    empty = FrozenMap(dict.fromkeys(STANDARD_GEMS, 0))
    public = dict(
        coins=25,
        hidden_gem_count=4,
        revealed_gems=(),
        won_gems=empty,
        auction_cards=(),
        mission_cards=(),
        invested_coins=0,
    )
    auction = AuctionCard(AuctionType.AUCTION, 1)
    auctions = Counter(AUCTION_CARDS)
    auctions.subtract([auction])
    return BotInfo(
        self_player=SelfInfo(id=0, hidden_gems=(BLUE, GREEN, GREEN, PINK), **public),
        other_players=tuple(PlayerInfo(id=i, **public) for i in range(1, 4)),
        num_players=4,
        round_number=1,
        current_auction=auction,
        gem_deck_count=12,
        gems_remaining=14,
        unknown_gems=FrozenMap(dict(zip(STANDARD_GEMS, (5, 4, 5, 5, 5)))),
        auction_gems=(PURPLE, YELLOW),
        value_display=empty,
        previous_winner_id=None,
        value_chart=VALUE_CHARTS[0],
        available_missions=(),
        auction_deck_count=24,
        auction_cards_remaining=FrozenMap(+auctions),
    )


class ValuationTests(unittest.TestCase):
    def test_hypergeometric_matches_enumerated_worlds_on_every_chart(self):
        info = view()
        info = replace(
            info,
            self_player=replace(info.self_player, hidden_gems=(BLUE,)),
            other_players=tuple(
                replace(p, hidden_gem_count=int(p.id < 3)) for p in info.other_players
            ),
            value_display=FrozenMap({BLUE: 1}),
            unknown_gems=FrozenMap(
                {g: 2 if g in (BLUE, GREEN) else 0 for g in STANDARD_GEMS}
            ),
            gem_deck_count=2,
        )
        unknown = (BLUE, BLUE, GREEN, GREEN)
        worlds = list(combinations(range(4), 2))
        for chart in VALUE_CHARTS:
            info = replace(info, value_chart=chart)
            prices = ValuationBot.expected_gem_values(info)
            for gem in STANDARD_GEMS:
                expected = sum(
                    chart.get_value(
                        (2 if gem is BLUE else 0)
                        + sum(unknown[i] is gem for i in world)
                    )
                    for world in worlds
                ) / len(worlds)
                self.assertAlmostEqual(prices[gem], expected)
            revealed = replace(
                info,
                self_player=replace(info.self_player, hidden_gems=()),
                value_display=FrozenMap({BLUE: 2}),
            )
            self.assertEqual(prices, ValuationBot.expected_gem_values(revealed))

    def test_wildcard_targets_match_engine_including_four_equal_gems(self):
        rng = Random(9)
        for mission in MISSION_CARDS:
            targets = _mission_targets(mission)
            for target in targets:
                self.assertTrue(mission.is_satisfied(dict(zip(STANDARD_GEMS, target))))
            for _ in range(80):
                counts = [rng.randrange(5) for _ in range(5)]
                expected = any(all(a >= b for a, b in zip(counts, t)) for t in targets)
                self.assertEqual(
                    mission.is_satisfied(dict(zip(STANDARD_GEMS, counts))), expected
                )

    def test_immediate_mission_and_opponent_denial_are_marginal_rewards(self):
        info = replace(
            view(),
            auction_gems=(GREEN,),
            gem_deck_count=0,
            gems_remaining=1,
            unknown_gems=FrozenMap(dict.fromkeys(STANDARD_GEMS, 0)),
            other_players=tuple(
                replace(p, hidden_gem_count=0) for p in view().other_players
            ),
        )
        bot = ValuationBot()
        plain = bot.evaluate(info)
        mission = MissionCard(5, {BLUE: 1, GREEN: 1})
        ours = replace(info.self_player, won_gems=FrozenMap({BLUE: 1}))
        completing = bot.evaluate(
            replace(info, self_player=ours, available_missions=(mission,))
        )
        self.assertEqual(completing.missions, 5)
        self.assertEqual(completing.gems, plain.gems)
        rivals = (
            replace(info.other_players[0], won_gems=FrozenMap({BLUE: 1})),
            *info.other_players[1:],
        )
        denying = bot.evaluate(
            replace(info, other_players=rivals, available_missions=(mission,))
        )
        self.assertGreater(denying.denial, plain.denial)
        self.assertLess(denying.denial, 5)
        self.assertEqual(denying.missions, 0)
        impossible = replace(info, available_missions=(MissionCard(10, {PINK: 3}),))
        self.assertEqual(bot.evaluate(impossible).missions, 0)

    def test_unfinished_missions_discount_races_cost_and_unavailable_supply(self):
        info = view()
        bot = ValuationBot()
        mission = MissionCard(10, {BLUE: 1, GREEN: 1, PURPLE: 1})
        ours = {BLUE: 1, GREEN: 1}
        values = bot.expected_gem_values(info)
        visible = Counter({PURPLE: 1})
        potential = bot._mission_potential(
            info, mission, info.self_player, ours, visible, values
        )
        self.assertTrue(0 < potential < mission.value)
        rival = replace(info.other_players[0], won_gems=FrozenMap(ours))
        raced = replace(info, other_players=(rival, *info.other_players[1:]))
        self.assertLess(
            bot._mission_potential(
                raced, mission, raced.self_player, ours, visible, values
            ),
            potential,
        )
        costly = {g: 100.0 for g in STANDARD_GEMS}
        self.assertLess(
            bot._mission_potential(
                info, mission, info.self_player, ours, visible, costly
            ),
            potential,
        )
        no_supply = replace(info, gem_deck_count=0)
        self.assertEqual(
            bot._mission_potential(
                no_supply, mission, info.self_player, ours, Counter(), values
            ),
            0,
        )

    def test_reserve_tapers_and_allows_final_purchase(self):
        bot = ValuationBot()
        info = view()
        early = bot.evaluate(info)
        self.assertGreater(early.reserve, 0)
        self.assertLessEqual(bot.get_bid(info), info.self_player.coins - early.reserve)
        broke = replace(info, self_player=replace(info.self_player, coins=1))
        self.assertEqual(bot.get_bid(broke), 0)
        last = replace(info, gems_remaining=1, gem_deck_count=0, auction_gems=(PURPLE,))
        self.assertEqual(bot.evaluate(last).reserve, 0)

    def test_investment_recoups_bid_and_loans_need_productive_opportunities(self):
        bot = ValuationBot()
        info = view()
        invest = replace(
            info,
            current_auction=AuctionCard(AuctionType.INVEST, 10),
            self_player=replace(info.self_player, coins=100),
            other_players=tuple(replace(p, coins=100) for p in info.other_players),
        )
        # Unlike treasure, the locked principal returns. With ample cash and
        # a low liquidity premium, bid more than for a spent 10-point reward.
        spent_bid = int(bot.bid_fraction * 10 / bot.evaluate(invest).liquidity)
        self.assertGreater(bot.get_bid(invest), spent_bid)
        loan = replace(
            info,
            current_auction=AuctionCard(AuctionType.LOAN, 20),
            self_player=replace(info.self_player, coins=0),
        )
        self.assertTrue(0 < bot.get_bid(loan) < 20)
        no_future = replace(loan, gems_remaining=0, gem_deck_count=0, auction_gems=())
        self.assertEqual(bot.get_bid(no_future), 0)
        rich = replace(loan, self_player=replace(info.self_player, coins=1000))
        self.assertEqual(bot.get_bid(rich), 0)

    def test_worst_case_decision_stays_under_callback_budget(self):
        info = replace(view(), available_missions=MISSION_CARDS[:4])
        start = perf_counter()
        for _ in range(5):
            bid = ValuationBot().get_bid(info)
            self.assertIs(type(bid), int)
            self.assertTrue(0 <= bid <= info.self_player.coins)
        self.assertLess((perf_counter() - start) / 5, 2.0)

    def test_real_worker_games_all_seats_and_charts_without_fallbacks(self):
        with self.assertNoLogs("bot_runner", level="WARNING"):
            for seats in (3, 4, 5):
                for chart in VALUE_CHARTS:
                    with self.subTest(seats=seats, chart=chart.name):
                        names = (
                            [
                                "ValuationBot",
                                "FixedRuleBot",
                                "SmallBidBot",
                                "ValuationBot",
                                "Bot",
                            ]
                        )[:seats]
                        replay = simulate(
                            {"bots": names, "chart": chart.name, "seed": 9}
                        )
                        self.assertEqual(replay["frames"][-1]["event"]["type"], "FINAL")
                        for frame in replay["frames"]:
                            self.assertTrue(
                                all(p["coins"] >= 0 for p in frame["state"]["players"])
                            )
            config = {"bots": ["ValuationBot"] * 3, "chart": "E", "seed": 7}
            self.assertEqual(simulate(config), simulate(config))


if __name__ == "__main__":
    unittest.main()
