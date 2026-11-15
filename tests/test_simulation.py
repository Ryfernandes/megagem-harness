"""The baseline policy and browser replays must obey the real game rules."""

import unittest
from collections import Counter
from dataclasses import replace
from random import Random

from bot_info import build_bot_info
from bots import Bot
from constants import CARDS_PER_PLAYER, AuctionType
from custom_bots import FixedRuleBot, SmallBidBot
from decks import GEM_CARDS, VALUE_CHARTS
from harness import create_game
from lib import AuctionCard
from simulation import simulate


class FixedRuleTests(unittest.TestCase):
    def test_fixed_fraction_and_financial_passes(self):
        with create_game(3, rng=Random(42)) as game:
            view = build_bot_info(game, game.players[0])
        bot = FixedRuleBot()
        for coins in (0, 1, 4, 5, 9, 25, 35, 100):
            for kind in AuctionType:
                info = replace(
                    view,
                    self_player=replace(view.self_player, coins=coins),
                    current_auction=AuctionCard(
                        kind, 1 if kind is AuctionType.AUCTION else 10
                    ),
                )
                bid = bot.get_bid(info)
                self.assertIs(type(bid), int)
                self.assertEqual(bid, coins // 5 if kind is AuctionType.AUCTION else 0)
                self.assertTrue(0 <= bid <= coins)
        self.assertEqual(bot.get_bid(view), 0)
        self.assertEqual(bot.choose_hidden_gem(view, {}), 0)


class ReplayTests(unittest.TestCase):
    def test_sampling_bot_is_selectable_and_finishes_legal_isolated_games(self):
        with self.assertNoLogs("bot_runner", level="WARNING"):
            for seats, chart in ((3, "E"), (4, "A"), (5, "B")):
                replay = simulate(
                    {
                        "bots": ["SamplingBot"] + ["ValuationBot"] * (seats - 1),
                        "chart": chart,
                        "seed": 3300,
                    }
                )
                self.assertEqual(replay["frames"][-1]["event"]["type"], "FINAL")
                for previous, current in zip(replay["frames"], replay["frames"][1:-1]):
                    event = current["event"]
                    budget = previous["state"]["players"][0]["coins"]
                    if event["type"] == "LOAN":
                        budget += event["value"]
                    self.assertTrue(0 <= event["bids"][0] <= budget)

    def test_strategic_reveals_are_legal_on_all_charts_and_seed_reproducible(self):
        with self.assertNoLogs("bot_runner", level="WARNING"):
            for seats in (3, 4, 5):
                for chart in "ABCDE":
                    replay = simulate(
                        {
                            "bots": ["SoftTrackingBot"]
                            + ["ValuationBot"] * (seats - 1),
                            "botOptions": [{"use_strategic_reveal": True}]
                            + [{}] * (seats - 1),
                            "chart": chart,
                            "seed": 3000,
                        }
                    )
                    self.assertTrue(replay["botOptions"][0]["use_strategic_reveal"])
                    self.assertEqual(replay["frames"][-1]["event"]["type"], "FINAL")
                    for previous, current in zip(
                        replay["frames"], replay["frames"][1:-1]
                    ):
                        event = current["event"]
                        if event["winner"] == 0 and event["reveal"] is not None:
                            self.assertIn(
                                event["reveal"], previous["state"]["players"][0]["hand"]
                            )
            config = {
                "bots": ["SoftTrackingBot"] * 3,
                "chart": "E",
                "seed": 3000,
                "botOptions": [{"use_strategic_reveal": True}] * 3,
            }
            self.assertEqual(simulate(config), simulate(config))

    def test_cash_fraction_switch_runs_legal_worker_games_at_all_seat_counts(self):
        with self.assertNoLogs("bot_runner", level="WARNING"):
            for seats in (3, 4, 5):
                replay = simulate(
                    {
                        "bots": ["SoftTrackingBot"] + ["ValuationBot"] * (seats - 1),
                        "botOptions": [{"use_cash_fraction_model": True}]
                        + [{}] * (seats - 1),
                        "chart": "E",
                        "seed": 2700,
                    }
                )
                self.assertTrue(replay["botOptions"][0]["use_cash_fraction_model"])
                self.assertFalse(replay["botOptions"][0]["use_finance_horizon"])
                self.assertEqual(replay["frames"][-1]["event"]["type"], "FINAL")
                for previous, current in zip(replay["frames"], replay["frames"][1:-1]):
                    event = current["event"]
                    budget = previous["state"]["players"][0]["coins"]
                    if event["type"] == "LOAN":
                        budget += event["value"]
                    self.assertTrue(0 <= event["bids"][0] <= budget)

    def test_soft_default_matches_current_valuation_full_games(self):
        for seats, chart in ((3, "B"), (4, "E"), (5, "D")):
            common = {"chart": chart, "seed": 2400}
            soft = simulate({**common, "bots": ["SoftTrackingBot"] * seats})
            level2 = simulate({**common, "bots": ["ValuationBot"] * seats})
            for frame in soft["frames"]:
                for player in frame["state"]["players"]:
                    player["bot"] = "ValuationBot"
            self.assertEqual(soft["frames"], level2["frames"])
            self.assertTrue(
                all(not any(options.values()) for options in soft["botOptions"])
            )

    def test_finance_switch_reaches_the_worker_and_changes_play(self):
        config = {
            "bots": ["SoftTrackingBot", "ValuationBot", "ValuationBot"],
            "chart": "A",
            "seed": 2400,
        }
        off = simulate(config)
        on_config = {**config, "botOptions": [{"use_finance_horizon": True}, {}, {}]}
        with self.assertNoLogs("bot_runner", level="WARNING"):
            on = simulate(on_config)
        self.assertTrue(on["botOptions"][0]["use_finance_horizon"])
        self.assertFalse(on["botOptions"][0]["use_bid_evidence"])
        self.assertNotEqual(on["frames"], off["frames"])
        self.assertEqual(on, simulate(on_config))

    def test_invalid_feature_switches_are_rejected(self):
        config = {"bots": ["SoftTrackingBot", "ValuationBot", "ValuationBot"]}
        for options in (
            [],
            {},
            [{"use_finance_horizon": 1}, {}, {}],
            [{"unknown": True}, {}, {}],
            [{}, {"use_finance_horizon": True}, {}],
        ):
            with self.assertRaises(ValueError):
                simulate({**config, "botOptions": options})

    def test_replay_all_seat_counts_and_charts(self):
        for seats in (3, 4, 5):
            for chart in VALUE_CHARTS:
                with self.subTest(seats=seats, chart=chart.name):
                    names = (
                        [FixedRuleBot.__name__, SmallBidBot.__name__, Bot.__name__] * 2
                    )[:seats]
                    replay = simulate({"bots": names, "chart": chart.name, "seed": 42})
                    frames = replay["frames"]
                    self.assertIsNone(frames[0]["event"])
                    self.assertEqual(frames[-1]["event"]["type"], "FINAL")
                    for f in frames:
                        state = f["state"]
                        total = (
                            sum(state["display"])
                            + state["deck"]
                            + len(state["row"])
                            + sum(
                                sum(p["gems"]) + len(p["hand"])
                                for p in state["players"]
                            )
                        )
                        self.assertEqual(total, len(GEM_CARDS))
                        self.assertTrue(all(p["coins"] >= 0 for p in state["players"]))
                    for previous, current in zip(frames, frames[1:-1]):
                        event = current["event"]
                        self.assertEqual(event["bid"], max(event["bids"]))
                        for player, bid in zip(
                            previous["state"]["players"], event["bids"]
                        ):
                            if player["bot"] == "FixedRuleBot":
                                self.assertEqual(
                                    bid,
                                    player["coins"] // 5
                                    if event["type"] == "AUCTION"
                                    else 0,
                                )
                    final = frames[-1]
                    self.assertEqual(
                        sum(final["state"]["display"]), CARDS_PER_PLAYER[seats] * seats
                    )
                    actual_gems = Counter()
                    for g, count in enumerate(final["state"]["display"]):
                        actual_gems[g] += count
                    for p in final["state"]["players"]:
                        self.assertEqual(p["hand"], [])
                        self.assertEqual(p["score"], sum(p["breakdown"].values()))
                        for g, count in enumerate(p["gems"]):
                            actual_gems[g] += count
                    self.assertEqual(list(actual_gems.values()), [6] * 5)
                    self.assertEqual(
                        final["event"]["scores"],
                        [p["score"] for p in final["state"]["players"]],
                    )

    def test_seed_reproduces_full_replay(self):
        config = {
            "bots": ["FixedRuleBot", "SmallBidBot", "Bot"],
            "chart": "E",
            "seed": 17,
        }
        self.assertEqual(simulate(config), simulate(config))

    def test_invalid_configurations(self):
        for config in (
            {},
            {"bots": ["Bot"] * 2},
            {"bots": ["missing"] * 3},
            {"bots": ["Bot"] * 3, "chart": "Z"},
            {"bots": ["Bot"] * 3, "seed": True},
            {"bots": ["Bot"] * 3, "seed": -1},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                simulate(config)


if __name__ == "__main__":
    unittest.main()
