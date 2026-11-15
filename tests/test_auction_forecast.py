"""Check exact auction stopping horizons against every labeled deck order."""

import pickle
import unittest
from collections import Counter
from dataclasses import replace
from itertools import permutations
from time import perf_counter

from bot_definitions.auction_forecast import (
    _forecast,
    finance_opportunity_cost,
    forecast_auctions,
    treasure_stopping_pmf,
)
from bot_definitions.soft_tracking_bot import LEVEL3_BOTS, SoftTrackingBot
from bot_definitions.valuation_bot import ValuationBot
from constants import AuctionType
from immutable import FrozenMap
from lib import AuctionCard
from tests.test_valuation_bot import BLUE, PURPLE, view

TREASURE, LOAN, INVEST = AuctionType.AUCTION, AuctionType.LOAN, AuctionType.INVEST


def forecast_view(gems, cards):
    return replace(
        view(),
        auction_gems=(PURPLE, BLUE) if gems else (PURPLE,),
        gems_remaining=gems + 1,
        gem_deck_count=max(0, gems - 1),
        auction_cards_remaining=FrozenMap(Counter(cards)),
    )


class AuctionForecastTests(unittest.TestCase):
    def test_stopping_horizon_and_finance_timing_match_all_deck_orders(self):
        cases = (
            (0, ((TREASURE, 1), (LOAN, 20), (INVEST, 10))),
            (1, ((TREASURE, 2), (TREASURE, 1), (LOAN, 20))),
            (3, ((TREASURE, 1), (TREASURE, 2), (TREASURE, 2), (LOAN, 10), (INVEST, 5))),
            (
                4,
                ((TREASURE, 1), (TREASURE, 1), (TREASURE, 2), (LOAN, 20), (INVEST, 10)),
            ),
            (
                2,
                (
                    (TREASURE, 1),
                    (TREASURE, 1),
                    (TREASURE, 2),
                    (LOAN, 10),
                    (LOAN, 20),
                    (INVEST, 5),
                ),
            ),
        )
        for gems, definitions in cases:
            cards = [AuctionCard(kind, value) for kind, value in definitions]
            forecast = forecast_auctions(forecast_view(gems, cards))
            rows = []
            for order in permutations(range(len(cards))):
                remaining, treasures, rounds = gems, 0, 0
                loan_coins = loans = investments = 0
                finance = []
                for index in order:
                    if not remaining:
                        break
                    card = cards[index]
                    rounds += 1
                    if card.type is TREASURE:
                        remaining = max(0, remaining - card.value)
                        treasures += 1
                    else:
                        finance.append((card, treasures))
                        if card.type is LOAN:
                            loan_coins += card.value
                            loans += 1
                        else:
                            investments += 1
                timed_loan = timed_invest = timed_count = 0.0
                for card, before in finance:
                    weight = (treasures - before) / treasures
                    if card.type is LOAN:
                        timed_loan += weight * card.value
                    else:
                        timed_invest += weight * card.value
                        timed_count += weight
                rows.append(
                    (
                        treasures,
                        rounds,
                        loans,
                        investments,
                        loan_coins,
                        timed_loan,
                        timed_invest,
                        timed_count,
                    )
                )
            self.assertEqual(
                (forecast.min_treasure, forecast.max_treasure),
                (min(r[0] for r in rows), max(r[0] for r in rows)),
            )
            self.assertEqual(
                (forecast.min_rounds, forecast.max_rounds),
                (min(r[1] for r in rows), max(r[1] for r in rows)),
            )
            actual = (
                forecast.expected_treasure,
                forecast.expected_rounds,
                forecast.expected_loans,
                forecast.expected_investments,
                forecast.expected_loan_coins,
                forecast.timed_loan_coins,
                forecast.timed_investment_coins,
                forecast.timed_investments,
            )
            for index, value in enumerate(actual):
                self.assertAlmostEqual(value, sum(r[index] for r in rows) / len(rows))

    def test_funding_reduces_cash_scarcity_and_investments_can_raise_it(self):
        treasure = [AuctionCard(TREASURE, 1)] * 8
        no_finance = forecast_view(6, treasure)
        loans = forecast_view(6, treasure + [AuctionCard(LOAN, 20)] * 2)
        investments = forecast_view(6, treasure + [AuctionCard(INVEST, 10)] * 2)

        def k(info):
            return finance_opportunity_cost(info, 2, 150, 1, 0.5, 0.45)

        self.assertLess(k(loans), k(no_finance))
        self.assertGreater(k(investments), k(no_finance))
        last = forecast_view(
            0, treasure + [AuctionCard(LOAN, 20), AuctionCard(INVEST, 10)]
        )
        self.assertEqual(k(last), 0)
        self.assertEqual(forecast_auctions(last).expected_loan_coins, 0)

    def test_feature_is_off_by_default_and_preserved_in_workers(self):
        self.assertFalse(SoftTrackingBot().use_finance_horizon)
        self.assertFalse(SoftTrackingBot().use_bid_evidence)
        self.assertFalse(SoftTrackingBot().use_endgame_rules)
        self.assertEqual(len(LEVEL3_BOTS), 32)
        bot = SoftTrackingBot(use_finance_horizon=True)
        clone = SoftTrackingBot.__new__(SoftTrackingBot)
        clone.load_state(pickle.loads(pickle.dumps(bot.save_state())))
        clone.on_game_start()
        self.assertTrue(clone.use_finance_horizon)
        with self.assertRaises(ValueError):
            SoftTrackingBot(use_finance_horizon=True, use_bid_model=False)

    def test_only_k_changes_not_valuation_financial_bids_reserve_or_reveals(self):
        on, off = SoftTrackingBot(use_finance_horizon=True), ValuationBot()
        for cash in (0, 1, 25):
            for kind, amount in ((TREASURE, 1), (INVEST, 10), (LOAN, 20)):
                info = replace(
                    view(),
                    self_player=replace(view().self_player, coins=cash),
                    current_auction=AuctionCard(kind, amount),
                )
                self.assertEqual(on.evaluate(info), off.evaluate(info))
                if kind is not TREASURE:
                    self.assertEqual(on.get_bid(info), off.get_bid(info))
                else:
                    self.assertTrue(
                        0 <= on.get_bid(info) <= cash - on.evaluate(info).reserve
                    )
                self.assertEqual(
                    on.choose_hidden_gem(info, {}), off.choose_hidden_gem(info, {})
                )

    def test_invalid_decks_and_forecast_runtime(self):
        for counts in ((1, 0, 0), (-1, 3, 2), (5, 1, 1)):
            with self.assertRaises(ValueError):
                treasure_stopping_pmf(*counts)
        _forecast.cache_clear()
        treasure_stopping_pmf.cache_clear()
        start = perf_counter()
        forecast = _forecast(15, 12, 5, 2, 2, 2, 2)
        self.assertLess(perf_counter() - start, 2)
        self.assertTrue(
            forecast.min_rounds <= forecast.expected_rounds <= forecast.max_rounds
        )


if __name__ == "__main__":
    unittest.main()
