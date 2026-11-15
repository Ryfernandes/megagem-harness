"""Log pre-bid cash/fraction forecasts and validated bids from one actual game."""

import argparse
import csv
import logging
import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot_definitions.cash_fraction_model import CashFractionModel, available_budget  # noqa: E402
from bot_definitions.expected_utility import win_probability  # noqa: E402
from bot_definitions.soft_tracking_bot import LEVEL3_BOTS  # noqa: E402
from bot_definitions.valuation_bot import ValuationBot  # noqa: E402
from bots import BotInfo  # noqa: E402
from decks import VALUE_CHARTS  # noqa: E402
from experiments.ablate_level3 import CallbackErrors  # noqa: E402
from harness import create_game, finish_game, play_round  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seats", type=int, choices=(3, 4, 5), default=4)
    parser.add_argument("--seed", type=int, default=2700)
    parser.add_argument("--chart", choices=list("ABCDE"), default="E")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors = CallbackErrors()
    logging.getLogger("bot_runner").addHandler(errors)
    model = CashFractionModel()
    before: BotInfo | None = None
    pmfs: dict[int, list[float]] = {}

    def observe(info: BotInfo) -> None:
        nonlocal before, pmfs
        before = info
        pmfs = {p.id: model.distribution(info, p) for p in info.other_players}

    rows = []
    with create_game(
        args.seats,
        bots=[LEVEL3_BOTS[(False, False, False, True, False)]]
        + [ValuationBot] * (args.seats - 1),
        rng=Random(args.seed),
        bot_seed=args.seed,
        value_chart=next(c for c in VALUE_CHARTS if c.name == args.chart),
        bot_timeout=2.0,
    ) as game:
        while game.auction_gems:
            result = play_round(game, before_bid=observe)
            assert before is not None
            chosen = result.bids[0]
            guarantee = (
                max(available_budget(before, p) for p in before.other_players) + 1
            )
            for player in before.other_players:
                mean, spread = model.fraction_moments(
                    player.id, result.auction_card.type
                )
                pmf = pmfs[player.id]
                expected_bid = sum(bid * p for bid, p in enumerate(pmf))
                model.observe(before, player, result.bids[player.id])
                record = model.history[player.id][result.auction_card.type][-1]
                rows.append(
                    {
                        "seed": args.seed,
                        "chart": args.chart,
                        "round": record.round_number,
                        "opponent": player.id,
                        "auction_type": record.auction_type.name,
                        "auction_value": record.auction_value,
                        "coins_before": record.coins_before,
                        "budget_before": record.budget_before,
                        "actual_bid": record.bid,
                        "fraction": record.fraction,
                        "evidence_weight": record.weight,
                        "prior_to_this_bid_fraction_mean": mean,
                        "prior_to_this_bid_fraction_sd": spread,
                        "predicted_bid_mean": expected_bid,
                        "our_bid": chosen,
                        "p_win_at_zero": win_probability(before, 0, pmfs),
                        "p_win_at_our_bid": win_probability(before, chosen, pmfs),
                        "p_win_above_every_rival_budget": win_probability(
                            before, guarantee, pmfs
                        ),
                    }
                )
        finish_game(game)
    if errors.messages:
        raise RuntimeError(f"Invalid log: {errors.messages}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Logged {len(rows)} opponent bids; forecasts frozen before learning each bid."
    )


if __name__ == "__main__":
    main()
