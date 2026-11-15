"""Matched before/after ValuationBot games against frozen fractional opponents.

The new bidder's k is fixed at the previously selected (scale=1, floor=0.5).
No tuning or selection uses these fresh deals. Every lineup has one candidate;
all other seats retain the former ValuationBot policy. Ties split win credit.
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot_definitions.valuation_bot import (  # noqa: E402
    FractionalValuationBot,
    ValuationBot,
)
from experiments.ablate_level3 import (  # noqa: E402
    CallbackErrors,
    paired_ci,
    run_variant,
    summarize,
)


def run(name: str, seats: int, start: int, count: int) -> dict[str, Any]:
    errors = CallbackErrors()
    logger = logging.getLogger("bot_runner")
    logger.addHandler(errors)
    try:
        cls = ValuationBot if name == "after" else FractionalValuationBot
        result = run_variant(name, range(start, start + count), seats, bot_class=cls)
        if errors.messages:
            raise RuntimeError(f"Invalid benchmark: {errors.messages}")
        result.pop("flags")  # Those switches describe SoftTrackingBot only.
        return result
    finally:
        logger.removeHandler(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--start-seed", type=int, default=2100)
    parser.add_argument(
        "--seats", nargs="+", type=int, choices=(3, 4, 5), default=[3, 4, 5]
    )
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games < 2 or args.start_seed < 0 or args.jobs < 1:
        parser.error("Invalid count, seed or jobs")
    counts = list(dict.fromkeys(args.seats))
    results: dict[int, dict[str, Any]] = {seats: {} for seats in counts}
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=get_context("spawn")
    ) as pool:
        pending = {
            pool.submit(run, name, seats, args.start_seed, args.games): (seats, name)
            for seats in counts
            for name in ("before", "after")
        }
        for future in as_completed(pending):
            seats, name = pending[future]
            result = results[seats][name] = future.result()
            credit = sum(r["win_credit"] for r in result["records"])
            print(
                f"{seats} seats, {name}: {credit:.1f}/{args.games} win credits",
                flush=True,
            )
    for variants in results.values():
        before = variants["before"]
        for result in variants.values():
            result["summary"] = summarize(result, before)
            deltas = [
                r["win_credit"] - b["win_credit"]
                for r, b in zip(result["records"], before["records"])
            ]
            mean, low, high = paired_ci(deltas)
            result["summary"].update(
                paired_win_credit_delta=mean,
                paired_win_credit_ci_low=low,
                paired_win_credit_ci_high=high,
            )
    report = {
        "games_per_policy_per_seat_count": args.games,
        "start_seed": args.start_seed,
        "end_seed": args.start_seed + args.games - 1,
        "opponents": "Every other seat uses the frozen FractionalValuationBot",
        "matching": "same deal seed, chart ABCDE[seed%5], candidate seat=(-seed)%seats",
        "ties": "fractional win credit for the single candidate",
        "formula": "p(win)*(E[value]-(1+k)*bid)",
        "k": "(liquidity-1)+0.5*future_lots/(future_lots+1.5)",
        "parameters_frozen_before_run": True,
        "inference": False,
        "mission_only_endgame": False,
        "unchanged": ["valuation", "missions/denial", "reserve", "finance", "reveals"],
        "ci": "approximate paired 95% interval: mean +/- 1.96 SE across seeds",
        "callback_errors": [],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "\nSeats | before wins | after wins | before/after score | "
        "margin change [95% CI]"
    )
    for seats, variants in results.items():
        b, a = (variants[name]["summary"] for name in ("before", "after"))
        print(
            f"{seats} | {b['win_credit']:.1f} | {a['win_credit']:.1f} | "
            f"{b['mean_score']:.2f}/{a['mean_score']:.2f} | "
            f"{a['paired_margin_delta']:+.2f} "
            f"[{a['paired_margin_ci_low']:+.2f}, {a['paired_margin_ci_high']:+.2f}]"
        )


if __name__ == "__main__":
    main()
