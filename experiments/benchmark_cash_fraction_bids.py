"""Matched cash-fraction opponent model ablation against current level 2.

Three policies: unchanged level 2; new distributions on treasure only; new
distributions plus correctly valued financial-auction EU bidding. All other
level-3 extras are off. Parameters are frozen before fresh matched games.
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

from bot_definitions.soft_tracking_bot import (  # noqa: E402
    LEVEL3_BOTS,
    _CashFractionTreasureOnlyBot,
)
from bot_definitions.valuation_bot import ValuationBot  # noqa: E402
from experiments.ablate_level3 import (  # noqa: E402
    CallbackErrors,
    paired_ci,
    run_variant,
    summarize,
)

VARIANTS = {
    "off": LEVEL3_BOTS[(False, False, False, False, False)],
    "treasure_only": _CashFractionTreasureOnlyBot,
    "all_types": LEVEL3_BOTS[(False, False, False, True, False)],
}


def run(name: str, seats: int, start: int, count: int) -> dict[str, Any]:
    errors = CallbackErrors()
    logger = logging.getLogger("bot_runner")
    logger.addHandler(errors)
    try:
        cls = VARIANTS[name]
        result = run_variant(
            name,
            range(start, start + count),
            seats,
            bot_class=cls,
            opponent_class=ValuationBot,
        )
        if errors.messages:
            raise RuntimeError(f"Invalid benchmark: {errors.messages}")
        result.pop("flags")  # Those switches describe SoftTrackingBot only.
        return result
    finally:
        logger.removeHandler(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--start-seed", type=int, default=2700)
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
            for name in VARIANTS
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
        before = variants["off"]
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
        "opponents": "Every other seat uses the current ValuationBot (level 2)",
        "matching": "same deal seed, chart ABCDE[seed%5], candidate seat=(-seed)%seats",
        "ties": "fractional win credit for the single candidate",
        "formula": "p(win)*(E[value]-(1+k)*bid)",
        "feature": "use_cash_fraction_model",
        "default_enabled": False,
        "priors": {
            "weight": 2.0,
            "treasure_mean_sd": [0.20, 0.14],
            "invest_mean_sd": [0.20, 0.16],
            "loan_mean_sd": [0.06, 0.10],
        },
        "evidence_weights": {"treasure": 0.25, "invest": 1.0, "loan": 1.0},
        "model": "rounded clipped-normal prior plus weighted empirical cash fractions",
        "loan_budget": "pre-bid cash + loan face value",
        "financial_control": "treasure_only keeps existing financial policy",
        "financial_eu": "all_types uses EU with investment return and loan debt offset",
        "parameters_frozen_before_run": True,
        "inference": False,
        "mission_only_endgame": False,
        "unchanged": ["gem/mission/denial valuation", "k", "reserve", "reveals"],
        "finance_horizon": False,
        "ci": "approximate paired 95% interval: mean +/- 1.96 SE across seeds",
        "callback_errors": [],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("\nSeats | policy | win credit | mean score | margin change [95% CI]")
    for seats, variants in results.items():
        for name in VARIANTS:
            s = variants[name]["summary"]
            print(
                f"{seats} | {name} | {s['win_credit']:.1f} | {s['mean_score']:.2f} | "
                f"{s['paired_margin_delta']:+.2f} "
                f"[{s['paired_margin_ci_low']:+.2f}, {s['paired_margin_ci_high']:+.2f}]"
            )


if __name__ == "__main__":
    main()
