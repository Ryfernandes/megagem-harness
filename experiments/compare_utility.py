"""Compare fractional bidding with expected utility, on matched game deals.

One candidate faces only ValuationBot opponents. Gem EV, missions, reserve rule,
reveal policy and financial bidding remain fixed. All hand inference/endgame
extras are off. EU variants share the same cash-bounded learned win predictor;
only k changes. Old EU retains its 95%-value action cap as a historical control.
Selection requires higher fractional win rate and positive paired-margin 95%
lower bound. Validation freezes the selected policy and uses untouched seeds.
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

from bot_definitions.soft_tracking_bot import UTILITY_BOTS, UTILITY_SETTINGS  # noqa: E402
from experiments.ablate_level3 import CallbackErrors, run_variant, summarize  # noqa: E402


def run(name: str, start: int, count: int, seats: int) -> dict[str, Any]:
    errors = CallbackErrors()
    logging.getLogger("bot_runner").addHandler(errors)
    result = run_variant(
        name, range(start, start + count), seats, bot_class=UTILITY_BOTS[name]
    )
    if errors.messages:
        raise RuntimeError(f"Invalid benchmark: {errors.messages}")
    return result


def choose(variants: dict[str, Any]) -> str:
    before = variants["before"]["summary"]["win_rate"]
    eligible = [
        name
        for name, r in variants.items()
        if name not in ("before", "old_eu")
        and r["summary"]["win_rate"] > before
        and r["summary"]["paired_margin_ci_low"] > 0
    ]
    return (
        max(
            eligible,
            key=lambda n: (
                variants[n]["summary"]["win_rate"],
                variants[n]["summary"]["paired_margin_delta"],
            ),
        )
        if eligible
        else "before"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=240)
    parser.add_argument("--start-seed", type=int, default=1560)
    parser.add_argument("--seats", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--variants", nargs="+", choices=list(UTILITY_BOTS))
    parser.add_argument("--validate", choices=list(UTILITY_BOTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games < 2 or args.start_seed < 0 or args.jobs < 1:
        parser.error("Invalid count, seed, or jobs")
    names = list(dict.fromkeys(["before", *(args.variants or UTILITY_BOTS)]))
    if args.validate and args.validate not in names:
        names.append(args.validate)
    variants = {}
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=get_context("spawn")
    ) as pool:
        pending = {
            pool.submit(run, name, args.start_seed, args.games, args.seats): name
            for name in names
        }
        for future in as_completed(pending):
            name = pending[future]
            variants[name] = future.result()
            credit = sum(g["win_credit"] for g in variants[name]["records"])
            print(
                f"Completed {name}: {credit:.1f}/{args.games} win credits", flush=True
            )
    for name in names:
        variants[name]["summary"] = summarize(variants[name], variants["before"])
        variants[name]["settings"] = dict(
            zip(("k_scale", "k_floor"), UTILITY_SETTINGS.get(name, (1, 0)))
        )
    report = {
        "games_per_variant": args.games,
        "start_seed": args.start_seed,
        "seats": args.seats,
        "opponents": ["ValuationBot"] * (args.seats - 1),
        "chart": "ABCDE[seed%5]",
        "seating": "candidate seat=(-seed)%seats",
        "ties": "fractional win credit",
        "formula": "p(win)*(E[value]-(1+k)*bid)",
        "k": "scale*(base_liquidity-1)+floor*remaining_auction_horizon",
        "fixed": [
            "hypergeometric gem EV",
            "mission/denial model",
            "reserve rule",
            "financial bidding",
            "first-card reveal",
        ],
        "hand_inference": False,
        "endgame_overrides": False,
        "callback_errors": [],
        "variants": {n: variants[n] for n in names},
    }
    if args.validate:
        r = variants[args.validate]["summary"]
        b = variants["before"]["summary"]
        report["validation"] = {
            "preselected_variant": args.validate,
            "confirmed": args.validate == "before"
            or (r["win_rate"] > b["win_rate"] and r["paired_margin_ci_low"] > 0),
        }
    else:
        report["selection"] = {
            "recommended_variant": choose(variants),
            "rule": "higher wins and positive paired-margin 95% lower bound; "
            "rank by win rate, then margin",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("\nPolicy | win credit | mean score | margin change [95% CI]")
    for name in names:
        r = variants[name]["summary"]
        print(
            f"{name} | {r['win_credit']:.1f}/{args.games} | {r['mean_score']:.2f} | "
            f"{r['paired_margin_delta']:+.2f} "
            f"[{r['paired_margin_ci_low']:+.2f}, {r['paired_margin_ci_high']:+.2f}]"
        )
    print(report.get("selection", report.get("validation")))


if __name__ == "__main__":
    main()
