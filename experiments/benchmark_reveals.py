"""Matched strategic-reveal on/off games against current level 2.

Only the reveal policy changes. All bid policies, value estimates and other
level-3 extras are held fixed. Private bot RNG starts at runner index and
continues/checkpoints through games, so reproduce the full ordered seed range.
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

from bot_definitions.soft_tracking_bot import LEVEL3_BOTS  # noqa: E402
from bot_definitions.valuation_bot import ValuationBot  # noqa: E402
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
        cls = LEVEL3_BOTS[(False, False, False, False, name == "on")]
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
    parser.add_argument("--start-seed", type=int, default=3000)
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
            for name in ("off", "on")
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
    per_chart: dict[int, dict[str, Any]] = {}
    for seats, variants in results.items():
        per_chart[seats] = {}
        for chart in "ABCDE":
            selected = {
                name: {"records": [r for r in data["records"] if r["chart"] == chart]}
                for name, data in variants.items()
            }
            if not selected["off"]["records"]:
                continue
            per_chart[seats][chart] = {
                name: summarize(data, selected["off"])
                for name, data in selected.items()
            }
    report = {
        "games_per_policy_per_seat_count": args.games,
        "start_seed": args.start_seed,
        "end_seed": args.start_seed + args.games - 1,
        "opponents": "Every other seat uses the current ValuationBot (level 2)",
        "matching": "same deal seed, chart ABCDE[seed%5], candidate seat=(-seed)%seats",
        "ties": "fractional win credit for the single candidate",
        "formula": "p(win)*(E[value]-(1+k)*bid)",
        "feature": "use_strategic_reveal",
        "default_enabled": False,
        "on": "chart-aware attention heuristic",
        "off": "first remaining hand card (current level 2)",
        "rng": "runner seed=index; private checkpointed RNG continues across games",
        "parameters_frozen_before_run": True,
        "inference": False,
        "mission_only_endgame": False,
        "unchanged": [
            "bid algorithms",
            "valuation",
            "missions/denial",
            "k",
            "reserve",
            "finance",
        ],
        "finance_horizon": False,
        "cash_fraction_model": False,
        "ci": "approximate paired 95% interval: mean +/- 1.96 SE across seeds",
        "callback_errors": [],
        "results": results,
        "per_chart": per_chart,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("\nSeats | off wins | on wins | off/on score | margin change [95% CI]")
    for seats, variants in results.items():
        b, a = (variants[name]["summary"] for name in ("off", "on"))
        print(
            f"{seats} | {b['win_credit']:.1f} | {a['win_credit']:.1f} | "
            f"{b['mean_score']:.2f}/{a['mean_score']:.2f} | "
            f"{a['paired_margin_delta']:+.2f} "
            f"[{a['paired_margin_ci_low']:+.2f}, {a['paired_margin_ci_high']:+.2f}]"
        )


if __name__ == "__main__":
    main()
