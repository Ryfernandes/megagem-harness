"""Paired level-3 feature ablations against two identical level-2 opponents.

All-off is the level-2 policy. Each seed uses the same setup, chart, and rotating
candidate seat across variants. The extra bid-model flag isolates the change
from fractional bidding to learned surplus optimization. Cash limits condition
that predictor; alone they add nothing to level 2's existing legal/max-coin caps.
Private importable classes pin the flags in spawned workers. No policy tuning or
selection is performed on a validation run; summaries report paired margin CIs.
"""

import argparse
import json
import logging
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from math import sqrt
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bot_definitions.soft_tracking_bot import ABLATION_BOTS, FEATURE_NAMES  # noqa: E402
from bot_definitions.valuation_bot import (  # noqa: E402
    FractionalValuationBot as ValuationBot,
)
from bot_runner import BotRunner  # noqa: E402
from bots import Bot  # noqa: E402
from decks import VALUE_CHARTS  # noqa: E402
from harness import play_game  # noqa: E402


class CallbackErrors(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def run_variant(
    name: str,
    seeds: range,
    seats: int,
    *,
    bot_class: type[Bot] | None = None,
    opponent_class: type[Bot] | None = None,
) -> dict[str, Any]:
    start = perf_counter()
    cls = bot_class if bot_class is not None else ABLATION_BOTS[name]
    opponent = opponent_class if opponent_class is not None else ValuationBot
    flags = dict(
        zip(
            FEATURE_NAMES,
            (
                getattr(cls, "use_bid_evidence", False),
                getattr(cls, "use_endgame_rules", False),
                getattr(cls, "use_cash_limits", False),
                getattr(cls, "use_bid_model", False),
            ),
        )
    )
    records = []
    with ExitStack() as stack:
        runners = [
            stack.enter_context(BotRunner(c, timeout=2.0, seed=i))
            for i, c in enumerate([cls] + [opponent] * (seats - 1))
        ]
        for seed in seeds:
            offset = seed % seats
            lineup = runners[offset:] + runners[:offset]
            chart = VALUE_CHARTS[seed % len(VALUE_CHARTS)]
            game = play_game(seats, bots=lineup, rng=Random(seed), value_chart=chart)
            seat = lineup.index(runners[0])
            others = [score for i, score in enumerate(game.scores) if i != seat]
            won = seat in game.winner_ids
            records.append(
                {
                    "seed": seed,
                    "chart": chart.name,
                    "seat": seat,
                    "candidate_score": game.scores[seat],
                    "rival_scores": others,
                    "margin": game.scores[seat] - max(others),
                    "win": won,
                    "win_credit": 1 / len(game.winner_ids) if won else 0.0,
                    "winner_count": len(game.winner_ids),
                }
            )
    return {
        "flags": flags,
        "records": records,
        "elapsed_seconds": perf_counter() - start,
    }


def paired_ci(differences: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(differences)
    radius = (
        1.96 * statistics.stdev(differences) / sqrt(len(differences))
        if len(differences) > 1
        else 0
    )
    return mean, mean - radius, mean + radius


def summarize(result: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    records, baseline = result["records"], control["records"]
    assert [(r["seed"], r["seat"], r["chart"]) for r in records] == [
        (r["seed"], r["seat"], r["chart"]) for r in baseline
    ]
    deltas = [float(r["margin"] - b["margin"]) for r, b in zip(records, baseline)]
    score_deltas = [
        float(r["candidate_score"] - b["candidate_score"])
        for r, b in zip(records, baseline)
    ]
    mean, low, high = paired_ci(deltas)
    score_mean, score_low, score_high = paired_ci(score_deltas)
    return {
        "games": len(records),
        "wins_including_ties": sum(r["win"] for r in records),
        "win_credit": sum(r["win_credit"] for r in records),
        "win_rate": statistics.mean(r["win_credit"] for r in records),
        "mean_score": statistics.mean(r["candidate_score"] for r in records),
        "mean_margin": statistics.mean(r["margin"] for r in records),
        "paired_margin_delta": mean,
        "paired_margin_ci_low": low,
        "paired_margin_ci_high": high,
        "paired_score_delta": score_mean,
        "paired_score_ci_low": score_low,
        "paired_score_ci_high": score_high,
    }


def recommend(variants: dict[str, Any]) -> str:
    """Select on training deals only; require wins and paired-margin evidence."""
    reference = variants["baseline"]["summary"]["win_rate"]
    eligible = [
        name
        for name, result in variants.items()
        if name != "baseline"
        and result["summary"]["win_rate"] > reference
        and result["summary"]["paired_margin_ci_low"] > 0
    ]
    if not eligible:
        return "baseline"
    return max(
        eligible,
        key=lambda name: (
            variants[name]["summary"]["win_rate"],
            variants[name]["summary"]["paired_margin_delta"],
            -sum(variants[name]["flags"].values()),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=240)
    parser.add_argument("--start-seed", type=int, default=360)
    parser.add_argument("--seats", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--variants", nargs="+", choices=sorted(ABLATION_BOTS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-variant", choices=sorted(ABLATION_BOTS))
    args = parser.parse_args()
    if args.games < 2 or args.start_seed < 0 or args.jobs < 1:
        parser.error("Use at least two games, a nonnegative seed, and positive jobs")
    names = list(dict.fromkeys(["baseline", *(args.variants or ABLATION_BOTS)]))
    if args.validate_variant and args.validate_variant not in names:
        names.append(args.validate_variant)
    seeds = range(args.start_seed, args.start_seed + args.games)
    errors = CallbackErrors()
    logging.getLogger("bot_runner").addHandler(errors)
    results = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_variant, name, seeds, args.seats): name for name in names
        }
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
            records = results[name]["records"]
            wins = sum(r["win_credit"] for r in records)
            score = statistics.mean(r["candidate_score"] for r in records)
            print(
                f"Completed {name}: {wins:.1f}/{args.games} wins; mean {score:.2f}",
                flush=True,
            )
    if errors.messages:
        raise RuntimeError(f"Invalid benchmark: callback failures: {errors.messages}")
    for name in names:
        results[name]["summary"] = summarize(results[name], results["baseline"])
    report = {
        "games_per_variant": args.games,
        "start_seed": args.start_seed,
        "seats": args.seats,
        "opponents": ["ValuationBot"] * (args.seats - 1),
        "seating": "candidate seat = (-seed) % seats",
        "chart": "ABCDE[seed % 5]",
        "win_counting": "fractional credit for tied winners",
        "ci": "approximate 95% paired margin interval, mean +/- 1.96 SE across seeds",
        "callback_errors": errors.messages,
        "variants": {name: results[name] for name in names},
    }
    if args.validate_variant:
        chosen = results[args.validate_variant]["summary"]
        reference = results["baseline"]["summary"]
        report["validation"] = {
            "variant_from_selection": args.validate_variant,
            "confirmed": args.validate_variant == "baseline"
            or (
                chosen["win_rate"] > reference["win_rate"]
                and chosen["paired_margin_ci_low"] > 0
            ),
            "rule": "higher win rate and positive paired-margin 95% lower bound",
        }
    else:
        report["selection"] = {
            "recommended_variant": recommend(results),
            "rule": "higher win rate and positive paired-margin 95% lower bound; "
            "rank by wins, margin, then fewer flags; default baseline if none",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("\nVariant | win credit | mean score | paired margin delta [95% CI]")
    for name in names:
        row = results[name]["summary"]
        print(
            f"{name} | {row['win_credit']:.1f}/{args.games} | "
            f"{row['mean_score']:.2f} | "
            f"{row['paired_margin_delta']:+.2f} "
            f"[{row['paired_margin_ci_low']:+.2f}, {row['paired_margin_ci_high']:+.2f}]"
        )
    print(f"Full results: {args.output}")


if __name__ == "__main__":
    main()
