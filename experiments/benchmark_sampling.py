"""Matched level-4 versus level-2 games, with real isolated-worker search metrics."""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot_definitions.sampling_bot import SamplingBot  # noqa: E402
from bot_definitions.valuation_bot import ValuationBot  # noqa: E402
from bots import BotInfo, GameSummary  # noqa: E402
from experiments.ablate_level3 import CallbackErrors, paired_ci, run_variant, summarize  # noqa: E402
from experiments.weighted_sampling_reference import WeightedSamplingReference  # noqa: E402


class _MeasuredSamplingBot(SamplingBot):
    """Benchmark-only file telemetry; the public bot has no filesystem side effects."""

    def on_game_end(self, info: BotInfo, result: GameSummary) -> None:
        super().on_game_end(info, result)
        path = os.environ.get("MEGAGEM_SAMPLING_METRICS")
        if path:
            rows = [
                [
                    d["round"],
                    d["seconds"],
                    d["worlds"],
                    d["rollouts"],
                    d.get("effective_worlds", d["worlds"]),
                    d["fallback"],
                ]
                for d in self.search_log
            ]
            record = {
                "seats": info.num_players,
                "seat": info.self_player.id,
                "score": result.scores[info.self_player.id],
                "decisions": rows,
            }
            with open(path, "a") as stream:
                stream.write(json.dumps(record) + "\n")


# BotRunner resolves workers by importable module path, including when this
# file is launched as a command or imported in multiprocessing's main module.
_MeasuredSamplingBot.__module__ = "experiments.benchmark_sampling"


class _MeasuredWeightedSamplingBot(_MeasuredSamplingBot, WeightedSamplingReference):
    """Private historical control, available only through an explicit CLI flag."""


_MeasuredWeightedSamplingBot.__module__ = "experiments.benchmark_sampling"


def run(
    name: str, seats: int, start: int, count: int, weighted: bool = False
) -> dict[str, Any]:
    errors = CallbackErrors()
    logger = logging.getLogger("bot_runner")
    logger.addHandler(errors)
    try:
        cls = _MeasuredSamplingBot if name == "sampling" else ValuationBot
        if name == "sampling" and weighted:
            cls = _MeasuredWeightedSamplingBot
        result = run_variant(
            name,
            range(start, start + count),
            seats,
            bot_class=cls,
            opponent_class=ValuationBot,
        )
        if errors.messages:
            raise RuntimeError(f"Invalid benchmark: {errors.messages}")
        result.pop("flags")
        return result
    finally:
        logger.removeHandler(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--start-seed", type=int, default=3300)
    parser.add_argument(
        "--seats", nargs="+", choices=(3, 4, 5), type=int, default=[3, 4, 5]
    )
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--weighted-reference",
        action="store_true",
        help="Reproduce the original weighted L4 control instead of current uniform L4",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games < 2 or args.start_seed < 0 or args.jobs < 1 or args.batch_size < 1:
        parser.error("Invalid count, seed or jobs")
    counts = list(dict.fromkeys(args.seats))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output.resolve().with_suffix(".metrics.jsonl")
    metrics_path.write_text("")
    os.environ["MEGAGEM_SAMPLING_METRICS"] = str(metrics_path)
    results: dict[int, dict[str, Any]] = {
        n: {
            name: {"records": [], "elapsed_seconds": 0.0}
            for name in ("level2", "sampling")
        }
        for n in counts
    }
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=get_context("spawn")
    ) as pool:
        pending = {
            pool.submit(
                run,
                name,
                n,
                args.start_seed + offset,
                min(args.batch_size, args.games - offset),
                args.weighted_reference,
            ): (n, name, offset)
            for n in counts
            for name in ("level2", "sampling")
            for offset in range(0, args.games, args.batch_size)
        }
        for future in as_completed(pending):
            n, name, offset = pending[future]
            batch = future.result()
            results[n][name]["records"].extend(batch["records"])
            results[n][name]["elapsed_seconds"] += batch["elapsed_seconds"]
            # Preserve full scores/wins, not just telemetry, if a long run is
            # externally interrupted. Only the final JSON is a valid comparison.
            args.output.with_suffix(".partial.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(
                f"{n} seats, {name}: completed seeds "
                f"{args.start_seed + offset}–"
                f"{args.start_seed + offset + len(batch['records']) - 1}",
                flush=True,
            )
    for variants in results.values():
        for value in variants.values():
            value["records"].sort(key=lambda record: record["seed"])
        control = variants["level2"]
        for value in variants.values():
            value["summary"] = summarize(value, control)
            d = [
                a["win_credit"] - b["win_credit"]
                for a, b in zip(value["records"], control["records"])
            ]
            m, lo, hi = paired_ci(d)
            value["summary"].update(
                paired_win_credit_delta=m,
                paired_win_credit_ci_low=lo,
                paired_win_credit_ci_high=hi,
            )
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    if len(metrics) != args.games * len(counts):
        raise RuntimeError("Missing isolated-worker search telemetry")
    performance = {}
    for n in counts:
        decisions = [
            d for game in metrics if game["seats"] == n for d in game["decisions"]
        ]
        seconds = sorted(d[1] for d in decisions)
        performance[n] = {
            "decisions": len(decisions),
            "mean_seconds": mean(seconds),
            "p95_seconds": seconds[round(0.95 * (len(seconds) - 1))],
            "max_seconds": max(seconds),
            "mean_worlds": mean(d[2] for d in decisions),
            "min_worlds": min(d[2] for d in decisions),
            "mean_rollouts": mean(d[3] for d in decisions),
            "mean_effective_worlds": mean(d[4] for d in decisions),
            "fallback_decisions": sum(d[5] is not None for d in decisions),
            "forced_zero_decisions": sum(d[2] == 0 and d[5] is None for d in decisions),
        }
    report = {
        "games_per_policy_per_seat_count": args.games,
        "start_seed": args.start_seed,
        "end_seed": args.start_seed + args.games - 1,
        "batch_size": args.batch_size,
        "parallel_jobs": args.jobs,
        "opponents": "all other seats current ValuationBot",
        "objective": (
            "importance-weighted mean own final coin score"
            if args.weighted_reference
            else "uniform mean own final coin score"
        ),
        "search": {
            "seconds": 0.85,
            "max_worlds": 16,
            "max_candidates": 7,
            "min_worlds": 2,
            "evidence_weight": 0.1 if args.weighted_reference else 0.0,
        },
        "tracker_enabled": args.weighted_reference,
        "other_level3_experiments_enabled": False,
        "matching": "same deal seed; chart ABCDE[seed%5]; candidate seat=(-seed)%seats",
        "win_counting": "fractional tie credit for the single candidate",
        "ci": "approximate paired 95% intervals, mean +/- 1.96 SE across seeds",
        "timing_note": "time-limited choices vary with CPU load despite fixed seeds",
        "callback_errors": [],
        "performance": performance,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".partial.json").unlink(missing_ok=True)
    print("\nSeats | L2/L4 wins | L2/L4 mean points | score change [95% CI]")
    for n, v in results.items():
        b, a = (v[k]["summary"] for k in ("level2", "sampling"))
        print(
            f"{n} | {b['win_credit']:.2f}/{a['win_credit']:.2f} | "
            f"{b['mean_score']:.2f}/{a['mean_score']:.2f} | "
            f"{a['paired_score_delta']:+.2f} "
            f"[{a['paired_score_ci_low']:+.2f}, {a['paired_score_ci_high']:+.2f}]"
        )
    print(json.dumps(performance, indent=2))


if __name__ == "__main__":
    main()
