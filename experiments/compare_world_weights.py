"""Matched L4 weighted/uniform-world ablation against current L2 opponents.

Only the candidate changes. Uniform uses evidence_weight=0, which also skips
L3's hand inference; bidder, rollout model, limits, and sampling RNG stay fixed.
Weighted decisions additionally compare both aggregations on IDENTICAL completed
worlds, without extra rollouts, to isolate weight effects from timing noise.
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot_definitions.valuation_bot import ValuationBot  # noqa: E402
from bots import BotInfo, GameSummary, RoundResult  # noqa: E402
from experiments.ablate_level3 import CallbackErrors, paired_ci, run_variant, summarize  # noqa: E402
from experiments.weighted_sampling_reference import WeightedSamplingReference  # noqa: E402


class WeightedReference(WeightedSamplingReference):
    """Private benchmark control; never exported as a selectable bot."""

    variant = "weighted"

    def on_game_start(self) -> None:
        super().on_game_start()
        self.same_world_comparisons = 0
        self.same_world_disagreements = 0
        self.round_end_seconds = 0.0

    def _aggregate_worlds(self, outcomes: list[list[int]]) -> list[float]:
        weighted = super()._aggregate_worlds(outcomes)
        uniform = [
            sum(row[i] for row in outcomes) / len(outcomes)
            for i in range(len(outcomes[0]))
        ]

        # Candidates are sorted ascending; both policies prefer the cheaper bid
        # for equal means, exactly as the public search does.
        def choice(scores: list[float]) -> int:
            return max(range(len(scores)), key=scores.__getitem__)

        self.same_world_comparisons += 1
        self.same_world_disagreements += choice(weighted) != choice(uniform)
        return weighted

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        started = perf_counter()
        super().on_round_end(info, result)
        self.round_end_seconds += perf_counter() - started

    def on_game_end(self, info: BotInfo, result: GameSummary) -> None:
        super().on_game_end(info, result)
        path = os.environ.get("MEGAGEM_WEIGHT_METRICS")
        if path:
            record = {
                "variant": self.variant,
                "seats": info.num_players,
                "seat": info.self_player.id,
                "score": result.scores[info.self_player.id],
                "same_world_comparisons": self.same_world_comparisons,
                "same_world_disagreements": self.same_world_disagreements,
                "round_end_seconds": self.round_end_seconds,
                "decisions": [
                    [
                        d["seconds"],
                        d["worlds"],
                        d["rollouts"],
                        d.get("effective_worlds", 0),
                        d["fallback"],
                    ]
                    for d in self.search_log
                ],
            }
            with open(path, "a") as stream:
                stream.write(json.dumps(record) + "\n")


class UniformCandidate(WeightedReference):
    variant = "uniform"

    def __init__(self) -> None:
        super().__init__(evidence_weight=0.0)


# Workers import the real module even when the experiment is run as __main__.
WeightedReference.__module__ = "experiments.compare_world_weights"
UniformCandidate.__module__ = "experiments.compare_world_weights"


def run(name: str, seats: int, start: int, count: int) -> dict[str, Any]:
    errors = CallbackErrors()
    logger = logging.getLogger("bot_runner")
    logger.addHandler(errors)
    try:
        result = run_variant(
            name,
            range(start, start + count),
            seats,
            bot_class=WeightedReference if name == "weighted" else UniformCandidate,
            opponent_class=ValuationBot,
        )
        if errors.messages:
            raise RuntimeError(f"Invalid benchmark: {errors.messages}")
        result.pop("flags")
        return result
    finally:
        logger.removeHandler(errors)


def timing(games: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [d for game in games for d in game["decisions"]]
    seconds = sorted(d[0] for d in decisions)
    comparisons = sum(g["same_world_comparisons"] for g in games)
    disagreements = sum(g["same_world_disagreements"] for g in games)
    return {
        "decisions": len(decisions),
        "mean_bid_seconds": mean(seconds),
        "p95_bid_seconds": seconds[round(0.95 * (len(seconds) - 1))],
        "max_bid_seconds": max(seconds),
        "mean_worlds": mean(d[1] for d in decisions),
        "mean_rollouts": mean(d[2] for d in decisions),
        "mean_effective_worlds": mean(d[3] for d in decisions),
        "mean_round_end_ms": 1000
        * sum(g["round_end_seconds"] for g in games)
        / len(decisions),
        "fallback_reasons": dict(Counter(d[4] for d in decisions if d[4])),
        "same_world_comparisons": comparisons,
        "same_world_disagreements": disagreements,
        "same_world_disagreement_rate": disagreements / comparisons
        if comparisons
        else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--start-seed", type=int, default=3600)
    parser.add_argument(
        "--seats", nargs="+", type=int, choices=(3, 4, 5), default=[3, 4, 5]
    )
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games < 2 or args.start_seed < 0 or args.jobs < 1 or args.batch_size < 1:
        parser.error("Invalid count, seed, jobs or batch size")
    counts = list(dict.fromkeys(args.seats))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output.resolve().with_suffix(".metrics.jsonl")
    metrics_path.write_text("")
    os.environ["MEGAGEM_WEIGHT_METRICS"] = str(metrics_path)
    results: dict[int, dict[str, Any]] = {
        n: {
            name: {"records": [], "elapsed_seconds": 0.0}
            for name in ("weighted", "uniform")
        }
        for n in counts
    }
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=get_context("spawn")
    ) as pool:
        tasks = {
            pool.submit(
                run,
                name,
                n,
                args.start_seed + offset,
                min(args.batch_size, args.games - offset),
            ): (n, name, offset)
            for n in counts
            for offset in range(0, args.games, args.batch_size)
            for name in ("weighted", "uniform")
        }
        for future in as_completed(tasks):
            n, name, offset = tasks[future]
            batch = future.result()
            results[n][name]["records"].extend(batch["records"])
            results[n][name]["elapsed_seconds"] += batch["elapsed_seconds"]
            args.output.with_suffix(".partial.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(
                f"{n} seats {name}: completed {args.start_seed + offset}–"
                f"{args.start_seed + offset + len(batch['records']) - 1}",
                flush=True,
            )
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    if len(metrics) != 2 * args.games * len(counts):
        raise RuntimeError("Missing isolated-worker telemetry")
    for n, variants in results.items():
        for value in variants.values():
            value["records"].sort(key=lambda r: r["seed"])
        for name, value in variants.items():
            value["summary"] = summarize(value, variants["weighted"])
            deltas = [
                a["win_credit"] - b["win_credit"]
                for a, b in zip(value["records"], variants["weighted"]["records"])
            ]
            m, lo, hi = paired_ci(deltas)
            value["summary"].update(
                paired_win_credit_delta=m,
                paired_win_credit_ci_low=lo,
                paired_win_credit_ci_high=hi,
            )
            value["timing"] = timing(
                [g for g in metrics if g["variant"] == name and g["seats"] == n]
            )
    report = {
        "games_per_policy_per_seat_count": args.games,
        "seeds": [args.start_seed, args.start_seed + args.games - 1],
        "batch_size": args.batch_size,
        "parallel_jobs": args.jobs,
        "control": "weighted L4, evidence weight 0.1",
        "candidate": "uniform L4, evidence weight zero; tracker inference skipped",
        "opponents": "all other seats current ValuationBot",
        "limits": {"decision_seconds": 0.85, "max_worlds": 16, "max_candidates": 7},
        "matching": "same deal seed, chart ABCDE[seed%5], candidate seat=(-seed)%seats",
        "win_counting": "one candidate per game, fractional tie credit",
        "ci": "approximate paired 95% intervals, mean +/- 1.96 SE across seeds",
        "practical_equivalence_margin_points": 3.0,
        "note": (
            "Wall-clock search can vary with load. "
            "Identical-world comparisons isolate aggregation effects."
        ),
        "callback_errors": [],
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".partial.json").unlink(missing_ok=True)
    print(
        "\nSeats | Weighted/uniform win credit | Mean points | "
        "Uniform score delta [95% CI]"
    )
    for n, variants in results.items():
        b, a = [variants[k]["summary"] for k in ("weighted", "uniform")]
        print(
            f"{n} | {b['win_credit']:.1f}/{a['win_credit']:.1f} | "
            f"{b['mean_score']:.2f}/{a['mean_score']:.2f} | "
            f"{a['paired_score_delta']:+.2f} [{a['paired_score_ci_low']:+.2f}, "
            f"{a['paired_score_ci_high']:+.2f}]"
        )


if __name__ == "__main__":
    main()
