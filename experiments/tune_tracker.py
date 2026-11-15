"""Tune evidence shrinkage on shared shadow trajectories, then validate untouched.

Weights temper RAW decaying evidence: posterior proportional to prior * exp(w E).
Zero is the prior, 0.1 is the requested conservative starting weight, and 0.4
reproduces the historical tracker. All shadows see identical chronological
observations; none can change play. Selection minimizes informative count MSE,
with weaker weights breaking exact ties. Validation never retunes its candidate.
"""

import argparse
import csv
import gzip
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from math import isfinite
from multiprocessing import get_context
from pathlib import Path
from random import Random
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot_definitions.valuation_bot import FractionalValuationBot as ValuationBot  # noqa: E402
from bot_info import build_bot_info  # noqa: E402
from bot_runner import BotRunner  # noqa: E402
from bots import Bot, BotInfo  # noqa: E402
from custom_bots import FixedRuleBot, SoftTrackingBot  # noqa: E402
from decks import VALUE_CHARTS  # noqa: E402
from experiments.evaluate_tracker import (  # noqa: E402
    METRICS,
    PROFILES,
    CallbackErrors,
    ShadowForecasts,
    aggregate,
    game_metrics,
    prediction_gate,
)
from harness import create_game, finish_game, play_round  # noqa: E402


def weight_key(weight: float) -> str:
    return "0" if weight == 0 else repr(float(weight))


def run_job(
    profile: str,
    seats: int,
    seeds: list[int],
    weights: list[float],
    output: str,
    log_weight: float | None,
) -> dict[str, Any]:
    errors = CallbackErrors()
    logging.getLogger("bot_runner").addHandler(errors)
    games: dict[str, list[dict[str, Any]]] = {weight_key(w): [] for w in weights}
    samples = []
    with ExitStack() as stack:
        classes: list[type[Bot]] = [ValuationBot]
        for i in range(seats - 1):
            classes.append(
                ValuationBot
                if profile == "valuation" or (profile == "mixed" and i % 2 == 0)
                else FixedRuleBot
            )
        runners = [
            stack.enter_context(BotRunner(cls, timeout=2, seed=i))
            for i, cls in enumerate(classes)
        ]
        writer = None
        if log_weight is not None:
            handle = stack.enter_context(
                gzip.open(
                    Path(output) / f"{profile}_{seats}_forecasts.csv.gz",
                    "wt",
                    newline="",
                )
            )
        for seed in seeds:
            observer = (seed // 15) % seats
            lineup = runners[-observer:] + runners[:-observer] if observer else runners
            chart = VALUE_CHARTS[seed % 5]
            shadows = {
                weight_key(w): ShadowForecasts(evidence_weight=w) for w in weights
            }

            def before_bid(info: BotInfo) -> None:
                for shadow in shadows.values():
                    shadow.before_bid(info)

            with create_game(
                seats, bots=lineup, rng=Random(seed), value_chart=chart
            ) as game:
                while game.auction_gems:
                    result = play_round(
                        game, before_bid=before_bid, observer_id=observer
                    )
                    info = build_bot_info(game, game.players[observer])
                    for shadow in shadows.values():
                        shadow.after_round(info, result)
                finish_game(game)
                terminal = build_bot_info(game, game.players[observer])
                for weight, shadow in shadows.items():
                    rows = shadow.finalize(terminal)
                    metadata = {
                        "seed": seed,
                        "profile": profile,
                        "seats": seats,
                        "chart": chart.name,
                        "observer_seat": observer,
                        "evidence_weight": float(weight),
                    }
                    games[weight].append({**metadata, **game_metrics(rows)})
                    for row in rows:
                        sample = seed == seeds[0] and row["round"] in (1, 5, 10)
                        should_log = log_weight is not None and weight == weight_key(
                            log_weight
                        )
                        if not sample and not should_log:
                            continue
                        record = {**metadata, **row}
                        for field in ("tracker_pmf", "prior_pmf"):
                            record[field] = json.dumps(
                                record[field], separators=(",", ":")
                            )
                        if sample:
                            samples.append(record)
                        if should_log:
                            if writer is None:
                                writer = csv.DictWriter(handle, fieldnames=list(record))
                                writer.writeheader()
                            writer.writerow(record)
    if errors.messages:
        raise RuntimeError(f"Invalid evaluation: {errors.messages}")
    return {
        "games": games,
        "samples": samples,
        "profile": profile,
        "seats": seats,
        "callback_errors": errors.messages,
    }


def compare_weights(
    candidate: list[dict[str, Any]], reference: list[dict[str, Any]]
) -> dict[str, Any]:
    assert [(r["seed"], r["profile"], r["seats"]) for r in candidate] == [
        (r["seed"], r["profile"], r["seats"]) for r in reference
    ]
    paired = []
    for a, b in zip(candidate, reference):
        entry = {k: a[k] for k in ("seed", "rows", "informative_rows", "known_rows")}
        entry["metrics"] = {
            m: {
                "tracker": a["metrics"][m]["tracker"],
                "prior": b["metrics"][m]["tracker"],
                "gain": b["metrics"][m]["tracker"] - a["metrics"][m]["tracker"],
            }
            for m in METRICS
        }
        paired.append(entry)
    report = aggregate(paired)
    for m in METRICS:
        report[m]["reference"] = report[m].pop("prior")
        report[m]["candidate"] = report[m].pop("tracker")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", nargs="+", type=float, default=[0, 0.025, 0.05, 0.1, 0.2, 0.4]
    )
    parser.add_argument("--games", type=int, default=90, help="Deal seeds per profile")
    parser.add_argument("--start-seed", type=int, default=840)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-weight", type=float)
    parser.add_argument("--log-weight", type=float)
    args = parser.parse_args()
    weights = sorted(set([0.0, *args.weights]))
    if args.games < 3 or args.start_seed < 0 or args.jobs < 1:
        parser.error("Use at least three games, nonnegative seeds, positive jobs")
    if any(not isfinite(w) or not 0 <= w <= 1 for w in weights):
        parser.error("Weights must be finite in [0,1]")
    if args.validate_weight is not None and args.validate_weight not in weights:
        parser.error("Validation weight must be in --weights")
    if args.log_weight is not None and args.log_weight not in weights:
        parser.error("Log weight must be in --weights")
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = range(args.start_seed, args.start_seed + args.games)
    jobs = [
        (p, n, [s for i, s in enumerate(seeds) if 3 + i % 3 == n])
        for p in PROFILES
        for n in (3, 4, 5)
    ]
    reports = []
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(
                run_job, p, n, values, weights, str(args.output), args.log_weight
            ): (p, n)
            for p, n, values in jobs
        }
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(
                f"Completed {report['profile']}, {report['seats']} seats: "
                f"{len(report['games']['0'])} games, {len(weights)} weights",
                flush=True,
            )
    models: dict[str, dict[str, Any]] = {}
    for w in weights:
        key = weight_key(w)
        games = sorted(
            (g for r in reports for g in r["games"][key]),
            key=lambda g: (g["seed"], g["profile"]),
        )
        models[key] = {
            "games": games,
            "overall": aggregate(games),
            "by_profile": {
                p: aggregate([g for g in games if g["profile"] == p]) for p in PROFILES
            },
        }
    selection = min(
        weights, key=lambda w: (models[weight_key(w)]["overall"]["mse"]["tracker"], w)
    )
    selected = args.validate_weight if args.validate_weight is not None else selection
    summary: dict[str, Any] = {
        "phase": "validation" if args.validate_weight is not None else "selection",
        "start_seed": args.start_seed,
        "deal_seeds": args.games,
        "game_trajectories": args.games * 3,
        "weights": weights,
        "weight_formula": "prior * exp(weight * raw decaying evidence)",
        "legacy_weight": 0.4,
        "selected_weight": selected,
        "selection_rule": "Minimum count MSE; weaker breaks ties; frozen on validation",
        "ci": "Paired trajectory losses; 95% intervals cluster by seed across profiles",
        "actual_play": "Level-2 observer; estimates never influence actual actions",
        "models": models,
        "logged_weight": args.log_weight,
        "forecast_logs": (
            [str(path) for path in sorted(args.output.glob("*_forecasts.csv.gz"))]
            if args.log_weight is not None
            else []
        ),
        "bidding_inference_enabled": SoftTrackingBot.use_bid_evidence,
        "callback_errors": [],
    }
    chosen = models[weight_key(selected)]
    summary["prediction_gate_passed"] = prediction_gate(
        chosen["overall"], chosen["by_profile"]
    )
    if 0.4 in weights:
        summary["selected_vs_legacy"] = compare_weights(
            chosen["games"], models["0.4"]["games"]
        )
        summary["comparisons_vs_legacy"] = {
            key: compare_weights(model["games"], models["0.4"]["games"])
            for key, model in models.items()
        }
    samples = [row for r in reports for row in r["samples"]]
    with (args.output / "examples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nWeight | count MSE | gain vs prior [95% CI]")
    for w in weights:
        r = models[weight_key(w)]["overall"]["mse"]
        print(
            f"{w:g} | {r['tracker']:.6f} | {r['gain']:+.6f} "
            f"[{r['gain_ci_low']:+.6f}, {r['gain_ci_high']:+.6f}]"
        )
    print(
        f"{summary['phase']} weight: {selected}; "
        f"prediction gate: {summary['prediction_gate_passed']}"
    )


if __name__ == "__main__":
    main()
