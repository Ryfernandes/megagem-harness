"""Evaluate shadow count forecasts without letting them influence any bids.

Forecasts are captured BEFORE current bids, from one normal BotInfo snapshot.
The shadow learns bids/reveals only afterward. Actual final counts are attached
only after finish_game reveals all remaining hands. All actions come from the
ordinary ValuationBot/FixedRuleBot workers; shadow get_bid is never called.

Primary losses average informative color-rounds within each game, then profiles
within seed. Approximate 95% paired intervals cluster on seed, not correlated
rounds/colors/profiles. Known-count forecasts are logged but excluded from the
primary comparison. Positive loss gain means the tracker beats the plain prior.

Prediction gate (declared before evaluation): positive lower bounds for MSE and
Brier gains, nonnegative mean log-loss gain, and no profile with clearly worse
MSE (upper bound below zero). Passing is necessary, not sufficient, for driving
bids: the separate strategy ablation must also have validated inference.
"""

import argparse
import csv
import gzip
import json
import logging
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from math import comb, isfinite, log, sqrt
from multiprocessing import get_context
from pathlib import Path
from random import Random
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bot_definitions.soft_tracking_bot import SoftTrackingBot  # noqa: E402
from bot_definitions.valuation_bot import FractionalValuationBot as ValuationBot  # noqa: E402
from bot_info import build_bot_info  # noqa: E402
from bot_runner import BotRunner  # noqa: E402
from bots import Bot, BotInfo, RoundResult  # noqa: E402
from constants import STANDARD_GEMS, GemType  # noqa: E402
from custom_bots import FixedRuleBot  # noqa: E402
from decks import VALUE_CHARTS  # noqa: E402
from harness import create_game, finish_game, play_round  # noqa: E402

METRICS = ("mse", "mae", "brier", "log_loss")
PROFILES = ("valuation", "fixed", "mixed")


def prior_final_pmf(info: BotInfo, gem: GemType) -> tuple[float, ...]:
    """Exact hypergeometric prior over FINAL display counts, including own hand."""
    known = info.value_display.get(gem, 0) + info.self_player.hidden_gems.count(gem)
    population = sum(info.unknown_gems.values())
    successes = info.unknown_gems[gem]
    draws = sum(p.hidden_gem_count for p in info.other_players)
    denominator = comb(population, draws)
    pmf = [0.0] * 7
    for n in range(max(0, draws - population + successes), min(successes, draws) + 1):
        pmf[known + n] = (
            comb(successes, n) * comb(population - successes, draws - n) / denominator
        )
    return tuple(pmf)


def moments(pmf: tuple[float, ...]) -> tuple[float, float]:
    if len(pmf) != 7 or min(pmf) < -1e-12 or abs(sum(pmf) - 1) > 1e-8:
        raise ValueError("Invalid final-count probability distribution")
    mean = sum(n * p for n, p in enumerate(pmf))
    return mean, sum(p * (n - mean) ** 2 for n, p in enumerate(pmf))


def losses(pmf: tuple[float, ...], actual: int) -> dict[str, float]:
    mean, _ = moments(pmf)
    return {
        "mse": (mean - actual) ** 2,
        "mae": abs(mean - actual),
        "brier": sum((p - int(n == actual)) ** 2 for n, p in enumerate(pmf)),
        "log_loss": -log(max(pmf[actual], 1e-300)),
    }


class ShadowForecasts:
    """No game state or truth access; only one seat's ordinary observations."""

    def __init__(self, *, evidence_weight: float | None = None) -> None:
        self.tracker = SoftTrackingBot(
            use_bid_evidence=True,
            use_endgame_rules=False,
            use_cash_limits=False,
            use_bid_model=False,
            evidence_weight=evidence_weight,
        )
        self.rows: list[dict[str, Any]] = []
        self.awaiting_result = False
        self.last_round = 0
        self.finished = False

    def before_bid(self, info: BotInfo) -> None:
        if (
            self.finished
            or self.awaiting_result
            or info.round_number <= self.last_round
        ):
            raise ValueError("Forecasts must precede each round's bids exactly once")
        self.tracker.observe_auction(info)
        belief = self.tracker.beliefs(info) if self.tracker.evidence_weight else None
        hand = Counter(info.self_player.hidden_gems)
        for gem in STANDARD_GEMS:
            known = info.value_display.get(gem, 0) + hand[gem]
            tracked = [0.0] * 7
            prior_pmf = prior_final_pmf(info, gem)
            if belief is not None:
                for hidden, mass in enumerate(belief.hidden_count_probabilities[gem]):
                    tracked[known + hidden] += mass
                tracker_pmf = tuple(tracked)
            else:
                tracker_pmf = prior_pmf
            tracker_mean, tracker_variance = moments(tracker_pmf)
            prior_mean, prior_variance = moments(prior_pmf)
            self.rows.append(
                {
                    "round": info.round_number,
                    "color": gem.name,
                    "deck_count": info.gem_deck_count,
                    "opponent_hidden_count": sum(
                        p.hidden_gem_count for p in info.other_players
                    ),
                    "public_count": info.value_display.get(gem, 0),
                    "own_hidden_count": hand[gem],
                    "unseen_color_count": info.unknown_gems[gem],
                    "tracker_mean": tracker_mean,
                    "prior_mean": prior_mean,
                    "tracker_variance": tracker_variance,
                    "prior_variance": prior_variance,
                    "tracker_pmf": tracker_pmf,
                    "prior_pmf": prior_pmf,
                    "informative": prior_variance > 1e-12,
                }
            )
        self.last_round = info.round_number
        self.awaiting_result = True

    def after_round(self, info: BotInfo, result: RoundResult) -> None:
        if not self.awaiting_result or info.round_number != self.last_round:
            raise ValueError("Observe bids/reveals only after their forecast")
        self.tracker.on_round_end(info, result)
        self.awaiting_result = False

    def finalize(self, final_info: BotInfo) -> list[dict[str, Any]]:
        if (
            self.finished
            or self.awaiting_result
            or final_info.gems_remaining
            or final_info.self_player.hidden_gems
            or any(p.hidden_gem_count for p in final_info.other_players)
        ):
            raise ValueError("Actual counts may only be attached after final reveals")
        scored = []
        for original in self.rows:
            row = dict(original)
            actual = final_info.value_display.get(GemType[row["color"]], 0)
            row["actual_count"] = actual
            for model in ("tracker", "prior"):
                for metric, value in losses(row[f"{model}_pmf"], actual).items():
                    row[f"{model}_{metric}"] = value
            scored.append(row)
        self.finished = True
        return scored


class CallbackErrors(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def game_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    informative = [r for r in rows if r["informative"]]
    if not informative:
        raise ValueError("No informative observations in this game")
    metrics = {}
    for metric in METRICS:
        prior = statistics.mean(r[f"prior_{metric}"] for r in informative)
        tracker = statistics.mean(r[f"tracker_{metric}"] for r in informative)
        metrics[metric] = {"prior": prior, "tracker": tracker, "gain": prior - tracker}
    known = [r for r in rows if not r["informative"]]
    if any(abs(r["tracker_mean"] - r["actual_count"]) > 1e-8 for r in known):
        raise ValueError("A supposedly known terminal count was predicted incorrectly")
    return {
        "rows": len(rows),
        "informative_rows": len(informative),
        "known_rows": len(known),
        "metrics": metrics,
    }


def run_job(
    profile: str,
    seats: int,
    seeds: list[int],
    output: str,
    evidence_weight: float | None = None,
) -> dict[str, Any]:
    errors = CallbackErrors()
    logging.getLogger("bot_runner").addHandler(errors)
    path = Path(output) / f"{profile}_{seats}_forecasts.csv.gz"
    games = []
    samples = []
    with ExitStack() as stack, gzip.open(path, "wt", newline="") as handle:
        classes: list[type[Bot]] = [ValuationBot]
        for i in range(seats - 1):
            classes.append(
                ValuationBot
                if profile == "valuation" or (profile == "mixed" and i % 2 == 0)
                else FixedRuleBot
            )
        runners = [
            stack.enter_context(BotRunner(c, timeout=2.0, seed=i))
            for i, c in enumerate(classes)
        ]
        writer = None
        for seed in seeds:
            observer = (seed // 15) % seats
            lineup = runners[-observer:] + runners[:-observer] if observer else runners
            chart = VALUE_CHARTS[seed % 5]
            shadow = ShadowForecasts(evidence_weight=evidence_weight)
            with create_game(
                seats, bots=lineup, rng=Random(seed), value_chart=chart
            ) as game:
                while game.auction_gems:
                    result = play_round(
                        game, before_bid=shadow.before_bid, observer_id=observer
                    )
                    shadow.after_round(
                        build_bot_info(game, game.players[observer]), result
                    )
                finish_game(game)
                rows = shadow.finalize(build_bot_info(game, game.players[observer]))
            metadata = {
                "seed": seed,
                "profile": profile,
                "seats": seats,
                "observer_seat": observer,
                "chart": chart.name,
            }
            for row in rows:
                record = {**metadata, **row}
                for key in ("tracker_pmf", "prior_pmf"):
                    record[key] = json.dumps(record[key], separators=(",", ":"))
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(record))
                    writer.writeheader()
                writer.writerow(record)
                if seed == seeds[0] and row["round"] in (1, 5, 10):
                    samples.append(record)
            games.append({**metadata, **game_metrics(rows)})
    if errors.messages:
        raise RuntimeError(
            f"Callback failures invalidate evaluation: {errors.messages}"
        )
    return {
        "profile": profile,
        "seats": seats,
        "log": str(path),
        "games": games,
        "samples": samples,
        "callback_errors": errors.messages,
    }


def aggregate(games: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        by_seed[game["seed"]].append(game)
    result: dict[str, Any] = {
        "games": len(games),
        "independent_seed_clusters": len(by_seed),
        "rows": sum(g["rows"] for g in games),
        "informative_rows": sum(g["informative_rows"] for g in games),
        "known_rows": sum(g["known_rows"] for g in games),
    }
    for metric in METRICS:
        seed_gains = [
            statistics.mean(g["metrics"][metric]["gain"] for g in group)
            for group in by_seed.values()
        ]
        gain = statistics.mean(seed_gains)
        radius = (
            1.96 * statistics.stdev(seed_gains) / sqrt(len(seed_gains))
            if len(seed_gains) > 1
            else 0
        )
        result[metric] = {
            "tracker": statistics.mean(
                statistics.mean(g["metrics"][metric]["tracker"] for g in group)
                for group in by_seed.values()
            ),
            "prior": statistics.mean(
                statistics.mean(g["metrics"][metric]["prior"] for g in group)
                for group in by_seed.values()
            ),
            "gain": gain,
            "gain_ci_low": gain - radius,
            "gain_ci_high": gain + radius,
        }
    return result


def prediction_gate(overall: dict[str, Any], profiles: dict[str, Any]) -> bool:
    return bool(
        overall["mse"]["gain_ci_low"] > 0
        and overall["brier"]["gain_ci_low"] > 0
        and overall["log_loss"]["gain"] >= 0
        and not any(p["mse"]["gain_ci_high"] < 0 for p in profiles.values())
    )


def summarize_colors(logs: list[str]) -> dict[str, Any]:
    """Derive per-color diagnostics from the recorded forecast/actual pairs."""
    groups: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    for filename in logs:
        with gzip.open(filename, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    int(row["seed"]),
                    row["profile"],
                    int(row["seats"]),
                    row["color"],
                )
                group = groups.setdefault(
                    key,
                    {
                        "seed": key[0],
                        "rows": 0,
                        "informative_rows": 0,
                        "known_rows": 0,
                        "sums": {m: {"prior": 0.0, "tracker": 0.0} for m in METRICS},
                        "means": {"prior": 0.0, "tracker": 0.0, "actual": 0.0},
                    },
                )
                group["rows"] += 1
                if row["informative"] != "True":
                    group["known_rows"] += 1
                    continue
                group["informative_rows"] += 1
                for metric in METRICS:
                    for model in ("prior", "tracker"):
                        group["sums"][metric][model] += float(row[f"{model}_{metric}"])
                group["means"]["prior"] += float(row["prior_mean"])
                group["means"]["tracker"] += float(row["tracker_mean"])
                group["means"]["actual"] += float(row["actual_count"])
    by_color: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, group in groups.items():
        n = group["informative_rows"]
        if not n:
            continue
        group["metrics"] = {}
        for metric in METRICS:
            prior = group["sums"][metric]["prior"] / n
            tracker = group["sums"][metric]["tracker"] / n
            group["metrics"][metric] = {
                "prior": prior,
                "tracker": tracker,
                "gain": prior - tracker,
            }
        group["means"] = {name: value / n for name, value in group["means"].items()}
        by_color[key[3]].append(group)
    result = {}
    for color, games in by_color.items():
        report = aggregate(games)
        seeds: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for game in games:
            seeds[game["seed"]].append(game)
        report["mean_counts"] = {
            model: statistics.mean(
                statistics.mean(g["means"][model] for g in group)
                for group in seeds.values()
            )
            for model in ("prior", "tracker", "actual")
        }
        result[color] = report
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=int, default=300, help="Fresh deal seeds per profile"
    )
    parser.add_argument("--start-seed", type=int, default=840)
    parser.add_argument(
        "--seats", nargs="+", type=int, choices=(3, 4, 5), default=[3, 4, 5]
    )
    parser.add_argument(
        "--profiles", nargs="+", choices=PROFILES, default=list(PROFILES)
    )
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evidence-weight", type=float, default=SoftTrackingBot.evidence_weight
    )
    args = parser.parse_args()
    if args.games < 2 or args.start_seed < 0 or args.jobs < 1:
        parser.error("Use at least two seeds, a nonnegative seed, and positive jobs")
    if not isfinite(args.evidence_weight) or not 0 <= args.evidence_weight <= 1:
        parser.error("Evidence weight must be finite and between zero and one")
    seats = list(dict.fromkeys(args.seats))
    profiles = list(dict.fromkeys(args.profiles))
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = range(args.start_seed, args.start_seed + args.games)
    jobs = [
        (
            profile,
            n,
            [seed for i, seed in enumerate(seeds) if seats[i % len(seats)] == n],
        )
        for profile in profiles
        for n in seats
    ]
    jobs = [j for j in jobs if j[2]]
    reports = []
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(
                run_job, p, n, values, str(args.output), args.evidence_weight
            ): (p, n)
            for p, n, values in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            reports.append(result)
            print(
                f"Completed {result['profile']}, {result['seats']} seats: "
                f"{len(result['games'])} games",
                flush=True,
            )
    games = sorted(
        (g for r in reports for g in r["games"]),
        key=lambda g: (g["seed"], g["profile"]),
    )
    overall = aggregate(games)
    by_profile = {
        p: aggregate([g for g in games if g["profile"] == p]) for p in profiles
    }
    by_seats = {str(n): aggregate([g for g in games if g["seats"] == n]) for n in seats}
    by_chart = {
        c: aggregate([g for g in games if g["chart"] == c])
        for c in "ABCDE"
        if any(g["chart"] == c for g in games)
    }
    qualifies = prediction_gate(overall, by_profile)
    samples = [
        row
        for r in sorted(reports, key=lambda r: (r["profile"], r["seats"]))
        for row in r["samples"]
    ]
    with (args.output / "examples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    report = {
        "start_seed": args.start_seed,
        "deal_seeds": args.games,
        "evidence_weight": args.evidence_weight,
        "profiles": profiles,
        "seats": seats,
        "observer": "seat = (seed // 15) % seats; always ValuationBot",
        "prediction_timing": (
            "before current bids; learn afterward; truth after final reveals"
        ),
        "primary_rows": "prior variance > 1e-12; equal game weights within seed",
        "ci": "approximate 95% paired intervals clustered by deal seed across profiles",
        "gate_rule": (
            "positive MSE/Brier gain lower bounds; mean log-loss gain >= 0; "
            "no profile with clearly worse MSE"
        ),
        "prediction_gate_passed": qualifies,
        "bidding_inference_enabled": SoftTrackingBot.use_bid_evidence,
        "deployment": "Diagnostic only; separate strategy validation required.",
        "overall": overall,
        "by_profile": by_profile,
        "by_seats": by_seats,
        "by_chart": by_chart,
        "logs": [r["log"] for r in reports],
        "callback_errors": [],
        "games": games,
    }
    report["by_color"] = summarize_colors(report["logs"])
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nInformative forecasts; positive gain means tracker is better:")
    for name, group in [("overall", overall), *by_profile.items()]:
        row = group["mse"]
        print(
            f"{name}: prior MSE={row['prior']:.6f}, tracker MSE={row['tracker']:.6f}, "
            f"gain={row['gain']:+.6f} "
            f"[{row['gain_ci_low']:+.6f}, {row['gain_ci_high']:+.6f}]"
        )
    print(
        f"Prediction gate passed: {qualifies}; "
        f"bidding inference remains: {SoftTrackingBot.use_bid_evidence}"
    )


if __name__ == "__main__":
    main()
