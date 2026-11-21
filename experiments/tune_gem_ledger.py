"""Search GemLedgerBot's tuning numbers against the other submitted bots.

Run from the project folder.

On Windows:

    py -3 -u experiments\tune_gem_ledger.py

On Linux:

    python3 -u experiments/tune_gem_ledger.py

That uses about two hours. It saves progress after every batch of games, so
stopping it and running the same command again continues where it left off.

The other bots are downloaded into a temporary folder from the open class
pull requests. SamplingBot is not included. The temporary folder is deleted
when the script exits.
"""

from __future__ import annotations

import argparse
import atexit
import itertools
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path
from time import perf_counter, time

ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = ROOT / "src"
WORK_ROOT = Path(tempfile.gettempdir()) / "gem-ledger-tune"
WORK_SRC = WORK_ROOT / "src"
RESULTS_PATH = ROOT / "experiments" / "gem_ledger_tune_results.json"

# These are the exact commits the last tournament was played against.
FIELD_FILES: tuple[tuple[str, str], ...] = (
    ("3b6e0baa499b1de630770edf9512406c2c118cc1", "adaptive_bot.py"),
    ("3b6e0baa499b1de630770edf9512406c2c118cc1", "aggressive_bot.py"),
    ("3b6e0baa499b1de630770edf9512406c2c118cc1", "pricing.py"),
    ("3b6e0baa499b1de630770edf9512406c2c118cc1", "value_liquidity_bot.py"),
    ("6e188e96d64b7cf66640c4a006403442ad36cc18", "yale_bot.py"),
    ("632c92cb534709b678593354c79633c286be1a70", "adaptive_finance_bots.py"),
    ("8da825bd50ceab2ddd7d608eef33a58327025abd", "valuation_bot.py"),
    ("8da825bd50ceab2ddd7d608eef33a58327025abd", "expected_utility.py"),
)

# GemLedgerBot sits in every game. These fill the other seats.
FIELD = (
    "YaleBot",
    "AdaptiveBot",
    "AggressiveBot",
    "ValueLiquidityBot",
    "ConservativeAdaptiveBot",
    "LoanAwareAdaptiveBot",
    "FinanceAwareAdaptiveBot",
    "ValuationBot",
)

# One-at-a-time values. The current number is left out on purpose, because
# the baseline games already measure it.
KNOBS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("GL_CFG_COIN_SCARCITY", ("2", "3", "5", "6"), "coin_scarcity_gain"),
    ("GL_CFG_ENDGAME_GEMS", ("4", "6", "12", "16"), "endgame_gem_threshold"),
    ("GL_CFG_ENDGAME_FLOOR", ("0", "0.2", "0.5", "0.7"), "endgame_premium_floor"),
    ("GL_CFG_MAX_COIN", ("1", "1.5", "3", "4"), "max_coin_value"),
    ("GL_CFG_RIVAL_CONTEST", ("0", "0.25", "0.75", "1"), "rival_mission_contest"),
    ("GL_CFG_MISSION_DIVISOR", ("5", "8", "15", "20"), "mission_urgency_divisor"),
    (
        "GL_CFG_OPENING_RATIOS",
        ("0.2,0.4,0.6", "0.35,0.5,0.7", "0.25,0.45,0.75"),
        "opening_ratios",
    ),
    ("GL_CFG_OPENING_WEIGHT", ("0", "0.1", "0.5", "1"), "opening_weight"),
    ("GL_CFG_OPENING_DROP", ("1", "4", "8"), "opening_drop_after"),
    ("GL_CFG_MAX_RECORDS", ("4", "8", "24"), "max_records"),
    ("GL_CFG_TABLE_WEIGHT", ("0", "0.5", "2"), "table_weight"),
    ("GL_CFG_SEAT_WEIGHT", ("1", "2", "5"), "seat_weight"),
)

SETTING_FILE = {
    "GL_CFG_COIN_SCARCITY": "src/bot_definitions/gem_ledger_pricing.py",
    "GL_CFG_ENDGAME_GEMS": "src/bot_definitions/gem_ledger_pricing.py",
    "GL_CFG_ENDGAME_FLOOR": "src/bot_definitions/gem_ledger_pricing.py",
    "GL_CFG_MAX_COIN": "src/bot_definitions/gem_ledger_pricing.py",
    "GL_CFG_RIVAL_CONTEST": "src/bot_definitions/gem_ledger_pricing.py",
    "GL_CFG_MISSION_DIVISOR": "src/bot_definitions/gem_ledger_pricing.py",
    "GL_CFG_OPENING_RATIOS": "src/bot_definitions/gem_ledger_book.py",
    "GL_CFG_OPENING_WEIGHT": "src/bot_definitions/gem_ledger_book.py",
    "GL_CFG_OPENING_DROP": "src/bot_definitions/gem_ledger_book.py",
    "GL_CFG_MAX_RECORDS": "src/bot_definitions/gem_ledger_book.py",
    "GL_CFG_TABLE_WEIGHT": "src/bot_definitions/gem_ledger_book.py",
    "GL_CFG_SEAT_WEIGHT": "src/bot_definitions/gem_ledger_book.py",
}

CLASSES: dict = {}
SECONDS_PER_GAME = 0.55


def prepare_work_src() -> None:
    """Copy this project and download the other submitted bots beside it."""
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    shutil.copytree(
        REPO_SRC,
        WORK_SRC,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    destination = WORK_SRC / "bot_definitions"
    for sha, name in FIELD_FILES:
        url = (
            "https://raw.githubusercontent.com/Ryfernandes/megagem-harness/"
            f"{sha}/src/bot_definitions/{name}"
        )
        print(f"Downloading {name}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "gem-ledger-tune"})
        with urllib.request.urlopen(request, timeout=60) as response:
            (destination / name).write_bytes(response.read())


def ensure_work_src() -> None:
    """Copy the code into a temp folder and put that folder on the import path.

    New bot processes inherit this environment. Windows cannot replace the
    running program with os.execv when the folder path contains spaces, so
    the path is inserted here instead of restarting.
    """
    if os.environ.get("GL_TUNE_SRC") != str(WORK_SRC):
        prepare_work_src()
        os.environ["GL_TUNE_SRC"] = str(WORK_SRC)
        previous = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = (
            str(WORK_SRC) if previous == "" else str(WORK_SRC) + os.pathsep + previous
        )
    if str(WORK_SRC) not in sys.path:
        sys.path.insert(0, str(WORK_SRC))


def load_classes() -> None:
    """Import every bot after the temporary copy is on the path."""
    from bot_definitions.adaptive_bot import AdaptiveBot
    from bot_definitions.adaptive_finance_bots import (
        ConservativeAdaptiveBot,
        FinanceAwareAdaptiveBot,
        LoanAwareAdaptiveBot,
    )
    from bot_definitions.aggressive_bot import AggressiveBot
    from bot_definitions.valuation_bot import ValuationBot
    from bot_definitions.value_liquidity_bot import ValueLiquidityBot
    from bot_definitions.yale_bot import YaleBot

    from bot_definitions.gem_ledger_bot import GemLedgerBot

    CLASSES.update(
        {
            "GemLedgerBot": GemLedgerBot,
            "YaleBot": YaleBot,
            "AdaptiveBot": AdaptiveBot,
            "AggressiveBot": AggressiveBot,
            "ValueLiquidityBot": ValueLiquidityBot,
            "ConservativeAdaptiveBot": ConservativeAdaptiveBot,
            "LoanAwareAdaptiveBot": LoanAwareAdaptiveBot,
            "FinanceAwareAdaptiveBot": FinanceAwareAdaptiveBot,
            "ValuationBot": ValuationBot,
        }
    )


def opponent_blocks(other_seats: int) -> list[tuple[str, ...]]:
    """Rotate opponents so each one sits the same number of times."""
    offsets = [0, 1, 3, 6][:other_seats]
    count = len(FIELD)
    return [
        tuple(FIELD[(start + offset) % count] for offset in offsets)
        for start in range(count)
    ]


def build_jobs(deals: int, seed_offset: int) -> list[dict]:
    """Same deals for every candidate, covering 3, 4, and 5 players and charts A-E."""
    from decks import VALUE_CHARTS

    jobs: list[dict] = []
    for players in (3, 4, 5):
        for block_index, block in enumerate(opponent_blocks(players - 1)):
            labels = ["GemLedgerBot", *block]
            for chart_index, chart in enumerate(VALUE_CHARTS):
                for deal in range(deals):
                    for rotation in range(players):
                        seated = labels[rotation:] + labels[:rotation]
                        jobs.append(
                            {
                                "labels": seated,
                                "chart_index": chart_index,
                                "chart": chart.name,
                                "players": players,
                                "seed": seed_offset
                                + players * 1_000_000
                                + block_index * 10_000
                                + chart_index * 100
                                + deal * 10
                                + rotation,
                            }
                        )
    return jobs


def play_one(payload: dict) -> dict:
    """Play one game. The environment is set before the bots are started."""
    import logging
    from random import Random

    # Spawned workers start empty. Load the bots inside this process.
    src = os.environ.get("GL_TUNE_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)
    if not CLASSES:
        load_classes()

    from decks import VALUE_CHARTS
    from harness import play_game

    for key in list(os.environ):
        if key.startswith("GL_CFG_"):
            del os.environ[key]
    os.environ.update(payload["env"])

    fallbacks: Counter[str] = Counter()

    class _Count(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if "(" in message and ")" in message:
                name = message.split("(", 1)[1].split(")", 1)[0]
            else:
                name = "unknown"
            fallbacks[name] += 1

    handler = _Count()
    logger = logging.getLogger("bot_runner")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        result = play_game(
            payload["players"],
            value_chart=VALUE_CHARTS[payload["chart_index"]],
            rng=Random(payload["seed"]),
            bots=[CLASSES[label] for label in payload["labels"]],
            bot_timeout=1.0,
            bot_seed=payload["seed"] + 1_000_003,
        )
    finally:
        logger.removeHandler(handler)
    seat = payload["labels"].index("GemLedgerBot")
    return {
        "score": int(result.scores[seat]),
        "win": int(seat in result.winner_ids),
        "players": payload["players"],
        "chart": payload["chart"],
        "fallbacks": int(sum(fallbacks.values())),
    }


def run_games(jobs: list[dict], env: dict[str, str], workers: int) -> list[dict]:
    """Play every job and return the rows in the same order."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    payloads = []
    for job in jobs:
        payload = dict(job)
        payload["env"] = env
        payloads.append(payload)
    # "spawn" is the only start method Windows provides. Linux can use it too.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(play_one, payloads, chunksize=4))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def paired_report(rows: list[dict], baseline: list[dict]) -> dict:
    """Compare one candidate with the baseline on the same deals."""
    deltas = [row["score"] - base["score"] for row, base in zip(rows, baseline)]
    by_players: dict[str, list[float]] = {}
    by_chart: dict[str, list[float]] = {}
    wins = 0
    base_wins = 0
    for row, base, delta in zip(rows, baseline, deltas):
        by_players.setdefault(str(row["players"]), []).append(delta)
        by_chart.setdefault(row["chart"], []).append(delta)
        wins += row["win"]
        base_wins += base["win"]
    return {
        "mean_score": round(mean([row["score"] for row in rows]), 3),
        "baseline_mean": round(mean([row["score"] for row in baseline]), 3),
        "delta": round(mean(deltas), 3),
        "wins": wins,
        "baseline_wins": base_wins,
        "fallbacks": sum(row["fallbacks"] for row in rows),
        "by_players": {
            key: round(mean(values), 3) for key, values in sorted(by_players.items())
        },
        "by_chart": {
            key: round(mean(values), 3) for key, values in sorted(by_chart.items())
        },
    }


def variant_id(overrides: dict[str, str]) -> str:
    if not overrides:
        return "baseline"
    return ",".join(f"{key}={overrides[key]}" for key in sorted(overrides))


def save_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def single_knob_candidates(base_env: dict[str, str]) -> list[dict]:
    candidates = []
    for key, values, field_name in KNOBS:
        for value in values:
            if value == base_env[key]:
                continue
            overrides = {key: value}
            candidates.append(
                {
                    "id": variant_id(overrides),
                    "overrides": overrides,
                    "field": field_name,
                    "file": SETTING_FILE[key],
                }
            )
    return candidates


def moved_knobs(phase1: list[dict]) -> list[dict]:
    """Keep knobs whose best one-at-a-time change gained at least 0.25 points."""
    grouped: dict[str, list[dict]] = {}
    for row in phase1:
        overrides = row["overrides"]
        if len(overrides) != 1:
            continue
        key = next(iter(overrides))
        grouped.setdefault(key, []).append(row)
    chosen = []
    for key, rows in grouped.items():
        best = max(rows, key=lambda item: item["report"]["delta"])
        if best["report"]["delta"] >= 0.25 and best["report"]["fallbacks"] == 0:
            chosen.append(best)
    chosen.sort(key=lambda item: -item["report"]["delta"])
    return chosen[:4]


def combination_candidates(chosen: list[dict]) -> list[dict]:
    """Try every mix of the knobs that helped on their own."""
    mixes = []
    knobs = []
    for row in chosen:
        key = next(iter(row["overrides"]))
        knobs.append((key, row["overrides"][key]))
    for size in range(2, len(knobs) + 1):
        for combo in itertools.combinations(knobs, size):
            overrides = dict(combo)
            mixes.append({"id": variant_id(overrides), "overrides": overrides})
    return mixes


def describe(overrides: dict[str, str], base_env: dict[str, str]) -> str:
    if not overrides:
        return "current numbers"
    parts = []
    fields = {key: field for key, _values, field in KNOBS}
    for key in sorted(overrides):
        parts.append(f"{fields[key]} {base_env[key]} -> {overrides[key]}")
    return ", ".join(parts)


def print_recommendation(
    overrides: dict[str, str], base_env: dict[str, str], report: dict
) -> None:
    print("\n========== Result ==========", flush=True)
    if not overrides or report["delta"] <= 0.05:
        print(
            "Keep today's GemLedgerBot numbers. On the fresh deals, no tested "
            "change scored more than a twentieth of a point per game.",
            flush=True,
        )
        return
    print(
        f"On fresh deals this scores {report['delta']:+.2f} points per game "
        f"versus today's GemLedgerBot "
        f"({report['mean_score']:.2f} vs {report['baseline_mean']:.2f}).",
        flush=True,
    )
    print(
        f"Win totals on those deals: {report['wins']} vs {report['baseline_wins']} "
        f"for today's bot.",
        flush=True,
    )
    print("By table size (points per game versus today):", flush=True)
    for players, delta in report["by_players"].items():
        print(f"  {players} players: {delta:+.2f}", flush=True)
    print("By value chart:", flush=True)
    for chart, delta in report["by_chart"].items():
        print(f"  chart {chart}: {delta:+.2f}", flush=True)
    print("\nChange these lines, and leave every other number as it is:", flush=True)
    fields = {key: field for key, _values, field in KNOBS}
    for key in sorted(overrides):
        print(
            f"  {SETTING_FILE[key]}   {fields[key]} = {overrides[key]}",
            flush=True,
        )


def games_fit(deadline: float, reserve: float, game_count: int, workers: int) -> bool:
    estimate = game_count / max(workers, 1) * SECONDS_PER_GAME * 1.3
    return time() + estimate < deadline - reserve


def evaluate_candidate(
    candidate: dict,
    jobs: list[dict],
    baseline_rows: list[dict],
    base_env: dict[str, str],
    workers: int,
) -> dict:
    global SECONDS_PER_GAME
    env = dict(base_env)
    env.update(candidate["overrides"])
    started = perf_counter()
    rows = run_games(jobs, env, workers)
    elapsed = perf_counter() - started
    SECONDS_PER_GAME = max(0.05, elapsed / max(len(jobs), 1))
    report = paired_report(rows, baseline_rows)
    print(
        f"  {describe(candidate['overrides'], base_env)}: "
        f"{report['delta']:+.2f} points/game, "
        f"fallbacks {report['fallbacks']}, {elapsed:.0f}s",
        flush=True,
    )
    stored = dict(candidate)
    stored["report"] = report
    stored["seconds"] = round(elapsed, 1)
    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune GemLedgerBot's numbers.")
    parser.add_argument(
        "--hours", type=float, default=2.0, help="Stop after this many hours."
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Games to play at the same time."
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Ignore a saved results file."
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Play a few games to check the setup."
    )
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    atexit.register(lambda: shutil.rmtree(WORK_ROOT, ignore_errors=True))
    load_classes()
    from bot_definitions.gem_ledger_bot import default_settings_environment

    workers = args.workers or min(10, os.cpu_count() or 4)
    base_env = default_settings_environment()
    deals = 1 if args.smoke else 2
    jobs = build_jobs(deals, seed_offset=0)
    confirm_jobs = build_jobs(deals, seed_offset=50_000_000)
    if args.smoke:
        jobs = jobs[:8]
        confirm_jobs = confirm_jobs[:8]
    signature = {
        "games": len(jobs),
        "confirm_games": len(confirm_jobs),
        "first_seed": jobs[0]["seed"],
        "last_seed": jobs[-1]["seed"],
    }

    if args.fresh and args.results.exists():
        args.results.unlink()
    saved: dict = {}
    if args.results.exists() and not args.smoke:
        saved = json.loads(args.results.read_text())
        if saved.get("signature") != signature:
            print(
                "Saved results used a different game list, so this run starts over.",
                flush=True,
            )
            saved = {}

    payload = {
        "signature": signature,
        "baseline_rows": saved.get("baseline_rows"),
        "phase1": list(saved.get("phase1", [])),
        "phase2": list(saved.get("phase2", [])),
        "confirmation": saved.get("confirmation"),
    }
    deadline = time() + args.hours * 3600
    reserve = 0.0 if args.smoke else 15 * 60

    print(
        f"Playing {len(jobs)} games per version, {workers} games at a time.",
        flush=True,
    )
    if payload["baseline_rows"] is None:
        print("Measuring today's GemLedgerBot...", flush=True)
        started = perf_counter()
        payload["baseline_rows"] = run_games(jobs, base_env, workers)
        elapsed = perf_counter() - started
        global SECONDS_PER_GAME
        SECONDS_PER_GAME = max(0.05, elapsed / len(jobs))
        base_mean = mean([row["score"] for row in payload["baseline_rows"]])
        print(f"  today's average score: {base_mean:.2f} ({elapsed:.0f}s)", flush=True)
        save_results(args.results, payload)

    done_ids = {row["id"] for row in payload["phase1"]}
    candidates = single_knob_candidates(base_env)
    if args.smoke:
        candidates = candidates[:1]
    noun = "change" if len(candidates) == 1 else "changes"
    print(f"\nPhase 1: try {len(candidates)} single {noun}.", flush=True)
    for index, candidate in enumerate(candidates, start=1):
        if candidate["id"] in done_ids:
            continue
        if not args.smoke and not games_fit(deadline, reserve, len(jobs), workers):
            print("Time is short, so the one-at-a-time search stops here.", flush=True)
            break
        print(f"[{index}/{len(candidates)}]", flush=True)
        stored = evaluate_candidate(
            candidate, jobs, payload["baseline_rows"], base_env, workers
        )
        payload["phase1"].append(stored)
        save_results(args.results, payload)

    chosen = moved_knobs(payload["phase1"])
    mixes = [] if args.smoke else combination_candidates(chosen)
    done_mixes = {row["id"] for row in payload["phase2"]}
    if mixes:
        print(
            f"\nPhase 2: try {len(mixes)} combinations of the helpful changes.",
            flush=True,
        )
    for index, candidate in enumerate(mixes, start=1):
        if candidate["id"] in done_mixes:
            continue
        if not games_fit(deadline, reserve, len(jobs), workers):
            print("Time is short, so the combination search stops here.", flush=True)
            break
        print(f"[mix {index}/{len(mixes)}]", flush=True)
        stored = evaluate_candidate(
            candidate, jobs, payload["baseline_rows"], base_env, workers
        )
        payload["phase2"].append(stored)
        save_results(args.results, payload)

    finished = [*payload["phase1"], *payload["phase2"]]
    finished = [row for row in finished if row["report"]["fallbacks"] == 0]
    best = max(finished, key=lambda row: row["report"]["delta"], default=None)
    best_overrides: dict[str, str] = {}
    if best is not None and best["report"]["delta"] > 0:
        best_overrides = best["overrides"]

    print("\nPhase 3: replay the best change on new deals.", flush=True)
    print("Measuring today's bot on the new deals...", flush=True)
    confirm_baseline = run_games(confirm_jobs, base_env, workers)
    confirm_env = dict(base_env)
    confirm_env.update(best_overrides)
    print(
        f"Measuring {describe(best_overrides, base_env)} on the same new deals...",
        flush=True,
    )
    confirm_rows = run_games(confirm_jobs, confirm_env, workers)
    confirmation = paired_report(confirm_rows, confirm_baseline)
    confirmation["overrides"] = best_overrides
    payload["confirmation"] = confirmation
    save_results(args.results, payload)
    ranked = sorted(finished, key=lambda row: -row["report"]["delta"])
    print("\nClosest changes on the search deals (not the fresh check):", flush=True)
    if not ranked:
        print("  none finished", flush=True)
    for row in ranked[:8]:
        print(
            f"  {row['report']['delta']:+.2f}  {describe(row['overrides'], base_env)}",
            flush=True,
        )
    print_recommendation(best_overrides, base_env, confirmation)
    print(f"\nFull numbers saved to {args.results}", flush=True)


if __name__ == "__main__":
    ensure_work_src()
    main()
