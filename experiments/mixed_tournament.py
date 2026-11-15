"""Time-bounded, matched mixed-opponent tournament with per-game checkpoints."""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime
from itertools import permutations
from multiprocessing import get_context
from pathlib import Path
from random import Random
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot_runner import BotRunner  # noqa: E402
from bots import Bot  # noqa: E402
from decks import VALUE_CHARTS  # noqa: E402
from experiments.mixed_policies import CANDIDATES, POLICIES, SCENARIOS  # noqa: E402
from harness import play_game  # noqa: E402


class BudgetStop(Exception):
    """Stop before starting a callback that could exceed the experiment deadline."""


class BudgetRunner(BotRunner):
    def __init__(self, cls: type[Bot], seed: int, deadline: float):
        super().__init__(cls, timeout=2.0, seed=seed)
        self.deadline = deadline
        self.count = 0
        self.max_seconds = 0.0

    def call(self, method: str, *args: object) -> int | None:
        if time.time() >= self.deadline - 3:
            raise BudgetStop()
        if self._channel is None:
            self._start()
        started = time.perf_counter()
        answer = super().call(method, *args)
        elapsed = time.perf_counter() - started
        self.count += 1
        self.max_seconds = max(self.max_seconds, elapsed)
        if self.last_error or elapsed >= 2.0:
            raise RuntimeError(
                f"Invalid game: {self.label}: {self.last_error}; {elapsed}s"
            )
        return answer


def run_batch(task: dict[str, Any]) -> dict[str, Any]:
    candidate = task["candidate"]
    scenario = task["scenario"]
    seats = task["seats"]
    chart_index = task["chart"]
    scenario_index = tuple(SCENARIOS).index(scenario)
    choices = tuple(permutations(SCENARIOS[scenario]))
    records = []
    with ExitStack() as stack:
        # The candidate retains one runner; each opponent identity has its own.
        # Reorder seats between games without transferring private policy state.
        opponent_names = SCENARIOS[scenario]
        role_names = (candidate, *opponent_names)
        runners = [
            cast(
                BudgetRunner,
                stack.enter_context(
                    BudgetRunner(
                        POLICIES[name],
                        task["seed_base"] * 10
                        + scenario_index * 1000
                        + chart_index * 10
                        + i,
                        task["deadline"],
                    )
                ),
            )
            for i, name in enumerate(role_names)
        ]
        try:
            for rep in range(task["reps"]):
                if rep in task.get("done_reps", ()):
                    continue
                if time.time() >= task["deadline"] - 3:
                    raise BudgetStop()
                order_names = choices[rep % len(choices)]
                # Repeated styles are separate players. Resolve a permutation
                # by consuming original role slots, then cycle for five seats.
                unused = list(range(1, 4))
                order = []
                for name in order_names:
                    index = next(i for i in unused if role_names[i] == name)
                    unused.remove(index)
                    order.append(index)
                chosen_roles = [order[i % 3] for i in range(seats - 1)]
                # A fifth seat needs a distinct worker, not the same policy
                # instance twice in one game.
                lineup = []
                used = set()
                for i in chosen_roles:
                    if i in used:
                        lineup.append(
                            cast(
                                BudgetRunner,
                                stack.enter_context(
                                    BudgetRunner(
                                        POLICIES[role_names[i]],
                                        task["seed_base"] * 10 + 100 + i,
                                        task["deadline"],
                                    )
                                ),
                            )
                        )
                    else:
                        lineup.append(runners[i])
                        used.add(i)
                seat = (rep + chart_index + scenario_index) % seats
                lineup.insert(seat, runners[0])
                names = [runner.bot_class.__name__ for runner in lineup]
                for runner in lineup:
                    runner.count = 0
                    runner.max_seconds = 0.0
                seed = (
                    task["seed_base"] + scenario_index * 10000 + chart_index * 100 + rep
                )
                started = time.perf_counter()
                result = play_game(
                    seats,
                    bots=lineup,
                    rng=Random(seed),
                    value_chart=VALUE_CHARTS[chart_index],
                )
                others = [score for i, score in enumerate(result.scores) if i != seat]
                record = {
                    "stage": task["stage"],
                    "candidate": candidate,
                    "scenario": scenario,
                    "chart": VALUE_CHARTS[chart_index].name,
                    "seats": seats,
                    "rep": rep,
                    "seed": seed,
                    "seat": seat,
                    "lineup": names,
                    "score": result.scores[seat],
                    "scores": result.scores,
                    "margin": result.scores[seat] - max(others),
                    "win_credit": 1 / len(result.winner_ids)
                    if seat in result.winner_ids
                    else 0,
                    "elapsed_seconds": time.perf_counter() - started,
                    "max_callback_seconds": max(r.max_seconds for r in lineup),
                    "callbacks": sum(r.count for r in lineup),
                }
                with open(task["records_path"], "a") as stream:
                    stream.write(json.dumps(record) + "\n")
                records.append(record)
        except BudgetStop:
            pass  # An incomplete game is never scored or appended.
    return {"records": records, "task": task}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="screen")
    parser.add_argument(
        "--candidates", nargs="+", choices=CANDIDATES, default=CANDIDATES
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=tuple(SCENARIOS), default=tuple(SCENARIOS)
    )
    parser.add_argument("--charts", default="ABCDE")
    parser.add_argument("--seats", nargs="+", type=int, choices=(3, 4, 5), default=[4])
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=41000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--deadline", required=True, help="Absolute ISO UTC deadline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.reps < 1
        or args.jobs < 1
        or not args.charts
        or any(c not in "ABCDE" for c in args.charts)
    ):
        parser.error("Invalid reps, jobs or charts")
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records_path = args.output.resolve().with_suffix(".games.jsonl")
    prior = []
    if args.resume and records_path.exists():
        prior = [json.loads(line) for line in records_path.read_text().splitlines()]
    else:
        records_path.write_text("")
    tasks = [
        dict(
            stage=args.stage,
            candidate=c,
            scenario=s,
            chart="ABCDE".index(chart),
            seats=n,
            reps=args.reps,
            seed_base=args.seed_base,
            deadline=deadline,
            records_path=str(records_path),
        )
        for n in args.seats
        for s in args.scenarios
        for chart in args.charts
        for c in args.candidates
    ]
    Random(901).shuffle(tasks)
    for task in tasks:
        task["done_reps"] = [
            r["rep"]
            for r in prior
            if r["candidate"] == task["candidate"]
            and r["scenario"] == task["scenario"]
            and r["chart"] == "ABCDE"[task["chart"]]
            and r["seats"] == task["seats"]
        ]
    expected = len(tasks) * args.reps
    tasks = [t for t in tasks if len(t["done_reps"]) < args.reps]
    started = time.time()
    complete = len(prior)
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=get_context("spawn")
    ) as pool:
        futures = [pool.submit(run_batch, task) for task in tasks]
        for future in as_completed(futures):
            batch = future.result()
            complete += len(batch["records"])
            if complete % 40 == 0 or not batch["records"]:
                print(
                    f"{args.stage}: {complete} complete games; "
                    f"{time.time() - started:.1f}s",
                    flush=True,
                )
    report = {
        "stage": args.stage,
        "candidates": args.candidates,
        "scenarios": args.scenarios,
        "charts": args.charts,
        "seats": args.seats,
        "reps": args.reps,
        "seed_base": args.seed_base,
        "jobs": args.jobs,
        "deadline": args.deadline,
        "expected_games": expected,
        "complete_games": complete,
        "elapsed_seconds": time.time() - started,
        "callback_errors": [],
        "records_file": records_path.name,
        "complete_matrix": complete == expected,
        "resumed_records": len(prior),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
