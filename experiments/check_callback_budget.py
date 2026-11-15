"""Time actual Python bot callbacks, including IPC/checkpoints, with a 2s limit.

Cover all public custom bots on every chart at every supported seat count.
Worker startup has a separate allowance and is excluded from callback timing.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import custom_bots  # noqa: E402
from bot_runner import BotRunner  # noqa: E402
from bots import Bot, BotInfo  # noqa: E402
from decks import VALUE_CHARTS  # noqa: E402
from harness import play_game  # noqa: E402


class TimedRunner(BotRunner):
    def __init__(self, bot_class: type[Bot], rows: list[dict[str, Any]], seed: int):
        super().__init__(bot_class, timeout=2.0, seed=seed)
        self.rows = rows

    def call(self, method: str, *args: object) -> int | None:
        if self._channel is None:
            self._start()
        started = perf_counter()
        result = super().call(method, *args)
        seconds = perf_counter() - started
        self.rows.append(
            {
                "bot": self.bot_class.__name__,
                "callback": method,
                "round": args[0].round_number
                if args and isinstance(args[0], BotInfo)
                else 0,
                "seconds": seconds,
                "error": self.last_error,
            }
        )
        if self.last_error or seconds >= 2.0:
            raise RuntimeError(f"Callback budget failed: {self.rows[-1]}")
        return result


def run(case: tuple[int, int, tuple[str, ...], int]) -> dict[str, Any]:
    seats, chart_index, names, seed = case
    rows: list[dict[str, Any]] = []
    with ExitStack() as stack:
        runners = [
            stack.enter_context(TimedRunner(getattr(custom_bots, name), rows, seed + i))
            for i, name in enumerate(names)
        ]
        result = play_game(
            seats, bots=runners, rng=Random(seed), value_chart=VALUE_CHARTS[chart_index]
        )
    return {
        "seats": seats,
        "chart": VALUE_CHARTS[chart_index].name,
        "bots": names,
        "seed": seed,
        "scores": result.scores,
        "callbacks": rows,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = tuple(custom_bots.__all__)
    cases: list[tuple[int, int, tuple[str, ...], int]] = []
    for seats in (3, 4, 5):
        for chart_index in range(len(VALUE_CHARTS)):
            for offset in range(0, len(names), seats):
                selected = names[offset : offset + seats]
                selected += ("FixedRuleBot",) * (seats - len(selected))
                cases.append((seats, chart_index, selected, 3900 + len(cases)))
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
        records = list(pool.map(run, cases))
    summary = {}
    for name in names:
        rows = [
            row for game in records for row in game["callbacks"] if row["bot"] == name
        ]
        bids = [row["seconds"] for row in rows if row["callback"] == "get_bid"]
        summary[name] = {
            "callbacks": len(rows),
            "bid_calls": len(bids),
            "max_callback_seconds": max(row["seconds"] for row in rows),
            "max_bid_seconds": max(bids),
            "mean_bid_seconds": sum(bids) / len(bids),
            "coverage": sorted(
                {(g["seats"], g["chart"]) for g in records if name in g["bots"]}
            ),
        }
        assert len(summary[name]["coverage"]) == 15
    report = {
        "language": "Python",
        "timeout_seconds": 2.0,
        "timing": (
            "wall time including argument transport and checkpointing, "
            "excluding worker startup"
        ),
        "games": len(records),
        "callback_errors": [],
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"games": len(records), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
