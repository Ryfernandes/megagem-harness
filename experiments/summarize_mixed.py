"""Freeze exploratory nominees, then summarize independent mixed-table validation."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "games": len(rows),
        "wins": sum(r["win_credit"] for r in rows),
        "win_rate": statistics.mean(r["win_credit"] for r in rows),
        "score": statistics.mean(r["score"] for r in rows),
        "margin": statistics.mean(r["margin"] for r in rows),
    }


def groups(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[row[field]].append(row)
    return dict(result)


def select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = groups(rows, "candidate")
    ranked = []
    for name, r in candidates.items():
        s = summary(r)
        worst = min(
            summary(g)["win_rate"]
            for f in ("scenario", "chart")
            for g in groups(r, f).values()
        )
        ranked.append((round(worst, 12), round(s["win_rate"], 12), s["margin"], name))
    general = max(ranked)[-1]
    reference = candidates[general]
    nominees = []
    excluded = {general}
    for field in ("scenario", "chart"):
        pool = []
        for name, r in candidates.items():
            if name in excluded:
                continue
            for value, g in groups(r, field).items():
                s, b = summary(g), summary([x for x in reference if x[field] == value])
                pool.append(
                    (
                        round(s["win_rate"] - b["win_rate"], 12),
                        s["margin"] - b["margin"],
                        s["win_rate"],
                        name,
                        value,
                    )
                )
        delta, margin, rate, name, value = max(pool)
        nominees.append(
            dict(
                candidate=name,
                field=field,
                value=value,
                screen_win_delta=delta,
                screen_margin_delta=margin,
                screen_win_rate=rate,
                positive_screen_win_edge=delta > 0,
            )
        )
        excluded.add(name)
    return {
        "generalist": general,
        "specialists": nominees,
        "generalist_ranking": [
            dict(candidate=n, worst_group_win_rate=w, win_rate=r, margin=m)
            for w, r, m, n in sorted(ranked, reverse=True)
        ],
        "selection_only": True,
        "validation_not_used_for_selection": True,
    }


def paired(
    rows: list[dict[str, Any]],
    candidate: str,
    reference: str,
    field: str | None = None,
    value: str | None = None,
) -> dict[str, Any]:
    def key(r: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(r[k] for k in ("scenario", "chart", "seats", "rep", "seed"))

    lookup = {
        name: {
            key(r): r
            for r in rows
            if r["candidate"] == name and (field is None or r[field] == value)
        }
        for name in (candidate, reference)
    }
    keys = sorted(set(lookup[candidate]) & set(lookup[reference]))
    result: dict[str, Any] = {
        "pairs": len(keys),
        "candidate": candidate,
        "reference": reference,
    }
    if not keys:
        return result
    for metric in ("win_credit", "score", "margin"):
        d = [lookup[candidate][k][metric] - lookup[reference][k][metric] for k in keys]
        mean = statistics.mean(d)
        se = statistics.stdev(d) / sqrt(len(d)) if len(d) > 1 else 0
        result[metric] = {
            "delta": mean,
            "ci_low": mean - 1.96 * se,
            "ci_high": mean + 1.96 * se,
        }
    result["candidate_summary"] = summary([lookup[candidate][k] for k in keys])
    result["reference_summary"] = summary([lookup[reference][k] for k in keys])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("experiments/results/mixed_tournament")
    )
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    root = args.root
    screen = load(root / "screen.games.jsonl")
    if args.select:
        meta = json.loads((root / "screen.json").read_text())
        if not meta["complete_matrix"]:
            raise RuntimeError("Complete balanced screen before selection")
        decision = select(screen)
        (root / "selection.json").write_text(json.dumps(decision, indent=2) + "\n")
        print(json.dumps(decision, indent=2))
        return
    selection = json.loads((root / "selection.json").read_text())
    result = {"selection": selection, "stages": {}, "comparisons": {}}
    for stage in ("screen", "validation", "robustness"):
        rows = load(root / f"{stage}.games.jsonl")
        if not rows:
            continue
        by_candidate = groups(rows, "candidate")
        result["stages"][stage] = {name: summary(r) for name, r in by_candidate.items()}
        matrix = []
        cells = defaultdict(list)
        for row in rows:
            cells[
                (row["candidate"], row["scenario"], row["chart"], row["seats"])
            ].append(row)
        for (candidate, scenario, chart, seats), r in sorted(cells.items()):
            matrix.append(
                dict(
                    candidate=candidate,
                    scenario=scenario,
                    chart=chart,
                    seats=seats,
                    **summary(r),
                )
            )
        with (root / f"{stage}_matrix.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(matrix[0]))
            writer.writeheader()
            writer.writerows(matrix)
        result["comparisons"][stage] = {
            "generalist_vs_L2": paired(rows, selection["generalist"], "L2"),
            "specialists": [
                paired(
                    rows,
                    s["candidate"],
                    selection["generalist"],
                    s["field"],
                    s["value"],
                )
                for s in selection["specialists"]
            ],
            "specialists_vs_L2": [
                paired(rows, s["candidate"], "L2", s["field"], s["value"])
                for s in selection["specialists"]
            ],
            "by_chart": {
                name: {v: summary(g) for v, g in groups(r, "chart").items()}
                for name, r in by_candidate.items()
            },
            "by_opponent": {
                name: {v: summary(g) for v, g in groups(r, "scenario").items()}
                for name, r in by_candidate.items()
            },
        }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
