"""Build a reviewable Markdown report from the frozen mixed-tournament results."""

# Long strings are Markdown paragraphs, kept intact for reviewing report copy.
# ruff: noqa: E501

import json
from pathlib import Path

from experiments.summarize_mixed import groups, load, summary

ROOT = Path("experiments/results/mixed_tournament")


def main() -> None:
    data = json.loads((ROOT / "summary.json").read_text())
    selection = data["selection"]
    gen = selection["generalist"]
    screens = load(ROOT / "screen.games.jsonl")
    validation = load(ROOT / "validation.games.jsonl")
    robust = load(ROOT / "robustness.games.jsonl")
    lines = [
        "# Mixed-opponent tournament: nominations and limits",
        "",
        "All policies are Python. Tests use real isolated workers with two-second callbacks; L4 keeps its production 0.85-second cutoff and level-2 fallback. The default, SmallBidBot and FixedRuleBot are excluded from every tournament lineup.",
        "",
        "## Nominations",
        "",
        f"**Generalist: {gen}.** Selected by the best worst opponent/chart aggregate win rate, then overall win rate and mean margin. This is a robustness choice, not a claim of universal dominance or necessarily the largest average win total.",
        "",
    ]
    adopted = []
    for i, s in enumerate(selection["specialists"]):
        comp = data["comparisons"].get("validation", {}).get("specialists", [])
        comp = comp[i] if len(comp) > i else {}
        base = data["comparisons"].get("validation", {}).get("specialists_vs_L2", [])
        base = base[i] if len(base) > i else {}
        name = s["candidate"]
        replacement = False
        target = [
            r
            for r in validation
            if r["candidate"] == name and r[s["field"]] == s["value"]
        ]
        control = {
            tuple(r[k] for k in ("scenario", "chart", "seats", "rep", "seed")): r
            for r in validation
            if r["candidate"] == "L2" and r[s["field"]] == s["value"]
        }
        matched = [
            (
                r,
                control[
                    tuple(r[k] for k in ("scenario", "chart", "seats", "rep", "seed"))
                ],
            )
            for r in target
            if tuple(r[k] for k in ("scenario", "chart", "seats", "rep", "seed"))
            in control
        ]
        if matched and all(a["win_credit"] == b["win_credit"] for a, b in matched):
            delta = sum(a["score"] - b["score"] for a, b in matched) / len(matched)
            if abs(delta) < 1:
                replacement = True
                name = "L2"
        adopted.append(
            dict(
                candidate=name,
                frozen_candidate=s["candidate"],
                field=s["field"],
                value=s["value"],
                simplified_to_L2=replacement,
            )
        )
        lines += [
            f"**Specialist {i + 1}: {name}, targeting {s['field']} `{s['value']}`.** The frozen screening finalist was {s['candidate']}."
        ]
        if replacement:
            lines += [
                "It matched L2's win credit on every target validation deal and differed by less than one mean point. Prefer L2 under the predeclared simplicity rule; the inference/reveal flags did not establish added value here."
            ]
        if comp.get("pairs"):
            a, b = comp["candidate_summary"], comp["reference_summary"]
            d = comp["win_credit"]
            m = comp["margin"]
            lines += [
                f"Fresh target check ({comp['pairs']} paired games): frozen finalist {a['wins']:.2f} win credit versus {b['wins']:.2f} for {gen}; win-rate difference {100 * d['delta']:+.1f} percentage points [{100 * d['ci_low']:+.1f}, {100 * d['ci_high']:+.1f}]. Margin difference {m['delta']:+.2f} [{m['ci_low']:+.2f}, {m['ci_high']:+.2f}]."
            ]
            lines += [
                "The paired win-credit interval is positive in this subset."
                if d["ci_low"] > 0
                else "The paired win-credit interval includes zero or favors the generalist: treat this specialty nomination as provisional, not a confirmed advantage."
            ]
        lines += [""]
    lines += [
        "These are recommendations for the tested synthetic lineups. No finite sweep proves a bot dominates every possible opponent. Small joint cells and selection across many configurations require caution. The nominees were frozen before validation; a failed specialty check is reported rather than replaced with a new validation winner.",
        "",
        "## Bots and opponent hypotheses",
        "",
        "| Token | Policy |",
        "| --- | --- |",
        "| L2 | Current ValuationBot; also L3 with every extra off |",
        "| I | Hidden-hand inference, evidence weight 0.1 |",
        "| F | Finance-aware horizon / opportunity cost |",
        "| R | Strategic reveal policy |",
        "| L3-I/F/R/IF/IR/FR/IFR | Every nonempty combination of those three features |",
        "| L4 | Current uniform-prior sampling bot, with L2 continuation policies |",
        "",
        "The mission-only endgame rule and cash-fraction predictor are omitted because earlier four-seat ablations showed clear harm. Legal cash limits and expected-utility bidding remain standard L2 behavior. L3-all-off is identical to L2 and is not counted twice.",
        "",
        "| Hypothetical opponent | Treasure spending fraction | Reserve relative to L2 |",
        "| --- | ---: | ---: |",
        "| RandomBidder | Uniform random fraction 0–1.4 each auction | 0.5× |",
        "| AggressiveBidder | 1.10 | 0.25× |",
        "| PassiveBidder | 0.12 | 1.5× |",
        "",
        "Fractions multiply chart/mission-aware value discounted for liquidity. Investments price locked liquidity; loans price productive funding and retain positive net proceeds. Normal affordability and rival-budget caps apply. These hypothetical styles are not fitted human behaviors, and random means value-bounded random spending rather than random over the entire bankroll. Reveals use the first-card rule. All three styles are available in the browser selector.",
        "",
        "## Lineup table",
        "",
        "| Scenario | Other three seats in the four-player screen |",
        "| --- | --- |",
        "| level2 | L2, L2, L2 |",
        "| level3 | L3-I, L3-F, L3-R |",
        "| random | Random, Random, Random |",
        "| aggressive | Aggressive, Aggressive, Aggressive |",
        "| passive | Passive, Passive, Passive |",
        "| mixed_styles | Random, Aggressive, Passive |",
        "| mixed_experts | L2, L3-IFR, L4 |",
        "",
        "The candidate is a separate fourth seat, including when it has the same type as an opponent. Candidate seats and opponent order rotate identically across variants. All five charts are crossed independently with the lineups. This representative matrix is not every multiset of every possible bot.",
        "",
        "## Screening: complete balanced matrix",
        "",
        "Nine unique candidates × seven lineups × five charts × four deals = 1,260 games, all at four seats. The first time slice checkpointed 1,179 complete games; a time-driven resume filled the remaining 81 without changing candidates or selecting from outcomes. Resumed worker RNG streams can restart; wall-clock L4 decisions also vary with load. Only complete games count.",
        "",
        "The following entries are single-candidate win-credit percentages. Each opponent aggregate has 20 games per candidate; each chart aggregate has 28. Tied winners split credit. Opponent wins are never summed and compared with one candidate.",
        "",
    ]
    for field, values, title in [
        (
            "scenario",
            [
                "level2",
                "level3",
                "random",
                "aggressive",
                "passive",
                "mixed_styles",
                "mixed_experts",
            ],
            "By opponent lineup",
        ),
        ("chart", list("ABCDE"), "By chart"),
    ]:
        lines += [
            f"### {title}",
            "",
            "| Candidate | " + " | ".join(values) + " | Overall |",
            "| --- | " + " | ".join(["---:"] * (len(values) + 1)) + " |",
        ]
        for rank in selection["generalist_ranking"]:
            name = rank["candidate"]
            r = [x for x in screens if x["candidate"] == name]
            cells = [
                f"{100 * summary([x for x in r if x[field] == v])['win_rate']:.1f}%"
                for v in values
            ]
            lines += [
                "| "
                + name
                + " | "
                + " | ".join(cells)
                + f" | {100 * summary(r)['win_rate']:.1f}% |"
            ]
        lines += [""]
    lines += [
        "Screening is exploratory. Ranking by average wins can differ from ranking by the worst opponent/chart group; the generalist rule deliberately emphasizes the latter. Do not interpret four deals per joint cell as a reliable dominance test.",
        "",
        "## Fresh validation",
        "",
    ]
    if validation:
        lines += [
            f"{len(validation)} completed validation games. Each nominee and L2 control uses fresh matched seeds, independently from screening. See the stage metadata for planned versus completed matrix counts.",
            "",
            "| Candidate | Games | Win credit | Win rate | Mean points | Mean margin |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, r in groups(validation, "candidate").items():
            s = summary(r)
            lines += [
                f"| {name} | {s['games']} | {s['wins']:.2f} | {100 * s['win_rate']:.1f}% | {s['score']:.2f} | {s['margin']:.2f} |"
            ]
        c = data["comparisons"]["validation"]["generalist_vs_L2"]
        if c.get("pairs"):
            d = c["win_credit"]
            m = c["margin"]
            lines += [
                "",
                f"{gen} versus L2 across {c['pairs']} matched validation games: win-rate change {100 * d['delta']:+.1f} pp [{100 * d['ci_low']:+.1f}, {100 * d['ci_high']:+.1f}], margin change {m['delta']:+.2f} [{m['ci_low']:+.2f}, {m['ci_high']:+.2f}].",
            ]
        lines += [
            "",
            "Validation opponent/chart aggregates and all paired specialty comparisons (including L2 controls) are in summary.json. Intervals are approximate paired 95% mean ±1.96 SE across matched deal cases; small target subsets are exploratory and intervals are not adjusted for multiple comparisons.",
        ]
    if validation:
        scenarios = [
            "level2",
            "level3",
            "random",
            "aggressive",
            "passive",
            "mixed_styles",
            "mixed_experts",
        ]
        lines += [
            "",
            "### Validation by opponent lineup",
            "",
            "Each entry is win credit as a percentage of ten fresh games.",
            "",
            "| Candidate | " + " | ".join(scenarios) + " |",
            "| --- | " + " | ".join(["---:"] * len(scenarios)) + " |",
        ]
        for name, values in data["comparisons"]["validation"]["by_opponent"].items():
            cells = [f"{100 * values[s]['win_rate']:.1f}%" for s in scenarios]
            lines += ["| " + name + " | " + " | ".join(cells) + " |"]
    if robust:
        lines += [
            "",
            "## Additional seat-count checks",
            "",
            f"{len(robust)} fresh games at three/five seats, targeting aggressive and mixed-expert lineups across all charts. Opponent roles cycle when the seat count changes; each repeated role gets a distinct worker. These sparse checks are separate from the main four-seat ranking.",
            "",
            "| Candidate | Seats | Games | Win credit | Win rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, r in groups(robust, "candidate").items():
            for n, g in groups(r, "seats").items():
                s = summary(g)
                lines += [
                    f"| {name} | {n} | {s['games']} | {s['wins']:.2f} | {100 * s['win_rate']:.1f}% |"
                ]
    all_rows = screens + validation + robust
    lines += [
        "",
        "## Timing, reproducibility and limits",
        "",
        f"Total completed benchmark games: **{len(all_rows)}**. Logged callbacks: {sum(r['callbacks'] for r in all_rows):,}. Maximum parent-side callback time, excluding startup: **{max(r['max_callback_seconds'] for r in all_rows):.3f} seconds**. Completed rows contain no callback errors or hard timeouts; an error aborts its batch instead of silently accepting fallback games.",
        "",
        "L4 still assumes L2 continuation policies for every simulated opponent. This experiment tests that existing assumption against other styles; it does not teach the sampler their true policies. Its earlier strong results against L2-only tables therefore need not carry over. This is a model limitation, not evidence that hypothetical opponents match human play.",
        "",
        "The session began at 19:50:05 UTC on 2026-11-15. Experiment deadlines stop starting callbacks before the cutoff and discard unfinished games. Testing ends by 20:08:00 UTC, leaving the rest of the twenty-minute budget for analysis. The protocol and phase metadata document counts/deadlines; no unbounded search was launched.",
        "",
        "Validation: 148 unit tests passed, including legal bids on every chart/auction/cash range and all seven L3 feature combinations. Ruff and strict mypy pass for the added code. Browser checks exercised the new selectors and a five-way Chart E game with L4, configured L3 and the three hypothetical opponents.",
        "",
        "## Artifacts",
        "",
        "- [Frozen selection](selection.json), [summary and paired intervals](summary.json), [protocol](protocol.md).",
        "- [Screen matrix](screen_matrix.csv), [validation matrix](validation_matrix.csv).",
        "- [Raw screen games](screen.games.jsonl), [raw validation games](validation.games.jsonl).",
        "- [Screen metadata](screen.json), [validation metadata](validation.json).",
        "- [Seat-count checks](robustness_matrix.csv), [session timing](run_timing.json).",
        "",
        "Run `npm run dev` and choose the new styles in any seat. For a reusable tournament run, use `experiments/mixed_tournament.py`; `--deadline` is an absolute future ISO UTC time. Candidate aliases are in `experiments/mixed_policies.py`. Use `--resume` to fill a fixed interrupted matrix. Run `experiments/summarize_mixed.py --select` only after a complete screen, freeze that selection, then summarize fresh validation with `experiments/summarize_mixed.py` and `python3 -m experiments.report_mixed`.",
        "",
    ]
    (ROOT / "report.md").write_text("\n".join(lines))
    (ROOT / "nominations.json").write_text(
        json.dumps(
            {
                "generalist": gen,
                "specialists": adopted,
                "frozen_selection": "selection.json",
                "universal_dominance_proven": False,
                "report": "report.md",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"generalist": gen, "specialists": adopted, "games": len(all_rows)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
