# Mixed-opponent tournament: nominations and limits

All policies are Python. Tests use real isolated workers with two-second callbacks; L4 keeps its production 0.85-second cutoff and level-2 fallback. The default, SmallBidBot and FixedRuleBot are excluded from every tournament lineup.

## Nominations

**Generalist: L4.** Selected by the best worst opponent/chart aggregate win rate, then overall win rate and mean margin. This is a robustness choice, not a claim of universal dominance or necessarily the largest average win total.

**Specialist 1: L2, targeting scenario `passive`.** The frozen screening finalist was L3-I.
It matched L2's win credit on every target validation deal and differed by less than one mean point. Prefer L2 under the predeclared simplicity rule; the inference/reveal flags did not establish added value here.
Fresh target check (10 paired games): frozen finalist 8.00 win credit versus 4.00 for L4; win-rate difference +40.0 percentage points [+8.0, +72.0]. Margin difference +37.00 [+19.99, +54.01].
The paired win-credit interval is positive in this subset.

**Specialist 2: L3-IR, targeting chart `C`.** The frozen screening finalist was L3-IR.
Fresh target check (14 paired games): frozen finalist 7.00 win credit versus 7.00 for L4; win-rate difference +0.0 percentage points [-41.1, +41.1]. Margin difference +3.36 [-12.99, +19.70].
The paired win-credit interval includes zero or favors the generalist: treat this specialty nomination as provisional, not a confirmed advantage.

These are recommendations for the tested synthetic lineups. No finite sweep proves a bot dominates every possible opponent. Small joint cells and selection across many configurations require caution. The nominees were frozen before validation; a failed specialty check is reported rather than replaced with a new validation winner.

## Bots and opponent hypotheses

| Token | Policy |
| --- | --- |
| L2 | Current ValuationBot; also L3 with every extra off |
| I | Hidden-hand inference, evidence weight 0.1 |
| F | Finance-aware horizon / opportunity cost |
| R | Strategic reveal policy |
| L3-I/F/R/IF/IR/FR/IFR | Every nonempty combination of those three features |
| L4 | Current uniform-prior sampling bot, with L2 continuation policies |

The mission-only endgame rule and cash-fraction predictor are omitted because earlier four-seat ablations showed clear harm. Legal cash limits and expected-utility bidding remain standard L2 behavior. L3-all-off is identical to L2 and is not counted twice.

| Hypothetical opponent | Treasure spending fraction | Reserve relative to L2 |
| --- | ---: | ---: |
| RandomBidder | Uniform random fraction 0–1.4 each auction | 0.5× |
| AggressiveBidder | 1.10 | 0.25× |
| PassiveBidder | 0.12 | 1.5× |

Fractions multiply chart/mission-aware value discounted for liquidity. Investments price locked liquidity; loans price productive funding and retain positive net proceeds. Normal affordability and rival-budget caps apply. These hypothetical styles are not fitted human behaviors, and random means value-bounded random spending rather than random over the entire bankroll. Reveals use the first-card rule. All three styles are available in the browser selector.

## Lineup table

| Scenario | Other three seats in the four-player screen |
| --- | --- |
| level2 | L2, L2, L2 |
| level3 | L3-I, L3-F, L3-R |
| random | Random, Random, Random |
| aggressive | Aggressive, Aggressive, Aggressive |
| passive | Passive, Passive, Passive |
| mixed_styles | Random, Aggressive, Passive |
| mixed_experts | L2, L3-IFR, L4 |

The candidate is a separate fourth seat, including when it has the same type as an opponent. Candidate seats and opponent order rotate identically across variants. All five charts are crossed independently with the lineups. This representative matrix is not every multiset of every possible bot.

## Screening: complete balanced matrix

Nine unique candidates × seven lineups × five charts × four deals = 1,260 games, all at four seats. The first time slice checkpointed 1,179 complete games; a time-driven resume filled the remaining 81 without changing candidates or selecting from outcomes. Resumed worker RNG streams can restart; wall-clock L4 decisions also vary with load. Only complete games count.

The following entries are single-candidate win-credit percentages. Each opponent aggregate has 20 games per candidate; each chart aggregate has 28. Tied winners split credit. Opponent wins are never summed and compared with one candidate.

### By opponent lineup

| Candidate | level2 | level3 | random | aggressive | passive | mixed_styles | mixed_experts | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L4 | 85.0% | 85.0% | 27.5% | 47.5% | 30.0% | 35.0% | 35.0% | 49.3% |
| L3-IR | 25.0% | 15.0% | 65.0% | 90.0% | 75.0% | 80.0% | 20.0% | 52.9% |
| L3-F | 25.0% | 14.2% | 62.5% | 90.0% | 75.0% | 75.0% | 15.0% | 51.0% |
| L3-IF | 25.0% | 14.2% | 62.5% | 90.0% | 75.0% | 75.0% | 15.0% | 51.0% |
| L3-IFR | 20.0% | 25.0% | 65.0% | 90.0% | 75.0% | 80.0% | 10.0% | 52.1% |
| L3-FR | 20.0% | 25.0% | 65.0% | 90.0% | 75.0% | 80.0% | 5.0% | 51.4% |
| L2 | 30.0% | 12.5% | 62.5% | 90.0% | 75.0% | 75.0% | 5.0% | 50.0% |
| L3-I | 30.0% | 12.5% | 62.5% | 90.0% | 75.0% | 75.0% | 5.0% | 50.0% |
| L3-R | 25.0% | 15.0% | 65.0% | 90.0% | 75.0% | 80.0% | 0.0% | 50.0% |

### By chart

| Candidate | A | B | C | D | E | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L4 | 60.7% | 57.1% | 41.1% | 55.4% | 32.1% | 49.3% |
| L3-IR | 60.7% | 41.1% | 62.5% | 58.9% | 41.1% | 52.9% |
| L3-F | 53.6% | 46.4% | 61.9% | 50.0% | 42.9% | 51.0% |
| L3-IF | 57.1% | 46.4% | 61.9% | 50.0% | 39.3% | 51.0% |
| L3-IFR | 57.1% | 41.1% | 60.7% | 58.9% | 42.9% | 52.1% |
| L3-FR | 53.6% | 41.1% | 60.7% | 58.9% | 42.9% | 51.4% |
| L2 | 60.7% | 42.9% | 60.7% | 46.4% | 39.3% | 50.0% |
| L3-I | 57.1% | 42.9% | 60.7% | 46.4% | 42.9% | 50.0% |
| L3-R | 57.1% | 41.1% | 60.7% | 51.8% | 39.3% | 50.0% |

Screening is exploratory. Ranking by average wins can differ from ranking by the worst opponent/chart group; the generalist rule deliberately emphasizes the latter. Do not interpret four deals per joint cell as a reliable dominance test.

## Fresh validation

280 completed validation games. Each nominee and L2 control uses fresh matched seeds, independently from screening. See the stage metadata for planned versus completed matrix counts.

| Candidate | Games | Win credit | Win rate | Mean points | Mean margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| L2 | 70 | 38.00 | 54.3% | 61.57 | 4.97 |
| L3-I | 70 | 39.00 | 55.7% | 62.00 | 6.23 |
| L3-IR | 70 | 39.00 | 55.7% | 62.64 | 6.69 |
| L4 | 70 | 38.00 | 54.3% | 61.56 | 5.07 |

L4 versus L2 across 70 matched validation games: win-rate change +0.0 pp [-18.3, +18.3], margin change +0.10 [-10.99, +11.19].

Validation opponent/chart aggregates and all paired specialty comparisons (including L2 controls) are in summary.json. Intervals are approximate paired 95% mean ±1.96 SE across matched deal cases; small target subsets are exploratory and intervals are not adjusted for multiple comparisons.

### Validation by opponent lineup

Each entry is win credit as a percentage of ten fresh games.

| Candidate | level2 | level3 | random | aggressive | passive | mixed_styles | mixed_experts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 | 0.0% | 10.0% | 80.0% | 100.0% | 80.0% | 100.0% | 10.0% |
| L3-I | 0.0% | 20.0% | 80.0% | 100.0% | 80.0% | 100.0% | 10.0% |
| L3-IR | 0.0% | 20.0% | 80.0% | 100.0% | 80.0% | 100.0% | 10.0% |
| L4 | 90.0% | 100.0% | 20.0% | 50.0% | 40.0% | 50.0% | 30.0% |

## Additional seat-count checks

60 fresh games at three/five seats, targeting aggressive and mixed-expert lineups across all charts. Opponent roles cycle when the seat count changes; each repeated role gets a distinct worker. These sparse checks are separate from the main four-seat ranking.

| Candidate | Seats | Games | Win credit | Win rate |
| --- | ---: | ---: | ---: | ---: |
| L3-IR | 3 | 10 | 7.00 | 70.0% |
| L3-IR | 5 | 10 | 5.00 | 50.0% |
| L2 | 5 | 10 | 5.00 | 50.0% |
| L2 | 3 | 10 | 7.00 | 70.0% |
| L4 | 3 | 10 | 8.00 | 80.0% |
| L4 | 5 | 10 | 5.00 | 50.0% |

## Timing, reproducibility and limits

Total completed benchmark games: **1600**. Logged callbacks: 231,916. Maximum parent-side callback time, excluding startup: **0.858 seconds**. Completed rows contain no callback errors or hard timeouts; an error aborts its batch instead of silently accepting fallback games.

L4 still assumes L2 continuation policies for every simulated opponent. This experiment tests that existing assumption against other styles; it does not teach the sampler their true policies. Its earlier strong results against L2-only tables therefore need not carry over. This is a model limitation, not evidence that hypothetical opponents match human play.

The session began at 19:50:05 UTC on 2026-11-15. Experiment deadlines stop starting callbacks before the cutoff and discard unfinished games. Testing ends by 20:08:00 UTC, leaving the rest of the twenty-minute budget for analysis. The protocol and phase metadata document counts/deadlines; no unbounded search was launched.

Validation: 148 unit tests passed, including legal bids on every chart/auction/cash range and all seven L3 feature combinations. Ruff and strict mypy pass for the added code. Browser checks exercised the new selectors and a five-way Chart E game with L4, configured L3 and the three hypothetical opponents.

## Artifacts

- [Frozen selection](selection.json), [summary and paired intervals](summary.json), [protocol](protocol.md).
- [Screen matrix](screen_matrix.csv), [validation matrix](validation_matrix.csv).
- [Raw screen games](screen.games.jsonl), [raw validation games](validation.games.jsonl).
- [Screen metadata](screen.json), [validation metadata](validation.json).
- [Seat-count checks](robustness_matrix.csv), [session timing](run_timing.json).

Run `npm run dev` and choose the new styles in any seat. For a reusable tournament run, use `experiments/mixed_tournament.py`; `--deadline` is an absolute future ISO UTC time. Candidate aliases are in `experiments/mixed_policies.py`. Use `--resume` to fill a fixed interrupted matrix. Run `experiments/summarize_mixed.py --select` only after a complete screen, freeze that selection, then summarize fresh validation with `experiments/summarize_mixed.py` and `python3 -m experiments.report_mixed`.
