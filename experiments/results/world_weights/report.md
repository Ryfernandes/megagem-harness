# L4 world-weight ablation: use uniform weights

Current `SamplingBot` uses the conditional prior and equal world weights.
The level-3 tracker, likelihood updates, per-world log weights, exponentiation
and effective-sample-size bookkeeping have been removed from the public L4 bot.
The browser option remains **Sampling search · level 4**. Level-2 public-bid
learning still models continuation opponents. The separate level-3 bot and its
optional features are unchanged.

## Matched full-game comparison

Fresh seeds **3600–3659**, 60 games per policy at each of three, four and five
seats: **360 real isolated-worker games**. Each game has one L4 candidate, with
current level-2 `ValuationBot` in all other seats. Weighted and uniform runs pair
the initial-deal seed, chart and candidate seat. These win credits belong to the
single candidate, never the opponents combined. Tied winners split credit.

Both policies use identical search settings: 0.85 seconds, at most 16 worlds,
seven initial candidate bids, and at least two complete comparison worlds.
The weighted control uses evidence weight 0.1; the uniform candidate sets it to
zero and skips posterior inference. No other L3 feature is enabled in either.
The [protocol](protocol.md) specifies a three-point mean-score loss as a
practically relevant deterioration; this was set before inspecting the main
comparison's outcomes. No bid-policy parameters were tuned on these games.

| Players | Weighted win credit / 60 | Uniform win credit / 60 | Mean points, weighted / uniform | Uniform score change [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| 3 | 54.00 | 57.00 | 87.15 / 91.30 | +4.15 [+0.80, +7.50] |
| 4 | 44.33 | 45.00 | 67.15 / 66.25 | -0.90 [-2.10, +0.30] |
| 5 | 53.00 | 53.00 | 66.18 / 65.73 | -0.45 [-2.70, +1.80] |

Every lower paired score bound is above **−3 points**. Uniform has a higher mean
score at three seats and small mean declines at four/five seats; win credit is
as high or higher at every count. This supports dropping weighting under the
stated practical-loss threshold. It does not prove exact equality: the
four/five-seat intervals allow modest changes in either direction. The search
objective is mean own points, not rollout win rate.

Approximate intervals are paired per-seed means ± 1.96 standard errors and are
reported per seat count, without adjustment across counts. Charts cycle as
`ABCDE[seed % 5]`; candidate seat is `(-seed) % seats`. Ten-game batches start
fresh fixed-per-seat bot RNGs, with six parallel jobs and saved batch checkpoints.
At five seats, chart and candidate seat are coupled by this rotation. The same
seed at different seat counts is not an independent replicate to pool. Results
apply to L2 opponents, not arbitrary tournament lineups.

## Does weighting change decisions?

On every successful weighted search, also compute the unweighted optimum from
its **exact same worlds, candidates and final-score outcomes**, including the
same cheaper-bid tie preference. This diagnostic adds no rollouts.

Weighting changed **12 of 2,841 choices (0.42%)**. It
usually produces the same action as the ordinary mean. Full-game searches can
still diverge because available CPU time changes sample counts and candidate
pruning, and a changed auction can alter the entire continuation. The observed
three-seat score gain should not be attributed solely to those few aggregation
changes. Fixed setup seeds do not make wall-clock searches bit-for-bit repeatable.

## Computation

| Players | Round-end ms, weighted / uniform | Bid seconds, weighted / uniform | Mean worlds, weighted / uniform | Same-world action differences |
| --- | ---: | ---: | ---: | ---: |
| 3 | 1.64 / 0.14 | 0.575 / 0.574 | 11.07 / 11.02 | 3 / 997 |
| 4 | 9.73 / 0.16 | 0.600 / 0.600 | 8.12 / 8.17 | 7 / 892 |
| 5 | 36.44 / 0.19 | 0.629 / 0.623 | 6.40 / 6.31 | 2 / 952 |

The clear speed saving is removal of hidden-hand posterior inference at round
end. Search still uses the same 0.85-second allowance, so this is not a large
end-to-end speedup. Means include all decisions and legal zero-bid skips; world
counts include fallback diagnostics. Computation timing excludes worker
transport/checkpointing. Both variants received the same diagnostic overhead.

Across **6,004 decisions**, there were no callback warnings, errors or hard
timeouts; maximum measured bid computation was **0.851 seconds**. The runner
rejects an experiment if any callback warning occurs. The weighted policy had
four explicit L2 fallbacks and uniform had eight; these are legal deliberate
responses, separate from harness timeout/error fallback.

## Cleanup and validation

L4 now derives directly from `Bot` and delegates its baseline bid learning and
reveal decision to `ValuationBot`. Sampling preserves all public facts and exact
hand/deck capacities, including the distinction between starting hands and
purchased gems. Each action uses an ordinary mean over the same complete worlds.
The original evidence-weight parameter is no longer part of public L4's API.

The full comparison tested the zero-evidence precursor before removing unused
tracker state. A regression test verifies that simplified public L4 and that
zero-weight control produce exactly the same bids, sample counts, terminal-score
means, public history and learned rates through fixed-world decisions. CPU-based
stopping can still differ after cleanup. Uniform prior sampling is checked
against exact prior marginals; tests reject any L4 call to hidden-hand inference.

The previous weighting policy remains only in the private
`experiments/weighted_sampling_reference.py`, outside the public bot API and
selector, to reproduce this ablation and the original weighted-versus-L2 report.
`benchmark_sampling.py` now defaults to current uniform L4; pass
`--weighted-reference` for the historical weighting control. Original results
are preserved and labeled historical.

Validation: **144 tests**, Ruff, strict mypy, browser games at three/four/five seats
and five sampling bots together. Extra two-game smoke checks exercise both
benchmark entry points after moving the reference code; they are excluded from
the main tables above.

## Reproduce

```sh
python3 experiments/compare_world_weights.py --games 60 --start-seed 3600 --jobs 6 --batch-size 10 --output experiments/results/world_weights/comparison.json
npm test
npm run dev
```

Select **Sampling search · level 4**. No inference or weighting toggle is needed.
[Per-game scores and paired summaries](comparison.json) ·
[Worker timings and identical-world diagnostics](comparison.metrics.jsonl) ·
[Decision](decision.json).
