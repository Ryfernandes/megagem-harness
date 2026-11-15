# Level 4: weighted sampling search

Historical results: current public L4 now uses uniform weights and has no
level-3 tracker, following the [world-weight ablation](../world_weights/report.md).
The results below describe the previous weighted policy. Reproduce that control
with `benchmark_sampling.py --weighted-reference`; the default runs current L4.

`SamplingBot` is a separate browser option, **Sampling search · level 4**. Current
`ValuationBot` and the unconfigured level-3 policy retain their decisions. This
experiment evaluates a bounded search; it does not assume search always wins.

## Policy

Sample the multiset of unseen gems without replacement into opponents' remaining
hands and the hidden gem deck. Preserve exact color totals, hand sizes, the
bot's own hand, auction row, collections and already revealed information.
Shuffle the known remaining auction-card multiset separately. Deck/auctioned
gems never count as display gems; starting hands alone set terminal prices.

The only additional level-3 feature enabled is bid-evidence inference, tempered
at `0.1`. Sample from the ordinary conditional prior, then importance-weight each
world by `exp(0.1 × sum of remaining-hand log evidence)`. Do not apply those
likelihoods to the deck or count reveals twice. Tests recover the exact tracker
marginal with a large importance sample and check full card conservation.
Endgame restrictions, finance-horizon k, cash-fraction prediction and strategic
reveals stay off. Cash limits and expected-utility learning are the shared level-2
behavior, not extra inference flags.

Every continuation player uses current level 2, with its own private snapshot.
Reconstruct each sampled opponent's learned bidding rates from the actual public
bid history and that world's possible private hand. Within a rollout, the real
harness resolves auctions, tie priority, reveals, missions, financial cards and
final scoring. A direct policy adapter avoids spawning workers inside workers.
Pure mission computations are memoized; no level-2 arithmetic changes.

Compare candidate bids on the same sampled worlds, future deck orders and tie
randomness. Choose the highest importance-weighted mean **own final coin score**;
rollout win rate is not the optimization target. Initially consider at most seven
legal bids, including the level-2 incumbent and sampled rival thresholds. Search
may spend reserve cash. A timed pilot adapts the number of actions by computation
cost alone, always retaining the incumbent; it does not select actions using the
pilot's score. Full continuation outcomes are committed only after every active
action finishes for that world.

The default allowance is 0.85 seconds, at most 16 worlds, with a two-world minimum.
Unix workers use an interruptible timer in addition to cooperative checks. It is
restored/disarmed before checkpointing. Too few completed worlds or preparation
expiry returns the level-2 action; if the baseline itself is interrupted the
explicit emergency response is legal bid zero. Cashless non-loan auctions skip
search because zero is their only legal bid. Other platforms retain cooperative
checks. The actual isolated harness still enforces two-second callbacks.

## Matched comparison

**One candidate** plays against current `ValuationBot` in every other seat. The
control substitutes a current level-2 bot in that same candidate seat. Each row
reports only that single candidate's wins, not the combined opponent wins.
Ties split win credit. Both policies use the same initial-deal seeds, chart and
rotating candidate seat.

There are 60 games per policy at each of three, four and five seats: **360 actual
games**, seeds 3300–3359, charts `ABCDE[seed % 5]`, candidate seat
`(-seed) % seats`. Settings were fixed before this run. No search settings were
tuned from these game outcomes. The runner uses ten-deal batches and six parallel
jobs, starting fresh fixed-per-seat worker RNGs in each batch and saving completed
batches to an incremental checkpoint. The same seed across different seat counts is
not an independent replicate to pool; at five seats, chart and candidate seat
are coupled by the rotation. This is not a fully crossed seat/chart tournament.

| Players | L2 win credit / 60 | L4 win credit / 60 | L2 / L4 mean points | Paired score change [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| 3 | 16.5 | 53.0 | 66.12 / 90.73 | +24.62 [+20.11, +29.12] |
| 4 | 16.5 | 47.0 | 47.42 / 68.98 | +21.57 [+15.66, +27.47] |
| 5 | 16.0 | 46.5 | 40.38 / 66.13 | +25.75 [+20.15, +31.35] |

The paired mean-score interval is above zero at 3, 4, 5 seats.

Intervals are approximate paired per-seed means ± 1.96 standard errors, without
adjustment across the three seat counts. Search optimizes points; its win credit
can move differently. These opponents all use level 2, so results do not establish
strength against other policies or mixed tournament lineups. Sixty deals per row
also provide less precision than the previous 300-game feature experiments.
The level-4 bot remains a separate option; level 2 and level-3 defaults are unchanged.

## Timing and validation

| Players | Decisions | Mean / p95 / max seconds | Mean worlds per successful search | Mean rollouts per decision | L2 fallbacks | Forced-zero skips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1009 | 0.582 / 0.837 / 0.850 | 11.13 | 51.34 | 2 | 45 |
| 4 | 940 | 0.624 / 0.838 / 0.850 | 8.08 | 32.81 | 1 | 54 |
| 5 | 1009 | 0.622 / 0.839 / 0.851 | 6.70 | 23.22 | 9 | 86 |

Timing covers `get_bid` computation, excluding host transport/checkpointing.
World means exclude fallbacks and forced-zero decisions; rollout means include
all decisions. Rollouts count completed equal-action comparison batches, not
abandoned partial work. Counts, importance effective sample size and fallback
reasons are in the raw telemetry and public `last_search`/`search_log`.
The 12 level-2 fallbacks comprise six interrupted searches and six searches with
fewer than two complete worlds. No emergency baseline-deadline zero bid occurred.

The accepted comparison produced **no callback warnings, errors or hard
timeouts**. The runner rejects an entire comparison if any callback warning
occurs. A prior cooperative-deadline run hit a hard timeout and was rejected;
its [log](invalid_cooperative_deadline.log) and
[telemetry](invalid_cooperative_deadline.metrics.jsonl) are retained for auditing
and excluded from every table above. Another long process was externally
terminated before saving full comparison results; its
[partial telemetry](interrupted_process.metrics.jsonl) is also excluded. The final
run uses the interruptible guard and checkpoints completed ten-game batches.

All **142 tests** pass, including complete-continuation parity with real isolated
level-2 games on all charts and all seat counts, private-view separation,
importance-weight agreement, history reconstruction, partial-world discard,
terminal scoring/ties, interruption cleanup and worker checkpoint/reset behavior.
Ruff and strict mypy pass. Browser checks passed at three/four/five seats, plus
five sampling bots playing together, legal bids, final scores, mobile layout
and no JavaScript errors. The five-sampling-bot browser game took about 46 seconds.

Search is approximate: few sampled worlds, a reduced bid set, heuristic bid
likelihoods and assumed level-2 continuation policies can all introduce error.
Time-limited choices vary with CPU load, even with fixed initial seeds. This is
bounded policy search, not full game-tree search or a guarantee of optimal play.

## Reproduce

```sh
npm install
npm run dev
# Select Sampling search · level 4 in any seat.
python3 experiments/benchmark_sampling.py --weighted-reference --games 60 --start-seed 3300 --jobs 6 --batch-size 10 --output experiments/results/sampling_search/comparison.json
npm test
```

[Per-game scores and paired summaries](comparison.json) ·
[Worker timing telemetry](comparison.metrics.jsonl).
