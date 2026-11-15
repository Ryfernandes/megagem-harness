# ValuationBot with the new expected-utility bidder

Current level-3 baseline update: SoftTrackingBot now matches current ValuationBot at all seat counts, with all three optional extras off. See the [finance-horizon follow-up](../finance_horizon/report.md). Historical results below remain unchanged.

ValuationBot now uses the previously selected expected-utility treasure bidder with three, four and five players. Its gem valuations, mission and denial estimates, reserve, investment/loan bids and reveal decisions retain the former policy. The existing browser selector loads the updated bot. No hidden-hand inference or mission-only endgame restriction was added.

## Implementation

Treasure bids maximize `p(win) × (E[value] − (1+k) × bid)` over all reserve-affordable integer bids up to the richest rival’s cash plus one. The penalty remains frozen at the earlier selected setting: `k = liquidity − 1 + 0.5 × L/(L+1.5)`, where `L` estimates future treasure lots, excluding the current lot. The extra premium expires at the last treasure purchase. No parameters were tuned on these comparison games.

Both bots share the same cash-bounded bid-distribution model, public-bid learning update and first-round/clockwise tie probabilities. Tests compare the updated valuation bot with forced-on SoftTrackingBot through every round of actual games across all five charts and all supported player counts, including learned bid-rate state. SoftTrackingBot’s default player-count scope remains unchanged.

The former policy is preserved as `FractionalValuationBot`, omitted from `custom_bots.__all__`. `ValuationBot(use_expected_utility=False)` also restores its former decisions. Historical ablation and tracker runners explicitly use the frozen class, so public-bot upgrades cannot silently replace their controls. This is not an extra submitted bot.

## Fresh matched comparison

Each run has **one candidate** and all remaining seats use the frozen former valuation policy. Before/after runs use the same seed, initial deal, chart and candidate seat. Charts A–E cycle with seed; seating rotates. The numbers below belong to the single candidate, not a sum of opponent wins. Tied winners split win credit.

There are 300 games per policy per seat count, on fresh seeds 2100–2399: 1,800 actual games in total. The same seeds at different seat counts are not independent blocks to pool. Every game uses the real isolated harness with two-second callbacks; no callback warnings, errors, timeouts or fallback decisions occurred.

| Players | Before win credit / 300 | After win credit / 300 | Before / after mean points | Paired margin change [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| 3 | 121.5 | 141.5 | 74.70 / 76.47 | +5.36 [+2.76, +7.95] |
| 4 | 58.0 | 82.0 | 49.80 / 49.49 | +1.11 [-1.39, +3.61] |
| 5 | 52.5 | 68.0 | 43.05 / 41.66 | +0.21 [-2.42, +2.83] |

Margin means candidate score minus the strongest opponent score. Approximate paired 95% intervals use per-seed differences, mean ± 1.96 standard errors. They concern this opponent population and do not guarantee general tournament superiority.

| Players | Before → after win credit rate | Paired change in percentage points [95% interval] |
| --- | ---: | ---: |
| 3 | 40.50% → 47.17% | +6.67 [+0.13, +13.20] |
| 4 | 19.33% → 27.33% | +8.00 [+1.70, +14.30] |
| 5 | 17.50% → 22.67% | +5.17 [-0.72, +11.05] |

Win credit rises at all tested seat counts. The paired win-credit interval excludes zero at three and four seats, while the five-seat interval includes zero. Only the three-seat score-margin interval excludes zero. Mean points decline slightly at four and five seats even as the candidate wins more often; changing auction winners also changes rival scores. These intervals are approximate and are not adjusted for the three seat-count comparisons.

The updated bidder is enabled at every supported seat count as requested. The table reports all tested seat counts; no seat count was chosen after seeing these deals. These are comparisons against the former valuation policy, not measurements of a whole table of upgraded bots or of mixed tournament lineups.

## Reproduce

```sh
python3 experiments/compare_valuation_bidder.py --games 300 --start-seed 2100 --jobs 3 --output experiments/results/valuation_expected_utility/comparison.json
npm test
npm run dev
```

Select **Valuation · expected value** in any browser seat. The JSON contains all per-game seeds, charts, candidate seats, scores, win credits, paired summaries and frozen parameters. All changes remain local and uncommitted.
