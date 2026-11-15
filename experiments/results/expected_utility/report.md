# Expected-utility bidding comparison

Current level-3 baseline update: SoftTrackingBot now matches current ValuationBot at all seat counts, with all three optional extras off. See the [finance-horizon follow-up](../finance_horizon/report.md). Historical results below remain unchanged.

Later follow-up: the public ValuationBot now also uses this bidder at every supported seat count. See the [fresh valuation comparison](../valuation_expected_utility/report.md). The historical control in this report is now preserved as FractionalValuationBot.

The selected expected-utility treasure bidder improves held-out three-player games. SoftTrackingBot now uses it for three players; four/five-player defaults retain the previous fractional policy because improvement was not established there. ValuationBot remains the unchanged level-2 control.

## What changed

Treasure bids maximize `p(win) × (E[value] − (1+k) × bid)` over every reserve-affordable integer bid, pruning only bids above every opponent’s cash plus one. Win probabilities use the existing learned, cash-bounded opponent bid distributions and the actual first-round/clockwise tie rules.

The old experimental optimizer already used the formula, but with the original liquidity premium and a 95%-of-value bid cap. It is included as `old_eu`. Removing the cap alone (`heuristic`) did not improve these selection games. The improvement came from calibrating the opportunity cost of cash, rather than adding hand inference.

The selected setting is `floor0.5`: `k = (liquidity − 1) + 0.5 × L/(L+1.5)`, where `L` estimates the number of future treasure lots from remaining gems and remaining auction-card composition, excluding the current lot. The extra premium expires at the final treasure lot. It is a heuristic selected from eight predeclared settings, not an exact value function.

Gem EV, chart selection, mission and denial values, the reserve rule, first-card reveal policy, investments and loans are held fixed. Hand inference and mission-only endgame rules stay off. Financial auctions keep their existing valuation policy: investment bids are returned at scoring and loan proceeds offset principal, so treasure cost arithmetic is not applied indiscriminately to them.

## Method

One candidate faces only ValuationBot opponents. Each variant uses the same seed, chart and candidate seat, with charts A–E and seats rotated deterministically. Outcomes may diverge after different bids, as intended. A win belongs to that one candidate; opponent wins are not pooled into the numbers below. Ties split win credit. All callbacks run through the real isolated harness with the two-second timeout; no callback errors or fallback decisions occurred.

Selection: 240 seeds (1560–1799), three players, ten policies including the before/old controls. Selection requires a higher fractional win rate and a positive paired score-margin 95% lower bound. Freeze `floor0.5` before fresh validation: 300 seeds (1800–2099) with three players, and 180 seeds (1800–1979) separately with four and five players. The same seeds across seat counts do not make those rows independent samples to pool.

Approximate 95% intervals use paired per-seed margin differences, mean ± 1.96 standard errors. Margin is candidate score minus the strongest rival score. These intervals assess score margin, not a guarantee of a higher tournament win rate.

## Selection

| Policy | Win credit / 240 | Mean score | Margin change vs before [95% interval] |
| --- | ---: | ---: | ---: |
| before | 92.0 | 73.49 | +0.00 [+0.00, +0.00] |
| old_eu | 59.0 | 66.22 | -4.85 [-8.07, -1.64] |
| myopic | 59.5 | 64.40 | -7.07 [-10.45, -3.68] |
| heuristic | 59.0 | 66.22 | -4.85 [-8.07, -1.64] |
| scale2 | 75.5 | 67.96 | -2.86 [-6.07, +0.36] |
| scale4 | 65.0 | 68.49 | -3.73 [-6.96, -0.49] |
| floor0.5 | 105.5 | 74.08 | +3.50 [+0.35, +6.65] |
| floor1 | 92.5 | 73.20 | +0.12 [-2.27, +2.51] |
| floor2 | 28.5 | 63.54 | -17.20 [-20.42, -13.98] |
| scale2_floor1 | 87.0 | 72.48 | -0.95 [-3.47, +1.58] |

## Fresh validation

| Players | Games per policy | Before win credit | EU win credit | Before / EU mean score | EU margin change [95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 300 | 113.5 | 147.0 | 74.44 / 76.40 | +6.26 [+3.54, +8.97] |
| 4 | 180 | 45.0 | 52.5 | 50.90 / 50.53 | +2.18 [-1.72, +6.09] |
| 5 | 180 | 43.0 | 41.5 | 43.69 / 41.16 | -2.01 [-5.49, +1.47] |

Three-player win credit rises from 37.8% to 49.0% (+11.2 percentage points); paired margin improves by 6.26 points. Four-player wins rise slightly but the margin interval spans zero; five-player point estimates are worse and inconclusive. Only the three-player setting passes the adoption rule. No claim of universal dominance or improvement against other opponent populations is made.

The held-out old optimizer earns 95.0/300 win credits and 68.10 mean points, versus the selected policy’s 147.0 and 76.40. Keeping sufficient future cash materially changes the outcome even with the same win predictor.

## Reproduce

```sh
python3 experiments/compare_utility.py --games 240 --start-seed 1560 --jobs 3 --output experiments/results/expected_utility/selection.json
python3 experiments/compare_utility.py --games 300 --start-seed 1800 --seats 3 --variants before old_eu floor0.5 --validate floor0.5 --output experiments/results/expected_utility/validation_3_seats.json
python3 experiments/compare_utility.py --games 180 --start-seed 1800 --seats 4 --variants before floor0.5 --validate floor0.5 --output experiments/results/expected_utility/validation_4_seats.json
python3 experiments/compare_utility.py --games 180 --start-seed 1800 --seats 5 --variants before floor0.5 --validate floor0.5 --output experiments/results/expected_utility/validation_5_seats.json
```

JSON artifacts include every per-game seed, chart, seat, score, win credit, settings and paired summary. Historical ablation classes pin their original flags, penalties and bid caps, so previous reports remain reproducible. The selection/validation experiments total 4,020 actual games.

Use the existing **Soft tracking · informed bids** selector in the browser. To force EU for experiments at any legal seat count, construct `SoftTrackingBot(use_bid_model=True, use_cash_limits=True, utility_player_counts=(3, 4, 5))`; `use_bid_model=False` restores fractional bidding. Default hand inference remains disabled. The no-option `valuation_experiment.py --candidate soft` runs the public default; explicit legacy feature overrides run the historically pinned policy and print its penalty settings.
