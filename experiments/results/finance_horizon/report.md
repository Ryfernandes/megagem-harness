# Switchable finance-aware k

Level 3 now defaults to the current level-2 ValuationBot policy at all three, four and five seats. Its previous four/five-player fractional fallback was stale after the level-2 upgrade and has been removed. The three level-3 extras are all OFF: `use_bid_evidence`, `use_endgame_rules`, and `use_finance_horizon`. Cash-bounded expected-utility bidding and learned public-bid rates belong to the shared level-2 baseline.

## Feature

Enable `SoftTrackingBot(use_finance_horizon=True)`, pass `--finance-horizon` to `experiments/valuation_experiment.py --candidate soft`, or select **Soft tracking · informed bids** in the viewer and check **Finance-aware cash planning** for that seat. The other extras stay off. Replay metadata and player cards identify the flags actually used, independently of later setup edits.

Only the treasure bidder’s k changes. Gem EV, mission and denial values, reserve, bid distributions/learning, financial bids and reveal policy remain fixed. Forecast loan proceeds never enter the legal bid budget or final score. Financial auctions continue using the existing finance policy.

## Exact horizon; heuristic funding

For G remaining gems after the current treasure lot, a dynamic program integrates the remaining one-/two-gem cards without replacement to find P(T=t), where T is the number of future treasure auctions before the gems are exhausted. Its support gives tight minimum/maximum treasure counts, including a truncated last two-gem lot. No auction order or opponent hand is inspected.

Let M be the number of remaining treasure cards and F the number of remaining financial cards. Given T=t, a particular financial card lies in one of M+1 equally likely gaps between treasure cards. Exactly t gaps occur before the terminal treasure. Thus:

```text
P(a particular financial card appears before game end) = E[T]/(M+1)
E[total future rounds] = E[T] + F*E[T]/(M+1)
min total rounds = min support(T)
max total rounds = max support(T) + F
```

For G=0, all future counts and k are zero, even with financial cards left. Cards after the final treasure never count. Loans and investments already drawn are excluded through the harness’s exact remaining-card inventory.

Financial availability is weighted by the fraction of treasure opportunities after each card. Given t, the weight averaged over the gaps is `(t+1)/(2*(M+1))`; this discounts late-arriving funding and late investment lockups. These means and bounds are exact for a uniformly shuffled remaining auction deck. They are not an exact game-value calculation.

Cash projections are deliberately approximate and fixed before testing: retain 50% of timing-weighted loan face value after acquisition costs, distribute it equally across players, and subtract timing-weighted predicted investment bids. Investment bids use the existing 45%-of-return/liquidity heuristic and a reserve-preserving average-bankroll cap. These assumptions do not guarantee that our bot wins a loan or investment.

Projected own/table cash and existing future gem EV feed the existing capped scarcity premium. The new horizon averages `E[T]/(E[T]+1.5)` and `E[R]/(E[R]+1.5)` for treasure and total rounds. `k = (scale*projected_scarcity + floor)*new_horizon`, with scale=1 and floor=0.5. Tight bounds constrain these expectations through the exact stopping-time distribution. At the final treasure, k and the unchanged reserve are zero.

Example: with 13 future gems, 11 single-gem cards, five double-gem cards and all eight financial cards remaining, treasure auctions are bounded by 8–12 and total rounds by 8–20. Expected treasure auctions are 10.011; expected total rounds are 14.722. Full metrics are in forecast_example.json.

## Paired on/off benchmark

One level-3 candidate faces only CURRENT level-2 ValuationBot opponents. Feature off and on use the same seed, initial deal, chart and rotating candidate seat. Other level-3 extras stay off. There is no tuning on these results and no post-result adoption. Ties split win credit; each number belongs to the one candidate, not the combined opponents.

300 fresh seeds (2400–2699) per policy at each of three, four and five players: 1,800 actual games. All charts A–E are balanced. Every callback runs through the real isolated harness with a two-second limit. There were no callback warnings/errors, timeouts or fallback decisions. These results use stronger current level-2 rivals; they should not be directly compared with old reports using fractional-policy rivals.

| Players | Off win credit / 300 | On win credit / 300 | Off/on mean points | Paired margin change [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| 3 | 84.5 | 89.5 | 67.47 / 67.19 | -0.13 [-1.21, +0.96] |
| 4 | 83.5 | 86.0 | 47.08 / 47.09 | -0.07 [-0.81, +0.67] |
| 5 | 48.5 | 45.0 | 38.02 / 37.70 | -0.26 [-1.09, +0.56] |

| Players | Paired win-credit rate change in percentage points [95% interval] |
| --- | ---: |
| 3 | +1.67 [-1.50, +4.83] |
| 4 | +0.83 [-1.21, +2.87] |
| 5 | -1.17 [-2.98, +0.65] |

Approximate paired 95% intervals use mean ± 1.96 standard errors of per-seed differences. Margin is candidate score minus the strongest rival score. The same seeds across different player counts are not independent blocks to pool, and intervals are not adjusted for multiple seat-count comparisons.

**No demonstrated improvement:** every score-margin and win-credit change interval includes zero. Win-credit point estimates rise slightly at three/four players and fall at five. The default remains OFF, as requested; accurate horizon arithmetic alone does not establish that the liquidity heuristic improves strategy. No additional hidden-hand or endgame feature was enabled.

## Validation and reproduction

104 tests pass. New checks compare exact horizon bounds, expected financial appearances and timing weights against every labeled deck ordering of small inventories. They verify final-auction k=0, funding/lockup effects, legal bids, unchanged valuations/finance/reveals, worker checkpoint/reset behavior and cold forecast runtime below two seconds. Default level-3 equality with current level 2 is checked through full replay games at 3–5 seats and per-round public-bid learning on all five charts. Browser checks cover per-seat switches, disabled defaults, worker options, replay labels, final scores and mobile layout.

```sh
python3 experiments/benchmark_finance_horizon.py --games 300 --start-seed 2400 --jobs 3 --output experiments/results/finance_horizon/comparison.json
python3 experiments/valuation_experiment.py --candidate soft --finance-horizon --games 120
npm test
npm run dev
```

Historical four-flag ablation classes explicitly keep finance planning off and retain their original parameters. The modern three-extra registry is separate. Historical CLI settings now require `--legacy-ablation`; modern inference and finance flags use the current level-2 baseline. All changes remain local and uncommitted.
