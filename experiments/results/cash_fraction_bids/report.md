# Cash-fraction opponent bid model

Implemented as the optional `use_cash_fraction_model` level-3 feature. It stays OFF by default. With all four extras off, level 3 continues to match current level 2 exactly. All other features, including finance-aware k, remain off in this comparison.

## Model and observations

The existing level-2 predictor already makes win probability depend on c. This feature replaces its value-conditioned prediction with type-specific cash-fraction histories; it is not a correction from a constant p(win).

Every validated opponent bid is logged using the pre-auction snapshot, with round, player, auction type/face, cash, legal budget, bid, fraction and evidence weight. Types are treasure, investment and loan; opponents never share histories. Ordinary fractions are bid/cash. For loans the denominator is cash+loan proceeds, since that is the legal bankroll; a cashless borrower can still bid. Forced zero-bankroll passes are logged but carry no preference evidence. Losing bids and voluntary passes are included. Histories are bounded at 32 observations per type and reset each game; all fields survive worker checkpoints.

Predictions mix a rounded/clipped normal prior with an empirical PMF of old fractions rescaled to the opponent’s CURRENT legal budget. Two prior pseudo-observations retain uncertainty. Prior fraction mean/spread: treasure .20/.14; investment .20/.16; loan .06/.10. Treasure observations have weight .25 versus 1 for finance, because treasure lot values and mission interest vary. Mean/spread inspection reflects latent fractions before prior clipping and integer rounding. Actual PMFs are normalized on bids 0 through the legal budget, with exactly zero mass beyond that budget.

For each possible c, p(win) multiplies rival CDFs. It includes rival bids equal to c only when our clockwise rank beats that rival; first-round tied winners are integrated uniformly. Scanning all affordable integer bids lets favorable tie priority save a coin. The existing own reserve and maximum-useful-bid cap remain in place.

## Financial utility and isolation control

The checkbox-enabled feature uses these distributions for all auction types. Extending the optimizer to finances is a policy change as well as a prediction change, so the benchmark separately includes a treasure-only predictor control:

- `off`: current level-2 value-conditioned prediction and existing finance policy.
- `treasure_only`: cash-fraction prediction for treasure, with financial bidding unchanged. All auction types are logged, but finance predictions do not drive bids in this control.
- `all_types`: the complete switchable feature, with cash-fraction prediction and EU for all three auction types.

Financial EU uses the existing financial value estimates and legal budgets:

```text
Treasure:   p(c) × (gem + mission + denial value − (1+k)×c)
Investment: p(c) × (face − max(.70, liquidity−1)×c)
Loan:       p(c) × ((liquidity−1)×min(face, productive_loan) − liquidity×c)
```

An investment returns its bid, so its principal is not charged as lost final-score coins. Loans add principal and matching debt; the reward is useful future buying power, not free score. Loans retain the useful-proceeds guard (bid < face); own loan affordability includes proceeds. Finance-aware k, when separately enabled, continues to affect treasure only. Gem/mission/denial estimates, treasure k, reserve and first-card reveals are unchanged by the new predictor.

## Fresh paired benchmark

300 games per policy per player count on fresh seeds 2700–2999, charts A–E, rotating seats: 2,700 actual games. Each has one candidate and only CURRENT level-2 ValuationBot opponents. Initial deal, chart and candidate seat match across policies. Each win-credit number belongs to that one candidate; tied winners split credit, including thirds. No hyperparameters were retuned after these results. All callbacks used isolated workers and the two-second allowance; there were no callback warnings, timeouts or fallback decisions.

| Players | Existing predictor win credit / 300 | Cash fractions, treasure only | Cash fractions, all types |
| --- | ---: | ---: | ---: |
| 3 | 103.50 | 27.50 | 35.00 |
| 4 | 64.00 | 22.50 | 48.00 |
| 5 | 52.33 | 34.83 | 53.50 |

| Players | Policy | Mean points | Paired margin change vs existing predictor [95% interval] |
| --- | --- | ---: | ---: |
| 3 | off | 67.33 | +0.00 [+0.00, +0.00] |
| 3 | treasure_only | 57.93 | -19.61 [-22.36, -16.86] |
| 3 | all_types | 61.06 | -15.60 [-18.42, -12.79] |
| 4 | off | 46.12 | +0.00 [+0.00, +0.00] |
| 4 | treasure_only | 40.01 | -10.84 [-13.15, -8.52] |
| 4 | all_types | 44.75 | -5.50 [-8.12, -2.88] |
| 5 | off | 38.52 | +0.00 [+0.00, +0.00] |
| 5 | treasure_only | 35.69 | -5.18 [-7.63, -2.73] |
| 5 | all_types | 39.50 | -0.38 [-2.85, +2.10] |

| Players | Complete feature win-credit rate change vs existing predictor, percentage points [95% interval] |
| --- | ---: |
| 3 | -22.83 [-28.45, -17.22] |
| 4 | -5.33 [-11.62, +0.95] |
| 5 | +0.39 [-5.29, +6.07] |

**The new model does not improve this lineup.** Treasure-only replacement hurts at every player count, showing that the poor result is not solely a financial-policy change. The complete feature hurts score margin substantially at three and four seats, and is inconclusive at five. Financial EU recovers some of the treasure-only loss, but does not establish an overall gain. The existing predictor remains the default.

Cash fractions discard the current treasure’s value as a predictor. That is a plausible explanation for losing to value-based rivals across changing lots, but this experiment does not establish the cause or measure performance against other opponent populations.

## Financial-policy control

Paired margin changes for `all_types` versus `treasure_only` (not versus level 2):

| Players | Margin change [95% interval] |
| --- | ---: |
| 3 | +4.00 [+1.97, +6.04] |
| 4 | +5.34 [+3.30, +7.37] |
| 5 | +4.80 [+2.78, +6.82] |

Intervals are approximate paired 95% mean ± 1.96 standard errors across seeds; margin is candidate score minus the strongest rival score. They are not adjusted for multiple comparisons. Charts and candidate seats are individually balanced; the existing seed%seats / seed%5 rotation links chart and seat at five players, so that block does not cover every chart-seat combination. Blocks at different player counts reuse seed labels and should not be pooled as independent samples. These are comparisons with current level-2 opponents, not older fractional-policy rivals.

## Validation, logs and use

114 tests pass, including cash/loan support, separate opponent/type records, pre-bid cash timing, low-weight treasure evidence, empirical scaling and uncertainty, forced passes, all-type learning, changing p(c), favorable/unfavorable ties, investment principal return, loan debt offset, checkpoint/reset behavior and real worker games across all player counts. Existing full-game/per-round tests continue to establish default level-3 equality with level 2. A separate combination smoke check and browser QA exercise per-seat switches, replay metadata/labels, final scoring and mobile layout.

`observations.csv` logs 57 opponent bids from a four-player Chart E game. It includes predicted fraction moments, actual fractions, predicted mean bids and p(win) at zero, the chosen bid and a bid above all rival budgets. Predictions use frozen pre-bid histories and never the current bids they are evaluated against. A shadow copy of the same deterministic predictor generates the audit alongside the actual isolated-worker game; it does not influence play.

```sh
python3 experiments/benchmark_cash_fraction_bids.py --games 300 --start-seed 2700 --jobs 3 --output experiments/results/cash_fraction_bids/comparison.json
python3 experiments/log_cash_fraction_bids.py --seats 4 --seed 2700 --chart E --output experiments/results/cash_fraction_bids/observations.csv
python3 experiments/valuation_experiment.py --candidate soft --cash-fraction-model --games 120
npm test
npm run dev
```

Select **Soft tracking · informed bids**, then check **Cash-fraction bid model**. Other extras stay off. Direct Python callers use `SoftTrackingBot(use_cash_fraction_model=True)` and can inspect `bot.cash_fraction_model.history`. The model is not an additional public bot submission. Historical ablation/utility classes explicitly pin it off, and the previous finance benchmark pins the new flag off. All changes remain local and uncommitted.
