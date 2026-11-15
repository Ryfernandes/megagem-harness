# Tracker forecast validation

The legacy-strength tracker (equivalent evidence weight 0.4) was tested as a shadow observer. It never supplied bids or reveals to the playing bots. Its estimates slightly beat the plain hypergeometric prior, but inference remains disabled for bidding because the independent strategy ablation did not validate an improvement in wins.

## Dataset and timing

- 900 games: 300 fresh deal seeds (840–1139) replayed under three opponent profiles.
- Each profile includes 100 games at each seat count (3, 4, 5), and 60 games per chart (A–E). Observer seats rotate by `(seed // 15) % seats`.
- The observed seat always plays ordinary level-2 ValuationBot. Profiles: all valuation rivals, all fixed-rule rivals, and alternating valuation/fixed-rule rivals.
- Every forecast is recorded before current bids are requested. The tracker receives the current round's bids/reveal only after that forecast.
- The observer receives one normal immutable BotInfo, with only its own hidden hand. It is never given opponents' hidden identities or draw-pile order.
- Truth is attached only after finish_game reveals all remaining hands. Logs have one row per auction/color, with both predicted means, both seven-count PMFs, and actual final display count.
- 74,355 total color-round rows; 58,334 informative rows; 16,021 already-known rows.
- Already-known forecasts are retained in logs and checked for correctness, but excluded from the primary accuracy comparison. Including them would dilute the test with trivial predictions.
- Forecasts within a game are averaged first; profiles are then averaged within seed. Confidence intervals cluster on 300 deal seeds, rather than pretending that correlated rounds/colors/profile replays are independent.
- Metrics and the prediction gate were specified in the evaluator before the run. No tracker weights were tuned on these seeds. No callback failures occurred.

## Accuracy on informative observations

| Metric (lower is better) | Plain prior | Tracker | Prior minus tracker [approx. 95% CI] |
| --- | ---: | ---: | ---: |
| Count MSE | 0.615812 | 0.613738 | +0.002074 [+0.001138, +0.003010] |
| Count MAE | 0.629655 | 0.628864 | +0.000790 [+0.000234, +0.001346] |
| Count-distribution Brier | 0.622230 | 0.621183 | +0.001048 [+0.000527, +0.001568] |
| Count log loss | 1.100451 | 1.098451 | +0.002000 [+0.001118, +0.002882] |

The count MSE improvement is **0.34%**, a small predictive gain. Brier and log-loss gains also favor the tracker.

| Opponent profile | Prior MSE | Tracker MSE | MSE gain [approx. 95% CI] |
| --- | ---: | ---: | ---: |
| valuation | 0.610827 | 0.607526 | +0.003301 [+0.002412, +0.004190] |
| fixed | 0.622046 | 0.620983 | +0.001063 [-0.000414, +0.002541] |
| mixed | 0.614562 | 0.612704 | +0.001857 [+0.000815, +0.002900] |

The fixed-rule profile is a negative control: its bids ignore private hands. That profile shows no clear gain, consistent with the weak-evidence concern. The strongest gain is against valuation opponents, whose bids use private information.

## Per-color count averages

These descriptive averages summarize forecasts across informative rounds; they are not substitutes for paired per-forecast losses above.

| Color | Prior forecast | Tracker forecast | Actual final count |
| --- | ---: | ---: | ---: |
| BLUE_CIRCLE | 3.114495 | 3.113765 | 3.136667 |
| GREEN_OVAL | 3.129852 | 3.130333 | 3.133333 |
| PURPLE_TRIANGLE | 3.011419 | 3.011923 | 3.030000 |
| YELLOW_SQUARE | 3.090094 | 3.089384 | 3.050000 |
| PINK_PENTAGON | 3.040226 | 3.040244 | 2.983333 |

## Decision

- The prediction gate passed: positive lower interval bounds for MSE and Brier gains; nonnegative mean log-loss gain; no profile with clearly worse MSE.
- The separate held-out strategy test did not pass. Prediction accuracy alone does not establish a bidding improvement.
- **Bid-driving inference stays off.** The diagnostic run cannot automatically enable any production feature. Experimental flags remain available for deliberate tests.

## Logs and reproduction

- [Uncompressed examples](examples.csv): per-round/per-color predicted means, PMFs, final actual count, and losses.
- [Summary and per-game metrics](summary.json): profile, seat-count, chart, and color breakdowns.
- Complete logs are the nine `*_forecasts.csv.gz` files in this directory. Each row includes the actual final count alongside the prediction made earlier.

```sh
python3 experiments/evaluate_tracker.py --games 300 --start-seed 840 --jobs 3 --output experiments/results/tracker_predictions --evidence-weight .4
```

Read a complete log with the Python standard library:

```python
import csv, gzip
with gzip.open("experiments/results/tracker_predictions/valuation_4_forecasts.csv.gz", "rt") as f:
    for row in csv.DictReader(f):
        print(row["seed"], row["round"], row["color"],
              row["prior_mean"], row["tracker_mean"], row["actual_count"])
```
