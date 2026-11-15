# Tracker evidence shrinkage

Updates now use `posterior ∝ prior × exp(evidence_weight × raw_evidence)`. The conservative experimental default is **0.1**. Inference remains disabled in production bids.

## Meaning of the weight

- `0`: plain hypergeometric prior, still conditioned on all known card facts.
- `0.1`: requested conservative starting point, four times weaker than the previous tracker.
- `0.4`: legacy update strength, retained as a control.
- Evidence is accumulated without the old embedded 0.4 multiplier, decays by 0.96, and is clipped to ±3. The weight is applied to the joint allocation likelihood. At 0.4 this is mathematically equivalent to the former ±1.2 effective log-evidence bound.
- The allocation model continues to enforce exact color totals, opponent hand sizes, and deck size. Hard observations are never softened.
- Weights must be finite and in [0,1]. Changing the weight invalidates cached beliefs; configured weights survive game resets and checkpoints.

## Method

- Same shadow-game protocol as the original prediction diagnostic: forecast before current bids, learn bids/reveal afterward, attach truth only after final reveals.
- Several independent shadows observe each actual game simultaneously. No shadow influences actions. All strengths see identical trajectories.
- Selection: 90 deal seeds 840–929, three opponent profiles (270 actual games), all 3–5 seat counts/charts A–E.
- Grid: 0, 0.025, 0.05, 0.1, 0.2, 0.4. Select minimum informative count MSE; smaller weights break exact ties.
- Validation: 180 untouched seeds 1140–1319, three profiles (540 actual games); evaluate 0, 0.1, 0.2 and the frozen selection winner 0.4.
- Losses average informative color-rounds within game and profiles within seed. Approximate paired 95% intervals cluster on deal seed, not individual forecasts.
- The legacy-strength control reproduced the previous tracker on the same 270 selection games, maximum game-MSE difference 2.22e-16.
- No callback failures occurred.

## Selection results

| Weight | Count MSE | Improvement over prior [95% CI] |
| --- | ---: | ---: |
| 0 | 0.613615 | +0.000000 [+0.000000, +0.000000] |
| 0.025 | 0.613464 | +0.000150 [+0.000048, +0.000252] |
| 0.05 | 0.613317 | +0.000298 [+0.000095, +0.000501] |
| 0.1 | 0.613028 | +0.000586 [+0.000180, +0.000992] |
| 0.2 | 0.612480 | +0.001135 [+0.000325, +0.001944] |
| 0.4 | 0.611500 | +0.002115 [+0.000506, +0.003723] |

The minimum-MSE setting was 0.4, the previous strength. Within the tested range, selection did not support additional shrinkage.

## Fresh validation results

| Weight | Count MSE | Improvement over prior [95% CI] | Improvement over legacy 0.4 [95% CI] |
| --- | ---: | ---: | ---: |
| 0 | 0.610322 | +0.000000 [+0.000000, +0.000000] | -0.001311 [-0.002503, -0.000119] |
| 0.1 | 0.609936 | +0.000386 [+0.000086, +0.000686] | -0.000925 [-0.001817, -0.000032] |
| 0.2 | 0.609589 | +0.000733 [+0.000135, +0.001331] | -0.000578 [-0.001172, +0.000017] |
| 0.4 | 0.609011 | +0.001311 [+0.000119, +0.002503] | +0.000000 [+0.000000, +0.000000] |

**Weight 0.1 beat the plain prior, but did not improve the previous tracker.** Weight 0.4 again had the lowest tested MSE. More prior shrinkage was not an accuracy improvement in this experiment.

## Adoption

- Implemented 0.1 as the requested conservative experimental default, with an explicit tuning parameter.
- The data-selected recommendation remains 0.4; the report does not label 0.1 as optimal.
- **Inference stays off for actual bids.** No shrunken policy has passed a held-out strategy test. The earlier 0.4 strategy ablation also failed its adoption rule.
- Historical flag ablations remain explicitly pinned to 0.4. Gameplay experiments accept `--evidence-weight` to override that legacy setting.

## Reproduce

```sh
python3 experiments/tune_tracker.py --games 90 --start-seed 840 --jobs 3 --output experiments/results/tracker_shrinkage/selection
python3 experiments/tune_tracker.py --weights 0 .1 .2 .4 --validate-weight .4 --log-weight .1 --games 180 --start-seed 1140 --jobs 3 --output experiments/results/tracker_shrinkage/validation
python3 experiments/valuation_experiment.py --candidate soft --inference --evidence-weight .1 --games 120
```

Direct configuration: `SoftTrackingBot(use_bid_evidence=True, evidence_weight=0.1)`.

Full data: [selection](selection/summary.json), [validation](validation/summary.json). Per-round/color forecast examples: [selection](selection/examples.csv), [validation](validation/examples.csv). The nine compressed forecast logs in `validation/` contain weight-0.1 predictions versus actual counts for every validation auction/color.
