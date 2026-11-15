# Level-3 ablation results

The selected production configuration disables all experimental extensions and exactly reproduces level-2 valuation decisions. No extension demonstrated a reproducible improvement on the independent validation block.

## Design

- One candidate versus **two ValuationBots** in each three-player game.
- Selection: seeds 360–599, 240 games for each of all 16 flag combinations (3,840 simulations).
- Validation: seeds 600–839, 240 games for each prespecified single-feature variant and the control (1,200 simulations).
- Charts A–E and candidate seats rotate identically across variants; each 240-game block has 48 games per chart and 80 per seat.
- The all-off candidate matches level 2 exactly, including complete-game round outcomes. Legal affordability checks are always active.
- Ties receive fractional win credit: a two-way tie gives each winner 0.5.
- The paired margin is candidate score minus the highest opponent score. Differences compare each variant with the all-off control on the same seed/seat/chart.
- Intervals are approximate 95% intervals for paired mean margin differences (mean ± 1.96 SE across seeds). Selection among multiple variants can overfit; validation uses a separate seed block.
- Selection required higher win credit and a positive lower interval bound. Eligible variants were ranked by wins, margin improvement, then fewer enabled flags.
- No callback failures or two-second timeouts occurred in either block.

## Flags

| Feature | Constructor flag | Meaning |
| --- | --- | --- |
| Inference | `use_bid_evidence` | Reweight hidden-hand/deck probabilities from bids. |
| Endgame | `use_endgame_rules` | Mission-only buying restriction, urgency, and reserve override near the end. |
| Cash limits | `use_cash_limits` | Condition bid predictions on each rival's current budget; alone this is a no-op because level 2 already enforces affordability/max-rival caps. |
| Bid model | `use_bid_model` | Learn bid rates and optimize surplus instead of fractional bidding. This separate fourth flag prevents confounding cash limits with the optimizer. |

## Selection block

| Enabled features | Win credit / 240 | Mean score | Paired margin change [95% CI] |
| --- | ---: | ---: | ---: |
| baseline | 76.5 | 72.41 | +0.00 [+0.00, +0.00] |
| bid_model | 60.5 | 65.45 | -3.37 [-6.93, +0.19] |
| cash_limits | 76.5 | 72.41 | +0.00 [+0.00, +0.00] |
| cash_limits+bid_model | 68.0 | 65.98 | -2.62 [-6.20, +0.95] |
| endgame | 54.5 | 68.02 | -6.80 [-8.24, -5.36] |
| endgame+bid_model | 50.5 | 63.49 | -6.68 [-10.26, -3.10] |
| endgame+cash_limits | 54.5 | 68.02 | -6.80 [-8.24, -5.36] |
| endgame+cash_limits+bid_model | 52.0 | 63.94 | -6.01 [-9.62, -2.40] |
| inference | 78.0 | 72.71 | +0.45 [+0.04, +0.86] |
| inference+bid_model | 61.5 | 65.51 | -3.36 [-6.90, +0.18] |
| inference+cash_limits | 78.0 | 72.71 | +0.45 [+0.04, +0.86] |
| inference+cash_limits+bid_model | 68.0 | 65.97 | -2.74 [-6.31, +0.83] |
| inference+endgame | 55.5 | 68.19 | -6.51 [-8.01, -5.01] |
| inference+endgame+bid_model | 49.5 | 63.47 | -6.80 [-10.36, -3.25] |
| inference+endgame+cash_limits | 55.5 | 68.19 | -6.51 [-8.01, -5.01] |
| inference+endgame+cash_limits+bid_model | 52.0 | 63.95 | -6.06 [-9.67, -2.46] |

Inference alone was the selection recommendation: a small +0.45 margin improvement with interval [+0.04, +0.86], and 78.0 versus 76.5 win credits. The equivalent inference+cash variant was rejected as redundant.

## Independent validation block

| Enabled features | Win credit / 240 | Mean score | Paired margin change [95% CI] |
| --- | ---: | ---: | ---: |
| baseline | 76.5 | 73.15 | +0.00 [+0.00, +0.00] |
| inference | 75.5 | 73.20 | +0.19 [-0.48, +0.85] |
| endgame | 59.5 | 68.72 | -6.69 [-8.00, -5.38] |
| cash_limits | 76.5 | 73.15 | +0.00 [+0.00, +0.00] |
| bid_model | 54.5 | 65.52 | -4.16 [-7.81, -0.52] |

Inference failed the validation rule: 75.5 versus 76.5 win credits, with paired margin +0.19 and interval [−0.48, +0.85]. It is disabled as unproven, not declared universally harmful. Endgame rules and the optimizer hurt in both blocks. Cash conditioning reduced some optimizer losses in selection but did not make the combined policy beat level 2.

**Adopted defaults: inference off, endgame off, cash conditioning off, bid model off.** The public SoftTrackingBot remains configurable, but its selected default is level 2. Normal own-cash, loan-funding, reserve, and maximum-rival spending rules are retained.

The evidence covers this three-player lineup and these seeds. It does not establish universal results for other opponent policies or four/five-player tournaments.

## Reproduce

```sh
python3 experiments/ablate_level3.py --games 240 --start-seed 360 --jobs 3 --output experiments/results/level3_selection.json
python3 experiments/ablate_level3.py --games 240 --start-seed 600 --jobs 3 --variants baseline inference endgame cash_limits bid_model --validate-variant inference --output experiments/results/level3_single_feature_holdout.json
```

Full per-game data: [selection](level3_selection.json), [validation](level3_single_feature_holdout.json), [adoption decision](level3_decision.json).
