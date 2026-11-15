# Optional strategic reveals

Implemented as `SoftTrackingBot(use_strategic_reveal=True)`, CLI `--strategic-reveal`, and the per-seat **Strategic reveals** checkbox. It remains OFF by default. All other level-3 extras stay independently switchable and off. Current level-2 and the default level-3 policy still reveal the first remaining hand card.

## Why chart direction matters

All starting-hand cards enter the final display whether shown early or at game end. A reveal therefore changes information and perceived prices, not the final supply or the scoring value of already owned gems. It never adds a hand card to a scoring collection.

On ascending A/C, a reveal raises the displayed quote, so the intended bait is a color we care less about. On descending B/D it lowers the quote, so applying the same least-value rationale unchanged would cool an unwanted color. The extension instead tries to cool competition for desired future purchases. These are information-release heuristics: rivals estimating final values accurately need not follow current quotes.

## Exact rules

### A and C

1. Among cards still in hand, minimize the number of that color already in the public INFORMATION display.
2. Among those ties, minimize the number of remaining missions in which that color could participate.
3. Among remaining ties, choose a hand-card index uniformly at random.

Private expected value, collection size and mission progress do not override these tiers. This preserves the user’s exact Chart-A ordering. Mission participation counts each mission once, regardless of required copies, and supports all wildcard patterns. The reveal callback happens after receiving the award but before automatic mission claims: missions already satisfied by that collection are about to be claimed and are omitted. Collections, auction-row cards and private cards never inflate public display counts.

### B and D

Prefer the highest expected-final-value color that still has possible future auction copies. Intrinsic value remains the primary factor, ahead of missions. For intrinsic-value ties, prefer more remaining missions involving that color, then greater collection-based progress toward completing them, then more expected future auction copies. Randomize final ties. Existing collections matter through mission progress, not through an invented gain from revealing owned colors.

Expected prices use the same full hidden-count distribution and active chart as gem bidding, including our own remaining hand and excluding auctioned gems from the final display. Expected future supply is the unsold row plus each unseen color’s expected deck copies. A color that cannot be bought again is not treated as a future purchase to protect.

### E and other non-monotonic curves

For each candidate color:

```text
delta = chart(public_count + 1) - chart(public_count)
market_average = unweighted mean of expected final values for colors still auctionable
exposure = expected_future_copies / (1 + expected_future_copies)
attention_gain = delta * (market_average - our_expected_final_value) * exposure
```

Maximize attention_gain. A positive delta can heat a relatively unimportant color; a negative delta can cool a preferred color. Zero price movement can be safer than advertising a desired target. On gain ties, raising quotes favors fewer missions/weaker progress; lowering quotes favors more missions/stronger progress. Randomize remaining ties. If no future auction supply exists, E has no information-driven preference. Private terminal values are averaged over their FULL distributions, not evaluated at a mean count; delta is the deterministic next public-quote movement, a different quantity.

On Chart E, revealing from three to four public copies changes the displayed quote from 18 to 6. The bot also checks its private final-value estimate; passing the public peak is not automatically a useful reveal. Gain scores are rounded to ten decimals to avoid numerical noise deciding otherwise tied candidates.

Final randomness uses the isolated worker’s seeded and checkpointed random stream. Duplicate tied cards remain separate eligible indices, matching a random choice among tied hand cards. A unique winner consumes no tie randomness. The harness skips the callback when the hand is empty.

## Paired benchmark

One candidate against only current level-2 ValuationBot opponents; 300 matched games per policy at each of 3, 4 and 5 seats, fresh seeds 3000–3299, all five charts. Total: 1,800 actual games. Initial deal, chart and candidate seat match across policies. Only reveal selection changes; bid algorithms, valuations, k, reserves and finance policy stay fixed. Later bids can respond differently to the different public information. All other level-3 extras are off.

Win credits belong to the single candidate, not the combined opponents; ties split credit, including thirds. All callbacks ran in isolated workers with the two-second allowance. No callback warnings, errors or timeouts occurred.

| Players | First-card win credit / 300 | Strategic win credit / 300 | First/strategic mean points | Paired margin change [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| 3 | 101.83 | 94.83 | 67.35 / 66.80 | -0.46 [-2.31, +1.38] |
| 4 | 82.00 | 82.50 | 47.44 / 47.32 | +0.43 [-0.96, +1.81] |
| 5 | 59.00 | 58.50 | 38.32 / 38.46 | +0.01 [-1.01, +1.02] |

| Players | Paired win-credit rate change, percentage points [95% interval] |
| --- | ---: |
| 3 | -2.33 [-7.58, +2.92] |
| 4 | +0.17 [-3.61, +3.94] |
| 5 | -0.17 [-2.80, +2.47] |

**No clear overall gain was demonstrated.** Every aggregate margin and win-credit change interval includes zero. The feature stays off and optional, consistent with the level-3 testing setup. There was no tuning or chart-specific adoption after these results.

## Exploratory breakdown by chart

Each row has 60 games per policy. These smaller subgroups are exploratory and not adjusted for multiple comparisons.

| Players | Chart | First-card win credit | Strategic win credit | Paired margin change [95% interval] |
| --- | --- | ---: | ---: | ---: |
| 3 | A | 21.00 | 19.50 | +0.15 [-4.21, +4.51] |
| 3 | B | 28.00 | 23.50 | -2.13 [-6.41, +2.14] |
| 3 | C | 14.00 | 18.00 | +1.35 [-1.97, +4.67] |
| 3 | D | 16.33 | 17.33 | +1.65 [-3.21, +6.51] |
| 3 | E | 22.50 | 16.50 | -3.33 [-6.97, +0.31] |
| 4 | A | 15.00 | 13.00 | -0.18 [-2.22, +1.85] |
| 4 | B | 15.00 | 20.00 | +3.35 [+0.65, +6.05] |
| 4 | C | 17.00 | 14.50 | +0.30 [-2.96, +3.56] |
| 4 | D | 20.00 | 15.50 | -2.33 [-6.29, +1.63] |
| 4 | E | 15.00 | 19.50 | +1.00 [-2.11, +4.11] |
| 5 | A | 14.00 | 14.50 | +1.58 [-1.30, +4.47] |
| 5 | B | 13.00 | 12.00 | +0.43 [-1.61, +2.48] |
| 5 | C | 15.00 | 16.00 | +1.42 [-0.48, +3.31] |
| 5 | D | 9.00 | 9.00 | -1.48 [-3.66, +0.69] |
| 5 | E | 8.00 | 7.00 | -1.92 [-4.07, +0.24] |

Intervals are approximate paired 95% mean ± 1.96 standard errors across seed differences. Margin is candidate score minus the strongest rival score. Different player-count blocks reuse seed labels and should not be pooled as independent samples. Charts and seats are individually balanced; the existing seed%seats/seed%5 rotation links candidate seat and chart at five players. No claim of universal superiority or of a different opponent population is made.

## Validation and reproduction

129 tests pass. Focused checks establish the exact A tiers, random eligible indices including duplicates, display/collection separation, already-satisfied and wildcard missions, C’s ascending order, B/D’s cooling preferences and mission progress, both directions around E’s peak, neutral reveals, unchanged terminal prices, preserved collections, unchanged bid algorithms, checkpoint/reset behavior and callback runtime. Isolated-worker replays cover all charts at 3–5 seats and repeat with the same seed. Browser checks cover the per-seat toggle, other extras staying off, replay labels, final scoring and mobile layout. An additional 64 actual games exercise every modern feature combination at 3 and 5 seats without worker warnings.

```sh
python3 experiments/benchmark_reveals.py --games 300 --start-seed 3000 --jobs 3 --output experiments/results/strategic_reveals/comparison.json
python3 experiments/valuation_experiment.py --candidate soft --strategic-reveal --games 120
npm test
npm run dev
```

For exact benchmark reproduction, run the full ordered range: each runner starts with private seed equal to its initial runner index and its private RNG continues across games. The game setup seeds alone do not encode the evolving private tie-randomness stream. Viewer replays initialize fresh seeded workers and are reproducible for the same complete viewer configuration.

Inspection: `assess_reveals(info, gem_values, deck_fractions)` returns each candidate’s public count, mission count/progress, expected final value, future supply, displayed-price change and priority tuple. Historical ablation/utility/cash-fraction controls explicitly keep this feature off. Existing private worker import paths are preserved when it is off. No additional public bot was submitted. All changes remain local and uncommitted.
