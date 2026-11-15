# Mixed-opponent tournament protocol

Session starts 2026-11-15 19:50:05 UTC; all work/testing stops within twenty
minutes. Screen deadline 20:02:15 UTC; validation deadline no later than
20:08:05 UTC, leaving time for final analysis. Save each completed game. Check
absolute deadline before each callback, reject callback errors, and discard any
unfinished game. Every normal callback has a two-second hard limit. L4 retains
its production 0.85-second allowance, 16-world maximum and two-world minimum.

Screen four seats: nine unique candidates × seven opponent lineups × five charts
× four deals = 1,260 games. L2 is also L3-all-extras-off, so do not duplicate it.
Test every combination of I=inference (weight 0.1), F=finance horizon and
R=strategic reveal; plus uniform-prior L4. Endgame restrictions and cash-fraction
bidding are excluded because previous four-seat studies showed clear harm.
Shared L2 expected-utility bidding and cash limits remain enabled.

Opponent lineups: all L2; I/F/R L3 mix; all random; all aggressive; all passive;
random/aggressive/passive mix; L2/L3-IFR/L4 mix. All auction/deck rules are real
harness rules. No default, small-bid or fixed-rule players enter these games.
Hypothetical opponents use chart-aware valuations but different spending
fractions/reserves, as documented in their source. They are stress policies,
not empirical human bid distributions.

Candidate seat rotates over all four seats independently of chart. Rotate
opponent orders. Match deal seeds, chart, seat and opponent ordering between
candidates; use independent worker RNGs. Charts/deals have distinct seeds. L4
still varies with wall-clock CPU availability. Four games per joint cell are
exploratory; inspect opponent/chart aggregates and report small-cell uncertainty.

Before examining screen scores: rank the generalist by the minimum of its
opponent-aggregate and chart-aggregate win rates, then overall win rate and
mean margin. Seek one opponent specialist and one chart specialist among other
candidates, ranking their improvement over the generalist in the relevant
aggregate. Require distinct candidates; if advantages are weak, label nominations
provisional rather than claim universal dominance. Freeze the nominations before
running fresh validation deals. Validate the nominees and L2 control on the same
matrix as time allows, then add 3/5-seat checks if the time budget permits.

Primary metric is single-candidate fractional win credit; secondary metrics are
own points and margin against the strongest opponent. Opponent wins are never
combined and compared with one candidate. Report paired per-deal differences
with approximate intervals for validation, along with exact game counts. A
single sweep cannot establish dominance over every possible opponent policy.

Adoption clarification, before inspecting validation outcomes: compare each
specialist with the included L2 control as well as the generalist. If a specialist
has identical per-deal win credits to L2 and less than one mean point of score
difference in its target validation subset, prefer simpler L2 for deployment;
still report the originally frozen finalist and its results. No feature should
receive credit for a contextual advantage it does not add over L2.
