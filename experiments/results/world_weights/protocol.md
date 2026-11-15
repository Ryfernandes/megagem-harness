# World-weight ablation protocol

Compare weighted L4 (evidence weight 0.1) with uniform L4 (weight zero, inference
skipped) on fresh seeds 3600–3659. Each has one candidate against current L2
opponents, at three, four and five seats, for 360 actual games. Pair initial deal,
chart and candidate seat. Use unchanged search limits: 0.85 seconds, 16 worlds,
seven initial candidates, two completed worlds minimum. Ten-game batches use
fresh fixed-per-seat worker RNGs. Reject any run with callback warnings.

Compare own mean points (the search objective), single-candidate fractional win
credit, time per bid, completed worlds, round-end processing time and fallbacks.
Report paired approximate 95% intervals. Treat a loss of three mean points as a
practically relevant deterioration for this comparison; a uniform-minus-weighted
lower score bound above -3 supports dropping weights without that size of loss.
An interval spanning zero alone does not establish equivalence. Report intervals
and any inconclusive seat counts explicitly; do not silently tune parameters.

For each successful weighted decision, also calculate the unweighted optimum
using its exact same worlds, candidates and terminal outcomes. This needs no
additional rollouts and directly measures how often weighting changes the chosen
action. Prefer the lower bid for exact ties in both aggregations. Full-game
outcomes still include the practical effects of the wall-clock budget, and can
vary with CPU load. The identical-world diagnostic isolates aggregation effects.

The zero-weight candidate skips the level-3 posterior calculations in
on_round_end. It still uses the same public-bid history to reconstruct level-2
rollout policies; that history learning is separate from hidden-hand inference.
The experiment does not disable level-2 opponent modeling.
