# Megagem harness

A standard-library-only simulator for 3–5 players, using Python 3.10 or newer.

All bot strategies are written in Python in `src/bot_definitions`. JavaScript
provides the browser interface and local server, which launches the Python engine.
See the [callback timing audit](experiments/results/callback_budget/report.md)
for measured worker timings and L4 cutoff/fallback checks.

## Sprint submissions

Only these two bots are included in `src/custom_bots.py::__all__`:

| Submission | Class / CLI name | Definition |
| --- | --- | --- |
| L4 generalist | `custom_bots:SamplingBot` | [sampling_bot.py](src/bot_definitions/sampling_bot.py) |
| L2 passive-opponent specialist | `custom_bots:ValuationBot` | [valuation_bot.py](src/bot_definitions/valuation_bot.py) |

Both definitions contain module documentation explaining their strategies and
listing Grace Guo (85375651b) and OpenAI Codex (implementation assistance).
The L2 specialist is the normal valuation policy used **against** passive
opponents; `PassiveBidder` is a synthetic test opponent, not that submission.

All other local bots are listed separately in `custom_bots.EXPERIMENTAL_BOTS`.
They remain available in the browser and through existing explicit CLI imports,
but are excluded from the submission `__all__`. Internal variants in `experiments`
and rollout/helper classes are controls, not extra submissions. Add a name to
the submission list only when intentionally submitting that bot.

## Browser simulation viewer

Requires Node.js 18+ and Python 3.10+ on your PATH. No third-party Python or
JavaScript runtime dependencies are needed.

```sh
npm install
npm run dev
```

Open **http://localhost:5173**. Select 3–5 seats and a bot for each seat, choose
chart A–E and a game seed, then click **Start a new game**. Use Play/Pause,
previous/next, the timeline slider, and playback speed to inspect the game.
The initial default game is loaded automatically. Setup changes take effect on
the next new game. The journal records bids, reveals, and mission claims;
the last frame reveals all hands and shows each player's score breakdown.
Live market quotes are provisional until that final reveal.

**Peek at hidden hands** is a spectator feature only: bots still receive the
harness's normal private information snapshots. Games run in the existing Python
engine with isolated bot workers and a two-second timeout per callback. The
browser plays back the resulting snapshots; it does not implement a second game
engine. The same seed, seating, and bot selection reproduce a game for bots
using the harness-provided random stream.
Sampling search also has a wall-clock budget, so its choices can vary with CPU
load even when the setup seed is the same.

The selector includes `Bot` (zero bids), both submission bots, and the local
opponents in `custom_bots.EXPERIMENTAL_BOTS`, including `SmallBidBot` (one coin
on all auctions). Restart the server
and refresh the page after adding a bot. `PORT` changes the local server port;
`PYTHON` selects another Python executable. The server binds to loopback only.
Selecting Soft tracking reveals five per-seat feature checkboxes. All start
off; enabled features are recorded in the replay and shown on that player's card.

### Level 4: sampling search

Choose **Sampling search · level 4** in any seat. It is a separate `SamplingBot`
in `src/bot_definitions`, exported by `src/custom_bots.py`. It uses uniform world
weights and requires no level-3 tracker or experimental features. The
unconfigured level-3 bot still matches level 2.

The bot draws unseen gems without replacement into opponents' remaining hands
and the hidden deck, preserving every public color count and hand/deck capacity.
Its own hand and auction row stay fixed. It independently shuffles the known
remaining auction-card multiset. Worlds are drawn from the conditional prior and
receive equal weight. Bid history does not shift hidden-hand probabilities.

For every candidate bid, it simulates the rest of the game using current level-2
policies for **all** players. Each simulated policy receives only its own private
hand and the normal immutable public snapshot. Actual past bid learning is
reconstructed separately for each sampled opponent's hand. The real harness
resolves auctions, tie priority, reveals, missions, investments, loans and final
scoring. Rollouts run in-process inside the existing isolated bot worker and
never spawn nested workers.

The action maximizes the ordinary mean of the bot's **own final coin score**,
rather than rollout win rate. Initial candidate bids include the level-2
bid, zero, and bids near sampled rival thresholds, with at most seven choices.
Small budgets initially cover every legal bid; cost adaptation may narrow that
set, always retaining the level-2 action. Search can spend the current reserve
when a full continuation justifies it. The same sampled worlds and tie randomness
compare all actions. An incomplete comparison world is discarded for every action.

Search uses up to 16 completed worlds and a default 0.85-second allowance,
below the hard two-second callback limit. Unix workers also use an interruptible
timer, disarmed before checkpointing; other environments use cooperative checks.
A pilot continuation adapts the number of candidates to cost; it never narrows
actions using the pilot reward. At least two completed worlds are required,
otherwise the ordinary level-2 bid is used and the fallback is recorded.
If even computing the baseline exceeds the timer,
the emergency response is a legal zero bid, recorded separately. Cashless
non-loan auctions have only one legal bid and skip search. Preparing a table with
several sampling bots takes longer.

Only repeated pure level-2 computations are memoized for speed; bidding formulas
and numeric results are unchanged. The [world-weight ablation](experiments/results/world_weights/report.md)
compares uniform and weighted L4; the [original benchmark](experiments/results/sampling_search/report.md)
compares the previous weighted version with level 2. Direct callers can configure
`SamplingBot(decision_seconds=0.85, max_worlds=16, max_candidates=7)`; the viewer
uses these defaults. Inspect `last_search` or `search_log` for completed-world
counts, candidate mean scores and fallback reasons.

```sh
python3 src/harness.py 3 --bots custom_bots:SamplingBot custom_bots:ValuationBot custom_bots:ValuationBot --seed 42 --bot-seed 42
python3 experiments/benchmark_sampling.py --games 60 --start-seed 3300 --jobs 6 --batch-size 10 --output experiments/results/world_weights/uniform_vs_level2.json
python3 experiments/compare_world_weights.py --games 60 --start-seed 3600 --jobs 6 --batch-size 10 --output experiments/results/world_weights/comparison.json
```

Limited sample counts, a reduced action set and the assumption of level-2
opponents make this an approximate policy search. The
benchmark determines its measured performance; no universal superiority is
assumed.

L4's former evidence weighting and tracker were removed after the matched
ablation found no useful advantage. The private `WeightedSamplingReference`
exists only in `experiments` to reproduce comparisons. Add `--weighted-reference`
to `benchmark_sampling.py` to reproduce the original weighted-versus-L2 protocol.

### Mixed-opponent tournament

The selector now includes **Random · value-bounded**, **Very aggressive · high
bids**, and **Very passive · low bids**. They use the active chart and mission
values, with different spending fractions and cash reserves. These are synthetic
stress opponents rather than models fitted to human players.

The [mixed-opponent report](experiments/results/mixed_tournament/report.md)
compares L2, all seven combinations of the inconclusive L3 inference/finance/
reveal features, and L4 across every chart and seven opponent lineups. It includes
fresh validation, specialist nominations, complete matrix CSVs and raw games.
The worst historically harmful features and the default/small/fixed bidders are
excluded from this tournament. Existing production defaults are unchanged.

Candidate aliases are in `experiments/mixed_policies.py`; the time-bounded runner
is `experiments/mixed_tournament.py`. Set `--deadline` to an absolute future ISO
UTC time. The saved protocol documents selection and validation rules.

### Fixed-rule baseline

`FixedRuleBot` bids **floor(current coins / 5)** on treasure and zero on loans
and investments. It reveals the first hidden gem. This deliberately simple
policy ignores valuation, mission bonuses, and opponents' actions, making it a
useful baseline for future strategies. With fewer than five coins, its treasure
bid is zero. All-zero auctions still have a winner, so it can acquire loans and
investments for free under the ordinary tie rules.

Missions are **automatically claimed without payment or consuming gems** when a
collection satisfies their requirements. There is no separate mission purchase
decision in Megagem; this bot uses the standard mission resolution.

```sh
python3 src/harness.py 4 --bots custom_bots:FixedRuleBot --seed 42 --bot-seed 42
npm test
```

`src/simulation.py` exports JSON replays for the viewer. `play_round(game)` now
returns its immutable `RoundResult`; existing callers may continue ignoring it.

### Valuation model

Choose **Valuation · expected value** in any seat, or run:

```sh
python3 src/harness.py 4 --bots custom_bots:ValuationBot --seed 42 --bot-seed 42
python3 experiments/valuation_experiment.py --games 120
```

`ValuationBot` uses an exact hypergeometric distribution over unseen gem
locations to estimate final prices, conditioning on public information and its
own private hand. It averages the actual chart values, so it handles the curved
and non-monotonic charts without pricing a gem at an average display count.

The bot adds gem value and marginal mission progress to a denial bonus (30% of
the largest opponent mission gain the purchase prevents). Treasure bids now
maximize `p(win) × (E[value] − (1+k) × bid)` with all three, four and five seats,
using the same bidder and penalty selected in the expected-utility experiment.
The opponent bid model learns from public bids and respects cash and tie priority.
The former policy bid 45% of value before liquidity adjustments; it is preserved
as the private `FractionalValuationBot` control. Direct callers can use
`ValuationBot(use_expected_utility=False)` to restore those fractional bids.
Unfinished missions use a conservative best-completion estimate across all
matching wildcard patterns, discounted by remaining supply, purchasing capacity,
missing-gem acquisition costs, and competitors' progress. An immediately
completed mission gets its full actual reward. Missions reuse gems under the
rules, so the bot does not impose an artificial exclusivity cost.

The bot reserves 40% of cash while several treasure lots remain, tapering toward
the final purchase. It retains at least one coin before the last treasure lot.
Bids are further reduced when future purchases make liquid cash especially
valuable. Investments charge only the opportunity cost of locking up the bid,
since the bid itself is returned; the bonus is the profit. Loans charge their
bid against the expected benefit of usable additional cash, with principal and
repayment cancelling out. Loan bids are netted against the proceeds and remain
below the loan face value. Opportunity estimates shrink as the remaining
treasure auctions disappear. As with all bots, bidding zero can still win a
financial card.

This is a deterministic valuation heuristic with public-bid learning, without
hidden-hand inference or rollout search. Gem EV is exact under a uniform
unseen-card assumption; mission completion, competitive discounts, and future
purchasing power are estimates.
The bot uses no third-party packages and reveals its first hidden gem.
Restart the dev server and refresh the browser if it was already running.

On the original validation set (`--games 120 --start-seed 120`), the former
fractional valuation bot
won 69 games versus 53 for the fixed-rule bot and zero for SmallBidBot; two
games had shared winners. Mean scores were 108.3, 100.1, and 39.7 respectively.
This checks the heuristic against simple baselines across all five charts and
rotating seats; it does not guarantee stronger play against other strategies.

The new bidder was compared with that frozen former policy on 300 fresh games
per seat count, with one candidate against former-valuation opponents. Win
credit rose from 121.5 to 141.5 (three players), 58 to 82 (four), and 52.5 to 68
(five). Ties split credit. Mean score rose only with three players; the strongest
evidence for an improved score margin is at three seats. The [full comparison](experiments/results/valuation_expected_utility/report.md)
contains confidence intervals, per-game scores, and reproduction commands.

### Level-3 feature ablations

**SoftTrackingBot now matches the current ValuationBot at all seat counts.**
Its five extras—hand inference, mission endgame rules, finance-aware k, the
cash-fraction bid model and strategic reveals—are
disabled by default. The old four/five-player fractional fallback has been
removed to align the level-3 baseline with the upgraded level-2 bidder.
Select **Soft tracking · informed bids** and use the per-seat checkboxes to
enable individual extras. Cash-bounded expected-utility bidding and public-bid
learning are shared level-2 behavior, not enabled level-3 extras.

Each extension has a constructor flag, preserved across game resets and worker
checkpoints:

| Extension | Flag | Default |
| --- | --- | --- |
| Bid-based hand inference | `use_bid_evidence` | `False` |
| Mission-focused endgame overrides | `use_endgame_rules` | `False` |
| Finance-aware k | `use_finance_horizon` | `False` |
| Type-specific cash-fraction bid model | `use_cash_fraction_model` | `False` |
| Chart-aware information reveal policy | `use_strategic_reveal` | `False` |
| Shared level-2 cash conditioning | `use_cash_limits` | `True` |
| Shared level-2 bid model / surplus optimizer | `use_bid_model` | `True`, all seats |

The last two switches remain available for historical ablations. Cash
conditioning only affects the bid predictor. **No flag disables legal affordability
checks**, loan funding rules, or level 2's own-cash reserve/max-rival spending cap.

The current public default runs with `--candidate soft` alone. Modern CLI
switches keep the current level-2 baseline:

```sh
python3 experiments/valuation_experiment.py --candidate soft --finance-horizon --games 120
python3 experiments/valuation_experiment.py --candidate soft --cash-fraction-model --games 120
python3 experiments/valuation_experiment.py --candidate soft --strategic-reveal --games 120
python3 experiments/valuation_experiment.py --candidate soft --inference --evidence-weight .1 --games 120
python3 experiments/valuation_experiment.py --candidate soft --legacy-ablation --inference --no-endgame-rules --no-cash-limits --no-bid-model --games 120
```

`--bid-evidence` remains an alias for `--inference`. Direct Python callers can
use `SoftTrackingBot(use_finance_horizon=True)` or
`SoftTrackingBot(use_bid_evidence=True)` or
`SoftTrackingBot(use_cash_fraction_model=True)` or
`SoftTrackingBot(use_strategic_reveal=True)`. Enabling one does not enable the others.
The `--legacy-ablation` option explicitly restores the former fractional control,
penalties and bid cap. Refresh the viewer to use the selected default on the next game.

When enabled, inference uses a joint allocation model that enforces all known
color totals, remaining hand sizes, and deck size. The aggregate probability is
`bot.beliefs(info).opponent_hand_probabilities[color]`: an unseen copy's chance of
being in any opponent's remaining hand rather than the deck. Pricing integrates
the full count distribution over the current Value Chart:

```text
final display = public starting-hand reveals
              + our remaining starting hand
              + opponents' remaining starting hands
```

Auction purchases and deck gems never contribute to display counts. Chosen
reveals and bid likelihoods still require behavioral assumptions; inference is
experimental rather than assumed to improve results.

The historical [ablation report](experiments/results/level3_ablation.md) includes all 16
configurations, per-game data, confidence intervals, selection rules, and the
adoption decision. Each game has one candidate and two former ValuationBot
opponents, now pinned to `FractionalValuationBot` for reproduction.

| Single feature on fresh validation deals | Win credit / 240 |
| --- | ---: |
| Level-2 control (all off) | 76.5 |
| Inference alone | 75.5 |
| Endgame rules alone | 59.5 |
| Cash conditioning alone | 76.5 |
| Bid optimizer alone | 54.5 |

Ties split win credit. Inference's small selection gain did not replicate;
endgame rules and the original optimizer hurt. Cash conditioning alone is redundant.
These results cover this three-player lineup, not every possible tournament.

```sh
python3 experiments/ablate_level3.py --games 240 --start-seed 360 --jobs 3 --output experiments/results/level3_selection.json
python3 experiments/ablate_level3.py --games 240 --start-seed 600 --jobs 3 --variants baseline inference endgame cash_limits bid_model --validate-variant inference --output experiments/results/level3_single_feature_holdout.json
```

### Expected-utility bidding

Treasure bids maximize `p(win) × (E[value] − (1+k) × bid)` over affordable
integer bids after keeping the normal reserve. Opponent bid distributions learn
from past public bids, respect current cash, and handle actual tie priority.
The old experimental optimizer already used this formula. The follow-up removes
its 95%-of-value bid cap and calibrates `k`: the existing liquidity premium plus
`0.5 × L/(L+1.5)`, where `L` estimates future treasure lots. That added future-cash
premium expires at the final treasure auction.

After selecting among eight settings on 240 deals, fresh matched validation
compared one candidate against only ValuationBot opponents, across all charts
and rotating seats. Ties split credit; the numbers belong to the single
candidate in separate before/after runs:

| Players | Games per policy | Before win credit | EU win credit |
| --- | ---: | ---: | ---: |
| 3 | 300 | 113.5 | 147.0 |
| 4 | 180 | 45.0 | 52.5 |
| 5 | 180 | 43.0 | 41.5 |

Three-player margin against the strongest rival improved by **6.26 points**
(approximate paired 95% interval **+3.54 to +8.97**). Four/five-player margin
intervals include zero, so the original soft-tracking adoption covered only three
players. The later level-2 upgrade and current baseline alignment use EU at all
seat counts. Valuation,
missions, denial, finance bids, reserve and reveal policy remain fixed;
inference and mission-only endgame rules stay off.

`use_bid_model=False` restores fractional bidding. All seat counts now use EU by
default. Parameters `utility_k_scale`
and `utility_k_floor` configure the penalty and survive worker checkpoints.
Private experiment classes pin all settings, preserving historical results.
See the [soft-tracking report and reproduction commands](experiments/results/expected_utility/report.md)
and [per-game results](experiments/results/expected_utility/validation_3_seats.json).
The later ValuationBot adoption has its own [fresh before/after comparison](experiments/results/valuation_expected_utility/report.md).

### Optional finance-aware k

`use_finance_horizon=True` changes only the cash penalty in treasure bidding.
A finite-deck calculation integrates all possible remaining one-/two-gem orders
to find the exact treasure stopping-time distribution. It uses the remaining
loan/investment counts to obtain tight minimum/maximum total-round bounds and
exact expected rounds and financial cards before the final gem sale. Cards that
would appear after game end are excluded.

Funding is a heuristic: anticipated loans retain half their principal after
acquisition costs and are split equally across players; predicted investment
bids reduce future liquidity. Finance cards are weighted by how many treasure
opportunities follow them, so late funds matter less. Neither forecast creates
actual coins, guarantees a loan, changes gem values or alters the reserve.
The horizon combines expected treasure auctions with expected total rounds.
At the final treasure lot, k remains zero.

The feature stays **off by default**. The [paired benchmark](experiments/results/finance_horizon/report.md)
compares it on/off against the **current** level-2 opponents across 3–5 seats and
all five charts. Reproduce it with:

```sh
python3 experiments/benchmark_finance_horizon.py --games 300 --start-seed 2400 --jobs 3 --output experiments/results/finance_horizon/comparison.json
```

### Optional cash-fraction opponent bid model

`use_cash_fraction_model=True` replaces the opponent predictor with a separate
cash-fraction history for every opponent and auction type. Observations record
validated bids and **pre-auction** cash. Loans use cash plus their proceeds as
the available budget, so a cashless borrower still has a valid fraction. Forced
zero-cash passes are logged without learning a bidding preference.

Each prediction mixes two broad normal-prior pseudo-observations with the
empirical distribution of past fractions scaled to current cash. Treasure
observations count one-quarter as much as investment/loan observations because
lot values differ. The PMF has no mass above legal cash. `p(win with c)` multiplies
each rival's CDF, with the correct inclusive/strict comparison for tie order;
the first auction integrates uniformly over tied winners. It is recalculated
for each candidate bid, and the bot maximizes expected utility.

The enabled feature also applies EU to investments and loans using their own
histories. Investments return the bid, so profit charges only the lost liquidity;
loan principal offsets debt, so the value is productive funding rather than free
coins. The reserve, gem/mission/denial values, k and reveal policy are unchanged.
The [paired benchmark](experiments/results/cash_fraction_bids/report.md) includes
a treasure-only control to isolate the predictor change from financial bidding.
All level-3 extras remain off by default.
Against current level-2 opponents, the complete feature hurt score margin at
three/four seats and showed no clear gain at five. Treasure-only replacement
also hurt at every seat count, so the existing predictor remains the default.

```sh
python3 experiments/benchmark_cash_fraction_bids.py --games 300 --start-seed 2700 --jobs 3 --output experiments/results/cash_fraction_bids/comparison.json
python3 experiments/log_cash_fraction_bids.py --seats 4 --seed 2700 --chart E --output experiments/results/cash_fraction_bids/observations.csv
```

Select **Soft tracking · informed bids**, then check **Cash-fraction bid model**
for that seat. Replay labels record the features actually used. The sample CSV
contains raw bids, bankrolls, fractions, weights and pre-bid win probabilities.
Python callers can inspect `bot.cash_fraction_model.history` directly. The
bounded history is checkpointed and resets between games.

### Optional strategic reveals

Enable `use_strategic_reveal=True`, use CLI `--strategic-reveal`, or check
**Strategic reveals** for a Soft tracking seat in the viewer. It remains off by
default, so the unconfigured level-3 bot still matches current level 2.

Revealing a hand card changes opponents' information, not the final gem values:
all starting-hand cards count at scoring. On A/C a reveal raises the displayed
price; on B/D it lowers it. Revealing a low-value color to attract bids therefore
does not extend unchanged to descending charts.

| Chart | Reveal heuristic |
| --- | --- |
| A, C | Fewest public **display** copies; then fewest remaining missions involving the color; then random among tied hand cards. |
| B, D | Prefer the highest expected-value color still available to buy, trying to cool competition. Break value ties by more missions, stronger progress toward a mission, then more future auction copies. |
| E | Use the sign and size of the next displayed-price change and the full expected final gem value: try to heat less useful colors or cool preferred ones. A neutral reveal can be better than advertising a valuable target. |

Chart A keeps the exact requested order: private value, owned copies and mission
progress do not override the two specified tiers or the final random tie.
Each mission counts once, including wildcard missions where the color could
participate. Missions already completed by the just-awarded collection are about
to be claimed and are excluded. Scoring collections and auction-row cards never
inflate public display counts.

For B/D/E, expected final prices use the full hidden-count distribution.
Remaining row/deck supply prevents pretending a reveal will affect purchases of
a color that cannot be bought again. Existing collections matter through
mission progress; revealing never increases their final scoring value. E weighs
display-price movement by the difference between the color's expected price and
the unweighted average for colors that could still be auctioned, with a
saturating future-supply factor.
Mission tie-breaks favor less relevant colors when raising apparent prices and
more relevant colors when lowering them.

Final ties use the bot worker's seeded, checkpointed random stream. Duplicate
cards are separate eligible indices, matching a random choice among tied cards.
This is a perception heuristic: rivals estimating terminal values accurately
need not follow current display quotes, so drawing or cooling competition is an
intention rather than a guarantee.

The [paired benchmark](experiments/results/strategic_reveals/report.md) compares
this policy with first-card reveals, keeping all other extras off:

```sh
python3 experiments/benchmark_reveals.py --games 300 --start-seed 3000 --jobs 3 --output experiments/results/strategic_reveals/comparison.json
```

### Tracker accuracy without bidding influence

The tracker has also been evaluated as a **shadow observer**: it records a final
count forecast per color before each auction, learns bids/reveals only afterward,
and attaches actual counts only after final scoring reveals all starting hands.
Actual actions come from frozen former-valuation/fixed-rule workers; tracker
estimates never reach those workers. Historical tracker/ablation runners pin
`FractionalValuationBot` so upgrading the public bot does not change their deals.

```sh
python3 experiments/evaluate_tracker.py --games 300 --start-seed 840 --jobs 3 --output experiments/results/tracker_predictions --evidence-weight .4
```

At the legacy evidence strength 0.4, across 900 games on 300 fresh deals,
charts A–E, 3–5 seats, and three opponent
profiles, informative count MSE was **0.613738 for the tracker versus 0.615812
for the prior** (0.34% improvement). Intervals cluster on deal seed; known-count
rows are excluded from the primary comparison. Fixed-rule rivals showed no
clear gain. The prediction gate passed, but the prior strategy gate did not:
**inference remains disabled for actual bids**.

The [diagnostic report](experiments/results/tracker_predictions/report.md),
[example forecast/actual rows](experiments/results/tracker_predictions/examples.csv),
and compressed full logs record predictions and outcomes per round and color.
The tests verify that adding the observer leaves real bids, reveals, and scores
unchanged, and that no actual-count labels are present before final reveals.

`play_round(game, before_bid=callback, observer_id=seat)` provides the host-only
hook. It passes only that seat's ordinary immutable `BotInfo` after the auction
card is known but before any current bid is collected. The return is ignored.

### Shrinkage toward the prior

`SoftTrackingBot` now supports `evidence_weight=0.1`. The allocation posterior is
proportional to `prior * exp(evidence_weight * raw_evidence)`. Zero gives the
plain prior; 0.4 reproduces the previous tracker. Raw evidence decays and is
bounded, while known card facts remain exact. The weight is preserved across
games/checkpoints and participates in the belief-cache key.

```sh
python3 experiments/evaluate_tracker.py --evidence-weight .1 --games 300 --start-seed 1140 --output /tmp/tracker-weight-01
python3 experiments/valuation_experiment.py --candidate soft --inference --evidence-weight .1 --games 120
```

The [shrinkage report](experiments/results/tracker_shrinkage/report.md) compares
weights 0, 0.025, 0.05, 0.1, 0.2 and 0.4 on shared shadow trajectories. Selection
used 90 seeds; validation used 180 separate fresh seeds, each under three
opponent profiles. Validation count MSE was 0.610322 for the prior, 0.609936 at
weight 0.1, 0.609589 at 0.2, and 0.609011 at the legacy 0.4 strength.

Shrinking helped relative to the plain prior but did not beat the previous
tracker. **0.1 is a conservative experimental default, not a data-proven optimum.**
Inference remains disabled for production bids. Historical ablation workers
stay pinned to 0.4 unless an explicit weight is supplied in gameplay experiments.

## Command-line simulator

```sh
python3 src/harness.py 3 --seed 42 --bot-seed 7
python3 src/harness.py 5 --chart C --bots custom_bots:SmallBidBot
python3 src/harness.py 3 --bot-timeout none
```

Without a player-count argument, the CLI asks for one. Chart A is the default.
`--bots` accepts one `module:Class` for everyone or one per player, in seating order.
Every player gets a separate bot instance. The default `Bot` bids zero and reveals
a random hidden gem. `--bot-timeout` is a positive number of seconds or `none`;
it defaults to one second per callback.

## Writing a bot

Put each custom bot implementation in the `src/bot_definitions/` package, then expose
its class through `src/custom_bots.py`. Implementations can live in separate files:

```text
src/
    bot_definitions/
        __init__.py
        small_bid_bot.py
        my_bot.py
    custom_bots.py
```

For example, define your subclass in `src/bot_definitions/my_bot.py`:

```python
from collections.abc import Mapping
from bots import Bot, BotInfo, RoundResult


class MyBot(Bot):
    def __init__(self) -> None:
        self.games_played = 0
        self.rounds_seen = 0

    def on_game_start(self) -> None:
        self.games_played += 1
        self.rounds_seen = 0

    def get_bid(self, info: BotInfo) -> int:
        return min(2, info.self_player.coins)

    def choose_hidden_gem(self, info: BotInfo, bids: Mapping[int, int]) -> int:
        return 0  # Index into info.self_player.hidden_gems.

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        self.rounds_seen += 1
```

For a local test bot, add its import to `src/custom_bots.py` and register it in
the development catalog, after the existing mapping is defined:

```python
from bot_definitions.my_bot import MyBot

# Leave the two actual submission entries in __all__ unchanged.
EXPERIMENTAL_BOTS["MyBot"] = MyBot
```

Run it with `python3 src/harness.py 3 --bots custom_bots:MyBot`. Multiple implementations
can be selected independently, such as
`--bots custom_bots:MyBot custom_bots:SmallBidBot custom_bots:MyBot`.
Explicit imports make each bot name easy to find. Keep definitions in their own
modules; importing them into `src/custom_bots.py` makes them available to the CLI,
and the development mapping makes them selectable in the viewer. Update
`__all__` separately only when intentionally changing the submitted team.
Workers still import each class from its original `bot_definitions` module, which
also allows checkpoint recovery without any special registration code.

All five callbacks are optional and synchronous:

- `on_game_start() -> None`: called once before the first decision of each game.
- `get_bid(info: BotInfo) -> int`: return a nonnegative, affordable integer bid.
  A loan bid may include the proceeds of that loan.
- `choose_hidden_gem(info: BotInfo, bids: Mapping[int, int]) -> int`: return a valid
  index. `bids` contains validated bids keyed by player ID. Empty hands skip this call.
- `on_round_end(info: BotInfo, result: RoundResult) -> None`: called for every bot
  after the reveal, replenishment, and mission claims. Results include the auction
  card, validated bids, winner ID, winning bid, and revealed gem (possibly `None`).
- `on_game_end(info: BotInfo, result: GameSummary) -> None`: called once after all
  remaining hands are revealed and scoring finishes. Results contain scores keyed
  by player ID, winner IDs, and the winning score. No live game state is included.

`get_bid` defaults to zero, `choose_hidden_gem` defaults to a random valid index,
and the three notifications default to doing nothing. No decorators or async
functions are required. Constructors must accept no arguments. Use ordinary
instance attributes for memory; globals are not checkpointed.

## Information available to bots

`BotInfo` and everything nested within it are immutable snapshots. Count mappings
support normal mapping operations, including `.get()`, `.items()`, and `.values()`.

| Field | Meaning |
|---|---|
| `self_player` | ID, coins, invested coins, hidden gems and hand size, revealed gems, acquired `won_gems`, auction cards, and mission cards |
| `other_players` | Public player information and hidden-hand sizes, ordered by seat; no hidden gem identities |
| `num_players`, `round_number` | Player count and current round; setup is round zero |
| `current_auction` | The current round's auction card, or `None` before the first round |
| `gem_deck_count` | Number of gems in the hidden draw pile |
| `gems_remaining` | Hidden draw pile plus exposed auction gems |
| `unknown_gems` | Full gem catalog minus the display, acquired gems, exposed gems, and this bot's hidden hand; includes opponents' hidden hands and the hidden draw pile together |
| `auction_gems` | Exposed gems in auction order, with the first gem first |
| `value_display` | Counts of revealed starting-hand gems |
| `previous_winner_id` | Previous auction's winner when bidding; current winner after resolution |
| `value_chart`, `available_missions` | The selected chart and unclaimed missions |
| `auction_deck_count` | Number of undrawn auction cards |
| `auction_cards_remaining` | Unordered counts of undrawn auction cards, never their deck order |

Each bidder gets the same public state and their own private hand; current bids
are not disclosed until everyone has bid. The reveal decision sees the auction
award and payment already applied, before any gem reveal or replenishment.
`current_auction` remains the just-resolved card in result notifications.

## Timeouts and memory

Each bot lives in its own persistent spawned worker. It receives snapshots, never
`Player`, `GameState`, or the game's RNG. Only validated integer decisions reach
the game. Invalid bids become zero; invalid reveal choices select a random valid
index. Booleans and floats are invalid decisions. Exceptions and timeouts use the
same fallbacks and log the player and callback responsible.

After every successful callback, the worker checkpoints the bot's instance
attributes and its private `random` module state. On failure, its worker is stopped.
The next call starts a replacement from the last successful checkpoint, without
rerunning `__init__` or `on_game_start`. Changes from the failed callback are lost.
Successful callbacks returning invalid decisions still retain their memory updates.

Ordinary pickleable attributes in `self.__dict__` need no extra code. Snapshots can
themselves be saved in bot memory. Slot-backed fields need custom hooks. For
files, locks, clients, or other resources, override both
`save_state() -> object` and `load_state(state: object) -> None`: export pickleable
memory and recreate resources during restoration. Recovery calls `load_state` on
an instance allocated without running `__init__`. Checkpoints are opaque to the
game process and are unpickled only in a worker. They are in-memory checkpoints,
not saves that persist after the harness exits.

The callback budget includes snapshot transfer and checkpoint creation. Worker
startup/import/restoration has a separate budget of at least five seconds (or the
callback timeout if greater). With `none`, both limits are disabled. Bots can read
`self.timeout_seconds` for their callback allowance. Messages/checkpoints have a
16 MiB transport limit. Process cleanup can add a small delay after a timeout.

`--seed` controls game setup and tiebreaks. `--bot-seed` independently seeds bot
randomness. Use both to reproduce randomized bot games. `import random` inside a
bot uses its own worker's stream. Fallback reveal choices also use separate randomness.
A bot using its own additional random generators should seed and store them itself.

## Python API and reusing bots

Runtime modules live in `src/`. When running your own API script from the project
root, use `PYTHONPATH=src python3 your_script.py` so Python and spawned workers can
find them. The experiment script and notebook add `src/` automatically, so their
commands require no extra setup. No installation or virtual environment is needed.

```python
from random import Random
from bot_runner import BotRunner
from custom_bots import SmallBidBot
from harness import play_game


def main():
    with (
        BotRunner(SmallBidBot) as first,
        BotRunner(SmallBidBot) as second,
        BotRunner(SmallBidBot) as third,
    ):
        bots = [first, second, third]
        for seed in range(2):
            result = play_game(3, bots=bots, rng=Random(seed))
            print(result.scores, result.winner_ids)


if __name__ == "__main__":
    main()
```

A caller can pass bot classes instead, for example `bots=[SmallBidBot] * 3`.
The harness creates and closes those workers automatically. Caller-owned runners
retain memory between games until `on_game_start` selectively resets it. Each
runner's own timeout/seed configuration applies when supplying existing runners.
Separate players must use separate runners.

`play_game()` returns a `GameResult` without printing; use `print_result()` for
console output. Its `state` contains mutable final game state for host inspection.
Bot game-end notifications receive only the immutable `GameSummary`.

For round-by-round inspection, use `with create_game(...) as game`, call
`play_round(game)` while `game.auction_gems` is nonempty, and then `finish_game(game)`.
Repeated finalization neither reveals cards twice nor repeats game-end callbacks.
The main guard shown above is required for scripts that launch spawned workers.
Define custom bot classes at module scope inside `src/bot_definitions/` and expose them
through `src/custom_bots.py`. Avoid defining them inside functions or `__main__`.

## Experiment examples

The `experiments/` directory contains script and notebook examples that reuse bot runners across
games, print each game's results, and then report total points and games won per
entrant:

- [Python script](experiments/sample_experiment.py): run
  `python3 experiments/sample_experiment.py` from the project root.
- [Notebook using existing bots](experiments/sample_experiment.ipynb): the same workflow
  split into editable cells.

Run notebook cells from top to bottom in a Python 3.10+ Jupyter environment. Each
notebook finds the project whether its kernel starts in the root or `experiments/`.
Edit the game count, seeds, timeout, and participant list to change the experiment.
Labels identify individual entrants, seats stay fixed, and each tied winner
receives one win. Runners retain memory between games and are closed when the
experiment finishes, including when an exception interrupts it.

## Layout and checks

- `src/constants.py`, `src/decks.py`, `src/lib.py`: setup rules, immutable catalogs/cards, typed decks.
- `src/player.py`: mutable player state and bot decision validation.
- `src/bots.py`: the developer-facing bot interface, snapshots, and result types.
- `src/bot_info.py`: builds each player's information without inspecting hidden deck order.
- `src/bot_runner.py`: worker lifecycle, transport, deadlines, and memory recovery.
- `src/immutable.py`: a pickleable read-only mapping used by cards and snapshots.
- `src/harness.py`: setup, round resolution, scoring, and CLI.
- `src/bot_definitions/`: custom bot implementations, including `SmallBidBot` in
  `small_bid_bot.py`.
- `src/custom_bots.py`: imports and exposes the bot classes available to the CLI.

Tests require no third-party packages. Ruff and mypy are optional development tools:

```sh
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
ruff check .
ruff format --check .
mypy
```
