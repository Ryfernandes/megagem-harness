# Python bot language and callback-budget audit

Historical development-catalog audit: the submission API now contains only
`SamplingBot` and `ValuationBot`. The other bots below remain local development
controls. The measurements are unchanged; `check_callback_budget.py` follows
the current submission `__all__` when rerun.

All five public custom bots are Python classes in `src/bot_definitions`, exported
by `src/custom_bots.py`: FixedRuleBot, SmallBidBot, ValuationBot, SoftTrackingBot
and SamplingBot. The default zero-bidding `Bot` is also Python (`src/bots.py`).
JavaScript implements the browser interface and Node server. The server launches
`python3 src/simulation.py`; it does not execute JavaScript bidding policies.
No bot language conversion or production runtime change was necessary.

## Actual worker timing

Run 25 complete games using the normal isolated Python bot workers with a
**2.0-second timeout per callback**. Every custom bot appears on each of charts
A–E at every supported seat count (3, 4 and 5). Pad shorter lineups with the
fixed-rule bot; these are timing checks, not competitive strength estimates.

Measure parent-side wall time around each actual BotRunner callback, including
argument transport, response handling and checkpointing. Process startup has a
separate allowance and is excluded from callback timing. Any callback error or
measured duration >=2 seconds invalidates the audit.

| Bot | Callbacks | Max callback seconds | Max bid seconds |
| --- | ---: | ---: | ---: |
| FixedRuleBot | 1348 | 0.002612 | 0.002612 |
| SmallBidBot | 533 | 0.002452 | 0.002383 |
| ValuationBot | 557 | 0.012449 | 0.012449 |
| SoftTrackingBot | 580 | 0.012972 | 0.012972 |
| SamplingBot | 613 | 0.851643 | 0.851643 |

All **3,631 callbacks passed**, with no callback errors or hard timeouts.
These measurements describe this run and hardware; the independent worker timeout
continues to enforce the callback budget. The browser sets that limit to two
seconds. The standalone harness/runner defaults to a stricter one-second limit;
explicit caller/CLI timeout overrides remain available for experimentation.

## L4 cutoff and level-2 fallback

Public L4 already uses a **0.85-second default allowance** for baseline calculation
and sampled search together. It runs cooperative deadline checks plus an
interruptible Unix timer inside the bot worker, disarming/restoring that timer
before checkpointing. Constructor allowances are bounded at 1.35 seconds; when
the host supplies a shorter budget, L4 reserves 0.30 seconds for returning its
response. The host independently enforces the callback limit.

L4 computes its ordinary level-2 (`ValuationBot`) bid before searching. Fewer than
**two complete comparison worlds** returns that bid. All candidate actions must
finish before a world counts; incomplete comparisons are discarded. A separate
emergency zero-bid path handles an interrupted baseline itself, preserving the
host deadline; it did not occur in this audit.

New regression coverage deliberately makes a continuation sleep for ten seconds,
with a configured 0.2-second allowance. For both zero and one completed world,
search is interrupted, returns the exact level-2 bid within two seconds, records
the fallback, and leaves no armed timer. Existing tests cover timer cleanup,
checkpointing, partial comparisons and real worker games.

The full suite has **145 passing tests**; Ruff and strict mypy also pass.

## Reproduce

```sh
python3 experiments/check_callback_budget.py --output experiments/results/callback_budget/audit.json
npm test
```

[Raw callback timings and game coverage](audit.json).
