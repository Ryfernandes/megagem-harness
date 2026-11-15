"""Compare the valuation policy to simple baselines with rotating seats/charts."""

import argparse
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bot_definitions.soft_tracking_bot import (  # noqa: E402
    EVIDENCE_WEIGHTS,
    FEATURE_NAMES,
    WEIGHTED_ABLATION_BOTS,
    WEIGHTED_LEVEL3_BOTS,
)
from bot_definitions.valuation_bot import FractionalValuationBot  # noqa: E402
from bot_runner import BotRunner  # noqa: E402
from custom_bots import (  # noqa: E402
    FixedRuleBot,
    SmallBidBot,
    SoftTrackingBot,
    ValuationBot,
)
from decks import VALUE_CHARTS  # noqa: E402
from harness import play_game  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument(
        "--candidate", choices=("valuation", "soft"), default="valuation"
    )
    parser.add_argument(
        "--bid-evidence",
        "--inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use bid-based hand updates (default: selected production setting)",
    )
    for feature in ("endgame-rules", "cash-limits", "bid-model"):
        parser.add_argument(
            f"--{feature}", action=argparse.BooleanOptionalAction, default=None
        )
    parser.add_argument(
        "--finance-horizon", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--cash-fraction-model", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--strategic-reveal", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--legacy-ablation",
        action="store_true",
        help="Reproduce the former fractional-baseline ablation policy",
    )
    parser.add_argument(
        "--evidence-weight",
        type=float,
        choices=EVIDENCE_WEIGHTS,
        default=None,
        help="Posterior evidence weight (default 0.1, legacy 0.4)",
    )
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    if args.start_seed < 0:
        parser.error("--start-seed must be nonnegative")
    requested = (
        args.bid_evidence,
        args.endgame_rules,
        args.cash_limits,
        args.bid_model,
    )
    if (
        any(value is not None for value in requested)
        or args.evidence_weight is not None
        or args.finance_horizon is not None
        or args.cash_fraction_model is not None
        or args.strategic_reveal is not None
        or args.legacy_ablation
    ) and args.candidate != "soft":
        parser.error("Feature options require --candidate soft")
    if not args.legacy_ablation and (
        args.cash_limits is not None or args.bid_model is not None
    ):
        parser.error(
            "Cash/bid-model overrides require --legacy-ablation; "
            "both are now level-2 defaults"
        )
    if args.legacy_ablation and (
        args.finance_horizon or args.cash_fraction_model or args.strategic_reveal
    ):
        parser.error("New extras require the current level-2 baseline")
    defaults = (
        SoftTrackingBot.use_bid_evidence,
        SoftTrackingBot.use_endgame_rules,
        SoftTrackingBot.use_cash_limits,
        SoftTrackingBot.use_bid_model,
    )
    if args.legacy_ablation:
        defaults = (False, False, False, False)
    flags = tuple(
        default if value is None else value
        for value, default in zip(requested, defaults)
    )
    name = "+".join(
        feature for feature, enabled in zip(FEATURE_NAMES, flags) if enabled
    )
    weight = (
        (0.4 if args.legacy_ablation else 0.1)
        if args.evidence_weight is None
        else args.evidence_weight
    )
    if args.legacy_ablation:
        soft_class = WEIGHTED_ABLATION_BOTS[(name or "baseline", weight)]
    else:
        soft_class = WEIGHTED_LEVEL3_BOTS[
            (
                (
                    flags[0],
                    flags[1],
                    bool(args.finance_horizon),
                    bool(args.cash_fraction_model),
                    bool(args.strategic_reveal),
                ),
                weight,
            )
        ]
    rival = FractionalValuationBot if args.legacy_ablation else ValuationBot
    classes = (
        (soft_class, rival, FixedRuleBot)
        if args.candidate == "soft"
        else (ValuationBot, FixedRuleBot, SmallBidBot)
    )
    wins: Counter[str] = Counter()
    points: Counter[str] = Counter()
    with ExitStack() as stack:
        runners = [
            stack.enter_context(BotRunner(cls, timeout=2.0, seed=i))
            for i, cls in enumerate(classes)
        ]
        for seed in range(args.start_seed, args.start_seed + args.games):
            offset = seed % len(runners)
            seats = runners[offset:] + runners[:offset]
            result = play_game(
                3,
                bots=seats,
                rng=Random(seed),
                value_chart=VALUE_CHARTS[seed % len(VALUE_CHARTS)],
            )
            for seat, score in enumerate(result.scores):
                points[seats[seat].bot_class.__name__] += score
            for seat in result.winner_ids:
                wins[seats[seat].bot_class.__name__] += 1
    print(
        f"{args.games} games; charts A–E; rotating seats; tied winners each get a win"
    )
    if args.candidate == "soft":
        print("Level-3 features:", dict(zip(FEATURE_NAMES, flags)))
        print("Evidence weight:", weight)
        print("Finance horizon:", soft_class.use_finance_horizon)
        print("Cash-fraction model:", soft_class.use_cash_fraction_model)
        print("Strategic reveal:", soft_class.use_strategic_reveal)
        print(
            "Utility k (scale, floor, legacy cap):",
            (
                soft_class.utility_k_scale,
                soft_class.utility_k_floor,
                soft_class.utility_legacy_cap,
            ),
        )
    for cls in classes:
        label = SoftTrackingBot.__name__ if cls is soft_class else cls.__name__
        print(
            f"{label:16} {wins[cls.__name__]:4} wins; "
            f"{points[cls.__name__] / args.games:6.2f} mean points"
        )


if __name__ == "__main__":
    main()
