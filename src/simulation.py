"""JSON replay bridge for the local viewer. Gameplay stays in the real harness.

Spectator snapshots include all hands; bots still receive only BotInfo snapshots.
No user-supplied module names or executable code are accepted by this bridge.
"""

import json
import sys
from random import Random
from typing import Any

import custom_bots
from bot_definitions.soft_tracking_bot import LEVEL3_BOTS, LEVEL3_FEATURES
from bots import Bot
from constants import STANDARD_GEMS, GemType
from decks import VALUE_CHARTS
from harness import create_game, finish_game, play_round
from lib import MissionCard


def bot_catalog() -> dict[str, type[Bot]]:
    return {
        "Bot": Bot,
        **{name: getattr(custom_bots, name) for name in custom_bots.__all__},
        **custom_bots.EXPERIMENTAL_BOTS,
    }


def mission_data(mission: MissionCard, index: int) -> dict[str, Any]:
    return {
        "id": index,
        "value": mission.value,
        "gems": [
            gem.value
            for gem, count in mission.gem_requirements.items()
            for _ in range(count)
            if gem is not GemType.WILD_CARD
        ],
        "wildCount": mission.gem_requirements.get(GemType.WILD_CARD, 0),
        "pattern": (
            mission.wild_card_requirement.name
            if mission.wild_card_requirement is not None
            else None
        ),
    }


def simulate(config: dict[str, Any]) -> dict[str, Any]:
    names = config.get("bots")
    catalog = bot_catalog()
    if not isinstance(names, list) or len(names) not in (3, 4, 5):
        raise ValueError("Select 3, 4, or 5 bots")
    if any(not isinstance(name, str) or name not in catalog for name in names):
        raise ValueError("Unknown bot selection")
    options = config.get("botOptions", [{} for _ in names])
    if not isinstance(options, list) or len(options) != len(names):
        raise ValueError("Provide one bot-options object per seat")
    classes = []
    normalized_options = []
    for name, selected in zip(names, options):
        if (
            not isinstance(selected, dict)
            or any(k not in LEVEL3_FEATURES for k in selected)
            or any(type(v) is not bool for v in selected.values())
            or (name != "SoftTrackingBot" and selected)
        ):
            raise ValueError(
                "Feature switches require SoftTrackingBot and Boolean values"
            )
        flags = tuple(selected.get(feature, False) for feature in LEVEL3_FEATURES)
        classes.append(
            LEVEL3_BOTS[flags] if name == "SoftTrackingBot" else catalog[name]
        )
        normalized_options.append(
            dict(zip(LEVEL3_FEATURES, flags)) if name == "SoftTrackingBot" else {}
        )
    seed = config.get("seed", 42)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be an integer from 0 to 4294967295")
    chart = next((c for c in VALUE_CHARTS if c.name == config.get("chart", "A")), None)
    if chart is None:
        raise ValueError("Select chart A, B, C, D, or E")
    with create_game(
        len(names),
        bots=classes,
        rng=Random(seed),
        bot_seed=seed,
        value_chart=chart,
        bot_timeout=2.0,
    ) as game:
        missions = list(game.available_missions)

        def snapshot() -> dict[str, Any]:
            display = [game.value_display[gem] for gem in STANDARD_GEMS]
            values = [chart.get_value(count) for count in display]
            players: list[dict[str, Any]] = []
            for p in game.players:
                gems = [p.gem_collection[gem] for gem in STANDARD_GEMS]
                breakdown = {
                    "cash": p.coins,
                    "gems": sum(n * v for n, v in zip(gems, values)),
                    "missions": sum(m.value for m in p.mission_cards),
                    "investments": p.invested_coins
                    + sum(
                        c.end_game_value
                        for c in p.auction_cards
                        if c.end_game_value > 0
                    ),
                    "loans": sum(
                        c.end_game_value
                        for c in p.auction_cards
                        if c.end_game_value < 0
                    ),
                }
                players.append(
                    {
                        "id": p.id,
                        "bot": names[p.id],
                        "coins": p.coins,
                        "invested": p.invested_coins,
                        "gems": gems,
                        "hand": [g.value for g in p.hidden_gem_hand],
                        "revealed": [g.value for g in p.public_gem_hand],
                        "missions": [missions.index(m) for m in p.mission_cards],
                        "breakdown": breakdown,
                        "score": sum(breakdown.values()),
                    }
                )
            return {
                "round": game.round_number,
                "players": players,
                "display": display,
                "values": values,
                "row": [g.value for g in game.auction_gems],
                "deck": len(game.gem_deck),
                "previousWinner": game.previous_winner_id,
                "availableMissions": [
                    missions.index(m) for m in game.available_missions
                ],
            }

        frames: list[dict[str, Any]] = [{"state": snapshot(), "event": None}]
        while game.auction_gems:
            before_row = list(game.auction_gems)
            before_missions = list(game.available_missions)
            result = play_round(game)
            card = result.auction_card
            frames.append(
                {
                    "state": snapshot(),
                    "event": {
                        "type": card.type.name,
                        "value": card.value,
                        "gems": [g.value for g in before_row[: card.value]]
                        if card.type.name == "AUCTION"
                        else [],
                        "bids": [result.bids[i] for i in range(len(names))],
                        "winner": result.winner_id,
                        "bid": result.winning_bid,
                        "reveal": result.revealed_gem.value
                        if result.revealed_gem is not None
                        else None,
                        "claimed": [
                            missions.index(m)
                            for m in before_missions
                            if m not in game.available_missions
                        ],
                    },
                }
            )
        final = finish_game(game)
        frames.append(
            {
                "state": snapshot(),
                "event": {
                    "type": "FINAL",
                    "winners": list(final.winner_ids),
                    "scores": list(final.scores),
                },
            }
        )
        return {
            "seed": seed,
            "chart": chart.name,
            "chartValues": list(chart.values),
            "botOptions": normalized_options,
            "missions": [mission_data(m, i) for i, m in enumerate(missions)],
            "frames": frames,
        }


if __name__ == "__main__":
    try:
        if "--catalog" in sys.argv:
            print(json.dumps(list(bot_catalog())))
        else:
            print(json.dumps(simulate(json.load(sys.stdin))))
    except (ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
