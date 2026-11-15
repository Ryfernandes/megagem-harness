"""Chart-aware, optional information-release heuristic for Megagem.

Contributors: Grace Guo (85375651b), OpenAI Codex (implementation assistance).

Initial hands fix final prices: showing a card changes information, not supply.
A/C: fewest public DISPLAY copies, then fewest remaining missions, then random.
This intentionally preserves the user's exact Chart-A priority even when private
information suggests a different final price. Collections/auction row are not
display copies. No hidden-hand card is treated as an owned scoring asset.
B/D: cool competition for high expected-value future purchases; mission count,
collection-based mission progress and remaining supply break intrinsic ties.
E: weigh the sign/size of displayed price movement against private expected
value, directing positive price attention away from our preferred purchases and
negative price attention toward them. Use full expected chart values, not the
chart at an expected display count. This is an opponent-perception proxy, not
a claim that rational rivals will follow current display quotes or be fooled.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from random import choice

from bot_definitions.valuation_bot import _missing, _mission_targets
from bots import BotInfo
from constants import STANDARD_GEMS, GemType


@dataclass(frozen=True)
class RevealAssessment:
    color: GemType
    public_count: int
    mission_count: int
    mission_progress: float
    expected_final_value: float
    future_copies: float
    display_price_change: int
    priority: tuple[float, ...]


def assess_reveals(
    info: BotInfo,
    values: Mapping[GemType, float],
    deck_fractions: Mapping[GemType, float],
) -> dict[GemType, RevealAssessment]:
    """Lower priority tuples win; equal tuples are broken randomly by card index."""
    chart = info.value_chart
    collection = info.self_player.won_gems
    # The reveal happens after the award but BEFORE automatic mission claims.
    # Already completed missions are about to leave the market and need no bait.
    missions = [m for m in info.available_missions if not m.is_satisfied(collection)]
    mission_counts = {
        gem: sum(any(target[i] > 0 for target in _mission_targets(m)) for m in missions)
        for i, gem in enumerate(STANDARD_GEMS)
    }
    visible = Counter(info.auction_gems)
    future = {
        gem: visible[gem] + info.unknown_gems.get(gem, 0) * deck_fractions[gem]
        for gem in STANDARD_GEMS
    }
    available_values = [values[g] for g in STANDARD_GEMS if future[g] > 0]
    market_value = sum(available_values) / max(1, len(available_values))
    ascending = all(a <= b for a, b in zip(chart.values, chart.values[1:]))
    descending = all(a >= b for a, b in zip(chart.values, chart.values[1:]))
    result = {}
    for gem in dict.fromkeys(info.self_player.hidden_gems):
        count = info.value_display.get(gem, 0)
        change = chart.get_value(count + 1) - chart.get_value(count)
        after = Counter(collection)
        after[gem] += 1  # Hypothetical FUTURE purchase, never the hand card itself.
        progress = sum(
            m.value / (1 + _missing(m, collection))
            for m in missions
            if _missing(m, after) < _missing(m, collection)
        )
        priority: tuple[float, ...]
        if ascending:
            # Exact requested lexicographic order: no extra tier before randomness.
            priority = (float(count), float(mission_counts[gem]))
        elif descending:
            # Only future auction copies can benefit from cheaper competition.
            active = future[gem] > 0
            priority = (
                -round(values[gem], 10) if active else 0.0,
                -float(mission_counts[gem]) if active else 0.0,
                -round(progress, 10) if active else 0.0,
                -round(future[gem], 10),
            )
        else:
            # E changes direction at its peak. Positive gain: heat an unwanted
            # color or cool a preferred one. Supply gates/saturates the effect.
            exposure = future[gem] / (1 + future[gem])
            gain = change * (market_value - values[gem]) * exposure
            direction = (change > 0) - (change < 0)
            priority = (
                -round(gain, 10),
                float(direction * mission_counts[gem]),
                round(direction * progress, 10),
            )
            if not available_values:
                priority = (0.0, 0.0, 0.0)
        result[gem] = RevealAssessment(
            gem,
            count,
            mission_counts[gem],
            progress,
            values[gem],
            future[gem],
            change,
            priority,
        )
    return result


def choose_reveal(
    info: BotInfo,
    values: Mapping[GemType, float],
    deck_fractions: Mapping[GemType, float],
) -> int:
    """Use the worker's seeded/checkpointed randomness for final ties."""
    if not info.self_player.hidden_gems:
        raise ValueError("There is no hidden gem to reveal")
    assessments = assess_reveals(info, values, deck_fractions)
    best = min(a.priority for a in assessments.values())
    indices = [
        i
        for i, gem in enumerate(info.self_player.hidden_gems)
        if assessments[gem].priority == best
    ]
    return indices[0] if len(indices) == 1 else choice(indices)
