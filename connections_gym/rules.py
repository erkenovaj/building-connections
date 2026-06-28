"""Python port of the Connections rules engine (see js/game-rules.js).

Provides the in-process ground-truth evaluation that reward.py and env.py
need: guess evaluation, remaining-term computation, and game status.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

MAX_MISTAKES = 4
CONCEPTS_PER_GROUP = 4


def evaluate_guess(
    board_groups: Mapping[str, Sequence[str]], items: Sequence[str]
) -> tuple[bool, Optional[str]]:
    """Evaluate a 4-term guess against the puzzle's groups.

    Returns ``(correct, matched_key)``. A guess is correct only when it is
    exactly four distinct terms forming the full membership of one group.
    """
    if len(items) != CONCEPTS_PER_GROUP:
        return False, None
    selected = set(items)
    if len(selected) != len(items):
        return False, None
    for key, members in board_groups.items():
        if set(members) == selected:
            return True, key
    return False, None


def remaining_terms(
    board: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    solved_keys: Iterable[str],
) -> list[str]:
    """Return board terms not belonging to any already-solved group, in board order."""
    solved: set[str] = set()
    for key in solved_keys:
        members = groups.get(key)
        if members:
            solved.update(members)
    return [term for term in board if term not in solved]


def game_status(mistakes: int, solved: int, num_categories: int) -> str:
    """Return ``'won'``, ``'lost'`` or ``'playing'`` for the current counts."""
    if solved == num_categories and mistakes < MAX_MISTAKES:
        return "won"
    if mistakes >= MAX_MISTAKES:
        return "lost"
    return "playing"
