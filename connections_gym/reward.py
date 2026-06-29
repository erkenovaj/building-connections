"""Dense per-guess reward for GRPO training (plan §3).

A single guess is scored by ``k`` = its overlap with the most-overlapping
*unsolved* group. Guesses that are not exactly four distinct terms drawn from
the current remaining terms are invalid. ``info`` carries step-level signal
for the faithfulness (H1) dataset.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from .rules import CONCEPTS_PER_GROUP

# k=2 overlap is a cheap, stable local optimum (the policy camps there and the
# intra-group reward variance collapses); only k>=3 now earns positive reward.
_REWARD_BY_K = {4: 1.0, 3: 0.25}
_LOW_OVERLAP_REWARD = -0.5
_INVALID_REWARD = -1.0


def reward(
    guess_items: Sequence[str],
    board_groups: Mapping[str, Sequence[str]],
    solved_keys: Iterable[str],
) -> tuple[float, dict[str, Any]]:
    """Score one single-group guess.

    Returns ``(reward, info)`` where ``info`` holds ``k`` (overlap with the
    best unsolved group, ``None`` when invalid), ``matched_group_key`` (set
    only on an exact k=4 match) and ``ground_truth_groups``.
    """
    solved = set(solved_keys)
    unsolved = {key: members for key, members in board_groups.items() if key not in solved}
    remaining = {term for members in unsolved.values() for term in members}

    selected = set(guess_items)
    is_valid = (
        len(guess_items) == CONCEPTS_PER_GROUP
        and len(selected) == CONCEPTS_PER_GROUP
        and selected <= remaining
    )
    if not is_valid:
        return _INVALID_REWARD, {
            "k": None,
            "matched_group_key": None,
            "ground_truth_groups": dict(board_groups),
        }

    best_k = 0
    best_key: Optional[str] = None
    for key, members in unsolved.items():
        overlap = len(selected & set(members))
        if overlap > best_k:
            best_k, best_key = overlap, key

    reward_value = _REWARD_BY_K.get(best_k, _LOW_OVERLAP_REWARD)
    matched_group_key = best_key if best_k == CONCEPTS_PER_GROUP else None
    return reward_value, {
        "k": best_k,
        "matched_group_key": matched_group_key,
        "ground_truth_groups": dict(board_groups),
    }
