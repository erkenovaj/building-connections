"""Prompt construction and guess parsing for the policy model.

The prompt string mirrors the served agent endpoint (api/llm/guess.py) so the
training distribution matches deployment. Parsing reuses the existing,
tested helpers from the agent loop rather than duplicating them.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from api.llm._ollamafree import extract_json, normalize_one_guess

_SCHEMA = (
    '{"guess":{"label":"short label","items":["term","term","term","term"]},'
    '"notes":"one short sentence"}'
)


def build_prompt(
    remaining: Sequence[str],
    num_categories: int,
    history: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    """Render the single-group guessing prompt for the current board state.

    ``history`` is the list of previously tried (wrong) guesses; each entry's
    ``items`` are surfaced so the model avoids repeating them.
    """
    tried = []
    for entry in history or []:
        if isinstance(entry, dict) and isinstance(entry.get("items"), list):
            items = [str(item) for item in entry["items"]]
            if items:
                tried.append(items)

    lines = [
        "You are playing a Connections-style puzzle, one group at a time.",
        f"This puzzle has {num_categories} groups total.",
        "Pick the single group of exactly 4 terms you are most confident about.",
        "Use only exact terms from the remaining list below.",
    ]
    if tried:
        lines.append(
            "Do not repeat any of these already-tried wrong groups: " + json.dumps(tried)
        )
    lines.extend(
        [
            "Return strict JSON only with this schema:",
            _SCHEMA,
            "",
            f"Remaining terms: {json.dumps(list(remaining))}",
        ]
    )
    return "\n".join(lines)


def parse_guess(text: str, remaining: Sequence[str]) -> Optional[dict]:
    """Extract a normalized single-group guess from raw model output.

    Returns ``{"label", "items"}`` with exactly four terms drawn from
    ``remaining``, or ``None`` if the text cannot be parsed into a valid guess.
    """
    return normalize_one_guess(extract_json(text), remaining)
