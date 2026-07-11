"""Terminal episode reward for agentic GRPO.

One scalar per finished episode: win bonus minus penalties for mistakes and
guess count. No per-step shaping — the dense per-guess reward in
``reward.py`` stays for the bandit trainer and probes only.
"""

from __future__ import annotations

WIN_REWARD = 1.0
MISTAKE_PENALTY = 0.25
TOOL_CALL_PENALTY = 0.02


def episode_reward(won: bool, mistakes: int, tool_calls: int) -> float:
    """Return ``1.0*[won] - 0.25*mistakes - 0.02*tool_calls``."""
    return (
        WIN_REWARD * float(won)
        - MISTAKE_PENALTY * mistakes
        - TOOL_CALL_PENALTY * tool_calls
    )
