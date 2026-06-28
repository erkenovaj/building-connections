"""Gymnasium-style environment driving full Connections episodes.

``reset(seed)`` samples a uniquely-solvable board via
:func:`connections_sampler.sample_game` (deterministic per seed) and
``step(items)`` plays one single-group guess: it scores the guess with the
dense training reward, applies the rules engine to remove solved groups and
count mistakes, and terminates when the board is solved or four mistakes are
made. Used for full-episode evaluation; ``reward.py`` is used directly for the
contextual-bandit training step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from connections_sampler import GameMode, SamplingParameters, sample_game

from .prompt import build_prompt
from .reward import reward
from .rules import evaluate_guess, game_status, remaining_terms

WIN_BONUS = 2.0

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "category-templates-new.json"
)


class ConnectionsEnv:
    """Full-episode Connections environment over the board sampler."""

    def __init__(
        self,
        config: Optional[Sequence[Mapping[str, Any]]] = None,
        mode: GameMode = GameMode.BASIC,
        sampling_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Build the environment.

        ``config`` is the category word-pool config passed to ``sample_game``;
        it defaults to ``configs/category-templates-new.json``. ``sampling_params``
        are extra :class:`SamplingParameters` fields (excluding ``seed``) merged
        into every ``reset``.
        """
        if config is None:
            with open(_DEFAULT_CONFIG_PATH) as handle:
                loaded: Sequence[Mapping[str, Any]] = json.load(handle)
            config = loaded
        self.config = config
        self.mode = mode
        self.sampling_params = dict(sampling_params or {})

        self.board: list[str] = []
        self.groups: dict[str, list[str]] = {}
        self.solved_keys: set[str] = set()
        self.mistakes = 0
        self.history: list[dict[str, Any]] = []
        self.num_categories = 0

    def reset(self, seed: int) -> dict[str, Any]:
        """Sample a fresh board for ``seed`` and return the initial observation."""
        params = SamplingParameters(seed=seed, **self.sampling_params)
        result = sample_game(self.config, mode=self.mode, parameters=params)
        if not result.ok or result.game is None:
            raise RuntimeError(f"board sampling failed: {result.reason}")

        game = result.game
        self.board = list(game.board)
        self.groups = {group.key: list(group.members) for group in game.groups}
        self.solved_keys = set()
        self.mistakes = 0
        self.history = []
        self.num_categories = len(self.groups)
        return self._obs()

    def step(self, items: Sequence[str]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Play one single-group guess and advance the episode."""
        reward_value, info = reward(items, self.groups, self.solved_keys)

        correct, matched_key = evaluate_guess(self.groups, items)
        if correct and matched_key is not None and matched_key not in self.solved_keys:
            self.solved_keys.add(matched_key)
        else:
            self.mistakes += 1
            self.history.append({"items": list(items)})

        status = game_status(self.mistakes, len(self.solved_keys), self.num_categories)
        done = status != "playing"
        if status == "won":
            reward_value += WIN_BONUS

        info = {
            **info,
            "status": status,
            "mistakes": self.mistakes,
            "solved_keys": sorted(self.solved_keys),
        }
        return self._obs(), reward_value, done, info

    def _obs(self) -> dict[str, Any]:
        """Build the observation for the current board state."""
        remaining = remaining_terms(self.board, self.groups, self.solved_keys)
        return {
            "remaining": remaining,
            "num_categories": self.num_categories,
            "history": list(self.history),
            "prompt": build_prompt(remaining, self.num_categories, self.history),
        }
