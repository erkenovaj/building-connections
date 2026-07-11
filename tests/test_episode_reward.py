"""Tests for the terminal episode reward."""

import pytest

from connections_gym.episode_reward import episode_reward


@pytest.mark.parametrize(
    "won,mistakes,tool_calls,expected",
    [
        (True, 0, 2, 0.96),    # clean 2-cat win
        (True, 0, 4, 0.92),    # clean 4-cat win
        (True, 2, 6, 0.38),    # sloppy win
        (False, 4, 4, -1.08),  # fast loss
        (False, 4, 7, -1.14),  # loss after solving some groups
        (False, 0, 0, 0.0),    # degenerate: nothing happened
    ],
)
def test_episode_reward(won, mistakes, tool_calls, expected):
    assert episode_reward(won, mistakes, tool_calls) == pytest.approx(expected)
