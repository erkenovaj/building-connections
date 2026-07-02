"""Tests for STaR trajectory filtering."""

import pytest

pytest.importorskip("torch")

from train.rollout import Episode
from train.star_sft import episode_to_record


def _episode(won, mistakes=1, tool_calls=3):
    return Episode(
        token_ids=[1, 2, 3], prompt_len=1, env_mask=[1, 1], logprobs=[0.0, 0.0],
        won=won, mistakes=mistakes, tool_calls=tool_calls, truncations=0,
        messages=[{"role": "user", "content": "p"},
                  {"role": "assistant", "content": "<think>x</think>guess"}],
    )


def test_won_episode_kept_with_think_intact():
    record = episode_to_record(_episode(won=True))
    assert record is not None
    assert record["mistakes"] == 1 and record["tool_calls"] == 3
    assert "<think>" in record["messages"][1]["content"]


def test_lost_episode_dropped():
    assert episode_to_record(_episode(won=False)) is None
