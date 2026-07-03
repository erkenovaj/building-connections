"""Tests for the multi-turn rollout driver, using scripted fakes (no GPU/HF)."""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from connections_gym.env import ConnectionsEnv
from train.rollout import play_episode

_SAMPLING_PARAMS = {
    "max_attempts": 20,
    "per_solve_timeout_seconds": 0.5,
    "total_timeout_seconds": 5.0,
}
SEED = 1_000_000


class FakeTokenizer:
    """Char-level tokenizer: token id = ord(char), lossless roundtrip."""

    pad_token_id = 0

    def apply_chat_template(
        self, messages, add_generation_prompt=True, tokenize=True, return_dict=False,
        enable_thinking=True,
    ):
        self.last_enable_thinking = enable_thinking
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return self.encode(text)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class FakeModel:
    """Replays scripted completions; logits are one-hot on the scripted token."""

    def __init__(self, tokenizer, scripted):
        self._outs = [tokenizer.encode(s) for s in scripted]
        self._calls = 0

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        new = self._outs[self._calls]
        self._calls += 1
        seq = torch.cat([input_ids[0], torch.tensor(new, dtype=input_ids.dtype)])
        scores = []
        for tok in new:
            row = torch.full((1, tok + 1), -1e9)
            row[0, tok] = 0.0
            scores.append(row)
        return SimpleNamespace(sequences=seq.unsqueeze(0), scores=scores)


def _guess(items):
    return json.dumps({"guess": {"label": "g", "items": items}, "notes": "n"})


def _board_groups():
    env = ConnectionsEnv(num_categories=2, sampling_params=_SAMPLING_PARAMS)
    env.reset(SEED)
    return env, list(env.groups.values())


def test_clean_win_stats_and_alignment():
    env, groups = _board_groups()
    tok = FakeTokenizer()
    scripted = [_guess(groups[0]), _guess(groups[1])]
    model = FakeModel(tok, scripted)

    ep = play_episode(model, tok, env, SEED)

    assert ep.won and ep.mistakes == 0 and ep.tool_calls == 2
    assert ep.truncations == 0
    tail = len(ep.token_ids) - ep.prompt_len
    assert len(ep.env_mask) == tail == len(ep.logprobs)
    model_tokens = sum(len(tok.encode(s)) for s in scripted)
    assert sum(ep.env_mask) == model_tokens
    # env tokens carry no logprob
    assert all(lp == 0.0 for lp, m in zip(ep.logprobs, ep.env_mask) if m == 0)
    roles = [m["role"] for m in ep.messages]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_wrong_then_recover():
    env, groups = _board_groups()
    tok = FakeTokenizer()
    near_miss = groups[0][:3] + [groups[1][0]]  # k=3 wrong guess
    scripted = [_guess(near_miss), _guess(groups[0]), _guess(groups[1])]
    model = FakeModel(tok, scripted)

    ep = play_episode(model, tok, env, SEED)

    assert ep.won and ep.mistakes == 1 and ep.tool_calls == 3
    # wrong guess surfaced back to the model in the next env turn
    assert "Wrong" in ep.messages[2]["content"]


def test_unparseable_output_costs_mistake_and_loses():
    env, _ = _board_groups()
    tok = FakeTokenizer()
    scripted = ["no idea"] * 4
    model = FakeModel(tok, scripted)

    ep = play_episode(model, tok, env, SEED)

    assert not ep.won and ep.mistakes == 4 and ep.tool_calls == 4


def test_truncation_counted():
    env, _ = _board_groups()
    tok = FakeTokenizer()
    scripted = ["x" * 8] * 4  # 8 tokens each, budget 8 -> flagged truncated
    model = FakeModel(tok, scripted)

    ep = play_episode(model, tok, env, SEED, max_new_tokens=8)

    assert ep.truncations == 4 and not ep.won


def test_no_think_reaches_template_and_env_turns():
    """enable_thinking=False is forwarded to the chat template and every
    injected env turn opens the assistant reply with an empty think block."""
    env, groups = _board_groups()
    tok = FakeTokenizer()
    near_miss = groups[0][:3] + [groups[1][0]]  # forces one extra env turn
    scripted = [_guess(near_miss), _guess(groups[0]), _guess(groups[1])]
    model = FakeModel(tok, scripted)

    ep = play_episode(model, tok, env, SEED, enable_thinking=False)

    assert ep.won
    assert tok.last_enable_thinking is False
    text = tok.decode(ep.token_ids)
    assert text.count("<|im_start|>assistant\n<think>\n\n</think>\n\n") == 2
