# Agentic GRPO: Tool-Loop Episodes + Terminal Reward — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the Connections policy on full multi-turn episodes (guess → env feedback → revise) with a single terminal reward, replacing the contextual-bandit proxy.

**Architecture:** A training-free rollout driver (`train/rollout.py`) plays whole episodes against `ConnectionsEnv` and records token ids + `env_mask` separating model tokens from injected feedback tokens. The driver feeds three consumers: a win@k probe (gate), STaR rejection-sampling SFT (warm start), and a TRL `rollout_func`-based GRPO trainer scoring episodes with `R = 1.0·[won] − 0.25·mistakes − 0.02·tool_calls`.

**Tech Stack:** Python, PyTorch, TRL ≥ 1.7 (GRPOTrainer `rollout_func`, SFTTrainer), PEFT/LoRA, transformers, `connections_sampler` (ortools), pytest.

**Spec:** `docs/superpowers/specs/2026-07-02-agentic-grpo-terminal-reward-design.md`

## Global Constraints

- Terminal reward exactly `R = 1.0·[won] − 0.25·mistakes − 0.02·tool_calls`; no per-step shaping.
- Unit tests run offline in `.venv` (has ortools): `./.venv/bin/python -m pytest tests/ -q --tb=short`. No HF model downloads in unit tests — use fakes.
- Model runs (smoke) use `.venv-train` (torch 2.12.1, transformers 5.12.1, MPS) with Qwen/Qwen3-0.6B; real runs are Kaggle T4 with Qwen/Qwen3-1.7B.
- Generation budget for Qwen3 thinking models: `max_new_tokens` default 1024 (lesson from `0a90d49`); always report truncation.
- LoRA: r=16, alpha=32, dropout=0.05, all-linear (same as `train/grpo_connections.py`).
- Mentor defaults in the new trainer: `--lr-scheduler cosine`, `--grad-accum 2` (effective batch = per-device 8 × 2 = 16).
- New scripts follow existing style: module docstring with run example, `sys.path.insert` repo-root bootstrap, argparse CLI, English docstrings.
- `train/grpo_connections.py` (bandit) stays untouched.
- Conventional Commits, subject ≤ 50 chars, no AI attribution.
- Do not use the word "truncate" in inline `git commit -m` — a hook blocks it; write the message to the scratchpad and use `git commit -F` if needed.

---

### Task 1: Terminal episode reward

**Files:**

- Create: `connections_gym/episode_reward.py`
- Test: `tests/test_episode_reward.py`

**Interfaces:**

- Consumes: nothing.
- Produces: `episode_reward(won: bool, mistakes: int, tool_calls: int) -> float`; constants `WIN_REWARD = 1.0`, `MISTAKE_PENALTY = 0.25`, `TOOL_CALL_PENALTY = 0.02`. Tasks 3 and 5 import these.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_episode_reward.py -q --tb=short`
Expected: FAIL — `ModuleNotFoundError: No module named 'connections_gym.episode_reward'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_episode_reward.py -q --tb=short`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add connections_gym/episode_reward.py tests/test_episode_reward.py
git commit -m "feat(gym): terminal episode reward"
```

---

### Task 2: Multi-turn rollout driver

**Files:**

- Create: `train/rollout.py`
- Test: `tests/test_rollout.py`

**Interfaces:**

- Consumes: `ConnectionsEnv` (`connections_gym/env.py`) — `reset(seed) -> obs` with `obs["prompt"]`/`obs["remaining"]`, `step(items) -> (obs, reward, done, info)` with `info["status"]`/`info["k"]`; `parse_guess(text, remaining)` from `connections_gym/prompt.py`.
- Produces (Tasks 3–5 rely on these exact names):

```python
@dataclass
class Episode:
    token_ids: list[int]        # full sequence: initial prompt + interleaved turns
    prompt_len: int             # length of the initial rendered prompt
    env_mask: list[int]         # aligned to token_ids[prompt_len:], 1=model token, 0=env token
    logprobs: list[float]       # aligned to token_ids[prompt_len:]; 0.0 on env tokens
    won: bool
    mistakes: int
    tool_calls: int             # number of guesses made
    truncations: int            # generations that hit max_new_tokens
    messages: list[dict[str, str]]  # full conversation, for STaR SFT

def play_episode(model, tokenizer, env, seed, *, device="cpu", temperature=1.0,
                 top_p=1.0, max_new_tokens=1024, max_turns=12) -> Episode
```

- [ ] **Step 1: Write the failing tests**

`tests/test_rollout.py`:

```python
"""Tests for the multi-turn rollout driver, using scripted fakes (no GPU/HF)."""

import json
from types import SimpleNamespace

import torch

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

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_rollout.py -q --tb=short`
Expected: FAIL — `ModuleNotFoundError: No module named 'train.rollout'`

(If `train/` lacks `__init__.py` and the import fails for that reason, keep the existing repo pattern: tests already import via repo-root conftest/sys.path — mirror how `tests/test_reward.py` imports `connections_gym`.)

- [ ] **Step 3: Write the driver**

`train/rollout.py`:

```python
"""Multi-turn rollout driver for agentic Connections episodes.

``play_episode`` plays one full board against :class:`ConnectionsEnv`:
generate a guess, parse it, step the env, inject the feedback as a new user
turn, repeat until the game ends. It records the interleaved token sequence
together with ``env_mask`` (1 = model-generated token, 0 = injected env
token) and per-token logprobs, so the same episodes serve the win@k probe,
STaR sampling, and the GRPO ``rollout_func``.

The env's dense per-step reward is ignored here; episodes are scored
terminally by ``connections_gym.episode_reward``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from connections_gym.prompt import parse_guess

# ChatML delta for an injected env turn. Both training model families (Qwen2.5,
# Qwen3) use ChatML; rendering the delta directly keeps earlier <think> blocks
# intact, which re-applying the chat template would strip.
_ENV_TURN = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


@dataclass
class Episode:
    """One finished episode with token-level bookkeeping for training."""

    token_ids: list[int]
    prompt_len: int
    env_mask: list[int]
    logprobs: list[float]
    won: bool
    mistakes: int
    tool_calls: int
    truncations: int
    messages: list[dict[str, str]] = field(default_factory=list)


def _feedback_line(info: dict[str, Any]) -> str:
    """One-line result of the last guess for the next env turn."""
    if info.get("k") == 4:
        return "Correct."
    if info.get("k") is None:
        return "Invalid guess (could not be used) — that costs one mistake."
    return "Wrong guess."


def play_episode(
    model,
    tokenizer,
    env,
    seed: int,
    *,
    device: str = "cpu",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 1024,
    max_turns: int = 12,
) -> Episode:
    """Play one episode on the board sampled for ``seed`` and return it."""
    obs = env.reset(seed)
    messages = [{"role": "user", "content": obs["prompt"]}]
    ids = list(
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    )
    prompt_len = len(ids)

    env_mask: list[int] = []
    logprobs: list[float] = []
    won = False
    tool_calls = 0
    truncations = 0

    for _ in range(max_turns):
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        new_ids = out.sequences[0][len(ids):].tolist()
        for tok, scores in zip(new_ids, out.scores):
            logprobs.append(
                torch.log_softmax(scores[0].float(), dim=-1)[tok].item()
            )
        ids.extend(new_ids)
        env_mask.extend([1] * len(new_ids))
        if len(new_ids) >= max_new_tokens:
            truncations += 1

        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": text})
        tool_calls += 1

        guess = parse_guess(text, obs["remaining"])
        items = guess["items"] if guess else []
        obs, _, done, info = env.step(items)
        if done:
            won = info["status"] == "won"
            break

        feedback = f"{_feedback_line(info)}\n\n{obs['prompt']}"
        messages.append({"role": "user", "content": feedback})
        delta = tokenizer.encode(_ENV_TURN.format(content=feedback), add_special_tokens=False)
        ids.extend(delta)
        env_mask.extend([0] * len(delta))
        logprobs.extend([0.0] * len(delta))

    return Episode(
        token_ids=ids,
        prompt_len=prompt_len,
        env_mask=env_mask,
        logprobs=logprobs,
        won=won,
        mistakes=env.mistakes,
        tool_calls=tool_calls,
        truncations=truncations,
        messages=messages,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_rollout.py -q --tb=short`
Expected: 4 passed

- [ ] **Step 5: Run the whole suite**

Run: `./.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: all pass (53 pre-existing + 10 new)

- [ ] **Step 6: MPS smoke test with a real model**

Run:

```bash
./.venv-train/bin/python - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
from connections_gym.env import ConnectionsEnv
from train.rollout import play_episode

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to("mps").eval()
env = ConnectionsEnv(num_categories=2, sampling_params={
    "max_attempts": 20, "per_solve_timeout_seconds": 0.5, "total_timeout_seconds": 5.0})
ep = play_episode(model, tok, env, 1_000_000, device="mps")
print("won", ep.won, "mistakes", ep.mistakes, "tool_calls", ep.tool_calls,
      "truncations", ep.truncations, "tokens", len(ep.token_ids))
EOF
```

Expected: completes without error, sane stats (tool_calls ≥ 1, token counts consistent). Win not required.

- [ ] **Step 7: Commit**

```bash
git add train/rollout.py tests/test_rollout.py
git commit -m "feat(train): multi-turn rollout driver"
```

---

### Task 3: Multi-turn win@k probe

**Files:**

- Create: `train/probe_wink.py`
- Reuse: `pass_at_k`, `load_policy`, `_pick_device`, `_SAMPLING_PARAMS` from `train/probe_passk.py`

**Interfaces:**

- Consumes: `play_episode` / `Episode` (Task 2), `episode_reward` (Task 1), `pass_at_k(n, c, k)` from `train/probe_passk.py`.
- Produces: CLI script printing a summary JSON; gates этапы 4–6. No importable API needed by later tasks.

- [ ] **Step 1: Write the probe**

`train/probe_wink.py`:

```python
"""Estimate base-model win@k on full multi-turn Connections episodes.

For each held-out board this plays ``N`` complete episodes with the rollout
driver and reports the unbiased win@k estimator plus mean mistakes, mean
tool calls, mean terminal reward, and the truncation rate. ``win@k ~ 0``
means agentic GRPO has no successes to amplify — gate every RL/SFT stage on
this probe.

Example::

    python train/probe_wink.py --model Qwen/Qwen3-1.7B --num-categories 2 \
        --num-boards 16 --num-episodes 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connections_gym.env import ConnectionsEnv
from connections_gym.episode_reward import episode_reward
from train.probe_passk import _SAMPLING_PARAMS, _pick_device, load_policy, pass_at_k
from train.rollout import play_episode


def main() -> None:
    """Play episodes over held-out boards and print the win@k summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--config", default=None,
                        help="path to a flat category config (default: env's built-in pool)")
    parser.add_argument("--num-categories", type=int, default=None,
                        help="board groups for curriculum (None = full 4-group game)")
    parser.add_argument("--num-boards", type=int, default=16)
    parser.add_argument("--num-episodes", type=int, default=8,
                        help="episodes per board (sequential; keep small, episodes are slow)")
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    device = _pick_device()
    model, tokenizer = load_policy(args.model, device)
    config = None
    if args.config:
        with open(args.config) as handle:
            config = json.load(handle)
    env = ConnectionsEnv(
        config=config, num_categories=args.num_categories, sampling_params=_SAMPLING_PARAMS
    )

    n = args.num_episodes
    ks = sorted({k for k in (1, 4, n) if k <= n})
    wink_sums = {k: 0.0 for k in ks}
    boards_with_win = 0
    mistakes_total = tool_calls_total = trunc_total = gen_total = 0
    reward_total = 0.0

    for i, seed in enumerate(range(args.seed_start, args.seed_start + args.num_boards), start=1):
        episodes = [
            play_episode(
                model, tokenizer, env, seed, device=device,
                temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
            for _ in range(n)
        ]
        c = sum(ep.won for ep in episodes)
        for k in ks:
            wink_sums[k] += pass_at_k(n, c, k)
        boards_with_win += int(c > 0)
        mistakes_total += sum(ep.mistakes for ep in episodes)
        tool_calls_total += sum(ep.tool_calls for ep in episodes)
        trunc_total += sum(ep.truncations for ep in episodes)
        gen_total += sum(ep.tool_calls for ep in episodes)
        reward_total += sum(
            episode_reward(ep.won, ep.mistakes, ep.tool_calls) for ep in episodes
        )
        print(
            f"  [{i}/{args.num_boards}] seed {seed}: wins {c}/{n} "
            f"mistakes {sum(ep.mistakes for ep in episodes)} "
            f"trunc {sum(ep.truncations for ep in episodes)}",
            file=sys.stderr,
            flush=True,
        )

    nb = args.num_boards
    total_eps = nb * n
    summary = {
        "model": args.model,
        "num_boards": nb,
        "num_episodes": n,
        "temperature": args.temperature,
        "win_at_k": {str(k): round(wink_sums[k] / nb, 4) for k in ks},
        "boards_with_any_win": boards_with_win,
        "mean_mistakes": round(mistakes_total / total_eps, 3),
        "mean_tool_calls": round(tool_calls_total / total_eps, 3),
        "mean_episode_reward": round(reward_total / total_eps, 4),
        "truncated_gen_rate": round(trunc_total / max(gen_total, 1), 4),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: MPS smoke test**

Run: `./.venv-train/bin/python train/probe_wink.py --model Qwen/Qwen3-0.6B --num-categories 2 --num-boards 2 --num-episodes 2`
Expected: 2 per-board stderr lines, then summary JSON with `win_at_k`, `mean_episode_reward`, `truncated_gen_rate` fields. Wins may be 0 for 0.6B — the probe running end-to-end is the success criterion.

- [ ] **Step 3: Run the full suite (imports untouched)**

Run: `./.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add train/probe_wink.py
git commit -m "feat(train): multi-turn win@k probe"
```

---

### Task 4: STaR sampling + SFT

**Files:**

- Create: `train/star_sft.py`
- Test: `tests/test_star_sft.py`

**Interfaces:**

- Consumes: `play_episode` / `Episode` (Task 2).
- Produces: `episode_to_record(episode) -> dict | None` (None unless won; otherwise `{"messages": [...], "mistakes": int, "tool_calls": int}`); CLI `sample` writes JSONL of such records, CLI `train` fits a LoRA on them. Task 5 consumes the saved adapter path via `--init-lora`.

- [ ] **Step 1: Write the failing test**

`tests/test_star_sft.py`:

```python
"""Tests for STaR trajectory filtering."""

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_star_sft.py -q --tb=short`
Expected: FAIL — `ModuleNotFoundError: No module named 'train.star_sft'`

- [ ] **Step 3: Verify TRL SFT masking support**

Run: `grep -rn "assistant_only_loss" .venv-train/lib/python*/site-packages/trl/ | head -5`
Expected: hits in `SFTConfig`/`SFTTrainer`. If absent in the installed TRL, replace `assistant_only_loss=True` below with `DataCollatorForCompletionOnlyLM(response_template="<|im_start|>assistant")` per TRL docs — check with context7 before deviating.

- [ ] **Step 4: Write the script**

`train/star_sft.py`:

```python
"""STaR / rejection-sampling SFT warm start (spec этап 4).

``sample`` plays base-model episodes on small curriculum boards via the
rollout driver and keeps only won trajectories — ``<think>`` blocks intact —
as JSONL conversations. ``train`` fits a LoRA on those conversations with
loss on assistant tokens only. The resulting adapter is the starting point
for agentic GRPO (``train/grpo_agentic.py --init-lora``).

Examples::

    python train/star_sft.py sample --model Qwen/Qwen3-1.7B --num-categories 2 \
        --num-boards 200 --episodes-per-board 4 --out data/star_2cat.jsonl
    python train/star_sft.py train --model Qwen/Qwen3-1.7B \
        --data data/star_2cat.jsonl --output outputs/star-sft-2cat
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def episode_to_record(episode) -> Optional[dict]:
    """Keep a won episode as an SFT record; drop lost ones."""
    if not episode.won:
        return None
    return {
        "messages": episode.messages,
        "mistakes": episode.mistakes,
        "tool_calls": episode.tool_calls,
    }


def _cmd_sample(args) -> None:
    """Sample base-model episodes and write won trajectories as JSONL."""
    from connections_gym.env import ConnectionsEnv
    from train.probe_passk import _SAMPLING_PARAMS, _pick_device, load_policy
    from train.rollout import play_episode

    device = _pick_device()
    model, tokenizer = load_policy(args.model, device)
    config = None
    if args.config:
        with open(args.config) as handle:
            config = json.load(handle)
    env = ConnectionsEnv(
        config=config, num_categories=args.num_categories, sampling_params=_SAMPLING_PARAMS
    )

    kept = played = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as out:
        for seed in range(args.seed_start, args.seed_start + args.num_boards):
            for _ in range(args.episodes_per_board):
                episode = play_episode(
                    model, tokenizer, env, seed, device=device,
                    temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                )
                played += 1
                record = episode_to_record(episode)
                if record is not None:
                    out.write(json.dumps(record) + "\n")
                    kept += 1
            print(f"  seed {seed}: kept {kept}/{played}", file=sys.stderr, flush=True)
    print(json.dumps({"played": played, "kept": kept, "out": args.out}))


def _cmd_train(args) -> None:
    """SFT a LoRA on won trajectories, loss on assistant tokens only."""
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.select_columns(["messages"])

    config = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        assistant_only_loss=True,
        logging_steps=1,
        report_to="none",
        gradient_checkpointing=True,
    )
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )
    trainer = SFTTrainer(
        model=args.model, args=config, train_dataset=dataset, peft_config=peft_config
    )
    trainer.train()
    trainer.save_model(args.output)


def main() -> None:
    """Dispatch the ``sample`` / ``train`` subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p_sample.add_argument("--config", default=None)
    p_sample.add_argument("--num-categories", type=int, default=2)
    p_sample.add_argument("--num-boards", type=int, default=200)
    p_sample.add_argument("--episodes-per-board", type=int, default=4)
    p_sample.add_argument("--seed-start", type=int, default=0)
    p_sample.add_argument("--temperature", type=float, default=1.0)
    p_sample.add_argument("--max-new-tokens", type=int, default=1024)
    p_sample.add_argument("--out", default="data/star.jsonl")
    p_sample.set_defaults(func=_cmd_sample)

    p_train = sub.add_parser("train")
    p_train.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--grad-accum", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-5)
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--output", default="outputs/star-sft")
    p_train.set_defaults(func=_cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_star_sft.py -q --tb=short`
Expected: 2 passed

- [ ] **Step 6: MPS smoke test of sampling**

Run: `./.venv-train/bin/python train/star_sft.py sample --model Qwen/Qwen3-0.6B --num-categories 2 --num-boards 2 --episodes-per-board 2 --out /tmp/star-smoke.jsonl`
Expected: final JSON `{"played": 4, "kept": <0..4>, ...}`; if kept > 0, each JSONL line has `messages` with roles alternating user/assistant.

- [ ] **Step 7: Commit**

```bash
git add train/star_sft.py tests/test_star_sft.py
git commit -m "feat(train): STaR sampling and SFT warm start"
```

---

### Task 5: Agentic GRPO trainer (`rollout_func` + `env_mask`)

**Files:**

- Create: `train/grpo_agentic.py`
- Reference: `train/grpo_connections.py` (LoRA config, `CompactLogCallback` — import it, don't copy)

**Interfaces:**

- Consumes: `play_episode` / `Episode` (Task 2), `episode_reward` (Task 1), `CompactLogCallback` from `train/grpo_connections.py`.
- Produces: CLI training script; final adapter saved to `--output`. Terminal deliverable — nothing imports it.

- [ ] **Step 1: Verify the installed TRL rollout contract (do not skip)**

Run:

```bash
grep -n "rollout_func" .venv-train/lib/python*/site-packages/trl/trainer/grpo_trainer.py | head -20
grep -rn "rollout_func" .venv-train/lib/python*/site-packages/trl/trainer/grpo_config.py | head
```

Confirm three things in the installed source (fetch TRL docs via context7 if unclear):

1. exact `rollout_func` signature (what it receives: prompt texts? repeated × num_generations?);
2. required return keys (`prompt_ids`, `completion_ids`, `logprobs`) and how extra keys reach reward funcs;
3. which key masks completion tokens out of the loss (`env_mask` vs `completion_mask` — adjust the code below to the real name).

Record findings as a comment at the top of `grpo_agentic.py`. If the installed TRL is < 1.7 (no `rollout_func`), stop and upgrade TRL in `.venv-train` first.

- [ ] **Step 2: Write the trainer**

`train/grpo_agentic.py` (adjust mask key per Step 1 findings):

```python
"""Agentic GRPO on full multi-turn episodes with terminal reward (spec этап 5).

A custom TRL ``rollout_func`` plays whole Connections episodes with the
rollout driver; ``env_mask`` keeps injected env-feedback tokens out of the
policy-gradient loss, and each episode is scored once with the terminal
reward from ``connections_gym.episode_reward``. Start from the STaR adapter
via ``--init-lora``.

Run::

    python train/grpo_agentic.py --model Qwen/Qwen3-1.7B --num-categories 2 \
        --init-lora outputs/star-sft-2cat --max-steps 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from trl import GRPOConfig, GRPOTrainer

from connections_gym.env import ConnectionsEnv
from connections_gym.episode_reward import episode_reward
from train.grpo_connections import CompactLogCallback
from train.probe_passk import _SAMPLING_PARAMS, _pick_device
from train.rollout import play_episode

# prompt text -> board seed, filled by build_dataset and read by the rollout
# func (TRL hands the rollout func prompt strings, not dataset rows).
_SEED_BY_PROMPT: dict[str, int] = {}


def build_dataset(num_boards, seed_start, num_categories, config):
    """One row per board: the rendered initial prompt, keyed back to its seed."""
    env = ConnectionsEnv(
        config=config, num_categories=num_categories, sampling_params=_SAMPLING_PARAMS
    )
    rows = []
    for seed in range(seed_start, seed_start + num_boards):
        obs = env.reset(seed)
        _SEED_BY_PROMPT[obs["prompt"]] = seed
        rows.append({"prompt": obs["prompt"]})
    return Dataset.from_list(rows)


def make_rollout_func(model, tokenizer, env, device, gen_kwargs):
    """Bind driver state into a TRL rollout function."""

    def rollout_func(prompts, args, processing_class):
        """Play one episode per (already repeated) prompt; return TRL fields."""
        out = {
            "prompt_ids": [], "completion_ids": [], "logprobs": [],
            "env_mask": [], "won": [], "mistakes": [], "tool_calls": [],
        }
        for prompt in prompts:
            seed = _SEED_BY_PROMPT[prompt]
            ep = play_episode(model, tokenizer, env, seed, device=device, **gen_kwargs)
            out["prompt_ids"].append(ep.token_ids[: ep.prompt_len])
            out["completion_ids"].append(ep.token_ids[ep.prompt_len :])
            out["logprobs"].append(ep.logprobs)
            out["env_mask"].append(ep.env_mask)
            out["won"].append(ep.won)
            out["mistakes"].append(ep.mistakes)
            out["tool_calls"].append(ep.tool_calls)
        return out

    return rollout_func


def terminal_reward(completions=None, won=None, mistakes=None, tool_calls=None, **kwargs):
    """Score each episode with the terminal reward from its rollout stats."""
    return [
        episode_reward(w, m, t) for w, m, t in zip(won, mistakes, tool_calls)
    ]


def main() -> None:
    """Parse arguments and run agentic GRPO."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--config", default=None)
    parser.add_argument("--num-categories", type=int, default=2)
    parser.add_argument("--num-boards", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--init-lora", default=None,
                        help="path to the STaR SFT adapter to start from")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output", default="outputs/grpo-agentic")
    args = parser.parse_args()

    config = None
    if args.config:
        with open(args.config) as handle:
            config = json.load(handle)
    dataset = build_dataset(args.num_boards, args.seed_start, args.num_categories, config)
    env = ConnectionsEnv(
        config=config, num_categories=args.num_categories, sampling_params=_SAMPLING_PARAMS
    )

    device = _pick_device()
    cuda = device == "cuda"
    use_bf16 = cuda and torch.cuda.get_device_capability()[0] >= 8

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if cuda else None
    )
    if args.init_lora:
        model = PeftModel.from_pretrained(model, args.init_lora, is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules="all-linear",
        )
    model.to(device)

    grpo_config = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens * 4,
        temperature=args.temperature,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=20,
        report_to="none",
        bf16=use_bf16,
        fp16=cuda and not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    gen_kwargs = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens}
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=terminal_reward,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
        rollout_func=make_rollout_func(model, tokenizer, env, device, gen_kwargs),
    )
    for _callback in (ProgressCallback, PrinterCallback):
        trainer.remove_callback(_callback)
    trainer.add_callback(CompactLogCallback())
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
```

Known adaptation points from Step 1 (fix during implementation, note in the header comment):

- if TRL masks loss via `completion_mask` instead of a custom `env_mask` key, rename;
- if `rollout_func` receives already-templated text or token ids instead of raw prompt strings, key `_SEED_BY_PROMPT` accordingly (e.g. strip template or key by the raw prompt substring);
- generation inside `play_episode` uses the driver's `model.generate` while GRPOTrainer wraps the model for training — verify the rollout uses the current (unwrapped, adapter-enabled) weights each step (`trainer.model`), not a stale reference.

- [ ] **Step 3: MPS smoke run**

Run: `./.venv-train/bin/python train/grpo_agentic.py --model Qwen/Qwen3-0.6B --num-categories 2 --num-boards 4 --num-generations 2 --batch-size 2 --max-steps 2 --max-new-tokens 256`
Expected: 2 training steps complete; `CompactLogCallback` lines show finite reward in [-2, 1]; no shape/mask errors. (256-token budget is fine for a smoke run — episodes will mostly lose, that's OK.)

- [ ] **Step 4: env_mask sanity check**

During the smoke run confirm no loss is attributed to env tokens: assert in a quick debug print (or breakpoint) that the per-token loss mask TRL builds matches `env_mask` zeros for injected turns. Simplest check: run once with a guess scripted long feedback and verify `completion_mask`/`env_mask` sum equals model-token count from the driver.

- [ ] **Step 5: Full test suite**

Run: `./.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add train/grpo_agentic.py
git commit -m "feat(train): agentic GRPO with rollout_func"
```

---

### Task 6: Empirical difficulty ordering

Mentor's difficulty-estimator idea, empirical variant (locked with user):
difficulty of a board = the model's own win rate from the win@k probe. The
probe writes per-board stats to JSONL; each GRPO stage trains on the easiest
seeds from that pool. This is board _selection_, not in-run ordering — the
TRL sampler shuffles dataset rows, so ordering rows would not survive.

**Files:**

- Create: `train/difficulty.py`
- Create: `tests/test_difficulty.py`
- Modify: `train/probe_wink.py` (add `--out-boards`)
- Modify: `train/grpo_agentic.py` (add `--seed-order`)

**Interfaces:**

- Consumes: probe loop internals from Task 3 (`episodes`, `c`, `n` per board), `build_dataset` and CLI from Task 5.
- Produces: `load_seed_order(path: str) -> list[int]` (easiest seed first); JSONL record schema `{"seed": int, "episodes": int, "wins": int, "win_rate": float, "mean_mistakes": float}` shared by probe (writer) and trainer (reader).

- [ ] **Step 1: Write the failing test**

`tests/test_difficulty.py`:

```python
"""Tests for empirical difficulty ordering of board seeds."""

import json

from train.difficulty import load_seed_order


def _write(tmp_path, records):
    """Write records as one-per-line JSONL and return the path."""
    path = tmp_path / "boards.jsonl"
    with open(path, "w") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    return str(path)


def test_orders_easiest_first(tmp_path):
    path = _write(tmp_path, [
        {"seed": 1, "episodes": 8, "wins": 0, "win_rate": 0.0, "mean_mistakes": 4.0},
        {"seed": 2, "episodes": 8, "wins": 6, "win_rate": 0.75, "mean_mistakes": 1.0},
        {"seed": 3, "episodes": 8, "wins": 2, "win_rate": 0.25, "mean_mistakes": 2.5},
    ])
    assert load_seed_order(path) == [2, 3, 1]


def test_ties_break_by_mistakes_then_seed(tmp_path):
    path = _write(tmp_path, [
        {"seed": 5, "episodes": 8, "wins": 4, "win_rate": 0.5, "mean_mistakes": 2.0},
        {"seed": 4, "episodes": 8, "wins": 4, "win_rate": 0.5, "mean_mistakes": 1.0},
        {"seed": 3, "episodes": 8, "wins": 4, "win_rate": 0.5, "mean_mistakes": 1.0},
    ])
    assert load_seed_order(path) == [3, 4, 5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_difficulty.py -q --tb=short`
Expected: FAIL with `ModuleNotFoundError: No module named 'train.difficulty'`

- [ ] **Step 3: Implement `train/difficulty.py`**

```python
"""Empirical board-difficulty ordering from win@k probe results.

``train/probe_wink.py --out-boards`` writes one JSONL record per board with
the model's observed win rate; this module turns that file into a seed list
ordered easiest-first. Curriculum stages train on the easiest slice of the
probed pool (board selection, not in-run ordering — the TRL sampler
shuffles dataset rows).
"""

from __future__ import annotations

import json


def load_seed_order(path: str) -> list[int]:
    """Return board seeds ordered easiest-first by probed win rate.

    Ties break by lower mean mistakes, then by seed for determinism.
    """
    with open(path) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    records.sort(key=lambda r: (-r["win_rate"], r["mean_mistakes"], r["seed"]))
    return [r["seed"] for r in records]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_difficulty.py -q --tb=short`
Expected: 2 passed

- [ ] **Step 5: Wire `--out-boards` into `train/probe_wink.py`**

Add the argument after `--max-new-tokens`:

```python
    parser.add_argument("--out-boards", default=None,
                        help="write per-board JSONL (seed, wins, win_rate, mean_mistakes) "
                             "for difficulty ordering (train/difficulty.py)")
```

Collect records inside the board loop (after `c` is computed) and write the file after the loop, before the summary:

```python
    board_records = []
    for i, seed in enumerate(range(args.seed_start, args.seed_start + args.num_boards), start=1):
        ...  # existing loop body
        board_records.append({
            "seed": seed,
            "episodes": n,
            "wins": c,
            "win_rate": round(c / n, 4),
            "mean_mistakes": round(sum(ep.mistakes for ep in episodes) / n, 3),
        })

    if args.out_boards:
        with open(args.out_boards, "w") as handle:
            for rec in board_records:
                handle.write(json.dumps(rec) + "\n")
```

- [ ] **Step 6: Wire `--seed-order` into `train/grpo_agentic.py`**

Add the import:

```python
from train.difficulty import load_seed_order
```

Replace `build_dataset` with a version that accepts an explicit seed list:

```python
def build_dataset(num_boards, seed_start, num_categories, config, seeds=None):
    """One row per board: the rendered initial prompt, keyed back to its seed.

    ``seeds`` (easiest-first, from ``load_seed_order``) overrides the
    contiguous ``seed_start`` range; only the first ``num_boards`` seeds are
    used, so the training pool is the easiest slice of the probed pool.
    """
    env = ConnectionsEnv(
        config=config, num_categories=num_categories, sampling_params=_SAMPLING_PARAMS
    )
    if seeds is None:
        seeds = range(seed_start, seed_start + num_boards)
    else:
        seeds = seeds[:num_boards]
    rows = []
    for seed in seeds:
        obs = env.reset(seed)
        _SEED_BY_PROMPT[obs["prompt"]] = seed
        rows.append({"prompt": obs["prompt"]})
    return Dataset.from_list(rows)
```

Add the CLI argument after `--seed-start`:

```python
    parser.add_argument("--seed-order", default=None,
                        help="per-board JSONL from probe_wink --out-boards; train on the "
                             "easiest --num-boards seeds instead of the seed-start range")
```

Update the `build_dataset` call in `main`:

```python
    seeds = load_seed_order(args.seed_order) if args.seed_order else None
    dataset = build_dataset(
        args.num_boards, args.seed_start, args.num_categories, config, seeds=seeds
    )
```

- [ ] **Step 7: Full test suite**

Run: `./.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add train/difficulty.py tests/test_difficulty.py train/probe_wink.py train/grpo_agentic.py
git commit -m "feat(train): empirical difficulty ordering"
```

---

### Task 7: Curriculum runbook (Kaggle, no code)

**Files:**

- Modify: none in repo — this is the operational sequence for the Kaggle notebook. Keep it in this plan; copy cells into the notebook as they are reached.

**Interfaces:**

- Consumes: all CLIs from Tasks 3–6.
- Produces: trained adapters per curriculum stage.

- [ ] **Step 1: Gate 2-cat with the win@k probe**

```bash
!python train/probe_wink.py --model Qwen/Qwen3-1.7B --num-categories 2 --num-boards 96 --num-episodes 4 --out-boards data/boards_2cat.jsonl
```

Gate: proceed only if `win_at_k["1"] >= 0.05` (GRPO/STaR need successes to amplify). If 0 — investigate prompt/budget before training.

Probe pool (`--num-boards 96`) must be ≥ the GRPO `--num-boards` (64) so the difficulty ordering has a slice to select from; `--out-boards` feeds Step 3.

- [ ] **Step 2: STaR on 2-cat**

```bash
!python train/star_sft.py sample --model Qwen/Qwen3-1.7B --num-categories 2 --num-boards 200 --episodes-per-board 4 --out data/star_2cat.jsonl
!python train/star_sft.py train --model Qwen/Qwen3-1.7B --data data/star_2cat.jsonl --output outputs/star-sft-2cat
```

Then re-probe with the adapter merged/loaded; expect win@1 up vs Step 1.

- [ ] **Step 3: Agentic GRPO on 2-cat from the STaR adapter**

```bash
!python train/grpo_agentic.py --model Qwen/Qwen3-1.7B --num-categories 2 --init-lora outputs/star-sft-2cat --seed-order data/boards_2cat.jsonl --max-steps 50 --output outputs/grpo-agentic-2cat
```

`--seed-order` selects the 64 easiest probed boards as the training pool (Task 6). Watch: reward EMA rising, no persistent zero-grad lines, `truncated` low.

- [ ] **Step 4: Advance the curriculum**

Advance 2→3 when post-GRPO probe `win_at_k["1"] >= 0.6` on 2-cat; repeat probe → (STaR if win@k collapsed) → GRPO at 3-cat, then 4-cat with the same threshold. Each stage starts from the previous stage's adapter via `--init-lora`, re-probes with `--out-boards data/boards_Ncat.jsonl`, and trains with the matching `--seed-order`.

- [ ] **Step 5: Commit checked-off plan progress**

```bash
git add docs/superpowers/plans/2026-07-02-agentic-grpo-terminal-reward.md
git commit -m "docs: record curriculum run results"
```

---

## Backlog (explicitly out of scope)

- Saturn-1.5B base-model comparison probe.
- Learned difficulty estimator trained on real NYT games (empirical win-rate ordering implemented in Task 6).
- Batched multi-turn generation in the rollout driver (episodes are sequential; fine for probe/STaR scale, revisit if GRPO wall-clock hurts).
