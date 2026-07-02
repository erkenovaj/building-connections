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
        tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )
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
