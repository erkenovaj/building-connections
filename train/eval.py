"""Full-episode evaluation for the Connections policy (plan §4).

Plays complete games via :class:`ConnectionsEnv` (4 lives, until solve or
loss) on a fixed held-out seed set, greedily decoded for reproducibility, and
reports aggregate metrics. Run before and after GRPO training on the same
seeds to show the quality delta.

Examples::

    ./.venv-train/bin/python train/eval.py --num-seeds 100
    ./.venv-train/bin/python train/eval.py --adapter outputs/grpo-connections --num-seeds 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from connections_gym.env import ConnectionsEnv
from connections_gym.prompt import parse_guess

_SAMPLING_PARAMS = {
    "max_attempts": 20,
    "per_solve_timeout_seconds": 0.5,
    "total_timeout_seconds": 5.0,
}


def _pick_device() -> str:
    """Return the best available torch device string."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_policy(model_id: str, adapter: str | None, device: str):
    """Load the base model (optionally with a LoRA adapter) and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_guess_text(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> tuple[str, int]:
    """Greedily decode the model's response to a single board-state prompt.

    Returns the decoded text and the count of newly generated tokens (a proxy
    for whether the model is hitting the ``max_new_tokens`` cap).
    """
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), int(new_tokens.shape[0])


def play_episode(model, tokenizer, env, seed, device, max_steps, max_new_tokens) -> dict:
    """Play one full greedy episode and return its per-episode metrics."""
    obs = env.reset(seed)
    done = False
    steps = invalid = 0
    total_reward = 0.0
    gen_tokens = 0
    info: dict = {"status": "playing", "mistakes": 0, "solved_keys": []}
    while not done and steps < max_steps:
        text, n_new = generate_guess_text(model, tokenizer, obs["prompt"], device, max_new_tokens)
        gen_tokens += n_new
        guess = parse_guess(text, obs["remaining"])
        items = guess["items"] if guess else []
        obs, reward_value, done, info = env.step(items)
        if info["k"] is None:
            invalid += 1
        total_reward += reward_value
        steps += 1
    return {
        "won": info["status"] == "won",
        "groups_solved": len(info["solved_keys"]),
        "mistakes": info["mistakes"],
        "steps": steps,
        "invalid": invalid,
        "total_reward": total_reward,
        "gen_tokens": gen_tokens,
    }


def aggregate(episodes: list[dict]) -> dict:
    """Reduce per-episode metrics to the before/after summary (plan §4)."""
    n = len(episodes)
    total_steps = sum(e["steps"] for e in episodes) or 1
    return {
        "num_episodes": n,
        "win_rate": sum(e["won"] for e in episodes) / n,
        "mean_groups_solved": sum(e["groups_solved"] for e in episodes) / n,
        "mean_mistakes": sum(e["mistakes"] for e in episodes) / n,
        "invalid_rate": sum(e["invalid"] for e in episodes) / total_steps,
        "mean_reward_per_step": sum(e["total_reward"] for e in episodes) / total_steps,
    }


def main() -> None:
    """Parse arguments, evaluate over the held-out seeds, and print the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=None, help="path to a trained LoRA adapter")
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=1_000_000)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    device = _pick_device()
    model, tokenizer = load_policy(args.model, args.adapter, device)
    env = ConnectionsEnv(sampling_params=_SAMPLING_PARAMS)

    episodes = []
    for i, seed in enumerate(range(args.eval_seed_start, args.eval_seed_start + args.num_seeds), start=1):
        ep = play_episode(
            model, tokenizer, env, seed, device, args.max_steps, args.max_new_tokens
        )
        episodes.append(ep)
        print(
            f"  [{i}/{args.num_seeds}] seed {seed}: groups {ep['groups_solved']} "
            f"mistakes {ep['mistakes']} steps {ep['steps']} invalid {ep['invalid']} "
            f"tok/step {ep['gen_tokens'] / max(ep['steps'], 1):.0f}",
            file=sys.stderr,
            flush=True,
        )

    summary = aggregate(episodes)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
