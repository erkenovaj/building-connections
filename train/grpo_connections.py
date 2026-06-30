"""GRPO training loop for the Connections single-group policy (plan §2).

Contextual-bandit setup: each dataset row is one board state (rendered prompt
plus the board's ground truth as JSON side columns). ``GRPOTrainer`` samples G
completions per prompt and scores each with the dense per-guess reward; the
group-relative advantage uses no value net, so the chain-of-thought is
preserved (the SATURN setup) for downstream faithfulness research.

Run from the repo root with the training venv, e.g.::

    ./.venv-train/bin/python train/grpo_connections.py --max-steps 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reduce allocator fragmentation on small GPUs (e.g. Kaggle T4); must be set
# before torch initializes its CUDA caching allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from trl import GRPOConfig, GRPOTrainer

from connections_gym.env import ConnectionsEnv
from connections_gym.prompt import parse_guess
from connections_gym.reward import reward

_SAMPLING_PARAMS = {
    "max_attempts": 20,
    "per_solve_timeout_seconds": 0.5,
    "total_timeout_seconds": 5.0,
}


def build_dataset(num_boards: int, seed_start: int) -> Dataset:
    """Build a dataset of initial-board prompts (all groups unsolved, plan v1).

    Each row carries the conversational ``prompt`` plus JSON side columns
    (``remaining_json``, ``board_groups_json``, ``solved_keys_json``) that the
    reward function needs to parse and score completions.
    """
    env = ConnectionsEnv(sampling_params=_SAMPLING_PARAMS)
    rows = []
    for seed in range(seed_start, seed_start + num_boards):
        obs = env.reset(seed)
        rows.append(
            {
                "prompt": [{"role": "user", "content": obs["prompt"]}],
                "remaining_json": json.dumps(obs["remaining"]),
                "board_groups_json": json.dumps(env.groups),
                "solved_keys_json": json.dumps([]),
            }
        )
    return Dataset.from_list(rows)


def _completion_text(completion) -> str:
    """Extract assistant text from a completion (conversational or plain str)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[0].get("content", "")
    return ""


def connections_reward(
    completions,
    remaining_json,
    board_groups_json,
    solved_keys_json,
    **kwargs,
):
    """Score each completion with the dense per-guess reward (plan §3)."""
    rewards = []
    for completion, rem_j, grp_j, solved_j in zip(
        completions, remaining_json, board_groups_json, solved_keys_json
    ):
        remaining = json.loads(rem_j)
        groups = json.loads(grp_j)
        solved = json.loads(solved_j)
        guess = parse_guess(_completion_text(completion), remaining)
        items = guess["items"] if guess else []
        reward_value, _ = reward(items, groups, solved)
        rewards.append(reward_value)
    return rewards


def _fmt_eta(seconds: float) -> str:
    """Format a seconds duration compactly as ``HhMMm`` / ``MmSSs`` / ``Ss``."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class CompactLogCallback(TrainerCallback):
    """Print one tidy line per GRPO step instead of TRL's raw metrics dict.

    Surfaces only the fields that matter for this reward setup (reward
    mean/std, completion clip ratio and length, grad/loss/entropy, lr) plus a
    self-computed ETA, and flags ``zero-grad`` steps where the whole group tied
    so nothing was learned. Registered in place of the default progress/printer
    callbacks.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        """Record the start time, reset the reward EMA, and print the header."""
        self._t0 = time.time()
        self._ema = None
        print(
            "   step   |      reward       |     ema     |  completions  |  grad    loss    ent |    lr    | pace",
            flush=True,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Format one training-step (or the final summary) log line."""
        if not logs:
            return
        if "train_runtime" in logs:
            print(
                f"  done: {state.global_step} steps in {_fmt_eta(logs['train_runtime'])}",
                flush=True,
            )
            return
        if "reward" not in logs and "loss" not in logs:
            return
        step, total = state.global_step, state.max_steps
        rew = logs.get("reward", float("nan"))
        rstd = logs.get("reward_std", 0.0)
        if rew == rew:  # update the smoothed reward only on real (non-NaN) values
            self._ema = rew if self._ema is None else 0.1 * rew + 0.9 * self._ema
        ema = self._ema if self._ema is not None else float("nan")
        clip = logs.get("completions/clipped_ratio", 0.0) * 100
        mlen = logs.get("completions/mean_length", 0.0)
        grad = logs.get("grad_norm", 0.0)
        loss = logs.get("loss", 0.0)
        ent = logs.get("entropy", float("nan"))
        lr = logs.get("learning_rate", 0.0)
        sit = (time.time() - self._t0) / max(step, 1)
        eta = _fmt_eta(sit * (total - step))
        flag = "  zero-grad" if not grad else ""
        print(
            f"  {step:>3}/{total:<3} | {rew:>7.3f} +/- {rstd:<5.3f} | ema {ema:>7.3f} | "
            f"clip{clip:>4.0f}% l{mlen:>4.0f} | {grad:>6.3f} {loss:>7.4f} {ent:>5.2f} | "
            f"{lr:>8.2e} | {sit:>4.1f}s eta {eta}{flag}",
            flush=True,
        )


def main() -> None:
    """Parse arguments and run the GRPO training loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-boards", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", default="linear")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--output", default="outputs/grpo-connections")
    args = parser.parse_args()

    dataset = build_dataset(args.num_boards, args.seed_start)

    cuda = torch.cuda.is_available()
    use_bf16 = cuda and torch.cuda.get_device_capability()[0] >= 8

    config = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
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
        use_vllm=args.use_vllm,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=connections_reward,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    for _callback in (ProgressCallback, PrinterCallback):
        trainer.remove_callback(_callback)
    trainer.add_callback(CompactLogCallback())
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
