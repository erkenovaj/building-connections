"""Estimate base-model pass@k on the single-group Connections subtask.

For each held-out board this samples ``N`` completions at temperature ``T``
from the (untrained) policy, scores each by its overlap with the best unsolved
group, and reports the unbiased pass@k estimator for a few k plus how close the
model gets (overlap histogram).

``pass@k ~ 0`` means GRPO has no successes to amplify: the verifiable reward
never fires, so the run can only learn the trivial format gain. Use this to
gate any RL run, and to compare base checkpoints / temperatures before training.

Examples::

    python train/probe_passk.py --model Qwen/Qwen2.5-1.5B-Instruct --num-boards 32 --num-samples 64
    python train/probe_passk.py --model Qwen/Qwen3-1.7B --num-boards 32 --num-samples 64 --max-new-tokens 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from connections_gym.env import ConnectionsEnv
from connections_gym.prompt import parse_guess
from connections_gym.reward import reward

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


def load_policy(model_id: str, device: str):
    """Load the base model (fp16 on CUDA) and its tokenizer for sampling."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def sample_overlaps(
    model,
    tokenizer,
    prompt: str,
    remaining,
    groups,
    device: str,
    n: int,
    gen_batch: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> tuple[list[int], list[bool]]:
    """Sample ``n`` completions for one board (chunked to cap KV memory).

    Returns per-sample best overlap with any group (0..4; 0 for an invalid
    guess) and a parallel list of validity flags.
    """
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    overlaps: list[int] = []
    valids: list[bool] = []
    done = 0
    while done < n:
        cur = min(gen_batch, n - done)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=cur,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        for seq in out:
            text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            guess = parse_guess(text, remaining)
            items = guess["items"] if guess else []
            _, info = reward(items, groups, [])
            k = info["k"]
            valids.append(k is not None)
            overlaps.append(k if k is not None else 0)
        done += cur
    return overlaps, valids


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: probability a random k-subset of n samples has a solve."""
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main() -> None:
    """Sample completions over held-out boards and print the pass@k summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-boards", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--gen-batch", type=int, default=16, help="completions per generate() call (caps GPU memory)")
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    device = _pick_device()
    model, tokenizer = load_policy(args.model, device)
    env = ConnectionsEnv(sampling_params=_SAMPLING_PARAMS)

    n = args.num_samples
    ks = sorted({k for k in (1, 4, 16, n) if k <= n})

    passk_sums = {k: 0.0 for k in ks}
    boards_with_solve = 0
    best_overlap_hist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    valid_total = 0
    sample_total = 0

    for i, seed in enumerate(range(args.seed_start, args.seed_start + args.num_boards), start=1):
        obs = env.reset(seed)
        overlaps, valids = sample_overlaps(
            model, tokenizer, obs["prompt"], obs["remaining"], env.groups,
            device, n, args.gen_batch, args.temperature, args.top_p, args.max_new_tokens,
        )
        c = sum(1 for o in overlaps if o == 4)
        for k in ks:
            passk_sums[k] += pass_at_k(n, c, k)
        boards_with_solve += int(c > 0)
        best = max(overlaps) if overlaps else 0
        best_overlap_hist[best] += 1
        valid_total += sum(valids)
        sample_total += len(overlaps)
        print(
            f"  [{i}/{args.num_boards}] seed {seed}: solves {c}/{n} "
            f"best_overlap {best} valid {sum(valids)}/{len(valids)}",
            file=sys.stderr,
            flush=True,
        )

    nb = args.num_boards
    summary = {
        "model": args.model,
        "num_boards": nb,
        "num_samples": n,
        "temperature": args.temperature,
        "pass_at_k": {str(k): round(passk_sums[k] / nb, 4) for k in ks},
        "boards_with_any_solve": boards_with_solve,
        "frac_boards_solvable": round(boards_with_solve / nb, 4),
        "best_overlap_hist": best_overlap_hist,
        "valid_rate": round(valid_total / max(sample_total, 1), 4),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
