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
