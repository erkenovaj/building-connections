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
                    enable_thinking=not args.no_think,
                )
                played += 1
                record = episode_to_record(episode)
                if record is not None:
                    out.write(json.dumps(record) + "\n")
                    kept += 1
            print(f"  seed {seed}: kept {kept}/{played}", file=sys.stderr, flush=True)
            out.flush()  # keep JSONL durable across Colab disconnects
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
    p_sample.add_argument("--no-think", action="store_true",
                          help="disable Qwen3 thinking mode (empty <think> block each turn)")
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
