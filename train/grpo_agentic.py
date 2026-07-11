"""Agentic GRPO on full multi-turn episodes with terminal reward (spec этап 5).

A custom TRL ``rollout_func`` plays whole Connections episodes with the
rollout driver; ``env_mask`` keeps injected env-feedback tokens out of the
policy-gradient loss, and each episode is scored once with the terminal
reward from ``connections_gym.episode_reward``. Start from the STaR adapter
via ``--init-lora``.

Run::

    python train/grpo_agentic.py --model Qwen/Qwen3-1.7B --num-categories 2 \
        --init-lora outputs/star-sft-2cat --max-steps 50

Installed TRL 1.7.0 rollout contract (verified in grpo_trainer.py):

- ``rollout_func(prompts, trainer)`` — second argument is the trainer
  instance, not ``(args, processing_class)``.
- ``prompts`` are the raw dataset ``"prompt"`` values, already repeated
  ``num_generations`` times by ``RepeatSampler`` — return one row per entry.
- Required return keys: ``prompt_ids``, ``completion_ids``, ``logprobs``.
  ``env_mask`` is popped from the extras and used as ``tool_mask``
  (1 = model token, 0 = env token, excluded from the loss); all other extra
  keys are forwarded per-index to reward functions as keyword lists.
- The rollout must use the trainer's live (adapter-wrapped) weights:
  ``trainer.accelerator.unwrap_model(trainer.model)``, not the base model
  passed to the constructor.
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
from train.difficulty import load_seed_order
from train.grpo_connections import CompactLogCallback
from train.probe_passk import _SAMPLING_PARAMS, _pick_device
from train.rollout import play_episode

# prompt text -> board seed, filled by build_dataset and read by the rollout
# func (TRL hands the rollout func prompt strings, not dataset rows).
_SEED_BY_PROMPT: dict[str, int] = {}


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


def make_rollout_func(env, gen_kwargs):
    """Bind the episode env and generation settings into a TRL rollout function."""

    def rollout_func(prompts, trainer):
        """Play one episode per (already repeated) prompt; return TRL fields."""
        model = trainer.accelerator.unwrap_model(trainer.model)
        tokenizer = trainer.processing_class
        device = str(trainer.accelerator.device)
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


def parse_args(argv=None):
    """CLI for agentic GRPO; --full-ft and --init-lora are mutually exclusive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--config", default=None)
    parser.add_argument("--num-categories", type=int, default=2)
    parser.add_argument("--num-boards", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-order", default=None,
                        help="per-board JSONL from probe_wink --out-boards; train on the "
                             "easiest --num-boards seeds instead of the seed-start range")
    parser.add_argument("--init-lora", default=None,
                        help="path to the STaR SFT adapter to start from")
    parser.add_argument("--full-ft", action="store_true",
                        help="full fine-tuning: no LoRA, save a full model dir")
    parser.add_argument("--optim", default="adamw_torch",
                        help="optimizer name for GRPOConfig (adamw_bnb_8bit for 8B)")
    parser.add_argument("--report-to", default="none",
                        help="TRL reporting target (comet_ml on the cluster)")
    parser.add_argument("--run-name", default=None,
                        help="experiment run name for the reporting backend")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--no-think", action="store_true",
                        help="disable Qwen3 thinking mode (empty <think> block each turn)")
    parser.add_argument("--output", default="outputs/grpo-agentic")
    args = parser.parse_args(argv)
    if args.full_ft and args.init_lora:
        parser.error("--full-ft conflicts with --init-lora")
    return args


def build_grpo_config(args, cuda, use_bf16):
    """GRPOConfig from CLI args; bf16 on cc>=8 CUDA, fp16 as the fallback."""
    return GRPOConfig(
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
        optim=args.optim,
        report_to=args.report_to,
        run_name=args.run_name,
        bf16=use_bf16,
        fp16=cuda and not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


def main() -> None:
    """Parse arguments and run agentic GRPO."""
    args = parse_args()

    config = None
    if args.config:
        with open(args.config) as handle:
            config = json.load(handle)
    seeds = load_seed_order(args.seed_order) if args.seed_order else None
    dataset = build_dataset(
        args.num_boards, args.seed_start, args.num_categories, config, seeds=seeds
    )
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
        args.model,
        dtype=(torch.bfloat16 if use_bf16 else torch.float16) if cuda else None,
    )
    if args.full_ft:
        peft_config = None
    elif args.init_lora:
        model = PeftModel.from_pretrained(model, args.init_lora, is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules="all-linear",
        )
    model.to(device)

    grpo_config = build_grpo_config(args, cuda, use_bf16)

    gen_kwargs = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": not args.no_think,
    }
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=terminal_reward,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        rollout_func=make_rollout_func(env, gen_kwargs),
    )
    for _callback in (ProgressCallback, PrinterCallback):
        trainer.remove_callback(_callback)
    trainer.add_callback(CompactLogCallback())
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
