"""Tests for grpo_agentic CLI parsing and GRPOConfig construction."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from train.grpo_agentic import build_grpo_config, parse_args


def test_parse_args_defaults():
    args = parse_args([])
    assert args.full_ft is False
    assert args.optim == "adamw_torch"
    assert args.report_to == "none"
    assert args.run_name is None


def test_full_ft_conflicts_with_init_lora():
    with pytest.raises(SystemExit):
        parse_args(["--full-ft", "--init-lora", "outputs/star-sft"])


def test_grpo_config_forwards_new_flags():
    args = parse_args([
        "--optim", "adamw_bnb_8bit", "--report-to", "comet_ml",
        "--run-name", "qwen3-8b-s2-grpo",
    ])
    config = build_grpo_config(args, cuda=False, use_bf16=False)
    assert config.optim == "adamw_bnb_8bit"
    assert config.report_to == ["comet_ml"]
    assert config.run_name == "qwen3-8b-s2-grpo"
    assert config.bf16 is False and config.fp16 is False


def test_grpo_config_bf16_precision_flags():
    args = parse_args([])
    config = build_grpo_config(args, cuda=True, use_bf16=True)
    assert config.bf16 is True and config.fp16 is False
