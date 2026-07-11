"""Tests for STaR trajectory filtering and the SFT/DFT train config."""

import argparse
import math
import types

import pytest

torch = pytest.importorskip("torch")

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


def _train_args(dft, full_ft=False):
    return argparse.Namespace(
        output="outputs/test-sft", batch_size=4, grad_accum=4, lr=1e-5,
        epochs=2, dft=dft, full_ft=full_ft, optim="adamw_torch",
        report_to="none", run_name=None,
    )


def test_train_config_defaults_to_nll_loss():
    pytest.importorskip("trl")
    from train.star_sft import build_train_config

    config = build_train_config(_train_args(dft=False))
    assert config.loss_type == "chunked_nll"
    assert config.assistant_only_loss is True


def test_train_config_dft_flag_selects_dft_loss():
    pytest.importorskip("trl")
    from train.star_sft import build_train_config

    config = build_train_config(_train_args(dft=True))
    assert config.loss_type == "dft"
    assert config.assistant_only_loss is True


def test_trl_dft_loss_matches_paper_formula():
    """Guard the pinned TRL: dft_loss must be -sg(p)*log p on unmasked tokens."""
    pytest.importorskip("trl")
    from trl.trainer.sft_trainer import dft_loss

    # Position 0 predicts token 0 with logits [2, 0]; position 1 predicts
    # token 1 with logits [0, 1]; first label is masked out by the shift.
    logits = torch.tensor([[[2.0, 0.0], [0.0, 1.0], [5.0, 5.0]]])
    labels = torch.tensor([[-100, 0, 1]])
    outputs = types.SimpleNamespace(logits=logits)

    p1 = math.exp(2.0) / (math.exp(2.0) + 1.0)
    p2 = math.exp(1.0) / (1.0 + math.exp(1.0))
    expected = -(p1 * math.log(p1) + p2 * math.log(p2)) / 2

    loss = dft_loss(outputs, labels)
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_train_config_forwards_optim_report_to_run_name():
    pytest.importorskip("trl")
    from train.star_sft import build_train_config

    args = _train_args(dft=False)
    args.optim = "adamw_bnb_8bit"
    args.report_to = "comet_ml"
    args.run_name = "smoke-s2-star"
    config = build_train_config(args)
    assert config.optim == "adamw_bnb_8bit"
    assert config.report_to == ["comet_ml"]
    assert config.run_name == "smoke-s2-star"


def test_full_ft_disables_peft_config():
    pytest.importorskip("peft")
    from train.star_sft import build_peft_config

    assert build_peft_config(_train_args(dft=False, full_ft=True)) is None


def test_default_mode_builds_lora_config():
    pytest.importorskip("peft")
    from train.star_sft import build_peft_config

    peft_config = build_peft_config(_train_args(dft=False))
    assert peft_config is not None
    assert peft_config.r == 16
