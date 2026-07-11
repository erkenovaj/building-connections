"""Tests for the pipeline orchestrator: config, state, gates, steps."""

import copy

import pytest

yaml = pytest.importorskip("yaml")

from train.pipeline import ConfigError, load_config, validate_config

VALID_CFG = {
    "run_name": "t",
    "model": "Qwen/Qwen3-0.6B",
    "full_ft": True,
    "optim": "adamw_torch",
    "report_to": "none",
    "max_new_tokens": 64,
    "stages": [2],
    "gates": {"start_grpo": 0.05, "advance": 0.6},
    "probe": {"num_boards": 2, "num_episodes": 2, "seed_start": 1000000},
    "eval": {"num_boards": 2, "num_episodes": 2, "seed_start": 2000000},
    "star": {"num_boards": 2, "episodes_per_board": 2, "seed_start": 150,
             "batch_size": 1, "grad_accum": 1, "lr": 1.0e-5, "epochs": 1,
             "dft": False},
    "grpo": {"num_boards": 2, "max_steps": 2, "num_generations": 2,
             "batch_size": 2, "grad_accum": 1, "lr": 1.0e-5},
}


def make_cfg(**overrides):
    cfg = copy.deepcopy(VALID_CFG)
    cfg.update(overrides)
    return cfg


def test_load_config_merges_defaults(tmp_path):
    raw = make_cfg()
    for key in ("full_ft", "optim", "report_to", "max_new_tokens"):
        del raw[key]
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(str(path))
    assert cfg["full_ft"] is False
    assert cfg["optim"] == "adamw_torch"
    assert cfg["report_to"] == "none"
    assert cfg["max_new_tokens"] == 1024


def test_unknown_top_level_key_rejected():
    with pytest.raises(ConfigError, match="unknown"):
        validate_config(make_cfg(learning_rate=1.0e-5))


def test_missing_required_key_rejected():
    cfg = make_cfg()
    del cfg["gates"]
    with pytest.raises(ConfigError, match="gates"):
        validate_config(cfg)


def test_missing_section_key_rejected():
    cfg = make_cfg()
    del cfg["grpo"]["max_steps"]
    with pytest.raises(ConfigError, match="grpo.max_steps"):
        validate_config(cfg)


def test_string_lr_rejected():
    """PyYAML parses bare 1e-5 as a string; catch it at validation time."""
    cfg = make_cfg()
    cfg["star"]["lr"] = "1e-5"
    with pytest.raises(ConfigError, match="star.lr"):
        validate_config(cfg)
