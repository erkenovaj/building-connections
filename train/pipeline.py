"""Curriculum pipeline orchestrator for unattended cluster runs.

Runs the notebook flow — win@k probe -> STaR SFT warm start (only when the
start gate fails, one retry) -> agentic GRPO -> held-out eval — per
curriculum stage (2/3/4 categories), driven by one YAML config. Every step
is a child-process invocation of the existing train scripts, so VRAM is
fully released between steps and one crash cannot corrupt the orchestrator.
Progress is checkpointed to ``<run_dir>/state.json`` after every step;
``--resume`` skips completed steps. Design:
``docs/superpowers/specs/2026-07-11-training-pipeline-design.md``.

Run::

    python train/pipeline.py --config configs_train/smoke.yaml \
        --run-dir outputs/runs/smoke --resume
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ConfigError(ValueError):
    """Invalid pipeline YAML config."""


class StepFailed(RuntimeError):
    """A pipeline subprocess exited non-zero or produced no summary JSON."""


REQUIRED_KEYS = {"run_name", "model", "stages", "gates", "probe", "eval", "star", "grpo"}
DEFAULTS = {"full_ft": False, "optim": "adamw_torch", "report_to": "none",
            "max_new_tokens": 1024}
SECTION_KEYS = {
    "gates": {"start_grpo", "advance"},
    "probe": {"num_boards", "num_episodes", "seed_start"},
    "eval": {"num_boards", "num_episodes", "seed_start"},
    "star": {"num_boards", "episodes_per_board", "seed_start", "batch_size",
             "grad_accum", "lr", "epochs", "dft"},
    "grpo": {"num_boards", "max_steps", "num_generations", "batch_size",
             "grad_accum", "lr"},
}


def validate_config(cfg: dict) -> None:
    """Reject unknown keys, missing keys, and non-numeric learning rates."""
    known = REQUIRED_KEYS | set(DEFAULTS)
    unknown = set(cfg) - known
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    missing = REQUIRED_KEYS - set(cfg)
    if missing:
        raise ConfigError(f"missing config keys: {sorted(missing)}")
    for section, keys in SECTION_KEYS.items():
        unknown = set(cfg[section]) - keys
        if unknown:
            raise ConfigError(f"unknown keys in {section}: {sorted(unknown)}")
        for key in sorted(keys - set(cfg[section])):
            raise ConfigError(f"missing config key: {section}.{key}")
    for section in ("star", "grpo"):
        lr = cfg[section]["lr"]
        if not isinstance(lr, (int, float)):
            # bare 1e-5 in YAML parses as a string; configs must write 1.0e-5
            raise ConfigError(f"{section}.lr must be a number, got {lr!r}")


def load_config(path: str) -> dict:
    """Load and validate a pipeline YAML config, merging defaults."""
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: config must be a YAML mapping")
    cfg = {**DEFAULTS, **cfg}
    validate_config(cfg)
    return cfg
