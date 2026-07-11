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


def new_state(cfg: dict) -> dict:
    """Fresh resume state for a run."""
    return {"run_name": cfg["run_name"], "completed": {},
            "model_path": None, "adapter_path": None, "stopped": None}


def _state_path(run_dir: str) -> str:
    return os.path.join(run_dir, "state.json")


def load_state(run_dir: str) -> dict | None:
    """Load state.json from a run dir; None if the run has no state yet."""
    path = _state_path(run_dir)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def save_state(run_dir: str, state: dict) -> None:
    """Write state.json atomically so a crash mid-save cannot corrupt resume."""
    path = _state_path(run_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(state, handle, indent=2)
    os.replace(tmp, path)


def decide_after_probe(win1: float, gates: dict, star_done: bool) -> str:
    """Gate before GRPO: start, warm-start via STaR (once), or stop."""
    if win1 >= gates["start_grpo"]:
        return "grpo"
    return "stop" if star_done else "star"


def decide_after_eval(win1: float, gates: dict) -> str:
    """Gate after held-out eval: advance the curriculum or stop."""
    return "advance" if win1 >= gates["advance"] else "stop"


def run_step(cmd: list[str], log_path: str) -> str:
    """Run one subprocess, teeing combined stdout+stderr to log and console."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    lines = []
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            lines.append(line)
        proc.wait()
    if proc.returncode != 0:
        raise StepFailed(f"{' '.join(cmd[1:3])} exited {proc.returncode}")
    return "".join(lines)


def parse_last_json(text: str) -> dict | None:
    """Last line of text that parses as a JSON object (probe/STaR summaries)."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _append_metrics(run_dir: str, name: str, metrics: dict | None) -> None:
    with open(os.path.join(run_dir, "metrics.jsonl"), "a") as handle:
        handle.write(json.dumps(
            {"ts": time.time(), "step": name, "metrics": metrics}) + "\n")


def step(state: dict, run_dir: str, name: str, cmd: list[str],
         require_json: bool = False) -> dict:
    """Run a named pipeline step once; on resume, return the recorded result."""
    if name in state["completed"]:
        print(f"pipeline: skip completed step {name}")
        return state["completed"][name]
    log_path = os.path.join(run_dir, "logs",
                            f"{len(state['completed']):02d}-{name}.log")
    print(f"pipeline: run {name}: {' '.join(cmd)}")
    stdout = run_step(cmd, log_path)
    metrics = parse_last_json(stdout)
    if require_json and metrics is None:
        raise StepFailed(f"{name}: no summary JSON found in output")
    record = {"metrics": metrics}
    state["completed"][name] = record
    _append_metrics(run_dir, name, metrics)
    save_state(run_dir, state)
    return record


def probe_cmd(cfg, stage, model, adapter, seeds, out_boards):
    """probe_wink invocation; seeds is the probe or eval config section."""
    cmd = [sys.executable, "train/probe_wink.py",
           "--model", model,
           "--num-categories", str(stage),
           "--num-boards", str(seeds["num_boards"]),
           "--num-episodes", str(seeds["num_episodes"]),
           "--seed-start", str(seeds["seed_start"]),
           "--max-new-tokens", str(cfg["max_new_tokens"])]
    if adapter:
        cmd += ["--adapter", adapter]
    if out_boards:
        cmd += ["--out-boards", out_boards]
    return cmd


def star_sample_cmd(cfg, stage, model, out):
    """star_sft sample invocation (STaR rejection sampling).

    Sampling has no adapter support, so in LoRA mode it plays the base
    model; full-FT mode passes the current full checkpoint as --model.
    """
    star = cfg["star"]
    return [sys.executable, "train/star_sft.py", "sample",
            "--model", model,
            "--num-categories", str(stage),
            "--num-boards", str(star["num_boards"]),
            "--episodes-per-board", str(star["episodes_per_board"]),
            "--seed-start", str(star["seed_start"]),
            "--max-new-tokens", str(cfg["max_new_tokens"]),
            "--out", out]


def star_train_cmd(cfg, stage, model, data, output):
    """star_sft train invocation on the kept trajectories."""
    star = cfg["star"]
    cmd = [sys.executable, "train/star_sft.py", "train",
           "--model", model,
           "--data", data,
           "--batch-size", str(star["batch_size"]),
           "--grad-accum", str(star["grad_accum"]),
           "--lr", str(star["lr"]),
           "--epochs", str(star["epochs"]),
           "--optim", cfg["optim"],
           "--report-to", cfg["report_to"],
           "--run-name", f"{cfg['run_name']}-s{stage}-star",
           "--output", output]
    if star["dft"]:
        cmd.append("--dft")
    if cfg["full_ft"]:
        cmd.append("--full-ft")
    return cmd


def grpo_cmd(cfg, stage, model, adapter, seed_order, output):
    """grpo_agentic invocation on the easiest probed boards."""
    grpo = cfg["grpo"]
    cmd = [sys.executable, "train/grpo_agentic.py",
           "--model", model,
           "--num-categories", str(stage),
           "--num-boards", str(grpo["num_boards"]),
           "--seed-order", seed_order,
           "--max-steps", str(grpo["max_steps"]),
           "--num-generations", str(grpo["num_generations"]),
           "--batch-size", str(grpo["batch_size"]),
           "--grad-accum", str(grpo["grad_accum"]),
           "--lr", str(grpo["lr"]),
           "--max-new-tokens", str(cfg["max_new_tokens"]),
           "--optim", cfg["optim"],
           "--report-to", cfg["report_to"],
           "--run-name", f"{cfg['run_name']}-s{stage}-grpo",
           "--output", output]
    if cfg["full_ft"]:
        cmd.append("--full-ft")
    elif adapter:
        cmd += ["--init-lora", adapter]
    return cmd
