"""Tests for the pipeline orchestrator: config, state, gates, steps."""

import copy
import json
import os
import sys

import pytest

yaml = pytest.importorskip("yaml")

from train.pipeline import ConfigError, load_config, validate_config
from train.pipeline import (
    StepFailed, decide_after_eval, decide_after_probe, load_state,
    new_state, parse_last_json, run_step, save_state, step,
)

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


GATES = {"start_grpo": 0.05, "advance": 0.6}


def test_decide_after_probe():
    assert decide_after_probe(0.1, GATES, star_done=False) == "grpo"
    assert decide_after_probe(0.0, GATES, star_done=False) == "star"
    assert decide_after_probe(0.0, GATES, star_done=True) == "stop"
    assert decide_after_probe(0.05, GATES, star_done=True) == "grpo"


def test_decide_after_eval():
    assert decide_after_eval(0.6, GATES) == "advance"
    assert decide_after_eval(0.59, GATES) == "stop"


def test_state_roundtrip(tmp_path):
    run_dir = str(tmp_path)
    assert load_state(run_dir) is None
    state = new_state(make_cfg())
    state["completed"]["s2-probe-base"] = {"metrics": {"win_at_k": {"1": 0.1}}}
    save_state(run_dir, state)
    assert load_state(run_dir) == state


def test_run_step_tees_output_and_returns_it(tmp_path):
    log = str(tmp_path / "logs" / "00-x.log")
    code = "import sys; print('progress', file=sys.stderr); print('{\"a\": 1}')"
    out = run_step([sys.executable, "-c", code], log)
    assert '{"a": 1}' in out
    assert '{"a": 1}' in open(log).read()


def test_run_step_raises_on_nonzero_exit(tmp_path):
    log = str(tmp_path / "00-x.log")
    with pytest.raises(StepFailed):
        run_step([sys.executable, "-c", "raise SystemExit(3)"], log)


def test_parse_last_json_picks_last_json_object():
    text = 'noise\n{"a": 1}\n{"b": 2}\ntrailing noise\n'
    assert parse_last_json(text) == {"b": 2}
    assert parse_last_json("no json here\n") is None


def test_step_records_skips_and_writes_metrics(tmp_path):
    run_dir = str(tmp_path)
    state = new_state(make_cfg())
    cmd = [sys.executable, "-c", "print('{\"win_at_k\": {\"1\": 0.5}}')"]
    rec = step(state, run_dir, "s2-probe-base", cmd, require_json=True)
    assert rec["metrics"]["win_at_k"]["1"] == 0.5
    assert "s2-probe-base" in load_state(run_dir)["completed"]
    lines = open(os.path.join(run_dir, "metrics.jsonl")).read().splitlines()
    assert json.loads(lines[0])["step"] == "s2-probe-base"

    # second call skips the subprocess: a failing cmd is never executed
    rec2 = step(state, run_dir, "s2-probe-base",
                [sys.executable, "-c", "raise SystemExit(1)"], require_json=True)
    assert rec2 == rec


def test_step_requires_json_when_asked(tmp_path):
    state = new_state(make_cfg())
    with pytest.raises(StepFailed):
        step(state, str(tmp_path), "s2-probe-base",
             [sys.executable, "-c", "print('no json')"], require_json=True)


from train.pipeline import grpo_cmd, probe_cmd, star_sample_cmd, star_train_cmd


def test_probe_cmd_includes_adapter_and_boards_only_when_given():
    cfg = make_cfg()
    cmd = probe_cmd(cfg, 2, "m", None, cfg["probe"], None)
    assert cmd[:2] == [sys.executable, "train/probe_wink.py"]
    assert "--adapter" not in cmd and "--out-boards" not in cmd
    assert cmd[cmd.index("--seed-start") + 1] == "1000000"

    cmd = probe_cmd(cfg, 2, "m", "outputs/star", cfg["eval"], "b.jsonl")
    assert cmd[cmd.index("--adapter") + 1] == "outputs/star"
    assert cmd[cmd.index("--out-boards") + 1] == "b.jsonl"
    assert cmd[cmd.index("--seed-start") + 1] == "2000000"


def test_star_train_cmd_full_ft_and_dft_flags():
    cfg = make_cfg()
    cfg["star"]["dft"] = True
    cmd = star_train_cmd(cfg, 3, "m", "d.jsonl", "out")
    assert cmd[1:3] == ["train/star_sft.py", "train"]
    assert "--full-ft" in cmd and "--dft" in cmd
    assert cmd[cmd.index("--run-name") + 1] == "t-s3-star"

    cfg = make_cfg(full_ft=False)
    cmd = star_train_cmd(cfg, 3, "m", "d.jsonl", "out")
    assert "--full-ft" not in cmd and "--dft" not in cmd


def test_grpo_cmd_full_ft_vs_lora_modes():
    cfg = make_cfg()
    cmd = grpo_cmd(cfg, 2, "m", None, "b.jsonl", "out")
    assert "--full-ft" in cmd and "--init-lora" not in cmd
    assert cmd[cmd.index("--seed-order") + 1] == "b.jsonl"

    cfg = make_cfg(full_ft=False)
    cmd = grpo_cmd(cfg, 2, "m", "outputs/star", "b.jsonl", "out")
    assert "--full-ft" not in cmd
    assert cmd[cmd.index("--init-lora") + 1] == "outputs/star"


def test_star_sample_cmd_uses_star_seeds():
    cfg = make_cfg()
    cmd = star_sample_cmd(cfg, 2, "m", "d.jsonl")
    assert cmd[1:3] == ["train/star_sft.py", "sample"]
    assert cmd[cmd.index("--seed-start") + 1] == "150"
    assert cmd[cmd.index("--out") + 1] == "d.jsonl"


import train.pipeline as pipeline


def _fake_runner(monkeypatch, probe_win1s):
    """Replace run_step; scripted win@1 per probe call, no-op otherwise."""
    calls = []
    win1s = iter(probe_win1s)

    def fake(cmd, log_path):
        calls.append(cmd)
        script = cmd[1]
        if script.endswith("probe_wink.py"):
            return json.dumps({"win_at_k": {"1": next(win1s), "4": 0, "8": 0}})
        if script.endswith("star_sft.py") and cmd[2] == "sample":
            return json.dumps({"played": 8, "kept": 2})
        return "trained\n"

    monkeypatch.setattr(pipeline, "run_step", fake)
    return calls


def test_happy_path_single_stage(monkeypatch, tmp_path):
    calls = _fake_runner(monkeypatch, [0.2, 0.9])  # probe-base, eval
    state = pipeline.run_pipeline(make_cfg(), str(tmp_path))
    assert list(state["completed"]) == ["s2-probe-base", "s2-grpo", "s2-eval"]
    assert state["stopped"] is None
    assert state["model_path"] == os.path.join(str(tmp_path), "grpo-s2")
    assert len(calls) == 3


def test_star_retry_then_gate_failed(monkeypatch, tmp_path):
    calls = _fake_runner(monkeypatch, [0.0, 0.0])  # probe-base, probe-star
    state = pipeline.run_pipeline(make_cfg(), str(tmp_path))
    assert list(state["completed"]) == [
        "s2-probe-base", "s2-star-sample", "s2-star-train", "s2-probe-star",
    ]
    assert state["stopped"] == "gate_failed"
    assert state["model_path"] == os.path.join(str(tmp_path), "star-s2")


def test_star_retry_recovers_and_advances(monkeypatch, tmp_path):
    _fake_runner(monkeypatch, [0.0, 0.3, 0.9])  # base, re-probe, eval
    state = pipeline.run_pipeline(make_cfg(), str(tmp_path))
    assert "s2-grpo" in state["completed"]
    assert state["stopped"] is None


def test_lora_mode_tracks_adapter_not_model(monkeypatch, tmp_path):
    calls = _fake_runner(monkeypatch, [0.0, 0.3, 0.9])
    state = pipeline.run_pipeline(make_cfg(full_ft=False), str(tmp_path))
    assert state["model_path"] is None
    assert state["adapter_path"] == os.path.join(str(tmp_path), "grpo-s2")
    grpo_call = next(c for c in calls if c[1].endswith("grpo_agentic.py"))
    assert grpo_call[grpo_call.index("--init-lora") + 1] == os.path.join(
        str(tmp_path), "star-s2")


def test_resume_skips_completed_steps(monkeypatch, tmp_path):
    _fake_runner(monkeypatch, [0.2, 0.9])
    pipeline.run_pipeline(make_cfg(), str(tmp_path))
    calls = _fake_runner(monkeypatch, [])  # any run_step call would StopIteration
    state = pipeline.run_pipeline(make_cfg(), str(tmp_path), resume=True)
    assert calls == []
    assert state["stopped"] is None


def test_multi_stage_chains_model_path(monkeypatch, tmp_path):
    calls = _fake_runner(monkeypatch, [0.2, 0.9, 0.2, 0.9])
    state = pipeline.run_pipeline(make_cfg(stages=[2, 3]), str(tmp_path))
    assert "s3-eval" in state["completed"]
    assert state["model_path"] == os.path.join(str(tmp_path), "grpo-s3")
    s3_probe = next(
        c for c in calls
        if c[1].endswith("probe_wink.py") and c[c.index("--num-categories") + 1] == "3"
    )
    assert s3_probe[s3_probe.index("--model") + 1] == os.path.join(
        str(tmp_path), "grpo-s2")


@pytest.mark.parametrize("name", ["smoke", "qwen3-4b-full", "qwen3-8b-full"])
def test_shipped_configs_are_valid(name):
    cfg = load_config(f"configs_train/{name}.yaml")
    assert cfg["run_name"] == name
    for section in ("star", "grpo"):
        assert isinstance(cfg[section]["lr"], float)
