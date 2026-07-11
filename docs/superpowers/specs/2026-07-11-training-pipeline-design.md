# Training Pipeline for Unattended Cluster Runs — Design

Date: 2026-07-11
Status: Approved (design decisions locked with user via chat)

## Goal

Convert the Kaggle curriculum notebook (probe → STaR SFT → agentic GRPO, staged
by board size 2→3→4 categories, with win@1 gates) into a config-driven script
setup for long (12–36h) unattended runs on a cluster: one YAML config + one
bash command, in-progress metric logging (comet-ml + local JSONL), stage-level
resume, and full fine-tuning (no PEFT) of 4–8B models. The operator (Sergey)
only edits a config and runs a bash script.

Hypotheses/experiments this serves: `docs/2026-07-11-hypotheses-experiments.md`.
Task tracker: `docs/2026-07-11-pre-call-tasks.md`.

## Decisions (locked with user)

- **Hardware baseline:** single GPU, assume 1×80GB. 8B full FT fits via bf16
  weights+grads (~32GB) + 8-bit Adam (`adamw_bnb_8bit`, ~16GB) + gradient
  checkpointing. No FSDP/multi-GPU — hardware unknown, and GRPO custom
  `rollout_func` under FSDP is separate work (backlog).
- **Experiment grid:** model scale only — Qwen3-4B and Qwen3-8B full FT
  (plus smoke config). No other axes in the initial config set.
- **Gate failure:** auto-STaR retry once (sample + train + re-probe), then
  clean stop with state saved. Never start a GRPO stage below win@1 ≥ 0.05.
- **Rollouts:** HF `generate` (sequential) as today; vLLM deferred to backlog.
- **Full-FT chaining:** each training stage saves a full model directory;
  the next stage receives it as `--model`. No adapters anywhere in full-FT mode.

## Architecture

Three units: minimal flag additions to the two existing train scripts, a new
subprocess orchestrator, and configs + launch script.

### 1. Flag additions (surgical)

`train/star_sft.py train`:

- `--full-ft` — pass `peft_config=None` to `SFTTrainer` (LoRA config becomes
  the else-branch). `trainer.save_model` then writes a full model dir.
- `--optim` (default `adamw_torch`) — forwarded to `SFTConfig.optim`;
  configs set `adamw_bnb_8bit` for 8B.
- `--report-to` (default `none`) — forwarded to `SFTConfig.report_to`
  (`comet_ml` on the cluster).
- `--run-name` (default None) — forwarded to `SFTConfig.run_name` so Comet
  experiments are identifiable per stage.

`train/grpo_agentic.py`: same four flags with the same semantics on
`GRPOConfig`. `--full-ft` conflicts with `--init-lora` (argparse error).
In full-FT mode the model is loaded plain (no `PeftModel`, `peft_config=None`)
and loaded in bf16 on cc≥8 GPUs (current code loads fp16 unconditionally on
CUDA — full FT of Qwen3 in fp16 is unstable; keep fp16 only as the non-bf16
fallback).

`train/probe_wink.py`: no changes (`--model` already accepts a checkpoint dir).

### 2. Orchestrator — `train/pipeline.py` (new)

`python train/pipeline.py --config configs_train/qwen3-4b-full.yaml
[--run-dir outputs/runs/<name>] [--resume]`.

**Config (YAML), flat and explicit** — example shape:

```yaml
run_name: qwen3-4b-full
model: Qwen/Qwen3-4B
full_ft: true
optim: adamw_bnb_8bit
report_to: comet_ml
max_new_tokens: 1024
stages: [2, 3, 4] # num-categories curriculum
gates: { start_grpo: 0.05, advance: 0.6 }
probe: { num_boards: 16, num_episodes: 8, seed_start: 1000000 }
eval: { num_boards: 16, num_episodes: 8, seed_start: 2000000 }
star:
  {
    num_boards: 200,
    episodes_per_board: 4,
    seed_start: 150,
    batch_size: 4,
    grad_accum: 4,
    lr: 1e-5,
    epochs: 2,
    dft: false,
  }
grpo:
  {
    num_boards: 64,
    max_steps: 200,
    num_generations: 8,
    batch_size: 8,
    grad_accum: 2,
    lr: 1e-5,
  }
```

**Per-curriculum-stage flow** (mirrors the notebook):

1. `probe_wink.py --out-boards` on probe seeds → parse summary JSON from stdout.
2. If `win@1 < start_grpo`: STaR (`sample` on star seeds, then `train`) →
   re-probe. If still below the gate: **stop** (retry budget = 1, per decision).
3. `grpo_agentic.py --seed-order <probe boards jsonl>` on the easiest boards.
4. Held-out eval: `probe_wink.py` on eval seeds (2,000,000+).
5. If `win@1 ≥ advance`: next stage with the just-saved model as `--model`;
   else stop cleanly.

**Mechanics:**

- Each step is a `subprocess.run([sys.executable, "train/<script>.py", ...])`
  — clean VRAM release between stages, and a crash in one stage can't corrupt
  the orchestrator.
- Step stdout/stderr tee'd to `run_dir/logs/<NN>-<step>.log`; the probe summary
  JSON is parsed from the last JSON line of stdout.
- `run_dir/state.json` records completed steps, their key outputs (model path,
  probe metrics, gate verdicts). `--resume` skips completed steps and picks up
  the model path from state. Written atomically after every step.
- `run_dir/metrics.jsonl` — one line per step: timestamp, step name, parsed
  metrics (probe summaries, STaR kept/played, gate decisions). This is the
  local mirror; TRL's own per-training-step metrics go to Comet via
  `--report-to` (and to the stage log file regardless).
- Comet credentials via standard `COMET_API_KEY` env var; `report_to: none`
  keeps everything local for an offline cluster.

### 3. Configs + launch script

- `configs_train/smoke.yaml` — Qwen3-0.6B (or 1.7B), LoRA (`full_ft: false`),
  tiny numbers (2 boards, 2 episodes, GRPO max_steps 4, stages [2]); validates
  the whole pipeline in <1h anywhere.
- `configs_train/qwen3-4b-full.yaml` — full FT, `adamw_torch`, 12–24h budget.
- `configs_train/qwen3-8b-full.yaml` — full FT, `adamw_bnb_8bit`,
  smaller `batch_size` + larger `grad_accum`, 24–36h budget.
- `scripts/run_pipeline.sh` — thin wrapper: `bash scripts/run_pipeline.sh
configs_train/qwen3-4b-full.yaml`. Sets `PYTHONUNBUFFERED=1`, creates the
  run dir (`outputs/runs/<run_name>-<date>`), invokes `train/pipeline.py`
  with `--resume`, exits with its code. Suitable for `nohup`/`tmux`/sbatch
  wrapping by the operator.

## Error handling

- Subprocess non-zero exit → orchestrator logs the step, leaves state.json at
  the last completed step, exits non-zero. `--resume` re-runs the failed step.
- Gate failure after the single STaR retry → clean stop, `state.json` marks
  `stopped: gate_failed`, exit 0 (expected outcome, not an error).
- Missing/unparseable probe summary JSON → treated as step failure (non-zero).
- Config validation up front: unknown keys rejected, required keys checked,
  `full_ft` + adapter-style options are mutually exclusive.

## Testing

- Unit: config parsing/validation; state.json round-trip + resume-skip logic;
  gate decision function (probe JSON → start/star-retry/advance/stop) —
  pure functions, no GPU.
- Smoke: `configs_train/smoke.yaml` end-to-end on MPS/CPU-capable tiny model —
  all stages execute, state.json and metrics.jsonl written, resume after a
  killed step works.
- Flag additions: `--full-ft` run of `star_sft.py train` on a toy dataset saves
  a full model dir (config.json + weights, no adapter_config.json).

## Out of scope (backlog)

- FSDP/multi-GPU, vLLM rollouts, Saturn-1.5B comparison, learned difficulty
  estimator, hyperparameter sweeps (all tracked in the pre-call task doc).
