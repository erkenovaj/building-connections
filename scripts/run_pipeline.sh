#!/usr/bin/env bash
# Operator entrypoint: bash scripts/run_pipeline.sh configs_train/qwen3-4b-full.yaml
# Creates a dated run dir and resumes automatically if re-launched after a
# crash. Wrap with nohup/tmux/sbatch as needed; COMET_API_KEY comes from env.
set -euo pipefail

CONFIG="${1:?usage: bash scripts/run_pipeline.sh <config.yaml>}"
export PYTHONUNBUFFERED=1

RUN_NAME=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "$CONFIG")
RUN_DIR="outputs/runs/${RUN_NAME}-$(date +%Y%m%d)"
mkdir -p "$RUN_DIR"

exec python train/pipeline.py --config "$CONFIG" --run-dir "$RUN_DIR" --resume
