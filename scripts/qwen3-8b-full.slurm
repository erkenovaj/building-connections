#!/usr/bin/env bash
# Submit from the repository root:
#   sbatch scripts/qwen3-8b-full.slurm
# Requests one type_h (80GB) GPU on the rocky partition.

#SBATCH --job-name=qwen3-8b-full
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-12:00
#SBATCH --output=logs/slurm-%A.out
#SBATCH --error=logs/slurm-%A.err
#SBATCH --constraint=type_h
#SBATCH --export=ALL

set -euo pipefail
umask 027

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_ROOT"

CONFIG="configs_train/qwen3-8b-full.yaml"
if [[ ! -f "$CONFIG" || ! -f scripts/run_pipeline.sh ]]; then
  echo "Project files were not found in $PROJECT_ROOT." >&2
  echo "Submit from the repository root or export PROJECT_ROOT first." >&2
  exit 2
fi

# TRAIN_VENV may point to a cluster-specific virtual environment. Otherwise,
# use the repository-local training environment when it exists, or inherit the
# Python environment exported by sbatch.
if [[ -n "${TRAIN_VENV:-}" ]]; then
  if [[ ! -f "$TRAIN_VENV/bin/activate" ]]; then
    echo "TRAIN_VENV does not contain bin/activate: $TRAIN_VENV" >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source "$TRAIN_VENV/bin/activate"
elif [[ -f .venv-train/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv-train/bin/activate
fi

command -v python >/dev/null || {
  echo "python is unavailable; activate the training environment first." >&2
  exit 2
}
python -c 'import torch, transformers, trl, datasets, yaml, bitsandbytes'

MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-79000}"
if [[ ! "$MIN_GPU_MEMORY_MIB" =~ ^[0-9]+$ ]]; then
  echo "MIN_GPU_MEMORY_MIB must be an integer: $MIN_GPU_MEMORY_MIB" >&2
  exit 2
fi

GPU_MEMORY_MIB="$(
  python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))'
)"
if (( GPU_MEMORY_MIB < MIN_GPU_MEMORY_MIB )); then
  echo "Allocated GPU has ${GPU_MEMORY_MIB} MiB; at least ${MIN_GPU_MEMORY_MIB} MiB is required." >&2
  exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_NAME="$(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "Starting qwen3-8b-full on ${GPU_NAME} (${GPU_MEMORY_MIB} MiB)"
echo "Project root: $PROJECT_ROOT"

srun --ntasks=1 bash scripts/run_pipeline.sh "$CONFIG"
