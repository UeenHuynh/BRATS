#!/bin/bash
#SBATCH --job-name=brats-p0p3-100c10e
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/pilot_100c10e_%j.out
#SBATCH --error=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/pilot_100c10e_%j.err

set -euo pipefail

REMOTE_ROOT=/mnt/beegfs/ydang/BRATS
source "$REMOTE_ROOT/transfer/remote_configs/ydang/brats.env"

mkdir -p "$BRATS_OUTPUT_BASE/logs" "$BRATS_OUTPUT_BASE/pilot_preprocessing_3d_100c_10e"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export BRATS_OUTPUT_ROOT="$BRATS_OUTPUT_BASE/pilot_preprocessing_3d_100c_10e"

"$BRATS_VENV_DIR/bin/python" - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA GPU is unavailable"
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("gpu_memory_gib", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
PY

"$BRATS_VENV_DIR/bin/python" -m brats.experiments.benchmark \
  --config "$REMOTE_ROOT/transfer/remote_configs/ydang/pilot_preprocessing_3d_100c_10e.yaml"
