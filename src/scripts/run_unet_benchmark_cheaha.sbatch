#!/bin/bash
#SBATCH --job-name=brats-unet
#SBATCH --output=brats-unet-%j.out
#SBATCH --error=brats-unet-%j.err
#SBATCH --partition=amperenodes-medium
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

ROOT="/home/duckhai.le@mail.utoronto.ca/Desktop/BRATS"
cd "$ROOT"
export PYTHONPATH="$ROOT"

UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ] && [ -x "/home/duckhai.le@mail.utoronto.ca/.local/bin/uv" ]; then
  UV_BIN="/home/duckhai.le@mail.utoronto.ca/.local/bin/uv"
fi
if [ -z "$UV_BIN" ]; then
  echo "ERROR: uv is not available. Load the Cheaha Python/uv module first." >&2
  exit 1
fi

if [ ! -x .venv-train/bin/python ]; then
  "$UV_BIN" venv --python python3 .venv-train
fi

if ! .venv-train/bin/python -c 'import torch' >/dev/null 2>&1; then
  mkdir -p /scratch/local/brats-uv-cache
  UV_CACHE_DIR="/scratch/local/brats-uv-cache" "$UV_BIN" pip install \
    --python .venv-train/bin/python \
    numpy==1.26.4 pyyaml==6.0.2 simpleitk==2.4.1 torch
fi

.venv-train/bin/python -c \
  'import torch; assert torch.cuda.is_available(), "CUDA GPU is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'

mkdir -p outputs/benchmark_2d outputs/benchmark_3d

echo "===== 2D U-Net benchmark: P0-P3 ====="
.venv-train/bin/python -m brats.experiments.benchmark \
  --config configs/benchmark_2d.yaml

echo "===== 3D U-Net benchmark: P0-P3 ====="
.venv-train/bin/python -m brats.experiments.benchmark \
  --config configs/benchmark_3d.yaml

echo "===== COMPLETE ====="
find outputs/benchmark_2d outputs/benchmark_3d \
  -name benchmark.json -print
