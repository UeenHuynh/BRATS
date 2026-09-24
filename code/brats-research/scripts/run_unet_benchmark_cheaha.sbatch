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

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${BRATS_VENV_DIR:-$ROOT/.venv-train}"
export BRATS_DATASET_ROOT="${BRATS_DATASET_ROOT:-$ROOT/outputs/preprocessing/datasets}"
OUTPUT_BASE="${BRATS_OUTPUT_BASE:-$ROOT/outputs}"
cd "$ROOT"
export PYTHONPATH="$ROOT"

UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ]; then
  echo "ERROR: uv is not available. Load the Cheaha Python/uv module first." >&2
  exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$UV_BIN" venv --python python3 "$VENV_DIR"
fi

if ! "$VENV_DIR/bin/python" -c 'import torch' >/dev/null 2>&1; then
  BRATS_UV_CACHE_DIR="${BRATS_UV_CACHE_DIR:-${TMPDIR:-/tmp}/brats-uv-cache}"
  mkdir -p "$BRATS_UV_CACHE_DIR"
  UV_CACHE_DIR="$BRATS_UV_CACHE_DIR" "$UV_BIN" pip install \
    --python "$VENV_DIR/bin/python" \
    numpy==1.26.4 pyyaml==6.0.2 simpleitk==2.4.1 torch
fi

"$VENV_DIR/bin/python" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA GPU is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'

mkdir -p "$OUTPUT_BASE/benchmark_2d" "$OUTPUT_BASE/benchmark_3d"

echo "===== 2D U-Net benchmark: P0-P3 ====="
BRATS_OUTPUT_ROOT="$OUTPUT_BASE/benchmark_2d" "$VENV_DIR/bin/python" -m brats.experiments.benchmark \
  --config configs/benchmark_2d.yaml

echo "===== 3D U-Net benchmark: P0-P3 ====="
BRATS_OUTPUT_ROOT="$OUTPUT_BASE/benchmark_3d" "$VENV_DIR/bin/python" -m brats.experiments.benchmark \
  --config configs/benchmark_3d.yaml

echo "===== COMPLETE ====="
find "$OUTPUT_BASE/benchmark_2d" "$OUTPUT_BASE/benchmark_3d" \
  -name benchmark.json -print
