#!/bin/bash
#SBATCH --job-name=brats-tune-full
#SBATCH --output=brats-tune-full-%j.out
#SBATCH --error=brats-tune-full-%j.err
#SBATCH --partition=amperenodes-medium
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${BRATS_VENV_DIR:-$ROOT/.venv-train}"
export BRATS_DATASET_ROOT="${BRATS_DATASET_ROOT:-$ROOT/outputs/preprocessing/datasets}"
OUTPUT_BASE="${BRATS_OUTPUT_BASE:-$ROOT/outputs}"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PILOT_CASES="${PILOT_CASES:-100}"
PILOT_EPOCHS="${PILOT_EPOCHS:-10}"
FULL_EPOCHS="${FULL_EPOCHS:-30}"

UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ]; then
  echo "ERROR: uv is not available." >&2
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

make_config() {
  local source="$1" destination="$2" mode="$3" epochs="$4" cases="$5" output_root="$6" profile="${7:-}"
  "$VENV_DIR/bin/python" - "$source" "$destination" "$mode" "$epochs" "$cases" "$output_root" "$profile" <<'PY'
import os
import sys
from pathlib import Path
import yaml

source, destination, mode, epochs, cases, output_root, profile = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
config["mode"] = mode
config["dataset_root"] = os.environ["BRATS_DATASET_ROOT"]
config["output_root"] = output_root
config["training"]["epochs"] = int(epochs)
if cases == "all":
    config.pop("case_limit", None)
else:
    config["case_limit"] = int(cases)
if profile:
    config["profiles"] = [profile]
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

latest_benchmark_json() {
  find "$1" -type f -name benchmark.json -printf '%T@ %p\n' | sort -nr | \
    awk 'NR == 1 { $1=""; sub(/^ /, ""); print; exit }'
}

select_best_profile() {
  "$VENV_DIR/bin/python" - "$1" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
ranked = []
for profile, result in record["results"].items():
    summary = result["validation"]["summary"]
    dice = float(summary["dice"]["mean"])
    hd95 = float(summary["hd95_mm"]["mean"])
    ranked.append((dice, -hd95, profile))
print(max(ranked)[2])
PY
}

mkdir -p "$OUTPUT_BASE"/pilot_2d "$OUTPUT_BASE"/pilot_3d \
  "$OUTPUT_BASE"/full_2d "$OUTPUT_BASE"/full_3d

echo "===== PILOT 2D: ${PILOT_CASES} cases, ${PILOT_EPOCHS} epochs ====="
make_config configs/benchmark_2d.yaml /tmp/brats_pilot_2d.yaml 2d "$PILOT_EPOCHS" "$PILOT_CASES" "$OUTPUT_BASE/pilot_2d"
BRATS_OUTPUT_ROOT="$OUTPUT_BASE/pilot_2d" "$VENV_DIR/bin/python" -m brats.experiments.benchmark \
  --config /tmp/brats_pilot_2d.yaml 2>&1 | tee "$OUTPUT_BASE/pilot_2d/console.log"
PILOT_2D_JSON="$(latest_benchmark_json "$OUTPUT_BASE/pilot_2d")"
BEST_2D="$(select_best_profile "$PILOT_2D_JSON")"
echo "BEST 2D: $BEST_2D"

echo "===== PILOT 3D: ${PILOT_CASES} cases, ${PILOT_EPOCHS} epochs ====="
make_config configs/benchmark_3d.yaml /tmp/brats_pilot_3d.yaml 3d "$PILOT_EPOCHS" "$PILOT_CASES" "$OUTPUT_BASE/pilot_3d"
BRATS_OUTPUT_ROOT="$OUTPUT_BASE/pilot_3d" "$VENV_DIR/bin/python" -m brats.experiments.benchmark \
  --config /tmp/brats_pilot_3d.yaml 2>&1 | tee "$OUTPUT_BASE/pilot_3d/console.log"
PILOT_3D_JSON="$(latest_benchmark_json "$OUTPUT_BASE/pilot_3d")"
BEST_3D="$(select_best_profile "$PILOT_3D_JSON")"
echo "BEST 3D: $BEST_3D"

echo "===== FULL 2D: $BEST_2D, all cases, ${FULL_EPOCHS} epochs ====="
make_config configs/benchmark_2d.yaml /tmp/brats_full_2d.yaml 2d "$FULL_EPOCHS" all "$OUTPUT_BASE/full_2d" "$BEST_2D"
BRATS_OUTPUT_ROOT="$OUTPUT_BASE/full_2d" "$VENV_DIR/bin/python" -m brats.experiments.benchmark \
  --config /tmp/brats_full_2d.yaml 2>&1 | tee "$OUTPUT_BASE/full_2d/console.log"

echo "===== FULL 3D: $BEST_3D, all cases, ${FULL_EPOCHS} epochs ====="
make_config configs/benchmark_3d.yaml /tmp/brats_full_3d.yaml 3d "$FULL_EPOCHS" all "$OUTPUT_BASE/full_3d" "$BEST_3D"
BRATS_OUTPUT_ROOT="$OUTPUT_BASE/full_3d" "$VENV_DIR/bin/python" -m brats.experiments.benchmark \
  --config /tmp/brats_full_3d.yaml 2>&1 | tee "$OUTPUT_BASE/full_3d/console.log"

echo "===== COMPLETE ====="
find "$OUTPUT_BASE" \
  -type f -name benchmark.json -print
