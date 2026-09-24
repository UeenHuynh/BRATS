#!/bin/bash
#SBATCH --job-name=brats-spacing1mm-prep
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/spacing1mm_full_%j.out
#SBATCH --error=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/spacing1mm_full_%j.err

set -euo pipefail
ROOT=/mnt/beegfs/ydang/BRATS
source "$ROOT/transfer/remote_configs/ydang/brats.env"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export TOTALSEG_HOME_DIR="$ROOT/cache/totalsegmentator"
export BRATS_FULL_PREPROCESS_OUTPUT_ROOT="$ROOT/experiments/preprocessing_spacing1mm_fold1/current"
cd "$PROJECT_ROOT"
"$BRATS_VENV_DIR/bin/python" -m brats.preprocessing \
  --config "$ROOT/transfer/remote_configs/ydang/preprocess_pilot_spacing1mm.yaml" \
  --manifest "$PROJECT_ROOT/manifests/local_dataset.csv" \
  --profiles P0_SPACING1MM_ZSCORE --case-limit 570 --fail-fast
"$BRATS_VENV_DIR/bin/python" "$ROOT/transfer/scripts/validate_preprocessed_dataset.py" \
  --dataset-root "$BRATS_FULL_PREPROCESS_OUTPUT_ROOT/datasets" \
  --manifest "$PROJECT_ROOT/manifests/local_dataset.csv" \
  --profiles P0_SPACING1MM_ZSCORE \
  --expected-train-count 500 --expected-validation-count 70 \
  --output "$BRATS_FULL_PREPROCESS_OUTPUT_ROOT/validation.json"
