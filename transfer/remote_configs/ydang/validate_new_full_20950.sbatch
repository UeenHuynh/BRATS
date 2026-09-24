#!/bin/bash
#SBATCH --job-name=brats-validate-20950
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/validate_new_full_20950_%j.out
#SBATCH --error=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/validate_new_full_20950_%j.err

set -euo pipefail
REMOTE_ROOT=/mnt/beegfs/ydang/BRATS
source "$REMOTE_ROOT/transfer/remote_configs/ydang/brats.env"
OUTPUT_ROOT="$REMOTE_ROOT/experiments/preprocessing_full_canonical/run_20950"
"$BRATS_VENV_DIR/bin/python" "$REMOTE_ROOT/transfer/scripts/validate_preprocessed_dataset.py" \
  --dataset-root "$OUTPUT_ROOT/datasets" \
  --manifest "$PROJECT_ROOT/manifests/local_dataset.csv" \
  --expected-train-count 500 \
  --expected-validation-count 70 \
  --output "$OUTPUT_ROOT/validation.json"
