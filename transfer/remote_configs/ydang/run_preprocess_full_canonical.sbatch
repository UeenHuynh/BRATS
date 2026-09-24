#!/bin/bash
#SBATCH --job-name=brats-prep-full
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00
#SBATCH --output=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/preprocess_full_%j.out
#SBATCH --error=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/preprocess_full_%j.err

set -euo pipefail
REMOTE_ROOT=/mnt/beegfs/ydang/BRATS
source "$REMOTE_ROOT/transfer/remote_configs/ydang/brats.env"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export TOTALSEG_HOME_DIR="$REMOTE_ROOT/cache/totalsegmentator"
export BRATS_FULL_PREPROCESS_OUTPUT_ROOT="$REMOTE_ROOT/experiments/preprocessing_full_canonical/run_${SLURM_JOB_ID}"

cd "$PROJECT_ROOT"
"$BRATS_VENV_DIR/bin/python" -m brats.preprocessing \
  --config "$REMOTE_ROOT/transfer/remote_configs/ydang/preprocess_full_canonical.yaml" \
  --manifest "$PROJECT_ROOT/manifests/local_dataset.csv" \
  --profiles P0_MEDIAN_ZSCORE P1_MEDIAN_CLIP_ZSCORE P2_MEDIAN_N4_ZSCORE P3_MEDIAN_N4_CLIP_ZSCORE \
  --fail-fast

"$BRATS_VENV_DIR/bin/python" "$REMOTE_ROOT/transfer/scripts/validate_preprocessed_dataset.py" \
  --dataset-root "$BRATS_FULL_PREPROCESS_OUTPUT_ROOT/datasets" \
  --manifest "$PROJECT_ROOT/manifests/local_dataset.csv" \
  --expected-train-count 500 \
  --expected-validation-count 70 \
  --output "$BRATS_FULL_PREPROCESS_OUTPUT_ROOT/validation.json"
