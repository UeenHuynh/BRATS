#!/bin/bash
#SBATCH --job-name=brats-old-new-smoke5
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/compare_preprocessing_smoke5_%j.out
#SBATCH --error=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/compare_preprocessing_smoke5_%j.err

set -euo pipefail

REMOTE_ROOT=/mnt/beegfs/ydang/BRATS
CONFIG_ROOT="$REMOTE_ROOT/transfer/remote_configs/ydang"
COMPARE_ROOT="$REMOTE_ROOT/experiments/preprocessing_compare_smoke5/run_${SLURM_JOB_ID}"
LEGACY_CODE="$REMOTE_ROOT/code/legacy_nnunet"

source "$CONFIG_ROOT/brats.env"
export PYTHONUNBUFFERED=1
export BRATS_COMPARE_OUTPUT_ROOT="$COMPARE_ROOT"

export PYTHONPATH="$LEGACY_CODE/src"
"$BRATS_VENV_DIR/bin/python" -m brats_men_rt.stages.preprocessing \
  --project-root "$LEGACY_CODE" \
  --manifest "$CONFIG_ROOT/manifest_smoke5_legacy.csv" \
  --config "$LEGACY_CODE/configs/preprocessing_config.json" \
  --data-root "$REMOTE_ROOT" \
  --output-root "$COMPARE_ROOT/legacy/data_processed" \
  --artifact-root "$COMPARE_ROOT/legacy/artifacts" \
  --progress-every 1

export PYTHONPATH="$PROJECT_ROOT"
"$BRATS_VENV_DIR/bin/python" -m brats.preprocessing \
  --config "$CONFIG_ROOT/preprocess_compare_0402_new.yaml" \
  --manifest "$CONFIG_ROOT/manifest_smoke5.csv" \
  --profiles P1_MEDIAN_CLIP_ZSCORE \
  --fail-fast

"$BRATS_VENV_DIR/bin/python" "$REMOTE_ROOT/transfer/scripts/validate_preprocessing_smoke_set.py" \
  --manifest "$CONFIG_ROOT/manifest_smoke5.csv" \
  --data-root "$REMOTE_ROOT" \
  --legacy-root "$COMPARE_ROOT/legacy/data_processed" \
  --new-root "$COMPARE_ROOT/new" \
  --output "$COMPARE_ROOT/validation.json"

echo "COMPARE_ROOT=$COMPARE_ROOT"
