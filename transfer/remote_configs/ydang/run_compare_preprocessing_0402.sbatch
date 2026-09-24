#!/bin/bash
#SBATCH --job-name=brats-old-new-0402
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/compare_preprocessing_0402_%j.out
#SBATCH --error=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/compare_preprocessing_0402_%j.err

set -euo pipefail

REMOTE_ROOT=/mnt/beegfs/ydang/BRATS
CONFIG_ROOT="$REMOTE_ROOT/transfer/remote_configs/ydang"
COMPARE_ROOT="$REMOTE_ROOT/experiments/preprocessing_compare_0402/run_${SLURM_JOB_ID}"
LEGACY_ROOT="$COMPARE_ROOT/legacy_workspace"
CASE_ID=BraTS-MEN-RT-0402-1

source "$CONFIG_ROOT/brats.env"
export PYTHONUNBUFFERED=1
export BRATS_COMPARE_OUTPUT_ROOT="$COMPARE_ROOT"

mkdir -p "$LEGACY_ROOT/configs" "$LEGACY_ROOT/manifests"
cp -a "$REMOTE_ROOT/code/legacy_nnunet/src" "$LEGACY_ROOT/src"
install -m 0644 "$CONFIG_ROOT/preprocessing_0402_legacy.json" "$LEGACY_ROOT/configs/preprocessing_config.json"
install -m 0644 "$CONFIG_ROOT/manifest_0402_legacy.csv" "$LEGACY_ROOT/manifests/manifest_raw.csv"

export PYTHONPATH="$LEGACY_ROOT/src"
"$BRATS_VENV_DIR/bin/python" -m brats_men_rt.stages.preprocessing --limit 1 --progress-every 1

export PYTHONPATH="$PROJECT_ROOT"
"$BRATS_VENV_DIR/bin/python" -m brats.preprocessing \
  --config "$CONFIG_ROOT/preprocess_compare_0402_new.yaml" \
  --manifest "$CONFIG_ROOT/manifest_0402.csv" \
  --profiles P1_MEDIAN_CLIP_ZSCORE \
  --fail-fast

"$BRATS_VENV_DIR/bin/python" "$REMOTE_ROOT/transfer/scripts/compare_preprocessing_outputs.py" \
  --raw-image "$REMOTE_ROOT/datasets/raw_canonical/train/$CASE_ID/${CASE_ID}_t1c.nii.gz" \
  --raw-label "$REMOTE_ROOT/datasets/raw_canonical/train/$CASE_ID/${CASE_ID}_gtv.nii.gz" \
  --legacy-image "$LEGACY_ROOT/data_processed/images/${CASE_ID}_t1c.nii.gz" \
  --legacy-label "$LEGACY_ROOT/data_processed/masks/${CASE_ID}_gtv.nii.gz" \
  --new-image "$COMPARE_ROOT/new/datasets/P1_MEDIAN_CLIP_ZSCORE/train/images/${CASE_ID}.nii.gz" \
  --new-label "$COMPARE_ROOT/new/datasets/P1_MEDIAN_CLIP_ZSCORE/train/labels/${CASE_ID}.nii.gz" \
  --output "$COMPARE_ROOT/comparison.json"

echo "COMPARE_ROOT=$COMPARE_ROOT"
