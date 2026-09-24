#!/bin/bash
#SBATCH --job-name=brats-prep-0402
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/preprocess_0402_%j.out
#SBATCH --error=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/preprocess_0402_%j.err

set -euo pipefail

REMOTE_ROOT=/mnt/beegfs/ydang/BRATS
source "$REMOTE_ROOT/transfer/remote_configs/ydang/brats.env"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export TOTALSEG_HOME_DIR="$REMOTE_ROOT/cache/totalsegmentator"

STAGING_ROOT="$REMOTE_ROOT/experiments/preprocessing_0402"
CONFIG="$REMOTE_ROOT/transfer/remote_configs/ydang/preprocess_0402.yaml"
MANIFEST="$REMOTE_ROOT/transfer/remote_configs/ydang/manifest_0402.csv"
CASE_ID=BraTS-MEN-RT-0402-1
PROFILES=(
  P0_MEDIAN_ZSCORE
  P1_MEDIAN_CLIP_ZSCORE
  P2_MEDIAN_N4_ZSCORE
  P3_MEDIAN_N4_CLIP_ZSCORE
)

cd "$PROJECT_ROOT"
"$BRATS_VENV_DIR/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA GPU is unavailable"
print("torch", torch.__version__)
print("gpu", torch.cuda.get_device_name(0))
PY

"$BRATS_VENV_DIR/bin/python" -m brats.preprocessing \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --profiles "${PROFILES[@]}" \
  --fail-fast

for profile in "${PROFILES[@]}"; do
  source_image="$STAGING_ROOT/datasets/$profile/train/images/$CASE_ID.nii.gz"
  source_label="$STAGING_ROOT/datasets/$profile/train/labels/$CASE_ID.nii.gz"
  target_image="$BRATS_DATASET_ROOT/$profile/train/images/$CASE_ID.nii.gz"
  target_label="$BRATS_DATASET_ROOT/$profile/train/labels/$CASE_ID.nii.gz"
  test -f "$source_image"
  test -f "$source_label"
  test ! -e "$target_image"
  test ! -e "$target_label"
done

for profile in "${PROFILES[@]}"; do
  install -m 0644 "$STAGING_ROOT/datasets/$profile/train/images/$CASE_ID.nii.gz" \
    "$BRATS_DATASET_ROOT/$profile/train/images/$CASE_ID.nii.gz"
  install -m 0644 "$STAGING_ROOT/datasets/$profile/train/labels/$CASE_ID.nii.gz" \
    "$BRATS_DATASET_ROOT/$profile/train/labels/$CASE_ID.nii.gz"
done

"$BRATS_VENV_DIR/bin/python" "$REMOTE_ROOT/transfer/scripts/validate_preprocessed_dataset.py" \
  --dataset-root "$BRATS_DATASET_ROOT" \
  --manifest "$PROJECT_ROOT/manifests/local_dataset.csv" \
  --output "$STAGING_ROOT/validation_after_install.json"
