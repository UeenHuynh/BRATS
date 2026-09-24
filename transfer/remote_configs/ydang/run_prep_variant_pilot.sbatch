#!/bin/bash
#SBATCH --job-name=brats-prep-pilot
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/prep_variant_%j.out
#SBATCH --error=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/prep_variant_%j.err

set -euo pipefail
ROOT=/mnt/beegfs/ydang/BRATS
source "$ROOT/transfer/remote_configs/ydang/brats.env"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
VARIANT="${VARIANT:?set VARIANT}"
PREP_CONFIG="${PREP_CONFIG:-}"
PROFILE="${PROFILE:?set PROFILE}"
OUT="$ROOT/experiments/unet/pilot_variants/$VARIANT"
mkdir -p "$OUT"

if [[ -n "$PREP_CONFIG" ]]; then
  export BRATS_FULL_PREPROCESS_OUTPUT_ROOT="$OUT"
  export TOTALSEG_HOME_DIR="$ROOT/cache/totalsegmentator"
  "$BRATS_VENV_DIR/bin/python" -m brats.preprocessing \
    --config "$ROOT/transfer/remote_configs/ydang/$PREP_CONFIG" \
    --manifest "$PROJECT_ROOT/manifests/local_dataset.csv" \
    --profiles "$PROFILE" --case-limit 100 --fail-fast
  DATASET_ROOT="$OUT/datasets"
else
  DATASET_ROOT="$ROOT/datasets/preprocessed"
fi

export BRATS_DATASET_ROOT="$DATASET_ROOT"
export BRATS_OUTPUT_ROOT="$OUT/benchmark"

cat > "$OUT/benchmark.yaml" <<EOF
schema_version: 1
seed: 42
mode: 3d
dataset_root: $DATASET_ROOT
output_root: $OUT/benchmark
source_split: train
case_limit: 100
profiles: [$PROFILE]
split: {validation_fraction: 0.2}
data: {depth: 64, patch_size: [128, 128]}
training: {epochs: 10, batch_size: 2, num_workers: 4, amp: true, grad_clip: 1.0, log_interval: 10}
model: {arch: unet3d, in_channels: 1, num_classes: 1, base_filters: 16}
loss: {name: dice_ce, dice_weight: 1.0, bce_weight: 1.0, bce_pos_weight: 100.0}
optimizer: {name: adam, lr: 0.0001, weight_decay: 0.00001}
scheduler: {name: cosine, warmup_epochs: 0}
evaluation: {batch_size: 2, threshold: 0.5}
EOF

if [[ "${AUGMENT:-0}" == "1" ]]; then
  export BRATS_INTENSITY_AUGMENT=1
fi
cd "$PROJECT_ROOT"
"$BRATS_VENV_DIR/bin/python" -m brats.experiments.benchmark --config "$OUT/benchmark.yaml"
