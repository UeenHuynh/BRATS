#!/bin/bash
#SBATCH --job-name=brats-spacing1mm-train
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/spacing1mm_train_%j.out
#SBATCH --error=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/spacing1mm_train_%j.err

set -euo pipefail
ROOT=/mnt/beegfs/ydang/BRATS
source "$ROOT/transfer/remote_configs/ydang/brats.env"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"
"$BRATS_VENV_DIR/bin/python" -m brats.experiments.ablation \
  --config "$ROOT/transfer/remote_configs/ydang/ablation_spacing1mm_fold1.yaml" \
  run-task --arm P0_SPACING1MM_ZSCORE --fold 1 --seed 42
