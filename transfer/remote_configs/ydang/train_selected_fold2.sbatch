#!/bin/bash
#SBATCH --job-name=brats-selected-f2
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/selected_fold2_%j.out
#SBATCH --error=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/selected_fold2_%j.err
set -euo pipefail
ROOT=/mnt/beegfs/ydang/BRATS
source "$ROOT/transfer/remote_configs/ydang/brats.env"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
ARM="${SELECTED_ARM:?SELECTED_ARM is required}"
cd "$PROJECT_ROOT"
"$BRATS_VENV_DIR/bin/python" -m brats.experiments.ablation \
  --config "$ROOT/transfer/remote_configs/ydang/ablation_spacing_decision_fold2.yaml" \
  run-task --arm "$ARM" --fold 2 --seed 42
