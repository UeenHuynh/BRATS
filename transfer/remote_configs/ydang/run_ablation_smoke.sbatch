#!/bin/bash
#SBATCH --job-name=brats-abl-smoke
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/beegfs/ydang/BRATS/experiments/ablation_5arm_5fold_3seed/logs/smoke_%j.out
#SBATCH --error=/mnt/beegfs/ydang/BRATS/experiments/ablation_5arm_5fold_3seed/logs/smoke_%j.err

set -euo pipefail

REMOTE_ROOT=/mnt/beegfs/ydang/BRATS
source "$REMOTE_ROOT/transfer/remote_configs/ydang/brats.env"
CONFIG="$REMOTE_ROOT/transfer/remote_configs/ydang/ablation_5arm_5fold_3seed.yaml"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

"$BRATS_VENV_DIR/bin/python" -m brats.experiments.ablation \
  --config "$CONFIG" run-task \
  --arm P1_MEDIAN_CLIP_ZSCORE --fold 0 --seed 42 --smoke
