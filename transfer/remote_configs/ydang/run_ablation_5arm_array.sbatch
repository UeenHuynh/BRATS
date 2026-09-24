#!/bin/bash
#SBATCH --job-name=brats-ablation
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-74%4
#SBATCH --output=/mnt/beegfs/ydang/BRATS/experiments/ablation_5arm_5fold_3seed/logs/task_%A_%a.out
#SBATCH --error=/mnt/beegfs/ydang/BRATS/experiments/ablation_5arm_5fold_3seed/logs/task_%A_%a.err

set -euo pipefail

REMOTE_ROOT=/mnt/beegfs/ydang/BRATS
source "$REMOTE_ROOT/transfer/remote_configs/ydang/brats.env"
CONFIG="$REMOTE_ROOT/transfer/remote_configs/ydang/ablation_5arm_5fold_3seed.yaml"
ARMS=(legacy_corrected P0_MEDIAN_ZSCORE P1_MEDIAN_CLIP_ZSCORE P2_MEDIAN_N4_ZSCORE P3_MEDIAN_N4_CLIP_ZSCORE)
SEEDS=(42 2026 3407)

mkdir -p "$REMOTE_ROOT/experiments/ablation_5arm_5fold_3seed/logs"

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
arm_index=$((task_id / 15))
within_arm=$((task_id % 15))
fold=$((within_arm / 3))
seed_index=$((within_arm % 3))
arm="${ARMS[$arm_index]}"
seed="${SEEDS[$seed_index]}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

echo "task=$task_id arm=$arm fold=$fold seed=$seed"
"$BRATS_VENV_DIR/bin/python" -m brats.experiments.ablation \
  --config "$CONFIG" run-task --arm "$arm" --fold "$fold" --seed "$seed"
