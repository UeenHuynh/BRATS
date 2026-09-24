#!/bin/bash
#SBATCH --job-name=brats-legacy-fixed
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/legacy_corrected_full_%j.out
#SBATCH --error=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/legacy_corrected_full_%j.err

set -euo pipefail
REMOTE_ROOT=/mnt/beegfs/ydang/BRATS
LEGACY_CODE="$REMOTE_ROOT/code/legacy_nnunet"
RUN_ROOT="$REMOTE_ROOT/experiments/legacy_corrected/run_${SLURM_JOB_ID}"
VENV="$REMOTE_ROOT/envs/brats-py311"
export PYTHONPATH="$LEGACY_CODE/src"
export PYTHONUNBUFFERED=1

"$VENV/bin/python" -m brats_men_rt.stages.preprocessing \
  --project-root "$LEGACY_CODE" \
  --manifest "$LEGACY_CODE/manifests/manifest_raw_canonical.csv" \
  --config "$LEGACY_CODE/configs/preprocessing_config.json" \
  --data-root "$REMOTE_ROOT" \
  --output-root "$RUN_ROOT/data_processed" \
  --artifact-root "$RUN_ROOT" \
  --progress-every 10

"$VENV/bin/python" "$REMOTE_ROOT/transfer/scripts/validate_legacy_preprocessed.py" \
  --manifest "$LEGACY_CODE/manifests/manifest_raw_canonical.csv" \
  --output-root "$RUN_ROOT/data_processed" \
  --report "$RUN_ROOT/validation.json"

"$VENV/bin/python" -m brats_men_rt.stages.splitting \
  --project-root "$LEGACY_CODE" \
  --manifest "$LEGACY_CODE/manifests/manifest_raw_canonical.csv" \
  --mask-qc "$LEGACY_CODE/qc/mask_qc_summary_canonical.csv" \
  --output-root "$RUN_ROOT" \
  --expected-cases 500

"$VENV/bin/python" -m brats_men_rt.stages.baseline_experiment \
  --project-root "$LEGACY_CODE" \
  --run-root "$RUN_ROOT" \
  --baseline-config "$LEGACY_CODE/configs/nnunet_baseline.yaml"
