#!/bin/bash
# Read-only/input-validation preflight plus dry-run records for both pipelines.

set -euo pipefail
REMOTE_ROOT=/mnt/beegfs/ydang/BRATS
LEGACY_CODE="$REMOTE_ROOT/code/legacy_nnunet"
NEW_CODE="$REMOTE_ROOT/code/brats-research"
VENV="$REMOTE_ROOT/envs/brats-py311"
PREFLIGHT_ROOT="$REMOTE_ROOT/experiments/preflight"

"$VENV/bin/python" "$REMOTE_ROOT/transfer/scripts/validate_canonical_dataset.py" \
  --workspace "$REMOTE_ROOT" \
  --manifest "$NEW_CODE/manifests/local_dataset.csv" \
  --output "$PREFLIGHT_ROOT/canonical_validation.json"

PYTHONPATH="$LEGACY_CODE/src" "$VENV/bin/python" -m unittest discover \
  -s "$LEGACY_CODE/tests" -v

PYTHONPATH="$LEGACY_CODE/src" "$VENV/bin/python" -m brats_men_rt.stages.preprocessing \
  --project-root "$LEGACY_CODE" \
  --manifest "$LEGACY_CODE/manifests/manifest_raw_canonical.csv" \
  --config "$LEGACY_CODE/configs/preprocessing_config.json" \
  --data-root "$REMOTE_ROOT" \
  --output-root "$PREFLIGHT_ROOT/legacy_dry_run/data_processed" \
  --artifact-root "$PREFLIGHT_ROOT/legacy_dry_run" \
  --dry-run

export BRATS_FULL_PREPROCESS_OUTPUT_ROOT="$PREFLIGHT_ROOT/new_dry_run"
PYTHONPATH="$NEW_CODE" "$VENV/bin/python" -m brats.preprocessing \
  --config "$REMOTE_ROOT/transfer/remote_configs/ydang/preprocess_full_canonical.yaml" \
  --manifest "$NEW_CODE/manifests/local_dataset.csv" \
  --profiles P0_MEDIAN_ZSCORE P1_MEDIAN_CLIP_ZSCORE P2_MEDIAN_N4_ZSCORE P3_MEDIAN_N4_CLIP_ZSCORE \
  --dry-run

for executable in nnUNetv2_plan_and_preprocess nnUNetv2_train nnUNetv2_predict; do
  test -x "$VENV/bin/$executable"
done

cd "$REMOTE_ROOT"
sbatch --test-only transfer/remote_configs/ydang/run_compare_preprocessing_smoke5.sbatch
sbatch --test-only transfer/remote_configs/ydang/run_legacy_corrected_full.sbatch
sbatch --test-only transfer/remote_configs/ydang/run_preprocess_full_canonical.sbatch
echo "FULL_PREFLIGHT_OK"
