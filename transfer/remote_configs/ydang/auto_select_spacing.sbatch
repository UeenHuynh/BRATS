#!/bin/bash
#SBATCH --job-name=brats-auto-select
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/auto_select_spacing_%j.out
#SBATCH --error=/mnt/beegfs/ydang/BRATS/experiments/unet/logs/auto_select_spacing_%j.err
set -euo pipefail
ROOT=/mnt/beegfs/ydang/BRATS
P0="$ROOT/experiments/ablation_5arm_5fold_3seed/P0_MEDIAN_ZSCORE/fold_1/seed_42/DONE.json"
SPACING="$ROOT/experiments/ablation_spacing1mm_fold1/P0_SPACING1MM_ZSCORE/fold_1/seed_42/DONE.json"
DECISION="$ROOT/experiments/spacing_decision_confirmation/decision.json"
mkdir -p "$(dirname "$DECISION")"
if [[ ! -s "$P0" || ! -s "$SPACING" ]]; then
  echo "P0 or spacing DONE.json is missing" >&2
  exit 1
fi
read -r P0_SCORE SPACING_SCORE < <(python3 -c 'import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); print(a["best_inner_validation_dice"], b["best_inner_validation_dice"])' "$P0" "$SPACING")
python3 -c 'import json,sys; p=float(sys.argv[2]); s=float(sys.argv[3]); d=s-p; arm="P0_SPACING1MM_ZSCORE" if d >= 0.01 else "P0_MEDIAN_ZSCORE"; json.dump({"metric":"best_inner_validation_dice","p0":p,"spacing1mm":s,"delta":d,"threshold":0.01,"selected_arm":arm},open(sys.argv[1],"w"),indent=2); print(f"P0 inner={p:.6f} spacing1mm inner={s:.6f} delta={d:+.6f} selected={arm}")' "$DECISION" "$P0_SCORE" "$SPACING_SCORE"
SELECTED_ARM="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_arm"])' "$DECISION")"
echo "Submitting fold 2 confirmation with $SELECTED_ARM"
sbatch --parsable --nodelist=mantis-05 --export=ALL,SELECTED_ARM="$SELECTED_ARM" "$ROOT/transfer/remote_configs/ydang/train_selected_fold2.sbatch"
