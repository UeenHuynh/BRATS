from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

import yaml

from brats_men_rt.reports import write_markdown_report
from brats_men_rt.utils import read_json, write_csv, write_json, write_text


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ensure_prerequisites(root: Path, n_folds: int) -> None:
    required = [
        root / "manifests" / "manifest_preprocessed.csv",
        root / "splits" / "split_config.json",
    ]
    for path in required:
        if not path.exists():
            raise SystemExit(f"Missing prerequisite: {path}")
    for fold in range(n_folds):
        for suffix in ("train", "val"):
            path = root / "splits" / f"fold_{fold}_{suffix}.txt"
            if not path.exists():
                raise SystemExit(f"Missing split file: {path}")


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected mapping in {path}")
    return payload


def _dataset_folder_name(dataset_id: int, dataset_name: str) -> str:
    return f"Dataset{dataset_id:03d}_{dataset_name}"


def _symlink_or_skip(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        raise FileExistsError(f"Refusing conflicting existing target: {target}")
    os.symlink(source, target)


def _write_dataset_json(path: Path, num_training: int, dataset_name: str) -> None:
    write_json(
        path,
        {
            "channel_names": {"0": "T1c"},
            "labels": {"background": 0, "gtv": 1},
            "file_ending": ".nii.gz",
            "numTraining": num_training,
            "dataset_name": dataset_name,
        },
    )


def run(
    n_folds: int | None = None,
    *,
    project_root_path: Path | None = None,
    run_root: Path | None = None,
    baseline_config_path: Path | None = None,
) -> dict[str, object]:
    project = (project_root_path or Path(__file__).resolve().parents[3]).resolve()
    root = (run_root or project).resolve()
    split_config = read_json(root / "splits" / "split_config.json")
    baseline_config = _read_yaml((baseline_config_path or project / "configs" / "nnunet_baseline.yaml").resolve())
    folds = int(n_folds or split_config.get("n_folds", 5))
    _ensure_prerequisites(root, folds)
    dataset_id = int(baseline_config.get("dataset_id", 701))
    dataset_name = str(baseline_config.get("dataset_name", "BRATSMENRT"))
    dataset_folder = _dataset_folder_name(dataset_id, dataset_name)

    manifest_rows = _read_csv(root / "manifests" / "manifest_preprocessed.csv")
    duplicate_ids = sorted(
        case_id for case_id, count in Counter(row["case_id"] for row in manifest_rows).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"preprocessed manifest contains duplicate case IDs: {', '.join(duplicate_ids)}")
    manifest_index = {row["case_id"]: row for row in manifest_rows if row.get("status") == "processed"}
    train_ids_all = sorted(
        {
            case_id
            for fold in range(folds)
            for case_id in _read_split(root / "splits" / f"fold_{fold}_train.txt")
            + _read_split(root / "splits" / f"fold_{fold}_val.txt")
        }
    )
    missing_manifest_cases = sorted(set(train_ids_all) - set(manifest_index))
    if missing_manifest_cases:
        raise ValueError(f"split cases missing successful preprocessing: {missing_manifest_cases}")
    raw_dataset_dir = root / "nnunet_raw" / dataset_folder
    images_tr = raw_dataset_dir / "imagesTr"
    labels_tr = raw_dataset_dir / "labelsTr"
    images_ts = raw_dataset_dir / "imagesTs"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)

    for case_id in train_ids_all:
        row = manifest_index.get(case_id)
        if row is None:
            continue
        _symlink_or_skip(Path(row["processed_image_path"]), images_tr / f"{case_id}_0000.nii.gz")
        _symlink_or_skip(Path(row["processed_mask_path"]), labels_tr / f"{case_id}.nii.gz")
    _write_dataset_json(raw_dataset_dir / "dataset.json", len(train_ids_all), dataset_name)

    created_files = []
    split_json_rows = []
    for fold in range(folds):
        train_ids = _read_split(root / "splits" / f"fold_{fold}_train.txt")
        val_ids = _read_split(root / "splits" / f"fold_{fold}_val.txt")
        fold_dir = root / "models" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_rows = [manifest_index[case_id] for case_id in train_ids if case_id in manifest_index]
        val_rows = [manifest_index[case_id] for case_id in val_ids if case_id in manifest_index]
        train_manifest = fold_dir / "train_manifest.csv"
        val_manifest = fold_dir / "val_manifest.csv"
        if train_rows:
            write_csv(train_manifest, train_rows, train_rows[0].keys())
        else:
            write_csv(train_manifest, [], ("case_id",))
        if val_rows:
            write_csv(val_manifest, val_rows, val_rows[0].keys())
        else:
            write_csv(val_manifest, [], ("case_id",))
        created_files.extend([str(train_manifest), str(val_manifest)])
        split_json_rows.append({"train": train_ids, "val": val_ids})

        inference_input_dir = root / "nnunet_inference" / f"fold_{fold}" / "images"
        inference_input_dir.mkdir(parents=True, exist_ok=True)
        for case_id in val_ids:
            row = manifest_index.get(case_id)
            if row is None:
                continue
            _symlink_or_skip(Path(row["processed_image_path"]), inference_input_dir / f"{case_id}_0000.nii.gz")

    export_dir = root / "nnunet_exports" / dataset_folder
    export_dir.mkdir(parents=True, exist_ok=True)
    write_json(export_dir / "splits_final.json", split_json_rows)

    command_script = f"""#!/usr/bin/env bash
set -euo pipefail

# Activate your prepared environment first.
# conda activate brats-men-rt-pipeline
# Install PyTorch matching your CUDA version before running nnU-Net.

export nnUNet_raw="{root / 'nnunet_raw'}"
export nnUNet_preprocessed="{root / 'nnunet_preprocessed'}"
export nnUNet_results="{root / 'nnunet_results'}"

# Dataset folder: {dataset_folder}
# Raw dataset path: $nnUNet_raw/{dataset_folder}

nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity
mkdir -p "$nnUNet_preprocessed/{dataset_folder}"
cp "{export_dir / 'splits_final.json'}" "$nnUNet_preprocessed/{dataset_folder}/splits_final.json"

nnUNetv2_train {dataset_id} 3d_fullres 0
nnUNetv2_train {dataset_id} 3d_fullres 1
nnUNetv2_train {dataset_id} 3d_fullres 2
nnUNetv2_train {dataset_id} 3d_fullres 3
nnUNetv2_train {dataset_id} 3d_fullres 4

nnUNetv2_predict -i "{root / 'nnunet_inference' / 'fold_0' / 'images'}" -o "{root / 'predictions' / 'fold_0'}" -d {dataset_id} -c 3d_fullres -f 0
"""
    write_text(root / "logs" / "nnunet_command_template.sh", command_script)
    write_json(
        root / "models" / "baseline_experiment.json",
        {
            "model": "nnunet_v2_3d_fullres",
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "dataset_folder": dataset_folder,
            "n_folds": folds,
            "preprocessed_manifest": "manifests/manifest_preprocessed.csv",
            "split_config": "splits/split_config.json",
            "nnunet_raw": str(root / "nnunet_raw"),
            "nnunet_preprocessed": str(root / "nnunet_preprocessed"),
            "nnunet_results": str(root / "nnunet_results"),
            "command_template": "logs/nnunet_command_template.sh",
        },
    )
    write_markdown_report(
        root / "reports" / "uncertainty_aware_segmentation_report.md",
        "Baseline Experiment Preparation",
        [
            "## Prepared artifacts",
            f"- Folds prepared: {folds}",
            f"- nnU-Net dataset folder: `{dataset_folder}`",
            f"- imagesTr/labelsTr exported under `nnunet_raw/{dataset_folder}` using symlinks.",
            "- Fold-specific inference input folders written under `nnunet_inference/fold_x/images`.",
            "- `splits_final.json` written under `nnunet_exports/` and should be copied into `nnUNet_preprocessed` after planning.",
            "- Per-fold train/val manifests written under `models/fold_x/`.",
            "- nnU-Net command template written to `logs/nnunet_command_template.sh`.",
            "",
            "## Scope",
            "- This stage exports nnU-Net raw structure, manifests, and command templates only.",
            "- Training, probability map export, and uncertainty generation are not executed here.",
        ],
    )
    write_csv(root / "metrics" / "uncertainty_case_scores.csv", [], ("case_id", "uncertainty_score"))
    return {"folds": folds, "artifacts": len(created_files), "dataset_folder": dataset_folder}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare baseline experiment manifests and command templates.")
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--baseline-config", type=Path)
    args = parser.parse_args()
    result = run(
        n_folds=args.n_folds,
        project_root_path=args.project_root,
        run_root=args.run_root,
        baseline_config_path=args.baseline_config,
    )
    print(f"Baseline experiment prepared for {result['folds']} folds")


if __name__ == "__main__":
    main()
