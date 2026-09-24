#!/usr/bin/env python3
"""Build a duplicate-free BraTS-MEN-RT raw view without modifying source data."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


PATCH_CASE_ID = "BraTS-MEN-RT-0402-1"
EXPECTED_TRAIN = 500
EXPECTED_VALIDATION = 70
FIELDS = (
    "case_id",
    "patient_id",
    "session_id",
    "split",
    "fold",
    "site",
    "t1c_path",
    "label_path",
    "brain_mask_path",
)


def case_files(case_dir: Path, *, labeled: bool) -> tuple[Path, Path | None]:
    case_id = case_dir.name
    image = case_dir / f"{case_id}_t1c.nii.gz"
    label = case_dir / f"{case_id}_gtv.nii.gz" if labeled else None
    if not image.is_file():
        raise FileNotFoundError(f"Missing image: {image}")
    if label is not None and not label.is_file():
        raise FileNotFoundError(f"Missing label: {label}")
    return image, label


def ensure_relative_symlink(link: Path, source: Path) -> None:
    expected = Path(os.path.relpath(source, start=link.parent))
    if link.is_symlink():
        if Path(os.readlink(link)) != expected:
            raise RuntimeError(f"Existing symlink has the wrong target: {link} -> {os.readlink(link)}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink path: {link}")
    link.symlink_to(expected, target_is_directory=True)


def manifest_row(workspace: Path, case_dir: Path, split: str) -> dict[str, str]:
    case_id = case_dir.name
    labeled = split == "train"
    image, label = case_files(case_dir, labeled=labeled)
    relative_image = image.relative_to(workspace).as_posix()
    relative_label = label.relative_to(workspace).as_posix() if label is not None else ""
    patient_id, session_id = case_id.rsplit("-", 1)
    return {
        "case_id": case_id,
        "patient_id": patient_id,
        "session_id": session_id,
        "split": split,
        "fold": "",
        "site": "",
        "t1c_path": relative_image,
        "label_path": relative_label,
        "brain_mask_path": "",
    }


def build(workspace: Path, manifest: Path) -> None:
    extracted = workspace / "data" / "extracted"
    train_source = extracted / "BraTS-MEN-RT-Train-v2"
    validation_source = extracted / "BraTS-MEN-RT-Val-v1"
    patch_source = extracted / PATCH_CASE_ID
    view = workspace / "datasets" / "raw_canonical"
    train_view = view / "train"
    validation_view = view / "validation"
    train_view.mkdir(parents=True, exist_ok=True)
    validation_view.mkdir(parents=True, exist_ok=True)

    train_cases = sorted(path for path in train_source.iterdir() if path.is_dir())
    validation_cases = sorted(path for path in validation_source.iterdir() if path.is_dir())
    if len(train_cases) != EXPECTED_TRAIN:
        raise RuntimeError(f"Expected {EXPECTED_TRAIN} train cases, found {len(train_cases)}")
    if len(validation_cases) != EXPECTED_VALIDATION:
        raise RuntimeError(f"Expected {EXPECTED_VALIDATION} validation cases, found {len(validation_cases)}")
    case_files(patch_source, labeled=True)

    rows: list[dict[str, str]] = []
    for original in train_cases:
        source = patch_source if original.name == PATCH_CASE_ID else original
        link = train_view / original.name
        ensure_relative_symlink(link, source)
        rows.append(manifest_row(workspace, link, "train"))
    for source in validation_cases:
        link = validation_view / source.name
        ensure_relative_symlink(link, source)
        rows.append(manifest_row(workspace, link, "validation"))

    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("Canonical manifest contains duplicate case IDs")
    patch_link = train_view / PATCH_CASE_ID
    if patch_link.resolve() != patch_source.resolve():
        raise RuntimeError("Canonical 0402 does not resolve to the standalone patch")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest)

    provenance = {
        "schema_version": 1,
        "train_cases": EXPECTED_TRAIN,
        "validation_cases": EXPECTED_VALIDATION,
        "training_source": {
            "path": "data/BraTS2024-MEN-RT-TrainingData.zip",
            "synapse_id": "syn60085033",
            "version": 2,
            "md5": "5bd14a6794e5874b6dd2a09a64795a4c",
        },
        "validation_source": {
            "path": "data/BraTS2024-MEN-RT-ValidationData.zip",
            "synapse_id": "syn61484746",
            "version": 1,
            "md5": "885678ebe03224a3cce4b3a82f861c99",
        },
        "case_overrides": {
            PATCH_CASE_ID: {
                "path": "data/BraTS-MEN-RT-0402-1.zip",
                "synapse_id": "syn64826221",
                "version": 1,
                "md5": "aadfd702c07ce72709e5249b73333ad8",
                "image_sha256": "5969ef53c6f43549c0fd42e1582e3caa2eac89d3e4dc74c453e4e319ac981fcf",
                "label_sha256": "72e1bea52eb70845bb123dcf69772269f8b59c822ebdaa4e65095379a36cb457",
                "replaces_image_sha256": "0033422b0b8c1ce6df10524641b7f9524c6c036515b8212a48c171574ccda059",
            }
        },
        "policy": "The standalone 0402 patch replaces, rather than supplements, the copy in training v2.",
        "manifest": manifest.relative_to(workspace).as_posix(),
    }
    (view / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


def main() -> None:
    default_workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=default_workspace)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_workspace / "code" / "brats-research" / "manifests" / "local_dataset.csv",
    )
    args = parser.parse_args()
    build(args.workspace.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main()
