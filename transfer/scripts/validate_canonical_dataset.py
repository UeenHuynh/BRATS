#!/usr/bin/env python3
"""Fail-fast validation gate for the canonical BraTS-MEN-RT raw view."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PATCH_CASE_ID = "BraTS-MEN-RT-0402-1"
PATCH_IMAGE_SHA256 = "5969ef53c6f43549c0fd42e1582e3caa2eac89d3e4dc74c453e4e319ac981fcf"
STALE_IMAGE_SHA256 = "0033422b0b8c1ce6df10524641b7f9524c6c036515b8212a48c171574ccda059"
ARCHIVE_MD5 = {
    "data/BraTS2024-MEN-RT-TrainingData.zip": "5bd14a6794e5874b6dd2a09a64795a4c",
    "data/BraTS2024-MEN-RT-ValidationData.zip": "885678ebe03224a3cce4b3a82f861c99",
    "data/BraTS-MEN-RT-0402-1.zip": "aadfd702c07ce72709e5249b73333ad8",
}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=workspace_default / "code" / "brats-research" / "manifests" / "local_dataset.csv",
    )
    parser.add_argument("--verify-archives", action="store_true", help="Also hash the three large source ZIP files")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    manifest = args.manifest.resolve()
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    counts = Counter(row["case_id"] for row in rows)
    duplicates = sorted(case_id for case_id, count in counts.items() if count > 1)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    errors: list[str] = []
    if len(rows) != 570 or len(train_rows) != 500 or len(validation_rows) != 70:
        errors.append(f"expected 570 rows (500 train, 70 validation), found {len(rows)} ({len(train_rows)}, {len(validation_rows)})")
    if duplicates:
        errors.append("duplicate case IDs: " + ", ".join(duplicates))

    missing_paths: list[str] = []
    for row in rows:
        required = [row["t1c_path"]]
        if row["split"] == "train":
            required.append(row["label_path"])
        for relative in required:
            if not relative or not (workspace / relative).is_file():
                missing_paths.append(relative or f"{row['case_id']}:empty-path")
    if missing_paths:
        errors.append(f"{len(missing_paths)} required manifest paths are missing")

    patch_link = workspace / "datasets" / "raw_canonical" / "train" / PATCH_CASE_ID
    patch_source = workspace / "data" / "extracted" / PATCH_CASE_ID
    if not patch_link.is_symlink():
        errors.append("canonical 0402 directory is not a symlink")
    elif patch_link.resolve() != patch_source.resolve():
        errors.append(f"canonical 0402 resolves to the wrong source: {patch_link.resolve()}")

    patch_image = patch_link / f"{PATCH_CASE_ID}_t1c.nii.gz"
    patch_sha = digest(patch_image, "sha256") if patch_image.is_file() else None
    if patch_sha != PATCH_IMAGE_SHA256:
        errors.append(f"canonical 0402 image checksum mismatch: {patch_sha}")
    if patch_sha == STALE_IMAGE_SHA256:
        errors.append("canonical 0402 still points to the stale training-v2 image")

    archive_checks: dict[str, object] = {}
    if args.verify_archives:
        for relative, expected in ARCHIVE_MD5.items():
            path = workspace / relative
            actual = digest(path, "md5") if path.is_file() else None
            archive_checks[relative] = {"expected_md5": expected, "actual_md5": actual, "valid": actual == expected}
            if actual != expected:
                errors.append(f"archive checksum mismatch: {relative}")

    report = {
        "manifest": str(manifest),
        "rows": len(rows),
        "train_cases": len(train_rows),
        "validation_cases": len(validation_rows),
        "duplicate_case_ids": duplicates,
        "missing_paths": missing_paths,
        "canonical_0402_target": str(patch_link.resolve()) if patch_link.exists() else None,
        "canonical_0402_image_sha256": patch_sha,
        "archive_checks": archive_checks,
        "errors": errors,
        "valid": not errors,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
