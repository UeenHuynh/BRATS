#!/usr/bin/env python3
"""Validate case membership of portable P0--P3 datasets without imaging dependencies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROFILES = (
    "P0_MEDIAN_ZSCORE",
    "P1_MEDIAN_CLIP_ZSCORE",
    "P2_MEDIAN_N4_ZSCORE",
    "P3_MEDIAN_N4_CLIP_ZSCORE",
)


def case_ids(path: Path) -> set[str]:
    return {item.name.removesuffix(".nii.gz") for item in path.glob("*.nii.gz")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_train = {row["case_id"] for row in rows if row["split"] == "train"}
    expected_validation = {row["case_id"] for row in rows if row["split"] == "validation"}

    report: dict[str, object] = {
        "manifest_train_cases": len(expected_train),
        "manifest_validation_cases": len(expected_validation),
        "profiles": {},
    }
    common_train: set[str] | None = None
    profile_reports: dict[str, object] = {}
    for profile in PROFILES:
        root = args.dataset_root / profile
        train_images = case_ids(root / "train" / "images")
        train_labels = case_ids(root / "train" / "labels")
        validation_images = case_ids(root / "validation" / "images")
        usable = train_images & train_labels
        common_train = usable if common_train is None else common_train & usable
        profile_reports[profile] = {
            "train_images": len(train_images),
            "train_labels": len(train_labels),
            "usable_train_cases": len(usable),
            "validation_images": len(validation_images),
            "missing_train_images": sorted(expected_train - train_images),
            "missing_train_labels": sorted(expected_train - train_labels),
            "unexpected_train_images": sorted(train_images - expected_train),
            "unexpected_train_labels": sorted(train_labels - expected_train),
            "missing_validation_images": sorted(expected_validation - validation_images),
            "unexpected_validation_images": sorted(validation_images - expected_validation),
        }
    report["profiles"] = profile_reports
    report["common_usable_train_cases"] = len(common_train or set())
    report["common_missing_train_cases"] = sorted(expected_train - (common_train or set()))

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
