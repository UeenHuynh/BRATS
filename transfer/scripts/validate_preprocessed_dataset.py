#!/usr/bin/env python3
"""Validate manifest integrity and case membership of portable P0--P3 datasets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=PROFILES,
        help="Dataset profile names to validate (defaults to the canonical P0--P3 profiles)",
    )
    parser.add_argument("--expected-train-count", type=int)
    parser.add_argument("--expected-validation-count", type=int)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    expected_train = {row["case_id"] for row in train_rows}
    expected_validation = {row["case_id"] for row in validation_rows}
    duplicate_case_ids = sorted(case_id for case_id, count in Counter(row["case_id"] for row in rows).items() if count > 1)
    split_overlap = sorted(expected_train & expected_validation)
    errors: list[str] = []
    if duplicate_case_ids:
        errors.append("manifest contains duplicate case IDs")
    if split_overlap:
        errors.append("manifest contains case IDs in both train and validation")
    if args.expected_train_count is not None and len(expected_train) != args.expected_train_count:
        errors.append(f"expected {args.expected_train_count} unique train cases, found {len(expected_train)}")
    if args.expected_validation_count is not None and len(expected_validation) != args.expected_validation_count:
        errors.append(
            f"expected {args.expected_validation_count} unique validation cases, found {len(expected_validation)}"
        )

    report: dict[str, object] = {
        "manifest_rows": len(rows),
        "manifest_train_rows": len(train_rows),
        "manifest_validation_rows": len(validation_rows),
        "manifest_train_cases": len(expected_train),
        "manifest_validation_cases": len(expected_validation),
        "duplicate_case_ids": duplicate_case_ids,
        "train_validation_overlap": split_overlap,
        "profiles": {},
    }
    common_train: set[str] | None = None
    profile_reports: dict[str, object] = {}
    for profile in args.profiles:
        root = args.dataset_root / profile
        train_images = case_ids(root / "train" / "images")
        train_labels = case_ids(root / "train" / "labels")
        validation_images = case_ids(root / "validation" / "images")
        usable = train_images & train_labels
        common_train = usable if common_train is None else common_train & usable
        profile_report = {
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
        profile_reports[profile] = profile_report
        if any(
            profile_report[key]
            for key in (
                "missing_train_images",
                "missing_train_labels",
                "unexpected_train_images",
                "unexpected_train_labels",
                "missing_validation_images",
                "unexpected_validation_images",
            )
        ):
            errors.append(f"{profile} membership differs from manifest")
    report["profiles"] = profile_reports
    report["common_usable_train_cases"] = len(common_train or set())
    report["common_missing_train_cases"] = sorted(expected_train - (common_train or set()))
    report["errors"] = errors
    report["valid"] = not errors

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
