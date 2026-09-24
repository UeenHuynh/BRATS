#!/usr/bin/env python3
"""Validate duplicate-free legacy preprocessing membership and required pairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def ids(path: Path, suffix: str) -> set[str]:
    return {item.name.removesuffix(suffix) for item in path.glob(f"*{suffix}")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=500)
    parser.add_argument("--expected-validation", type=int, default=70)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    counts = Counter(row["case_id"] for row in rows)
    duplicates = sorted(case_id for case_id, count in counts.items() if count > 1)
    train = {row["case_id"] for row in rows if row["split_group"] == "train"}
    validation = {row["case_id"] for row in rows if row["split_group"] == "val_unlabeled"}
    images = ids(args.output_root / "images", "_t1c.nii.gz")
    labels = ids(args.output_root / "masks", "_gtv.nii.gz")
    expected_images = train | validation
    errors: list[str] = []
    if duplicates:
        errors.append(f"duplicate manifest IDs: {duplicates}")
    if len(train) != args.expected_train or len(validation) != args.expected_validation:
        errors.append(f"expected {args.expected_train}/{args.expected_validation}, found {len(train)}/{len(validation)}")
    if images != expected_images:
        errors.append(f"image membership mismatch: missing={sorted(expected_images-images)}, extra={sorted(images-expected_images)}")
    if labels != train:
        errors.append(f"label membership mismatch: missing={sorted(train-labels)}, extra={sorted(labels-train)}")
    report = {
        "train_cases": len(train),
        "validation_cases": len(validation),
        "images": len(images),
        "labels": len(labels),
        "duplicates": duplicates,
        "errors": errors,
        "valid": not errors,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
