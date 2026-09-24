#!/usr/bin/env python3
"""Validate corrected legacy and new-P1 preprocessing over a small case set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_preprocessing_outputs import image_summary, label_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    records: list[dict[str, object]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for row in rows:
        case_id = row["case_id"]
        raw_image = args.data_root / row["t1c_path"]
        raw_label = args.data_root / row["label_path"]
        legacy_image = args.legacy_root / "images" / f"{case_id}_t1c.nii.gz"
        legacy_label = args.legacy_root / "masks" / f"{case_id}_gtv.nii.gz"
        new_image = args.new_root / "datasets" / "P1_MEDIAN_CLIP_ZSCORE" / "train" / "images" / f"{case_id}.nii.gz"
        new_label = args.new_root / "datasets" / "P1_MEDIAN_CLIP_ZSCORE" / "train" / "labels" / f"{case_id}.nii.gz"
        try:
            raw_i, raw_l = image_summary(raw_image), label_summary(raw_label)
            old_i, old_l = image_summary(legacy_image), label_summary(legacy_label)
            new_i, new_l = image_summary(new_image), label_summary(new_label)
        except Exception as exc:
            errors.append(f"{case_id}: missing/unreadable output: {type(exc).__name__}: {exc}")
            continue
        old_total = int(np.prod(old_i["geometry"]["size_xyz"]))
        old_volume_delta = 100.0 * (old_l["foreground_volume_mm3"] - raw_l["foreground_volume_mm3"]) / raw_l["foreground_volume_mm3"]
        new_volume_delta = 100.0 * (new_l["foreground_volume_mm3"] - raw_l["foreground_volume_mm3"]) / raw_l["foreground_volume_mm3"]
        checks = {
            "finite": old_i["finite_fraction"] == 1.0 and new_i["finite_fraction"] == 1.0,
            "legacy_geometry_match": old_i["geometry"] == old_l["geometry"],
            "new_geometry_match": new_i["geometry"] == new_l["geometry"],
            "legacy_preserves_zero_background": old_i["nonzero_voxels"] < old_total,
            "legacy_label_volume_preserved": abs(old_volume_delta) < 1e-6,
            "new_label_volume_within_10pct": abs(new_volume_delta) <= 10.0,
        }
        if case_id == "BraTS-MEN-RT-0402-1":
            checks["corrected_0402_input"] = raw_i["sha256"] == "5969ef53c6f43549c0fd42e1582e3caa2eac89d3e4dc74c453e4e319ac981fcf"
        for name, passed in checks.items():
            if not passed:
                errors.append(f"{case_id}: {name}")
        if new_l["connected_components"] != raw_l["connected_components"]:
            warnings.append(f"{case_id}: new resampling changed connected components")
        records.append(
            {
                "case_id": case_id,
                "checks": checks,
                "legacy_nonzero_fraction": old_i["nonzero_voxels"] / old_total,
                "legacy_label_volume_change_pct": old_volume_delta,
                "new_label_volume_change_pct": new_volume_delta,
                "raw_components": raw_l["connected_components"],
                "legacy_components": old_l["connected_components"],
                "new_components": new_l["connected_components"],
            }
        )
    report = {"cases": records, "errors": errors, "warnings": warnings, "valid": not errors}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
