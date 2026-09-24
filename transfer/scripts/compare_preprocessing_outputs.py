#!/usr/bin/env python3
"""Compare raw, legacy, and new preprocessing outputs without requiring a shared grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def geometry(image: sitk.Image) -> dict[str, Any]:
    return {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz_mm": list(image.GetSpacing()),
        "origin_xyz_mm": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }


def image_summary(path: Path) -> dict[str, Any]:
    image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    values = sitk.GetArrayViewFromImage(image)
    nonzero = values[values != 0]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "geometry": geometry(image),
        "finite_fraction": float(np.isfinite(values).mean()),
        "nonzero_voxels": int(nonzero.size),
        "nonzero_mean": float(nonzero.mean()) if nonzero.size else None,
        "nonzero_std": float(nonzero.std()) if nonzero.size else None,
        "nonzero_min": float(nonzero.min()) if nonzero.size else None,
        "nonzero_max": float(nonzero.max()) if nonzero.size else None,
    }


def label_summary(path: Path) -> dict[str, Any]:
    label = sitk.ReadImage(str(path))
    binary = sitk.Cast(label > 0, sitk.sitkUInt8)
    values = sitk.GetArrayViewFromImage(binary)
    components = sitk.ConnectedComponent(binary)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(components)
    voxel_count = int(values.sum())
    voxel_volume = float(np.prod(label.GetSpacing()))
    return {
        "path": str(path),
        "sha256": sha256(path),
        "geometry": geometry(label),
        "foreground_voxels": voxel_count,
        "foreground_volume_mm3": voxel_count * voxel_volume,
        "connected_components": len(stats.GetLabels()),
        "component_voxels_desc": sorted((int(stats.GetNumberOfPixels(item)) for item in stats.GetLabels()), reverse=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-image", "raw-label", "legacy-image", "legacy-label", "new-image", "new-label"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "raw": {"image": image_summary(args.raw_image), "label": label_summary(args.raw_label)},
        "legacy": {"image": image_summary(args.legacy_image), "label": label_summary(args.legacy_label)},
        "new_p1": {"image": image_summary(args.new_image), "label": label_summary(args.new_label)},
    }
    raw_hash = report["raw"]["image"]["sha256"]
    report["checks"] = {
        "corrected_0402_input": raw_hash == "5969ef53c6f43549c0fd42e1582e3caa2eac89d3e4dc74c453e4e319ac981fcf",
        "all_images_finite": all(report[key]["image"]["finite_fraction"] == 1.0 for key in ("raw", "legacy", "new_p1")),
        "legacy_image_label_geometry_match": report["legacy"]["image"]["geometry"] == report["legacy"]["label"]["geometry"],
        "new_image_label_geometry_match": report["new_p1"]["image"]["geometry"] == report["new_p1"]["label"]["geometry"],
    }
    raw_voxels = int(np.prod(report["raw"]["image"]["geometry"]["size_xyz"]))
    legacy_voxels = int(np.prod(report["legacy"]["image"]["geometry"]["size_xyz"]))
    raw_volume = report["raw"]["label"]["foreground_volume_mm3"]
    legacy_volume = report["legacy"]["label"]["foreground_volume_mm3"]
    new_volume = report["new_p1"]["label"]["foreground_volume_mm3"]
    report["derived"] = {
        "raw_nonzero_fraction": report["raw"]["image"]["nonzero_voxels"] / raw_voxels,
        "legacy_nonzero_fraction": report["legacy"]["image"]["nonzero_voxels"] / legacy_voxels,
        "legacy_label_volume_change_pct": 100.0 * (legacy_volume - raw_volume) / raw_volume,
        "new_label_volume_change_pct": 100.0 * (new_volume - raw_volume) / raw_volume,
    }
    warnings: list[str] = []
    if report["derived"]["legacy_nonzero_fraction"] == 1.0 and report["derived"]["raw_nonzero_fraction"] < 1.0:
        warnings.append("legacy_clipping_filled_zero_background")
    if report["new_p1"]["label"]["connected_components"] != report["raw"]["label"]["connected_components"]:
        warnings.append("new_resampling_changed_label_component_count")
    report["warnings"] = warnings
    report["valid"] = all(report["checks"].values())
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
