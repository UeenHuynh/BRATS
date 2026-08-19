from __future__ import annotations

import hashlib
import json
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from SimpleITK import Image

from .config import effective_profile, resolve_path
from .imaging import (
    apply_source_support,
    binary_volume,
    connected_components,
    create_source_support,
    crop_box,
    crop_image,
    geometry,
    geometry_matches,
    get_brain_mask,
    import_sitk,
    label_values,
    n4_correct,
    otsu_mask,
    read_spacing,
    rectangular_mask,
    resample,
    sha256_file,
    support_leakage,
    target_spacing,
    transform_intensity,
    write_image_verified,
)
from .manifest import case_paths, manifest_value, validate_inputs

VERSION = "0.1.0"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def training_spacing_plan(rows: Sequence[Mapping[str, str]], spec: Mapping[str, Any], spec_dir: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        if manifest_value(row, spec, "split").lower() != "train":
            continue
        path = case_paths(row, spec, spec_dir)["image"]
        if path is None or not path.exists():
            raise FileNotFoundError(f"Training image missing while planning spacing: {path}")
        entries.append({"case_id": manifest_value(row, spec, "case_id"), "spacing_mm": read_spacing(path)})
    if not entries:
        raise ValueError("No training rows available for median spacing")
    median = np.median(np.asarray([entry["spacing_mm"] for entry in entries], dtype=float), axis=0)
    plan: dict[str, Any] = {
        "source_split": "train",
        "case_count": len(entries),
        "cases": entries,
        "median_spacing_mm": median.tolist(),
    }
    plan["sha256"] = sha256_text(canonical_json(plan))
    return plan


@dataclass
class PreparedCase:
    image: Image
    label: Image | None
    brain: Image
    head: Image
    source_support: Image
    n4_image: Image | None
    n4_bias: Image | None
    n4_metrics: dict[str, Any]
    n4_error: str | None
    native_geometry: dict[str, Any]
    input_label_volume: float | None
    input_label_values: list[int]
    input_components: int | None
    brain_source: str
    input_hashes: dict[str, str]


def prepare_case(
    row: Mapping[str, str], spec: Mapping[str, Any], spec_dir: Path, cache_root: Path, needs_n4: bool
) -> PreparedCase:
    sitk = import_sitk()
    paths = case_paths(row, spec, spec_dir)
    image_path = paths["image"]
    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    label = None
    if paths["label"] is not None:
        if not paths["label"].exists():
            raise FileNotFoundError(f"Label not found: {paths['label']}")
        label = sitk.ReadImage(str(paths["label"]))
        if not geometry_matches(image, label):
            raise ValueError("Input image and label geometry differ")
    orientation = str(spec["pipeline"].get("orientation", "RAS"))
    image_oriented = sitk.DICOMOrient(image, orientation)
    label_oriented = sitk.DICOMOrient(label, orientation) if label is not None else None
    source_support = create_source_support(image_oriented)
    head = otsu_mask(image_oriented)
    brain_oriented = head
    brain_source = "not_used"
    n4_image = n4_bias = None
    n4_metrics: dict[str, Any] = {}
    n4_error = None
    if needs_n4:
        try:
            case_id = manifest_value(row, spec, "case_id")
            cache = cache_root / case_id
            cache.mkdir(parents=True, exist_ok=True)
            brain, brain_source, _ = get_brain_mask(
                image, image_path, paths["brain_mask"], spec["pipeline"]["brain_mask"], cache
            )
            if not geometry_matches(image, brain):
                raise ValueError("Input image and brain mask geometry differ")
            brain_oriented = sitk.Cast(sitk.DICOMOrient(brain, orientation) > 0, sitk.sitkUInt8)
            n4_image, n4_bias, n4_metrics = n4_correct(image_oriented, brain_oriented, spec["pipeline"]["n4"])
        except Exception as exc:
            n4_error = f"{type(exc).__name__}: {exc}"
            brain_oriented = head
            brain_source = "unavailable"
    hashes = {"image_sha256": sha256_file(image_path)}
    if paths["label"] is not None:
        hashes["label_sha256"] = sha256_file(paths["label"])
    if needs_n4 and paths["brain_mask"] is not None:
        hashes["brain_mask_sha256"] = sha256_file(paths["brain_mask"])
    return PreparedCase(
        image=image_oriented,
        label=label_oriented,
        brain=brain_oriented,
        head=head,
        source_support=source_support,
        n4_image=n4_image,
        n4_bias=n4_bias,
        n4_metrics=n4_metrics,
        n4_error=n4_error,
        native_geometry=geometry(image),
        input_label_volume=binary_volume(label_oriented) if label_oriented is not None else None,
        input_label_values=label_values(label_oriented) if label_oriented is not None else [],
        input_components=connected_components(label_oriented) if label_oriented is not None else None,
        brain_source=brain_source,
        input_hashes=hashes,
    )


def process_profile(
    prepared: PreparedCase,
    row: Mapping[str, str],
    spec: Mapping[str, Any],
    output_root: Path,
    profile_id: str,
    median_spacing: Sequence[float] | None,
    run_id: str,
    spec_hash: str,
    manifest_hash: str,
    overwrite: bool,
) -> dict[str, Any]:
    sitk = import_sitk()
    started = time.time()
    config = effective_profile(spec, profile_id)
    case_id = manifest_value(row, spec, "case_id")
    split = manifest_value(row, spec, "split").lower()
    case_root = output_root / "datasets" / profile_id / split
    image_path = case_root / "images" / f"{case_id}.nii.gz"
    label_path = case_root / "labels" / f"{case_id}.nii.gz"
    expects_label = bool(manifest_value(row, spec, "label"))
    outputs_complete = image_path.exists() and (not expects_label or label_path.exists())
    if outputs_complete and not overwrite:
        return {
            "run_id": run_id,
            "profile_id": profile_id,
            "case_id": case_id,
            "split": split,
            "status": "SKIPPED",
            "qc_status": "NOT_RUN",
            "output_image": str(image_path),
        }
    uses_n4 = bool(config["n4"].get("enabled", False))
    source = prepared.n4_image if uses_n4 else prepared.image
    if source is None:
        raise RuntimeError(prepared.n4_error or "N4 output was not prepared")
    crop_config = config["crop"]
    crop_enabled = bool(crop_config.get("enabled", False))
    rectangle_config = config.get("rectangular_mask", {})
    rectangle_enabled = bool(rectangle_config.get("enabled", False))
    rectangle_margin_mm = float(rectangle_config.get("margin_mm", 0))
    rectangle = rectangular_mask(prepared.head, rectangle_margin_mm) if rectangle_enabled else None
    if crop_enabled:
        index, size = crop_box(prepared.head, float(crop_config.get("margin_mm", 0)))
        source = crop_image(source, index, size)
        label = crop_image(prepared.label, index, size) if prepared.label is not None else None
        brain = crop_image(prepared.brain, index, size)
        head = crop_image(prepared.head, index, size)
        source_support = crop_image(prepared.source_support, index, size)
        rectangle = crop_image(rectangle, index, size) if rectangle is not None else None
    else:
        index, size = [0, 0, 0], list(geometry(source)["size"])
        label, brain, head = prepared.label, prepared.brain, prepared.head
        source_support = prepared.source_support
    effective_support = sitk.Cast(source_support > 0, sitk.sitkUInt8)
    if rectangle is not None:
        effective_support = sitk.And(effective_support, sitk.Cast(rectangle > 0, sitk.sitkUInt8))
    image, intensity_metrics = transform_intensity(source, brain, head, effective_support, config)
    after_crop_volume = binary_volume(label) if label is not None else None
    if prepared.input_label_volume and after_crop_volume is not None:
        retained = 100.0 * after_crop_volume / prepared.input_label_volume
        if retained < 100.0 - float(config["qc"].get("crop_volume_tolerance_pct", 0.01)):
            raise ValueError(f"Crop removed labeled volume: retained={retained:.6f}%")
    spacing = target_spacing(config["resample"], median_spacing)
    if spacing is not None:
        image = resample(image, spacing, False)
        label = resample(label, spacing, True) if label is not None else None
        brain = resample(brain, spacing, True)
        head = resample(head, spacing, True)
        source_support = resample(source_support, spacing, True)
        effective_support = resample(effective_support, spacing, True)
    brain = sitk.Cast(brain > 0, sitk.sitkUInt8)
    head = sitk.Cast(head > 0, sitk.sitkUInt8)
    source_support = sitk.Cast(source_support > 0, sitk.sitkUInt8)
    effective_support = sitk.Cast(effective_support > 0, sitk.sitkUInt8)
    pre_mask_leakage_count, pre_mask_leakage_abs_max = support_leakage(image, effective_support)
    image = apply_source_support(image, effective_support)
    post_mask_leakage_count, post_mask_leakage_abs_max = support_leakage(image, effective_support)
    if post_mask_leakage_count:
        raise RuntimeError(f"Output has {post_mask_leakage_count} nonzero voxels outside effective support")
    outside_head = sitk.GetArrayViewFromImage(head) == 0
    outside_head_nonzero_count = int(np.count_nonzero(sitk.GetArrayViewFromImage(image)[outside_head]))
    if label is not None and not geometry_matches(image, label):
        raise RuntimeError("Output image and label geometry differ")
    output_values = label_values(label) if label is not None else []
    output_components = connected_components(label) if label is not None else None
    if label is not None and set(output_values) != set(prepared.input_label_values):
        raise ValueError("Label classes changed during preprocessing")
    finite_fraction = float(np.isfinite(sitk.GetArrayViewFromImage(image)).mean())
    if finite_fraction < float(config["qc"].get("minimum_finite_fraction", 1.0)):
        raise ValueError(f"Output finite fraction is {finite_fraction}")
    output_label_volume = binary_volume(label) if label is not None else None
    volume_change = None
    if prepared.input_label_volume and output_label_volume is not None:
        volume_change = 100.0 * (output_label_volume - prepared.input_label_volume) / prepared.input_label_volume
    warnings: list[str] = []
    if volume_change is not None and abs(volume_change) > float(config["qc"].get("max_label_volume_change_pct", 10.0)):
        warnings.append("label_volume_change")
    if prepared.input_components is not None and output_components != prepared.input_components:
        warnings.append("connected_component_count_changed")
    output_hashes = {"image_sha256": write_image_verified(image, image_path)}
    if label is not None:
        output_hashes["label_sha256"] = write_image_verified(label, label_path)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "profile_id": profile_id,
        "case_id": case_id,
        "patient_id": manifest_value(row, spec, "patient_id"),
        "split": manifest_value(row, spec, "split"),
        "fold": manifest_value(row, spec, "fold"),
        "site": manifest_value(row, spec, "site"),
        "spec_sha256": spec_hash,
        "manifest_sha256": manifest_hash,
        "effective_config": config,
        "input_hashes": prepared.input_hashes,
        "output_hashes": output_hashes,
        "native_geometry": prepared.native_geometry,
        "oriented_geometry": geometry(prepared.image),
        "output_geometry": geometry(image),
        "operations": {
            "orientation": config.get("orientation"),
            "brain_mask_source": prepared.brain_source if uses_n4 else "not_used",
            "n4": prepared.n4_metrics if uses_n4 else {"enabled": False},
            "intensity": intensity_metrics,
            "crop": {
                "enabled": crop_enabled,
                "mask_source": "largest_component_otsu_head" if crop_enabled else None,
                "index_xyz": index,
                "size_xyz": size,
            },
            "source_support": {
                "source": "oriented_raw_nonzero",
                "interpolation": "nearest_neighbor" if spacing is not None else "none",
                "applied_after_resampling": True,
                "outside_value": 0.0,
            },
            "rectangular_mask": {
                "enabled": rectangle_enabled,
                "source": "largest_component_otsu_head_bounding_box" if rectangle_enabled else None,
                "margin_mm": rectangle_margin_mm if rectangle_enabled else None,
                "combined_with_source_support": rectangle_enabled,
                "outside_value": 0.0 if rectangle_enabled else None,
            },
            "target_spacing_mm": spacing,
        },
        "qc": {
            "status": "WARN" if warnings else "PASS",
            "warnings": warnings,
            "finite_fraction": finite_fraction,
            "input_label_volume_mm3": prepared.input_label_volume,
            "after_crop_label_volume_mm3": after_crop_volume,
            "output_label_volume_mm3": output_label_volume,
            "label_volume_change_pct": volume_change,
            "input_label_values": prepared.input_label_values,
            "output_label_values": output_values,
            "input_connected_components": prepared.input_components,
            "output_connected_components": output_components,
            "brain_volume_mm3": binary_volume(brain),
            "head_volume_mm3": binary_volume(head),
            "source_support_volume_mm3": binary_volume(source_support),
            "effective_support_volume_mm3": binary_volume(effective_support),
            "pre_mask_nonzero_voxels_outside_effective_support": pre_mask_leakage_count,
            "pre_mask_max_abs_outside_effective_support": pre_mask_leakage_abs_max,
            "post_mask_nonzero_voxels_outside_effective_support": post_mask_leakage_count,
            "post_mask_max_abs_outside_effective_support": post_mask_leakage_abs_max,
            "outside_head_nonzero_voxels": outside_head_nonzero_count,
        },
        "created_utc": utc_now(),
    }
    return {
        "run_id": run_id,
        "profile_id": profile_id,
        "case_id": case_id,
        "patient_id": metadata["patient_id"],
        "split": metadata["split"],
        "fold": metadata["fold"],
        "site": metadata["site"],
        "status": "SUCCESS",
        "qc_status": metadata["qc"]["status"],
        "warnings": ";".join(warnings),
        "output_image": str(image_path),
        "output_label": str(label_path) if label is not None else "",
        "metadata": metadata,
        "runtime_sec": round(time.time() - started, 3),
    }


def run(
    spec_path: Path,
    manifest_path: Path,
    profiles: Sequence[str] | None,
    case_limit: int | None,
    overwrite: bool,
    fail_fast: bool,
    dry_run: bool,
) -> int:
    spec, all_rows = validate_inputs(spec_path, manifest_path)
    spec_dir = spec_path.parent
    selected = list(
        profiles or [profile_id for profile_id, value in spec["profiles"].items() if bool(value.get("enabled", True))]
    )
    for profile_id in selected:
        effective_profile(spec, profile_id)
    output_root = resolve_path(spec["project"]["output_root"], spec_dir)
    if output_root is None:
        raise ValueError("project.output_root is required")
    output_root.mkdir(parents=True, exist_ok=True)
    spec_hash = sha256_text(canonical_json(spec))
    manifest_hash = sha256_file(manifest_path)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + spec_hash[:8]
    run_root = output_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    needs_median = any(
        str(effective_profile(spec, profile_id)["resample"].get("target_spacing_mm", "")).lower() == "training_median"
        for profile_id in selected
    )
    spacing_plan = training_spacing_plan(all_rows, spec, spec_dir) if needs_median else None
    summary = {
        "status": "DRY_RUN_OK" if dry_run else "RUNNING",
        "run_id": run_id,
        "cases_in_manifest": len(all_rows),
        "profiles": selected,
        "spec_sha256": spec_hash,
        "manifest_sha256": manifest_hash,
        "output_root": str(output_root),
        "training_median_spacing_mm": spacing_plan["median_spacing_mm"] if spacing_plan else None,
    }
    run_record: dict[str, Any] = {
        "schema_version": 1,
        "summary": summary,
        "resolved_config": spec,
        "manifest": {"source": str(manifest_path), "sha256": manifest_hash, "rows": all_rows},
        "training_spacing_plan": spacing_plan,
        "cases": [],
    }
    run_path = run_root / "run.json"
    if dry_run:
        run_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    rows = all_rows[:case_limit] if case_limit is not None else all_rows
    needs_n4 = any(bool(effective_profile(spec, profile_id)["n4"].get("enabled", False)) for profile_id in selected)
    records: list[dict[str, Any]] = []
    failures = 0
    for position, row in enumerate(rows, start=1):
        case_id = manifest_value(row, spec, "case_id")
        print(f"[{position}/{len(rows)}] {case_id}", flush=True)
        try:
            prepared = prepare_case(row, spec, spec_dir, output_root / "cache", needs_n4)
            for profile_id in selected:
                try:
                    records.append(
                        process_profile(
                            prepared,
                            row,
                            spec,
                            output_root,
                            profile_id,
                            spacing_plan["median_spacing_mm"] if spacing_plan else None,
                            run_id,
                            spec_hash,
                            manifest_hash,
                            overwrite,
                        )
                    )
                except Exception as exc:
                    failures += 1
                    records.append(failure_record(row, spec, profile_id, run_id, exc))
                    if fail_fast:
                        raise
        except Exception as exc:
            failures += len(selected)
            records.extend(failure_record(row, spec, profile_id, run_id, exc) for profile_id in selected)
            if fail_fast:
                raise
    summary.update(
        {
            "status": "DONE" if failures == 0 else "DONE_WITH_FAILURES",
            "processed_cases": len(rows),
            "failure_records": failures,
            "finished_utc": utc_now(),
        }
    )
    run_record["cases"] = records
    run_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if failures == 0 else 2


def failure_record(
    row: Mapping[str, str], spec: Mapping[str, Any], profile_id: str, run_id: str, exc: BaseException
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "profile_id": profile_id,
        "case_id": manifest_value(row, spec, "case_id"),
        "patient_id": manifest_value(row, spec, "patient_id"),
        "split": manifest_value(row, spec, "split"),
        "fold": manifest_value(row, spec, "fold"),
        "site": manifest_value(row, spec, "site"),
        "status": "FAILED",
        "qc_status": "FAIL",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(limit=5),
    }
