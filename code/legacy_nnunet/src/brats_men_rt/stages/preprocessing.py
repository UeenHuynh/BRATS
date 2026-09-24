from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from brats_men_rt.reports import write_markdown_report
from brats_men_rt.utils import append_jsonl, read_json, write_csv


def _require_nibabel():
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency `nibabel`. Create the conda env from configs/environment.yml "
            "or install from requirements.txt before running preprocessing."
        ) from exc
    return nib


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_unique_case_ids(rows: list[dict[str, str]]) -> None:
    counts = Counter(row.get("case_id", "").strip() for row in rows if row.get("case_id", "").strip())
    duplicates = sorted(case_id for case_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Manifest contains duplicate case IDs: {', '.join(duplicates)}")


def _resolve_input_path(value: str, data_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def _crop_bounds(
    image: np.ndarray,
    mask: np.ndarray | None,
    margin_voxels: tuple[int, int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    nonzero_coords = np.argwhere(image != 0)
    mask_coords = np.argwhere(mask > 0) if mask is not None else np.empty((0, 3), dtype=np.int64)
    if nonzero_coords.size == 0 and mask_coords.size == 0:
        shape = image.shape[:3]
        return tuple((0, int(shape[idx])) for idx in range(3))  # type: ignore[return-value]
    coords = nonzero_coords if mask_coords.size == 0 else np.vstack([nonzero_coords, mask_coords])
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    bounds = []
    for axis in range(3):
        lo = max(0, int(mins[axis]) - int(margin_voxels[axis]))
        hi = min(image.shape[axis], int(maxs[axis]) + int(margin_voxels[axis]))
        bounds.append((lo, hi))
    return bounds[0], bounds[1], bounds[2]


def _apply_crop(array: np.ndarray, bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> np.ndarray:
    x, y, z = bounds
    return array[x[0] : x[1], y[0] : y[1], z[0] : z[1]]


def _normalize_image(
    image: np.ndarray,
    clipping_percentiles: tuple[float, float],
    region_mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    work = image.astype(np.float32, copy=True)
    if region_mode == "nonzero":
        support = work != 0
        region = work[support]
    else:
        region = work.reshape(-1)
    if region.size == 0:
        return work, {"clip_low": 0.0, "clip_high": 0.0, "mean": 0.0, "std": 1.0}
    low, high = np.percentile(region, clipping_percentiles)
    if region_mode == "nonzero":
        work[support] = np.clip(work[support], low, high)
        masked = work[support]
    else:
        work = np.clip(work, low, high)
        masked = work.reshape(-1)
    mean = float(masked.mean()) if masked.size else 0.0
    std = float(masked.std()) if masked.size else 1.0
    std = std if std > 0 else 1.0
    if region_mode == "nonzero":
        work[support] = (work[support] - mean) / std
    else:
        work = (work - mean) / std
    return work, {"clip_low": float(low), "clip_high": float(high), "mean": mean, "std": std}


def _cropped_affine(affine: np.ndarray, bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> np.ndarray:
    start = np.asarray([bounds[0][0], bounds[1][0], bounds[2][0], 1.0], dtype=np.float64)
    cropped = affine.copy()
    cropped[:3, 3] = (affine @ start)[:3]
    return cropped


def _save_nifti(nib, data: np.ndarray, affine: np.ndarray, header, path: Path, *, is_mask: bool) -> None:
    out_header = header.copy()
    if is_mask:
        payload = data.astype(np.uint8, copy=False)
        out_header.set_data_dtype(np.uint8)
    else:
        payload = data.astype(np.float32, copy=False)
        out_header.set_data_dtype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(payload, affine, header=out_header), str(path))


def run(
    limit: int | None = None,
    progress_every: int = 10,
    *,
    project_root_path: Path | None = None,
    manifest_path: Path | None = None,
    config_path: Path | None = None,
    data_root: Path | None = None,
    output_root: Path | None = None,
    artifact_root: Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    nib = _require_nibabel()
    root = (project_root_path or Path(__file__).resolve().parents[3]).resolve()
    manifest = (manifest_path or root / "manifests" / "manifest_raw.csv").resolve()
    config_file = (config_path or root / "configs" / "preprocessing_config.json").resolve()
    input_root = (data_root or root).resolve()
    outputs = (output_root or root / "data_processed").resolve()
    artifacts = (artifact_root or root).resolve()
    config = read_json(config_file)
    manifest_rows = _read_manifest(manifest)
    _validate_unique_case_ids(manifest_rows)

    resampling = config.get("resampling", {})
    if isinstance(resampling, dict) and bool(resampling.get("enabled")):
        raise SystemExit("Resampling is intentionally not implemented yet. Set `resampling.enabled` to false first.")

    clip_percentiles = tuple(config.get("intensity_clipping_percentiles", [0.5, 99.5]))
    crop_margin = tuple(config.get("crop_margin_voxels", [8, 8, 8]))
    region_mode = str(config.get("normalization_region", "nonzero"))

    if dry_run:
        missing: list[str] = []
        for row in manifest_rows:
            if row.get("image_path") and not _resolve_input_path(row["image_path"], input_root).is_file():
                missing.append(row["image_path"])
            if row.get("mask_path") and not _resolve_input_path(row["mask_path"], input_root).is_file():
                missing.append(row["mask_path"])
        if missing:
            raise FileNotFoundError(f"dry-run found {len(missing)} missing inputs; first={missing[0]}")
        summary = {
            "status": "DRY_RUN_OK",
            "manifest": str(manifest),
            "config": str(config_file),
            "data_root": str(input_root),
            "output_root": str(outputs),
            "artifact_root": str(artifacts),
            "cases": len(manifest_rows),
        }
        print(json.dumps(summary, indent=2))
        return {"processed": 0, "failed": 0, "dry_run_cases": len(manifest_rows)}

    processed_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    processed = 0

    for row in manifest_rows:
        if not row["image_path"]:
            continue
        case_id = row["case_id"]
        image_path = _resolve_input_path(row["image_path"], input_root)
        mask_path = _resolve_input_path(row["mask_path"], input_root) if row["mask_path"] else None
        try:
            image_nii = nib.as_closest_canonical(nib.load(str(image_path)))
            image = np.asarray(image_nii.get_fdata(dtype=np.float32), dtype=np.float32)
            mask_nii = None
            mask = None
            if mask_path is not None:
                mask_nii = nib.as_closest_canonical(nib.load(str(mask_path)))
                mask = np.asarray(mask_nii.get_fdata(), dtype=np.float32)
                if image.shape != mask.shape:
                    raise ValueError(f"shape mismatch after canonical orientation: image={image.shape}, mask={mask.shape}")
                if not np.allclose(image_nii.affine, mask_nii.affine, rtol=0, atol=1e-5):
                    raise ValueError("affine mismatch after canonical orientation")

            normalized, norm_stats = _normalize_image(image, clip_percentiles, region_mode)
            # Derive spatial support from the raw oriented image. A valid foreground
            # voxel can normalize to exactly zero and must not disappear from the crop.
            bounds = _crop_bounds(image, mask, crop_margin)
            cropped_image = _apply_crop(normalized, bounds)
            cropped_mask = _apply_crop(mask, bounds) if mask is not None else None
            if cropped_mask is not None:
                cropped_mask = (cropped_mask > 0).astype(np.uint8)

            image_out = outputs / "images" / f"{case_id}_t1c.nii.gz"
            mask_out = outputs / "masks" / f"{case_id}_gtv.nii.gz" if cropped_mask is not None else None
            if not overwrite and (image_out.exists() or (mask_out is not None and mask_out.exists())):
                raise FileExistsError(f"Refusing to overwrite existing output for {case_id} under {outputs}")
            image_affine = _cropped_affine(image_nii.affine, bounds)
            _save_nifti(nib, cropped_image, image_affine, image_nii.header, image_out, is_mask=False)
            if mask_out is not None and mask_nii is not None:
                mask_affine = _cropped_affine(mask_nii.affine, bounds)
                _save_nifti(nib, cropped_mask, mask_affine, mask_nii.header, mask_out, is_mask=True)

            volume_before = int((mask > 0).sum()) if mask is not None else 0
            volume_after = int(cropped_mask.sum()) if cropped_mask is not None else 0
            bbox_text = "|".join(f"{lo}:{hi}" for lo, hi in bounds)
            processed_rows.append(
                {
                    "case_id": case_id,
                    "split_group": row["split_group"],
                    "raw_image_path": str(image_path),
                    "raw_mask_path": str(mask_path) if mask_path else "",
                    "processed_image_path": str(image_out),
                    "processed_mask_path": str(mask_out) if mask_out else "",
                    "has_label": bool(mask_out),
                    "raw_shape": "x".join(str(v) for v in image.shape),
                    "processed_shape": "x".join(str(v) for v in cropped_image.shape),
                    "crop_bbox_vox": bbox_text,
                    "status": "processed",
                    "warnings": "",
                }
            )
            qc_rows.append(
                {
                    "case_id": case_id,
                    "split_group": row["split_group"],
                    "image_shape_match": True,
                    "processed_shape": "x".join(str(v) for v in cropped_image.shape),
                    "mask_unique_values": "0|1" if cropped_mask is not None else "",
                    "foreground_voxels_before": volume_before,
                    "foreground_voxels_after": volume_after,
                    "foreground_voxel_delta": volume_after - volume_before,
                    "nonzero_voxels_after": int(np.count_nonzero(cropped_image)),
                    "severity": "info" if volume_before == volume_after else "warning",
                }
            )
            append_jsonl(
                artifacts / "logs" / "preprocessing_log.jsonl",
                {
                    "case_id": case_id,
                    "split_group": row["split_group"],
                    "image_path": str(image_path),
                    "mask_path": str(mask_path) if mask_path else "",
                    "crop_bbox_vox": bbox_text,
                    "clip_low": norm_stats["clip_low"],
                    "clip_high": norm_stats["clip_high"],
                    "mean": norm_stats["mean"],
                    "std": norm_stats["std"],
                    "status": "processed",
                },
            )
            processed += 1
            if progress_every and processed % progress_every == 0:
                print(f"[preprocessing] processed={processed}", flush=True)
            if limit is not None and processed >= limit:
                break
        except Exception as exc:
            failures.append(
                {
                    "case_id": case_id,
                    "split_group": row["split_group"],
                    "raw_image_path": str(image_path),
                    "raw_mask_path": str(mask_path) if mask_path else "",
                    "processed_image_path": "",
                    "processed_mask_path": "",
                    "has_label": bool(mask_path),
                    "raw_shape": "",
                    "processed_shape": "",
                    "crop_bbox_vox": "",
                    "status": "failed",
                    "warnings": str(exc),
                }
            )
            append_jsonl(
                artifacts / "logs" / "preprocessing_log.jsonl",
                {
                    "case_id": case_id,
                    "split_group": row["split_group"],
                    "status": "failed",
                    "error": str(exc),
                },
            )

    all_rows = processed_rows + failures
    fieldnames = (
        "case_id",
        "split_group",
        "raw_image_path",
        "raw_mask_path",
        "processed_image_path",
        "processed_mask_path",
        "has_label",
        "raw_shape",
        "processed_shape",
        "crop_bbox_vox",
        "status",
        "warnings",
    )
    write_csv(artifacts / "manifests" / "manifest_preprocessed.csv", all_rows, fieldnames)
    write_csv(
        artifacts / "qc" / "post_preprocessing_qc.csv",
        qc_rows,
        (
            "case_id",
            "split_group",
            "image_shape_match",
            "processed_shape",
            "mask_unique_values",
            "foreground_voxels_before",
            "foreground_voxels_after",
            "foreground_voxel_delta",
            "nonzero_voxels_after",
            "severity",
        ),
    )
    write_markdown_report(
        artifacts / "reports" / "preprocessing_report.md",
        "Preprocessing Report",
        [
            "## Summary",
            f"- Processed cases: {len(processed_rows)}",
            f"- Failed cases: {len(failures)}",
            f"- Clipping percentiles: {clip_percentiles}",
            f"- Normalization region: {region_mode}",
            f"- Crop margin voxels: {crop_margin}",
            "",
            "## Notes",
            "- Reorientation uses nibabel canonical orientation.",
            "- Resampling is currently disabled by design and must be implemented deliberately before enabling it in config.",
        ],
    )
    write_markdown_report(
        artifacts / "reports" / "preprocessing_delta_report.md",
        "Preprocessing Delta Report",
        [
            "## Delta checks",
            f"- Post-QC rows: {len(qc_rows)}",
            "- Any non-zero foreground voxel delta should be reviewed before using the case for training.",
        ],
    )
    return {"processed": len(processed_rows), "failed": len(failures)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing on raw manifest entries.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on processed cases.")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(
        limit=args.limit,
        progress_every=args.progress_every,
        project_root_path=args.project_root,
        manifest_path=args.manifest,
        config_path=args.config,
        data_root=args.data_root,
        output_root=args.output_root,
        artifact_root=args.artifact_root,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"Preprocessing complete: processed={result['processed']} failed={result['failed']}")


if __name__ == "__main__":
    main()
