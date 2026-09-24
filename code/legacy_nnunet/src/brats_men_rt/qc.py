from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from brats_men_rt.nifti import NiftiHeader, load_array, orientation_code, read_header


def image_summary(path: str | Path) -> dict[str, object]:
    header = read_header(path)
    array = load_array(path, header=header, apply_scaling=True).astype(np.float32, copy=False)
    finite = np.isfinite(array)
    nonzero = array[array != 0]
    stats = {
        "shape": "x".join(str(v) for v in header.shape[:3]),
        "spacing_x": header.spacing[0],
        "spacing_y": header.spacing[1],
        "spacing_z": header.spacing[2],
        "affine": np.array2string(header.affine, precision=3, separator=","),
        "orientation_code": orientation_code(header.affine),
        "dtype": header.dtype,
        "intensity_min": float(np.nanmin(array)),
        "intensity_max": float(np.nanmax(array)),
        "intensity_mean": float(np.nanmean(array)),
        "intensity_std": float(np.nanstd(array)),
        "p0_5": float(np.nanpercentile(array, 0.5)),
        "p1": float(np.nanpercentile(array, 1)),
        "p5": float(np.nanpercentile(array, 5)),
        "p50": float(np.nanpercentile(array, 50)),
        "p95": float(np.nanpercentile(array, 95)),
        "p99": float(np.nanpercentile(array, 99)),
        "p99_5": float(np.nanpercentile(array, 99.5)),
        "nonzero_mean": float(np.nanmean(nonzero)) if nonzero.size else 0.0,
        "nonzero_std": float(np.nanstd(nonzero)) if nonzero.size else 0.0,
        "zero_voxel_count": int(np.size(array) - nonzero.size),
        "zero_voxel_ratio": float((np.size(array) - nonzero.size) / np.size(array)),
        "nan_count": int(np.size(array) - np.count_nonzero(finite)),
        "inf_count": int(np.count_nonzero(np.isinf(array))),
    }
    return stats


def mask_summary(path: str | Path, spacing: tuple[float, float, float] | None = None) -> dict[str, object]:
    header = read_header(path)
    mask = load_array(path, header=header, apply_scaling=False)
    spacing_values = spacing or header.spacing
    mask_bool = mask > 0
    total_voxels = int(mask.size)
    foreground_voxels = int(mask_bool.sum())
    background_voxels = total_voxels - foreground_voxels
    voxel_volume_mm3 = float(spacing_values[0] * spacing_values[1] * spacing_values[2])
    volume_mm3 = float(foreground_voxels * voxel_volume_mm3)
    volume_ml = float(volume_mm3 / 1000.0)
    unique_values = sorted(np.unique(mask).astype(float).tolist())
    bbox_dims_vox, bbox_dims_mm, centroid = _bbox_and_centroid(mask_bool, spacing_values)
    components, component_sizes = _component_stats(mask_bool)
    boundary_voxels = int(mask_bool.sum() - ndimage.binary_erosion(mask_bool, border_value=0).sum()) if foreground_voxels else 0
    imbalance_ratio = float(background_voxels / foreground_voxels) if foreground_voxels else float("inf")
    return {
        "mask_shape": "x".join(str(v) for v in header.shape[:3]),
        "mask_dtype": header.dtype,
        "mask_unique_values": "|".join(str(v) for v in unique_values),
        "total_voxels": total_voxels,
        "foreground_voxels": foreground_voxels,
        "background_voxels": background_voxels,
        "foreground_ratio": float(foreground_voxels / total_voxels) if total_voxels else 0.0,
        "imbalance_ratio": imbalance_ratio,
        "volume_mm3": volume_mm3,
        "volume_ml": volume_ml,
        "connected_components": components,
        "largest_component_ratio": float(max(component_sizes) / foreground_voxels) if component_sizes and foreground_voxels else 0.0,
        "small_component_count": int(sum(1 for size in component_sizes if size < 10)),
        "bbox_dims_vox": "x".join(str(v) for v in bbox_dims_vox),
        "bbox_dims_mm": "x".join(f"{v:.3f}" for v in bbox_dims_mm),
        "centroid_vox": ",".join(f"{v:.3f}" for v in centroid[0]),
        "centroid_mm": ",".join(f"{v:.3f}" for v in centroid[1]),
        "boundary_voxels": boundary_voxels,
        "boundary_to_volume_ratio": float(boundary_voxels / foreground_voxels) if foreground_voxels else 0.0,
        "empty_mask_flag": foreground_voxels == 0,
    }


def compare_image_mask(image_header: NiftiHeader, mask_header: NiftiHeader) -> dict[str, object]:
    shape_match = tuple(image_header.shape[:3]) == tuple(mask_header.shape[:3])
    spacing_delta = np.abs(np.array(image_header.spacing) - np.array(mask_header.spacing))
    affine_delta = np.abs(image_header.affine - mask_header.affine)
    return {
        "shape_match": shape_match,
        "spacing_match": bool(np.all(spacing_delta < 1e-5)),
        "spacing_delta_max": float(spacing_delta.max()),
        "affine_delta_max": float(affine_delta.max()),
        "image_orientation": orientation_code(image_header.affine),
        "mask_orientation": orientation_code(mask_header.affine),
    }


def _bbox_and_centroid(mask_bool: np.ndarray, spacing: tuple[float, float, float]) -> tuple[tuple[int, int, int], tuple[float, float, float], tuple[np.ndarray, np.ndarray]]:
    if not mask_bool.any():
        zeros = (0, 0, 0)
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        return zeros, (0.0, 0.0, 0.0), (origin, origin)
    coords = np.argwhere(mask_bool)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    dims_vox = tuple((maxs - mins + 1).astype(int).tolist())
    dims_mm = tuple(float(dims_vox[index] * spacing[index]) for index in range(3))
    centroid_vox = coords.mean(axis=0)
    centroid_mm = centroid_vox * np.array(spacing)
    return dims_vox, dims_mm, (centroid_vox, centroid_mm)


def _component_stats(mask_bool: np.ndarray) -> tuple[int, list[int]]:
    if not mask_bool.any():
        return 0, []
    labeled, components = ndimage.label(mask_bool)
    sizes = ndimage.sum(mask_bool, labeled, index=np.arange(1, components + 1))
    return int(components), [int(size) for size in np.atleast_1d(sizes)]
