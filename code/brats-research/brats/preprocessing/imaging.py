from __future__ import annotations

import math
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from SimpleITK import Image


def import_sitk() -> Any:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK is required for image processing") from exc
    return sitk


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _image_size(image: Image) -> tuple[int, ...]:
    return cast(tuple[int, ...], image.GetSize())


def _image_vector(value: Any) -> tuple[float, ...]:
    return cast(tuple[float, ...], value)


def geometry(image: Image) -> dict[str, Any]:
    return {
        "size": list(_image_size(image)),
        "spacing": list(_image_vector(image.GetSpacing())),
        "origin": list(_image_vector(image.GetOrigin())),
        "direction": list(_image_vector(image.GetDirection())),
    }


def geometry_matches(left: Image, right: Image, tolerance: float = 1e-5) -> bool:
    if _image_size(left) != _image_size(right):
        return False
    vector_pairs = (
        (_image_vector(left.GetSpacing()), _image_vector(right.GetSpacing())),
        (_image_vector(left.GetOrigin()), _image_vector(right.GetOrigin())),
        (_image_vector(left.GetDirection()), _image_vector(right.GetDirection())),
    )
    return all(
        abs(a - b) <= tolerance
        for left_values, right_values in vector_pairs
        for a, b in zip(left_values, right_values, strict=True)
    )


def voxel_volume(image: Image) -> float:
    return float(np.prod(np.asarray(image.GetSpacing(), dtype=float)))


def binary_volume(image: Image) -> float:
    sitk = import_sitk()
    return float(np.count_nonzero(sitk.GetArrayViewFromImage(image)) * voxel_volume(image))


def label_values(image: Image) -> list[int]:
    sitk = import_sitk()
    values = cast(NDArray[np.integer[Any]], np.unique(sitk.GetArrayViewFromImage(image)))
    return [int(value) for value in values]


def connected_components(image: Image) -> int:
    sitk = import_sitk()
    connected = sitk.ConnectedComponent(sitk.Cast(image > 0, sitk.sitkUInt8))
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(connected)
    return int(len(statistics.GetLabels()))


def largest_component(mask: Image) -> Image:
    sitk = import_sitk()
    connected = sitk.ConnectedComponent(sitk.Cast(mask > 0, sitk.sitkUInt8))
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(connected)
    labels = list(statistics.GetLabels())
    if not labels:
        raise ValueError("Foreground mask is empty")
    largest = max(labels, key=statistics.GetPhysicalSize)
    return sitk.Cast(connected == largest, sitk.sitkUInt8)


def otsu_mask(image: Image) -> Image:
    sitk = import_sitk()
    mask = largest_component(sitk.OtsuThreshold(image, 0, 1, 200))
    mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 2])
    return sitk.Cast(sitk.BinaryFillhole(mask), sitk.sitkUInt8)


def get_brain_mask(
    image: Image,
    image_path: Path,
    configured_path: Path | None,
    config: Mapping[str, Any],
    cache: Path,
) -> tuple[Image, str, dict[str, Any]]:
    sitk = import_sitk()
    source = str(config.get("source", "otsu")).lower()
    if source == "precomputed":
        if configured_path is None or not configured_path.exists():
            raise FileNotFoundError("Precomputed brain mask is missing")
        return sitk.ReadImage(str(configured_path), sitk.sitkUInt8), "precomputed", {}
    if source == "otsu":
        return otsu_mask(image), "otsu", {}
    if source != "totalsegmentator":
        raise ValueError(f"Unknown brain mask source: {source}")
    settings = config.get("totalsegmentator", {})
    output = cache / "totalsegmentator"
    output.mkdir(parents=True, exist_ok=True)
    brain_path = output / "brain.nii.gz"
    command = [
        str(settings.get("executable", "TotalSegmentator")),
        "-i",
        str(image_path),
        "-o",
        str(output),
        "--task",
        str(settings.get("task", "total_mr")),
        "--roi_subset",
        str(settings.get("roi", "brain")),
    ]
    if settings.get("device"):
        command.extend(["--device", str(settings["device"])])
    if bool(settings.get("fast", False)):
        command.append("--fast")
    command.extend(str(value) for value in settings.get("extra_args", []))
    if not brain_path.exists() or bool(settings.get("overwrite", False)):
        environment = os.environ.copy()
        environment.setdefault("TOTALSEG_HOME_DIR", str(cache.parent / "totalsegmentator-home"))
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment)
        (output / "command.log").write_text(process.stdout or "", encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"TotalSegmentator failed with exit code {process.returncode}")
    if not brain_path.exists():
        candidates = list(output.glob("**/brain.nii.gz"))
        if candidates:
            shutil.copy2(candidates[0], brain_path)
    if not brain_path.exists():
        raise RuntimeError("TotalSegmentator did not produce brain.nii.gz")
    return sitk.ReadImage(str(brain_path), sitk.sitkUInt8), "totalsegmentator", {}


def n4_correct(image: Image, mask: Image, config: Mapping[str, Any]) -> tuple[Image, Image, dict[str, Any]]:
    sitk = import_sitk()
    image = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    shrink = int(config.get("shrink_factor", 2))
    small_image = sitk.Shrink(image, [shrink] * 3) if shrink > 1 else image
    small_mask = sitk.Shrink(mask, [shrink] * 3) if shrink > 1 else mask
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    iterations = cast(Sequence[int], config.get("iterations", [50, 50, 30, 20]))
    corrector.SetMaximumNumberOfIterations(list(iterations))
    if config.get("convergence_threshold") is not None:
        corrector.SetConvergenceThreshold(float(config["convergence_threshold"]))
    corrector.Execute(small_image, small_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(image)
    corrected = sitk.Cast(image / sitk.Exp(log_bias), sitk.sitkFloat32)
    array = cast(NDArray[np.float32], sitk.GetArrayViewFromImage(log_bias))
    summary = {
        "elapsed_iterations": int(corrector.GetElapsedIterations()),
        "convergence_measurement": float(corrector.GetCurrentConvergenceMeasurement()),
        "log_bias_min": float(np.min(array)),
        "log_bias_max": float(np.max(array)),
        "log_bias_mean": float(np.mean(array)),
        "log_bias_std": float(np.std(array)),
    }
    return corrected, log_bias, summary


def create_source_support(image: Image) -> Image:
    sitk = import_sitk()
    return sitk.Cast(image != 0, sitk.sitkUInt8)


def selector(
    name: str,
    image_array: NDArray[Any],
    brain_array: NDArray[Any],
    head_array: NDArray[Any],
    support_array: NDArray[Any],
) -> NDArray[np.bool_]:
    support = support_array > 0
    selectors = {
        "brain": brain_array > 0,
        "head": head_array > 0,
        "head_nonzero": (head_array > 0) & support,
        "nonzero": support,
        "all": np.ones_like(image_array, dtype=bool),
    }
    if name not in selectors:
        raise ValueError(f"Unknown mask selector: {name}")
    return selectors[name]


def transform_intensity(
    image: Image,
    brain: Image,
    head: Image,
    source_support: Image,
    config: Mapping[str, Any],
) -> tuple[Image, dict[str, Any]]:
    sitk = import_sitk()
    if not geometry_matches(image, source_support):
        raise ValueError("Image and source support geometry differ")
    array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Image contains non-finite intensities")
    brain_array = sitk.GetArrayViewFromImage(brain)
    head_array = sitk.GetArrayViewFromImage(head)
    support_array = sitk.GetArrayViewFromImage(source_support)
    metrics: dict[str, Any] = {}
    clipping = config["clipping"]
    if bool(clipping.get("enabled", False)):
        stats = selector(str(clipping.get("stats_mask", "brain")), array, brain_array, head_array, support_array)
        values = array[stats]
        if values.size < int(clipping.get("minimum_voxels", 100)):
            raise ValueError("Too few voxels for clipping statistics")
        percentiles = cast(
            NDArray[np.float64],
            np.percentile(values, [float(clipping["lower_percentile"]), float(clipping["upper_percentile"])]),
        )
        low, high = float(percentiles[0]), float(percentiles[1])
        if not high > low:
            raise ValueError("Invalid clipping range")
        apply = selector(str(clipping.get("apply_mask", "nonzero")), array, brain_array, head_array, support_array)
        below = np.count_nonzero(array[apply] < low)
        above = np.count_nonzero(array[apply] > high)
        count = np.count_nonzero(apply)
        array = array.copy()
        array[apply] = np.clip(array[apply], low, high)
        metrics.update(
            {
                "clip_low_value": float(low),
                "clip_high_value": float(high),
                "clip_low_fraction": below / count,
                "clip_high_fraction": above / count,
            }
        )
    normalization = config["normalization"]
    method = str(normalization.get("method", "zscore")).lower()
    if method == "zscore":
        stats = selector(str(normalization.get("stats_mask", "brain")), array, brain_array, head_array, support_array)
        values = array[stats]
        if values.size < int(normalization.get("minimum_voxels", 100)):
            raise ValueError("Too few voxels for normalization statistics")
        mean, standard_deviation = float(values.mean()), float(values.std())
        if not math.isfinite(standard_deviation) or standard_deviation <= 1e-8:
            raise ValueError("Invalid normalization variance")
        apply = selector(str(normalization.get("apply_mask", "nonzero")), array, brain_array, head_array, support_array)
        output = np.zeros_like(array)
        output[apply] = (array[apply] - mean) / standard_deviation
        array = output
        metrics.update({"zscore_mean": mean, "zscore_std": standard_deviation})
    elif method not in {"none", "off"}:
        raise ValueError(f"Unknown normalization method: {method}")
    output_image = sitk.GetImageFromArray(array)
    output_image.CopyInformation(image)
    return sitk.Cast(output_image, sitk.sitkFloat32), metrics


def support_leakage(image: Image, source_support: Image) -> tuple[int, float]:
    sitk = import_sitk()
    if not geometry_matches(image, source_support):
        raise ValueError("Image and source support geometry differ")
    image_array = sitk.GetArrayViewFromImage(image)
    outside_support = sitk.GetArrayViewFromImage(source_support) == 0
    outside_values = image_array[outside_support]
    count = int(np.count_nonzero(outside_values))
    maximum = float(np.max(np.abs(outside_values))) if outside_values.size else 0.0
    return count, maximum


def apply_source_support(image: Image, source_support: Image) -> Image:
    sitk = import_sitk()
    if not geometry_matches(image, source_support):
        raise ValueError("Image and source support geometry differ")
    return sitk.Mask(
        sitk.Cast(image, sitk.sitkFloat32),
        sitk.Cast(source_support > 0, sitk.sitkUInt8),
        outsideValue=0,
    )


def crop_box(mask: Image, margin_mm: float) -> tuple[list[int], list[int]]:
    sitk = import_sitk()
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(sitk.Cast(mask > 0, sitk.sitkUInt8))
    if not statistics.HasLabel(1):
        raise ValueError("Crop mask is empty")
    bounding_box = cast(tuple[int, ...], statistics.GetBoundingBox(1))
    index = list(bounding_box[:3])
    size = list(bounding_box[3:])
    full_size = _image_size(mask)
    margin = [int(math.ceil(margin_mm / spacing)) for spacing in _image_vector(mask.GetSpacing())]
    starts = [max(0, index[axis] - margin[axis]) for axis in range(3)]
    ends = [min(full_size[axis], index[axis] + size[axis] + margin[axis]) for axis in range(3)]
    return starts, [ends[axis] - starts[axis] for axis in range(3)]


def rectangular_mask(mask: Image, margin_mm: float) -> Image:
    sitk = import_sitk()
    index, size = crop_box(mask, margin_mm)
    rectangle = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
    rectangle.CopyInformation(mask)
    region = sitk.Image([int(value) for value in size], sitk.sitkUInt8) + 1
    return sitk.Paste(rectangle, region, size, [0, 0, 0], index)


def crop_image(image: Image, index: Sequence[int], size: Sequence[int]) -> Image:
    sitk = import_sitk()
    return sitk.RegionOfInterest(image, [int(value) for value in size], [int(value) for value in index])


def resample(image: Image, spacing: Sequence[float], is_label: bool) -> Image:
    sitk = import_sitk()
    old_size = np.asarray(image.GetSize(), dtype=float)
    old_spacing = np.asarray(image.GetSpacing(), dtype=float)
    target = np.asarray(spacing, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all() or np.any(target <= 0):
        raise ValueError("Target spacing must contain three positive finite values")
    new_size = np.maximum(1, np.rint((old_size - 1) * old_spacing / target).astype(int) + 1)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    old_center_offset = direction @ ((old_size - 1) * old_spacing / 2)
    new_center_offset = direction @ ((new_size - 1) * target / 2)
    new_origin = np.asarray(image.GetOrigin(), dtype=float) + old_center_offset - new_center_offset
    return sitk.Resample(
        image,
        new_size.tolist(),
        sitk.Transform(),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
        new_origin.tolist(),
        target.tolist(),
        image.GetDirection(),
        0,
        image.GetPixelID(),
    )


def target_spacing(config: Mapping[str, Any], cohort_median: Sequence[float] | None) -> list[float] | None:
    if not bool(config.get("enabled", False)):
        return None
    value = config.get("target_spacing_mm", "native")
    if isinstance(value, str):
        if value.lower() == "native":
            return None
        if value.lower() == "training_median":
            if cohort_median is None:
                raise ValueError("Training median spacing is unavailable")
            return [float(item) for item in cohort_median]
        raise ValueError(f"Unknown spacing mode: {value}")
    if len(value) != 3 or any(float(item) <= 0 for item in value):
        raise ValueError("Target spacing must contain three positive values")
    return [float(item) for item in value]


def read_spacing(path: Path) -> list[float]:
    sitk = import_sitk()
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return [float(value) for value in reader.GetSpacing()]


def write_image_verified(image: Image, path: Path) -> str:
    sitk = import_sitk()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.nii.gz")
    sitk.WriteImage(image, str(temporary), True)
    reread = sitk.ReadImage(str(temporary))
    if not geometry_matches(image, reread):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Write verification failed: {path}")
    temporary.replace(path)
    return sha256_file(path)
