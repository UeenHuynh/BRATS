"""Voxel-wise and surface metrics for binary 3D segmentation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray


def _surface_distances_mm(
    prediction: NDArray[np.bool_], target: NDArray[np.bool_], spacing_xyz: Sequence[float]
) -> NDArray[np.float64]:
    """Return distances from prediction surface to target surface in millimetres."""
    prediction_image = sitk.GetImageFromArray(prediction.astype(np.uint8, copy=False))
    target_image = sitk.GetImageFromArray(target.astype(np.uint8, copy=False))
    prediction_image.SetSpacing(tuple(float(value) for value in spacing_xyz))
    target_image.SetSpacing(tuple(float(value) for value in spacing_xyz))

    prediction_surface = sitk.GetArrayFromImage(sitk.LabelContour(prediction_image)).astype(bool)
    target_surface = sitk.GetArrayFromImage(sitk.LabelContour(target_image)).astype(bool)
    if not prediction_surface.any() or not target_surface.any():
        return np.empty(0, dtype=np.float64)

    distance_to_target = np.abs(
        sitk.GetArrayFromImage(
            sitk.SignedMaurerDistanceMap(target_image, squaredDistance=False, useImageSpacing=True)
        )
    )
    distance_to_prediction = np.abs(
        sitk.GetArrayFromImage(
            sitk.SignedMaurerDistanceMap(prediction_image, squaredDistance=False, useImageSpacing=True)
        )
    )
    return np.concatenate(
        [distance_to_target[prediction_surface], distance_to_prediction[target_surface]]
    ).astype(np.float64, copy=False)


def compute_binary_metrics(
    prediction: NDArray[np.bool_] | NDArray[np.uint8],
    target: NDArray[np.bool_] | NDArray[np.uint8],
    spacing_xyz: Sequence[float],
) -> dict[str, float]:
    """Compute Dice, HD95, sensitivity, and precision for one 3D case."""
    prediction_bool = np.asarray(prediction, dtype=bool)
    target_bool = np.asarray(target, dtype=bool)
    if prediction_bool.shape != target_bool.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction_bool.shape} vs {target_bool.shape}")
    if len(spacing_xyz) != 3 or any(float(value) <= 0 for value in spacing_xyz):
        raise ValueError(f"Invalid spacing: {spacing_xyz}")

    prediction_count = int(prediction_bool.sum())
    target_count = int(target_bool.sum())
    true_positive = int(np.logical_and(prediction_bool, target_bool).sum())
    dice_denominator = prediction_count + target_count
    dice = 1.0 if dice_denominator == 0 else (2.0 * true_positive / dice_denominator)
    sensitivity = 1.0 if target_count == 0 and prediction_count == 0 else (
        true_positive / target_count if target_count else 0.0
    )
    precision = 1.0 if target_count == 0 and prediction_count == 0 else (
        true_positive / prediction_count if prediction_count else 0.0
    )

    if prediction_count == 0 and target_count == 0:
        hd95 = 0.0
    elif prediction_count == 0 or target_count == 0:
        hd95 = float("inf")
    else:
        distances = _surface_distances_mm(prediction_bool, target_bool, spacing_xyz)
        hd95 = float(np.percentile(distances, 95)) if distances.size else float("inf")
    return {
        "dice": float(dice),
        "hd95_mm": hd95,
        "sensitivity": float(sensitivity),
        "precision": float(precision),
    }


def summarize_metrics(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Summarize per-case metrics without hiding failed/empty predictions."""
    names = ("dice", "hd95_mm", "sensitivity", "precision")
    summary: dict[str, dict[str, float]] = {}
    for name in names:
        values = np.asarray([float(record[name]) for record in records], dtype=np.float64)
        summary[name] = {
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "median": float(np.median(values)) if values.size else float("nan"),
            "std": float(np.std(values)) if values.size else float("nan"),
        }
    return summary
