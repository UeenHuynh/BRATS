from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage


@dataclass(slots=True)
class SegmentationMetrics:
    dice: float
    precision: float
    recall: float
    hd95_mm: float
    surface_dice_mm: float
    fp_components: int
    fn_components: int
    pred_foreground_voxels: int
    gt_foreground_voxels: int


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    denom = pred_sum + gt_sum
    if denom == 0:
        return 0.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 1.0
    return precision, recall


def false_component_counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int]:
    structure = ndimage.generate_binary_structure(3, 1)
    pred_labels, pred_count = ndimage.label(pred, structure=structure)
    gt_labels, gt_count = ndimage.label(gt, structure=structure)

    fp_components = 0
    for index in range(1, pred_count + 1):
        component = pred_labels == index
        if not np.logical_and(component, gt).any():
            fp_components += 1

    fn_components = 0
    for index in range(1, gt_count + 1):
        component = gt_labels == index
        if not np.logical_and(component, pred).any():
            fn_components += 1

    return fp_components, fn_components


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def _surface_distances(source_surface: np.ndarray, target_mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    if not source_surface.any():
        return np.array([], dtype=np.float64)
    if not target_mask.any():
        return np.full(int(source_surface.sum()), np.inf, dtype=np.float64)
    distance_map = ndimage.distance_transform_edt(np.logical_not(target_mask), sampling=spacing)
    return distance_map[source_surface]


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]) -> float:
    if not pred.any() and not gt.any():
        return 0.0
    pred_surface = surface_voxels(pred)
    gt_surface = surface_voxels(gt)
    d_pred_to_gt = _surface_distances(pred_surface, gt, spacing)
    d_gt_to_pred = _surface_distances(gt_surface, pred, spacing)
    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    if all_distances.size == 0:
        return 0.0
    return float(np.percentile(all_distances, 95))


def surface_dice(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float], tolerance_mm: float) -> float:
    if not pred.any() and not gt.any():
        return 1.0
    pred_surface = surface_voxels(pred)
    gt_surface = surface_voxels(gt)
    d_pred_to_gt = _surface_distances(pred_surface, gt, spacing)
    d_gt_to_pred = _surface_distances(gt_surface, pred, spacing)
    pred_hits = int(np.count_nonzero(d_pred_to_gt <= tolerance_mm))
    gt_hits = int(np.count_nonzero(d_gt_to_pred <= tolerance_mm))
    denom = int(pred_surface.sum() + gt_surface.sum())
    if denom == 0:
        return 0.0
    return float((pred_hits + gt_hits) / denom)


def evaluate_case(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float], tolerance_mm: float) -> SegmentationMetrics:
    pred_bool = pred.astype(bool, copy=False)
    gt_bool = gt.astype(bool, copy=False)
    precision, recall = precision_recall(pred_bool, gt_bool)
    fp_components, fn_components = false_component_counts(pred_bool, gt_bool)
    return SegmentationMetrics(
        dice=dice_coefficient(pred_bool, gt_bool),
        precision=precision,
        recall=recall,
        hd95_mm=hd95(pred_bool, gt_bool, spacing),
        surface_dice_mm=surface_dice(pred_bool, gt_bool, spacing, tolerance_mm),
        fp_components=fp_components,
        fn_components=fn_components,
        pred_foreground_voxels=int(pred_bool.sum()),
        gt_foreground_voxels=int(gt_bool.sum()),
    )


def find_prediction_path(prediction_dir: Path, case_id: str) -> Path | None:
    for suffix in (".nii.gz", ".nii"):
        for candidate in (
            prediction_dir / f"{case_id}_pred{suffix}",
            prediction_dir / f"{case_id}{suffix}",
            prediction_dir / f"{case_id}_gtv{suffix}",
        ):
            if candidate.exists():
                return candidate
    return None
