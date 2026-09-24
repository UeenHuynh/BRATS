"""Deterministic full-volume inference and evaluation for 3D models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
from torch import nn

from brats.training.dataset import DatasetSpec, load_volume

from .metrics import compute_binary_metrics, summarize_metrics


def _make_patches(
    image: np.ndarray, depth: int, height: int, width: int
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int, int, int]]]:
    patches: list[np.ndarray] = []
    locations: list[tuple[int, int, int, int, int, int]] = []
    d, h, w = image.shape
    for d0 in range(0, d, depth):
        for h0 in range(0, h, height):
            for w0 in range(0, w, width):
                gd = min(depth, d - d0)
                gh = min(height, h - h0)
                gw = min(width, w - w0)
                patch = np.zeros((depth, height, width), dtype=np.float32)
                patch[:gd, :gh, :gw] = image[d0 : d0 + gd, h0 : h0 + gh, w0 : w0 + gw]
                patches.append(patch)
                locations.append((d0, h0, w0, gd, gh, gw))
    return patches, locations


def _predict_case(
    model: nn.Module,
    image: np.ndarray,
    depth: int,
    patch_size: tuple[int, int],
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
    height, width = patch_size
    patches, locations = _make_patches(image, depth, height, width)
    probabilities = np.zeros(image.shape, dtype=np.float32)
    counts = np.zeros(image.shape, dtype=np.float32)
    for start in range(0, len(patches), batch_size):
        batch = torch.from_numpy(np.stack(patches[start : start + batch_size], axis=0)[:, None])
        batch = batch.to(device, non_blocking=True)
        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch)
        else:
            logits = model(batch)
        batch_probabilities = torch.sigmoid(logits).detach().float().cpu().numpy()[:, 0]
        for probability, (d0, h0, w0, gd, gh, gw) in zip(
            batch_probabilities, locations[start : start + batch_size], strict=True
        ):
            probabilities[d0 : d0 + gd, h0 : h0 + gh, w0 : w0 + gw] += probability[:gd, :gh, :gw]
            counts[d0 : d0 + gd, h0 : h0 + gh, w0 : w0 + gw] += 1.0
    return probabilities / np.maximum(counts, 1.0)


def _predict_case_2d(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
    """Run slice-wise inference, padding each slice to U-Net-compatible dimensions."""
    height, width = image.shape[1:]
    padded_height = int(np.ceil(height / 16.0) * 16)
    padded_width = int(np.ceil(width / 16.0) * 16)
    padded = np.zeros((image.shape[0], padded_height, padded_width), dtype=np.float32)
    padded[:, :height, :width] = image
    probabilities = np.zeros_like(padded, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, padded.shape[0], batch_size):
            batch = torch.from_numpy(padded[start : start + batch_size, None]).to(device, non_blocking=True)
            if amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(batch)
            else:
                logits = model(batch)
            probabilities[start : start + batch.shape[0]] = (
                torch.sigmoid(logits).detach().float().cpu().numpy()[:, 0]
            )
    return probabilities[:, :height, :width]


def evaluate_model_2d(
    model: nn.Module,
    dataset_root: Path,
    case_ids: Sequence[str],
    split: str,
    device: torch.device,
    batch_size: int = 8,
    threshold: float = 0.5,
    amp: bool = True,
) -> dict[str, Any]:
    """Run slice-wise inference and return aggregate 3D metrics per reconstructed volume."""
    spec = DatasetSpec(dataset_root=dataset_root, split=split, mode="2d", augment=False, case_ids=tuple(case_ids))
    model.eval()
    records: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids, start=1):
        image, label = load_volume(case_id, spec)
        image_path = dataset_root / split / "images" / f"{case_id}.nii.gz"
        spacing_xyz = sitk.ReadImage(str(image_path)).GetSpacing()
        probability = _predict_case_2d(model, image, device, batch_size, amp)
        metrics = compute_binary_metrics(probability >= threshold, label > 0, spacing_xyz)
        records.append({"case_id": case_id, **metrics})
        print(
            f"  Validation {index}/{len(case_ids)} | {case_id} | "
            f"Dice={metrics['dice']:.4f} | HD95={metrics['hd95_mm']:.2f} mm",
            flush=True,
        )
    return {"cases": records, "summary": summarize_metrics(records)}


def evaluate_model_3d(
    model: nn.Module,
    dataset_root: Path,
    case_ids: Sequence[str],
    split: str,
    depth: int,
    patch_size: tuple[int, int],
    device: torch.device,
    batch_size: int = 2,
    threshold: float = 0.5,
    amp: bool = True,
) -> dict[str, Any]:
    """Run tiled inference and return per-case plus aggregate 3D metrics."""
    spec = DatasetSpec(
        dataset_root=dataset_root,
        split=split,
        depth=depth,
        patch_size=patch_size,
        mode="3d",
        augment=False,
        case_ids=tuple(case_ids),
    )
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, case_id in enumerate(case_ids, start=1):
            image, label = load_volume(case_id, spec)
            image_path = dataset_root / split / "images" / f"{case_id}.nii.gz"
            spacing_xyz = sitk.ReadImage(str(image_path)).GetSpacing()
            probability = _predict_case(
                model=model,
                image=image,
                depth=depth,
                patch_size=patch_size,
                device=device,
                batch_size=batch_size,
                amp=amp,
            )
            metrics = compute_binary_metrics(probability >= threshold, label > 0, spacing_xyz)
            records.append({"case_id": case_id, **metrics})
            print(
                f"  Validation {index}/{len(case_ids)} | {case_id} | "
                f"Dice={metrics['dice']:.4f} | HD95={metrics['hd95_mm']:.2f} mm",
                flush=True,
            )
    return {"cases": records, "summary": summarize_metrics(records)}
