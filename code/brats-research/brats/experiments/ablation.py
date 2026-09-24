"""Matched five-arm, five-fold preprocessing ablation for BraTS-MEN-RT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
import yaml
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from brats.evaluation.metrics import compute_binary_metrics
from brats.training.losses import DiceCELoss
from brats.training.models import UNet3D, model_parameter_count


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Ablation config must be a YAML mapping")
    return value


def _resolve(value: str, config_path: Path) -> Path:
    expanded = Path(os.path.expandvars(value)).expanduser()
    return expanded if expanded.is_absolute() else (config_path.parent / expanded).resolve()


def _read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate case ID in {path}")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Arm:
    name: str
    image_template: str
    label_template: str

    def image_path(self, case_id: str) -> Path:
        return Path(self.image_template.format(case_id=case_id))

    def label_path(self, case_id: str) -> Path:
        return Path(self.label_template.format(case_id=case_id))


def _arms(config: Mapping[str, Any], config_path: Path) -> dict[str, Arm]:
    result: dict[str, Arm] = {}
    raw_arms = config.get("arms")
    if not isinstance(raw_arms, Mapping):
        raise ValueError("arms must be a mapping")
    for name, raw in raw_arms.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"arms.{name} must be a mapping")
        root = _resolve(str(raw["root"]), config_path)
        image = str(root / str(raw["image_template"]))
        label = str(root / str(raw["label_template"]))
        result[str(name)] = Arm(str(name), image, label)
    return result


def _raw_label_path(config: Mapping[str, Any], config_path: Path, case_id: str) -> Path:
    root = _resolve(str(config["raw_label_root"]), config_path)
    template = str(config.get("raw_label_template", "{case_id}/{case_id}_gtv.nii.gz"))
    return root / template.format(case_id=case_id)


def _fold_ids(config: Mapping[str, Any], config_path: Path, fold: int) -> tuple[list[str], list[str]]:
    fold_root = _resolve(str(config["fold_root"]), config_path)
    return _read_ids(fold_root / f"fold_{fold}_train.txt"), _read_ids(fold_root / f"fold_{fold}_val.txt")


def _inner_split(train_ids: Sequence[str], seed: int, fraction: float) -> tuple[list[str], list[str]]:
    ordered = sorted(train_ids)
    random.Random(seed).shuffle(ordered)
    count = max(1, round(len(ordered) * fraction))
    return sorted(ordered[count:]), sorted(ordered[:count])


def _read_pair(arm: Arm, case_id: str) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(arm.image_path(case_id)))).astype(np.float32, copy=False)
    label = sitk.GetArrayFromImage(sitk.ReadImage(str(arm.label_path(case_id)))) > 0
    if image.shape != label.shape:
        raise ValueError(f"{arm.name}/{case_id}: image-label shape mismatch {image.shape} != {label.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"{arm.name}/{case_id}: non-finite image")
    return image, label.astype(np.float32)


def _crop_around(
    image: NDArray[np.float32],
    label: NDArray[np.float32],
    shape: tuple[int, int, int],
    rng: np.random.Generator,
    foreground_probability: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    starts: list[int] = []
    foreground = np.argwhere(label > 0)
    use_foreground = foreground.size > 0 and rng.random() < foreground_probability
    center = foreground[int(rng.integers(len(foreground)))] if use_foreground else None
    for axis, wanted in enumerate(shape):
        size = image.shape[axis]
        if size <= wanted:
            starts.append(0)
        elif center is None:
            starts.append(int(rng.integers(0, size - wanted + 1)))
        else:
            jitter = int(rng.integers(-wanted // 4, wanted // 4 + 1))
            starts.append(max(0, min(size - wanted, int(center[axis]) - wanted // 2 + jitter)))
    slices = tuple(slice(start, min(start + wanted, size)) for start, wanted, size in zip(starts, shape, image.shape))
    image_crop = image[slices]
    label_crop = label[slices]
    if image_crop.shape != shape:
        image_out = np.zeros(shape, dtype=np.float32)
        label_out = np.zeros(shape, dtype=np.float32)
        target = tuple(slice(0, size) for size in image_crop.shape)
        image_out[target] = image_crop
        label_out[target] = label_crop
        return image_out, label_out
    return image_crop.copy(), label_crop.copy()


class AblationDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        arm: Arm,
        case_ids: Sequence[str],
        patch_shape: tuple[int, int, int],
        patches_per_case: int,
        seed: int,
        foreground_probability: float,
    ) -> None:
        self.arm = arm
        self.case_ids = tuple(case_ids)
        self.patch_shape = patch_shape
        self.patches_per_case = patches_per_case
        self.seed = seed
        self.foreground_probability = foreground_probability
        self.epoch = 0
        self._cache: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.case_ids) * self.patches_per_case

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        case_id = self.case_ids[index // self.patches_per_case]
        if case_id not in self._cache:
            self._cache[case_id] = _read_pair(self.arm, case_id)
            if len(self._cache) > 2:
                self._cache.pop(next(iter(self._cache)))
        image, label = self._cache[case_id]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        image, label = _crop_around(image, label, self.patch_shape, rng, self.foreground_probability)
        for axis in (0, 1, 2):
            if rng.random() < 0.5:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        return torch.from_numpy(image[None]).float(), torch.from_numpy(label[None]).float()


def _starts(size: int, patch: int, overlap: float) -> list[int]:
    if size <= patch:
        return [0]
    stride = max(1, int(round(patch * (1.0 - overlap))))
    values = list(range(0, size - patch + 1, stride))
    if values[-1] != size - patch:
        values.append(size - patch)
    return values


def _gaussian(shape: tuple[int, int, int]) -> NDArray[np.float32]:
    axes = [np.linspace(-1.0, 1.0, size, dtype=np.float32) for size in shape]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    weight = np.exp(-0.5 * (zz * zz + yy * yy + xx * xx) / (0.5**2))
    return np.maximum(weight, 1e-3).astype(np.float32)


def _predict(
    model: nn.Module,
    image: NDArray[np.float32],
    patch_shape: tuple[int, int, int],
    overlap: float,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> NDArray[np.float32]:
    padded_shape = tuple(max(size, patch) for size, patch in zip(image.shape, patch_shape))
    padded = np.zeros(padded_shape, dtype=np.float32)
    padded[tuple(slice(0, size) for size in image.shape)] = image
    locations = [
        (z, y, x)
        for z in _starts(padded_shape[0], patch_shape[0], overlap)
        for y in _starts(padded_shape[1], patch_shape[1], overlap)
        for x in _starts(padded_shape[2], patch_shape[2], overlap)
    ]
    weight = _gaussian(patch_shape)
    total = np.zeros(padded_shape, dtype=np.float32)
    weights = np.zeros(padded_shape, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(locations), batch_size):
            batch_locations = locations[offset : offset + batch_size]
            patches = np.stack([
                padded[z : z + patch_shape[0], y : y + patch_shape[1], x : x + patch_shape[2]]
                for z, y, x in batch_locations
            ])
            tensor = torch.from_numpy(patches[:, None]).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                probabilities = torch.sigmoid(model(tensor)).float().cpu().numpy()[:, 0]
            for probability, (z, y, x) in zip(probabilities, batch_locations, strict=True):
                region = np.s_[z : z + patch_shape[0], y : y + patch_shape[1], x : x + patch_shape[2]]
                total[region] += probability * weight
                weights[region] += weight
    result = total / np.maximum(weights, 1e-6)
    return result[tuple(slice(0, size) for size in image.shape)]


def _dice_only(probability: NDArray[np.float32], label: NDArray[np.float32], threshold: float) -> float:
    prediction = probability >= threshold
    target = label > 0
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else 2.0 * float(np.logical_and(prediction, target).sum()) / denominator


def _surface_dice(
    prediction: NDArray[np.bool_],
    target: NDArray[np.bool_],
    spacing: Sequence[float],
    tolerance: float,
) -> float:
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    prediction_image = sitk.GetImageFromArray(prediction.astype(np.uint8))
    target_image = sitk.GetImageFromArray(target.astype(np.uint8))
    prediction_image.SetSpacing(tuple(float(v) for v in spacing))
    target_image.SetSpacing(tuple(float(v) for v in spacing))
    prediction_surface = sitk.GetArrayFromImage(sitk.LabelContour(prediction_image)).astype(bool)
    target_surface = sitk.GetArrayFromImage(sitk.LabelContour(target_image)).astype(bool)
    to_target = np.abs(sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(target_image, False, False, True)))
    to_prediction = np.abs(
        sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(prediction_image, False, False, True))
    )
    hits = int((to_target[prediction_surface] <= tolerance).sum())
    hits += int((to_prediction[target_surface] <= tolerance).sum())
    denominator = int(prediction_surface.sum()) + int(target_surface.sum())
    return hits / denominator if denominator else 0.0


def _evaluate_case(
    model: nn.Module,
    arm: Arm,
    case_id: str,
    raw_label_path: Path,
    patch_shape: tuple[int, int, int],
    overlap: float,
    batch_size: int,
    threshold: float,
    tolerance_mm: float,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    image_itk = sitk.ReadImage(str(arm.image_path(case_id)))
    image = sitk.GetArrayFromImage(image_itk).astype(np.float32, copy=False)
    probability = _predict(model, image, patch_shape, overlap, batch_size, device, amp)
    prediction_itk = sitk.GetImageFromArray((probability >= threshold).astype(np.uint8))
    prediction_itk.CopyInformation(image_itk)
    raw_label_itk = sitk.ReadImage(str(raw_label_path))
    prediction_raw = sitk.Resample(
        prediction_itk,
        raw_label_itk,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    prediction = sitk.GetArrayFromImage(prediction_raw).astype(bool)
    target = sitk.GetArrayFromImage(raw_label_itk).astype(bool)
    spacing = raw_label_itk.GetSpacing()
    metrics = compute_binary_metrics(prediction, target, spacing)
    metrics["surface_dice_2mm"] = float(_surface_dice(prediction, target, spacing, tolerance_mm))
    metrics["prediction_voxels"] = int(prediction.sum())
    metrics["target_voxels"] = int(target.sum())
    return {"case_id": case_id, **metrics}


def _train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    grad_clip: float,
) -> tuple[float, float]:
    model.train()
    scaler = torch.amp.GradScaler(enabled=amp)
    losses: list[float] = []
    dices: list[float] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            prediction = torch.sigmoid(logits) >= 0.5
            target = labels > 0
            dims = tuple(range(1, prediction.ndim))
            numerator = 2.0 * (prediction & target).sum(dim=dims)
            denominator = prediction.sum(dim=dims) + target.sum(dim=dims)
            dices.extend(((numerator + 1e-5) / (denominator + 1e-5)).float().cpu().tolist())
        losses.append(float(loss.detach()))
    return float(np.mean(losses)), float(np.mean(dices))


def _inner_score(
    model: nn.Module,
    arm: Arm,
    case_ids: Sequence[str],
    config: Mapping[str, Any],
    config_path: Path,
    device: torch.device,
    amp: bool,
) -> float:
    evaluation = config["evaluation"]
    data = config["data"]
    values: list[float] = []
    for case_id in case_ids:
        image_itk = sitk.ReadImage(str(arm.image_path(case_id)))
        image = sitk.GetArrayFromImage(image_itk).astype(np.float32, copy=False)
        probability = _predict(
            model,
            image,
            tuple(int(v) for v in data["patch_shape"]),
            float(evaluation["overlap"]),
            int(evaluation["batch_size"]),
            device,
            amp,
        )
        prediction_itk = sitk.GetImageFromArray(
            (probability >= float(evaluation["threshold"])).astype(np.uint8)
        )
        prediction_itk.CopyInformation(image_itk)
        raw_label_itk = sitk.ReadImage(str(_raw_label_path(config, config_path, case_id)))
        prediction_raw = sitk.Resample(
            prediction_itk,
            raw_label_itk,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        prediction = sitk.GetArrayFromImage(prediction_raw).astype(bool)
        label = sitk.GetArrayFromImage(raw_label_itk).astype(bool)
        values.append(_dice_only(prediction.astype(np.float32), label.astype(np.float32), 0.5))
    return float(np.mean(values))


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _save_restart_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    """Atomically save state used to resume a preempted/requeued task."""
    temporary = path.with_suffix(".tmp")
    torch.save(dict(state), temporary)
    temporary.replace(path)


def run_task(config_path: Path, arm_name: str, fold: int, seed: int, smoke: bool = False) -> Path:
    config = _load_yaml(config_path)
    arms = _arms(config, config_path)
    if arm_name not in arms:
        raise ValueError(f"Unknown arm {arm_name}; choose from {sorted(arms)}")
    expected_folds = [int(v) for v in config["folds"]]
    expected_seeds = [int(v) for v in config["seeds"]]
    if fold not in expected_folds or seed not in expected_seeds:
        raise ValueError(f"fold/seed not declared in config: {fold}/{seed}")
    arm = arms[arm_name]
    outer_train, outer_test = _fold_ids(config, config_path, fold)
    training = config["training"]
    if smoke:
        outer_train = outer_train[:8]
        outer_test = outer_test[:2]
    inner_train, inner_validation = _inner_split(
        outer_train,
        int(config.get("inner_split_seed", 4242)) + fold,
        0.25 if smoke else float(config["inner_validation_fraction"]),
    )
    output_root = _resolve(str(config["output_root"]), config_path)
    task_root = output_root / ("_smoke" if smoke else arm_name)
    if smoke:
        task_root = task_root / arm_name
    task_root = task_root / f"fold_{fold}" / f"seed_{seed}"
    done_path = task_root / "DONE.json"
    if done_path.exists():
        raise FileExistsError(f"Completed task already exists: {done_path}")
    partial_task = task_root.exists() and any(task_root.iterdir())
    if partial_task and not (task_root / "last.pt").exists() and not (task_root / "best.pt").exists():
        raise FileExistsError(f"Partial task has no resumable checkpoint: {task_root}")
    task_root.mkdir(parents=True, exist_ok=True)
    _seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bool(config.get("require_cuda", True)) and device.type != "cuda":
        raise RuntimeError("CUDA is required by this protocol")
    amp = bool(training.get("amp", True)) and device.type == "cuda"
    patch_shape = tuple(int(v) for v in config["data"]["patch_shape"])
    dataset = AblationDataset(
        arm,
        inner_train,
        patch_shape,
        1 if smoke else int(config["data"].get("patches_per_case", 2)),
        seed,
        float(config["data"].get("foreground_probability", 0.67)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    model = UNet3D(1, 1, int(config["model"].get("base_filters", 16))).to(device)
    loss_fn = DiceCELoss(
        dice_weight=float(config["loss"].get("dice_weight", 1.0)),
        bce_weight=float(config["loss"].get("bce_weight", 1.0)),
        bce_pos_weight=float(config["loss"].get("bce_pos_weight", 100.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    epochs = 2 if smoke else int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    evaluation_interval = 1 if smoke else int(training.get("evaluation_interval", 5))
    patience = 2 if smoke else int(training.get("early_stopping_patience", 30))
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    start_epoch = 0
    resume_source: str | None = None
    restart_path = task_root / "last.pt"
    if partial_task:
        resume_path = restart_path if restart_path.exists() else task_root / "best.pt"
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = int(checkpoint["epoch"])
        best_epoch = int(checkpoint.get("best_epoch", checkpoint["epoch"]))
        best_score = float(checkpoint.get("best_score", checkpoint.get("score", -1.0)))
        history = list(checkpoint.get("history", []))
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            if "generator_state" in checkpoint:
                generator.set_state(checkpoint["generator_state"].cpu())
        else:
            # Older tasks only saved the best model. Match the cosine schedule at
            # that epoch; optimizer moments cannot be reconstructed.
            progress = min(start_epoch, epochs)
            learning_rate = float(training["learning_rate"]) * (1.0 + math.cos(math.pi * progress / epochs)) / 2.0
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            scheduler.last_epoch = progress
            scheduler._step_count = progress + 1
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        resume_source = str(resume_path)
        print(f"resuming from {resume_path} at epoch={start_epoch}", flush=True)
    started = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        dataset.set_epoch(epoch)
        loss, train_dice = _train_epoch(
            model, loader, loss_fn, optimizer, device, amp, float(training.get("grad_clip", 1.0))
        )
        scheduler.step()
        record: dict[str, Any] = {"epoch": epoch, "loss": loss, "train_patch_dice": train_dice}
        should_evaluate = epoch == 1 or epoch % evaluation_interval == 0 or epoch == epochs
        if should_evaluate:
            score = _inner_score(model, arm, inner_validation, config, config_path, device, amp)
            record["inner_validation_dice"] = score
            print(f"epoch={epoch} loss={loss:.5f} train_dice={train_dice:.5f} inner_dice={score:.5f}", flush=True)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "score": score}, task_root / "best.pt")
            should_stop = epoch - best_epoch >= patience
        else:
            print(f"epoch={epoch} loss={loss:.5f} train_dice={train_dice:.5f}", flush=True)
            should_stop = False
        history.append(record)
        _save_restart_checkpoint(
            restart_path,
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "generator_state": generator.get_state(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "history": history,
            },
        )
        if should_stop:
            break
    checkpoint = torch.load(task_root / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    evaluation = config["evaluation"]
    cases: list[dict[str, Any]] = []
    for index, case_id in enumerate(outer_test, start=1):
        result = _evaluate_case(
            model,
            arm,
            case_id,
            _raw_label_path(config, config_path, case_id),
            patch_shape,
            float(evaluation["overlap"]),
            int(evaluation["batch_size"]),
            float(evaluation["threshold"]),
            float(evaluation.get("surface_dice_tolerance_mm", 2.0)),
            device,
            amp,
        )
        cases.append(result)
        print(
            f"test={index}/{len(outer_test)} case={case_id} dice={result['dice']:.5f} "
            f"hd95={result['hd95_mm']}",
            flush=True,
        )
    fieldnames = list(cases[0])
    with (task_root / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)
    result = {
        "schema_version": 1,
        "status": "DONE",
        "created_utc": _utc_now(),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__)),
        "smoke": smoke,
        "arm": arm_name,
        "image_template": arm.image_template,
        "label_template": arm.label_template,
        "fold": fold,
        "seed": seed,
        "device": str(device),
        "model_parameters": model_parameter_count(model),
        "outer_train_cases": len(outer_train),
        "inner_train_cases": len(inner_train),
        "inner_validation_cases": len(inner_validation),
        "outer_test_cases": len(outer_test),
        "inner_train_case_ids": inner_train,
        "inner_validation_case_ids": inner_validation,
        "outer_test_case_ids": outer_test,
        "best_epoch": best_epoch,
        "best_inner_validation_dice": best_score,
        "resumed_from": resume_source,
        "start_epoch": start_epoch,
        "elapsed_seconds": round(time.time() - started, 3),
        "history": history,
        "cases": cases,
    }
    done_path.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    return done_path


def preflight(config_path: Path) -> None:
    config = _load_yaml(config_path)
    arms = _arms(config, config_path)
    expected_cases = int(config.get("expected_labeled_cases", 500))
    all_test: list[str] = []
    all_cases: set[str] = set()
    for fold in [int(v) for v in config["folds"]]:
        train, test = _fold_ids(config, config_path, fold)
        overlap = set(train) & set(test)
        if overlap:
            raise ValueError(f"Fold {fold}: train/test overlap: {sorted(overlap)[:5]}")
        if len(set(train) | set(test)) != expected_cases:
            raise ValueError(f"Fold {fold}: train/test union is not {expected_cases} cases")
        all_test.extend(test)
        all_cases.update(train)
        all_cases.update(test)
    if len(all_test) != len(set(all_test)):
        raise ValueError("A case appears in more than one outer test fold")
    missing: list[str] = []
    for case_id in sorted(all_cases):
        raw_label = _raw_label_path(config, config_path, case_id)
        if not raw_label.is_file():
            missing.append(str(raw_label))
        for arm in arms.values():
            for path in (arm.image_path(case_id), arm.label_path(case_id)):
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required files; first: {missing[:10]}")
    print(
        f"PASS: {len(arms)} arms, {len(config['folds'])} folds, "
        f"{len(config['seeds'])} seeds, {len(all_cases)} cases, {len(all_test)} test assignments"
    )


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _wilcoxon_approx(differences: NDArray[np.float64]) -> float:
    values = differences[np.isfinite(differences) & (differences != 0)]
    if not len(values):
        return 1.0
    ranks = _average_ranks(np.abs(values))
    positive = float(ranks[values > 0].sum())
    mean = float(ranks.sum()) / 2.0
    variance = float(np.square(ranks).sum()) / 4.0
    if variance == 0:
        return 1.0
    z = (abs(positive - mean) - 0.5) / math.sqrt(variance)
    return float(math.erfc(max(0.0, z) / math.sqrt(2.0)))


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: p_values[key])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, p_values[key] * (count - index)))
        adjusted[key] = running
    return adjusted


def aggregate(config_path: Path) -> Path:
    config = _load_yaml(config_path)
    arms = list(_arms(config, config_path))
    folds = [int(v) for v in config["folds"]]
    seeds = [int(v) for v in config["seeds"]]
    output_root = _resolve(str(config["output_root"]), config_path)
    by_arm_case: dict[str, dict[str, list[dict[str, float]]]] = {arm: {} for arm in arms}
    missing: list[str] = []
    for arm in arms:
        for fold in folds:
            for seed in seeds:
                path = output_root / arm / f"fold_{fold}" / f"seed_{seed}" / "DONE.json"
                if not path.is_file():
                    missing.append(str(path))
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                for case in record["cases"]:
                    by_arm_case[arm].setdefault(case["case_id"], []).append(case)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} task results; first: {missing[:10]}")
    metrics = ["dice", "hd95_mm", "surface_dice_2mm", "sensitivity", "precision"]
    averaged: dict[str, dict[str, dict[str, float]]] = {arm: {} for arm in arms}
    summaries: dict[str, Any] = {}
    rng = np.random.default_rng(int(config.get("bootstrap_seed", 911)))
    bootstrap_samples = int(config.get("bootstrap_samples", 10000))
    for arm in arms:
        for case_id, records in by_arm_case[arm].items():
            if len(records) != len(seeds):
                raise ValueError(f"{arm}/{case_id}: expected {len(seeds)} seed records, got {len(records)}")
            averaged[arm][case_id] = {
                metric: float(np.mean([float(record[metric]) for record in records])) for metric in metrics
            }
        summaries[arm] = {}
        for metric in metrics:
            values = np.asarray([record[metric] for record in averaged[arm].values()], dtype=np.float64)
            finite = values[np.isfinite(values)]
            draws = (
                rng.choice(finite, size=(bootstrap_samples, len(finite)), replace=True).mean(axis=1)
                if len(finite)
                else np.asarray([], dtype=np.float64)
            )
            summaries[arm][metric] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "finite_cases": int(np.isfinite(values).sum()),
                "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
                if len(draws)
                else None,
            }
    comparisons: dict[str, Any] = {}
    for reference in ("P0_MEDIAN_ZSCORE", "legacy_corrected"):
        if reference not in arms:
            continue
        for arm in arms:
            if arm == reference:
                continue
            common = sorted(set(averaged[reference]) & set(averaged[arm]))
            left = np.asarray([averaged[reference][case]["dice"] for case in common])
            right = np.asarray([averaged[arm][case]["dice"] for case in common])
            difference = right - left
            draws = rng.choice(difference, size=(bootstrap_samples, len(difference)), replace=True).mean(axis=1)
            comparisons[f"{arm}_minus_{reference}"] = {
                "cases": len(common),
                "mean_delta_dice": float(np.mean(difference)),
                "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
                "wilcoxon_normal_approx_p": _wilcoxon_approx(difference),
            }
    adjusted = _holm_adjust(
        {key: float(value["wilcoxon_normal_approx_p"]) for key, value in comparisons.items()}
    )
    for key, value in adjusted.items():
        comparisons[key]["holm_adjusted_p"] = value
    result = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "analysis_unit": "case; metrics averaged across seeds before case-level aggregation",
        "summaries": summaries,
        "paired_dice_comparisons": comparisons,
    }
    path = output_root / "aggregate.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(path)
    return path


def status(config_path: Path) -> int:
    config = _load_yaml(config_path)
    output_root = _resolve(str(config["output_root"]), config_path)
    arms = list(_arms(config, config_path))
    folds = [int(v) for v in config["folds"]]
    seeds = [int(v) for v in config["seeds"]]
    complete = 0
    total = len(arms) * len(folds) * len(seeds)
    missing_task_ids: list[int] = []
    for arm_index, arm in enumerate(arms):
        arm_complete = 0
        for fold_index, fold in enumerate(folds):
            for seed_index, seed in enumerate(seeds):
                task_id = arm_index * len(folds) * len(seeds) + fold_index * len(seeds) + seed_index
                done = output_root / arm / f"fold_{fold}" / f"seed_{seed}" / "DONE.json"
                if done.is_file():
                    arm_complete += 1
                else:
                    missing_task_ids.append(task_id)
        complete += arm_complete
        print(f"{arm}: {arm_complete}/{len(folds) * len(seeds)} complete")
    print(f"TOTAL: {complete}/{total} complete")
    if missing_task_ids:
        print("MISSING_TASK_IDS=" + ",".join(str(value) for value in missing_task_ids))
    return 0 if complete == total else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    task_parser = subparsers.add_parser("run-task")
    task_parser.add_argument("--arm", required=True)
    task_parser.add_argument("--fold", type=int, required=True)
    task_parser.add_argument("--seed", type=int, required=True)
    task_parser.add_argument("--smoke", action="store_true")
    subparsers.add_parser("status")
    subparsers.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.config)
    elif args.command == "run-task":
        print(run_task(args.config, args.arm, args.fold, args.seed, args.smoke))
    elif args.command == "status":
        return status(args.config)
    else:
        aggregate(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
