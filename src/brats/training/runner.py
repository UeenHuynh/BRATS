"""Training loop for 2D and 3D U-Net segmentation."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .config import effective_training_profile, load_training_spec, resolve_path
from .dataset import DatasetSpec, SliceDataset2D, VolumeDataset3D, collate_2d, collate_3d
from .losses import DiceCELoss, DiceLoss, FocalLoss
from .models import build_model, model_parameter_count


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def build_loss(config: Mapping[str, Any]) -> nn.Module:
    name = str(config.get("name", "dice_ce")).lower()
    if name == "dice":
        return DiceLoss()
    if name == "dice_ce":
        pw = config.get("bce_pos_weight")
        return DiceCELoss(
            dice_weight=float(config.get("dice_weight", 1.0)),
            bce_weight=float(config.get("bce_weight", 1.0)),
            bce_pos_weight=float(pw) if pw is not None else None,
        )
    if name == "focal":
        return FocalLoss(alpha=float(config.get("alpha", 0.25)), gamma=float(config.get("gamma", 2.0)))
    raise ValueError(f"Unknown loss: {name}")


def build_optimizer(config: Mapping[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    name = str(config.get("name", "adam")).lower()
    lr = float(config.get("lr", 1e-4))
    wd = float(config.get("weight_decay", 0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=float(config.get("momentum", 0.9)))
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    config: Mapping[str, Any], optimizer: torch.optim.Optimizer, steps_per_epoch: int, epochs: int
) -> object | None:
    name = str(config.get("name", "none")).lower()
    if name == "none":
        return None
    if name == "cosine":
        warmup = int(config.get("warmup_epochs", 0))
        total_steps = max(1, steps_per_epoch * epochs)
        warmup_steps = steps_per_epoch * warmup
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (
                min(1.0, step / max(1, warmup_steps))
                if warmup > 0
                else 1.0
                if step < warmup_steps
                else 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
            ),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(config.get("step_size", 10)), gamma=float(config.get("gamma", 0.5))
        )
    raise ValueError(f"Unknown scheduler: {name}")


def build_dataloader(
    config: Mapping[str, Any], spec_dir: Path, seed: int
) -> tuple[DataLoader[tuple[Tensor, Tensor]], str, int]:
    data_cfg = config["data"]
    dataset_root = resolve_path(str(data_cfg["dataset_root"]), spec_dir)
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    mode = str(data_cfg.get("mode", "2d")).lower()
    patch_size = tuple(data_cfg.get("patch_size", [256, 256]))
    depth = int(data_cfg.get("depth", 64))
    case_limit_raw = data_cfg.get("case_limit")
    case_limit = int(case_limit_raw) if case_limit_raw is not None else None
    train_cfg = config["training"]
    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 4))
    if mode == "2d":
        ds_spec = DatasetSpec(
            dataset_root=dataset_root,
            split="train",
            patch_size=patch_size,
            mode="2d",
            augment=True,
            case_limit=case_limit,
        )
        dataset = SliceDataset2D(ds_spec, seed=seed)
        collate = collate_2d
    elif mode == "3d":
        ds_spec = DatasetSpec(
            dataset_root=dataset_root,
            split="train",
            depth=depth,
            patch_size=patch_size,
            mode="3d",
            augment=True,
            case_limit=case_limit,
        )
        dataset = VolumeDataset3D(ds_spec, seed=seed)
        collate = collate_3d
    else:
        raise ValueError(f"Unknown data mode: {mode}")
    loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate, pin_memory=True
    )
    return loader, mode, len(dataset)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    device: torch.device,
    epoch: int,
    epochs: int,
    amp: bool,
    grad_clip: float | None,
    log_interval: int,
) -> dict[str, float]:
    model.train()
    scaler: torch.amp.GradScaler[torch.device] | None = torch.amp.GradScaler() if amp else None
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    start = time.time()
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        if amp and scaler is not None:
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            dims = tuple(range(1, preds.ndim))
            intersection = (preds * labels).sum(dim=dims)
            denominator = preds.sum(dim=dims) + labels.sum(dim=dims)
            batch_dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
            total_dice += float(batch_dice.mean().detach())
        total_loss += float(loss.detach())
        num_batches += 1
        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
            elapsed = time.time() - start
            print(
                f"  Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {float(loss.detach()):.4f} | Dice: {float(batch_dice.mean().detach()):.4f} | "
                f"{elapsed:.1f}s",
                flush=True,
            )
    if scheduler is not None:
        scheduler.step()
    avg_loss = total_loss / max(1, num_batches)
    avg_dice = total_dice / max(1, num_batches)
    elapsed = time.time() - start
    return {"loss": avg_loss, "dice": avg_dice, "time_sec": elapsed}


def run_training(spec_path: Path, profile_id: str) -> int:
    spec = load_training_spec(spec_path)
    config = effective_training_profile(spec, profile_id)
    spec_dir = spec_path.parent
    seed = int(config["training"].get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output_root = resolve_path(str(config["project"]["output_root"]), spec_dir)
    if output_root is None:
        raise ValueError("project.output_root is required")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + profile_id
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    loader, mode, dataset_len = build_dataloader(config, spec_dir, seed)
    print(f"Dataset: {mode} | {dataset_len} samples", flush=True)
    model = build_model(
        arch=str(config["model"]["arch"]),
        in_channels=int(config["model"].get("in_channels", 1)),
        num_classes=int(config["model"].get("num_classes", 1)),
        base_filters=int(config["model"].get("base_filters", 32)),
    ).to(device)
    print(f"Model: {config['model']['arch']} | Params: {model_parameter_count(model):,}", flush=True)
    loss_fn = build_loss(config["loss"]).to(device)
    optimizer = build_optimizer(config["optimizer"], model)
    train_cfg = config["training"]
    epochs = int(train_cfg.get("epochs", 10))
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    grad_clip_raw = train_cfg.get("grad_clip")
    grad_clip = float(grad_clip_raw) if grad_clip_raw is not None else None
    log_interval = int(train_cfg.get("log_interval", 10))
    steps_per_epoch = max(1, len(loader))
    scheduler = build_scheduler(config.get("scheduler", {"name": "none"}), optimizer, steps_per_epoch, epochs)
    save_checkpoint = bool(train_cfg.get("save_checkpoint", True))
    print(f"Training: {epochs} epochs | batch={train_cfg.get('batch_size')} | amp={amp}", flush=True)
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---", flush=True)
        metrics = train_one_epoch(
            model=model,
            loader=loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            epochs=epochs,
            amp=amp,
            grad_clip=grad_clip,
            log_interval=log_interval,
        )
        lr_val = optimizer.param_groups[0]["lr"]
        print(
            f"  Summary: Loss={metrics['loss']:.4f} | Dice={metrics['dice']:.4f} | "
            f"LR={lr_val:.2e} | Time={metrics['time_sec']:.1f}s",
            flush=True,
        )
        history.append({"epoch": epoch + 1, "lr": lr_val, **metrics})
    if save_checkpoint:
        ckpt_path = run_root / "model_final.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
                "history": history,
                "profile_id": profile_id,
                "run_id": run_id,
            },
            ckpt_path,
        )
        print(f"Checkpoint: {ckpt_path}", flush=True)
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "profile_id": profile_id,
        "device": str(device),
        "model_arch": config["model"]["arch"],
        "model_params": model_parameter_count(model),
        "dataset_len": dataset_len,
        "data_mode": mode,
        "epochs": epochs,
        "resolved_config": config,
        "history": history,
        "created_utc": utc_now(),
    }
    (run_root / "train.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nDone. Output: {run_root}", flush=True)
    return 0
