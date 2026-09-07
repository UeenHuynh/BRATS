"""Fair P0--P3 preprocessing benchmark with a fixed 3D train/validation split."""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from brats.evaluation.volume import evaluate_model_2d, evaluate_model_3d
from brats.training.dataset import DatasetSpec, SliceDataset2D, VolumeDataset3D, collate_2d, collate_3d
from brats.training.models import build_model, model_parameter_count
from brats.training.runner import build_loss, build_optimizer, build_scheduler, train_one_epoch


DEFAULT_PROFILES = (
    "P0_MEDIAN_ZSCORE",
    "P1_MEDIAN_CLIP_ZSCORE",
    "P2_MEDIAN_N4_ZSCORE",
    "P3_MEDIAN_N4_CLIP_ZSCORE",
)


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Benchmark config must be a YAML mapping")
    return loaded


def _resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _profile_case_ids(dataset_root: Path, profile: str, split: str) -> set[str]:
    root = dataset_root / profile / split
    image_ids = {path.name.removesuffix(".nii.gz") for path in (root / "images").glob("*.nii.gz")}
    label_ids = {path.name.removesuffix(".nii.gz") for path in (root / "labels").glob("*.nii.gz")}
    return image_ids & label_ids


def _fixed_split(
    dataset_root: Path,
    profiles: list[str],
    split: str,
    seed: int,
    validation_fraction: float,
    case_limit: int | None = None,
) -> tuple[list[str], list[str]]:
    case_sets = [_profile_case_ids(dataset_root, profile, split) for profile in profiles]
    common = set.intersection(*case_sets)
    if not common:
        raise RuntimeError("No common image+label cases found across all benchmark profiles")
    cases = sorted(common)
    if case_limit is not None:
        if case_limit < 2:
            raise ValueError("case_limit must be at least 2")
        cases = cases[:case_limit]
    random.Random(seed).shuffle(cases)
    validation_count = max(1, int(round(len(cases) * validation_fraction)))
    if validation_count >= len(cases):
        raise ValueError("validation_fraction must leave at least one training case")
    return cases[validation_count:], cases[:validation_count]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_benchmark(config_path: Path) -> Path:
    config = _load_config(config_path)
    seed = int(config.get("seed", 42))
    profiles = [str(value) for value in config.get("profiles", DEFAULT_PROFILES)]
    dataset_root = _resolve_path(str(config["dataset_root"]), config_path)
    source_split = str(config.get("source_split", "train"))
    case_limit_raw = config.get("case_limit")
    case_limit = int(case_limit_raw) if case_limit_raw is not None else None
    split_cfg = config.get("split", {})
    validation_fraction = float(split_cfg.get("validation_fraction", 0.2))
    train_ids, validation_ids = _fixed_split(
        dataset_root, profiles, source_split, seed, validation_fraction, case_limit
    )

    data_cfg = config["data"]
    training_cfg = config["training"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    optimizer_cfg = config["optimizer"]
    scheduler_cfg = config.get("scheduler", {"name": "none"})
    depth = int(data_cfg.get("depth", 64))
    patch_size = tuple(int(value) for value in data_cfg.get("patch_size", [128, 128]))
    mode = str(config.get("mode", "3d")).lower()
    if mode not in {"2d", "3d"}:
        raise ValueError(f"Unknown benchmark mode: {mode}")
    batch_size = int(training_cfg.get("batch_size", 2))
    num_workers = int(training_cfg.get("num_workers", 2))
    epochs = int(training_cfg.get("epochs", 5))
    log_interval = int(training_cfg.get("log_interval", 25))
    grad_clip_raw = training_cfg.get("grad_clip")
    grad_clip = float(grad_clip_raw) if grad_clip_raw is not None else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    output_root = _resolve_path(str(config.get("output_root", "../outputs/benchmark_3d")), config_path)
    run_root = output_root / f"seed{seed}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_root.mkdir(parents=True, exist_ok=True)
    split_record = {
        "seed": seed,
        "source_split": source_split,
        "validation_fraction": validation_fraction,
        "train_case_ids": train_ids,
        "validation_case_ids": validation_ids,
        "profiles": profiles,
        "mode": mode,
    }
    (run_root / "split.json").write_text(json.dumps(split_record, indent=2), encoding="utf-8")
    print(
        f"Device: {device} | common cases={len(train_ids) + len(validation_ids)} | "
        f"train={len(train_ids)} | validation={len(validation_ids)} | seed={seed}",
        flush=True,
    )

    results: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "device": str(device),
        "seed": seed,
        "dataset_root": str(dataset_root),
        "source_split": source_split,
        "profiles": profiles,
        "train_cases": len(train_ids),
        "validation_cases": len(validation_ids),
        "case_limit": case_limit,
        "model": model_cfg,
        "data": data_cfg,
        "training": training_cfg,
        "results": {},
    }
    for profile in profiles:
        print(f"\n===== {profile} =====", flush=True)
        _seed_everything(seed)
        profile_root = dataset_root / profile
        train_spec = DatasetSpec(
            dataset_root=profile_root,
            split=source_split,
            depth=depth,
            patch_size=patch_size,
            mode=mode,
            augment=True,
            case_ids=tuple(train_ids),
        )
        if mode == "2d":
            dataset = SliceDataset2D(train_spec, seed=seed)
            collate = collate_2d
        else:
            dataset = VolumeDataset3D(train_spec, seed=seed)
            collate = collate_3d
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
        )
        model = build_model(
            arch=str(model_cfg.get("arch", "unet2d" if mode == "2d" else "unet3d")),
            in_channels=int(model_cfg.get("in_channels", 1)),
            num_classes=int(model_cfg.get("num_classes", 1)),
            base_filters=int(model_cfg.get("base_filters", 16)),
        ).to(device)
        loss_fn = build_loss(loss_cfg).to(device)
        optimizer = build_optimizer(optimizer_cfg, model)
        scheduler = build_scheduler(scheduler_cfg, optimizer, len(loader), epochs)
        print(
            f"Model: {model_cfg.get('arch', 'unet3d')} | Params: {model_parameter_count(model):,} | "
            f"samples={len(dataset)} | epochs={epochs} | batch={batch_size} | amp={amp}",
            flush=True,
        )
        history: list[dict[str, Any]] = []
        for epoch in range(epochs):
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
            lr_value = float(optimizer.param_groups[0]["lr"])
            history.append({"epoch": epoch + 1, "lr": lr_value, **metrics})
            print(
                f"  Summary: Loss={metrics['loss']:.4f} | train Dice={metrics['dice']:.4f} | "
                f"LR={lr_value:.2e} | Time={metrics['time_sec']:.1f}s",
                flush=True,
            )
        evaluation_cfg = config.get("evaluation", {})
        if mode == "2d":
            evaluation = evaluate_model_2d(
                model=model,
                dataset_root=profile_root,
                case_ids=validation_ids,
                split=source_split,
                device=device,
                batch_size=int(evaluation_cfg.get("batch_size", batch_size)),
                threshold=float(evaluation_cfg.get("threshold", 0.5)),
                amp=amp,
            )
        else:
            evaluation = evaluate_model_3d(
                model=model,
                dataset_root=profile_root,
                case_ids=validation_ids,
                split=source_split,
                depth=depth,
                patch_size=patch_size,
                device=device,
                batch_size=int(evaluation_cfg.get("batch_size", batch_size)),
                threshold=float(evaluation_cfg.get("threshold", 0.5)),
                amp=amp,
            )
        result = {
            "profile": profile,
            "model_params": model_parameter_count(model),
            "history": history,
            "validation": evaluation,
        }
        results["results"][profile] = result
        profile_root_out = run_root / profile
        profile_root_out.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "profile": profile, "seed": seed}, profile_root_out / "model_final.pt")
        (profile_root_out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary = evaluation["summary"]
        print(
            f"{profile} FINAL | Dice={summary['dice']['mean']:.4f} | "
            f"HD95={summary['hd95_mm']['mean']:.2f} mm | "
            f"Sensitivity={summary['sensitivity']['mean']:.4f} | "
            f"Precision={summary['precision']['mean']:.4f}",
            flush=True,
        )
    results_path = run_root / "benchmark.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nBenchmark complete: {results_path}", flush=True)
    return results_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_benchmark(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
