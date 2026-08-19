"""Datasets for 2D and 3D training from preprocessed NIfTI volumes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class DatasetSpec:
    dataset_root: Path
    split: str = "train"
    depth: int = 128
    patch_size: tuple[int, int] = (256, 256)
    mode: str = "2d"
    augment: bool = True
    case_limit: int | None = None


def list_cases(spec: DatasetSpec) -> list[str]:
    image_dir = spec.dataset_root / spec.split / "images"
    files = sorted(image_dir.glob("*.nii.gz"))
    cases = [f.stem.replace(".nii", "") for f in files]
    if spec.case_limit is not None:
        cases = cases[: spec.case_limit]
    return cases


def load_volume(case_id: str, spec: DatasetSpec) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    image_path = spec.dataset_root / spec.split / "images" / f"{case_id}.nii.gz"
    label_path = spec.dataset_root / spec.split / "labels" / f"{case_id}.nii.gz"
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32, copy=False)
    label = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))).astype(np.float32, copy=False)
    label = (label > 0).astype(np.float32)
    return image, label


def random_crop(
    image: NDArray[np.float32],
    label: NDArray[np.float32],
    depth: int,
    height: int,
    width: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    d, h, w = image.shape
    d0 = rng.integers(0, max(0, d - depth) + 1) if d > depth else 0
    h0 = rng.integers(0, max(0, h - height) + 1) if h > height else 0
    w0 = rng.integers(0, max(0, w - width) + 1) if w > width else 0
    img_patch = image[d0 : d0 + depth, h0 : h0 + height, w0 : w0 + width]
    lbl_patch = label[d0 : d0 + depth, h0 : h0 + height, w0 : w0 + width]
    if img_patch.shape != (depth, height, width):
        padded = np.zeros((depth, height, width), dtype=np.float32)
        gd, gh, gw = img_patch.shape
        padded[:gd, :gh, :gw] = img_patch[: min(gd, depth), : min(gh, height), : min(gw, width)]
        img_patch = padded
        padded_l = np.zeros((depth, height, width), dtype=np.float32)
        padded_l[:gd, :gh, :gw] = lbl_patch[: min(gd, depth), : min(gh, height), : min(gw, width)]
        lbl_patch = padded_l
    return img_patch, lbl_patch


def random_flip_z(
    image: NDArray[np.float32], label: NDArray[np.float32], rng: np.random.Generator
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if rng.random() < 0.5:
        image = image[:, ::-1].copy()
        label = label[:, ::-1].copy()
    return image, label


def random_flip_y(
    image: NDArray[np.float32], label: NDArray[np.float32], rng: np.random.Generator
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if rng.random() < 0.5:
        image = image[:, :, ::-1].copy()
        label = label[:, :, ::-1].copy()
    return image, label


class SliceDataset2D(Dataset[tuple[Tensor, Tensor]]):
    """2D slice-wise dataset for 2D U-Net training."""

    def __init__(self, spec: DatasetSpec, seed: int = 42) -> None:
        self.spec = spec
        self.case_ids = list_cases(spec)
        self.rng = np.random.default_rng(seed)
        self._cache: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]] = {}
        self._slices: list[tuple[str, int]] = []
        for case_id in self.case_ids:
            image, _ = self._load(case_id)
            for z in range(image.shape[0]):
                self._slices.append((case_id, z))

    def _load(self, case_id: str) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if case_id not in self._cache:
            self._cache[case_id] = load_volume(case_id, self.spec)
            if len(self._cache) > 4:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[case_id]

    def __len__(self) -> int:
        return len(self._slices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        case_id, z = self._slices[index]
        image, label = self._load(case_id)
        img_slice = image[z]
        lbl_slice = label[z]
        ph, pw = self.spec.patch_size
        ih, iw = img_slice.shape
        if ih < ph or iw < pw:
            padded = np.zeros((ph, pw), dtype=np.float32)
            padded[: min(ih, ph), : min(iw, pw)] = img_slice[: min(ih, ph), : min(iw, pw)]
            img_slice = padded
            padded_l = np.zeros((ph, pw), dtype=np.float32)
            padded_l[: min(ih, ph), : min(iw, pw)] = lbl_slice[: min(ih, ph), : min(iw, pw)]
            lbl_slice = padded_l
        else:
            h0 = self.rng.integers(0, ih - ph + 1)
            w0 = self.rng.integers(0, iw - pw + 1)
            img_slice = img_slice[h0 : h0 + ph, w0 : w0 + pw]
            lbl_slice = lbl_slice[h0 : h0 + ph, w0 : w0 + pw]
        if self.spec.augment:
            if self.rng.random() < 0.5:
                img_slice = img_slice[:, ::-1].copy()
                lbl_slice = lbl_slice[:, ::-1].copy()
            if self.rng.random() < 0.5:
                img_slice = img_slice[::-1].copy()
                lbl_slice = lbl_slice[::-1].copy()
        img_t = torch.from_numpy(img_slice[np.newaxis, ...]).float()
        lbl_t = torch.from_numpy(lbl_slice[np.newaxis, ...]).float()
        return img_t, lbl_t


class VolumeDataset3D(Dataset[tuple[Tensor, Tensor]]):
    """3D patch dataset for 3D U-Net training."""

    def __init__(self, spec: DatasetSpec, seed: int = 42) -> None:
        self.spec = spec
        self.case_ids = list_cases(spec)
        self.rng = np.random.default_rng(seed)
        self._cache: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]] = {}

    def _load(self, case_id: str) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if case_id not in self._cache:
            self._cache[case_id] = load_volume(case_id, self.spec)
            if len(self._cache) > 2:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[case_id]

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, label = self._load(self.case_ids[index])
        ph, pw = self.spec.patch_size
        depth = self.spec.depth
        image, label = random_crop(image, label, depth, ph, pw, self.rng)
        if self.spec.augment:
            image, label = random_flip_z(image, label, self.rng)
            image, label = random_flip_y(image, label, self.rng)
        img_t = torch.from_numpy(image[np.newaxis, ...]).float()
        lbl_t = torch.from_numpy(label[np.newaxis, ...]).float()
        return img_t, lbl_t


class SequentialVolumeDataset3D(Dataset[tuple[Tensor, Tensor]]):
    """3D dataset that produces patches by sequential tiling (no random crops) for deterministic validation."""

    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec
        self.case_ids = list_cases(spec)
        self._cache: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]] = {}
        self._patches: list[tuple[str, int, int, int]] = []
        for case_id in self.case_ids:
            image, _ = self._load(case_id)
            d, h, w = image.shape
            ph, pw = self.spec.patch_size
            depth = self.spec.depth
            for dz in range(0, math.ceil(d / depth)):
                for hy in range(0, math.ceil(h / ph)):
                    for wx in range(0, math.ceil(w / pw)):
                        self._patches.append((case_id, dz * depth, hy * ph, wx * pw))

    def _load(self, case_id: str) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if case_id not in self._cache:
            self._cache[case_id] = load_volume(case_id, self.spec)
            if len(self._cache) > 2:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[case_id]

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        case_id, d0, h0, w0 = self._patches[index]
        image, label = self._load(case_id)
        ph, pw = self.spec.patch_size
        depth = self.spec.depth
        img_patch = np.zeros((depth, ph, pw), dtype=np.float32)
        lbl_patch = np.zeros((depth, ph, pw), dtype=np.float32)
        gd = min(depth, image.shape[0] - d0)
        gh = min(ph, image.shape[1] - h0)
        gw = min(pw, image.shape[2] - w0)
        img_patch[:gd, :gh, :gw] = image[d0 : d0 + gd, h0 : h0 + gh, w0 : w0 + gw]
        lbl_patch[:gd, :gh, :gw] = label[d0 : d0 + gd, h0 : h0 + gh, w0 : w0 + gw]
        img_t = torch.from_numpy(img_patch[np.newaxis, ...]).float()
        lbl_t = torch.from_numpy(lbl_patch[np.newaxis, ...]).float()
        return img_t, lbl_t


def build_datasets(
    dataset_root: Path,
    mode: str = "2d",
    depth: int = 64,
    patch_size: tuple[int, int] = (256, 256),
    case_limit: int | None = None,
    seed: int = 42,
) -> tuple[Dataset[tuple[Tensor, Tensor]], int]:
    if mode == "2d":
        train_spec = DatasetSpec(
            dataset_root=dataset_root,
            split="train",
            patch_size=patch_size,
            mode="2d",
            augment=True,
            case_limit=case_limit,
        )
        return SliceDataset2D(train_spec, seed=seed), len(list_cases(train_spec))
    if mode == "3d":
        train_spec = DatasetSpec(
            dataset_root=dataset_root,
            split="train",
            depth=depth,
            patch_size=patch_size,
            mode="3d",
            augment=True,
            case_limit=case_limit,
        )
        return VolumeDataset3D(train_spec, seed=seed), len(list_cases(train_spec))
    raise ValueError(f"Unknown mode: {mode}")


def collate_2d(batch: Sequence[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return images, labels


def collate_3d(batch: Sequence[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return images, labels
