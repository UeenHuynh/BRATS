from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import load_spec, resolve_path


def read_manifest(path: Path, spec: Mapping[str, Any]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    columns = spec["manifest"]["columns"]
    required = [columns[key] for key in ("case_id", "patient_id", "split", "image")]
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    if not rows:
        raise ValueError("Manifest is empty")
    case_ids: set[str] = set()
    patient_partitions: dict[str, tuple[str, str]] = {}
    allowed_splits = {str(value).lower() for value in spec["manifest"].get("allowed_splits", [])}
    require_train_labels = bool(spec["manifest"].get("require_train_labels", True))
    for number, row in enumerate(rows, start=2):
        case_id = manifest_value(row, spec, "case_id")
        patient_id = manifest_value(row, spec, "patient_id")
        split = manifest_value(row, spec, "split").lower()
        fold = manifest_value(row, spec, "fold")
        if not case_id or not patient_id or not split or not manifest_value(row, spec, "image"):
            raise ValueError(f"Manifest row {number} has empty required values")
        if case_id in case_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        case_ids.add(case_id)
        if allowed_splits and split not in allowed_splits:
            raise ValueError(f"Invalid split for {case_id}: {split}")
        if split == "train" and require_train_labels and not manifest_value(row, spec, "label"):
            raise ValueError(f"Training case has no label: {case_id}")
        partition = (split, fold)
        if patient_id in patient_partitions and patient_partitions[patient_id] != partition:
            raise ValueError(f"Patient {patient_id} crosses split/fold boundaries")
        patient_partitions[patient_id] = partition
    return rows


def manifest_value(row: Mapping[str, str], spec: Mapping[str, Any], logical: str) -> str:
    column = spec["manifest"]["columns"].get(logical)
    return row.get(column, "") if column else ""


def case_paths(row: Mapping[str, str], spec: Mapping[str, Any], spec_dir: Path) -> dict[str, Path | None]:
    data_root = resolve_path(spec["project"].get("data_root", "."), spec_dir) or spec_dir
    return {key: resolve_path(manifest_value(row, spec, key), data_root) for key in ("image", "label", "brain_mask")}


def validate_inputs(spec_path: Path, manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    spec = load_spec(spec_path)
    rows = read_manifest(manifest_path, spec)
    spec_dir = spec_path.parent
    missing: list[str] = []
    for row in rows:
        paths = case_paths(row, spec, spec_dir)
        for key in ("image", "label", "brain_mask"):
            path = paths[key]
            if path is not None and not path.exists():
                missing.append(f"{manifest_value(row, spec, 'case_id')}:{key}:{path}")
    if missing:
        raise FileNotFoundError("Missing manifest files:\n" + "\n".join(missing[:20]))
    return spec, rows
