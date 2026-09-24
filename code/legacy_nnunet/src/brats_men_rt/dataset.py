from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(slots=True)
class CaseEntry:
    case_id: str
    split_group: str
    source_dir: str
    image_path: str
    mask_path: str
    has_label: bool
    pairing_status: str
    warnings: str

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def infer_split_group(case_dir: Path) -> str:
    text = str(case_dir)
    if "Train" in text:
        return "train"
    if "Val" in text:
        return "val_unlabeled"
    if "0402" in text:
        return "ad_hoc_labeled"
    return "unknown"


def find_case_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    case_dirs = []
    for child in sorted(data_root.rglob("*")):
        if child.is_dir():
            has_t1c = any(p.name.endswith("_t1c.nii.gz") for p in child.iterdir() if p.is_file())
            has_nii = any(p.name.endswith(".nii") or p.name.endswith(".nii.gz") for p in child.iterdir() if p.is_file())
            if has_t1c or has_nii:
                case_dirs.append(child)
    return case_dirs


def scan_dataset(data_root: Path) -> list[CaseEntry]:
    entries: list[CaseEntry] = []
    for case_dir in find_case_dirs(data_root):
        image_candidates = sorted(case_dir.glob("*_t1c.nii.gz")) + sorted(case_dir.glob("*_t1c.nii"))
        mask_candidates = sorted(case_dir.glob("*_gtv.nii.gz")) + sorted(case_dir.glob("*_gtv.nii"))
        case_id = case_dir.name
        warnings = []
        pairing_status = "paired"
        image_path = str(image_candidates[0]) if image_candidates else ""
        mask_path = str(mask_candidates[0]) if mask_candidates else ""

        if len(image_candidates) != 1:
            pairing_status = "invalid_image_count"
            warnings.append(f"image_candidates={len(image_candidates)}")
        if len(mask_candidates) > 1:
            pairing_status = "invalid_mask_count"
            warnings.append(f"mask_candidates={len(mask_candidates)}")
        if not image_candidates:
            pairing_status = "missing_image"
            warnings.append("missing_image")
        if not mask_candidates:
            warnings.append("missing_mask")
        entries.append(
            CaseEntry(
                case_id=case_id,
                split_group=infer_split_group(case_dir),
                source_dir=str(case_dir),
                image_path=image_path,
                mask_path=mask_path,
                has_label=bool(mask_candidates),
                pairing_status=pairing_status,
                warnings=";".join(warnings),
            )
        )
    return entries


def duplicate_case_rows(entries: list[CaseEntry]) -> list[dict[str, str]]:
    counts = Counter(entry.case_id for entry in entries)
    duplicates = []
    for entry in entries:
        if counts[entry.case_id] > 1:
            duplicates.append(
                {
                    "case_id": entry.case_id,
                    "split_group": entry.split_group,
                    "source_dir": entry.source_dir,
                    "reason": "duplicate_case_id",
                    "severity": "critical",
                }
            )
    return duplicates


def failed_case_rows(entries: list[CaseEntry]) -> list[dict[str, str]]:
    rows = duplicate_case_rows(entries)
    for entry in entries:
        if entry.pairing_status == "paired" and entry.image_path:
            continue
        reason = entry.pairing_status or "unknown"
        rows.append(
            {
                "case_id": entry.case_id,
                "split_group": entry.split_group,
                "source_dir": entry.source_dir,
                "reason": reason,
                "severity": "blocked" if "missing" in reason else "critical",
            }
        )
    return rows
