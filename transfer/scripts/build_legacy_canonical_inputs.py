#!/usr/bin/env python3
"""Build duplicate-free legacy manifest and label-QC inputs from canonical data."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


LEGACY_FIELDS = (
    "case_id",
    "split_group",
    "source_dir",
    "image_path",
    "mask_path",
    "has_label",
    "pairing_status",
    "warnings",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def assert_unique(rows: list[dict[str, str]], label: str) -> None:
    counts = Counter(row["case_id"] for row in rows)
    duplicates = sorted(case_id for case_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicate case IDs: {', '.join(duplicates)}")


def build_manifest(canonical_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    assert_unique(canonical_rows, "canonical manifest")
    rows: list[dict[str, str]] = []
    for row in canonical_rows:
        image = Path(row["t1c_path"])
        label = row["label_path"]
        rows.append(
            {
                "case_id": row["case_id"],
                "split_group": "train" if row["split"] == "train" else "val_unlabeled",
                "source_dir": image.parent.as_posix(),
                "image_path": image.as_posix(),
                "mask_path": label,
                "has_label": str(bool(label)),
                "pairing_status": "paired" if label else "image_only",
                "warnings": "",
            }
        )
    if Counter(row["split_group"] for row in rows) != Counter({"train": 500, "val_unlabeled": 70}):
        raise ValueError("legacy canonical manifest must contain 500 train and 70 val_unlabeled rows")
    return rows


def build_mask_qc(canonical_rows: list[dict[str, str]], legacy_qc_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in legacy_qc_rows:
        grouped[row["case_id"]].append(row)
    output: list[dict[str, str]] = []
    train_ids = [row["case_id"] for row in canonical_rows if row["split"] == "train"]
    for case_id in train_ids:
        candidates = grouped.get(case_id, [])
        if not candidates:
            raise ValueError(f"missing legacy label QC for canonical train case: {case_id}")
        comparable = [{key: value for key, value in row.items() if key != "split_group"} for row in candidates]
        if any(row != comparable[0] for row in comparable[1:]):
            raise ValueError(f"conflicting duplicate label QC rows for: {case_id}")
        chosen = dict(candidates[0])
        chosen["split_group"] = "train"
        output.append(chosen)
    assert_unique(output, "canonical mask QC")
    if len(output) != 500:
        raise ValueError(f"expected 500 canonical mask QC rows, found {len(output)}")
    return output


def main() -> int:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-manifest",
        type=Path,
        default=workspace / "code" / "brats-research" / "manifests" / "local_dataset.csv",
    )
    parser.add_argument("--legacy-mask-qc", type=Path, default=workspace / "qc" / "mask_qc_summary.csv")
    parser.add_argument("--output-manifest", type=Path, default=workspace / "manifests" / "manifest_raw_canonical.csv")
    parser.add_argument("--output-mask-qc", type=Path, default=workspace / "qc" / "mask_qc_summary_canonical.csv")
    args = parser.parse_args()

    canonical = read_csv(args.canonical_manifest)
    manifest = build_manifest(canonical)
    mask_qc = build_mask_qc(canonical, read_csv(args.legacy_mask_qc))
    write_csv(args.output_manifest, manifest, LEGACY_FIELDS)
    write_csv(args.output_mask_qc, mask_qc, tuple(mask_qc[0].keys()))
    print(f"wrote {len(manifest)} manifest rows and {len(mask_qc)} unique train mask-QC rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
