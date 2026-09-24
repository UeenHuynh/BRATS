#!/usr/bin/env python3
"""Select a deterministic five-case preprocessing smoke set across label extremes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--output-dir", type=Path, default=workspace / "transfer" / "remote_configs" / "ydang")
    args = parser.parse_args()
    root = args.workspace.resolve()

    qc = read(root / "qc" / "mask_qc_summary_canonical.csv")
    ordered = sorted(qc, key=lambda row: (float(row["volume_ml"]), row["case_id"]))
    selected = {
        ordered[0]["case_id"],
        ordered[len(ordered) // 2]["case_id"],
        ordered[-1]["case_id"],
        max(qc, key=lambda row: (int(row["connected_components"]), row["case_id"]))["case_id"],
        "BraTS-MEN-RT-0402-1",
    }
    if len(selected) != 5:
        raise ValueError(f"smoke selection must contain five unique cases, found {len(selected)}")

    new_rows = [row for row in read(root / "code" / "brats-research" / "manifests" / "local_dataset.csv") if row["case_id"] in selected]
    legacy_rows = [row for row in read(root / "manifests" / "manifest_raw_canonical.csv") if row["case_id"] in selected]
    if len(new_rows) != 5 or len(legacy_rows) != 5:
        raise ValueError("selected cases are missing from a canonical manifest")
    new_rows.sort(key=lambda row: row["case_id"])
    legacy_rows.sort(key=lambda row: row["case_id"])
    write(args.output_dir / "manifest_smoke5.csv", new_rows, list(new_rows[0].keys()))
    write(args.output_dir / "manifest_smoke5_legacy.csv", legacy_rows, list(legacy_rows[0].keys()))
    print("selected", ",".join(row["case_id"] for row in new_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
