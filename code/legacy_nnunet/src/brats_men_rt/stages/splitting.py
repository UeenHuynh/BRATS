from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from brats_men_rt.paths import project_root
from brats_men_rt.reports import write_markdown_report
from brats_men_rt.utils import write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _volume_group(volume_ml: float, cutoffs: tuple[float, float]) -> str:
    if volume_ml <= cutoffs[0]:
        return "small"
    if volume_ml <= cutoffs[1]:
        return "medium"
    return "large"


def _unique_index(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    counts = Counter(row["case_id"] for row in rows)
    duplicates = sorted(case_id for case_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicate case IDs: {', '.join(duplicates)}")
    return {row["case_id"]: row for row in rows}


def _validate_folds(folds: list[dict[str, list[str]]], full_set: set[str]) -> None:
    validation_assignments: list[str] = []
    for index, fold in enumerate(folds):
        train = set(fold["train"])
        validation = set(fold["val"])
        if train & validation:
            raise ValueError(f"fold {index} has train/validation overlap")
        if train | validation != full_set:
            raise ValueError(f"fold {index} does not cover the full training set")
        validation_assignments.extend(fold["val"])
    counts = Counter(validation_assignments)
    invalid = sorted(case_id for case_id in full_set if counts[case_id] != 1)
    if invalid or len(validation_assignments) != len(full_set):
        raise ValueError(f"validation assignment must be exactly once per case; invalid={invalid}")


def run(
    n_folds: int = 5,
    *,
    project_root_path: Path | None = None,
    manifest_path: Path | None = None,
    mask_qc_path: Path | None = None,
    output_root: Path | None = None,
    expected_cases: int | None = None,
) -> dict[str, object]:
    root = (project_root_path or project_root()).resolve()
    manifest_file = (manifest_path or root / "manifests" / "manifest_raw.csv").resolve()
    mask_qc_file = (mask_qc_path or root / "qc" / "mask_qc_summary.csv").resolve()
    outputs = (output_root or root).resolve()
    manifest = _unique_index(_read_csv(manifest_file), "manifest")
    mask_rows = _read_csv(mask_qc_file)
    _unique_index(mask_rows, "mask QC")
    labeled_rows = [row for row in mask_rows if manifest.get(row["case_id"], {}).get("split_group") == "train"]
    train_ids = {case_id for case_id, row in manifest.items() if row.get("split_group") == "train"}
    qc_ids = {row["case_id"] for row in labeled_rows}
    if qc_ids != train_ids:
        raise ValueError(f"mask QC/train manifest mismatch: missing={sorted(train_ids - qc_ids)}, extra={sorted(qc_ids - train_ids)}")
    if expected_cases is not None and len(train_ids) != expected_cases:
        raise ValueError(f"expected {expected_cases} train cases, found {len(train_ids)}")
    volumes = np.array([float(row["volume_ml"]) for row in labeled_rows], dtype=np.float64)
    if not len(volumes):
        raise SystemExit("No labeled training rows found in qc/mask_qc_summary.csv")
    cutoffs = tuple(float(x) for x in np.quantile(volumes, [1 / 3, 2 / 3]))
    grouped = {"small": [], "medium": [], "large": []}
    for row in sorted(labeled_rows, key=lambda item: item["case_id"]):
        grouped[_volume_group(float(row["volume_ml"]), cutoffs)].append(row["case_id"])

    folds = [{"train": [], "val": []} for _ in range(n_folds)]
    for case_ids in grouped.values():
        for index, case_id in enumerate(case_ids):
            folds[index % n_folds]["val"].append(case_id)
    full_set = {row["case_id"] for row in labeled_rows}
    for fold in folds:
        fold["train"] = sorted(full_set - set(fold["val"]))
        fold["val"] = sorted(fold["val"])
    _validate_folds(folds, full_set)

    for index, fold in enumerate(folds):
        path_train = outputs / "splits" / f"fold_{index}_train.txt"
        path_val = outputs / "splits" / f"fold_{index}_val.txt"
        path_train.parent.mkdir(parents=True, exist_ok=True)
        path_train.write_text("\n".join(fold["train"]) + "\n", encoding="utf-8")
        path_val.write_text("\n".join(fold["val"]) + "\n", encoding="utf-8")
    write_json(
        outputs / "splits" / "split_config.json",
        {
            "strategy": "five_fold_cv",
            "n_folds": n_folds,
            "volume_cutoffs_ml": list(cutoffs),
            "source_manifest": str(manifest_file),
            "source_mask_qc": str(mask_qc_file),
            "validation_assignment_policy": "each case exactly once",
        },
    )
    summary_rows = []
    for index, fold in enumerate(folds):
        val_rows = [row for row in labeled_rows if row["case_id"] in set(fold["val"])]
        summary_rows.append(
            {
                "fold": index,
                "n_train": len(fold["train"]),
                "n_val": len(fold["val"]),
                "median_volume_ml": float(np.median([float(row["volume_ml"]) for row in val_rows])),
                "small_medium_large": ",".join(
                    f"{name}:{sum(_volume_group(float(row['volume_ml']), cutoffs) == name for row in val_rows)}"
                    for name in ("small", "medium", "large")
                ),
                "warnings": "",
            }
        )
    summary_path = outputs / "reports" / "split_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    write_markdown_report(
        outputs / "reports" / "split_balance_report.md",
        "Split Balance Report",
        [
            "## Summary",
            f"- Folds created: {n_folds}",
            f"- Volume tertile cutoffs (ml): {cutoffs}",
            "- Public validation without labels remains excluded from these fold files.",
        ],
    )
    write_markdown_report(
        outputs / "reports" / "leakage_check_report.md",
        "Leakage Check Report",
        [
            "## Checks",
            "- Fold construction uses only labeled training rows from the raw manifest.",
            "- No public validation unlabeled case is inserted into training or validation metric folds.",
            "- Every labeled training case is assigned to validation exactly once across folds.",
            "- Patient-level leakage still requires metadata if patient identifiers are available later.",
        ],
    )
    return {"folds": n_folds, "cases": len(labeled_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create labeled training folds from mask QC summaries.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mask-qc", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-cases", type=int)
    args = parser.parse_args()
    result = run(
        n_folds=args.n_folds,
        project_root_path=args.project_root,
        manifest_path=args.manifest,
        mask_qc_path=args.mask_qc,
        output_root=args.output_root,
        expected_cases=args.expected_cases,
    )
    print(f"Created {result['folds']} folds from {result['cases']} labeled training cases")


if __name__ == "__main__":
    main()
