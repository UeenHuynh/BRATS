from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from brats_men_rt.metrics_engine import evaluate_case, find_prediction_path
from brats_men_rt.reports import write_markdown_report
from brats_men_rt.utils import write_csv


def _require_nibabel():
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency `nibabel`. Create the conda env from configs/environment.yml "
            "or install from requirements.txt before running metrics."
        ) from exc
    return nib


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _summary_rows(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    summaries = []
    for group, group_rows in sorted(groups.items()):
        dices = np.array([float(item["dice"]) for item in group_rows], dtype=np.float64)
        hd95s = np.array([float(item["hd95_mm"]) for item in group_rows], dtype=np.float64)
        summaries.append(
            {
                key: group,
                "n_cases": len(group_rows),
                "dice_mean": float(np.mean(dices)),
                "dice_median": float(np.median(dices)),
                "hd95_mean_mm": float(np.mean(hd95s)),
                "hd95_median_mm": float(np.median(hd95s)),
                "surface_dice_mean": float(np.mean([float(item["surface_dice_mm"]) for item in group_rows])),
            }
        )
    return summaries


def _overall_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = ["dice", "precision", "recall", "hd95_mm", "surface_dice_mm"]
    result = []
    for metric in metrics:
        values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
        result.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n_cases": len(values),
            }
        )
    return result


def _volume_cutoffs(gt_rows: list[dict[str, str]]) -> tuple[float, float]:
    values = np.array([float(row["volume_ml"]) for row in gt_rows], dtype=np.float64)
    return tuple(float(x) for x in np.quantile(values, [1 / 3, 2 / 3]))


def _volume_group(volume_ml: float, cutoffs: tuple[float, float]) -> str:
    if volume_ml <= cutoffs[0]:
        return "small"
    if volume_ml <= cutoffs[1]:
        return "medium"
    return "large"


def run(prediction_dir: Path, tolerance_mm: float = 2.0, limit: int | None = None, progress_every: int = 10) -> dict[str, object]:
    nib = _require_nibabel()
    root = Path(__file__).resolve().parents[3]
    manifest_rows = _read_csv(root / "manifests" / "manifest_raw.csv")
    mask_rows = {row["case_id"]: row for row in _read_csv(root / "qc" / "mask_qc_summary.csv")}
    spacing_rows = {row["case_id"]: row for row in _read_csv(root / "qc" / "spacing_summary.csv")}
    labeled_rows = [row for row in manifest_rows if row["mask_path"] and row["split_group"] != "val_unlabeled"]
    if not labeled_rows:
        raise SystemExit("No labeled rows available in manifests/manifest_raw.csv")

    cutoffs = _volume_cutoffs([mask_rows[row["case_id"]] for row in labeled_rows if row["case_id"] in mask_rows])
    spacing_threshold = float(np.median([float(spacing_rows[row["case_id"]]["spacing_z"]) for row in labeled_rows if row["case_id"] in spacing_rows]))

    per_case_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    processed = 0
    for row in labeled_rows:
        case_id = row["case_id"]
        pred_path = find_prediction_path(prediction_dir, case_id)
        if pred_path is None:
            continue
        gt_nii = nib.as_closest_canonical(nib.load(row["mask_path"]))
        pred_nii = nib.as_closest_canonical(nib.load(str(pred_path)))
        gt = np.asarray(gt_nii.get_fdata(), dtype=np.float32) > 0
        pred = np.asarray(pred_nii.get_fdata(), dtype=np.float32) > 0
        if gt.shape != pred.shape:
            raise SystemExit(f"Shape mismatch for {case_id}: pred={pred.shape}, gt={gt.shape}")
        spacing = tuple(float(v) for v in gt_nii.header.get_zooms()[:3])
        metrics = evaluate_case(pred, gt, spacing, tolerance_mm)
        volume_ml = float(mask_rows[case_id]["volume_ml"])
        spacing_z = float(spacing_rows[case_id]["spacing_z"])
        per_case_rows.append(
            {
                "case_id": case_id,
                "split_group": row["split_group"],
                "prediction_path": str(pred_path),
                "gt_path": row["mask_path"],
                "dice": metrics.dice,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "hd95_mm": metrics.hd95_mm,
                "surface_dice_mm": metrics.surface_dice_mm,
                "fp_components": metrics.fp_components,
                "fn_components": metrics.fn_components,
                "pred_foreground_voxels": metrics.pred_foreground_voxels,
                "gt_foreground_voxels": metrics.gt_foreground_voxels,
                "volume_ml": volume_ml,
                "volume_group": _volume_group(volume_ml, cutoffs),
                "spacing_z_mm": spacing_z,
                "spacing_group": "high_z_spacing" if spacing_z > spacing_threshold else "low_z_spacing",
            }
        )
        component_rows.append(
            {
                "case_id": case_id,
                "fp_components": metrics.fp_components,
                "fn_components": metrics.fn_components,
            }
        )
        processed += 1
        if progress_every and processed % progress_every == 0:
            print(f"[metric_protocol] processed={processed}", flush=True)
        if limit is not None and processed >= limit:
            break

    if not per_case_rows:
        raise SystemExit(
            f"No prediction files found in {prediction_dir}. Expected names like <case_id>_pred.nii.gz or <case_id>.nii.gz."
        )

    overall_rows = _overall_summary(per_case_rows)
    by_volume = _summary_rows(per_case_rows, "volume_group")
    by_spacing = _summary_rows(per_case_rows, "spacing_group")
    write_csv(root / "metrics" / "metrics_per_case.csv", per_case_rows, per_case_rows[0].keys())
    write_csv(root / "metrics" / "metrics_overall.csv", overall_rows, overall_rows[0].keys())
    write_csv(root / "metrics" / "metrics_by_volume_group.csv", by_volume, by_volume[0].keys())
    write_csv(root / "metrics" / "metrics_by_spacing_group.csv", by_spacing, by_spacing[0].keys())
    write_csv(root / "metrics" / "component_error_metrics.csv", component_rows, component_rows[0].keys())
    write_markdown_report(
        root / "reports" / "metric_protocol.md",
        "Metric Protocol",
        [
            "## Evaluation inputs",
            f"- Prediction directory: {prediction_dir}",
            f"- Evaluated cases: {len(per_case_rows)}",
            f"- Surface Dice tolerance (mm): {tolerance_mm}",
            "",
            "## Metrics",
            "- Dice, precision, recall, HD95 in mm, surface Dice in mm, false-positive components, false-negative components.",
            "- Stratified summaries are written for volume group and spacing group.",
        ],
    )
    return {"evaluated_cases": len(per_case_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute segmentation metrics from predicted masks and labeled ground truth.")
    parser.add_argument("--prediction-dir", type=Path, required=True, help="Directory containing predicted NIfTI masks.")
    parser.add_argument("--surface-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    result = run(
        prediction_dir=args.prediction_dir,
        tolerance_mm=args.surface_tolerance_mm,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(f"Metric evaluation complete for {result['evaluated_cases']} cases")


if __name__ == "__main__":
    main()
