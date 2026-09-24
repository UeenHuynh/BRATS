from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brats_men_rt.nifti import orientation_code, read_header
from brats_men_rt.paths import project_root
from brats_men_rt.qc import compare_image_mask
from brats_men_rt.reports import write_markdown_report
from brats_men_rt.utils import write_csv


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save_histogram(values: list[float], title: str, xlabel: str, output: Path) -> None:
    if not values:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(20, max(5, len(values) // 5)), color="#3b82f6", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run() -> dict[str, object]:
    root = project_root()
    manifest_path = root / "manifests" / "manifest_raw.csv"
    rows = _read_manifest(manifest_path)
    orientation_rows = []
    spacing_rows = []
    alignment_rows = []
    spacing_outliers = []
    spacing_z = []
    spacing_x = []
    spacing_y = []
    anisotropy = []
    for row in rows:
        if not row["image_path"]:
            continue
        image_header = read_header(row["image_path"])
        orientation_rows.append(
            {
                "case_id": row["case_id"],
                "split_group": row["split_group"],
                "image_orientation": orientation_code(image_header.affine),
                "image_qform_code": image_header.qform_code,
                "image_sform_code": image_header.sform_code,
            }
        )
        spacing_row = {
            "case_id": row["case_id"],
            "split_group": row["split_group"],
            "spacing_x": image_header.spacing[0],
            "spacing_y": image_header.spacing[1],
            "spacing_z": image_header.spacing[2],
            "anisotropy_ratio": max(image_header.spacing) / min(image_header.spacing),
        }
        spacing_rows.append(spacing_row)
        spacing_x.append(image_header.spacing[0])
        spacing_y.append(image_header.spacing[1])
        spacing_z.append(image_header.spacing[2])
        anisotropy.append(spacing_row["anisotropy_ratio"])
        if spacing_row["spacing_z"] > np.median(spacing_z or [spacing_row["spacing_z"]]) * 2:
            spacing_outliers.append(
                {
                    "case_id": row["case_id"],
                    "split_group": row["split_group"],
                    "reason": "z_spacing_outlier_candidate",
                    "severity": "warning",
                }
            )
        if row["mask_path"]:
            mask_header = read_header(row["mask_path"])
            alignment = compare_image_mask(image_header, mask_header)
            alignment_rows.append(
                {
                    "case_id": row["case_id"],
                    "split_group": row["split_group"],
                    "shape_match": alignment["shape_match"],
                    "spacing_match": alignment["spacing_match"],
                    "spacing_delta_max": alignment["spacing_delta_max"],
                    "affine_delta_max": alignment["affine_delta_max"],
                    "image_orientation": alignment["image_orientation"],
                    "mask_orientation": alignment["mask_orientation"],
                    "severity": "critical" if (not alignment["shape_match"] or alignment["affine_delta_max"] > 1e-3) else "info",
                }
            )
    write_csv(root / "qc" / "orientation_summary.csv", orientation_rows, orientation_rows[0].keys() if orientation_rows else ("case_id",))
    write_csv(root / "qc" / "spacing_summary.csv", spacing_rows, spacing_rows[0].keys() if spacing_rows else ("case_id",))
    write_csv(root / "qc" / "affine_alignment_report.csv", alignment_rows, alignment_rows[0].keys() if alignment_rows else ("case_id",))
    write_csv(root / "qc" / "spacing_outliers.csv", spacing_outliers, spacing_outliers[0].keys() if spacing_outliers else ("case_id", "split_group", "reason", "severity"))
    _save_histogram(spacing_x, "Spacing X", "mm", root / "figures" / "spacing_x_histogram.png")
    _save_histogram(spacing_y, "Spacing Y", "mm", root / "figures" / "spacing_y_histogram.png")
    _save_histogram(spacing_z, "Spacing Z", "mm", root / "figures" / "spacing_z_histogram.png")
    _save_histogram(anisotropy, "Anisotropy Ratio", "ratio", root / "figures" / "anisotropy_histogram.png")
    lines = [
        "## Orientation and spacing QC",
        f"- Processed image headers: {len(spacing_rows)}",
        f"- Paired image-mask alignment checks: {len(alignment_rows)}",
        f"- Candidate spacing outliers: {len(spacing_outliers)}",
        "",
        "## Caveat",
        "- This stage checks headers and affine consistency. It does not replace visual overlay review.",
    ]
    write_markdown_report(root / "reports" / "orientation_spacing_qc_report.md", "Orientation Spacing QC", lines)
    return {"headers": len(spacing_rows), "aligned_pairs": len(alignment_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run orientation, spacing, and affine QC.")
    parser.parse_args()
    result = run()
    print(f"Orientation QC complete for {result['headers']} images")


if __name__ == "__main__":
    main()
