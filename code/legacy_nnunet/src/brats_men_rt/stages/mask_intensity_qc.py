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

from brats_men_rt.paths import project_root
from brats_men_rt.qc import image_summary, mask_summary
from brats_men_rt.reports import write_markdown_report
from brats_men_rt.utils import write_csv


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save_distribution(values: list[float], title: str, output: Path) -> None:
    if not values:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(30, max(5, len(values) // 4)), color="#10b981", edgecolor="black")
    ax.set_title(title)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run(limit: int | None = None, progress_every: int = 25) -> dict[str, object]:
    root = project_root()
    rows = _read_manifest(root / "manifests" / "manifest_raw.csv")
    mask_rows = []
    intensity_rows = []
    outlier_rows = []
    volumes = []
    fg_ratios = []
    components = []
    processed = 0
    for row in rows:
        if not row["image_path"]:
            continue
        image_row = {"case_id": row["case_id"], "split_group": row["split_group"]}
        image_row.update(image_summary(row["image_path"]))
        intensity_rows.append(image_row)
        if image_row["nan_count"] or image_row["inf_count"]:
            outlier_rows.append({"case_id": row["case_id"], "reason": "non_finite_intensity", "severity": "critical"})
        if row["mask_path"]:
            mask_row = {"case_id": row["case_id"], "split_group": row["split_group"]}
            mask_row.update(mask_summary(row["mask_path"]))
            mask_rows.append(mask_row)
            volumes.append(float(mask_row["volume_ml"]))
            fg_ratios.append(float(mask_row["foreground_ratio"]))
            components.append(int(mask_row["connected_components"]))
            if mask_row["empty_mask_flag"]:
                outlier_rows.append({"case_id": row["case_id"], "reason": "empty_mask", "severity": "critical"})
        processed += 1
        if progress_every and processed % progress_every == 0:
            print(f"[mask_intensity_qc] processed={processed}", flush=True)
        if limit is not None and processed >= limit:
            break
    write_csv(root / "qc" / "mask_qc_summary.csv", mask_rows, mask_rows[0].keys() if mask_rows else ("case_id",))
    write_csv(root / "qc" / "intensity_summary.csv", intensity_rows, intensity_rows[0].keys() if intensity_rows else ("case_id",))
    write_csv(root / "qc" / "mask_outliers.csv", outlier_rows, outlier_rows[0].keys() if outlier_rows else ("case_id", "reason", "severity"))
    _save_distribution(volumes, "Target Volume (ml)", root / "figures" / "volume_distribution.png")
    _save_distribution(fg_ratios, "Foreground Ratio", root / "figures" / "foreground_ratio_distribution.png")
    _save_distribution(components, "Connected Components", root / "figures" / "component_count_distribution.png")
    write_markdown_report(
        root / "reports" / "intensity_qc_report.md",
        "Mask and Intensity QC",
        [
            "## Summary",
            f"- Image summaries: {len(intensity_rows)}",
            f"- Mask summaries: {len(mask_rows)}",
            f"- Outlier rows: {len(outlier_rows)}",
            "",
            "## Reminder",
            "- Foreground, boundary, and small-volume cases must be reported separately from whole-volume averages.",
        ],
    )
    return {"images": len(intensity_rows), "masks": len(mask_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mask and intensity QC for available cases.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on processed cases.")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    result = run(limit=args.limit, progress_every=args.progress_every)
    print(f"Mask/intensity QC complete for {result['images']} images")


if __name__ == "__main__":
    main()
