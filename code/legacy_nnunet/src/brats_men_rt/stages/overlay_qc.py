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

from brats_men_rt.nifti import load_array, read_header
from brats_men_rt.paths import project_root
from brats_men_rt.reports import write_markdown_report
from brats_men_rt.utils import write_csv


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _render_overlay(image2d: np.ndarray, mask2d: np.ndarray, title: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image2d.T, cmap="gray", origin="lower")
    if np.any(mask2d):
        ax.contour(mask2d.T, levels=[0.5], colors="red", linewidths=0.8)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _case_slices(mask: np.ndarray) -> dict[str, int]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        center = [dim // 2 for dim in mask.shape[:3]]
        return {"axial": center[2], "coronal": center[1], "sagittal": center[0], "maxfg": center[2]}
    centroid = coords.mean(axis=0).astype(int)
    area_by_slice = mask.sum(axis=(0, 1))
    return {
        "sagittal": int(centroid[0]),
        "coronal": int(centroid[1]),
        "axial": int(centroid[2]),
        "maxfg": int(np.argmax(area_by_slice)),
    }


def run(limit: int | None = None, progress_every: int = 25) -> dict[str, object]:
    root = project_root()
    rows = _read_manifest(root / "manifests" / "manifest_raw.csv")
    review_rows = []
    html_items = []
    processed = 0
    for row in rows:
        if not row["mask_path"]:
            continue
        image_header = read_header(row["image_path"])
        image = load_array(row["image_path"], header=image_header, apply_scaling=True)
        mask = load_array(row["mask_path"], apply_scaling=False) > 0
        indices = _case_slices(mask)
        case_id = row["case_id"]
        _render_overlay(image[indices["sagittal"], :, :], mask[indices["sagittal"], :, :], f"{case_id} sagittal", root / "qc" / "overlay" / f"{case_id}_sagittal.png")
        _render_overlay(image[:, indices["coronal"], :], mask[:, indices["coronal"], :], f"{case_id} coronal", root / "qc" / "overlay" / f"{case_id}_coronal.png")
        _render_overlay(image[:, :, indices["axial"]], mask[:, :, indices["axial"]], f"{case_id} axial", root / "qc" / "overlay" / f"{case_id}_axial.png")
        _render_overlay(image[:, :, indices["maxfg"]], mask[:, :, indices["maxfg"]], f"{case_id} maxfg", root / "qc" / "overlay" / f"{case_id}_maxfg.png")
        html_items.append(f"<li>{case_id}</li>")
        review_rows.append({"case_id": case_id, "severity": "pending_review", "notes": ""})
        processed += 1
        if progress_every and processed % progress_every == 0:
            print(f"[overlay_qc] processed={processed}", flush=True)
        if limit is not None and processed >= limit:
            break
    index_html = "<html><body><h1>Overlay QC</h1><ul>" + "".join(html_items) + "</ul></body></html>"
    (root / "qc" / "overlay_index.html").write_text(index_html, encoding="utf-8")
    write_csv(root / "qc" / "manual_review_overlay.csv", review_rows, ("case_id", "severity", "notes"))
    write_markdown_report(
        root / "reports" / "qc_overlay_report.md",
        "QC Overlay Report",
        [
            "## Overlay generation",
            f"- Generated overlay sets: {processed}",
            "- Reviewers must verify alignment, FOV plausibility, and suspicious morphology manually.",
        ],
    )
    return {"processed": processed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate overlay QC images for labeled cases.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on processed labeled cases.")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    result = run(limit=args.limit, progress_every=args.progress_every)
    print(f"Generated overlays for {result['processed']} cases")


if __name__ == "__main__":
    main()
