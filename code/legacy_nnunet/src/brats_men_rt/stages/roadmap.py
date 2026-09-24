from __future__ import annotations

import argparse

from brats_men_rt.paths import project_root
from brats_men_rt.utils import write_text


def run() -> dict[str, object]:
    root = project_root()
    readme = """# BRATS-MEN-RT Pipeline Scaffold

## Recommended stage order

1. `python scripts/00_project_setup.py`
2. `python scripts/01_dataset_understanding.py`
3. `python scripts/02_label_semantics_and_annotation_risk.py`
4. `python scripts/03_raw_inventory_and_pairing_qc.py`
5. `python scripts/04_orientation_spacing_alignment_qc.py`
6. `python scripts/05_visual_overlay_qc.py --limit 25`
7. `python scripts/06_mask_and_intensity_qc.py`
8. `python scripts/08_split_and_leakage_protocol.py`
9. Implement and run preprocessing before training.
10. Implement and run baseline segmentation before uncertainty, boundary calibration, and HITL.

## Current status

This repository contains a scaffold for the planning documents. Stages after QC and splitting still need algorithmic implementation before they can produce scientific results.
"""
    write_text(root / "README.md", readme)
    return {"status": "written"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the top-level command roadmap.")
    parser.parse_args()
    run()
    print("README command roadmap written")


if __name__ == "__main__":
    main()
