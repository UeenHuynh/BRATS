from __future__ import annotations

import argparse

from brats_men_rt.constants import LABEL_RISK_COLUMNS
from brats_men_rt.paths import project_root
from brats_men_rt.reports import write_markdown_report
from brats_men_rt.utils import write_csv


def run() -> dict[str, object]:
    root = project_root()
    lines = [
        "## Label scope",
        "- GTV/target-volume is treated as the segmentation label for this technical research pipeline.",
        "- The label must not be treated as diagnosis, treatment recommendation, or clinical validity evidence.",
        "",
        "## Review risks",
        "- Skull-base adjacency, dural-tail-like extension, partial volume effects, and multi-component targets require manual review.",
        "- Out-of-FOV-looking masks should be flagged, not silently removed.",
        "",
        "## Metadata policy",
        "- If scanner, institution, preoperative, or postoperative metadata are missing, record `not available` instead of inferring them.",
    ]
    write_markdown_report(root / "reports" / "label_semantics.md", "Label Semantics", lines)
    write_csv(root / "qc" / "manual_review_label_risk.csv", [], LABEL_RISK_COLUMNS)
    return {"status": "written"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create label semantics and manual review templates.")
    parser.parse_args()
    run()
    print("Label semantics templates created")


if __name__ == "__main__":
    main()
