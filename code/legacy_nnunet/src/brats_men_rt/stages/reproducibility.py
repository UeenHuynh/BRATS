from __future__ import annotations

import argparse

from brats_men_rt.artifacts import init_artifact_manifest
from brats_men_rt.paths import project_root
from brats_men_rt.reports import write_markdown_report


def run() -> dict[str, object]:
    root = project_root()
    init_artifact_manifest(root / "manifests" / "artifact_manifest.csv")
    write_markdown_report(
        root / "reports" / "reproducibility_report.md",
        "Reproducibility Report",
        [
            "## Required tracked artifacts",
            "- Raw manifest",
            "- QC tables and figures",
            "- Processed manifest",
            "- Split files",
            "- Training configs and checkpoints",
            "",
            "## Policy",
            "- Do not claim dataset cleanliness, segmentation performance, or uncertainty utility until the corresponding QC/evaluation stages run.",
        ],
    )
    return {"status": "initialized"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize reproducibility manifest and report.")
    parser.parse_args()
    run()
    print("Reproducibility templates created")


if __name__ == "__main__":
    main()
