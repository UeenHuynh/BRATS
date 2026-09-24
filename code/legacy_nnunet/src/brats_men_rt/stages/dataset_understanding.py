from __future__ import annotations

import argparse
from collections import Counter

from brats_men_rt.dataset import scan_dataset
from brats_men_rt.paths import default_data_root, project_root
from brats_men_rt.reports import write_markdown_report
from brats_men_rt.utils import write_csv


def run() -> dict[str, object]:
    root = project_root()
    entries = scan_dataset(default_data_root())
    split_counts = Counter(entry.split_group for entry in entries)
    label_counts = Counter(entry.has_label for entry in entries)
    failed_rows = [
        {
            "case_id": entry.case_id,
            "split_group": entry.split_group,
            "reason": entry.pairing_status,
            "notes": entry.warnings,
        }
        for entry in entries
        if entry.pairing_status != "paired"
    ]
    lines = [
        "## Inventory snapshot",
        f"- Total discovered case directories: {len(entries)}",
        f"- Split counts: {dict(sorted(split_counts.items()))}",
        f"- Labeled cases: {label_counts.get(True, 0)}",
        f"- Unlabeled cases: {label_counts.get(False, 0)}",
        "",
        "## Caveats",
        "- This report is file-inventory only. It does not claim alignment, label correctness, or dataset cleanliness.",
        "- Validation folders without masks are treated as unlabeled and excluded from segmentation metrics until labels exist.",
    ]
    write_markdown_report(root / "reports" / "dataset_summary.md", "Dataset Summary", lines)
    write_csv(
        root / "reports" / "failed_pairing_cases.csv",
        failed_rows,
        ("case_id", "split_group", "reason", "notes"),
    )
    return {"cases": len(entries), "failed": len(failed_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a high-level dataset understanding report.")
    parser.parse_args()
    result = run()
    print(f"Wrote dataset summary for {result['cases']} case directories")


if __name__ == "__main__":
    main()
