from __future__ import annotations

import argparse
from collections import Counter

from brats_men_rt.constants import FAILED_CASE_COLUMNS, PAIRING_COLUMNS
from brats_men_rt.dataset import failed_case_rows, scan_dataset
from brats_men_rt.paths import default_data_root, project_root
from brats_men_rt.reports import write_markdown_report
from brats_men_rt.utils import write_csv


def run() -> dict[str, object]:
    root = project_root()
    entries = scan_dataset(default_data_root())
    manifest_rows = [entry.to_row() for entry in entries]
    failed_rows = failed_case_rows(entries)
    write_csv(root / "manifests" / "manifest_raw.csv", manifest_rows, PAIRING_COLUMNS)
    write_csv(root / "manifests" / "pairing_report.csv", manifest_rows, PAIRING_COLUMNS)
    write_csv(root / "reports" / "failed_cases.csv", failed_rows, FAILED_CASE_COLUMNS)
    write_csv(root / "qc" / "manual_review_pairing.csv", failed_rows, FAILED_CASE_COLUMNS)
    status_counts = Counter(entry.pairing_status for entry in entries)
    lines = [
        "## Pairing summary",
        f"- Total entries: {len(entries)}",
        f"- Pairing status counts: {dict(sorted(status_counts.items()))}",
        f"- Failed rows: {len(failed_rows)}",
        "",
        "## Policy",
        "- No case was removed automatically.",
        "- Validation folders without labels remain in the manifest but must not be used for Dice/HD95.",
    ]
    write_markdown_report(root / "reports" / "integrity_report.md", "Integrity Report", lines)
    return {"entries": len(entries), "failed": len(failed_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw manifest and pairing report.")
    parser.parse_args()
    result = run()
    print(f"Manifest written for {result['entries']} entries")


if __name__ == "__main__":
    main()
