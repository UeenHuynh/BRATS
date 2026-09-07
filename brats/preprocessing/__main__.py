"""Command-line entrypoint for the reproducible preprocessing pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Preprocessing YAML config")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV manifest of raw cases")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help="Profiles to run; default is every enabled profile in the config",
    )
    parser.add_argument("--case-limit", type=int, default=None, help="Process only the first N manifest rows")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing profile outputs")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed case/profile")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write the resolved run plan only")
    parser.add_argument(
        "--brain-mask-source",
        choices=("otsu", "totalsegmentator", "precomputed"),
        default=None,
        help="Override brain-mask source for a controlled comparison run",
    )
    args = parser.parse_args()
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("--case-limit must be positive")
    return run(
        spec_path=args.config,
        manifest_path=args.manifest,
        profiles=args.profiles,
        case_limit=args.case_limit,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
        dry_run=args.dry_run,
        brain_mask_source=args.brain_mask_source,
    )


if __name__ == "__main__":
    raise SystemExit(main())
