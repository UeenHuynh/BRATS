"""Entry point for training runs.

Usage:
    uv run python -m brats.training configs/training.yaml test_2d
    uv run python -m brats.training configs/training.yaml test_3d
"""

from __future__ import annotations

import sys
from pathlib import Path

from .runner import run_training


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python -m brats.training <spec_path> <profile_id>", file=sys.stderr)
        return 1
    spec_path = Path(sys.argv[1]).resolve()
    profile_id = sys.argv[2]
    return run_training(spec_path, profile_id)


if __name__ == "__main__":
    raise SystemExit(main())
