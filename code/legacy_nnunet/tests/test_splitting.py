from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brats_men_rt.stages.splitting import _unique_index, _validate_folds


class SplitValidationTest(unittest.TestCase):
    def test_duplicate_input_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate case IDs"):
            _unique_index([{"case_id": "a"}, {"case_id": "a"}], "manifest")

    def test_each_case_must_be_validation_once(self) -> None:
        folds = [
            {"train": ["b", "c"], "val": ["a"]},
            {"train": ["a", "c"], "val": ["b"]},
            {"train": ["a", "b"], "val": ["c"]},
        ]
        _validate_folds(folds, {"a", "b", "c"})

    def test_repeated_validation_case_fails(self) -> None:
        folds = [
            {"train": ["b"], "val": ["a"]},
            {"train": ["b"], "val": ["a"]},
        ]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            _validate_folds(folds, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
