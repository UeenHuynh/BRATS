from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brats_men_rt.stages.preprocessing import _cropped_affine, _normalize_image, _validate_unique_case_ids


class LegacyPreprocessingRegressionTest(unittest.TestCase):
    def test_nonzero_normalization_preserves_zero_background(self) -> None:
        image = np.asarray([[[0.0, 10.0, 20.0, 30.0]]], dtype=np.float32)

        normalized, _ = _normalize_image(image, (0.0, 100.0), "nonzero")

        self.assertEqual(float(normalized[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(normalized[image != 0].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[image != 0].std()), 1.0, places=6)

    def test_crop_affine_moves_origin_to_crop_start(self) -> None:
        affine = np.asarray(
            [[2.0, 0.0, 0.0, 10.0], [0.0, 3.0, 0.0, 20.0], [0.0, 0.0, 4.0, 30.0], [0, 0, 0, 1]],
            dtype=np.float64,
        )

        cropped = _cropped_affine(affine, ((5, 10), (6, 11), (7, 12)))

        np.testing.assert_allclose(cropped[:3, 3], [20.0, 38.0, 58.0])
        np.testing.assert_allclose(cropped[:3, :3], affine[:3, :3])

    def test_duplicate_case_ids_fail_before_processing(self) -> None:
        rows = [{"case_id": "case-1"}, {"case_id": "case-1"}]

        with self.assertRaisesRegex(ValueError, "duplicate case IDs: case-1"):
            _validate_unique_case_ids(rows)


if __name__ == "__main__":
    unittest.main()
