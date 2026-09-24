from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brats_men_rt.stages.nnunet_env_check import _install_hints


class InstallHintsTest(unittest.TestCase):
    def test_blosc2_and_graphviz_requirements_hint(self) -> None:
        hints = _install_hints(["blosc2", "graphviz"])

        self.assertTrue(any("pip install --no-deps -r requirements.txt" in hint for hint in hints))
        self.assertTrue(any("conda install -c conda-forge graphviz" in hint for hint in hints))

    def test_blosc2_runtime_dependency_has_requirements_hint(self) -> None:
        hints = _install_hints(["numexpr"])

        self.assertEqual(len(hints), 1)
        self.assertIn("pip install --no-deps -r requirements.txt", hints[0])

    def test_pydantic_runtime_dependency_has_requirements_hint(self) -> None:
        hints = _install_hints(["pydantic_core"])

        self.assertEqual(len(hints), 1)
        self.assertIn("pip install --no-deps -r requirements.txt", hints[0])

    def test_torch_has_separate_hint(self) -> None:
        hints = _install_hints(["torch"])

        self.assertEqual(len(hints), 1)
        self.assertIn("Torch is missing.", hints[0])


if __name__ == "__main__":
    unittest.main()
