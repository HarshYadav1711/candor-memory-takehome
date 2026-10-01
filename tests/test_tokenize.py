from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candor_memory.retrieval import tokenize  # noqa: E402


class TokenizeTests(unittest.TestCase):
    def test_normalization_is_deterministic(self) -> None:
        text = "Route-Planner V2 — JOHN’S 2026/09/30"
        expected = ("route", "planner", "v2", "john's", "2026", "09", "30")
        self.assertEqual(expected, tokenize(text))
        self.assertEqual(tokenize(text), tokenize(text))

    def test_punctuation_variants_match(self) -> None:
        self.assertEqual(tokenize("route-planner"), tokenize("route planner"))


if __name__ == "__main__":
    unittest.main()

