from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candor_memory.retrieval import BM25Index, aggregate_record_scores, tokenize  # noqa: E402


class BM25Tests(unittest.TestCase):
    def test_relevant_document_ranks_above_unrelated_document(self) -> None:
        index = BM25Index(
            (
                ("relevant", tokenize("routing latency p95")),
                ("unrelated", tokenize("designer onboarding mockups")),
            )
        )
        ranked = index.rank(tokenize("routing latency"))
        self.assertEqual("relevant", ranked[0].document_id)
        self.assertNotIn("unrelated", [result.document_id for result in ranked])

    def test_formula_uses_documented_idf(self) -> None:
        index = BM25Index((("one", ("alpha",)), ("two", ("beta",))))
        expected_idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
        self.assertAlmostEqual(expected_idf, index.inverse_document_frequency("alpha"))
        self.assertAlmostEqual(expected_idf, index.rank(("alpha",))[0].score)

    def test_record_aggregation_uses_only_strongest_passages(self) -> None:
        score = aggregate_record_scores([10.0, 4.0, 2.0, 1.0])
        self.assertAlmostEqual(10.0 + 0.35 * 4.0 + 0.15 * 2.0, score)

    def test_ties_keep_insertion_order(self) -> None:
        index = BM25Index((("second", ("same",)), ("first", ("same",))))
        self.assertEqual(
            ["second", "first"],
            [result.document_id for result in index.rank(("same",))],
        )


if __name__ == "__main__":
    unittest.main()

