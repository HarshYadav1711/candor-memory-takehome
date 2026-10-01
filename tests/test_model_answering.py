from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candor_memory import GroundedMemory, MemoryUnit, build_query_plan  # noqa: E402
from candor_memory.answering import build_evidence_bundle, validate_model_output  # noqa: E402


NOW = datetime.fromisoformat("2026-10-05T12:00:00-07:00")
UNIT = MemoryUnit(
    unit_id="GOOD",
    record_id="GOOD",
    source_type="slack",
    available_at=NOW,
    text="Project status is green.",
    safe_text="Project status is green.",
)


def bundle():
    question = "What is the project status?"
    return build_evidence_bundle((UNIT,), ("GOOD",), question, NOW, build_query_plan(question, NOW))


class ModelAnsweringTests(unittest.TestCase):
    def test_invalid_model_json_is_rejected(self) -> None:
        self.assertIsNone(validate_model_output("not json", bundle()))

    def test_hallucinated_model_source_is_rejected(self) -> None:
        value = {"answer": "The status is green.", "sources": ["MADE-UP"]}
        self.assertIsNone(validate_model_output(value, bundle()))

    def test_unsafe_model_answer_is_rejected(self) -> None:
        value = {"answer": "Use sk-example-secret-value", "sources": ["GOOD"]}
        self.assertIsNone(validate_model_output(value, bundle()))

    def test_invalid_model_output_falls_back_safely(self) -> None:
        memory = GroundedMemory((UNIT,), model_writer=lambda _: "not json")
        answer, _ = memory.answer("What is the project status?", NOW, mode="auto")
        self.assertIn("green", answer.answer)
        self.assertEqual(("GOOD",), answer.sources)

    def test_model_outage_falls_back_in_auto_mode(self) -> None:
        def outage(_):
            raise OSError("offline")

        memory = GroundedMemory((UNIT,), model_writer=outage)
        answer, _ = memory.answer("What is the project status?", NOW, mode="auto")
        self.assertFalse(answer.abstained)
        self.assertIn("green", answer.answer)

    def test_valid_model_output_is_accepted(self) -> None:
        writer = lambda _: {"answer": "The project status is green.", "sources": ["GOOD"]}
        answer, _ = GroundedMemory((UNIT,), model_writer=writer).answer(
            "What is the project status?", NOW, mode="auto"
        )
        self.assertEqual("The project status is green.", answer.answer)


if __name__ == "__main__":
    unittest.main()

