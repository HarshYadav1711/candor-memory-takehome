from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candor_memory import QuestionFormatError, load_questions, parse_question  # noqa: E402


BASE = {
    "id": "Q-1",
    "question": "What happened?",
    "as_of": "2026-09-18T12:00:00-07:00",
}


class QuestionTests(unittest.TestCase):
    def test_extra_gold_fields_do_not_affect_parsing(self) -> None:
        plain = parse_question(BASE)
        annotated = parse_question(
            {
                **BASE,
                "gold_answer": "ignored",
                "needed": [["ignored-id"]],
                "evidence": ["ignored-id"],
                "answerable": True,
            }
        )
        self.assertEqual(plain, annotated)
        self.assertEqual({"question_id", "question", "as_of"}, set(plain.__dataclass_fields__))

    def test_malformed_question_fails_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_text(json.dumps(BASE) + "\n" + "not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(QuestionFormatError, "line 2: invalid JSON"):
                load_questions(path)

    def test_missing_required_field_fails_clearly(self) -> None:
        with self.assertRaisesRegex(QuestionFormatError, "'question'"):
            parse_question({"id": "Q", "as_of": BASE["as_of"]})

    def test_duplicate_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            line = json.dumps(BASE)
            path.write_text(line + "\n" + line + "\n", encoding="utf-8")
            with self.assertRaisesRegex(QuestionFormatError, "duplicate question id"):
                load_questions(path)


if __name__ == "__main__":
    unittest.main()

