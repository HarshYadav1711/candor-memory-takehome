from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MemoryCliTests(unittest.TestCase):
    def test_output_preserves_input_order_and_ids(self) -> None:
        questions = [
            {
                "id": "ORDER-2",
                "question": "route planner launch",
                "as_of": "2026-09-18T18:00:00-07:00",
                "gold_answer": "must be ignored",
            },
            {
                "id": "ORDER-1",
                "question": "flight to Denver",
                "as_of": "2026-09-18T18:00:00-07:00",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            output_path = Path(directory) / "output.jsonl"
            input_path.write_text(
                "".join(json.dumps(item) + "\n" for item in questions),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "main.py"),
                    "memory",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--data",
                    str(ROOT / "data"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(["ORDER-2", "ORDER-1"], [row["id"] for row in rows])
            for row in rows:
                self.assertEqual("I don't know", row["answer"])
                self.assertEqual([], row["sources"])
                self.assertTrue(row["abstained"])
                self.assertLessEqual(len(row["retrieved"]), 20)

    def test_debug_output_is_structured_and_contains_no_source_text(self) -> None:
        question = {
            "id": "DEBUG-1",
            "question": "What did I dictate yesterday?",
            "as_of": "2026-09-18T18:00:00-07:00",
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            output_path = Path(directory) / "output.jsonl"
            debug_path = Path(directory) / "debug.jsonl"
            input_path.write_text(json.dumps(question) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "main.py"),
                    "memory",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--debug-output",
                    str(debug_path),
                    "--data",
                    str(ROOT / "data"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            debug = json.loads(debug_path.read_text(encoding="utf-8"))
            self.assertEqual("DEBUG-1", debug["id"])
            self.assertIn("plan", debug)
            self.assertIn("top_units", debug)
            serialized = json.dumps(debug)
            self.assertNotIn("safe_text", serialized)
            self.assertNotIn('"text"', serialized)
            self.assertNotIn("<!--", serialized)

    def test_gold_fields_do_not_change_retrieval(self) -> None:
        base = {
            "id": "GOLD-INDEPENDENCE",
            "question": "What did I dictate about the launch?",
            "as_of": "2026-09-18T18:00:00-07:00",
        }
        annotated = {
            **base,
            "needed": [["SHOULD-NOT-BE-READ"]],
            "evidence": ["SHOULD-NOT-BE-READ"],
            "gold_answer": "SHOULD-NOT-BE-READ",
            "category": "SHOULD-NOT-BE-READ",
            "storyline": "SHOULD-NOT-BE-READ",
            "answerable": False,
            "harm_check": True,
            "rubric": ["SHOULD-NOT-BE-READ"],
            "key_terms": [["SHOULD-NOT-BE-READ"]],
        }
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            retrieved = []
            for index, row in enumerate((base, annotated)):
                input_path = directory_path / f"input-{index}.jsonl"
                output_path = directory_path / f"output-{index}.jsonl"
                input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "main.py"),
                        "memory",
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--data",
                        str(ROOT / "data"),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                retrieved.append(json.loads(output_path.read_text(encoding="utf-8"))["retrieved"])
            self.assertEqual(retrieved[0], retrieved[1])

    def test_malformed_cli_input_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "bad.jsonl"
            output_path = Path(directory) / "output.jsonl"
            input_path.write_text('{"id": "missing-fields"}\n', encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "main.py"),
                    "memory",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(0, result.returncode)
            self.assertIn("line 1", result.stderr)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()

