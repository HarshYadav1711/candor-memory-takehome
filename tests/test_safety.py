from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candor_memory import load_units, sanitize_text, visible_units  # noqa: E402


SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


class SafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.units = load_units(ROOT / "data")

    def test_known_dataset_secret_is_redacted(self) -> None:
        raw_secret_units = [unit for unit in self.units if SECRET_PATTERN.search(unit.text)]
        self.assertTrue(raw_secret_units, "expected the safety fixture to contain a secret")
        cutoff = max(unit.available_at for unit in self.units)
        visible = visible_units(self.units, cutoff)
        for unit in visible:
            self.assertIsNone(
                SECRET_PATTERN.search(unit.safe_text),
                f"a credential pattern remained in safe_text for {unit.unit_id}",
            )

    def test_planted_html_instruction_is_absent_from_safe_text(self) -> None:
        planted = [unit for unit in self.units if HTML_COMMENT.search(unit.text)]
        self.assertTrue(planted, "expected the safety fixture to contain an HTML comment")
        for unit in planted:
            comment = HTML_COMMENT.search(unit.text)
            self.assertIsNotNone(comment)
            self.assertIsNone(HTML_COMMENT.search(unit.safe_text))
            self.assertFalse(
                comment.group(0) in unit.safe_text,
                f"untrusted comment remained in safe_text for {unit.unit_id}",
            )

    def test_common_credentials_are_redacted_without_an_llm(self) -> None:
        examples = (
            "password=hunter-example",
            "Authorization: Bearer abcdefghijklmnop",
            "api_key: example-token-value",
            "".join(("xox", "b-", "1234567890-", "abcdefghijklmnop")),
        )
        for example in examples:
            with self.subTest(kind=example.split("=")[0].split(":")[0]):
                safe = sanitize_text(example)
                self.assertIn("[REDACTED]", safe)

    def test_sanitization_is_deterministic(self) -> None:
        text = "<!-- untrusted --> token password=example-value"
        self.assertEqual(sanitize_text(text), sanitize_text(text))


if __name__ == "__main__":
    unittest.main()

