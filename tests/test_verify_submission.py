from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_submission import (  # noqa: E402
    VerificationError,
    audit_action_rows,
    audit_deliverable_text,
    audit_memory_rows,
)


class VerificationAuditTests(unittest.TestCase):
    def test_memory_audit_accepts_bounded_schema(self) -> None:
        rows = [
            {
                "id": "m1",
                "answer": "Known.",
                "sources": ["SL-F-0001"],
                "retrieved": ["SL-F-0001", "EM-F-002"],
                "abstained": False,
            }
        ]
        audit_memory_rows(rows, expected_count=1)

    def test_memory_audit_rejects_oversized_lists(self) -> None:
        rows = [
            {
                "id": "m1",
                "answer": "Known.",
                "sources": [f"SL-F-000{i}" for i in range(6)],
                "retrieved": [],
                "abstained": False,
            }
        ]
        with self.assertRaises(VerificationError):
            audit_memory_rows(rows, expected_count=1)

    def test_memory_audit_rejects_source_outside_retrieved(self) -> None:
        rows = [
            {
                "id": "m1",
                "answer": "Known.",
                "sources": ["SL-F-0001"],
                "retrieved": ["EM-F-002"],
                "abstained": False,
            }
        ]
        with self.assertRaises(VerificationError):
            audit_memory_rows(rows, expected_count=1)

    def test_memory_audit_rejects_malformed_source_id(self) -> None:
        rows = [
            {
                "id": "m1",
                "answer": "Known.",
                "sources": ["not a source"],
                "retrieved": ["not a source"],
                "abstained": False,
            }
        ]
        with self.assertRaises(VerificationError):
            audit_memory_rows(rows, expected_count=1)

    def test_memory_audit_rejects_abstention_with_sources(self) -> None:
        rows = [
            {
                "id": "m1",
                "answer": "I don't know.",
                "sources": ["SL-F-0001"],
                "retrieved": ["SL-F-0001"],
                "abstained": True,
            }
        ]
        with self.assertRaises(VerificationError):
            audit_memory_rows(rows, expected_count=1)

    def test_action_audit_rejects_unknown_action_type(self) -> None:
        rows = [{"id": "a1", "actions": [{"type": "external.execute", "args": {}}]}]
        with self.assertRaises(VerificationError):
            audit_action_rows(rows, expected_count=1)

    def test_action_audit_rejects_confirm_with_side_effect(self) -> None:
        rows = [
            {
                "id": "a1",
                "actions": [
                    {"type": "confirm", "args": {"summary": "Confirm deletion"}},
                    {"type": "app.open", "args": {"name": "Figma"}},
                ],
            }
        ]
        with self.assertRaises(VerificationError):
            audit_action_rows(rows, expected_count=1)

    def test_deliverable_audit_rejects_unsafe_marker(self) -> None:
        with self.assertRaises(VerificationError):
            audit_deliverable_text("answer contains [REDACTED] material")

    def test_deliverable_audit_rejects_absolute_windows_path(self) -> None:
        with self.assertRaises(VerificationError):
            audit_deliverable_text("generated at C:\\Users\\example\\output.jsonl")


if __name__ == "__main__":
    unittest.main()
