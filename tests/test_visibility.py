from __future__ import annotations

import sys
import unittest
from datetime import timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candor_memory import load_units, visible_units  # noqa: E402


class VisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.units = load_units(ROOT / "data")
        cls.by_id = {unit.unit_id: unit for unit in cls.units}

    def test_future_unit_is_invisible_before_availability(self) -> None:
        future = self.by_id["CGPT-0913-BOARD#m3"]
        before = visible_units(self.units, future.available_at - timedelta(microseconds=1))
        self.assertNotIn(future.unit_id, {unit.unit_id for unit in before})
        at_time = visible_units(self.units, future.available_at)
        self.assertIn(future.unit_id, {unit.unit_id for unit in at_time})

    def test_original_slack_content_is_visible_before_edit(self) -> None:
        edit = next(unit for unit in self.units if unit.event_type == "message_changed")
        original = self.by_id[edit.target_id]
        projected = {
            unit.unit_id: unit
            for unit in visible_units(self.units, edit.available_at - timedelta(microseconds=1))
        }
        self.assertEqual(original.text, projected[original.unit_id].text)
        self.assertNotIn(edit.unit_id, projected)

    def test_latest_slack_edit_applies_and_event_remains_citable(self) -> None:
        edit = next(unit for unit in self.units if unit.event_type == "message_changed")
        original = self.by_id[edit.target_id]
        original_text = original.text
        projected = {unit.unit_id: unit for unit in visible_units(self.units, edit.available_at)}
        self.assertEqual(edit.text, projected[original.unit_id].text)
        self.assertEqual(edit.safe_text, projected[original.unit_id].safe_text)
        self.assertEqual(edit.unit_id, projected[original.unit_id].metadata["applied_edit_id"])
        self.assertIn(edit.unit_id, projected)
        self.assertEqual(original_text, original.text, "raw source unit was mutated")

    def test_deleted_target_is_absent_but_deletion_event_remains(self) -> None:
        for deletion in (unit for unit in self.units if unit.event_type == "message_deleted"):
            projected = {
                unit.unit_id: unit for unit in visible_units(self.units, deletion.available_at)
            }
            self.assertNotIn(deletion.target_id, projected)
            self.assertIn(deletion.unit_id, projected)

    def test_target_is_visible_immediately_before_deletion(self) -> None:
        deletion = next(unit for unit in self.units if unit.event_type == "message_deleted")
        projected = {
            unit.unit_id: unit
            for unit in visible_units(self.units, deletion.available_at - timedelta(microseconds=1))
        }
        self.assertIn(deletion.target_id, projected)

    def test_naive_as_of_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone"):
            visible_units(self.units, "2026-09-18T12:00:00")


if __name__ == "__main__":
    unittest.main()

