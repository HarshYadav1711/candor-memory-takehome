from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from candor_memory import load_units, parse_datetime  # noqa: E402


class IngestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.units = load_units(DATA)
        cls.by_id = {unit.unit_id: unit for unit in cls.units}

    def test_all_eight_meeting_files_load(self) -> None:
        files = list((DATA / "native" / "meetings").glob("*.json"))
        records = {unit.record_id for unit in self.units if unit.source_type == "meeting"}
        self.assertEqual(8, len(files))
        self.assertEqual({path.stem for path in files}, records)

    def test_meeting_segment_ids_remain_exact(self) -> None:
        unit = self.by_id["MTG-0909-ACME#0042"]
        self.assertEqual("MTG-0909-ACME#0042", unit.unit_id)
        self.assertEqual("MTG-0909-ACME", unit.record_id)

    def test_meeting_segment_availability_uses_segment_end(self) -> None:
        path = DATA / "native" / "meetings" / "MTG-0909-ACME.json"
        meeting = json.loads(path.read_text(encoding="utf-8"))
        segment = next(item for item in meeting["segments"] if item["seg_id"].endswith("#0042"))
        elapsed = (self.by_id[segment["seg_id"]].available_at - parse_datetime(meeting["start"])).total_seconds()
        self.assertEqual(segment["end_s"], elapsed)

    def test_chatgpt_messages_are_individually_citable(self) -> None:
        unit = self.by_id["CGPT-0913-BOARD#m3"]
        self.assertEqual("CGPT-0913-BOARD#m3", unit.unit_id)
        self.assertEqual("CGPT-0913-BOARD", unit.record_id)
        raw = json.loads((DATA / "connectors" / "chatgpt" / "conversations.json").read_text(encoding="utf-8"))
        expected = sum(len(conversation["messages"]) for conversation in raw)
        actual = sum(unit.source_type == "chatgpt" for unit in self.units)
        self.assertEqual(expected, actual)

    def test_codex_sessions_are_session_level_units(self) -> None:
        session_files = list((DATA / "connectors" / "codex" / "sessions").glob("*.jsonl"))
        codex_units = [unit for unit in self.units if unit.source_type == "codex"]
        self.assertEqual(len(session_files), len(codex_units))
        self.assertEqual({"CDX-0912", "CDX-0915", "CDX-F1", "CDX-F2"}, {unit.unit_id for unit in codex_units})
        self.assertTrue(all(unit.unit_id == unit.record_id for unit in codex_units))

    def test_codex_availability_uses_final_event(self) -> None:
        path = DATA / "connectors" / "codex" / "sessions" / "CDX-0912.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(parse_datetime(records[-1]["timestamp"]), self.by_id["CDX-0912"].available_at)

    def test_calendar_availability_uses_updated(self) -> None:
        path = DATA / "connectors" / "google_calendar" / "events.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        for row in rows:
            self.assertEqual(parse_datetime(row["updated"]), self.by_id[row["id"]].available_at)

    def test_all_availability_times_are_timezone_aware(self) -> None:
        self.assertTrue(all(unit.available_at.utcoffset() is not None for unit in self.units))

    def test_unit_ids_are_unique(self) -> None:
        self.assertEqual(len(self.units), len(self.by_id))


if __name__ == "__main__":
    unittest.main()

