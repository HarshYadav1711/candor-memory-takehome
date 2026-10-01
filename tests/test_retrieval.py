from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candor_memory import HierarchicalRetriever, MemoryUnit, load_units, visible_units  # noqa: E402


NOW = datetime(2026, 9, 18, 12, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    record_id: str,
    source_type: str,
    safe_text: str,
    *,
    text: str | None = None,
    available_at: datetime = NOW,
    event_type: str | None = None,
    target_id: str | None = None,
) -> MemoryUnit:
    return MemoryUnit(
        unit_id=unit_id,
        record_id=record_id,
        source_type=source_type,
        available_at=available_at,
        text=safe_text if text is None else text,
        safe_text=safe_text,
        metadata={},
        event_type=event_type,
        target_id=target_id,
    )


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.units = load_units(ROOT / "data")
        cls.by_id = {item.unit_id: item for item in cls.units}

    def test_repeated_invocation_is_identical_and_bounded(self) -> None:
        retriever = HierarchicalRetriever(self.units)
        as_of = max(item.available_at for item in self.units)
        first = retriever.retrieve("route planner launch", as_of)
        second = retriever.retrieve("route planner launch", as_of)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 20)
        self.assertEqual(len(first), len(set(first)))

    def test_every_result_is_from_visible_units(self) -> None:
        retriever = HierarchicalRetriever(self.units)
        as_of = datetime.fromisoformat("2026-09-12T12:00:00-07:00")
        allowed = {item.unit_id for item in visible_units(self.units, as_of)}
        retrieved = retriever.retrieve("route planner launch", as_of)
        self.assertTrue(set(retrieved) <= allowed)

    def test_future_unit_cannot_appear(self) -> None:
        future = unit(
            "FUTURE",
            "FUTURE",
            "gmail",
            "uniquefuturetoken",
            available_at=NOW + timedelta(hours=1),
        )
        retriever = HierarchicalRetriever((future,))
        self.assertEqual((), retriever.retrieve("uniquefuturetoken", NOW))

    def test_deleted_slack_target_cannot_appear_after_deletion(self) -> None:
        deletion = next(item for item in self.units if item.event_type == "message_deleted")
        target = self.by_id[deletion.target_id]
        retriever = HierarchicalRetriever((target, deletion))
        before = retriever.retrieve(target.safe_text, deletion.available_at - timedelta(microseconds=1))
        after = retriever.retrieve(target.safe_text, deletion.available_at)
        self.assertIn(target.unit_id, before)
        self.assertNotIn(target.unit_id, after)

    def test_raw_only_token_cannot_drive_retrieval(self) -> None:
        raw_only = unit(
            "SAFE-1",
            "SAFE-1",
            "slack",
            "sanitized content",
            text="sanitized content rawonlyuniquetoken",
        )
        retriever = HierarchicalRetriever((raw_only,))
        self.assertEqual((), retriever.retrieve("rawonlyuniquetoken", NOW))
        self.assertEqual(("SAFE-1",), retriever.retrieve("sanitized", NOW))

    def test_exact_citable_granularity_is_preserved(self) -> None:
        controlled = (
            unit("MEETING#0001", "MEETING", "meeting", "orchidmeetingtoken"),
            unit("CHAT#m1", "CHAT", "chatgpt", "orchidchattoken"),
            unit("CODEX", "CODEX", "codex", "orchidcodextoken"),
        )
        retriever = HierarchicalRetriever(controlled)
        self.assertEqual(("MEETING#0001",), retriever.retrieve("orchidmeetingtoken", NOW))
        self.assertEqual(("CHAT#m1",), retriever.retrieve("orchidchattoken", NOW))
        self.assertEqual(("CODEX",), retriever.retrieve("orchidcodextoken", NOW))


if __name__ == "__main__":
    unittest.main()

