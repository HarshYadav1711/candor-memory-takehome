from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candor_memory import HierarchicalRetriever, MemoryUnit  # noqa: E402


NOW = datetime.fromisoformat("2026-09-18T12:00:00-07:00")


def unit(
    unit_id: str,
    safe_text: str,
    *,
    record_id: str | None = None,
    source_type: str = "slack",
    actor: str | None = None,
    metadata: dict[str, object] | None = None,
    available_at: datetime = NOW,
    text: str | None = None,
    event_type: str | None = None,
    target_id: str | None = None,
) -> MemoryUnit:
    return MemoryUnit(
        unit_id=unit_id,
        record_id=record_id or unit_id,
        source_type=source_type,
        available_at=available_at,
        text=safe_text if text is None else text,
        safe_text=safe_text,
        actor=actor,
        metadata=metadata or {},
        event_type=event_type,
        target_id=target_id,
    )


class Phase3RetrievalTests(unittest.TestCase):
    def test_source_hint_boosts_without_eliminating_cross_source_evidence(self) -> None:
        units = (
            unit("DICT", "project cobalt message prepared", source_type="dictation", actor="Alex"),
            unit("MAIL", "project cobalt message prepared", source_type="gmail", actor="Alex"),
        )
        result = HierarchicalRetriever(units).retrieve(
            "What did I dictate about project cobalt, and did it go out?", NOW
        )
        self.assertEqual("DICT", result[0])
        self.assertIn("MAIL", result)

    def test_reported_speech_text_is_retrievable_without_false_authorship(self) -> None:
        units = (
            unit("REPORT", "John said cobalt storage was ready", actor="Dana Reed"),
            unit("DIRECT", "I prefer amber storage", actor="John Lane"),
        )
        result = HierarchicalRetriever(units).retrieve(
            "What did Dana Reed say John said about cobalt storage?", NOW
        )
        self.assertEqual("REPORT", result[0])

    def test_exact_unit_reranking_prefers_full_coverage_in_parent(self) -> None:
        units = (
            unit("CHAT#m1", "prototype database options", record_id="CHAT", source_type="chatgpt"),
            unit(
                "CHAT#m2",
                "prototype database choice is postgres because spatial queries need it",
                record_id="CHAT",
                source_type="chatgpt",
            ),
        )
        result = HierarchicalRetriever(units).retrieve(
            "Which database did I pick for the prototype and why?", NOW
        )
        self.assertEqual("CHAT#m2", result[0])

    def test_proximity_improves_target_without_creating_neighbor_result(self) -> None:
        units = (
            unit(
                "MEET#1", "alpha beta decision", record_id="MEET", source_type="meeting",
                metadata={"segment_start_s": 0},
            ),
            unit(
                "MEET#2", "alpha discussion", record_id="MEET", source_type="meeting",
                metadata={"segment_start_s": 5},
            ),
            unit(
                "MEET#3", "completely unrelated", record_id="MEET", source_type="meeting",
                metadata={"segment_start_s": 10},
            ),
        )
        result = HierarchicalRetriever(units).retrieve("alpha beta", NOW)
        self.assertEqual("MEET#1", result[0])
        self.assertNotIn("MEET#3", result)

    def test_causal_change_prefers_explanation_over_date_repetition(self) -> None:
        units = (
            unit("DATE", "launch date September 30 September 30"),
            unit("CAUSE", "launch moved because a blocker delayed verification"),
        )
        result = HierarchicalRetriever(units).retrieve(
            "Why did the launch move from September 30?", NOW
        )
        self.assertEqual("CAUSE", result[0])

    def test_disagreement_diversity_preserves_multiple_speakers(self) -> None:
        units = tuple(
            unit(f"A{index}", "cobalt storage opinion", actor="Alice Stone")
            for index in range(4)
        ) + (unit("B", "cobalt storage different view", actor="Bob Vale"),)
        result = HierarchicalRetriever(units).retrieve(
            "Did Alice Stone and Bob Vale agree on cobalt storage?", NOW, limit=3
        )
        actors = {next(item.actor for item in units if item.unit_id == unit_id) for unit_id in result}
        self.assertEqual({"Alice Stone", "Bob Vale"}, actors)

    def test_temporal_anchor_two_hop_retrieves_anchor_and_target_date(self) -> None:
        units = (
            unit(
                "TRIP",
                "Flight to Cedar City on September 23, 2026.",
                source_type="gmail",
                available_at=NOW - timedelta(days=3),
            ),
            unit(
                "CAL-23",
                "Board review",
                source_type="google_calendar",
                metadata={"start": {"dateTime": "2026-09-23T09:00:00-07:00"}},
                available_at=NOW - timedelta(days=10),
            ),
            unit(
                "CAL-24",
                "Planning review",
                source_type="google_calendar",
                metadata={"start": {"dateTime": "2026-09-24T09:00:00-07:00"}},
                available_at=NOW - timedelta(days=10),
            ),
        )
        retriever = HierarchicalRetriever(units)
        result, debug = retriever.retrieve_with_debug(
            "What's on my calendar the day I fly to Cedar City?", NOW
        )
        self.assertIn("TRIP", result[:5])
        self.assertIn("CAL-23", result[:5])
        self.assertEqual("2026-09-23", debug["multi_hop"]["resolved_date"])

    def test_unresolved_anchor_falls_back_to_ordinary_retrieval(self) -> None:
        units = (unit("UNKNOWN", "I fly to Cedar City but the date is unknown"),)
        result, debug = HierarchicalRetriever(units).retrieve_with_debug(
            "What happened the day I fly to Cedar City?", NOW
        )
        self.assertIn("UNKNOWN", result)
        self.assertTrue(debug["multi_hop"]["attempted"])
        self.assertIsNone(debug["multi_hop"]["resolved_date"])

    def test_multi_hop_never_returns_future_unit(self) -> None:
        units = (
            unit("TRIP", "Flight to Cedar City on September 23, 2026."),
            unit(
                "FUTURE-CAL",
                "Future board review",
                source_type="google_calendar",
                metadata={"start": {"dateTime": "2026-09-23T09:00:00-07:00"}},
                available_at=NOW + timedelta(seconds=1),
            ),
        )
        result = HierarchicalRetriever(units).retrieve(
            "What's on my calendar the day I fly to Cedar City?", NOW
        )
        self.assertNotIn("FUTURE-CAL", result)

    def test_multi_hop_respects_slack_deletion(self) -> None:
        target = unit("TARGET", "I fly to Cedar City on September 23, 2026.")
        deletion = unit(
            "DELETE",
            "Message TARGET was deleted.",
            event_type="message_deleted",
            target_id="TARGET",
            available_at=NOW,
        )
        result, debug = HierarchicalRetriever((target, deletion)).retrieve_with_debug(
            "What happened the day I fly to Cedar City?", NOW
        )
        self.assertNotIn("TARGET", result)
        self.assertIsNone(debug["multi_hop"]["resolved_date"])

    def test_phase3_scoring_still_uses_safe_text_only(self) -> None:
        unsafe = unit(
            "SAFE",
            "sanitized body",
            text="sanitized body rawphase3token",
        )
        retriever = HierarchicalRetriever((unsafe,))
        self.assertEqual((), retriever.retrieve("rawphase3token", NOW))

    def test_phase3_ranking_is_deterministic_and_bounded(self) -> None:
        units = tuple(unit(str(index), f"shared token {index}") for index in range(30))
        retriever = HierarchicalRetriever(units)
        first = retriever.retrieve("shared token", NOW)
        second = retriever.retrieve("shared token", NOW)
        self.assertEqual(first, second)
        self.assertEqual(20, len(first))
        self.assertEqual(20, len(set(first)))


if __name__ == "__main__":
    unittest.main()

