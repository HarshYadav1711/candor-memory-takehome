from __future__ import annotations

import sys
import unittest
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candor_memory import MemoryUnit  # noqa: E402
from candor_memory.retrieval import (  # noqa: E402
    Intent,
    build_query_plan,
    derive_corpus_aliases,
    extract_explicit_dates,
    normalize_terms,
    resolve_relative_dates,
    tokenize,
)


AS_OF = datetime.fromisoformat("2026-10-05T12:00:00-07:00")


def synthetic_unit(
    safe_text: str,
    *,
    metadata: dict[str, object] | None = None,
) -> MemoryUnit:
    return MemoryUnit(
        unit_id="SYNTHETIC",
        record_id="SYNTHETIC",
        source_type="meeting",
        available_at=AS_OF,
        text=safe_text,
        safe_text=safe_text,
        metadata=metadata or {},
    )


class PlannerTests(unittest.TestCase):
    def test_detects_dictation_source_hint(self) -> None:
        plan = build_query_plan("What did I dictate yesterday?", AS_OF)
        self.assertIn("dictation", plan.source_hints)

    def test_detects_slack_email_and_calendar_source_hints(self) -> None:
        question = "Was the Slack message emailed and put on my calendar?"
        plan = build_query_plan(question, AS_OF)
        self.assertEqual(
            {"slack", "gmail", "google_calendar"},
            set(plan.source_hints),
        )

    def test_extracts_common_explicit_date_forms(self) -> None:
        found = extract_explicit_dates(
            "Sep 10, then 9/11/2026, then 2026-09-12", AS_OF
        )
        self.assertEqual(
            (date(2026, 9, 10), date(2026, 9, 11), date(2026, 9, 12)),
            found,
        )

    def test_relative_weekday_resolution_uses_as_of(self) -> None:
        resolved = dict(resolve_relative_dates("this Tuesday and next Tuesday", AS_OF))
        self.assertEqual(date(2026, 10, 6), resolved["this tuesday"])
        self.assertEqual(date(2026, 10, 13), resolved["next tuesday"])

    def test_last_tuesday_crosses_week_boundary(self) -> None:
        resolved = dict(resolve_relative_dates("last Tuesday", AS_OF))
        self.assertEqual(date(2026, 9, 29), resolved["last tuesday"])

    def test_relative_dates_use_as_of_not_wall_clock(self) -> None:
        utc_reference = datetime.fromisoformat("2031-01-01T01:00:00+00:00")
        resolved = dict(resolve_relative_dates("today", utc_reference))
        self.assertEqual(date(2030, 12, 31), resolved["today"])

    def test_person_and_attribution_detection(self) -> None:
        plan = build_query_plan("What did Morgan Reed say about access?", AS_OF)
        self.assertEqual(("Morgan Reed",), plan.speaker_hints)
        self.assertIn("Morgan Reed", plan.entity_hints)
        self.assertIn(Intent.ATTRIBUTION, plan.intent_flags)

    def test_causal_and_current_intents_can_coexist(self) -> None:
        plan = build_query_plan("Why is the date currently delayed?", AS_OF)
        self.assertIn(Intent.CAUSAL_CHANGE, plan.intent_flags)
        self.assertIn(Intent.CURRENT_STATE, plan.intent_flags)

    def test_inflection_normalization_is_generic(self) -> None:
        dictated = set(normalize_terms(tokenize("dictated routes")))
        dictation = set(normalize_terms(tokenize("dictation route")))
        self.assertIn("dictate", dictated & dictation)
        self.assertIn("route", dictated & dictation)

    def test_hyphen_and_space_normalization(self) -> None:
        self.assertEqual(tokenize("north-star"), tokenize("north star"))

    def test_corpus_derived_acronym_alias(self) -> None:
        unit = synthetic_unit(
            "PA kickoff is tomorrow.",
            metadata={"meeting_title": "Project Atlas"},
        )
        aliases = derive_corpus_aliases((unit,))
        self.assertEqual(("project", "atlas"), aliases["pa"])
        plan = build_query_plan("Status of Project Atlas?", AS_OF, aliases=aliases)
        self.assertIn("pa", plan.expanded_terms)

    def test_temporal_anchor_detection_separates_clauses(self) -> None:
        plan = build_query_plan(
            "What's on my calendar the day I visit Cedar City?", AS_OF
        )
        self.assertEqual("I visit Cedar City", plan.anchor_query)
        self.assertEqual("What's on my calendar", plan.target_query)
        self.assertIn(Intent.TEMPORAL_ANCHOR, plan.intent_flags)


if __name__ == "__main__":
    unittest.main()

