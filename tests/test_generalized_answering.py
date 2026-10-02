from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candor_memory import GroundedMemory, MemoryUnit, build_query_plan  # noqa: E402
from candor_memory.answering import (  # noqa: E402
    AnswerKind,
    SupportStatus,
    build_evidence_bundle,
    decide_support,
    parse_question_frame,
    write_extractive_answer,
)


NOW = datetime.fromisoformat("2026-06-10T12:00:00-07:00")


def unit(
    unit_id: str,
    safe_text: str,
    *,
    actor: str | None = None,
    available_at: datetime = NOW,
    source_type: str = "slack",
    metadata: dict[str, object] | None = None,
    text: str | None = None,
) -> MemoryUnit:
    return MemoryUnit(
        unit_id=unit_id,
        record_id=unit_id,
        source_type=source_type,
        available_at=available_at,
        text=safe_text if text is None else text,
        safe_text=safe_text,
        actor=actor,
        metadata=metadata or {},
    )


def evidence(question: str, *units: MemoryUnit, as_of: datetime = NOW):
    return build_evidence_bundle(
        units,
        (item.unit_id for item in units),
        question,
        as_of,
        build_query_plan(question, as_of),
    )


def answer_from(question: str, *units: MemoryUnit, as_of: datetime = NOW):
    bundle = evidence(question, *units, as_of=as_of)
    if decide_support(bundle).status == SupportStatus.INSUFFICIENT_EVIDENCE:
        return "I don't know.", (), True
    draft = write_extractive_answer(bundle)
    return draft.answer, draft.source_ids, draft.answer == "I don't know."


class QuestionFrameTests(unittest.TestCase):
    def test_active_and_passive_ownership_frames_match(self) -> None:
        active = parse_question_frame("Who owns Project Juniper?")
        passive = parse_question_frame("Project Juniper belongs to whom?")
        self.assertEqual(AnswerKind.IDENTITY, active.answer_kind)
        self.assertEqual(active.answer_kind, passive.answer_kind)
        self.assertIn("ownership", active.relation_terms)
        self.assertIn("ownership", passive.relation_terms)

    def test_alternate_when_phrasing_is_temporal(self) -> None:
        frames = [
            parse_question_frame("When is the Juniper review?"),
            parse_question_frame("What date is the Juniper review?"),
            parse_question_frame("Remind me when the Juniper review happens."),
        ]
        self.assertTrue(all(frame.answer_kind in {AnswerKind.TIME, AnswerKind.DATE, AnswerKind.DATETIME} for frame in frames))
        self.assertTrue(all("time" in frame.relation_terms for frame in frames))


class RequestedSlotTests(unittest.TestCase):
    def assert_abstains(self, question: str, *units: MemoryUnit) -> None:
        answer, sources, abstained = answer_from(question, *units)
        self.assertTrue(abstained)
        self.assertEqual("I don't know.", answer)
        self.assertEqual((), sources)

    def test_salary_near_miss_abstains(self) -> None:
        self.assert_abstains("What is Rowan's salary?", unit("U", "Rowan owns the launch checklist."))

    def test_phone_near_miss_abstains(self) -> None:
        self.assert_abstains("What is Rowan's phone number?", unit("U", "Rowan approved the launch checklist."))

    def test_unsupported_reason_abstains_despite_date(self) -> None:
        self.assert_abstains("Why did the launch move?", unit("U", "The launch moved to June 18."))

    def test_numeric_question_requires_numeric_relation(self) -> None:
        self.assert_abstains("What budget was allocated to Juniper?", unit("U", "Juniper launches in 12 days."))

    def test_identity_question_rejects_unrelated_person(self) -> None:
        self.assert_abstains("Who owns Project Juniper?", unit("U", "Mira said Project Juniper launches Friday.", actor="Mira Vale"))

    def test_reported_speech_is_not_direct_speech(self) -> None:
        self.assert_abstains("Did Eli personally say the release was approved?", unit("U", "Eli said the release was approved.", actor="Mara West"))

    def test_proposal_is_not_a_decision(self) -> None:
        self.assert_abstains("What date was decided for the Juniper launch?", unit("U", "We proposed June 18 for the Juniper launch."))

    def test_conditional_is_not_a_confirmed_fact(self) -> None:
        self.assert_abstains("When will Juniper launch?", unit("U", "If legal approves, Juniper launches June 18."))

    def test_topical_overlap_without_relation_support_abstains(self) -> None:
        self.assert_abstains("What hotel was booked for the Juniper offsite?", unit("U", "The Juniper offsite is in Denver on June 18."))

    def test_multiple_unrelated_records_do_not_create_support(self) -> None:
        self.assert_abstains(
            "What is Rowan's salary?",
            unit("A", "Rowan owns Project Juniper."),
            unit("B", "Rowan attends the review Friday."),
            unit("C", "Rowan approved the checklist."),
        )

    def test_no_candidate_uses_exact_abstention_contract(self) -> None:
        answer, _ = GroundedMemory((unit("U", "Juniper launches Friday."),)).answer("What is Rowan's salary?", NOW)
        self.assertEqual("I don't know.", answer.answer)
        self.assertEqual((), answer.sources)
        self.assertTrue(answer.abstained)


class GenericCompositionTests(unittest.TestCase):
    def test_conditional_answers_condition_question(self) -> None:
        answer, _, abstained = answer_from(
            "Under what condition will Juniper launch?",
            unit("U", "Juniper launches June 18 only if legal approves."),
        )
        self.assertFalse(abstained)
        self.assertIn("only if", answer)

    def test_cancelled_commitment_is_not_open(self) -> None:
        answer, _, _ = answer_from(
            "Do we still owe Mira the Juniper draft?",
            unit("P", "We promised Mira the Juniper draft.", available_at=NOW - timedelta(days=2)),
            unit("C", "Scratch the Juniper draft; Mira no longer needs it.", available_at=NOW - timedelta(days=1)),
        )
        self.assertTrue(answer.startswith("No"))

    def test_completed_commitment_answers_status(self) -> None:
        answer, _, _ = answer_from("Was the Juniper report delivered?", unit("U", "The Juniper report was delivered this morning."))
        self.assertTrue(answer.startswith("Yes"))

    def test_conflicting_claims_preserve_both_speakers(self) -> None:
        answer, sources, _ = answer_from(
            "Did Mira Vale and Noor Reed agree about the launch?",
            unit("A", "The launch should proceed Friday.", actor="Mira Vale"),
            unit("B", "The launch should not proceed Friday.", actor="Noor Reed"),
        )
        self.assertIn("Mira Vale", answer)
        self.assertIn("Noor Reed", answer)
        self.assertEqual({"A", "B"}, set(sources))

    def test_corrected_value_wins(self) -> None:
        answer, _, _ = answer_from("What's the p95 latency?", unit("U", "p95 was 800 ms—sorry, p95 is 1.8 seconds."))
        self.assertIn("1.8", answer)
        self.assertNotIn("800", answer)

    def test_historical_state_cannot_use_later_unit(self) -> None:
        units = (
            unit("OLD", "Juniper launches June 18.", available_at=NOW - timedelta(days=1)),
            unit("NEW", "Juniper launches June 25.", available_at=NOW + timedelta(seconds=1)),
        )
        answer, _ = GroundedMemory(units).answer("When will Juniper launch?", NOW)
        self.assertIn("June 18", answer.answer)
        self.assertNotIn("NEW", answer.sources)

    def test_reason_question_uses_explanatory_clause(self) -> None:
        answer, _, _ = answer_from("Why did Juniper slip?", unit("U", "Juniper slipped because the vendor feed was late."))
        self.assertIn("vendor feed", answer)

    def test_structured_date_is_preferred(self) -> None:
        answer, _, _ = answer_from(
            "When is the Juniper review?",
            unit(
                "CAL",
                "Juniper review When: June 12.",
                source_type="google_calendar",
                metadata={"start": {"dateTime": "2026-06-12T09:30:00-07:00"}},
            ),
        )
        self.assertIn("Friday, June 12, 2026 at 9:30 AM", answer)

    def test_sources_are_exactly_used_claims(self) -> None:
        _, sources, _ = answer_from("Why did Juniper slip?", unit("USED", "Juniper slipped because the vendor feed was late."), unit("NOISE", "Juniper has a green logo."))
        self.assertEqual(("USED",), sources)

    def test_answer_uses_safe_text_only(self) -> None:
        secret = "raw-secret-value"
        answer, _, _ = answer_from("What is the Juniper status?", unit("U", "Juniper status is green.", text=f"Juniper status {secret}."))
        self.assertNotIn(secret, answer)

    def test_repeated_answers_are_deterministic(self) -> None:
        memory = GroundedMemory((unit("U", "Juniper status is green."),))
        first = memory.answer("What is the Juniper status?", NOW)[0]
        second = memory.answer("What is the Juniper status?", NOW)[0]
        self.assertEqual(first, second)

    def test_answer_is_at_most_120_words(self) -> None:
        answer, _, _ = answer_from("Why is Juniper delayed?", unit("U", "Juniper is delayed because " + "detail " * 200))
        self.assertLessEqual(len(answer.split()), 120)


if __name__ == "__main__":
    unittest.main()
