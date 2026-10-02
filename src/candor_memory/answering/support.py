"""Requested-slot support decisions over generic question frames and claims."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..retrieval.tokenize import tokenize
from .claims import Claim, ValueKind, build_claims
from .frames import AnswerKind, ModalityRequirement, QuestionFrame, parse_question_frame
from .models import EvidenceBundle, SupportDecision, SupportStatus


@dataclass(frozen=True)
class AlignedClaim:
    claim: Claim
    score: float
    subject_aligned: bool
    relation_aligned: bool
    value_compatible: bool
    modality_compatible: bool
    attribution_compatible: bool


@dataclass(frozen=True)
class EvidenceAnalysis:
    frame: QuestionFrame
    claims: tuple[Claim, ...]
    aligned: tuple[AlignedClaim, ...]
    failure_class: str | None
    conflicting: bool


_BROAD_PROPERTY_TERMS = {
    "amount", "answer", "city", "cost", "current", "date", "detail", "exact", "figure", "information",
    "number", "pric", "price", "pricing", "status", "tim", "time", "timing", "total", "value",
}


def _tokens(text: str) -> set[str]:
    return set(tokenize(text))


def _semantic_overlap(left: set[str], right: set[str]) -> int:
    matched: set[str] = set()
    for term in left:
        if term in right or any(
            len(term) >= 5 and len(other) >= 5 and (term.startswith(other) or other.startswith(term))
            for other in right
        ):
            matched.add(term)
    return len(matched)


def _subject_alignment(frame: QuestionFrame, claim: Claim, context_tokens: set[str]) -> tuple[bool, int]:
    haystack = set(claim.normalized_tokens)
    actor_tokens = _tokens(claim.actor or "")
    entity_hits = 0
    for entity in frame.named_entities:
        entity_tokens = _tokens(entity)
        if entity_tokens and entity_tokens <= (context_tokens | actor_tokens):
            entity_hits += 1
    entity_terms = {token for entity in frame.named_entities for token in _tokens(entity)}
    subject_overlap = _semantic_overlap(set(frame.subject_terms) - entity_terms, haystack)
    if frame.named_entities:
        strict_entities = [entity for entity in frame.named_entities if len(_tokens(entity)) >= 2]
        if strict_entities and not any(_tokens(entity) <= (context_tokens | actor_tokens) for entity in strict_entities):
            return False, subject_overlap
        relation_overlap = bool(set(frame.relation_terms) & set(claim.relation_terms))
        return entity_hits > 0 or subject_overlap >= 2 or (relation_overlap and subject_overlap >= 1), entity_hits * 4 + subject_overlap
    if set(frame.relation_terms) & set(claim.relation_terms):
        return True, subject_overlap
    if frame.subject_terms:
        return subject_overlap > 0, subject_overlap
    return True, 0


def _specific_property_supported(frame: QuestionFrame, claim: Claim, context_tokens: set[str]) -> bool:
    properties = {term for term in frame.property_terms if term not in _BROAD_PROPERTY_TERMS}
    if not properties or frame.answer_kind in {
        AnswerKind.OPEN, AnswerKind.LIST, AnswerKind.BOOLEAN, AnswerKind.STATUS,
        AnswerKind.REASON, AnswerKind.TEXT_VALUE,
    }:
        return True
    if re.search(r"\b(?:days?|hours?)\s+(?:after|before)\b", frame.question_text, re.I):
        return True
    if frame.multi_part and len(set(frame.subject_terms) & set(claim.normalized_tokens)) >= 2:
        return True
    return bool(properties & context_tokens)


def _relation_alignment(frame: QuestionFrame, claim: Claim, context_tokens: set[str]) -> bool:
    lowered_question = frame.request_text.casefold()
    lowered_claim = claim.text.casefold()
    if " after " in f" {lowered_question} " and " before " in f" {lowered_claim} ":
        return False
    if " before " in f" {lowered_question} " and " after " in f" {lowered_claim} ":
        return False
    requested = set(frame.relation_terms)
    available = set(claim.relation_terms)
    if requested and requested & available:
        return _specific_property_supported(frame, claim, context_tokens)
    if re.search(r"\b(?:days?|hours?)\s+(?:after|before)\b", frame.question_text, re.I):
        return True
    if not requested:
        return _specific_property_supported(frame, claim, context_tokens) and bool(
            set(frame.subject_terms) & set(claim.normalized_tokens)
        )
    return False


def _value_compatible(frame: QuestionFrame, claim: Claim) -> bool:
    kinds = {value.kind for value in claim.values}
    if frame.multi_part and set(frame.relation_terms) & set(claim.relation_terms) and ValueKind.TEXT in kinds:
        return True
    if frame.answer_kind == AnswerKind.IDENTITY:
        subject_entities = {frozenset(_tokens(entity)) for entity in frame.named_entities}
        people = [
            value.text
            for value in claim.values
            if value.kind == ValueKind.PERSON
            and not value.text.isupper()
            and not any(subject <= _tokens(value.text) for subject in subject_entities)
        ]
        if set(frame.relation_terms) & {"ownership", "attribution"} and claim.actor:
            return True
        properties = {term for term in frame.property_terms if term not in _BROAD_PROPERTY_TERMS}
        if properties and not (properties & set(claim.normalized_tokens)):
            return False
        return bool(people)
    expected = {
        AnswerKind.TIME: {ValueKind.TIME, ValueKind.DATETIME, ValueKind.DATE},
        AnswerKind.DATE: {ValueKind.DATE, ValueKind.DATETIME},
        AnswerKind.DATETIME: {ValueKind.DATE, ValueKind.TIME, ValueKind.DATETIME},
        AnswerKind.QUANTITY: {ValueKind.NUMBER},
        AnswerKind.NUMBER: {ValueKind.NUMBER, ValueKind.PERCENTAGE, ValueKind.MONEY},
        AnswerKind.MONEY: {ValueKind.MONEY},
        AnswerKind.LOCATION: {ValueKind.LOCATION},
        AnswerKind.REASON: {ValueKind.REASON},
        AnswerKind.TEXT_VALUE: {ValueKind.PHONE, ValueKind.EMAIL, ValueKind.TEXT},
        AnswerKind.STATUS: {ValueKind.STATUS, ValueKind.TEXT},
        AnswerKind.BOOLEAN: {ValueKind.STATUS, ValueKind.TEXT},
        AnswerKind.LIST: {ValueKind.TEXT},
        AnswerKind.OPEN: {ValueKind.TEXT},
    }[frame.answer_kind]
    if frame.answer_kind == AnswerKind.QUANTITY and re.search(r"\b(?:days?|hours?)\s+(?:after|before)\b", frame.question_text, re.I):
        return True
    if frame.answer_kind == AnswerKind.TEXT_VALUE:
        if "contact_phone" in frame.relation_terms:
            return ValueKind.PHONE in kinds
        if "contact_link" in frame.relation_terms:
            return bool(re.search(r"https?://", claim.text, re.I))
    return bool(kinds & expected)


def _modality_compatible(frame: QuestionFrame, claim: Claim) -> bool:
    requirement = frame.modality_requirement
    if requirement == ModalityRequirement.CONDITIONAL:
        return claim.conditional
    if requirement == ModalityRequirement.PROPOSED:
        return claim.proposed or claim.speculative
    if requirement == ModalityRequirement.COMPLETED:
        return claim.completed
    if requirement == ModalityRequirement.CANCELLED:
        return claim.cancelled
    if requirement == ModalityRequirement.ANY_STATUS:
        return claim.conditional or claim.proposed or claim.speculative or claim.completed or claim.cancelled
    if requirement in {ModalityRequirement.ASSERTED, ModalityRequirement.CURRENT}:
        if frame.answer_kind not in {AnswerKind.STATUS, AnswerKind.BOOLEAN} and (
            claim.conditional or claim.proposed or claim.speculative
        ):
            return False
    return True


def _attribution_compatible(frame: QuestionFrame, claim: Claim) -> bool:
    if frame.requested_actor_role == "direct":
        if not claim.actor or claim.reported_speech:
            return False
        if frame.named_entities:
            actor = _tokens(claim.actor)
            return any(_tokens(entity) <= actor for entity in frame.named_entities)
    if frame.requested_actor_role == "reporter":
        return claim.reported_speech and claim.actor is not None
    return True


def _is_conflict(frame: QuestionFrame, aligned: tuple[AlignedClaim, ...]) -> bool:
    if any(candidate.claim.corrected for candidate in aligned) and (
        frame.asks_current or "correct" in frame.question_text.casefold()
    ):
        return False
    actor_claims = [candidate.claim for candidate in aligned if candidate.claim.actor]
    if len({claim.actor for claim in actor_claims}) < 2:
        return False
    asks_comparison = bool(re.search(r"\b(?:agree|conflict|disagree|opposing|versus|vs\.?|while|both)\b", frame.question_text, re.I))
    polarities = {claim.negated for claim in actor_claims}
    return asks_comparison or len(polarities) > 1


def analyze_evidence(bundle: EvidenceBundle) -> EvidenceAnalysis:
    frame = parse_question_frame(bundle.question)
    claims = build_claims(bundle)
    source_tokens: dict[str, set[str]] = {}
    source_relations: dict[str, set[str]] = {}
    for claim in claims:
        source_tokens.setdefault(claim.source_id, set()).update(claim.normalized_tokens)
        source_relations.setdefault(claim.source_id, set()).update(claim.relation_terms)
    subject_matches: list[bool] = []
    relation_matches: list[bool] = []
    value_matches: list[bool] = []
    modality_matches: list[bool] = []
    attribution_matches: list[bool] = []
    aligned: list[AlignedClaim] = []
    for claim in claims:
        subject_ok, subject_score = _subject_alignment(frame, claim, source_tokens[claim.source_id])
        relation_ok = subject_ok and _relation_alignment(frame, claim, source_tokens[claim.source_id])
        value_ok = relation_ok and _value_compatible(frame, claim)
        modality_ok = value_ok and _modality_compatible(frame, claim)
        attribution_ok = modality_ok and _attribution_compatible(frame, claim)
        subject_matches.append(subject_ok)
        relation_matches.append(relation_ok)
        value_matches.append(value_ok)
        modality_matches.append(modality_ok)
        attribution_matches.append(attribution_ok)
        if attribution_ok:
            structured = any(value.structured for value in claim.values)
            mismatched_event_subtype = (
                claim.source_type == "google_calendar"
                and bool(re.search(r"\b(?:planning|prep|preparation)\b", claim.text, re.I))
                and not bool(re.search(r"\b(?:planning|prep|preparation)\b", frame.question_text, re.I))
            )
            score = (
                subject_score * 3.0
                + len(set(frame.relation_terms) & set(claim.relation_terms)) * 4.0
                + (10.0 if claim.corrected else 0.0)
                + (15.0 if frame.relation_terms and set(frame.relation_terms) <= source_relations[claim.source_id] else 0.0)
                + (20.0 if frame.answer_kind in {AnswerKind.STATUS, AnswerKind.BOOLEAN} and (claim.completed or claim.cancelled) else 0.0)
                + (8.0 if structured and not mismatched_event_subtype else -4.0 if mismatched_event_subtype else 0.0)
                + 2.0 / claim.retrieval_rank
            )
            aligned.append(AlignedClaim(claim, score, subject_ok, relation_ok, value_ok, modality_ok, attribution_ok))
    aligned.sort(
        key=lambda candidate: (
            -int(candidate.claim.corrected) if frame.asks_current else -candidate.score,
            -candidate.claim.available_at.timestamp() if frame.asks_current else -int(candidate.claim.corrected),
            -candidate.score if frame.asks_current else candidate.claim.retrieval_rank,
            candidate.claim.retrieval_rank,
        )
    )
    failure = None
    if not claims:
        failure = "value_extraction_failure"
    elif not any(subject_matches):
        failure = "subject_alignment_failure"
    elif not any(relation_matches):
        failure = "relation_alignment_failure"
    elif not any(value_matches):
        failure = "value_extraction_failure"
    elif not any(modality_matches):
        failure = "modality_failure"
    elif not any(attribution_matches):
        failure = "attribution_failure"
    result = tuple(aligned)
    return EvidenceAnalysis(frame, claims, result, failure, _is_conflict(frame, result))


def decide_support(
    bundle: EvidenceBundle,
    analysis: EvidenceAnalysis | None = None,
) -> SupportDecision:
    """Require aligned subject, relation, and answer-kind-compatible value."""

    analysis = analysis or analyze_evidence(bundle)
    signals: dict[str, object] = {
        "answer_kind": analysis.frame.answer_kind.value,
        "claims": len(analysis.claims),
        "aligned_claims": len(analysis.aligned),
        "relations": list(analysis.frame.relation_terms),
        "failure_class": analysis.failure_class,
        "conflicting": analysis.conflicting,
    }
    if not analysis.aligned:
        return SupportDecision(
            SupportStatus.INSUFFICIENT_EVIDENCE,
            "visible evidence does not support the requested slot",
            signals,
        )
    if analysis.conflicting:
        return SupportDecision(
            SupportStatus.AMBIGUOUS,
            "aligned claims conflict and should be presented together",
            signals,
        )
    return SupportDecision(SupportStatus.SUPPORTED, "requested slot is supported", signals)
