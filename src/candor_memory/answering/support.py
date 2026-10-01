"""Deterministic multi-signal support and abstention decisions."""

from __future__ import annotations

import re

from ..retrieval import Intent
from ..retrieval.planner import informative_terms, normalize_terms
from ..retrieval.tokenize import tokenize
from .models import EvidenceBundle, SupportDecision, SupportStatus


_MECHANIC_TERMS = frozenset(
    {
        "about", "after", "before", "current", "currently", "day", "days",
        "did", "do", "does", "done", "going", "many", "now", "owe", "say",
        "send", "should", "still", "tell", "then", "they", "want", "when",
        "why", "who",
    }
)


def _terms(text: str) -> set[str]:
    return set(normalize_terms(informative_terms(tokenize(text))))


def _focus_terms(question: str) -> set[str]:
    lowered = question.casefold().strip()
    focus = ""
    about = re.search(r"\babout\s+(.+?)[?.!]*$", lowered)
    possessive = re.search(r"\bwhat(?:'s| is)\s+.+?'s\s+([a-z0-9 -]+?)[?.!]*$", lowered)
    if about:
        focus = about.group(1)
    elif possessive:
        focus = possessive.group(1)
    return {term for term in _terms(focus) if term not in _MECHANIC_TERMS}


def decide_support(bundle: EvidenceBundle) -> SupportDecision:
    """Use coverage, focus, metadata, and evidence independence to gate answering."""

    if not bundle.items:
        return SupportDecision(
            SupportStatus.INSUFFICIENT_EVIDENCE,
            "no visible retrieved evidence",
            {"evidence_units": 0},
        )
    query_terms = _terms(bundle.question)
    documents = [_terms(item.safe_text) for item in bundle.items]
    union = set().union(*documents)
    best_coverage = max(
        (len(query_terms & document) / len(query_terms) for document in documents),
        default=0.0,
    ) if query_terms else 0.0
    union_coverage = len(query_terms & union) / len(query_terms) if query_terms else 0.0
    focus = _focus_terms(bundle.question)
    focus_coverage = len(focus & union) / len(focus) if focus else 1.0
    source_match = any(item.source_type in bundle.plan.source_hints for item in bundle.items)
    entity_match = any(
        set(tokenize(entity)) <= set(tokenize(item.safe_text + " " + (item.actor or "")))
        for entity in bundle.plan.entity_hints
        for item in bundle.items
    )
    question_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", bundle.question))
    number_match = bool(question_numbers & set(re.findall(r"\b\d+(?:\.\d+)?\b", " ".join(item.safe_text for item in bundle.items))))
    independent_records = len({item.record_id for item in bundle.items if _terms(item.safe_text) & query_terms})
    independent_actors = len({item.actor for item in bundle.items if item.actor and _terms(item.safe_text) & query_terms})
    signals: dict[str, float | int | bool] = {
        "evidence_units": len(bundle.items),
        "best_coverage": round(best_coverage, 4),
        "union_coverage": round(union_coverage, 4),
        "focus_coverage": round(focus_coverage, 4),
        "source_match": source_match,
        "entity_match": entity_match,
        "number_match": number_match,
        "independent_records": independent_records,
        "independent_actors": independent_actors,
    }
    if focus and focus_coverage == 0.0:
        return SupportDecision(
            SupportStatus.INSUFFICIENT_EVIDENCE,
            "the requested fact is absent from the evidence",
            signals,
        )
    if best_coverage < 0.25 and union_coverage < 0.5:
        return SupportDecision(
            SupportStatus.INSUFFICIENT_EVIDENCE,
            "retrieved evidence has insufficient lexical support",
            signals,
        )
    if best_coverage < 0.35 and not (source_match or entity_match or number_match):
        return SupportDecision(
            SupportStatus.INSUFFICIENT_EVIDENCE,
            "no strong exact, source, entity, or numeric support",
            signals,
        )
    comparison = bool(
        {Intent.DISAGREEMENT, Intent.ATTRIBUTION} & set(bundle.plan.intent_flags)
    )
    if comparison and independent_records >= 2 and independent_actors >= 2:
        return SupportDecision(
            SupportStatus.AMBIGUOUS,
            "multiple independently attributed records should be preserved",
            signals,
        )
    return SupportDecision(SupportStatus.SUPPORTED, "evidence is sufficient", signals)

