"""Generic immutable question frames for offline grounded answering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..retrieval.planner import informative_terms, normalize_terms
from ..retrieval.tokenize import tokenize


class AnswerKind(str, Enum):
    IDENTITY = "IDENTITY"
    TIME = "TIME"
    DATE = "DATE"
    DATETIME = "DATETIME"
    QUANTITY = "QUANTITY"
    NUMBER = "NUMBER"
    MONEY = "MONEY"
    BOOLEAN = "BOOLEAN"
    STATUS = "STATUS"
    LOCATION = "LOCATION"
    REASON = "REASON"
    TEXT_VALUE = "TEXT_VALUE"
    LIST = "LIST"
    OPEN = "OPEN"


class ModalityRequirement(str, Enum):
    ASSERTED = "ASSERTED"
    ANY_STATUS = "ANY_STATUS"
    CONDITIONAL = "CONDITIONAL"
    PROPOSED = "PROPOSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"


@dataclass(frozen=True)
class QuestionFrame:
    question_text: str
    request_text: str
    answer_kind: AnswerKind
    subject_terms: tuple[str, ...]
    relation_terms: tuple[str, ...]
    property_terms: tuple[str, ...]
    named_entities: tuple[str, ...]
    requested_actor_role: str | None
    modality_requirement: ModalityRequirement
    polarity: bool | None
    asks_current: bool
    asks_historical: bool
    multi_part: bool


_RELATION_CUES: dict[str, tuple[str, ...]] = {
    "approval": ("approve", "approved", "approval", "sign off", "authorized"),
    "attribution": ("say", "said", "tell", "told", "report", "reported", "according", "position", "explain"),
    "commitment": ("promise", "promised", "commit", "committed", "owe", "due", "sign", "signed", "signing", "close", "closes", "closing"),
    "choice": ("choose", "chose", "chosen", "choice", "pick", "picked", "select", "selected", "decision", "decided", "went with"),
    "condition": ("condition", "conditional", "if", "unless", "depend", "depending", "only if", "provided"),
    "contact_phone": ("phone", "mobile", "telephone", "cell", "number"),
    "contact_link": ("url", "link", "video-call", "video call", "meeting link"),
    "delivery": ("deliver", "delivered", "send", "sent", "posted", "went out", "land", "landed"),
    "location": ("where", "location", "address", "city", "hotel", "venue"),
    "money": ("salary", "compensation", "budget", "cost", "price", "pricing", "quote", "quoted", "arr", "revenue", "invoice", "expense", "total", "fee", "commercial terms"),
    "outcome": ("outcome", "goal", "result", "meant to", "deliver for", "benefit"),
    "ownership": ("own", "owns", "owner", "belong", "belongs", "responsible", "responsibility", "doing", "on my plate"),
    "quantity": ("how many", "count", "quantity", "pages", "number of", "latency", "throughput", "rate", "metric", "measurement", "p95"),
    "reason": ("why", "reason", "cause", "caused", "root cause", "because", "due to", "drove", "rationale", "explain"),
    "sequence": ("after", "before", "next step", "came next", "followed"),
    "status": ("status", "state", "current", "confirmed", "complete", "completed", "done", "open", "pending", "cancelled", "canceled", "approved", "renewal", "whether"),
    "time": ("when", "date", "day", "time", "timing", "deadline", "birthday", "start", "starts", "begin", "begins", "depart", "departure", "scheduled", "follow-up"),
    "validation": ("check", "checks", "validate", "validation", "quality"),
}

_QUESTION_PREFIXES = {
    "after", "are", "before", "can", "connect", "could", "do", "does", "did", "during", "for", "give", "has", "have", "how", "i", "is", "me",
    "please", "recall", "remind", "state", "tell", "the", "there", "was", "we", "were", "what",
    "summarize", "under", "when", "where", "which", "who", "whom", "whose", "why", "will", "would", "you",
}
_GENERIC_PROPERTY = {
    "answer", "detail", "information", "name", "person", "thing", "value", "what", "which", "who",
}


def _request_clause(question: str) -> str:
    clauses = [part.strip() for part in re.split(r"[;]", question) if part.strip()]
    for clause in reversed(clauses):
        if re.search(r"\b(what|when|where|which|who|whom|whose|why|how|is|was|did|does|has|will)\b", clause, re.I):
            return clause
    return question.strip()


def _relations(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    found = [
        relation
        for relation, cues in _RELATION_CUES.items()
        if any(re.search(rf"\b{re.escape(cue)}\b", lowered) for cue in cues)
    ]
    return tuple(dict.fromkeys(found))


def _entities(question: str) -> tuple[str, ...]:
    found: list[str] = []
    date_words = {
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
    }
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", question):
        parts = match.group(0).split()
        if parts and parts[0].casefold() in _QUESTION_PREFIXES:
            parts = parts[1:]
        value = " ".join(parts)
        if not value or value.casefold() in date_words:
            continue
        if value not in found:
            found.append(value)
    for match in re.finditer(r"\b[A-Z][A-Z0-9]{1,7}\b", question):
        value = match.group(0)
        prefix = question[max(0, match.start() - 12) : match.start()].casefold()
        if re.search(r"(?:'s|s'|name of)\s*$", prefix):
            continue
        if value not in found:
            found.append(value)
    return tuple(found)


def _property_terms(request: str) -> tuple[str, ...]:
    lowered = request.casefold().replace("’", "'")
    candidates: list[str] = []
    possessive = re.search(r"(?:[a-z][a-z -]+)'s?\s+([a-z][a-z -]+?)(?:\?|\bis\b|\bwas\b|\bdoes\b|\bdid\b|$)", lowered)
    if possessive:
        candidates.extend(tokenize(possessive.group(1)))
    lead = re.search(
        r"\b(?:what|which)\s+(?:is|was|are|were|did|does|do|has|have|the|exact|annual|current|corrected|total|promised|presently|agreed|commercial\s+terms\s+)?([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2})",
        lowered,
    )
    if lead:
        candidates.extend(tokenize(lead.group(1)))
    quantity = re.search(r"\bhow\s+many\s+([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2})", lowered)
    if quantity:
        candidates.extend(tokenize(quantity.group(1)))
    values = [term for term in normalize_terms(candidates) if term not in _QUESTION_PREFIXES and term not in _GENERIC_PROPERTY]
    return tuple(dict.fromkeys(values))


def _kind(request: str, relations: tuple[str, ...]) -> AnswerKind:
    lowered = request.casefold()
    relation_set = set(relations)
    if "reason" in relation_set:
        kind = AnswerKind.REASON
    elif "quantity" in relation_set:
        kind = AnswerKind.QUANTITY
    elif "contact_phone" in relation_set or "contact_link" in relation_set:
        kind = AnswerKind.TEXT_VALUE
    elif "location" in relation_set:
        kind = AnswerKind.LOCATION
    elif "time" in relation_set:
        has_clock = bool(re.search(r"\b(?:clock|time|begins?|starts?|depart)\b", lowered))
        has_date = bool(re.search(r"\b(?:date|day|deadline)\b", lowered))
        kind = AnswerKind.DATETIME if has_clock and has_date else AnswerKind.DATE if has_date else AnswerKind.TIME
    elif re.match(r"\s*(?:are|can|did|do|does|has|have|is|was|were|will|would)\b", lowered):
        kind = AnswerKind.BOOLEAN
    elif "money" in relation_set:
        kind = AnswerKind.MONEY
    elif re.search(r"\b(?:percentage|percent|figure|measurement|metric|p\d{2}|nrr|target)\b", lowered):
        kind = AnswerKind.NUMBER
    elif re.search(r"\b(?:who|whom|whose)\b|\bname\s+of\b|\bwhich\s+[a-z-]+\s+(?:will|did|does|is|was)\b", lowered) or "ownership" in relation_set:
        kind = AnswerKind.IDENTITY
    elif "status" in relation_set or "condition" in relation_set or "delivery" in relation_set:
        kind = AnswerKind.STATUS
    else:
        kind = AnswerKind.OPEN
    if kind == AnswerKind.OPEN and (
        re.search(r"\b(?:both|two|each)\b", lowered) or (" and " in lowered and len(relation_set) > 1)
    ):
        return AnswerKind.LIST
    return kind


def _modality(request: str) -> ModalityRequirement:
    lowered = request.casefold()
    states = sum(bool(re.search(pattern, lowered)) for pattern in (r"\bpropos", r"\bcondition", r"\bapprov|\bconfirm"))
    if states >= 2:
        return ModalityRequirement.ANY_STATUS
    if re.search(r"\bunder what condition\b|\bconditional\b|\bon what condition\b", lowered):
        return ModalityRequirement.CONDITIONAL
    if re.search(r"\b(?:was|is|merely|only)\s+proposed\b|\bproposed\s+(?:option|plan|change)\b|\bsuggest(?:ed|ion)\b", lowered):
        return ModalityRequirement.PROPOSED
    if re.search(r"\bcompleted?\b|\bdone\b|\bfulfilled\b|\blanded\b|\bposted\b", lowered):
        return ModalityRequirement.COMPLETED
    if re.search(r"\bcancell?ed\b|\bcalled off\b|\bscratched\b", lowered):
        return ModalityRequirement.CANCELLED
    if re.search(
        r"\bcurrent(?:ly)?\b|\bpresently\b|\bnow\b|\bstill\b|\bagreed\b|\bconfirmed\b|\bdecided\b"
        r"|\bwhen\s+(?:is|are|do|does|will)\b",
        lowered,
    ):
        return ModalityRequirement.CURRENT
    if re.search(r"\bbefore\b|\bafter\b|\bas things stood\b|\bpreviously\b|\bthen\b", lowered):
        return ModalityRequirement.HISTORICAL
    return ModalityRequirement.ASSERTED


def parse_question_frame(question: str) -> QuestionFrame:
    request = _request_clause(question)
    relations = _relations(request)
    entities = _entities(question)
    properties = _property_terms(request)
    cue_words = {
        token
        for cues in _RELATION_CUES.values()
        for cue in cues
        for token in tokenize(cue)
    }
    entity_words = {token for entity in entities for token in tokenize(entity)}
    properties = tuple(term for term in properties if term not in entity_words)
    subject = [
        term
        for term in normalize_terms(informative_terms(tokenize(question)))
        if term not in cue_words
        and term not in _QUESTION_PREFIXES
        and term not in properties
    ]
    # Entity tokens are useful subject anchors even when the surrounding words are generic.
    subject.extend(token for token in entity_words if token not in subject)
    lowered = question.casefold()
    actor_role = None
    if re.search(r"\bpersonally\b|\bdirectly\b|\bown words\b", lowered):
        actor_role = "direct"
    elif re.search(r"\b(?:cause|reason|explanation)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:gave|explained|stated)\b", question, re.I):
        actor_role = "direct"
    elif re.search(r"\bwhat did\s+.+?\s+(?:say|tell)\s+.+?\s+(?:said|told)\b", lowered):
        actor_role = "reporter"
    elif re.search(r"\bdid\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:agree|say|tell|promise|commit|give|explain|state)\b", question, re.I):
        actor_role = "direct"
    elif re.search(r"\bwho said\b.+\bsaid\b|\breported\b|\baccording to\b", lowered):
        actor_role = "reporter"
    modality = _modality(request)
    polarity = False if re.search(r"\bnot\b|\bnever\b|\bwon't\b|\bunlikely\b", lowered) else None
    return QuestionFrame(
        question_text=question,
        request_text=request,
        answer_kind=_kind(request, relations),
        subject_terms=tuple(dict.fromkeys(subject)),
        relation_terms=relations,
        property_terms=properties,
        named_entities=entities,
        requested_actor_role=actor_role,
        modality_requirement=modality,
        polarity=polarity,
        asks_current=modality == ModalityRequirement.CURRENT,
        asks_historical=modality == ModalityRequirement.HISTORICAL,
        multi_part=(
            " and " in request.casefold()
            or bool(re.search(r"\b(?:both|connect|each|two)\b", request, re.I))
        ),
    )


def relation_cues() -> dict[str, tuple[str, ...]]:
    """Return the shared generic cue families used by question and claim parsing."""

    return _RELATION_CUES
