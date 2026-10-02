"""Clause-level claims and reusable value extraction from sanitized evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

from ..ingest import parse_datetime
from ..retrieval.planner import extract_explicit_dates, informative_terms, normalize_terms
from ..retrieval.tokenize import tokenize
from .frames import relation_cues
from .models import EvidenceBundle, EvidenceItem


class ValueKind(str, Enum):
    PERSON = "PERSON"
    DATE = "DATE"
    TIME = "TIME"
    DATETIME = "DATETIME"
    NUMBER = "NUMBER"
    PERCENTAGE = "PERCENTAGE"
    MONEY = "MONEY"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    LOCATION = "LOCATION"
    STATUS = "STATUS"
    REASON = "REASON"
    TEXT = "TEXT"


@dataclass(frozen=True)
class ExtractedValue:
    kind: ValueKind
    text: str
    structured: bool = False


@dataclass(frozen=True)
class Claim:
    source_id: str
    source_type: str
    record_id: str
    actor: str | None
    text: str
    normalized_tokens: tuple[str, ...]
    entities: tuple[str, ...]
    values: tuple[ExtractedValue, ...]
    relation_terms: tuple[str, ...]
    available_at: datetime
    retrieval_rank: int
    negated: bool
    conditional: bool
    speculative: bool
    proposed: bool
    committed: bool
    completed: bool
    cancelled: bool
    corrected: bool
    reported_speech: bool
    reported_speaker: str | None


_STATUS_RE = re.compile(
    r"\b(?:approved|blocked|cancelled|canceled|closed|complete|completed|confirmed|done|"
    r"delivered|landed|open|passing|pending|posted|received|renewed|reviewing|sent|signed|"
    r"scheduled|went out|will get back|not yet|still owe)\b",
    re.I,
)
_TIME_RELATION_RE = re.compile(
    r"\b(?:at|by|date|deadline|depart|departure|due|follow-up|from|launch|meet|meeting|"
    r"on|rescheduled|scheduled|start|starts|timing|until|when)\b",
    re.I,
)
_CONDITIONAL_RE = re.compile(r"\b(?:if|only if|unless|provided that|depending on|conditional on)\b", re.I)
_SPECULATIVE_RE = re.compile(r"\b(?:maybe|might|possibly|probably|unlikely|expect|think)\b", re.I)
_PROPOSED_RE = re.compile(r"\b(?:propose|proposed|suggest(?:ed|ion)|option|recommend(?:ed|ation))\b", re.I)
_COMMITTED_RE = re.compile(r"\b(?:promise[ds]?|commit(?:ted)?|will|due|owe|agreed to)\b", re.I)
_COMPLETED_RE = re.compile(r"\b(?:sent|went out|received|delivered|done|completed|landed|is in|are up|posted|signed)\b", re.I)
_CANCELLED_RE = re.compile(r"\b(?:cancelled|canceled|scratch(?:ed)?|no need|called off|won't|will not|discarded)\b", re.I)
_CORRECTED_RE = re.compile(r"\b(?:actually|correct(?:ed|ion)?|instead|now|sorry|updated?)\b|,\s*not\s+\d", re.I)
_NEGATED_RE = re.compile(r"\b(?:no|not|never|don't|do not|won't|will not|didn't|did not|unlikely)\b", re.I)
_REPORTED_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:said|told|reported|explained|wanted|thought)\b"
)


def _clean(text: str) -> str:
    text = re.split(r"\bRaw transcript:\s*", text, maxsplit=1, flags=re.I)[0]
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\s+", " ", text).strip(" -\n")


def _clauses(item: EvidenceItem) -> tuple[str, ...]:
    clean = re.split(r"\bRaw transcript:\s*", item.safe_text, maxsplit=1, flags=re.I)[0]
    pieces = re.split(r"(?:\r?\n){2,}|(?<=[.!?])\s+|\s+[•*]\s+|\s*[;—]\s*", clean)
    clauses: list[str] = []
    for piece in pieces:
        value = _clean(piece)
        if value and value not in clauses:
            clauses.append(value)
    return tuple(clauses)


def _metadata_datetime(value: object) -> datetime | None:
    if isinstance(value, dict):
        value = value.get("dateTime") or value.get("date")
    if not isinstance(value, str):
        return None
    try:
        return parse_datetime(value) if "T" in value else datetime.fromisoformat(value + "T00:00:00+00:00")
    except ValueError:
        return None


def _entities(text: str, actor: str | None) -> tuple[str, ...]:
    found: list[str] = [actor] if actor else []
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text):
        value = match.group(0)
        if value not in found and value.casefold() not in {"i", "the", "when", "status", "summary"}:
            found.append(value)
    for match in re.finditer(r"\b[A-Z][A-Z0-9]{1,7}\b", text):
        if match.group(0) not in found:
            found.append(match.group(0))
    return tuple(found)


def _values(text: str, item: EvidenceItem) -> tuple[ExtractedValue, ...]:
    values: list[ExtractedValue] = []
    for field in ("start", "end"):
        value = _metadata_datetime(item.metadata.get(field))
        if value:
            values.append(ExtractedValue(ValueKind.DATETIME, value.isoformat(), True))
    location = item.metadata.get("location")
    if isinstance(location, str) and location.strip():
        values.append(ExtractedValue(ValueKind.LOCATION, location.strip(), True))
    delivery_state = item.metadata.get("delivery_state")
    if isinstance(delivery_state, str) and delivery_state.strip():
        values.append(ExtractedValue(ValueKind.STATUS, delivery_state.replace("_", " "), True))
    for value in extract_explicit_dates(text, item.available_at):
        values.append(ExtractedValue(ValueKind.DATE, value.isoformat()))
    for match in re.finditer(
        r"\b(?:(?:early|mid|late)\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)|(?:after|before)\s+[a-z][a-z -]{1,30})\b",
        text,
        re.I,
    ):
        values.append(ExtractedValue(ValueKind.DATE, match.group(0)))
    for match in re.finditer(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\s*(?:[ap]\.?m\.?)?|\b\d{1,2}\s*(?:[ap]\.?m\.?)\b", text, re.I):
        values.append(ExtractedValue(ValueKind.TIME, match.group(0)))
    for match in re.finditer(r"(?<!\w)(?:[$€£]\s*\d[\d,]*(?:\.\d{1,2})?|\d[\d,]*(?:\.\d{1,2})?\s*(?:dollars?|usd))\b", text, re.I):
        values.append(ExtractedValue(ValueKind.MONEY, match.group(0)))
    for match in re.finditer(r"\b\d+(?:\.\d+)?\s*%", text):
        values.append(ExtractedValue(ValueKind.PERCENTAGE, match.group(0)))
    for match in re.finditer(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I):
        values.append(ExtractedValue(ValueKind.EMAIL, match.group(0)))
    for match in re.finditer(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)", text):
        values.append(ExtractedValue(ValueKind.PHONE, match.group(0)))
    for match in re.finditer(r"\b(?:at|in|venue:?|location:?)\s+([A-Z][A-Za-z0-9&' -]{2,40}?)(?=\s+(?:on|from|for|with|to)\b|[,.;]|$)", text):
        values.append(ExtractedValue(ValueKind.LOCATION, match.group(1).strip()))
    occupied = {value.text for value in values}
    for match in re.finditer(r"(?<![\w$])\d+(?:\.\d+)?(?:\s*(?:days?|hours?|users?|vehicles?|cases?|pages?))?", text, re.I):
        if match.group(0) not in occupied:
            values.append(ExtractedValue(ValueKind.NUMBER, match.group(0)))
    people: list[str] = [item.actor] if item.actor else []
    people.extend(match.group(0) for match in re.finditer(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text))
    people.extend(
        match.group(1)
        for match in re.finditer(r"\b(?:by|from|to|with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text)
    )
    for person in dict.fromkeys(value for value in people if value):
        values.append(ExtractedValue(ValueKind.PERSON, person, person == item.actor))
    status = _STATUS_RE.search(text)
    if status:
        values.append(ExtractedValue(ValueKind.STATUS, status.group(0)))
    reason = re.search(r"\b(?:root cause:?|because|due to|caused by|blocked by|so that|so|we need(?:ed)?|need(?:ed)? to)\s+(.+)$", text, re.I)
    if reason:
        values.append(ExtractedValue(ValueKind.REASON, reason.group(1).strip()))
    values.append(ExtractedValue(ValueKind.TEXT, text))
    return tuple(dict.fromkeys(values))


def _relations(text: str, values: tuple[ExtractedValue, ...], item: EvidenceItem) -> tuple[str, ...]:
    lowered = text.casefold()
    found = [
        relation
        for relation, cues in relation_cues().items()
        if any(re.search(rf"\b{re.escape(cue)}\b", lowered) for cue in cues)
    ]
    kinds = {value.kind for value in values}
    if ValueKind.MONEY in kinds:
        found.append("money")
    if ValueKind.NUMBER in kinds and re.search(r"\b(?:count|cases?|passing|total|out of)\b|\d+\s*/\s*\d+", text, re.I):
        found.append("quantity")
    if ValueKind.PHONE in kinds:
        found.append("contact_phone")
    if ValueKind.LOCATION in kinds:
        found.append("location")
    if ValueKind.REASON in kinds:
        found.append("reason")
    if ValueKind.STATUS in kinds:
        found.append("status")
        if isinstance(item.metadata.get("delivery_state"), str):
            found.append("delivery")
    if _COMMITTED_RE.search(text) or _COMPLETED_RE.search(text) or _CANCELLED_RE.search(text):
        found.append("commitment")
    if _COMPLETED_RE.search(text):
        found.append("delivery")
    if kinds & {ValueKind.DATE, ValueKind.TIME, ValueKind.DATETIME} and (
        _TIME_RELATION_RE.search(text) or re.search(r"\blaunch(?:es|ed|ing)?\b", text, re.I) or item.source_type == "google_calendar"
    ):
        found.append("time")
    return tuple(dict.fromkeys(found))


def build_claims(bundle: EvidenceBundle) -> tuple[Claim, ...]:
    claims: list[Claim] = []
    for item in bundle.items:
        for text in _clauses(item):
            values = _values(text, item)
            report = _REPORTED_RE.search(text)
            claims.append(
                Claim(
                    source_id=item.unit_id,
                    source_type=item.source_type,
                    record_id=item.record_id,
                    actor=item.actor,
                    text=text,
                    normalized_tokens=normalize_terms(informative_terms(tokenize(text))),
                    entities=_entities(text, item.actor),
                    values=values,
                    relation_terms=_relations(text, values, item),
                    available_at=item.available_at,
                    retrieval_rank=item.retrieval_rank,
                    negated=bool(_NEGATED_RE.search(text)),
                    conditional=bool(_CONDITIONAL_RE.search(text)),
                    speculative=bool(_SPECULATIVE_RE.search(text)),
                    proposed=bool(_PROPOSED_RE.search(text)),
                    committed=bool(_COMMITTED_RE.search(text)),
                    completed=bool(_COMPLETED_RE.search(text)),
                    cancelled=bool(_CANCELLED_RE.search(text)),
                    corrected=bool(_CORRECTED_RE.search(text)),
                    reported_speech=report is not None,
                    reported_speaker=report.group(1) if report else None,
                )
            )
    return tuple(claims)
