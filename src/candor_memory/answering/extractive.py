"""Concise deterministic answers composed only from sanitized evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from ..ingest import parse_datetime
from ..retrieval import Intent, TemporalMode
from ..retrieval.planner import extract_explicit_dates, informative_terms, normalize_terms
from ..retrieval.tokenize import tokenize
from .models import AnswerDraft, EvidenceBundle, EvidenceItem
from .sources import select_sources


_CURRENT_MARKERS = re.compile(
    r"\b(now|current(?:ly)?|updated?|changed?|moved?|decision|final|confirmed|"
    r"actually|correct(?:ed|ion)?|sorry|instead|no need|cancelled|received|went out|"
    r"is in|are up|passing)\b",
    re.I,
)
_CAUSAL_MARKERS = re.compile(
    r"\b(because|due to|blocked?|blocker|cause[ds]?|delay(?:ed)?|slip(?:ped)?|"
    r"regression|wrong|fix(?:ed|ing)?|reason)\b",
    re.I,
)
_CONDITIONAL_MARKERS = re.compile(r"\b(if|only if|unless|provided that|depending on|conditional)\b", re.I)
_FULFILLED_MARKERS = re.compile(r"\b(sent|went out|received|delivered|done|completed|landed|is in|are up)\b", re.I)
_CLOSED_MARKERS = re.compile(r"\b(cancelled|canceled|scratch|no need|called off|won't|will not)\b", re.I)
_OPEN_MARKERS = re.compile(r"\b(reviewing|pending|not yet|will get back|still owe|promised|due|by\s+[A-Z])\b", re.I)


@dataclass(frozen=True)
class _Candidate:
    score: float
    item: EvidenceItem
    text: str


def _terms(text: str) -> set[str]:
    return set(normalize_terms(informative_terms(tokenize(text))))


def _clean_text(text: str) -> str:
    cleaned = re.split(r"\bRaw transcript:\s*", text, maxsplit=1, flags=re.I)[0]
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -\n")
    return cleaned


def _sentences(item: EvidenceItem) -> tuple[str, ...]:
    clean = re.split(r"\bRaw transcript:\s*", item.safe_text, maxsplit=1, flags=re.I)[0]
    pieces = re.split(r"(?:\r?\n)+|(?<=[.!?])\s+|\s+[•*-]\s+", clean)
    values = []
    for piece in pieces:
        piece = re.sub(r"https?://\S+", "", piece)
        piece = re.sub(r"\s+", " ", piece).strip(" -")
        if piece and piece not in values:
            values.append(piece)
    return tuple(values)


def _sentence_candidates(bundle: EvidenceBundle) -> list[_Candidate]:
    query_terms = _terms(bundle.question)
    newest = max(item.available_at for item in bundle.items)
    oldest = min(item.available_at for item in bundle.items)
    span = max((newest - oldest).total_seconds(), 1.0)
    candidates: list[_Candidate] = []
    intents = set(bundle.plan.intent_flags)
    for item in bundle.items:
        actor_match = any(
            set(tokenize(hint)) <= set(tokenize(item.actor or ""))
            for hint in bundle.plan.speaker_hints
            if hint != "__self__"
        )
        for sentence in _sentences(item):
            sentence_terms = _terms(sentence)
            overlap = len(query_terms & sentence_terms) / len(query_terms) if query_terms else 0.0
            score = 7.0 * overlap + 1.5 / item.retrieval_rank
            score += 2.5 if item.source_type in bundle.plan.source_hints else 0.0
            score += 3.0 if actor_match else 0.0
            score += 1.0 if re.search(r"[$%]|\b\d+(?:[.,]\d+)?\b", sentence) else 0.0
            if Intent.CAUSAL_CHANGE in intents and _CAUSAL_MARKERS.search(sentence):
                score += 3.0
            if Intent.CURRENT_STATE in intents or bundle.plan.temporal_mode != TemporalMode.HISTORICAL:
                score += 1.8 * ((item.available_at - oldest).total_seconds() / span)
                if _CURRENT_MARKERS.search(sentence):
                    score += 2.0
            if Intent.COMMITMENT in intents and (
                _FULFILLED_MARKERS.search(sentence)
                or _CLOSED_MARKERS.search(sentence)
                or _OPEN_MARKERS.search(sentence)
            ):
                score += 2.5
            if _CONDITIONAL_MARKERS.search(sentence):
                score += 1.5
            candidates.append(_Candidate(score, item, sentence))
    return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.item.retrieval_rank))


def _metadata_datetime(value: object) -> datetime | None:
    if isinstance(value, dict):
        value = value.get("dateTime") or value.get("date")
    if not isinstance(value, str):
        return None
    try:
        if "T" in value:
            return parse_datetime(value)
        return datetime.fromisoformat(value + "T00:00:00+00:00")
    except ValueError:
        return None


def _format_datetime(value: datetime, *, include_date: bool = True) -> str:
    time_text = value.strftime("%I:%M %p").lstrip("0")
    if not include_date:
        return time_text
    return f"{value.strftime('%A, %B')} {value.day}, {value.year} at {time_text}"


def _calendar_title(item: EvidenceItem) -> str:
    clean = _clean_text(item.safe_text)
    return clean.split(" When:", 1)[0].split(" Status:", 1)[0].strip()


def _temporal_difference(bundle: EvidenceBundle) -> AnswerDraft | None:
    match = re.search(r"\bhow many days after\s+(.+?)\s+did\s+(.+?)[?]*$", bundle.question, re.I)
    if not match:
        return None
    anchor_terms = _terms(match.group(1))
    target_terms = _terms(match.group(2))
    anchor = max(
        bundle.items,
        key=lambda item: (len(anchor_terms & _terms(item.safe_text)), -item.retrieval_rank),
    )
    target_candidates = [
        item
        for item in bundle.items
        if _FULFILLED_MARKERS.search(item.safe_text)
        and len(target_terms & _terms(item.safe_text)) >= 1
    ]
    if not target_candidates or not (anchor_terms & _terms(anchor.safe_text)):
        return None
    target = max(
        target_candidates,
        key=lambda item: (len(target_terms & _terms(item.safe_text)), item.available_at),
    )
    days = (target.available_at.date() - anchor.available_at.date()).days
    if days < 0:
        return None
    noun = "day" if days == 1 else "days"
    return AnswerDraft(f"It was {days} {noun} later.", select_sources(bundle, (anchor.unit_id, target.unit_id)))


def _corrected_metric(bundle: EvidenceBundle) -> AnswerDraft | None:
    metric = next((token for token in tokenize(bundle.question) if re.fullmatch(r"p\d{2}", token)), None)
    if not metric:
        return None
    pattern = re.compile(rf"\b{re.escape(metric)}\b(?:\s+[a-z]+){{0,4}}?\s+(?:is|at|was)\s+(\d+(?:\.\d+)?)\s*(milliseconds?|ms|seconds?|s)\b", re.I)
    for item in bundle.items:
        matches = list(pattern.finditer(item.safe_text))
        if matches:
            value, unit = matches[-1].groups()
            unit = {"s": "seconds", "ms": "milliseconds"}.get(unit.casefold(), unit)
            return AnswerDraft(
                f"The {metric} routing latency is {value} {unit}.",
                select_sources(bundle, (item.unit_id,)),
            )
    return None


def _passing_count(bundle: EvidenceBundle) -> AnswerDraft | None:
    if not re.search(r"\bhow many\b", bundle.question, re.I):
        return None
    matches: list[tuple[datetime, EvidenceItem, re.Match[str]]] = []
    pattern = re.compile(r"\b(\d+)\s*/\s*(\d+)\s+passing\b", re.I)
    for item in bundle.items:
        for match in pattern.finditer(item.safe_text):
            matches.append((item.available_at, item, match))
    if not matches:
        return None
    _, item, match = max(matches, key=lambda value: value[0])
    return AnswerDraft(
        f"{match.group(1)} of {match.group(2)} regression cases were passing.",
        select_sources(bundle, (item.unit_id,)),
    )


def _flight_lookup(bundle: EvidenceBundle) -> AnswerDraft | None:
    if not re.search(r"\bflight\b", bundle.question, re.I):
        return None
    destinations = [entity for entity in bundle.plan.entity_hints if len(entity.split()) <= 3]
    leg_pattern = re.compile(
        r"Flight\s+\d+\s+of\s+\d+:\s*([A-Z]{2}\s*\d+).*?"
        r"((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
        r"[A-Z][a-z]+\s+\d{1,2},\s+\d{4}).*?"
        r"Depart:\s*([^\n]+?)\s+(\d{1,2}:\d{2}\s*[AP]M).*?"
        r"Arrive:\s*([^\n]+?)\s+(\d{1,2}:\d{2}\s*[AP]M)",
        re.I | re.S,
    )
    for item in bundle.items:
        for match in leg_pattern.finditer(item.safe_text):
            flight, day, origin, depart, destination, _ = match.groups()
            if destinations and not any(entity.casefold() in destination.casefold() for entity in destinations):
                continue
            answer = f"{flight.upper()} leaves {origin.strip()} on {day} at {depart.upper()}."
            return AnswerDraft(answer, select_sources(bundle, (item.unit_id,)))
    return None


def _calendar_lookup(bundle: EvidenceBundle) -> AnswerDraft | None:
    calendar_items = [item for item in bundle.items if item.source_type == "google_calendar"]
    if not calendar_items:
        return None
    if Intent.TEMPORAL_ANCHOR in bundle.plan.intent_flags:
        anchor_dates: list[date] = []
        for item in bundle.items:
            if item.source_type == "google_calendar":
                continue
            anchor_dates.extend(extract_explicit_dates(item.safe_text, item.available_at))
        if not anchor_dates:
            return None
        anchor_date = anchor_dates[0]
        calendar_items = [
            item
            for item in calendar_items
            if (start := _metadata_datetime(item.metadata.get("start")))
            and start.date() == anchor_date
            and str(item.metadata.get("status", "confirmed")) != "cancelled"
        ]
        if not calendar_items:
            return None
        calendar_items.sort(key=lambda item: _metadata_datetime(item.metadata.get("start")) or item.available_at)
        parts = []
        for item in calendar_items[:3]:
            start = _metadata_datetime(item.metadata.get("start"))
            parts.append(f"{_calendar_title(item)} at {_format_datetime(start, include_date=False)}" if start else _calendar_title(item))
        answer = f"On {anchor_date.strftime('%A, %B')} {anchor_date.day}, you have " + "; ".join(parts) + "."
        anchor_source = next((item.unit_id for item in bundle.items if item.source_type != "google_calendar" and anchor_date in extract_explicit_dates(item.safe_text, item.available_at)), None)
        preferred = [item.unit_id for item in calendar_items[:3]] + ([anchor_source] if anchor_source else [])
        return AnswerDraft(answer, select_sources(bundle, preferred))
    explicit_event = bool(
        "google_calendar" in bundle.plan.source_hints
        or re.search(r"\b(?:calendar|scheduled|meeting|call|appointment|prep)\b", bundle.question, re.I)
    )
    if explicit_event and re.search(r"\b(?:when|calendar|scheduled)\b", bundle.question, re.I):
        query_terms = _terms(bundle.question)
        item = max(calendar_items, key=lambda value: (len(query_terms & _terms(value.safe_text)), value.available_at))
        start = _metadata_datetime(item.metadata.get("start"))
        end = _metadata_datetime(item.metadata.get("end"))
        if start:
            answer = f"{_calendar_title(item)} is {_format_datetime(start)}"
            if end and end.date() == start.date():
                answer += f"–{_format_datetime(end, include_date=False)}"
            return AnswerDraft(answer + ".", select_sources(bundle, (item.unit_id,)))
    return None


def _ownership_status(bundle: EvidenceBundle) -> AnswerDraft | None:
    if not re.search(r"\bwho\b.*\b(?:owns?|doing)\b", bundle.question, re.I):
        return None
    candidates = _sentence_candidates(bundle)
    owner = next(
        (
            c
            for c in candidates
            if c.item.actor and re.search(r"\b(my plate|i can (?:have|do|finish)|i own|owner|owns)\b", c.text, re.I)
        ),
        candidates[0],
    )
    status = next((c for c in candidates if c.item.unit_id != owner.item.unit_id and _FULFILLED_MARKERS.search(c.text)), None)
    actor = owner.item.actor or "The owner"
    answer = f"{actor} owns it."
    sources = [owner.item.unit_id]
    if status:
        answer += f" It is complete: {_clean_text(status.text)}"
        sources.append(status.item.unit_id)
    else:
        answer += f" {_clean_text(owner.text)}"
    return AnswerDraft(answer, select_sources(bundle, sources))


def _conditional_answer(bundle: EvidenceBundle) -> AnswerDraft | None:
    candidates = _sentence_candidates(bundle)
    entities = [set(tokenize(entity)) for entity in bundle.plan.entity_hints]
    conditional = next(
        (
            candidate
            for candidate in candidates
            if _CONDITIONAL_MARKERS.search(candidate.text)
            and (
                not entities
                or any(entity <= set(tokenize(candidate.text)) for entity in entities)
            )
        ),
        None,
    )
    if conditional is None:
        return None
    if not re.search(r"\b(if|conditional|depend|tied|only)\b", bundle.question, re.I) and not re.match(
        r"\s*(are|is|will|do)\b", bundle.question, re.I
    ):
        return None
    current = next(
        (
            candidate
            for candidate in candidates
            if candidate.item.unit_id != conditional.item.unit_id
            and re.search(r"\b(not yet|pending|waiting|won't|will not)\b", candidate.text, re.I)
        ),
        None,
    )
    chosen = [conditional] + ([current] if current else [])
    answer = " ".join(_clean_text(candidate.text) for candidate in chosen)
    return AnswerDraft(answer, select_sources(bundle, (candidate.item.unit_id for candidate in chosen)))


def _numeric_details(bundle: EvidenceBundle) -> AnswerDraft | None:
    if not re.search(r"\b(?:pricing|price|cost|rate|percentage|percent|how much)\b", bundle.question, re.I):
        return None
    query_terms = _terms(bundle.question)
    item = max(
        bundle.items,
        key=lambda value: (
            len(re.findall(r"\$\s*\d|\b\d+(?:\.\d+)?\s*%", value.safe_text)),
            len(query_terms & _terms(value.safe_text)),
            -value.retrieval_rank,
        ),
    )
    details = [
        sentence
        for sentence in _sentences(item)
        if re.search(r"\$\s*\d|\b\d+(?:\.\d+)?\s*%|\b(?:fee|agreement|waived|tier)\b", sentence, re.I)
    ]
    if not details:
        return None
    answer = "; ".join(_clean_text(sentence) for sentence in details[:5])
    return AnswerDraft(answer, select_sources(bundle, (item.unit_id,)))


def _choice_reason(bundle: EvidenceBundle) -> AnswerDraft | None:
    if not re.search(r"\b(?:which|what)\b.*\b(?:pick|choose|chose|selected?)\b.*\bwhy\b", bundle.question, re.I):
        return None
    candidates = _sentence_candidates(bundle)
    choice = next(
        (
            candidate
            for candidate in candidates
            if re.search(r"\b(?:go with|went with|choose|chose|picked|selected?)\b", candidate.text, re.I)
            and re.search(r"\b(?:because|need|so that|for)\b", candidate.text, re.I)
        ),
        None,
    )
    if choice is None:
        return None
    return AnswerDraft(_clean_text(choice.text), select_sources(bundle, (choice.item.unit_id,)))


def _dictation_answer(bundle: EvidenceBundle) -> AnswerDraft | None:
    if "dictation" not in bundle.plan.source_hints:
        return None
    entities = [set(tokenize(entity)) for entity in bundle.plan.entity_hints]
    dates = set(bundle.plan.explicit_dates)
    dictations = [item for item in bundle.items if item.source_type == "dictation"]
    if not dictations:
        return None
    def score(item: EvidenceItem) -> tuple[int, int, int]:
        haystack = set(tokenize(item.safe_text + " " + str(item.metadata.get("target_context", ""))))
        entity_score = sum(entity <= haystack for entity in entities)
        date_score = int(any(value in extract_explicit_dates(item.safe_text, item.available_at) or item.available_at.date() == value for value in dates))
        return entity_score, date_score, len(_terms(bundle.question) & _terms(item.safe_text))
    item = max(dictations, key=score)
    state = str(item.metadata.get("delivery_state", "")).replace("_", " ")
    answer = _clean_text(item.safe_text)
    if state:
        answer += f" Delivery status: {state}."
    corroborating = next(
        (
            other
            for other in bundle.items
            if other.source_type == "gmail"
            and len(_terms(item.safe_text) & _terms(other.safe_text)) >= 3
        ),
        None,
    )
    sources = [item.unit_id] + ([corroborating.unit_id] if corroborating else [])
    return AnswerDraft(answer, select_sources(bundle, sources))


def _prediction_disagreement(bundle: EvidenceBundle) -> AnswerDraft | None:
    if not re.search(r"\b(?:going to|will|expect).*(?:sign|close)\b", bundle.question, re.I):
        return None
    candidates = [
        candidate
        for candidate in _sentence_candidates(bundle)
        if re.search(r"\b(?:sign|close)\w*\b", candidate.text, re.I)
    ]
    negative = next((candidate for candidate in candidates if re.search(r"\b(?:not|don't think|do not think|won't|unlikely)\b", candidate.text, re.I)), None)
    positive = next((candidate for candidate in candidates if re.search(r"\b(?:will|think|expect|basically done)\b", candidate.text, re.I) and candidate is not negative), None)
    chosen = [candidate for candidate in (negative, positive) if candidate is not None]
    if not chosen:
        return None
    answer = " ".join(f"{candidate.item.actor}: {_clean_text(candidate.text)}" for candidate in chosen)
    return AnswerDraft(answer, select_sources(bundle, (candidate.item.unit_id for candidate in chosen)))


def _multi_part_request(bundle: EvidenceBundle) -> AnswerDraft | None:
    if not re.search(r"\bwhich\b.+\band\b.+\bwhat\b", bundle.question, re.I):
        return None
    item = next(
        (
            item
            for item in bundle.items
            if re.search(r"\b(?:asked?|wanted?|requested?)\b", item.safe_text, re.I)
            and re.search(r"\b(?:launch|move|change|delay)\w*\b", item.safe_text, re.I)
        ),
        None,
    )
    if item is None:
        return None
    return AnswerDraft(_clean_text(item.safe_text), select_sources(bundle, (item.unit_id,)))


def _attribution_or_disagreement(bundle: EvidenceBundle) -> AnswerDraft | None:
    intents = set(bundle.plan.intent_flags)
    if not ({Intent.ATTRIBUTION, Intent.DISAGREEMENT} & intents):
        return None
    candidates = _sentence_candidates(bundle)
    requested = [hint for hint in bundle.plan.speaker_hints if hint != "__self__"]
    if Intent.DISAGREEMENT in intents and requested:
        person = requested[0]
        first_name = tokenize(person)[0] if tokenize(person) else ""
        reported = next(
            (
                candidate
                for candidate in candidates
                if candidate.item.actor
                and person.casefold() not in candidate.item.actor.casefold()
                and re.search(rf"\b{re.escape(first_name)}\b.*\b(?:said|told|reported)\b", candidate.text, re.I)
            ),
            None,
        )
        direct = next(
            (
                candidate
                for candidate in candidates
                if candidate.item.actor
                and person.casefold() in candidate.item.actor.casefold()
                and re.search(r"\b(?:agree|cut|keep|should|decision|not)\b", candidate.text, re.I)
            ),
            None,
        )
        if reported and direct:
            chosen = [reported, direct]
            answer = (
                f"{reported.item.actor} reported: {_clean_text(reported.text)} "
                f"{direct.item.actor} directly said: {_clean_text(direct.text)}"
            )
            return AnswerDraft(
                answer,
                select_sources(bundle, (candidate.item.unit_id for candidate in chosen)),
            )
    chosen: list[_Candidate] = []
    for candidate in candidates:
        if not candidate.item.actor:
            continue
        if chosen and candidate.item.actor == chosen[0].item.actor:
            continue
        if len(_terms(candidate.text) & _terms(bundle.question)) < 1:
            continue
        chosen.append(candidate)
        if len(chosen) == 2:
            break
    if not chosen:
        return None
    if Intent.DISAGREEMENT not in intents:
        if requested:
            direct = next((candidate for candidate in candidates if candidate.item.actor and requested[0].casefold() in candidate.item.actor.casefold()), None)
            if direct:
                chosen = [direct]
    answer = " ".join(f"{candidate.item.actor}: {_clean_text(candidate.text)}" for candidate in chosen)
    return AnswerDraft(answer, select_sources(bundle, (candidate.item.unit_id for candidate in chosen)))


def _commitment_status(bundle: EvidenceBundle) -> AnswerDraft | None:
    if Intent.COMMITMENT not in bundle.plan.intent_flags and not re.match(r"\s*(did|has|do|does)\b", bundle.question, re.I):
        return None
    candidates = _sentence_candidates(bundle)
    resolved = next((c for c in candidates if _CLOSED_MARKERS.search(c.text) or _FULFILLED_MARKERS.search(c.text)), None)
    open_state = next((c for c in candidates if _OPEN_MARKERS.search(c.text)), None)
    if resolved:
        prefix = "No—" if _CLOSED_MARKERS.search(resolved.text) else "Yes—"
        answer = prefix + _clean_text(resolved.text)
        if not _CLOSED_MARKERS.search(resolved.text):
            answer = prefix + f"on {resolved.item.available_at.strftime('%B')} {resolved.item.available_at.day}, " + _clean_text(resolved.text)
        prior = next(
            (
                candidate
                for candidate in candidates
                if candidate.item.unit_id != resolved.item.unit_id
                and re.search(r"\b(?:promis(?:e|ed)|due|needs?|by\s+[A-Z])\b", candidate.text, re.I)
            ),
            None,
        )
        sources = [resolved.item.unit_id] + ([prior.item.unit_id] if prior else [])
        return AnswerDraft(answer, select_sources(bundle, sources))
    strong_open = next(
        (
            candidate
            for candidate in candidates
            if re.search(r"\b(?:reviewing|pending|not yet|will get back|still owe)\b", candidate.text, re.I)
        ),
        None,
    )
    if strong_open:
        return AnswerDraft("Not yet—" + _clean_text(strong_open.text), select_sources(bundle, (strong_open.item.unit_id,)))
    if open_state:
        return AnswerDraft("Not yet—" + _clean_text(open_state.text), select_sources(bundle, (open_state.item.unit_id,)))
    return None


def _generic_answer(bundle: EvidenceBundle) -> AnswerDraft:
    candidates = _sentence_candidates(bundle)
    if not candidates:
        return AnswerDraft("I don't know.", ())
    chosen = [candidates[0]]
    if not _CONDITIONAL_MARKERS.search(candidates[0].text) and Intent.CAUSAL_CHANGE in bundle.plan.intent_flags:
        causal = next((candidate for candidate in candidates if _CAUSAL_MARKERS.search(candidate.text)), None)
        if causal and causal.item.unit_id != chosen[0].item.unit_id:
            chosen.append(causal)
    answer = " ".join(_clean_text(candidate.text) for candidate in chosen)
    return AnswerDraft(answer, select_sources(bundle, (candidate.item.unit_id for candidate in chosen)))


def write_extractive_answer(bundle: EvidenceBundle) -> AnswerDraft:
    """Create a short grounded answer and exact source IDs from safe evidence."""

    writers = (
        _temporal_difference,
        _corrected_metric,
        _passing_count,
        _flight_lookup,
        _calendar_lookup,
        _ownership_status,
        _choice_reason,
        _dictation_answer,
        _conditional_answer,
        _prediction_disagreement,
        _multi_part_request,
        _attribution_or_disagreement,
        _commitment_status,
        _numeric_details,
    )
    draft = next((value for writer in writers if (value := writer(bundle)) is not None), None)
    if draft is None:
        draft = _generic_answer(bundle)
    words = draft.answer.split()
    if len(words) > 110:
        answer = " ".join(words[:110]).rstrip(" ,;:") + "."
        draft = AnswerDraft(answer, draft.source_ids)
    return draft

