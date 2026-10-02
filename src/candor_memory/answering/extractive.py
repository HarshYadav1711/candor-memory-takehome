"""Generic concise composition from aligned sanitized claims."""

from __future__ import annotations

from datetime import datetime

from .claims import Claim, ValueKind
from .frames import AnswerKind, QuestionFrame
from .models import AnswerDraft, EvidenceBundle
from .sources import select_sources
from .support import AlignedClaim, EvidenceAnalysis, analyze_evidence


def _format_datetime(value: str, *, include_date: bool = True) -> str:
    parsed = datetime.fromisoformat(value)
    clock = parsed.strftime("%I:%M %p").lstrip("0")
    if not include_date:
        return clock
    return f"{parsed.strftime('%A, %B')} {parsed.day}, {parsed.year} at {clock}"


def _structured_datetime(claim: Claim) -> str | None:
    values = [
        value.text
        for value in claim.values
        if value.kind == ValueKind.DATETIME and value.structured
    ]
    if not values:
        return None
    answer = _format_datetime(values[0])
    if len(values) > 1:
        end = datetime.fromisoformat(values[1])
        start = datetime.fromisoformat(values[0])
        answer += "–" + _format_datetime(values[1], include_date=end.date() != start.date())
    return answer


def _conflict(claims: tuple[AlignedClaim, ...]) -> list[Claim]:
    selected: list[Claim] = []
    actors: set[str] = set()
    for candidate in claims:
        claim = candidate.claim
        if not claim.actor or claim.actor in actors:
            continue
        selected.append(claim)
        actors.add(claim.actor)
        if len(selected) == 2:
            break
    return selected


def _required_values(frame: QuestionFrame) -> dict[ValueKind, int]:
    lowered = frame.request_text.casefold()
    required: dict[ValueKind, int] = {}
    if frame.answer_kind == AnswerKind.MONEY:
        required[ValueKind.MONEY] = 2 if any(term in lowered for term in ("both", "pricing", "rates", "terms", "tiers")) else 1
    elif frame.answer_kind == AnswerKind.QUANTITY:
        required[ValueKind.NUMBER] = 1
    elif frame.answer_kind == AnswerKind.NUMBER:
        required[ValueKind.NUMBER] = 1
    elif frame.answer_kind == AnswerKind.IDENTITY:
        required[ValueKind.PERSON] = 1
    elif frame.answer_kind == AnswerKind.REASON:
        required[ValueKind.REASON] = 1
    if "time" in frame.relation_terms:
        if "date" in lowered or "dates" in lowered or "day" in lowered or "deadline" in lowered:
            required[ValueKind.DATE] = 2 if "dates" in lowered else 1
        if "time" in lowered or "clock" in lowered or "when" in lowered:
            required[ValueKind.TIME] = 1
    if "location" in frame.relation_terms:
        required[ValueKind.LOCATION] = 1
    return required


def _value_counts(claim: Claim) -> dict[ValueKind, int]:
    counts: dict[ValueKind, int] = {}
    for value in claim.values:
        counts[value.kind] = counts.get(value.kind, 0) + 1
        if value.kind == ValueKind.DATETIME:
            counts[ValueKind.DATE] = counts.get(ValueKind.DATE, 0) + 1
            counts[ValueKind.TIME] = counts.get(ValueKind.TIME, 0) + 1
    return counts


def _cover(frame: QuestionFrame, candidates: tuple[AlignedClaim, ...], multi_part: bool) -> list[Claim]:
    frame_relations = set(frame.relation_terms)
    selected: list[Claim] = []
    covered: set[str] = set()
    required_values = _required_values(frame)
    covered_values: dict[ValueKind, int] = {}
    covered_value_texts: dict[ValueKind, set[str]] = {}
    limit = 4 if multi_part else 1
    for candidate in candidates:
        claim = candidate.claim
        new_relations = frame_relations & set(claim.relation_terms) - covered
        values = _value_counts(claim)
        claim_value_texts = {
            kind: {
                value.text.casefold()
                for value in claim.values
                if value.kind == kind or (value.kind == ValueKind.DATETIME and kind in {ValueKind.DATE, ValueKind.TIME})
            }
            for kind in required_values
        }
        adds_value = any(
            covered_values.get(kind, 0) < needed
            and bool(claim_value_texts.get(kind, set()) - covered_value_texts.get(kind, set()))
            for kind, needed in required_values.items()
        )
        adds_untyped_part = multi_part and not required_values and claim.text not in {value.text for value in selected}
        if not selected or new_relations or adds_value or adds_untyped_part:
            selected.append(claim)
            covered.update(frame_relations & set(claim.relation_terms))
            for kind, count in values.items():
                covered_values[kind] = covered_values.get(kind, 0) + count
            for kind, texts in claim_value_texts.items():
                covered_value_texts.setdefault(kind, set()).update(texts)
                covered_values[kind] = len(covered_value_texts[kind])
        values_done = all(covered_values.get(kind, 0) >= needed for kind, needed in required_values.items())
        relations_done = not frame_relations or covered >= frame_relations
        minimum = 2 if multi_part and not required_values else 1
        if len(selected) >= limit or (len(selected) >= minimum and values_done and relations_done):
            break
    return selected


def _compose_claim(claim: Claim, frame: QuestionFrame) -> str:
    kind = frame.answer_kind
    if kind in {AnswerKind.TIME, AnswerKind.DATE, AnswerKind.DATETIME}:
        structured = _structured_datetime(claim)
        if structured:
            title = claim.text.split(" When:", 1)[0].split(" Status:", 1)[0].strip()
            return f"{title} is {structured}."
    if kind in {AnswerKind.STATUS, AnswerKind.BOOLEAN}:
        if claim.cancelled:
            return f"No—{claim.text}"
        if claim.completed:
            return f"Yes—on {claim.available_at.strftime('%B')} {claim.available_at.day}, {claim.text}"
    answer = claim.text
    if claim.corrected:
        answer = answer.split(", not ", 1)[0].rstrip(" ,;:")
    if claim.actor and "ownership" in frame.relation_terms:
        answer = f"{claim.actor} owns it: {answer}"
    elif kind == AnswerKind.IDENTITY and claim.actor:
        answer = f"{claim.actor}: {answer}"
    elif "attribution" in frame.relation_terms and claim.actor:
        answer = f"{claim.actor}: {answer}"
    structured_status = next(
        (value.text for value in claim.values if value.kind == ValueKind.STATUS and value.structured),
        None,
    )
    if structured_status and structured_status.casefold() not in answer.casefold():
        answer += f" Delivery status: {structured_status}."
    return answer


def write_extractive_answer(
    bundle: EvidenceBundle,
    analysis: EvidenceAnalysis | None = None,
) -> AnswerDraft:
    """Compose an answer only from claims supporting the requested slot."""

    analysis = analysis or analyze_evidence(bundle)
    if not analysis.aligned:
        return AnswerDraft("I don't know.", ())
    frame = analysis.frame
    if frame.answer_kind == AnswerKind.QUANTITY and " after " in frame.question_text.casefold():
        distinct = {candidate.claim.source_id: candidate.claim for candidate in analysis.aligned}
        if len(distinct) >= 2:
            ordered = sorted(distinct.values(), key=lambda claim: claim.available_at)
            days = (ordered[-1].available_at.date() - ordered[0].available_at.date()).days
            noun = "day" if days == 1 else "days"
            selected = [ordered[0], ordered[-1]]
            answer = f"It was {days} {noun} later."
            return AnswerDraft(answer, select_sources(bundle, (claim.source_id for claim in selected)))
    if analysis.conflicting:
        selected = _conflict(analysis.aligned)
        if any(claim.reported_speech for claim in selected):
            parts = [
                f"{claim.actor} reported: {claim.text}" if claim.reported_speech
                else f"{claim.actor} directly said: {claim.text}"
                for claim in selected
            ]
        else:
            parts = [f"{claim.actor}: {claim.text}" for claim in selected]
    else:
        selected = _cover(
            frame,
            analysis.aligned,
            frame.multi_part or frame.answer_kind in {AnswerKind.LIST, AnswerKind.MONEY},
        )
        if selected and frame.answer_kind in {AnswerKind.STATUS, AnswerKind.BOOLEAN} and (
            selected[0].completed or selected[0].cancelled
        ):
            prior = next(
                (
                    candidate.claim
                    for candidate in analysis.aligned
                    if candidate.claim.source_id != selected[0].source_id
                    and candidate.claim.committed
                    and candidate.claim.available_at <= selected[0].available_at
                ),
                None,
            )
            if prior:
                selected.append(prior)
        if frame.requested_actor_role == "direct" and selected:
            direct = selected[0]
            mentioned = {token.casefold() for entity in frame.named_entities for token in entity.split()}
            report = next(
                (
                    claim
                    for claim in analysis.claims
                    if claim.reported_speech
                    and claim.actor
                    and any(token in claim.text.casefold() for token in mentioned)
                ),
                None,
            )
            if report and report.source_id != direct.source_id:
                selected = [report, direct]
                parts = [
                    f"{report.actor} reported: {report.text}",
                    f"{direct.actor} directly said: {direct.text}",
                ]
            else:
                parts = [f"{direct.actor}: {_compose_claim(direct, frame)}"]
        else:
            parts = [_compose_claim(claim, frame) for claim in selected]
    answer = " ".join(part.strip() for part in parts if part.strip())
    words = answer.split()
    if len(words) > 80:
        answer = " ".join(words[:80]).rstrip(" ,;:") + "."
    if answer and answer[-1] not in ".!?":
        answer += "."
    sources = select_sources(bundle, (claim.source_id for claim in selected))
    return AnswerDraft(answer or "I don't know.", sources)
