"""Planned hierarchical retrieval over the temporal visibility projection."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Mapping, Sequence

from ..ingest import parse_datetime
from ..models import MemoryUnit
from ..safety import sanitize_text
from ..visibility import visible_units
from .bm25 import BM25Index
from .planner import (
    Intent,
    QueryPlan,
    build_query_plan,
    derive_corpus_aliases,
    extract_explicit_dates,
    date_terms,
    normalize_terms,
    with_added_date,
)
from .tokenize import tokenize


@dataclass(frozen=True)
class RetrievalConfig:
    max_results: int = 20
    record_candidates: int = 40
    first_pass_per_record: int = 3
    record_metadata_weight: float = 0.25
    unit_metadata_weight: float = 0.12
    coverage_weight: float = 3.0
    rare_coverage_weight: float = 2.0
    proximity_weight: float = 1.5
    source_hint_weight: float = 5.0
    actor_hint_weight: float = 3.0
    entity_hint_weight: float = 2.5
    date_weight: float = 3.0
    causal_weight: float = 5.0
    state_weight: float = 2.5
    record_prior_weight: float = 0.08
    neighbor_weight: float = 0.15
    second_neighbor_weight: float = 0.04

    def __post_init__(self) -> None:
        if not 1 <= self.max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")
        if self.record_candidates < 1:
            raise ValueError("record_candidates must be positive")
        if self.first_pass_per_record < 1:
            raise ValueError("first_pass_per_record must be positive")


@dataclass(frozen=True)
class _UnitScore:
    unit_id: str
    record_id: str
    score: float
    components: Mapping[str, float]


_RECORD_PASSAGE_WEIGHTS = (1.0, 0.35, 0.15)
_RECORD_METADATA_FIELDS = (
    "meeting_title", "channel_name", "conversation_title", "target_app",
    "target_context", "repo",
)
_UNIT_METADATA_FIELDS = ("target_app", "target_context", "repo")
_CAUSAL_TERMS = frozenset(
    {
        "because", "block", "blocked", "blocker", "cause", "caused", "change",
        "changed", "correction", "delay", "delayed", "dependency", "due", "fix",
        "issue", "move", "moved", "reason", "regression", "slip", "slipped", "update",
        "updated", "wrong",
    }
)
_CURRENT_TERMS = frozenset(
    {
        "actually", "confirmed", "corrected", "correction", "current", "currently",
        "decision", "final", "latest", "now", "sorry", "still", "updated", "verified",
        "wrong",
    }
)


def aggregate_record_scores(
    unit_scores: Sequence[float],
    *,
    metadata_score: float = 0.0,
    metadata_weight: float = 0.25,
) -> float:
    """Combine only the three strongest passages plus weak record metadata."""

    strongest = sorted((score for score in unit_scores if score > 0), reverse=True)
    passage_score = sum(
        weight * score for weight, score in zip(_RECORD_PASSAGE_WEIGHTS, strongest)
    )
    return passage_score + metadata_weight * max(metadata_score, 0.0)


def _unit_metadata_parts(unit: MemoryUnit) -> tuple[str, ...]:
    values: list[str] = [unit.source_type.replace("_", " ")]
    if unit.actor:
        values.append(unit.actor)
    for field in _UNIT_METADATA_FIELDS:
        value = unit.metadata.get(field)
        if isinstance(value, str) and value:
            values.append(value)
    return tuple(sanitize_text(value) for value in values)


def _record_metadata(units: Sequence[MemoryUnit]) -> str:
    unique: dict[str, None] = {}
    for unit in units:
        unique.setdefault(unit.source_type.replace("_", " "), None)
        if unit.actor:
            unique.setdefault(sanitize_text(unit.actor), None)
        for field in _RECORD_METADATA_FIELDS:
            value = unit.metadata.get(field)
            if isinstance(value, str) and value:
                unique.setdefault(sanitize_text(value), None)
    return " ".join(unique)


def _normalized_text_tokens(text: str) -> tuple[str, ...]:
    return normalize_terms(tokenize(text))


def _coverage(query: Sequence[str], document: set[str]) -> float:
    unique = set(query)
    return len(unique & document) / len(unique) if unique else 0.0


def _rare_coverage(query: Sequence[str], document: set[str], index: BM25Index) -> float:
    unique = set(query)
    if not unique:
        return 0.0
    total = sum(index.inverse_document_frequency(term) for term in unique)
    matched = sum(index.inverse_document_frequency(term) for term in unique if term in document)
    return matched / total if total else 0.0


def _proximity(query: Sequence[str], document: Sequence[str]) -> float:
    useful_query = tuple(dict.fromkeys(query))
    if len(useful_query) < 2 or len(document) < 2:
        return 0.0
    positions: dict[str, list[int]] = defaultdict(list)
    for position, token in enumerate(document):
        if token in useful_query:
            positions[token].append(position)
    matched_terms = [term for term in useful_query if term in positions]
    if len(matched_terms) < 2:
        return 0.0
    for left, right in zip(useful_query, useful_query[1:]):
        if any(
            position + 1 < len(document) and document[position + 1] == right
            for position in positions.get(left, ())
        ):
            return 1.0
    spans = [
        abs(left - right) + 1
        for index, left_term in enumerate(matched_terms)
        for right_term in matched_terms[index + 1 :]
        for left in positions[left_term]
        for right in positions[right_term]
    ]
    return min(1.0, 2.0 / min(spans)) if spans else 0.0


def _actor_matches(unit: MemoryUnit, hints: Sequence[str], primary_actor: str | None) -> float:
    if not hints or not unit.actor:
        return 0.0
    actor_tokens = set(tokenize(unit.actor))
    for hint in hints:
        if hint == "__self__":
            if primary_actor and unit.actor == primary_actor:
                return 1.0
        elif set(tokenize(hint)) <= actor_tokens:
            return 1.0
    return 0.0


def _entity_matches(unit: MemoryUnit, hints: Sequence[str]) -> float:
    document = set(tokenize(" ".join((unit.safe_text, *_unit_metadata_parts(unit)))))
    return float(any(set(tokenize(hint)) <= document for hint in hints)) if hints else 0.0


def _parse_structured_date(value: object) -> date | None:
    if isinstance(value, Mapping):
        value = value.get("dateTime") or value.get("date")
    if not isinstance(value, str):
        return None
    try:
        if "T" in value:
            return parse_datetime(value).date()
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return None


def _structured_event_dates(unit: MemoryUnit) -> tuple[date, ...]:
    found: dict[date, None] = {}
    for field in ("start", "end", "meeting_start", "meeting_end"):
        parsed = _parse_structured_date(unit.metadata.get(field))
        if parsed:
            found.setdefault(parsed, None)
    return tuple(found)


def _content_dates(unit: MemoryUnit) -> tuple[date, ...]:
    return extract_explicit_dates(unit.safe_text, unit.available_at)


def _unit_dates(unit: MemoryUnit) -> tuple[date, ...]:
    values = (*_structured_event_dates(unit), *_content_dates(unit), unit.available_at.date())
    return tuple(dict.fromkeys(values))


def _date_matches(unit: MemoryUnit, dates: Sequence[date]) -> float:
    return float(bool(set(dates) & set(_unit_dates(unit)))) if dates else 0.0


class HierarchicalRetriever:
    """Plan and rank only units returned by ``visible_units(as_of)``."""

    def __init__(self, units: Iterable[MemoryUnit], *, config: RetrievalConfig | None = None) -> None:
        self._units = tuple(units)
        self.config = config or RetrievalConfig()

    def retrieve(
        self, question: str, as_of: datetime | str, *, limit: int | None = None
    ) -> tuple[str, ...]:
        retrieved, _ = self.retrieve_with_debug(question, as_of, limit=limit)
        return retrieved

    def retrieve_with_debug(
        self, question: str, as_of: datetime | str, *, limit: int | None = None
    ) -> tuple[tuple[str, ...], dict[str, object]]:
        # Visibility is the first retrieval operation and the only source pool.
        visible = visible_units(self._units, as_of)
        requested_limit = self.config.max_results if limit is None else limit
        if not 0 <= requested_limit <= 20:
            raise ValueError("limit must be between 0 and 20")
        reference = parse_datetime(as_of) if isinstance(as_of, str) else as_of
        aliases = derive_corpus_aliases(visible)
        plan = build_query_plan(question, reference, aliases=aliases)
        debug: dict[str, object] = {
            "plan": plan.debug_dict(),
            "visible_units": len(visible),
            "derived_aliases": len(aliases),
            "multi_hop": {"attempted": False, "resolved_date": None, "anchor_unit": None},
        }
        if requested_limit == 0 or not visible:
            return (), debug
        ranked, rank_debug = self._rank_visible(plan, visible)
        debug.update(rank_debug)
        if plan.anchor_query:
            merged, hop_debug = self._temporal_anchor_retrieval(
                plan, visible, aliases, ranked, requested_limit
            )
            debug["multi_hop"] = hop_debug
            if merged is not None:
                return merged, debug
        return self._select_diverse(ranked, visible, plan, requested_limit), debug

    def _rank_visible(
        self, plan: QueryPlan, visible: Sequence[MemoryUnit]
    ) -> tuple[tuple[_UnitScore, ...], dict[str, object]]:
        query_terms = plan.expanded_terms
        if Intent.CAUSAL_CHANGE in plan.intent_flags and plan.explicit_dates:
            generated_date_terms = {
                term
                for value in plan.explicit_dates
                for term in normalize_terms(date_terms(value))
            }
            query_terms = tuple(
                term
                for term in query_terms
                if term not in generated_date_terms
            )
        if not query_terms:
            return (), {"top_records": [], "top_units": []}
        units_by_record: dict[str, list[MemoryUnit]] = defaultdict(list)
        for unit in visible:
            units_by_record[unit.record_id].append(unit)
        primary_actor = self._primary_actor(visible)

        body_tokens = {unit.unit_id: _normalized_text_tokens(unit.safe_text) for unit in visible}
        body_index = BM25Index(body_tokens.items())
        body_scores = body_index.scores(query_terms)
        unit_metadata_tokens = {
            unit.unit_id: _normalized_text_tokens(" ".join(_unit_metadata_parts(unit)))
            for unit in visible
        }
        unit_metadata_scores = BM25Index(unit_metadata_tokens.items()).scores(query_terms)
        record_metadata_index = BM25Index(
            (record_id, _normalized_text_tokens(_record_metadata(record_units)))
            for record_id, record_units in units_by_record.items()
        )
        record_metadata_scores = record_metadata_index.scores(query_terms)

        initial: dict[str, _UnitScore] = {}
        causal_query = Intent.CAUSAL_CHANGE in plan.intent_flags
        current_query = Intent.CURRENT_STATE in plan.intent_flags
        for unit in visible:
            unit_id = unit.unit_id
            document_set = set(body_tokens[unit_id])
            body = body_scores.get(unit_id, 0.0)
            metadata = unit_metadata_scores.get(unit_id, 0.0)
            source = float(unit.source_type in plan.source_hints)
            actor = _actor_matches(unit, plan.speaker_hints, primary_actor)
            entity = _entity_matches(unit, plan.entity_hints)
            date_match = _date_matches(unit, plan.explicit_dates)
            causal_matches = len(document_set & _CAUSAL_TERMS)
            causal = min(1.5, causal_matches / 2.0) if causal_query else 0.0
            current = float(current_query and bool(document_set & _CURRENT_TERMS))
            if not (body or metadata or actor or entity or date_match):
                source = 0.0
            components = {
                "bm25": body,
                "metadata": self.config.unit_metadata_weight * metadata,
                "coverage": self.config.coverage_weight * _coverage(query_terms, document_set),
                "rare_coverage": self.config.rare_coverage_weight
                * _rare_coverage(query_terms, document_set, body_index),
                "proximity": self.config.proximity_weight
                * _proximity(plan.lexical_terms, tokenize(unit.safe_text)),
                "source_hint": self.config.source_hint_weight * source,
                "actor_hint": self.config.actor_hint_weight * actor,
                "entity_hint": self.config.entity_hint_weight * entity,
                "date_match": self.config.date_weight
                * date_match
                * (0.35 if causal_query else 1.0),
                "causal": self.config.causal_weight * causal,
                "current_state": self.config.state_weight * current,
            }
            initial[unit_id] = _UnitScore(
                unit_id, unit.record_id, sum(components.values()), components
            )

        record_order = {record_id: position for position, record_id in enumerate(units_by_record)}
        record_scores = {
            record_id: aggregate_record_scores(
                [initial[unit.unit_id].score for unit in record_units],
                metadata_score=record_metadata_scores.get(record_id, 0.0),
                metadata_weight=self.config.record_metadata_weight,
            )
            for record_id, record_units in units_by_record.items()
        }
        candidate_records = [
            record_id
            for record_id, score in sorted(
                record_scores.items(), key=lambda item: (-item[1], record_order[item[0]])
            )
            if score > 0.0
        ][: self.config.record_candidates]
        candidate_set = set(candidate_records)

        exact_scores = {unit_id: scored.score for unit_id, scored in initial.items()}
        self._apply_meeting_context(units_by_record, exact_scores)
        visible_order = {unit.unit_id: position for position, unit in enumerate(visible)}
        ranked: list[_UnitScore] = []
        for unit_id, scored in initial.items():
            if scored.record_id not in candidate_set or exact_scores[unit_id] <= 0.0:
                continue
            prior = self.config.record_prior_weight * record_scores[scored.record_id]
            context = exact_scores[unit_id] - scored.score
            components = dict(scored.components)
            components["record_prior"] = prior
            components["neighbor_context"] = context
            ranked.append(
                _UnitScore(
                    unit_id, scored.record_id, exact_scores[unit_id] + prior, components
                )
            )
        ranked.sort(key=lambda item: (-item.score, visible_order[item.unit_id]))
        debug = {
            "top_records": [
                {"record_id": record_id, "score": round(record_scores[record_id], 6)}
                for record_id in candidate_records[:20]
            ],
            "top_units": [
                {
                    "unit_id": item.unit_id,
                    "record_id": item.record_id,
                    "score": round(item.score, 6),
                    "components": {
                        key: round(value, 6) for key, value in item.components.items() if value
                    },
                }
                for item in ranked[:20]
            ],
        }
        return tuple(ranked), debug

    def _apply_meeting_context(
        self, units_by_record: Mapping[str, Sequence[MemoryUnit]], scores: dict[str, float]
    ) -> None:
        original = dict(scores)
        for record_units in units_by_record.values():
            if not record_units or record_units[0].source_type != "meeting":
                continue
            ordered = sorted(
                record_units,
                key=lambda unit: (float(unit.metadata.get("segment_start_s", 0.0)), unit.unit_id),
            )
            for position, unit in enumerate(ordered):
                # Neighbors strengthen an already-matching target, never create one.
                if original.get(unit.unit_id, 0.0) <= 0.0:
                    continue
                direct = max(
                    (
                        original.get(ordered[index].unit_id, 0.0)
                        for index in (position - 1, position + 1)
                        if 0 <= index < len(ordered)
                    ),
                    default=0.0,
                )
                second = max(
                    (
                        original.get(ordered[index].unit_id, 0.0)
                        for index in (position - 2, position + 2)
                        if 0 <= index < len(ordered)
                    ),
                    default=0.0,
                )
                scores[unit.unit_id] = (
                    original[unit.unit_id]
                    + self.config.neighbor_weight * direct
                    + self.config.second_neighbor_weight * second
                )

    def _select_diverse(
        self,
        ranked: Sequence[_UnitScore],
        visible: Sequence[MemoryUnit],
        plan: QueryPlan,
        limit: int,
    ) -> tuple[str, ...]:
        units = {unit.unit_id: unit for unit in visible}
        selected: list[str] = []
        selected_set: set[str] = set()
        per_record: Counter[str] = Counter()
        per_actor: Counter[str] = Counter()
        comparison = bool(
            {Intent.DISAGREEMENT, Intent.ATTRIBUTION}
            & set(plan.intent_flags)
        )
        record_cap = 2 if comparison else self.config.first_pass_per_record
        actor_cap = 2 if comparison else limit
        for item in ranked:
            unit = units[item.unit_id]
            actor_key = unit.actor or "__unknown__"
            if per_record[item.record_id] >= record_cap or per_actor[actor_key] >= actor_cap:
                continue
            selected.append(item.unit_id)
            selected_set.add(item.unit_id)
            per_record[item.record_id] += 1
            per_actor[actor_key] += 1
            if len(selected) == limit:
                return tuple(selected)
        for item in ranked:
            if item.unit_id in selected_set:
                continue
            selected.append(item.unit_id)
            selected_set.add(item.unit_id)
            if len(selected) == limit:
                break
        return tuple(selected)

    def _temporal_anchor_retrieval(
        self,
        plan: QueryPlan,
        visible: Sequence[MemoryUnit],
        aliases: Mapping[str, Sequence[str]],
        ordinary_ranked: Sequence[_UnitScore],
        limit: int,
    ) -> tuple[tuple[str, ...] | None, dict[str, object]]:
        hop_debug: dict[str, object] = {
            "attempted": True, "resolved_date": None, "anchor_unit": None,
        }
        anchor_plan = build_query_plan(
            plan.anchor_query or "", plan.reference_time, aliases=aliases
        )
        anchor_ranked, _ = self._rank_visible(anchor_plan, visible)
        resolution = self._resolve_anchor_date(anchor_ranked, visible, plan.reference_time)
        if resolution is None:
            return None, hop_debug
        resolved_date, anchor_unit_id = resolution
        hop_debug["resolved_date"] = resolved_date.isoformat()
        hop_debug["anchor_unit"] = anchor_unit_id

        target_plan = build_query_plan(
            plan.target_query or plan.original_question, plan.reference_time, aliases=aliases
        )
        target_plan = with_added_date(target_plan, resolved_date)
        target_ranked, _ = self._rank_visible(target_plan, visible)
        anchor_ids = self._select_diverse(anchor_ranked, visible, anchor_plan, min(5, limit))
        target_ids = self._select_diverse(target_ranked, visible, target_plan, limit)
        ordinary_ids = self._select_diverse(ordinary_ranked, visible, plan, limit)

        merged: list[str] = []
        for stream in (target_ids[:4], anchor_ids[:2], target_ids[4:], ordinary_ids):
            for unit_id in stream:
                if unit_id not in merged:
                    merged.append(unit_id)
                if len(merged) == limit:
                    return tuple(merged), hop_debug
        return tuple(merged), hop_debug

    def _resolve_anchor_date(
        self,
        ranked: Sequence[_UnitScore],
        visible: Sequence[MemoryUnit],
        reference: datetime,
    ) -> tuple[date, str] | None:
        units = {unit.unit_id: unit for unit in visible}
        reference_date = reference.date()
        for item in ranked[:5]:
            unit = units[item.unit_id]
            candidates = tuple(
                dict.fromkeys((*_structured_event_dates(unit), *_content_dates(unit)))
            )
            if not candidates:
                continue
            future = [value for value in candidates if value >= reference_date]
            chosen = min(future) if future else max(candidates)
            return chosen, item.unit_id
        return None

    @staticmethod
    def _primary_actor(units: Sequence[MemoryUnit]) -> str | None:
        counts = Counter(
            unit.actor
            for unit in units
            if unit.actor and unit.source_type in {"dictation", "gmail", "slack", "meeting"}
        )
        return counts.most_common(1)[0][0] if counts else None

