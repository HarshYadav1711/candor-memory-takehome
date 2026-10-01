"""Two-stage lexical retrieval over the temporal visibility projection."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from ..models import MemoryUnit
from ..safety import sanitize_text
from ..visibility import visible_units
from .bm25 import BM25Index
from .tokenize import tokenize


@dataclass(frozen=True)
class RetrievalConfig:
    max_results: int = 20
    record_candidates: int = 40
    first_pass_per_record: int = 3
    record_metadata_weight: float = 0.25
    unit_metadata_weight: float = 0.15
    neighbor_weight: float = 0.20
    second_neighbor_weight: float = 0.05

    def __post_init__(self) -> None:
        if not 1 <= self.max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")
        if self.record_candidates < 1:
            raise ValueError("record_candidates must be positive")
        if self.first_pass_per_record < 1:
            raise ValueError("first_pass_per_record must be positive")


_RECORD_PASSAGE_WEIGHTS = (1.0, 0.35, 0.15)
_SAFE_METADATA_FIELDS = (
    "meeting_title",
    "channel_name",
    "conversation_title",
    "target_app",
    "target_context",
    "repo",
)


def aggregate_record_scores(
    unit_scores: Sequence[float],
    *,
    metadata_score: float = 0.0,
    metadata_weight: float = 0.25,
) -> float:
    """Combine the strongest passages without rewarding record length.

    Only the top three passage scores contribute, with sharply decreasing
    weights. A separately indexed, non-repeated metadata document supplies a
    small title/source contribution.
    """

    strongest = sorted((score for score in unit_scores if score > 0), reverse=True)
    passage_score = sum(
        weight * score
        for weight, score in zip(_RECORD_PASSAGE_WEIGHTS, strongest)
    )
    return passage_score + metadata_weight * max(metadata_score, 0.0)


def _metadata_parts(unit: MemoryUnit) -> tuple[str, ...]:
    values: list[str] = [unit.source_type.replace("_", " ")]
    if unit.actor:
        values.append(unit.actor)
    for field in _SAFE_METADATA_FIELDS:
        value = unit.metadata.get(field)
        if isinstance(value, str) and value:
            values.append(value)
    return tuple(sanitize_text(value) for value in values)


def _record_metadata(units: Sequence[MemoryUnit]) -> str:
    unique: dict[str, None] = {}
    for unit in units:
        for value in _metadata_parts(unit):
            unique.setdefault(value, None)
    return " ".join(unique)


class HierarchicalRetriever:
    """Build a fresh lexical index from ``visible_units(as_of)`` per query."""

    def __init__(
        self,
        units: Iterable[MemoryUnit],
        *,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._units = tuple(units)
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        question: str,
        as_of: datetime | str,
        *,
        limit: int | None = None,
    ) -> tuple[str, ...]:
        # This projection deliberately happens before tokenization or indexing.
        visible = visible_units(self._units, as_of)
        requested_limit = self.config.max_results if limit is None else limit
        if not 0 <= requested_limit <= 20:
            raise ValueError("limit must be between 0 and 20")
        if requested_limit == 0 or not visible:
            return ()
        return self._retrieve_visible(question, visible, requested_limit)

    def _retrieve_visible(
        self,
        question: str,
        visible: Sequence[MemoryUnit],
        limit: int,
    ) -> tuple[str, ...]:
        query_tokens = tokenize(question)
        if not query_tokens:
            return ()

        units_by_id = {unit.unit_id: unit for unit in visible}
        units_by_record: dict[str, list[MemoryUnit]] = defaultdict(list)
        for unit in visible:
            units_by_record[unit.record_id].append(unit)

        # Body scoring has exactly one content source: sanitized safe_text.
        body_index = BM25Index(
            (unit.unit_id, tokenize(unit.safe_text)) for unit in visible
        )
        body_scores = body_index.scores(query_tokens)

        # Metadata is sanitized, indexed separately, and kept weak so titles do
        # not drown out exact passage matches.
        unit_metadata_index = BM25Index(
            (unit.unit_id, tokenize(" ".join(_metadata_parts(unit)))) for unit in visible
        )
        unit_metadata_scores = unit_metadata_index.scores(query_tokens)
        record_metadata_index = BM25Index(
            (
                record_id,
                tokenize(_record_metadata(record_units)),
            )
            for record_id, record_units in units_by_record.items()
        )
        record_metadata_scores = record_metadata_index.scores(query_tokens)

        combined_unit_scores = {
            unit.unit_id: body_scores.get(unit.unit_id, 0.0)
            + self.config.unit_metadata_weight
            * unit_metadata_scores.get(unit.unit_id, 0.0)
            for unit in visible
        }

        record_order = {
            record_id: position for position, record_id in enumerate(units_by_record)
        }
        record_scores = {
            record_id: aggregate_record_scores(
                [combined_unit_scores[unit.unit_id] for unit in record_units],
                metadata_score=record_metadata_scores.get(record_id, 0.0),
                metadata_weight=self.config.record_metadata_weight,
            )
            for record_id, record_units in units_by_record.items()
        }
        candidate_records = [
            record_id
            for record_id, score in sorted(
                record_scores.items(),
                key=lambda item: (-item[1], record_order[item[0]]),
            )
            if score > 0.0
        ][: self.config.record_candidates]
        candidate_set = set(candidate_records)
        if not candidate_set:
            return ()

        exact_scores = dict(combined_unit_scores)
        self._apply_meeting_context(
            units_by_record,
            combined_unit_scores,
            exact_scores,
        )
        visible_order = {unit.unit_id: position for position, unit in enumerate(visible)}
        ranked_units = sorted(
            (
                (unit_id, score)
                for unit_id, score in exact_scores.items()
                if score > 0.0 and units_by_id[unit_id].record_id in candidate_set
            ),
            key=lambda item: (-item[1], visible_order[item[0]]),
        )

        selected: list[str] = []
        selected_set: set[str] = set()
        per_record: Counter[str] = Counter()
        for unit_id, _ in ranked_units:
            record_id = units_by_id[unit_id].record_id
            if per_record[record_id] >= self.config.first_pass_per_record:
                continue
            selected.append(unit_id)
            selected_set.add(unit_id)
            per_record[record_id] += 1
            if len(selected) == limit:
                return tuple(selected)

        # Backfill preserves relevance order when the diverse pass is short.
        for unit_id, _ in ranked_units:
            if unit_id in selected_set:
                continue
            selected.append(unit_id)
            selected_set.add(unit_id)
            if len(selected) == limit:
                break
        return tuple(selected)

    def _apply_meeting_context(
        self,
        units_by_record: Mapping[str, Sequence[MemoryUnit]],
        base_scores: Mapping[str, float],
        output_scores: dict[str, float],
    ) -> None:
        """Give meeting segments a small boost from nearby safe-text matches."""

        for record_units in units_by_record.values():
            if not record_units or record_units[0].source_type != "meeting":
                continue
            ordered = sorted(
                record_units,
                key=lambda unit: (
                    float(unit.metadata.get("segment_start_s", 0.0)),
                    unit.unit_id,
                ),
            )
            for position, unit in enumerate(ordered):
                direct_neighbors = [
                    base_scores.get(ordered[index].unit_id, 0.0)
                    for index in (position - 1, position + 1)
                    if 0 <= index < len(ordered)
                ]
                second_neighbors = [
                    base_scores.get(ordered[index].unit_id, 0.0)
                    for index in (position - 2, position + 2)
                    if 0 <= index < len(ordered)
                ]
                output_scores[unit.unit_id] = (
                    base_scores.get(unit.unit_id, 0.0)
                    + self.config.neighbor_weight * max(direct_neighbors, default=0.0)
                    + self.config.second_neighbor_weight
                    * max(second_neighbors, default=0.0)
                )

