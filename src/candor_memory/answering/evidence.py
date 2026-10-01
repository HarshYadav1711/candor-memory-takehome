"""Build bounded, sanitized evidence bundles from frozen retrieval results."""

from __future__ import annotations

from datetime import datetime
from types import MappingProxyType
from typing import Iterable, Mapping

from ..models import MemoryUnit
from ..retrieval import QueryPlan
from ..safety import sanitize_text
from ..visibility import visible_units
from .models import EvidenceBundle, EvidenceItem


_METADATA_FIELDS = (
    "delivery_state",
    "target_app",
    "target_context",
    "meeting_title",
    "meeting_start",
    "meeting_end",
    "location",
    "start",
    "end",
    "status",
    "organizer",
    "attendees",
    "conversation_title",
    "repo",
    "applied_edit_id",
    "applied_edit_at",
)


def _safe_value(value: object) -> object:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Mapping):
        return {str(key): _safe_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(child) for child in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return sanitize_text(str(value))


def build_evidence_bundle(
    units: Iterable[MemoryUnit],
    retrieved: Iterable[str],
    question: str,
    as_of: datetime,
    plan: QueryPlan,
    *,
    limit: int = 10,
) -> EvidenceBundle:
    """Project retrieved IDs onto visible safe units, preserving frozen order."""

    if not 1 <= limit <= 20:
        raise ValueError("evidence limit must be between 1 and 20")
    retrieved_ids = tuple(dict.fromkeys(retrieved))
    visible = {unit.unit_id: unit for unit in visible_units(units, as_of)}
    items: list[EvidenceItem] = []
    for rank, unit_id in enumerate(retrieved_ids[:limit], start=1):
        unit = visible.get(unit_id)
        if unit is None:
            continue
        metadata = {
            field: _safe_value(unit.metadata[field])
            for field in _METADATA_FIELDS
            if field in unit.metadata and unit.metadata[field] is not None
        }
        items.append(
            EvidenceItem(
                unit_id=unit.unit_id,
                record_id=unit.record_id,
                source_type=unit.source_type,
                available_at=unit.available_at,
                actor=sanitize_text(unit.actor) if unit.actor else None,
                safe_text=unit.safe_text,
                metadata=MappingProxyType(metadata),
                retrieval_rank=rank,
            )
        )
    return EvidenceBundle(
        question=question,
        as_of=as_of,
        plan=plan,
        retrieved=retrieved_ids,
        items=tuple(items),
    )

