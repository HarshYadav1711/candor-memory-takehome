"""Explicit canonical model shared by ingestion and visibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class MemoryUnit:
    """One independently citable unit from a source record.

    ``text`` retains the canonical raw source content for traceability.
    Retrieval and answer code must consume ``safe_text`` instead. Slack edit
    and deletion records use ``event_type`` and ``target_id`` to preserve the
    relationship without changing the original source unit.
    """

    unit_id: str
    record_id: str
    source_type: str
    available_at: datetime
    text: str
    safe_text: str
    actor: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    event_type: Optional[str] = None
    target_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.unit_id or not self.record_id:
            raise ValueError("unit_id and record_id must be non-empty")
        if self.available_at.tzinfo is None or self.available_at.utcoffset() is None:
            raise ValueError(f"available_at must be timezone-aware for {self.unit_id}")
        if self.event_type in {"message_changed", "message_deleted"} and not self.target_id:
            raise ValueError(f"{self.event_type} requires target_id for {self.unit_id}")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

