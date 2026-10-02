"""Immutable models shared by the grounded answering pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Mapping

from ..retrieval import QueryPlan


class SupportStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class EvidenceItem:
    unit_id: str
    record_id: str
    source_type: str
    available_at: datetime
    actor: str | None
    safe_text: str
    metadata: Mapping[str, object]
    retrieval_rank: int


@dataclass(frozen=True)
class EvidenceBundle:
    question: str
    as_of: datetime
    plan: QueryPlan
    retrieved: tuple[str, ...]
    items: tuple[EvidenceItem, ...]


@dataclass(frozen=True)
class SupportDecision:
    status: SupportStatus
    reason: str
    signals: Mapping[str, object]


@dataclass(frozen=True)
class AnswerDraft:
    answer: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class MemoryAnswer:
    answer: str
    sources: tuple[str, ...]
    retrieved: tuple[str, ...]
    abstained: bool

