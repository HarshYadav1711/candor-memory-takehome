"""Grounded answering, abstention, and source selection."""

from .engine import AnswererMode, GroundedMemory
from .evidence import build_evidence_bundle
from .extractive import write_extractive_answer
from .model_writer import OpenAIAnswerWriter, OpenAIConfig, validate_model_output
from .models import (
    AnswerDraft,
    EvidenceBundle,
    EvidenceItem,
    MemoryAnswer,
    SupportDecision,
    SupportStatus,
)
from .sources import select_sources
from .support import decide_support

__all__ = [
    "AnswerDraft",
    "AnswererMode",
    "EvidenceBundle",
    "EvidenceItem",
    "GroundedMemory",
    "MemoryAnswer",
    "OpenAIAnswerWriter",
    "OpenAIConfig",
    "SupportDecision",
    "SupportStatus",
    "build_evidence_bundle",
    "decide_support",
    "select_sources",
    "validate_model_output",
    "write_extractive_answer",
]

