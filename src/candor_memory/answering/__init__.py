"""Grounded answering, abstention, and source selection."""

from .engine import AnswererMode, GroundedMemory
from .evidence import build_evidence_bundle
from .extractive import write_extractive_answer
from .frames import AnswerKind, ModalityRequirement, QuestionFrame, parse_question_frame
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
from .support import EvidenceAnalysis, analyze_evidence, decide_support

__all__ = [
    "AnswerDraft",
    "AnswerKind",
    "AnswererMode",
    "EvidenceBundle",
    "EvidenceItem",
    "EvidenceAnalysis",
    "GroundedMemory",
    "MemoryAnswer",
    "ModalityRequirement",
    "OpenAIAnswerWriter",
    "OpenAIConfig",
    "SupportDecision",
    "SupportStatus",
    "QuestionFrame",
    "analyze_evidence",
    "build_evidence_bundle",
    "decide_support",
    "select_sources",
    "parse_question_frame",
    "validate_model_output",
    "write_extractive_answer",
]

