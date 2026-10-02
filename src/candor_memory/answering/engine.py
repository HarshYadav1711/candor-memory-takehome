"""End-to-end grounded answering over frozen Phase 3 retrieval."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Literal

from ..models import MemoryUnit
from ..retrieval import HierarchicalRetriever, build_query_plan, derive_corpus_aliases
from ..visibility import visible_units
from .evidence import build_evidence_bundle
from .extractive import write_extractive_answer
from .model_writer import OpenAIAnswerWriter, OpenAIConfig, validate_model_output
from .models import EvidenceBundle, MemoryAnswer, SupportStatus
from .support import analyze_evidence, decide_support


AnswererMode = Literal["extractive", "openai", "auto"]
ModelWriter = Callable[[EvidenceBundle], object]


class GroundedMemory:
    def __init__(
        self,
        units: Iterable[MemoryUnit],
        *,
        model_writer: ModelWriter | None = None,
        openai_config: OpenAIConfig | None = None,
    ) -> None:
        self._units = tuple(units)
        self._retriever = HierarchicalRetriever(self._units)
        self._model_writer = model_writer
        self._openai_config = openai_config

    def answer(
        self,
        question: str,
        as_of: datetime,
        *,
        mode: AnswererMode = "extractive",
    ) -> tuple[MemoryAnswer, dict[str, object]]:
        if mode not in {"extractive", "openai", "auto"}:
            raise ValueError(f"unsupported answerer mode: {mode}")
        retrieved, retrieval_debug = self._retriever.retrieve_with_debug(question, as_of)
        visible = visible_units(self._units, as_of)
        plan = build_query_plan(
            question,
            as_of,
            aliases=derive_corpus_aliases(visible),
        )
        bundle = build_evidence_bundle(
            self._units,
            retrieved,
            question,
            as_of,
            plan,
            limit=20,
        )
        analysis = analyze_evidence(bundle)
        decision = decide_support(bundle, analysis)
        debug = {
            **retrieval_debug,
            "support": {
                "status": decision.status.value,
                "reason": decision.reason,
                "signals": dict(decision.signals),
            },
            "answerer": mode,
            "frame": {
                "answer_kind": analysis.frame.answer_kind.value,
                "relations": list(analysis.frame.relation_terms),
                "modality": analysis.frame.modality_requirement.value,
            },
        }
        if decision.status == SupportStatus.INSUFFICIENT_EVIDENCE:
            return MemoryAnswer("I don't know.", (), retrieved, True), debug

        fallback = write_extractive_answer(bundle, analysis)
        draft = fallback
        writer = self._model_writer
        if writer is None:
            config = self._openai_config or OpenAIConfig.from_environment()
            if config is not None:
                writer = OpenAIAnswerWriter(config)
        if mode == "openai" and writer is None:
            raise ValueError(
                "openai answerer requires OPENAI_API_KEY and OPENAI_MODEL"
            )
        if mode in {"openai", "auto"} and writer is not None:
            try:
                model_value = writer(bundle)
                draft = validate_model_output(model_value, bundle) or fallback
            except Exception:
                draft = fallback

        if not draft.source_ids or draft.answer == "I don't know.":
            return MemoryAnswer("I don't know.", (), retrieved, True), debug
        return MemoryAnswer(draft.answer, draft.source_ids, retrieved, False), debug

