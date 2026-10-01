"""Strict parsing for the public memory-question contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .ingest import parse_datetime


class QuestionFormatError(ValueError):
    """Raised when a memory question does not satisfy the public contract."""


@dataclass(frozen=True)
class MemoryQuestion:
    question_id: str
    question: str
    as_of: datetime


def parse_question(value: Mapping[str, Any]) -> MemoryQuestion:
    """Extract only ``id``, ``question``, and ``as_of`` from one object.

    Evaluation annotations and all other extra fields are intentionally ignored
    and are not retained on the returned model.
    """

    if not isinstance(value, Mapping):
        raise QuestionFormatError("question must be a JSON object")

    question_id = value.get("id")
    question = value.get("question")
    as_of = value.get("as_of")
    if not isinstance(question_id, str) or not question_id.strip():
        raise QuestionFormatError("question field 'id' must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise QuestionFormatError("question field 'question' must be a non-empty string")
    if not isinstance(as_of, str) or not as_of.strip():
        raise QuestionFormatError("question field 'as_of' must be an ISO-8601 string")
    try:
        timestamp = parse_datetime(as_of)
    except (TypeError, ValueError) as exc:
        raise QuestionFormatError(f"question field 'as_of' is invalid: {exc}") from exc

    return MemoryQuestion(
        question_id=question_id,
        question=question,
        as_of=timestamp,
    )


def load_questions(path: str | Path) -> tuple[MemoryQuestion, ...]:
    """Load JSONL questions in input order and reject ambiguous duplicate IDs."""

    input_path = Path(path)
    questions: list[MemoryQuestion] = []
    seen: set[str] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QuestionFormatError(
                    f"{input_path}: line {line_number}: invalid JSON"
                ) from exc
            try:
                parsed = parse_question(value)
            except QuestionFormatError as exc:
                raise QuestionFormatError(
                    f"{input_path}: line {line_number}: {exc}"
                ) from exc
            if parsed.question_id in seen:
                raise QuestionFormatError(
                    f"{input_path}: line {line_number}: duplicate question id "
                    f"{parsed.question_id!r}"
                )
            seen.add(parsed.question_id)
            questions.append(parsed)
    return tuple(questions)

