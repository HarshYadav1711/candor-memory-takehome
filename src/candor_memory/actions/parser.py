"""Strict JSONL parsing and conservative atomic-command decomposition."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from ..ingest import parse_datetime
from .models import ActionCommand


class ActionCommandFormatError(ValueError):
    """Raised when an action command violates the runtime contract."""


def parse_action_command(value: Mapping[str, Any]) -> ActionCommand:
    if not isinstance(value, Mapping):
        raise ActionCommandFormatError("action command must be a JSON object")
    command_id = value.get("id")
    command = value.get("command")
    as_of = value.get("as_of")
    if not isinstance(command_id, str) or not command_id.strip():
        raise ActionCommandFormatError("field 'id' must be a non-empty string")
    if not isinstance(command, str) or not command.strip():
        raise ActionCommandFormatError("field 'command' must be a non-empty string")
    if not isinstance(as_of, str) or not as_of.strip():
        raise ActionCommandFormatError("field 'as_of' must be an ISO-8601 string")
    try:
        timestamp = parse_datetime(as_of)
    except (TypeError, ValueError) as exc:
        raise ActionCommandFormatError(f"field 'as_of' is invalid: {exc}") from exc
    # Deliberately retain only runtime fields. Evaluation annotations never enter the model.
    return ActionCommand(command_id.strip(), command.strip(), timestamp)


def load_action_commands(path: str | Path) -> tuple[ActionCommand, ...]:
    input_path = Path(path)
    commands: list[ActionCommand] = []
    seen: set[str] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ActionCommandFormatError(
                    f"{input_path}: line {line_number}: invalid JSON"
                ) from exc
            try:
                command = parse_action_command(value)
            except ActionCommandFormatError as exc:
                raise ActionCommandFormatError(
                    f"{input_path}: line {line_number}: {exc}"
                ) from exc
            if command.command_id in seen:
                raise ActionCommandFormatError(
                    f"{input_path}: line {line_number}: duplicate action id {command.command_id!r}"
                )
            seen.add(command.command_id)
            commands.append(command)
    return tuple(commands)


_ATOMIC_START = (
    r"email|message|tell|thank|remind|book|schedule|move|reschedule|"
    r"open|delete|remove|wipe|clear"
)


def decompose_command(command: str) -> tuple[str, ...]:
    """Split only where a conjunction introduces another supported imperative."""

    pieces = re.split(rf"\s+and\s+(?=(?:{_ATOMIC_START})\b)", command.strip(), flags=re.I)
    return tuple(piece.strip(" ,") for piece in pieces if piece.strip(" ,"))

