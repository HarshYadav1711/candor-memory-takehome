"""Fail-closed destructive-action and untrusted-content safety rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .frames import ActionFrame

_UNSAFE_MARKERS = ("[REDACTED]", "[UNTRUSTED COMMENT REMOVED]")


class SafetyStatus(str, Enum):
    SAFE = "safe"
    CONFIRM = "confirm"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class SafetyDecision:
    status: SafetyStatus
    reason: str


def assess_safety(frame: ActionFrame) -> SafetyDecision:
    """Apply one fail-closed policy before any ordinary action dispatch."""
    if not frame.destructive:
        return SafetyDecision(SafetyStatus.SAFE, "non_destructive")

    text = frame.normalized_command.casefold()
    if frame.object_type == "attendee" or re.search(r"\bfrom\s+(?:the\s+)?attendee", text):
        return SafetyDecision(SafetyStatus.CLARIFY, "attendee_update_not_event_deletion")
    if frame.object_type == "draft" or re.search(r"\b(?:wording|copy|text)\s+from\b", text):
        return SafetyDecision(SafetyStatus.CLARIFY, "content_edit_requires_clarification")
    if frame.object_type == "reminder":
        return SafetyDecision(SafetyStatus.CLARIFY, "reminder_cancellation_unsupported")
    if frame.object_type in {"event", "email", "message", "file", "record", "account", "history", "data"}:
        return SafetyDecision(SafetyStatus.CONFIRM, f"destructive_{frame.object_type}")
    if frame.irreversible or re.search(r"\b(?:everything|all|every)\b", text):
        return SafetyDecision(SafetyStatus.CONFIRM, "broad_destructive_scope")
    return SafetyDecision(SafetyStatus.CLARIFY, "unclear_removal_scope")


def needs_confirmation(command: str) -> bool:
    from .frames import parse_action_frame

    return assess_safety(parse_action_frame(command)).status == SafetyStatus.CONFIRM


def confirmation_summary(command: str) -> str:
    concise = re.sub(r"\s+", " ", command).strip().rstrip(".?!")
    return f"Confirm destructive request: {concise}."


def is_safe_grounded_text(text: str) -> bool:
    return not any(marker.casefold() in text.casefold() for marker in _UNSAFE_MARKERS)

