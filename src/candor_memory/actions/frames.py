"""Composable, immutable command understanding for dry-run actions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Operation(str, Enum):
    COMMUNICATE = "communicate"
    CREATE_EVENT = "create_event"
    UPDATE_EVENT = "update_event"
    CREATE_REMINDER = "create_reminder"
    QUERY_MEMORY = "query_memory"
    OPEN_APP = "open_app"
    DESTRUCTIVE = "destructive"
    UNKNOWN = "unknown"


class Channel(str, Enum):
    SLACK = "slack"
    EMAIL = "email"
    CALENDAR = "calendar"
    MEMORY = "memory"
    APP = "app"
    UNSPECIFIED = "unspecified"


_POLITE_PREFIX = re.compile(
    r"^(?:(?:please|kindly)\s+|(?:could|can|would|will)\s+you\s+)+",
    re.I,
)
_FILLER = re.compile(r"\b(?:real quick|for me|again)\b", re.I)
_DESTRUCTIVE_CUE = re.compile(
    r"\b(?:delet(?:e|ing)|eras(?:e|ing)|destroy(?:ing)?|wip(?:e|ing)|purg(?:e|ing)|"
    r"trash(?:ing)?|discard(?:ing)?|clear(?:ing)?(?:\s+out)?|remov(?:e|ing)|cancel(?:ling|ing)?|"
    r"shred(?:ding)?|obliterat(?:e|ing)|nuk(?:e|ing))\b|"
    r"\b(?:get\s+rid\s+of|throw\s+away)\b",
    re.I,
)
_QUERY_START = re.compile(
    r"^(?:what(?:'s|\s+is)?|who|when|where|why|how|which|do\s+we\s+know|"
    r"can\s+you\s+recall|recall)\b",
    re.I,
)
_COMMUNICATE_START = re.compile(
    r"^(?:message|tell|notify|inform|ping|thank|email|write(?:\s+an?\s+email)?(?:\s+to)?|"
    r"send|drop|let|reach\s+out\s+to)\b",
    re.I,
)
_CREATE_EVENT_START = re.compile(
    r"^(?:schedule|book|arrange|set\s+up|put\b.*\bon\s+(?:my\s+)?calendar|create\b.*\bmeeting)\b",
    re.I,
)
_UPDATE_EVENT_START = re.compile(
    r"^(?:move|reschedule|shift|change|push\s+back|bring\b.*\bforward)\b",
    re.I,
)
_REMINDER_START = re.compile(
    r"^(?:remind\s+me|do\s+not\s+let\s+me\s+forget|don't\s+let\s+me\s+forget|"
    r"give\s+me\s+(?:a\s+)?reminder|set\s+(?:a\s+)?reminder|ping\s+me)\b",
    re.I,
)
_OPEN_START = re.compile(r"^(?:open|launch|start|fire\s+up|bring\s+up)\b", re.I)
_TEMPORAL_CUE = re.compile(
    r"\b(?:today|tomorrow|tonight|next\s+\w+|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|at\s+\d|\d(?::\d{2})?\s*(?:am|pm)|before|after|minutes?|hours?)\b",
    re.I,
)
_ACTION_CUE = (
    r"message|tell|notify|inform|ping|thank|email|write|send|drop|let|reach\s+out|"
    r"schedule|book|arrange|set\s+up|put|create|move|reschedule|shift|change|push\s+back|"
    r"bring|remind|give|open|launch|start|fire\s+up|delete|erase|destroy|wipe|purge|trash|"
    r"discard|clear|remove|get\s+rid|throw\s+away|cancel|shred|nuke"
)


@dataclass(frozen=True)
class ActionFrame:
    original_command: str
    normalized_command: str
    operation: Operation
    communication_channel: Channel = Channel.UNSPECIFIED
    object_type: str = "unknown"
    target_phrase: str = ""
    content_phrase: str = ""
    event_phrase: str = ""
    temporal_phrase: str = ""
    duration: str = ""
    relative_offset: str = ""
    application_phrase: str = ""
    destructive: bool = False
    irreversible: bool = False
    requires_target: bool = False
    compound_clauses: tuple[str, ...] = ()


def normalize_command(command: str) -> str:
    text = command.strip().replace("’", "'")
    text = re.sub(r"\bdon't\b", "do not", text, flags=re.I)
    text = _POLITE_PREFIX.sub("", text)
    text = _FILLER.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def decompose_action_clauses(command: str) -> tuple[str, ...]:
    """Split only when a conjunction introduces another operation-bearing clause."""
    normalized = command.strip().replace("’", "'")
    normalized = _POLITE_PREFIX.sub("", normalized)
    pieces = re.split(
        rf"\s+and\s+(?=(?:(?:please|could\s+you|can\s+you|would\s+you)\s+)?(?:{_ACTION_CUE})\b)",
        normalized,
        flags=re.I,
    )
    return tuple(piece.strip(" ,") for piece in pieces if piece.strip(" ,"))


def _object_type(text: str) -> str:
    objects = (
        ("attendee", r"\battendee(?:s|\s+list)?\b"),
        ("draft", r"\b(?:draft|wording|copy)\b"),
        ("reminder", r"\breminder\b"),
        ("event", r"\b(?:calendar\s+event|event|meeting|appointment)\b"),
        ("email", r"\b(?:email|mail)s?\b"),
        ("message", r"\bmessages?\b"),
        ("file", r"\b(?:file|folder)s?\b"),
        ("record", r"\brecords?\b"),
        ("account", r"\baccount\b"),
        ("history", r"\bhistory\b"),
        ("data", r"\bdata\b"),
    )
    return next((name for name, pattern in objects if re.search(pattern, text, re.I)), "unknown")


def _channel(text: str) -> Channel:
    if re.search(r"\b(?:email|e-mail|gmail|by\s+mail)\b", text, re.I):
        return Channel.EMAIL
    if re.search(r"\bslack\b|#[a-z0-9_-]+|\bchannel\b", text, re.I):
        return Channel.SLACK
    return Channel.UNSPECIFIED


def _content_phrase(text: str) -> str:
    matches = list(re.finditer(r"\b(?:saying|that|about|to\s+say|asking\s+about)\s+(.+)$", text, re.I))
    if matches:
        return matches[-1].group(1).strip(" .")
    match = re.search(r"\bto\s+(thank\b.+)$|:\s*(.+)$", text, re.I)
    if match:
        return (match.group(1) or match.group(2) or "").strip(" .")
    return ""


def parse_action_frame(command: str) -> ActionFrame:
    normalized = normalize_command(command)
    clauses = decompose_action_clauses(normalized)
    destructive = bool(_DESTRUCTIVE_CUE.search(normalized))
    object_type = _object_type(normalized)
    channel = _channel(normalized)

    reminder_query = bool(
        re.match(r"^remind\s+me\s+(?:what|who|when|where|whether|how)\b", normalized, re.I)
    ) and not bool(_TEMPORAL_CUE.search(normalized))
    if destructive:
        operation = Operation.DESTRUCTIVE
    elif reminder_query or normalized.endswith("?") or _QUERY_START.match(normalized):
        operation = Operation.QUERY_MEMORY
        channel = Channel.MEMORY
    elif _REMINDER_START.match(normalized):
        operation = Operation.CREATE_REMINDER
    elif _CREATE_EVENT_START.match(normalized):
        operation = Operation.CREATE_EVENT
        channel = Channel.CALENDAR
    elif _UPDATE_EVENT_START.match(normalized):
        operation = Operation.UPDATE_EVENT
        channel = Channel.CALENDAR
    elif _OPEN_START.match(normalized):
        operation = Operation.OPEN_APP
        channel = Channel.APP
    elif _COMMUNICATE_START.match(normalized):
        operation = Operation.COMMUNICATE
    else:
        operation = Operation.UNKNOWN

    app_match = _OPEN_START.match(normalized)
    application = normalized[app_match.end():].strip(" .") if app_match else ""
    target = ""
    if operation == Operation.COMMUNICATE:
        prefix = _COMMUNICATE_START.sub("", normalized, count=1).strip()
        target = re.split(r"\b(?:on\s+slack|by\s+email|saying|that|about|to\s+say)\b", prefix, maxsplit=1, flags=re.I)[0]
        target = re.sub(r"^(?:an?\s+email\s+to|a\s+(?:message|note)\s+to|this\s+to)\s+", "", target, flags=re.I)
        target = re.sub(r"\s+(?:an?\s+email|a\s+(?:message|note))$", "", target, flags=re.I).strip(" ,")

    irreversible = destructive and bool(
        re.search(r"\b(?:all|every|permanent(?:ly)?|account|history|records?|data)\b", normalized, re.I)
    )
    return ActionFrame(
        original_command=command,
        normalized_command=normalized,
        operation=operation,
        communication_channel=channel,
        object_type=object_type,
        target_phrase=target,
        content_phrase=_content_phrase(normalized),
        event_phrase=normalized if operation in {Operation.CREATE_EVENT, Operation.UPDATE_EVENT} else "",
        temporal_phrase=normalized if _TEMPORAL_CUE.search(normalized) else "",
        duration=(re.search(r"\b(?:\d+|one|an?)\s+(?:minutes?|hours?)\b", normalized, re.I) or [""])[0],
        relative_offset=(re.search(r"\b(?:\d+|one|an?)\s+(?:minutes?|hours?)\s+before\b", normalized, re.I) or [""])[0],
        application_phrase=application,
        destructive=destructive,
        irreversible=irreversible,
        requires_target=operation in {Operation.COMMUNICATE, Operation.CREATE_EVENT},
        compound_clauses=clauses,
    )
