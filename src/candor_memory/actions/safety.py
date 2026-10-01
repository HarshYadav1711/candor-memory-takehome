"""Fail-closed destructive-action and untrusted-content safety rules."""

from __future__ import annotations

import re


_DESTRUCTIVE = re.compile(
    r"\b(?:delete|erase|destroy|wipe|purge|clear|remove)\b.*"
    r"\b(?:email|mail|message|event|calendar|file|folder|record|data|account|history)s?\b|"
    r"\b(?:wipe|purge|clear)\s+(?:everything|all)\b",
    re.I,
)
_CONFIRMED = re.compile(r"\b(?:i confirm|confirmed|yes,?\s+delete|already confirmed)\b", re.I)
_UNSAFE_MARKERS = ("[REDACTED]", "[UNTRUSTED COMMENT REMOVED]")


def needs_confirmation(command: str) -> bool:
    return bool(_DESTRUCTIVE.search(command)) and not bool(_CONFIRMED.search(command))


def confirmation_summary(command: str) -> str:
    concise = re.sub(r"\s+", " ", command).strip().rstrip(".?!")
    return f"Confirm destructive request: {concise}."


def is_safe_grounded_text(text: str) -> bool:
    return not any(marker.casefold() in text.casefold() for marker in _UNSAFE_MARKERS)

