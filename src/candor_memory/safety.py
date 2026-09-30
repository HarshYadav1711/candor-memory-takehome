"""Deterministic safety transformations for untrusted source content."""

from __future__ import annotations

import re


REDACTION = "[REDACTED]"
UNTRUSTED_COMMENT = "[UNTRUSTED COMMENT REMOVED]"

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_OPENAI_STYLE_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_SLACK_TOKEN = re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{8,}\b")
_GITHUB_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")
_AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<label>api[ _-]?key|password|passwd|pwd|secret|"
    r"access[ _-]?token|auth[ _-]?token|credential)\b"
    r"(?P<separator>\s*[:=]\s*)"
    r"(?P<quote>['\"]?)"
    r"(?P<value>[^\s'\"`,;<>]+)"
    r"(?P=quote)"
)


def _redact_assignment(match: re.Match[str]) -> str:
    return f"{match.group('label')}{match.group('separator')}{REDACTION}"


def sanitize_text(text: str) -> str:
    """Return deterministic, non-executable text safe for later retrieval.

    All HTML comments are removed rather than trying to decide which comments
    are instructions. Common credential formats are redacted without relying
    on an LLM or on knowledge of the dataset's specific secret.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    safe = _HTML_COMMENT.sub(UNTRUSTED_COMMENT, text)
    safe = _PRIVATE_KEY.sub(REDACTION, safe)
    safe = _BEARER_TOKEN.sub(f"Bearer {REDACTION}", safe)
    safe = _OPENAI_STYLE_KEY.sub(REDACTION, safe)
    safe = _SLACK_TOKEN.sub(REDACTION, safe)
    safe = _GITHUB_TOKEN.sub(REDACTION, safe)
    safe = _AWS_ACCESS_KEY.sub(REDACTION, safe)
    safe = _JWT.sub(REDACTION, safe)
    safe = _CREDENTIAL_ASSIGNMENT.sub(_redact_assignment, safe)
    return safe

