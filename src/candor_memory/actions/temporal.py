"""Relative date, clock-time, and duration resolution from command as_of."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from ..retrieval.planner import extract_explicit_dates, resolve_relative_dates


LOCAL = ZoneInfo("America/Los_Angeles")
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
}


class TemporalResolutionError(ValueError):
    pass


def local_reference(as_of: datetime) -> datetime:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise TemporalResolutionError("as_of must be timezone-aware")
    return as_of.astimezone(LOCAL)


def resolve_date(text: str, as_of: datetime, *, default: date | None = None) -> date | None:
    reference = local_reference(as_of)
    relative = resolve_relative_dates(text, reference)
    if relative:
        return relative[0][1]
    explicit = extract_explicit_dates(text, reference)
    if explicit:
        return explicit[0]
    ordinal = re.search(r"\b(?:on\s+)?the\s+(\d{1,2})(?:st|nd|rd|th)\b", text, re.I)
    if ordinal:
        day = int(ordinal.group(1))
        year, month = reference.year, reference.month
        try:
            candidate = date(year, month, day)
        except ValueError as exc:
            raise TemporalResolutionError(f"invalid day of month: {day}") from exc
        if candidate < reference.date():
            month += 1
            if month == 13:
                year, month = year + 1, 1
            candidate = date(year, month, day)
        return candidate
    bare_weekday = re.search(
        r"\b(?:on\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        text,
        re.I,
    )
    if bare_weekday:
        target = _WEEKDAYS[bare_weekday.group(1).casefold()]
        delta = (target - reference.weekday()) % 7
        return reference.date() + timedelta(days=delta)
    return default


def resolve_time(text: str) -> time | None:
    match = re.search(
        r"\b(?:at|to)\s*(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?\b|"
        r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b",
        text,
        re.I,
    )
    if not match:
        return None
    hour = int(match.group(1) or match.group(4))
    minute = int(match.group(2) or match.group(5) or 0)
    marker = (match.group(3) or match.group(6) or "").casefold().replace(".", "")
    if marker == "pm" and hour < 12:
        hour += 12
    elif marker == "am" and hour == 12:
        hour = 0
    elif not marker and 1 <= hour <= 6:
        hour += 12
    if hour > 23 or minute > 59:
        raise TemporalResolutionError("invalid clock time")
    return time(hour, minute)


def resolve_datetime(
    text: str,
    as_of: datetime,
    *,
    default_date: date | None = None,
) -> datetime | None:
    day = resolve_date(text, as_of, default=default_date)
    clock = resolve_time(text)
    if day is None or clock is None:
        return None
    return datetime.combine(day, clock, LOCAL)


def parse_duration(text: str) -> timedelta | None:
    match = re.search(
        r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight)\s*"
        r"(minutes?|mins?|hours?|hrs?)\b",
        text,
        re.I,
    )
    if not match:
        return None
    token = match.group(1).casefold()
    value = int(token) if token.isdigit() else _NUMBER_WORDS[token]
    return timedelta(hours=value) if match.group(2).casefold().startswith(("hour", "hr")) else timedelta(minutes=value)


def iso_local(value: datetime) -> str:
    return value.astimezone(LOCAL).isoformat()

