"""Deterministic query planning for source, entity, intent, and time hints."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from ..models import MemoryUnit
from ..safety import sanitize_text
from .tokenize import tokenize


LOCAL_TIMEZONE = ZoneInfo("America/Los_Angeles")


class Intent(str, Enum):
    CURRENT_STATE = "CURRENT_STATE"
    HISTORICAL_STATE = "HISTORICAL_STATE"
    ATTRIBUTION = "ATTRIBUTION"
    COMMITMENT = "COMMITMENT"
    CAUSAL_CHANGE = "CAUSAL_CHANGE"
    DISAGREEMENT = "DISAGREEMENT"
    EVENT_LOOKUP = "EVENT_LOOKUP"
    TEMPORAL_ANCHOR = "TEMPORAL_ANCHOR"


class TemporalMode(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class QueryPlan:
    original_question: str
    reference_time: datetime
    lexical_terms: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    source_hints: tuple[str, ...]
    entity_hints: tuple[str, ...]
    speaker_hints: tuple[str, ...]
    explicit_dates: tuple[date, ...]
    relative_dates: tuple[tuple[str, date], ...]
    temporal_mode: TemporalMode
    intent_flags: tuple[Intent, ...]
    anchor_query: str | None = None
    target_query: str | None = None

    def debug_dict(self) -> dict[str, object]:
        return {
            "original_question": self.original_question,
            "lexical_terms": list(self.lexical_terms),
            "expanded_terms": list(self.expanded_terms),
            "source_hints": list(self.source_hints),
            "entity_hints": list(self.entity_hints),
            "speaker_hints": list(self.speaker_hints),
            "explicit_dates": [value.isoformat() for value in self.explicit_dates],
            "relative_dates": [
                {"expression": expression, "date": value.isoformat()}
                for expression, value in self.relative_dates
            ],
            "temporal_mode": self.temporal_mode.value,
            "intent_flags": [flag.value for flag in self.intent_flags],
            "anchor_query": self.anchor_query,
            "target_query": self.target_query,
        }


_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
        "for", "from", "had", "has", "have", "how", "i", "in", "is", "it", "me",
        "my", "of", "on", "or", "our", "the", "their", "them", "this", "to", "was",
        "we", "were", "what", "when", "where", "which", "who", "with", "would",
    }
)
_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_SOURCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("dictation", re.compile(r"\bdictat(?:e|ed|es|ing|ion|ions)?\b", re.I)),
    ("slack", re.compile(r"\bslack(?:ed|ing)?\b|\bin slack\b", re.I)),
    ("gmail", re.compile(r"\b(?:e-?mail|emailed|emailing|gmail)\b", re.I)),
    ("google_calendar", re.compile(r"\bcalendar\b|\bscheduled\b", re.I)),
    ("meeting", re.compile(r"\bmeeting\b|\bcall\b", re.I)),
    ("chatgpt", re.compile(r"\bchatgpt\b", re.I)),
    ("codex", re.compile(r"\bcodex\b", re.I)),
)
_QUESTION_WORDS = {
    "Are", "Did", "Do", "Does", "Has", "How", "In", "Is", "On", "The", "What",
    "When", "Where", "Which", "Who", "Why",
}
_SPEAKER_PATTERNS = (
    re.compile(r"\bwhat did\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:say|tell|promise)", re.I),
    re.compile(r"\bdid\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:say|agree|promise|commit)", re.I),
    re.compile(r"\baccording to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.I),
)
_ANCHOR_PATTERNS = (
    re.compile(r"^(?P<target>.*?)\bthe day\s+(?:that\s+)?(?P<anchor>(?:i|we)\s+.+?)[?.!]*$", re.I),
    re.compile(r"^(?P<target>.*?)\bon the day of\s+(?P<anchor>.+?)[?.!]*$", re.I),
    re.compile(r"^(?P<target>.*?)\bthe same day as\s+(?P<anchor>.+?)[?.!]*$", re.I),
)
_IRREGULAR_VARIANTS: Mapping[str, tuple[str, ...]] = {
    "did": ("do",),
    "done": ("do",),
    "flew": ("fly", "flight"),
    "fly": ("flight",),
    "flight": ("fly",),
    "said": ("say",),
    "sent": ("send",),
    "told": ("tell",),
    "went": ("go",),
}


def informative_terms(tokens: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            token
            for token in tokens
            if token not in _STOPWORDS
            and not (token.endswith("'s") and token[:-2] in _STOPWORDS)
        )
    )


def term_variants(term: str) -> tuple[str, ...]:
    """Return conservative English morphology variants for one token."""

    variants: dict[str, None] = {term: None}
    for value in _IRREGULAR_VARIANTS.get(term, ()):
        variants[value] = None
    base = term[:-2] if term.endswith("'s") and len(term) > 3 else term
    variants[base] = None
    if len(base) >= 5 and base.endswith("ies"):
        variants[base[:-3] + "y"] = None
    elif len(base) >= 5 and base.endswith("s") and not base.endswith(("ss", "us", "is")):
        variants[base[:-1]] = None
    if len(base) >= 6 and base.endswith("ing"):
        stem = base[:-3]
        variants[stem] = None
        variants[stem + "e"] = None
        if len(stem) > 2 and stem[-1] == stem[-2]:
            variants[stem[:-1]] = None
    if len(base) >= 5 and base.endswith("ed"):
        stem = base[:-2]
        variants[stem] = None
        variants[stem + "e"] = None
        if len(stem) > 2 and stem[-1] == stem[-2]:
            variants[stem[:-1]] = None
    if len(base) >= 7 and base.endswith("ation"):
        variants[base[:-5] + "ate"] = None
    return tuple(value for value in variants if len(value) > 1)


def normalize_terms(tokens: Sequence[str]) -> tuple[str, ...]:
    expanded: dict[str, None] = {}
    for token in tokens:
        for variant in term_variants(token):
            expanded.setdefault(variant, None)
    return tuple(expanded)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_explicit_dates(text: str, reference: datetime) -> tuple[date, ...]:
    """Extract ISO, month-name, and numeric dates using ``reference`` for year."""

    local_reference = reference.astimezone(LOCAL_TIMEZONE)
    positioned: list[tuple[int, date]] = []
    for match in re.finditer(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text):
        value = _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if value:
            positioned.append((match.start(), value))
    month_pattern = "|".join(sorted(_MONTHS, key=len, reverse=True))
    for match in re.finditer(
        rf"\b({month_pattern})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d{{2}}))?\b",
        text,
        re.I,
    ):
        year = int(match.group(3)) if match.group(3) else local_reference.year
        value = _safe_date(year, _MONTHS[match.group(1).lower()], int(match.group(2)))
        if value:
            positioned.append((match.start(), value))
    for match in re.finditer(r"(?<!\d)(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?(?!\d)", text):
        year_text = match.group(3)
        year = local_reference.year if not year_text else int(year_text)
        if year < 100:
            year += 2000
        value = _safe_date(year, int(match.group(1)), int(match.group(2)))
        if value:
            positioned.append((match.start(), value))
    found: dict[date, None] = {}
    for _, value in sorted(positioned, key=lambda item: item[0]):
        found.setdefault(value, None)
    return tuple(found)


def resolve_relative_dates(text: str, reference: datetime) -> tuple[tuple[str, date], ...]:
    local_date = reference.astimezone(LOCAL_TIMEZONE).date()
    resolved: list[tuple[str, date]] = []
    lowered = text.casefold()
    for expression, offset in (("today", 0), ("yesterday", -1), ("tomorrow", 1)):
        if re.search(rf"\b{expression}\b", lowered):
            resolved.append((expression, local_date + timedelta(days=offset)))
    for match in re.finditer(
        r"\b(last|this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        lowered,
    ):
        modifier, weekday_name = match.groups()
        target = _WEEKDAYS[weekday_name]
        week_start = local_date - timedelta(days=local_date.weekday())
        if modifier == "this":
            value = week_start + timedelta(days=target)
        elif modifier == "last":
            value = week_start - timedelta(days=7) + timedelta(days=target)
        else:
            value = week_start + timedelta(days=7 + target)
        resolved.append((match.group(0), value))
    return tuple(resolved)


def date_terms(value: date) -> tuple[str, ...]:
    month = _MONTH_NAMES[value.month - 1]
    return (
        value.isoformat(),
        str(value.year),
        month,
        month[:3],
        str(value.month),
        f"{value.month:02d}",
        str(value.day),
        f"{value.day:02d}",
        value.strftime("%A").casefold(),
    )


def _detect_anchor(question: str) -> tuple[str | None, str | None]:
    for pattern in _ANCHOR_PATTERNS:
        match = pattern.search(question.strip())
        if match:
            return match.group("anchor").strip(), match.group("target").strip()
    return None, None


def _detect_entities(question: str) -> tuple[str, ...]:
    found: dict[str, None] = {}
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", question):
        parts = match.group(0).split()
        if parts and parts[0] in _QUESTION_WORDS:
            parts = parts[1:]
        value = " ".join(parts)
        if value and value.casefold() not in _MONTHS:
            found.setdefault(value, None)
    for match in re.finditer(r"\b[A-Z][A-Z0-9]{1,5}\b", question):
        found.setdefault(match.group(0), None)
    return tuple(found)


def _detect_speakers(question: str) -> tuple[str, ...]:
    if re.search(r"\bwhat did i\s+(?:say|tell|dictate|send|promise)", question, re.I):
        return ("__self__",)
    for pattern in _SPEAKER_PATTERNS:
        match = pattern.search(question)
        if match:
            return (match.group(1),)
    return ()


def _detect_intents(question: str, anchor_query: str | None) -> tuple[Intent, ...]:
    lowered = question.casefold()
    flags: list[Intent] = []
    if re.search(r"\b(now|currently|still|latest|as of|this year|going to)\b|what(?:'s| is)", lowered):
        flags.append(Intent.CURRENT_STATE)
    if re.search(r"\b(last|then|before|after|previously|was|were)\b", lowered):
        flags.append(Intent.HISTORICAL_STATE)
    if re.search(r"\b(say|said|tell|told|according|agree|reported)\b", lowered):
        flags.append(Intent.ATTRIBUTION)
    if re.search(r"\b(promise|promised|commit|committed|owe|due|send|sent|sign|close|done|land)\b", lowered):
        flags.append(Intent.COMMITMENT)
    if re.search(r"\bwhy\b|\b(cause|caused|change|changed|move|moved|slip|slipped|delay|delayed)\b", lowered):
        flags.append(Intent.CAUSAL_CHANGE)
    if re.search(r"\b(disagree|disagreed|agree|agreed|versus|vs\.?|conflict)\b", lowered):
        flags.append(Intent.DISAGREEMENT)
    if re.search(r"\b(when|calendar|scheduled|meeting|call|flight|event|leave|depart)\b", lowered):
        flags.append(Intent.EVENT_LOOKUP)
    if anchor_query:
        flags.append(Intent.TEMPORAL_ANCHOR)
    return tuple(dict.fromkeys(flags))


def derive_corpus_aliases(units: Iterable[MemoryUnit]) -> dict[str, tuple[str, ...]]:
    """Derive corroborated acronym-to-name aliases from sanitized corpus data."""

    units = tuple(units)
    observed_acronyms: set[str] = set()
    direct_candidates: dict[str, set[tuple[str, ...]]] = {}
    metadata_values: list[str] = []
    for unit in units:
        safe = unit.safe_text
        observed_acronyms.update(match.group(0).casefold() for match in re.finditer(r"\b[A-Z]{2,6}\b", safe))
        for match in re.finditer(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})\s*\(([A-Z]{2,6})\)", safe):
            direct_candidates.setdefault(match.group(2).casefold(), set()).add(tokenize(match.group(1)))
        for field in ("meeting_title", "conversation_title", "channel_name", "repo"):
            value = unit.metadata.get(field)
            if isinstance(value, str) and value:
                metadata_values.append(sanitize_text(value).replace("-", " "))

    candidates = dict(direct_candidates)
    for value in metadata_values:
        words = [word for word in tokenize(value) if word not in _STOPWORDS and not word.isdigit()]
        for size in range(2, min(5, len(words) + 1)):
            for start in range(0, len(words) - size + 1):
                phrase = tuple(words[start : start + size])
                acronym = "".join(word[0] for word in phrase)
                if acronym in observed_acronyms:
                    candidates.setdefault(acronym, set()).add(phrase)
    return {
        acronym: next(iter(phrases))
        for acronym, phrases in sorted(candidates.items())
        if len(phrases) == 1
    }


def expand_aliases(tokens: Sequence[str], aliases: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    expanded: dict[str, None] = dict.fromkeys(tokens)
    token_set = set(tokens)
    for acronym, phrase in aliases.items():
        if acronym in token_set:
            for token in phrase:
                expanded.setdefault(token, None)
        if phrase and set(phrase) <= token_set:
            expanded.setdefault(acronym, None)
    return tuple(expanded)


def build_query_plan(
    question: str,
    as_of: datetime,
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> QueryPlan:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    lexical = tokenize(question)
    content_terms = informative_terms(lexical)
    expanded = normalize_terms(content_terms)
    if aliases:
        expanded = expand_aliases(expanded, aliases)
    relative = resolve_relative_dates(question, as_of)
    dates = tuple(dict.fromkeys((*extract_explicit_dates(question, as_of), *(value for _, value in relative))))
    for value in dates:
        expanded = tuple(dict.fromkeys((*expanded, *normalize_terms(date_terms(value)))))
    source_hints = tuple(source for source, pattern in _SOURCE_PATTERNS if pattern.search(question))
    anchor_query, target_query = _detect_anchor(question)
    intents = _detect_intents(question, anchor_query)
    if Intent.CURRENT_STATE in intents:
        temporal_mode = TemporalMode.CURRENT
    elif Intent.HISTORICAL_STATE in intents or dates:
        temporal_mode = TemporalMode.HISTORICAL
    else:
        temporal_mode = TemporalMode.UNSPECIFIED
    return QueryPlan(
        original_question=question,
        reference_time=as_of,
        lexical_terms=lexical,
        expanded_terms=expanded,
        source_hints=source_hints,
        entity_hints=_detect_entities(question),
        speaker_hints=_detect_speakers(question),
        explicit_dates=dates,
        relative_dates=relative,
        temporal_mode=temporal_mode,
        intent_flags=intents,
        anchor_query=anchor_query,
        target_query=target_query,
    )


def with_added_date(plan: QueryPlan, value: date) -> QueryPlan:
    dates = tuple(dict.fromkeys((*plan.explicit_dates, value)))
    expanded = tuple(dict.fromkeys((*plan.expanded_terms, *normalize_terms(date_terms(value)))))
    return replace(plan, explicit_dates=dates, expanded_terms=expanded)

