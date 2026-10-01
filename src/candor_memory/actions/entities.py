"""Dataset-derived contact and Slack-channel directory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ..ingest import parse_datetime
from ..models import MemoryUnit
from ..retrieval.tokenize import tokenize
from ..safety import sanitize_text
from ..visibility import visible_units


_ADDRESS = re.compile(r"^\s*(?:(?P<name>[^<>]+?)\s*)?<(?P<angle>[^<>\s]+@[^<>\s]+)>\s*$|^\s*(?P<plain>[^\s<>]+@[^\s<>]+)\s*$")


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _norm(value: str) -> str:
    return " ".join(tokenize(value))


def _parse_address(value: str) -> tuple[str | None, str | None]:
    match = _ADDRESS.match(value or "")
    if not match:
        return None, None
    email = (match.group("angle") or match.group("plain") or "").casefold()
    name = (match.group("name") or "").strip().strip('"') or None
    return name, email


def _name_from_email(email: str) -> str:
    local = email.split("@", 1)[0]
    return " ".join(part.capitalize() for part in re.split(r"[._+-]+", local) if part)


@dataclass
class Person:
    full_name: str
    aliases: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    slack_user_id: str | None = None
    dm_channel_id: str | None = None
    title: str = ""
    topics: set[str] = field(default_factory=set)

    @property
    def first_name(self) -> str:
        return _norm(self.full_name).split()[0]


@dataclass(frozen=True)
class Channel:
    channel_id: str
    name: str
    topic: str


class ContactDirectory:
    def __init__(self, people: Iterable[Person], channels: Iterable[Channel]) -> None:
        ordered = sorted(people, key=lambda person: (_norm(person.full_name), sorted(person.emails)))
        self.people = tuple(ordered)
        self.channels = tuple(sorted(channels, key=lambda channel: channel.name))
        self.people_by_full_name = {_norm(person.full_name): person for person in self.people}
        self.channels_by_name = {_norm(channel.name): channel for channel in self.channels}

    @classmethod
    def from_data(
        cls,
        data_dir: str | Path,
        units: Iterable[MemoryUnit],
        as_of: datetime,
    ) -> "ContactDirectory":
        root = Path(data_dir)
        people_by_email: dict[str, Person] = {}
        people_by_slack_id: dict[str, Person] = {}

        def add_person(name: str | None, email: str | None = None) -> Person | None:
            if not email and not name:
                return None
            clean_email = email.casefold() if email else None
            person = people_by_email.get(clean_email or "")
            if person is None:
                normalized_name = (name or _name_from_email(clean_email or "unknown@example.com")).strip()
                key = _norm(normalized_name)
                person = next(
                    (candidate for candidate in people_by_email.values() if _norm(candidate.full_name) == key),
                    None,
                )
                if person is None:
                    person = Person(normalized_name)
            if name:
                person.aliases.add(_norm(name))
            person.aliases.add(_norm(person.full_name))
            if clean_email:
                person.emails.add(clean_email)
                people_by_email[clean_email] = person
            return person

        slack_dir = root / "connectors" / "slack"
        users_path = slack_dir / "users.json"
        if users_path.exists():
            for item in _read_json(users_path):
                if item.get("is_bot"):
                    continue
                person = add_person(item.get("real_name") or item.get("name"), item.get("email"))
                if person:
                    person.slack_user_id = item.get("id")
                    person.title = item.get("title") or ""
                    person.topics.update(tokenize(person.title))
                    person.aliases.add(_norm(item.get("name") or ""))
                    people_by_slack_id[item["id"]] = person

        channels: list[Channel] = []
        channels_path = slack_dir / "channels.json"
        if channels_path.exists():
            for item in _read_json(channels_path):
                if item.get("is_dm"):
                    members = [member for member in item.get("members", []) if member in people_by_slack_id]
                    non_self = [member for member in members if people_by_slack_id[member].first_name != "alex"]
                    if len(non_self) == 1:
                        people_by_slack_id[non_self[0]].dm_channel_id = item.get("id")
                else:
                    channels.append(Channel(item["id"], item["name"], item.get("topic") or ""))

        gmail_path = root / "connectors" / "gmail" / "messages.jsonl"
        for item in _read_jsonl(gmail_path) or ():
            try:
                if parse_datetime(item["date"]) > as_of:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            addresses = [item.get("from") or "", *item.get("to", []), *item.get("cc", [])]
            observed: list[Person] = []
            for address in addresses:
                name, email = _parse_address(address)
                person = add_person(name, email)
                if person:
                    observed.append(person)
            evidence = sanitize_text(f"{item.get('subject', '')} {item.get('body', '')}")
            evidence_terms = set(tokenize(evidence))
            for person in observed:
                person.topics.update(evidence_terms)

        calendar_path = root / "connectors" / "google_calendar" / "events.jsonl"
        for item in _read_jsonl(calendar_path) or ():
            try:
                if parse_datetime(item["updated"]) > as_of:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            emails = [item.get("organizer") or "", *(attendee.get("email") or "" for attendee in item.get("attendees", []))]
            event_terms = set(tokenize(f"{item.get('summary', '')} {item.get('description', '')}"))
            for email in emails:
                if not email:
                    continue
                person = add_person(None, email)
                if person:
                    person.topics.update(event_terms)

        for unit in visible_units(tuple(units), as_of):
            evidence_terms = set(tokenize(unit.safe_text))
            actor_key = _norm(unit.actor or "")
            unique_people = {id(person): person for person in people_by_email.values()}
            for person in unique_people.values():
                if actor_key in person.aliases or any(email in unit.safe_text.casefold() for email in person.emails):
                    person.topics.update(evidence_terms)

        unique = {id(person): person for person in people_by_email.values()}
        unique.update({id(person): person for person in people_by_slack_id.values()})
        return cls(unique.values(), channels)

    def resolve_channel(self, query: str) -> Channel | None:
        normalized = _norm(re.sub(r"\b(?:the|channel)\b|#", " ", query, flags=re.I))
        exact = self.channels_by_name.get(normalized)
        if exact:
            return exact
        matches = [channel for channel in self.channels if normalized and normalized in _norm(channel.name)]
        return matches[0] if len(matches) == 1 else None

    def resolve_person(
        self,
        query: str,
        *,
        medium: str | None = None,
        context: str = "",
    ) -> tuple[Person | None, tuple[Person, ...]]:
        normalized = _norm(query)
        exact = [person for person in self.people if normalized in person.aliases]
        if not exact:
            exact = [person for person in self.people if normalized == person.first_name]
        candidates = exact
        if medium == "slack":
            medium_candidates = [person for person in candidates if person.slack_user_id]
            if len(medium_candidates) == 1:
                return medium_candidates[0], tuple(candidates)
            candidates = medium_candidates or candidates
        elif medium == "gmail":
            candidates = [person for person in candidates if person.emails]
        if len(candidates) == 1:
            return candidates[0], tuple(candidates)
        if len(candidates) > 1 and context:
            context_terms = set(tokenize(context)) - {normalized}
            scored = sorted(
                ((len(context_terms & person.topics), person) for person in candidates),
                key=lambda item: (-item[0], _norm(item[1].full_name)),
            )
            if scored[0][0] >= 2 and (len(scored) == 1 or scored[0][0] >= scored[1][0] + 2):
                return scored[0][1], tuple(candidates)
        return None, tuple(candidates)

    def names(self) -> tuple[str, ...]:
        return tuple(person.full_name for person in self.people)

