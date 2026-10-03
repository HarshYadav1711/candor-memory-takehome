"""Deterministic, auditable, side-effect-free TextOS action planner."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ..ingest import parse_datetime
from ..models import MemoryUnit
from ..retrieval.tokenize import tokenize
from ..safety import sanitize_text
from ..visibility import visible_units
from .entities import ContactDirectory, Person
from .frames import ActionFrame, Channel, Operation, parse_action_frame
from .models import ActionPlan, PlannedAction
from .parser import decompose_command
from .safety import SafetyStatus, assess_safety, confirmation_summary, is_safe_grounded_text
from .temporal import iso_local, parse_duration, resolve_datetime
from .validation import validate_action


_CONTENT_STOP = frozenset(
    {
        "about", "ask", "corrected", "current", "email", "latest", "message",
        "our", "please", "send", "slack", "tell", "thank", "that", "the", "to",
    }
)
_MONTH_DATE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?\b",
    re.I,
)


@dataclass(frozen=True)
class _CalendarEvent:
    event_id: str
    summary: str
    description: str
    start: datetime
    end: datetime


class ActionPlanner:
    """Create only structured plans; this class has no integration clients."""

    def __init__(self, data_dir: str | Path, *, units: Iterable[MemoryUnit] = ()) -> None:
        self.data_dir = Path(data_dir)
        self.units = tuple(units)

    def plan(self, command: str, as_of: datetime | str) -> ActionPlan:
        reference = parse_datetime(as_of) if isinstance(as_of, str) else as_of
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        command = command.strip()
        clauses = decompose_command(command)
        frames = tuple(parse_action_frame(clause) for clause in clauses)
        safety = tuple(assess_safety(frame) for frame in frames)
        if any(decision.status == SafetyStatus.CONFIRM for decision in safety):
            action = PlannedAction("confirm", {"summary": confirmation_summary(command)})
            validate_action(action)
            return ActionPlan((action,), {"safety_decision": "confirmation_required", "action_types": [action.type]})
        if any(decision.status == SafetyStatus.CLARIFY for decision in safety):
            action = self._clarify("What exactly should be removed or changed?")
            validate_action(action)
            return ActionPlan((action,), {"safety_decision": "clarification_required", "action_types": [action.type]})

        directory = ContactDirectory.from_data(self.data_dir, self.units, reference)
        actions: list[PlannedAction] = []
        debug_intents: list[str] = []
        for frame in frames:
            action, intent = self._plan_frame(frame, reference, directory)
            debug_intents.append(intent)
            if action.type == "clarify":
                validate_action(action)
                return ActionPlan(
                    (action,),
                    {"atomic_intents": debug_intents, "safety_decision": "none", "action_types": ["clarify"]},
                )
            validate_action(action)
            actions.append(action)
        return ActionPlan(
            tuple(actions),
            {"atomic_intents": debug_intents, "safety_decision": "none", "action_types": [a.type for a in actions]},
        )

    def _plan_frame(
        self,
        frame: ActionFrame,
        as_of: datetime,
        directory: ContactDirectory,
    ) -> tuple[PlannedAction, str]:
        clause = frame.normalized_command
        if frame.operation == Operation.QUERY_MEMORY:
            return PlannedAction("memory.ask", {"question": frame.original_command.strip()}), "memory_question"
        if frame.operation == Operation.OPEN_APP:
            app = frame.application_phrase
            if not app:
                return self._clarify("Which application should I open?"), "app_open"
            return PlannedAction("app.open", {"app": app}), "app_open"
        if frame.operation == Operation.CREATE_REMINDER:
            return self._reminder(clause, as_of), "reminder_create"
        if frame.operation == Operation.UPDATE_EVENT:
            return self._calendar_update(clause, as_of), "calendar_update"
        if frame.operation == Operation.CREATE_EVENT:
            return self._calendar_create(clause, as_of, directory), "calendar_create"
        if frame.operation == Operation.COMMUNICATE:
            return self._communicate(frame, as_of, directory)
        return self._clarify("What action would you like me to plan?"), "unknown"

    @staticmethod
    def _clarify(question: str) -> PlannedAction:
        return PlannedAction("clarify", {"question": question})

    def _person_from_clause(
        self,
        clause: str,
        directory: ContactDirectory,
        *,
        medium: str | None,
    ) -> tuple[Person | None, tuple[Person, ...], str]:
        lowered = clause.casefold()
        full_matches = [
            person for person in directory.people
            if re.search(rf"(?<![\w.-]){re.escape(person.full_name.casefold())}(?![\w.-])", lowered)
        ]
        if len(full_matches) == 1:
            return full_matches[0], tuple(full_matches), full_matches[0].full_name
        mentioned: list[tuple[int, str]] = []
        for person in directory.people:
            first = person.full_name.split()[0]
            if first.casefold() in {"a", "an", "the", "team", "me", "my"}:
                continue
            match = re.search(rf"(?<![\w.-]){re.escape(first)}(?![\w.-])", clause, re.I)
            if match:
                mentioned.append((match.start(), first))
        for query in dict.fromkeys(name for _, name in sorted(mentioned)):
            topical = set(tokenize(clause)) - _CONTENT_STOP - {
                "drop", "inform", "know", "let", "note", "notify", "ping", "reach", "write",
                query.casefold(),
            }
            context = clause if len(topical) >= 2 else ""
            person, candidates = directory.resolve_person(query, medium=medium, context=context)
            if person is not None or candidates:
                return person, candidates, query
        return None, (), ""

    @staticmethod
    def _ambiguity_question(query: str, candidates: tuple[Person, ...]) -> str:
        names = [person.full_name for person in candidates]
        if names:
            return f"Which {query} do you mean: {', '.join(names)}?"
        return f"Who do you mean by {query or 'that person'}?"

    def _communicate(
        self, frame: ActionFrame, as_of: datetime, directory: ContactDirectory
    ) -> tuple[PlannedAction, str]:
        clause = frame.normalized_command
        resolved_channel = next(
            (
                channel for channel in sorted(directory.channels, key=lambda item: -len(item.name))
                if re.search(
                    rf"(?<![\w-])#?{r'[-\s]+'.join(re.escape(part) for part in channel.name.split('-'))}(?:\s+channel)?(?![\w-])",
                    clause,
                    re.I,
                )
            ),
            None,
        )
        medium = "gmail" if frame.communication_channel == Channel.EMAIL else (
            "slack" if frame.communication_channel == Channel.SLACK or resolved_channel else None
        )
        person, candidates, query = self._person_from_clause(clause, directory, medium=medium)

        if frame.communication_channel == Channel.SLACK or resolved_channel:
            if resolved_channel:
                requested = self._communication_content(frame, clause, channel_name=resolved_channel.name)
                text = self._resolve_requested_content(requested, as_of)
                if text is None:
                    return self._clarify(f"What should I tell #{resolved_channel.name}?"), "message_send"
                return PlannedAction("slack.send_message", {"to": resolved_channel.channel_id, "text": text}), "message_send"
            if person is None:
                return self._clarify(self._ambiguity_question(query, candidates)), "message_send"
            if not person.slack_user_id:
                return self._clarify(f"I cannot resolve a Slack target for {person.full_name}."), "message_send"
            requested = self._communication_content(frame, clause, person=person)
            if re.match(r"^thank\b", clause, re.I) and not requested:
                requested = "Thank you"
            text = self._resolve_requested_content(requested, as_of)
            if text is None:
                return self._clarify(f"What should I message {person.full_name}?"), "message_send"
            return PlannedAction("slack.send_message", {"to": person.slack_user_id, "text": text}), "message_send"

        if frame.communication_channel == Channel.EMAIL:
            if person is None:
                return self._clarify(self._ambiguity_question(query, candidates)), "gmail_send"
            if not person.emails:
                return self._clarify(f"What email address should I use for {person.full_name}?"), "gmail_send"
            return self._gmail_for_person(person, frame, as_of), "gmail_send"

        if person is None:
            return self._clarify(self._ambiguity_question(query, candidates)), "communication"
        if not person.slack_user_id:
            return self._clarify(f"Should I contact {person.full_name} by Slack or email?"), "communication"
        requested = self._communication_content(frame, clause, person=person)
        text = self._resolve_requested_content(requested, as_of)
        if text is None:
            return self._clarify(f"What should I message {person.full_name}?"), "message_send"
        return PlannedAction("slack.send_message", {"to": person.slack_user_id, "text": text}), "message_send"

    @staticmethod
    def _communication_content(
        frame: ActionFrame,
        clause: str,
        *,
        person: Person | None = None,
        channel_name: str = "",
    ) -> str:
        if frame.content_phrase:
            return frame.content_phrase
        requested = re.sub(
            r"^\s*(?:message|tell|notify|inform|ping|thank|email|write(?:\s+an?\s+email)?(?:\s+to)?|"
            r"send|drop|let|reach\s+out\s+to)\s+",
            "",
            clause,
            flags=re.I,
        )
        requested = re.sub(r"^(?:a\s+(?:message|note)\s+to|this\s+to|an?\s+email\s+to)\s+", "", requested, flags=re.I)
        if person:
            requested = re.sub(re.escape(person.full_name), "", requested, count=1, flags=re.I)
            requested = re.sub(rf"^\s*{re.escape(person.full_name.split()[0])}\b", "", requested, count=1, flags=re.I)
        if channel_name:
            pattern = r"[-\s]+".join(re.escape(part) for part in channel_name.split("-"))
            requested = re.sub(rf"^(?:the\s+)?#?{pattern}(?:\s+channel)?\b", "", requested, count=1, flags=re.I)
        requested = re.sub(r"\b(?:on\s+slack|by\s+email)\b", " ", requested, flags=re.I)
        requested = re.sub(r"^(?:an?\s+email|a\s+(?:message|note))\s+", "", requested, flags=re.I)
        requested = re.sub(r"^(?:and\s+)?(?:ask|tell|saying|that)\s+", "", requested, flags=re.I)
        return re.sub(r"\s+", " ", requested).strip(" ,.")

    def _gmail_for_person(self, person: Person, frame: ActionFrame, as_of: datetime) -> PlannedAction:
        requested = self._communication_content(frame, frame.normalized_command, person=person)
        if re.search(r"\b(?:corrected|current|latest)\b", requested, re.I):
            body = self._resolve_requested_content(requested, as_of)
            if body is None:
                return self._clarify(f"Which current fact should I email {person.full_name}?")
        else:
            body = requested.strip(" ,.") or "Hello."
            body = body[:1].upper() + body[1:]
        subject_terms = [term for term in tokenize(requested) if term not in _CONTENT_STOP]
        subject = " ".join(subject_terms[:6]).strip().title() or "Follow-up"
        return PlannedAction(
            "gmail.send",
            {"to": [sorted(person.emails)[0]], "cc": [], "subject": subject, "body": body},
        )

    def _calendar_create(
        self, clause: str, as_of: datetime, directory: ContactDirectory
    ) -> PlannedAction:
        duration = parse_duration(clause)
        start = resolve_datetime(clause, as_of)
        if duration is None or start is None:
            return self._clarify("What date, time, and duration should I use for the event?")
        person, candidates, query = self._person_from_clause(clause, directory, medium="gmail")
        if person is None or not person.emails:
            return self._clarify(self._ambiguity_question(query, candidates))
        about = re.search(r"\babout\s+(.+)$", clause, re.I)
        title = about.group(1).strip(" .") if about else f"Meeting with {person.full_name}"
        return PlannedAction(
            "calendar.create_event",
            {
                "title": title,
                "start": iso_local(start),
                "end": iso_local(start + duration),
                "attendees": [sorted(person.emails)[0]],
            },
        )

    def _calendar_update(self, clause: str, as_of: datetime) -> PlannedAction:
        patterns = (
            r"^\s*(?:move|reschedule|shift)\s+(.+?)\s+to\s+(.+)$",
            r"^\s*change(?:\s+the\s+time\s+of)?\s+(.+?)\s+to\s+(.+)$",
            r"^\s*push\s+back\s+(.+?)\s+to\s+(.+)$",
            r"^\s*bring\s+(.+?)\s+forward\s+to\s+(.+)$",
        )
        match = next((candidate for pattern in patterns if (candidate := re.match(pattern, clause, re.I))), None)
        if not match:
            return self._clarify("Which event should I move, and to when?")
        event = self._match_event(match.group(1), as_of)
        if event is None:
            return self._clarify(f"Which calendar event do you mean by {match.group(1).strip()}?")
        start = resolve_datetime(match.group(2), as_of, default_date=event.start.date())
        if start is None:
            return self._clarify("What new date and time should I use?")
        duration = event.end - event.start
        return PlannedAction(
            "calendar.update_event",
            {"event_id": event.event_id, "start": iso_local(start), "end": iso_local(start + duration)},
        )

    def _reminder(self, clause: str, as_of: datetime) -> PlannedAction:
        before = re.search(r"\b(.+?)\s+before\s+(.+)$", clause, re.I)
        if before:
            offset = parse_duration(before.group(1))
            event_phrase = before.group(2).strip()
            reminder_text: str | None = None
            text_split = re.match(r"(.+?)\s+to\s+(.+)$", event_phrase, re.I)
            if text_split:
                event_phrase = text_split.group(1).strip()
                reminder_text = text_split.group(2).strip()
            event = self._match_event(event_phrase, as_of)
            if offset is None or event is None:
                return self._clarify("Which event and reminder offset should I use?")
            text = (reminder_text or f"Prepare for {event.summary}").strip(" .")
            return PlannedAction("reminder.create", {"text": text, "due": iso_local(event.start - offset)})
        due = resolve_datetime(clause, as_of)
        if due is None:
            return self._clarify("When should I set the reminder?")
        text_match = re.search(r"\bto\s+(.+)$", clause, re.I)
        if text_match:
            text = text_match.group(1).strip(" .")
        else:
            after_time = re.search(r"\b(?:am|pm)\s+to\s+(.+)$", clause, re.I)
            prefix = (
                r"^\s*(?:remind\s+me|do\s+not\s+let\s+me\s+forget|don't\s+let\s+me\s+forget|"
                r"give\s+me\s+(?:a\s+)?reminder|set\s+(?:a\s+)?reminder|ping\s+me)\s+"
            )
            text = (after_time.group(1) if after_time else re.sub(prefix, "", clause, flags=re.I)).strip(" .")
        return PlannedAction("reminder.create", {"text": text, "due": iso_local(due)})

    def _events(self, as_of: datetime) -> tuple[_CalendarEvent, ...]:
        path = self.data_dir / "connectors" / "google_calendar" / "events.jsonl"
        if not path.exists():
            return ()
        events: list[_CalendarEvent] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("status") == "cancelled" or parse_datetime(item["updated"]) > as_of:
                    continue
                start_value = item.get("start", {}).get("dateTime")
                end_value = item.get("end", {}).get("dateTime")
                if not start_value or not end_value:
                    continue
                events.append(
                    _CalendarEvent(
                        item["id"], item["summary"], item.get("description") or "",
                        parse_datetime(start_value), parse_datetime(end_value),
                    )
                )
        return tuple(events)

    def _match_event(self, query: str, as_of: datetime) -> _CalendarEvent | None:
        ignored = {"the", "meeting", "event", "calendar"}
        query_terms = set(tokenize(query)) - ignored
        if len(query_terms) == 1 and query_terms <= {"planning", "sync", "check", "review"}:
            return None
        normalized_query = " ".join(
            term for term in tokenize(query) if term not in {"the", "event", "calendar"}
        )
        scored: list[tuple[float, _CalendarEvent]] = []
        for event in self._events(as_of):
            terms = set(tokenize(event.summary + " " + event.description)) - ignored
            overlap = len(query_terms & terms)
            if overlap:
                normalized_summary = " ".join(tokenize(event.summary))
                phrase_bonus = 1.0 if normalized_query and normalized_query in normalized_summary else 0.0
                scored.append((overlap / max(1, len(query_terms)) + phrase_bonus, event))
        scored.sort(key=lambda item: (-item[0], item[1].event_id))
        if not scored or scored[0][0] < 0.5:
            return None
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            return None
        return scored[0][1]

    def _resolve_requested_content(self, requested: str, as_of: datetime) -> str | None:
        requested = requested.strip(" .")
        if not requested:
            return None
        needs_memory = bool(re.search(r"\b(?:current|corrected|latest)\b", requested, re.I))
        needs_memory = needs_memory or bool(
            re.fullmatch(
                r"(?:the|our)\s+.+\b(?:code|date|metric|number|status|time|value)",
                requested,
                re.I,
            )
        )
        if not needs_memory:
            return requested[:1].upper() + requested[1:]
        fact = self._ground_fact(requested, as_of)
        return fact

    def _ground_fact(self, request: str, as_of: datetime) -> str | None:
        focus = {
            term for term in tokenize(request)
            if term not in _CONTENT_STOP and term not in {"we", "re", "is", "date", "fact"}
        }
        candidates: list[tuple[int, datetime, str]] = []
        for unit in visible_units(self.units, as_of):
            safe = sanitize_text(unit.safe_text)
            if not is_safe_grounded_text(safe):
                continue
            terms = set(tokenize(safe))
            overlap = len(focus & terms)
            if overlap == 0:
                continue
            state = int(bool(re.search(r"\b(?:correction|corrected|current|latest|changed|updated|now)\b", safe, re.I)))
            candidates.append((overlap * 10 + state * 3, unit.available_at, safe))
        if not candidates:
            return None
        _, _, text = max(candidates, key=lambda item: (item[0], item[1]))
        if "launch" in focus:
            date_match = _MONTH_DATE.search(text)
            if date_match:
                return f"Our current launch date is {date_match.group(0)}."
        numeric_focus = next((term.upper() for term in focus if term in {"nrr", "arr", "mrr"}), None)
        if numeric_focus:
            pattern = re.compile(rf"\b{re.escape(numeric_focus)}\b.{{0,60}}?\b(?:is|to|at)\s+(\d+(?:\.\d+)?)\s*(%|percent)?", re.I)
            match = pattern.search(text)
            if match:
                suffix = "%" if match.group(2) else ""
                return f"The corrected {numeric_focus} is {match.group(1)}{suffix}."
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        safe_sentences = [sentence for sentence in sentences if len(sentence.split()) <= 40 and is_safe_grounded_text(sentence)]
        if safe_sentences:
            return safe_sentences[0]
        return None

