"""Load every assignment source into canonical citable memory units."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

from .models import MemoryUnit
from .safety import sanitize_text


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and require an explicit timezone."""

    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp must include a timezone offset: {value!r}")
    return parsed


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} at line {line_number}") from exc


def _unit(
    *,
    unit_id: str,
    record_id: str,
    source_type: str,
    available_at: datetime,
    text: str,
    actor: str | None = None,
    metadata: dict[str, Any] | None = None,
    event_type: str | None = None,
    target_id: str | None = None,
) -> MemoryUnit:
    return MemoryUnit(
        unit_id=unit_id,
        record_id=record_id,
        source_type=source_type,
        available_at=available_at,
        text=text,
        safe_text=sanitize_text(text),
        actor=actor,
        metadata=metadata or {},
        event_type=event_type,
        target_id=target_id,
    )


def _source_path(path: Path, data_dir: Path) -> str:
    return path.relative_to(data_dir).as_posix()


def load_meetings(data_dir: Path) -> list[MemoryUnit]:
    units: list[MemoryUnit] = []
    for path in sorted((data_dir / "native" / "meetings").glob("*.json")):
        meeting = _read_json(path)
        start = parse_datetime(meeting["start"])
        for segment in meeting["segments"]:
            units.append(
                _unit(
                    unit_id=segment["seg_id"],
                    record_id=meeting["id"],
                    source_type="meeting",
                    available_at=start + timedelta(seconds=float(segment["end_s"])),
                    text=segment["text"],
                    actor=segment.get("speaker_name") or segment.get("speaker_label"),
                    metadata={
                        "source_path": _source_path(path, data_dir),
                        "meeting_title": meeting["title"],
                        "meeting_type": meeting["type"],
                        "meeting_start": meeting["start"],
                        "meeting_end": meeting["end"],
                        "location": meeting.get("location"),
                        "calendar_event_id": meeting.get("calendar_event_id"),
                        "participants_known": meeting.get("participants_known", []),
                        "segment_start_s": segment["start_s"],
                        "segment_end_s": segment["end_s"],
                        "speaker_label": segment.get("speaker_label"),
                        "speaker_name": segment.get("speaker_name"),
                        "speaker_confidence": segment.get("speaker_confidence"),
                        "channel": segment.get("channel"),
                    },
                )
            )
    return units


def load_dictations(data_dir: Path) -> list[MemoryUnit]:
    path = data_dir / "native" / "dictation" / "dictations.jsonl"
    units: list[MemoryUnit] = []
    for item in _read_jsonl(path):
        cleaned = item.get("cleaned_text") or ""
        raw = item.get("raw_transcript") or ""
        text = cleaned if not raw or raw == cleaned else f"{cleaned}\nRaw transcript: {raw}"
        units.append(
            _unit(
                unit_id=item["id"],
                record_id=item["id"],
                source_type="dictation",
                available_at=parse_datetime(item["timestamp"]),
                text=text,
                actor="Alex Rivera",
                metadata={
                    "source_path": _source_path(path, data_dir),
                    "mode": item["mode"],
                    "target_app": item["target_app"],
                    "target_context": item["target_context"],
                    "delivery_state": item["delivery_state"],
                },
            )
        )
    return units


def load_slack(data_dir: Path) -> list[MemoryUnit]:
    base = data_dir / "connectors" / "slack"
    path = base / "messages.jsonl"
    users = {item["id"]: item for item in _read_json(base / "users.json")}
    channels = {item["id"]: item for item in _read_json(base / "channels.json")}
    units: list[MemoryUnit] = []

    for item in _read_jsonl(path):
        subtype = item.get("subtype")
        user = users.get(item.get("user"), {})
        actor = user.get("real_name") or item.get("bot_name") or item.get("user")
        channel = channels.get(item.get("channel_id"), {})
        metadata = {
            "source_path": _source_path(path, data_dir),
            "channel_id": item.get("channel_id"),
            "channel_name": channel.get("name"),
            "is_dm": channel.get("is_dm"),
            "user_id": item.get("user"),
            "thread_parent_id": item.get("thread_parent_id"),
            "reactions": item.get("reactions", []),
            "subtype": subtype,
        }
        if subtype == "message_deleted":
            text = f"Message {item['target_id']} was deleted."
        else:
            text = item.get("text") or ""
        units.append(
            _unit(
                unit_id=item["id"],
                record_id=item["id"],
                source_type="slack",
                available_at=parse_datetime(item["ts"]),
                text=text,
                actor=actor,
                metadata=metadata,
                event_type=subtype if subtype in {"message_changed", "message_deleted"} else None,
                target_id=item.get("target_id"),
            )
        )
    return units


def load_gmail(data_dir: Path) -> list[MemoryUnit]:
    path = data_dir / "connectors" / "gmail" / "messages.jsonl"
    units: list[MemoryUnit] = []
    for item in _read_jsonl(path):
        text = f"Subject: {item['subject']}\n{item['body']}"
        units.append(
            _unit(
                unit_id=item["id"],
                record_id=item["id"],
                source_type="gmail",
                available_at=parse_datetime(item["date"]),
                text=text,
                actor=item.get("from"),
                metadata={
                    "source_path": _source_path(path, data_dir),
                    "thread_id": item["thread_id"],
                    "from": item["from"],
                    "to": item.get("to", []),
                    "cc": item.get("cc", []),
                    "labels": item.get("labels", []),
                    "attachments": item.get("attachments", []),
                },
            )
        )
    return units


def _calendar_text(item: dict[str, Any]) -> str:
    start = item["start"].get("dateTime") or item["start"].get("date")
    end = item["end"].get("dateTime") or item["end"].get("date")
    attendees = ", ".join(a["email"] for a in item.get("attendees", []))
    return "\n".join(
        part
        for part in (
            item["summary"],
            item.get("description") or "",
            item.get("location") or "",
            f"When: {start} to {end}",
            f"Status: {item['status']}",
            f"Attendees: {attendees}" if attendees else "",
        )
        if part
    )


def load_calendar(data_dir: Path) -> list[MemoryUnit]:
    path = data_dir / "connectors" / "google_calendar" / "events.jsonl"
    units: list[MemoryUnit] = []
    for item in _read_jsonl(path):
        units.append(
            _unit(
                unit_id=item["id"],
                record_id=item["id"],
                source_type="google_calendar",
                available_at=parse_datetime(item["updated"]),
                text=_calendar_text(item),
                actor=item.get("organizer"),
                metadata={
                    "source_path": _source_path(path, data_dir),
                    "start": item["start"],
                    "end": item["end"],
                    "organizer": item.get("organizer"),
                    "attendees": item.get("attendees", []),
                    "status": item["status"],
                    "recurrence": item.get("recurrence"),
                    "created": item["created"],
                    "updated": item["updated"],
                },
            )
        )
    return units


def _codex_event_text(event: dict[str, Any]) -> str:
    if event.get("type") == "message":
        return f"{event.get('role', 'message')}: {event.get('content') or ''}"
    if event.get("type") == "tool_call":
        pieces = [f"tool {event.get('tool', 'unknown')}"]
        if event.get("input"):
            pieces.append(f"input: {event['input']}")
        if event.get("output"):
            pieces.append(f"output: {event['output']}")
        return " | ".join(pieces)
    return f"{event.get('type', 'event')}: {event.get('content') or ''}"


def load_codex(data_dir: Path) -> list[MemoryUnit]:
    units: list[MemoryUnit] = []
    directory = data_dir / "connectors" / "codex" / "sessions"
    for path in sorted(directory.glob("*.jsonl")):
        records = list(_read_jsonl(path))
        if not records:
            continue
        metadata, events = records[0], records[1:]
        final_timestamp = events[-1]["timestamp"] if events else metadata["started_at"]
        units.append(
            _unit(
                unit_id=metadata["id"],
                record_id=metadata["id"],
                source_type="codex",
                available_at=parse_datetime(final_timestamp),
                text="\n".join(_codex_event_text(event) for event in events),
                actor="Alex Rivera",
                metadata={
                    "source_path": _source_path(path, data_dir),
                    "started_at": metadata["started_at"],
                    "cwd": metadata.get("cwd"),
                    "repo": metadata.get("repo"),
                    "event_count": len(events),
                },
            )
        )
    return units


def load_chatgpt(data_dir: Path) -> list[MemoryUnit]:
    path = data_dir / "connectors" / "chatgpt" / "conversations.json"
    units: list[MemoryUnit] = []
    for conversation in _read_json(path):
        for message in conversation["messages"]:
            units.append(
                _unit(
                    unit_id=message["id"],
                    record_id=conversation["id"],
                    source_type="chatgpt",
                    available_at=parse_datetime(message["create_time"]),
                    text=message.get("content") or "",
                    actor=message.get("role"),
                    metadata={
                        "source_path": _source_path(path, data_dir),
                        "conversation_title": conversation["title"],
                        "conversation_create_time": conversation["create_time"],
                        "conversation_update_time": conversation["update_time"],
                        "role": message.get("role"),
                    },
                )
            )
    return units


def _all_loaders() -> Iterable:
    return (
        load_meetings,
        load_dictations,
        load_slack,
        load_gmail,
        load_calendar,
        load_codex,
        load_chatgpt,
    )


def load_units(data_dir: str | Path) -> tuple[MemoryUnit, ...]:
    """Load all supported sources under ``data_dir`` in deterministic order."""

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {root}")

    units: list[MemoryUnit] = []
    for loader in _all_loaders():
        units.extend(loader(root))

    seen: set[str] = set()
    duplicates: set[str] = set()
    for unit in units:
        if unit.unit_id in seen:
            duplicates.add(unit.unit_id)
        seen.add(unit.unit_id)
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate citable unit ids: {joined}")

    return tuple(sorted(units, key=lambda unit: (unit.available_at, unit.source_type, unit.unit_id)))

