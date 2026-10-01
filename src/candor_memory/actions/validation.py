"""Schema allowlist for dry-run action outputs."""

from __future__ import annotations

from .models import PlannedAction


ACTION_SCHEMAS = {
    "slack.send_message": frozenset({"to", "text"}),
    "gmail.send": frozenset({"to", "cc", "subject", "body"}),
    "calendar.create_event": frozenset({"title", "start", "end", "attendees"}),
    "calendar.update_event": frozenset({"event_id"}),
    "reminder.create": frozenset({"text", "due"}),
    "memory.ask": frozenset({"question"}),
    "app.open": frozenset({"app"}),
    "clarify": frozenset({"question"}),
    "confirm": frozenset({"summary"}),
}
ACTION_TYPES = frozenset(ACTION_SCHEMAS)


def validate_action(action: PlannedAction) -> None:
    required = ACTION_SCHEMAS.get(action.type)
    if required is None:
        raise ValueError(f"unsupported action type: {action.type}")
    fields = set(action.args)
    if action.type == "calendar.update_event":
        if "event_id" not in fields or len(fields) < 2:
            raise ValueError("calendar.update_event requires event_id and a changed field")
    elif fields != set(required):
        raise ValueError(f"{action.type} requires exactly {sorted(required)}")

