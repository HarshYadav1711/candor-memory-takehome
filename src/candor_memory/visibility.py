"""Temporal projection of canonical units without mutating ingestion output."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Iterable

from .ingest import parse_datetime
from .models import MemoryUnit


def _as_datetime(value: datetime | str) -> datetime:
    parsed = parse_datetime(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    return parsed


def visible_units(
    units: Iterable[MemoryUnit], as_of: datetime | str
) -> tuple[MemoryUnit, ...]:
    """Return the immutable state visible at ``as_of``.

    Slack edit and deletion records remain citable event units. A delivered
    deletion removes only its target. The latest delivered edit replaces the
    target's projected text and safe text. Input units are never modified.
    """

    cutoff = _as_datetime(as_of)
    delivered = [unit for unit in units if unit.available_at <= cutoff]

    deleted_targets = {
        unit.target_id
        for unit in delivered
        if unit.event_type == "message_deleted" and unit.target_id
    }
    latest_edits: dict[str, MemoryUnit] = {}
    for unit in delivered:
        if unit.event_type != "message_changed" or not unit.target_id:
            continue
        current = latest_edits.get(unit.target_id)
        if current is None or current.available_at <= unit.available_at:
            latest_edits[unit.target_id] = unit

    visible: list[MemoryUnit] = []
    for unit in delivered:
        if unit.unit_id in deleted_targets:
            continue
        edit = latest_edits.get(unit.unit_id)
        if edit is None:
            visible.append(unit)
            continue
        metadata = dict(unit.metadata)
        metadata["applied_edit_id"] = edit.unit_id
        metadata["applied_edit_at"] = edit.available_at.isoformat()
        visible.append(
            replace(
                unit,
                text=edit.text,
                safe_text=edit.safe_text,
                metadata=metadata,
            )
        )

    return tuple(visible)

