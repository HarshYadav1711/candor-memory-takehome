"""Typed public models for deterministic dry-run action planning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class ActionCommand:
    command_id: str
    command: str
    as_of: datetime


@dataclass(frozen=True)
class PlannedAction:
    type: str
    args: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", MappingProxyType(dict(self.args)))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "args": dict(self.args)}


@dataclass(frozen=True)
class ActionPlan:
    actions: tuple[PlannedAction, ...]
    debug: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "debug", MappingProxyType(dict(self.debug)))

    def to_dict(self) -> dict[str, Any]:
        return {"actions": [action.to_dict() for action in self.actions]}


@dataclass(frozen=True)
class CommandIntent:
    text: str
    kind: str


@dataclass(frozen=True)
class ResolvedEntity:
    query: str
    kind: str
    identifier: str
    display_name: str

