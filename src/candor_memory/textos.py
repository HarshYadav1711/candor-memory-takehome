"""Small terminal interface over frozen memory and dry-run action systems."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, TextIO
from zoneinfo import ZoneInfo

from .actions import ActionPlan, ActionPlanner
from .answering import GroundedMemory
from .ingest import load_units, parse_datetime
from .models import MemoryUnit


LOCAL_TIMEZONE = ZoneInfo("America/Los_Angeles")
DRY_RUN_NOTICE = "Dry run — nothing was executed."
_HELP = "Commands: /help, /asof <ISO timestamp>, /quit, /exit"


class TextOS:
    """Route one line to grounded memory or the dry-run action planner."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        units: Iterable[MemoryUnit] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.units = tuple(load_units(self.data_dir) if units is None else units)
        self.memory = GroundedMemory(self.units)
        self.planner = ActionPlanner(self.data_dir, units=self.units)

    def handle(self, text: str, as_of: datetime | str | None = None) -> dict[str, Any]:
        query = text.strip()
        if not query:
            raise ValueError("TextOS input must be non-empty")
        reference = _reference_time(as_of)
        if query.endswith("?"):
            return self._answer_memory(query, reference)
        plan = self.planner.plan(query, reference)
        return self._resolve_plan(plan, reference)

    def _resolve_plan(self, plan: ActionPlan, as_of: datetime) -> dict[str, Any]:
        if len(plan.actions) == 1 and plan.actions[0].type == "memory.ask":
            question = str(plan.actions[0].args["question"])
            return self._answer_memory(question, as_of)
        return {
            "kind": "actions",
            "actions": [action.to_dict() for action in plan.actions],
            "dry_run": True,
        }

    def _answer_memory(self, question: str, as_of: datetime) -> dict[str, Any]:
        answer, _ = self.memory.answer(question, as_of, mode="extractive")
        return {
            "kind": "memory",
            "answer": answer.answer,
            "sources": list(answer.sources),
            "abstained": answer.abstained,
        }


def _reference_time(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(LOCAL_TIMEZONE).replace(microsecond=0)
    return parse_datetime(value) if isinstance(value, str) else value


def format_textos_response(result: dict[str, Any]) -> str:
    if result.get("kind") == "memory":
        lines = ["Memory", str(result.get("answer", ""))]
        sources = result.get("sources") or []
        if sources:
            lines.extend(("", "Sources:"))
            lines.extend(f"- {source}" for source in sources)
        return "\n".join(lines)

    lines: list[str] = []
    for index, action in enumerate(result.get("actions", [])):
        if index:
            lines.append("")
        lines.extend(("Dry-run action", str(action["type"])))
        for key, value in action.get("args", {}).items():
            rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
            lines.append(f"  {key}: {rendered}")
    lines.extend(("", DRY_RUN_NOTICE))
    return "\n".join(lines)


def run_textos(
    shell: TextOS,
    *,
    once: str | None = None,
    as_of: datetime | str | None = None,
    json_output: bool = False,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> int:
    reference = _reference_time(as_of)
    if once is not None:
        _print_result(shell.handle(once, reference), json_output, output_stream)
        return 0

    if json_output:
        for line in input_stream:
            value = line.strip()
            if value:
                _print_result(shell.handle(value, reference), True, output_stream)
        return 0

    print("Candor TextOS", file=output_stream)
    print("Memory + dry-run actions", file=output_stream)
    print("Type /help for commands, /quit to exit.", file=output_stream)
    while True:
        try:
            if input_stream is sys.stdin:
                value = input("textos> ").strip()
            else:
                value = input_stream.readline()
                if value == "":
                    break
                value = value.strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not value:
            continue
        lowered = value.casefold()
        if lowered in {"/quit", "/exit"}:
            break
        if lowered == "/help":
            print(_HELP, file=output_stream)
            continue
        if lowered.startswith("/asof"):
            timestamp = re.sub(r"^/asof\s*", "", value, flags=re.I).strip()
            if not timestamp:
                print("Usage: /asof <ISO timestamp>", file=output_stream)
                continue
            try:
                reference = parse_datetime(timestamp)
            except (TypeError, ValueError) as exc:
                print(f"Invalid as_of: {exc}", file=output_stream)
                continue
            print(f"as_of set to {reference.isoformat()}", file=output_stream)
            continue
        try:
            _print_result(shell.handle(value, reference), False, output_stream)
        except ValueError as exc:
            print(f"Error: {exc}", file=output_stream)
    return 0


def _print_result(result: dict[str, Any], json_output: bool, output: TextIO) -> None:
    if json_output:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), file=output)
    else:
        print(format_textos_response(result), file=output)

