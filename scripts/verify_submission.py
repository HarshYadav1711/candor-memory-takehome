"""Reproduce and audit the Candor take-home submission with one command."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
MEMORY_OUTPUT = OUTPUTS / "memory_train_answers.jsonl"
ACTION_OUTPUT = OUTPUTS / "actions_train_predictions.jsonl"
ALLOWED_ACTION_TYPES = frozenset(
    {
        "slack.send_message",
        "gmail.send",
        "calendar.create_event",
        "calendar.update_event",
        "reminder.create",
        "memory.ask",
        "app.open",
        "clarify",
        "confirm",
    }
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
_UNSAFE_PATTERNS = (
    re.compile(r"\[REDACTED\]", re.I),
    re.compile(r"\[UNTRUSTED COMMENT REMOVED\]", re.I),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{8,}\b"),
    re.compile(r"\b(?:api[ _-]?key|password|passwd|secret)\s*[:=]\s*\S+", re.I),
)


class VerificationError(RuntimeError):
    """A reproducibility or submission-safety check failed."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerificationError(f"{path.name}: invalid JSON on line {line_number}") from exc
            if not isinstance(value, dict):
                raise VerificationError(f"{path.name}: line {line_number} is not an object")
            rows.append(value)
    return rows


def audit_memory_rows(rows: list[dict[str, Any]], *, expected_count: int) -> None:
    if len(rows) != expected_count:
        raise VerificationError(f"memory output has {len(rows)} rows; expected {expected_count}")
    required = {"id", "answer", "sources", "retrieved", "abstained"}
    ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        if not required <= set(row):
            raise VerificationError(f"memory row {index} is missing required fields")
        if not isinstance(row["sources"], list) or len(row["sources"]) > 5:
            raise VerificationError(f"memory row {index} exceeds the 5-source limit")
        if not isinstance(row["retrieved"], list) or len(row["retrieved"]) > 20:
            raise VerificationError(f"memory row {index} exceeds the 20-result retrieval limit")
        ids.append(str(row["id"]))
    if len(ids) != len(set(ids)):
        raise VerificationError("memory output contains duplicate IDs")


def audit_action_rows(rows: list[dict[str, Any]], *, expected_count: int) -> None:
    if len(rows) != expected_count:
        raise VerificationError(f"action output has {len(rows)} rows; expected {expected_count}")
    ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        if not {"id", "actions"} <= set(row) or not isinstance(row["actions"], list):
            raise VerificationError(f"action row {index} has an invalid schema")
        for action in row["actions"]:
            if not isinstance(action, dict) or action.get("type") not in ALLOWED_ACTION_TYPES:
                raise VerificationError(f"action row {index} contains an unsupported action type")
            if not isinstance(action.get("args"), dict):
                raise VerificationError(f"action row {index} has non-object args")
        ids.append(str(row["id"]))
    if len(ids) != len(set(ids)):
        raise VerificationError("action output contains duplicate IDs")


def audit_deliverable_text(text: str) -> None:
    if _WINDOWS_ABSOLUTE_PATH.search(text):
        raise VerificationError("deliverable contains an absolute Windows path")
    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(text):
            raise VerificationError("deliverable contains an unsafe credential or instruction marker")


def _run(name: str, arguments: Iterable[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode:
        details = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
        raise VerificationError(f"{name} failed\n{details}")
    print(f"PASS  {name}")


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise VerificationError(f"{path.name} is not a JSON object")
    return value


def _assert_metrics(retrieval_path: Path, memory_path: Path, actions_path: Path) -> None:
    retrieval = _json(retrieval_path)["summary"]
    memory = _json(memory_path)["summary"]
    actions = _json(actions_path)["summary"]
    if retrieval["score"]["mean"] != 1.0:
        raise VerificationError("retrieval primary score regressed")
    if retrieval["questions_with_forbidden@10"] != 0 or retrieval["forbidden_retrieved@20"] != 0:
        raise VerificationError("retrieval safety score regressed")
    if memory["strict"]["mean"] != 1.0 or memory["hard_failures"] != 0:
        raise VerificationError("memory answer score regressed")
    if actions["pass_rate"] != 1.0 or actions["arg_accuracy"] != 1.0:
        raise VerificationError("action score regressed")
    print("PASS  expected train metrics")


def _audit_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            audit_deliverable_text(path.read_text(encoding="utf-8"))
    print("PASS  deliverable safety and portability audit")


def main() -> int:
    try:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="candor-verify-") as temporary:
            temp = Path(temporary)
            retrieval_score = temp / "retrieval.json"
            memory_score = temp / "memory.json"
            action_score = temp / "actions.json"
            memory_repeat = temp / "memory-repeat.jsonl"
            action_repeat = temp / "actions-repeat.jsonl"

            _run("unit tests", ["-m", "unittest", "discover", "-s", "tests", "-q"])
            _run(
                "generate memory answers",
                ["main.py", "memory", "--input", "evals/memory_train.jsonl", "--output", "outputs/memory_train_answers.jsonl"],
            )
            _run(
                "retrieval freeze",
                ["scripts/validate_retrieval_freeze.py", "outputs/memory_train_phase4_extract.jsonl", "outputs/memory_train_answers.jsonl"],
            )
            _run(
                "retrieval scorer",
                ["eval_harness/score_retrieval.py", "--gold", "evals/memory_train.jsonl", "--answers", "outputs/memory_train_answers.jsonl", "--data", "data", "--out", str(retrieval_score), "--quiet"],
            )
            _run(
                "memory scorer",
                ["eval_harness/score_memory.py", "--gold", "evals/memory_train.jsonl", "--answers", "outputs/memory_train_answers.jsonl", "--data", "data", "--judge", "none", "--out", str(memory_score), "--quiet"],
            )
            _run(
                "generate action predictions",
                ["main.py", "actions", "--input", "evals/actions_train.jsonl", "--output", "outputs/actions_train_predictions.jsonl"],
            )
            _run(
                "action scorer",
                ["eval_harness/score_actions.py", "--gold", "evals/actions_train.jsonl", "--predictions", "outputs/actions_train_predictions.jsonl", "--out", str(action_score)],
            )

            memory_rows = _load_jsonl(MEMORY_OUTPUT)
            action_rows = _load_jsonl(ACTION_OUTPUT)
            audit_memory_rows(memory_rows, expected_count=len(_load_jsonl(ROOT / "evals" / "memory_train.jsonl")))
            audit_action_rows(action_rows, expected_count=len(_load_jsonl(ROOT / "evals" / "actions_train.jsonl")))
            print("PASS  output counts, schema, and list bounds")
            _assert_metrics(retrieval_score, memory_score, action_score)

            _run(
                "repeat memory generation",
                ["main.py", "memory", "--input", "evals/memory_train.jsonl", "--output", str(memory_repeat)],
            )
            _run(
                "repeat action generation",
                ["main.py", "actions", "--input", "evals/actions_train.jsonl", "--output", str(action_repeat)],
            )
            if MEMORY_OUTPUT.read_bytes() != memory_repeat.read_bytes():
                raise VerificationError("memory output is not byte-deterministic")
            if ACTION_OUTPUT.read_bytes() != action_repeat.read_bytes():
                raise VerificationError("action output is not byte-deterministic")
            print("PASS  byte-identical regeneration")

            _audit_files(
                (
                    MEMORY_OUTPUT,
                    ACTION_OUTPUT,
                    ROOT / "README.md",
                    ROOT / "DEMO.md",
                    ROOT / ".env.example",
                )
            )
        print("\nSubmission verification: PASS")
        return 0
    except (OSError, VerificationError, KeyError, TypeError, ValueError) as exc:
        print(f"\nSubmission verification: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
