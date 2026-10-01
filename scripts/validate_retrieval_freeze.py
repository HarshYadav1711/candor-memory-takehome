"""Compare ranked retrieved lists in two memory-answer JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare_retrieval(reference: Path, candidate: Path) -> list[str]:
    expected = _load(reference)
    actual = _load(candidate)
    problems: list[str] = []
    if [row.get("id") for row in expected] != [row.get("id") for row in actual]:
        problems.append("row IDs/order differ")
    actual_by_id = {row.get("id"): row for row in actual}
    for row in expected:
        question_id = row.get("id")
        other = actual_by_id.get(question_id, {})
        if row.get("retrieved") != other.get("retrieved"):
            problems.append(f"{question_id}: retrieved list changed")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    problems = compare_retrieval(args.reference, args.candidate)
    if problems:
        raise SystemExit("\n".join(problems))
    print("retrieval freeze: OK")


if __name__ == "__main__":
    main()

