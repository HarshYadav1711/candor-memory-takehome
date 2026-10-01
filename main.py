"""Safe development CLI for Phase 1 ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from candor_memory import (  # noqa: E402
    HierarchicalRetriever,
    QuestionFormatError,
    load_questions,
    load_units,
)


def _write_memory_answers(input_path: Path, output_path: Path, data_dir: Path) -> int:
    questions = load_questions(input_path)
    retriever = HierarchicalRetriever(load_units(data_dir))
    rows = []
    for question in questions:
        rows.append(
            {
                "id": question.question_id,
                "answer": "I don't know",
                "sources": [],
                "retrieved": list(
                    retriever.retrieve(question.question, question.as_of)
                ),
                "abstained": True,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _print_summary(data_dir: Path) -> None:
    units = load_units(data_dir)
    counts = Counter(unit.source_type for unit in units)
    changed_by_sanitation = sum(unit.text != unit.safe_text for unit in units)
    summary = {
        "total_units": len(units),
        "units_by_source": dict(sorted(counts.items())),
        "earliest_available_at": units[0].available_at.isoformat() if units else None,
        "latest_available_at": units[-1].available_at.isoformat() if units else None,
        "units_changed_by_sanitation": changed_by_sanitation,
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Candor memory development CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary", help="print a safe ingestion summary")
    summary_parser.add_argument("--data", type=Path, default=ROOT / "data")

    memory_parser = subparsers.add_parser("memory", help="retrieve memory for JSONL questions")
    memory_parser.add_argument("--input", type=Path, required=True)
    memory_parser.add_argument("--output", type=Path, required=True)
    memory_parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    try:
        if args.command == "summary":
            _print_summary(args.data)
        else:
            _write_memory_answers(args.input, args.output, args.data)
    except (FileNotFoundError, QuestionFormatError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

