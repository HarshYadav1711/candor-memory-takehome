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
    GroundedMemory,
    QuestionFormatError,
    load_questions,
    load_units,
)


def _write_memory_answers(
    input_path: Path,
    output_path: Path,
    data_dir: Path,
    answerer: str = "extractive",
    debug_output: Path | None = None,
) -> int:
    questions = load_questions(input_path)
    memory = GroundedMemory(load_units(data_dir))
    rows = []
    debug_rows = []
    for question in questions:
        answer, debug = memory.answer(
            question.question,
            question.as_of,
            mode=answerer,
        )
        rows.append(
            {
                "id": question.question_id,
                "answer": answer.answer,
                "sources": list(answer.sources),
                "retrieved": list(answer.retrieved),
                "abstained": answer.abstained,
            }
        )
        if debug_output is not None:
            debug_rows.append({"id": question.question_id, **debug})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if debug_output is not None:
        debug_output.parent.mkdir(parents=True, exist_ok=True)
        with debug_output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in debug_rows:
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
    memory_parser.add_argument(
        "--answerer",
        choices=("extractive", "openai", "auto"),
        default="extractive",
        help="grounded answer writer (default: deterministic offline extractive)",
    )
    memory_parser.add_argument(
        "--debug-output",
        type=Path,
        help="optional safe JSONL scoring trace",
    )
    args = parser.parse_args()
    try:
        if args.command == "summary":
            _print_summary(args.data)
        else:
            _write_memory_answers(
                args.input,
                args.output,
                args.data,
                args.answerer,
                args.debug_output,
            )
    except (FileNotFoundError, QuestionFormatError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

