"""Evaluate the frozen memory runtime on the independent development holdout."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candor_memory import GroundedMemory, load_units, parse_datetime  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _matches(requirement: object, text: str) -> bool:
    choices = requirement if isinstance(requirement, list) else [requirement]
    return any(str(choice).casefold() in text for choice in choices)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=ROOT / "evals" / "memory_generalization.jsonl")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--failures", type=Path, default=ROOT / "outputs" / "generalization_memory_failures.jsonl")
    args = parser.parse_args()

    rows = load_jsonl(args.suite)
    memory = GroundedMemory(load_units(args.data))
    results: list[dict[str, object]] = []
    for row in rows:
        runtime = {key: row[key] for key in ("id", "question", "as_of")}
        answer, _ = memory.answer(
            str(runtime["question"]), parse_datetime(str(runtime["as_of"])), mode="extractive"
        )
        expected = row["expected"]
        assert isinstance(expected, dict)
        text = answer.answer.casefold()
        should_abstain = bool(expected.get("abstained", False))
        required = expected.get("answer_contains", [])
        excluded = expected.get("answer_excludes", [])
        missing = [term for term in required if not _matches(term, text)]
        forbidden = [term for term in excluded if _matches(term, text)]
        passed = answer.abstained == should_abstain and not missing and not forbidden
        results.append(
            {
                "id": runtime["id"],
                "category": row["category"],
                "expected_abstained": should_abstain,
                "pass": passed,
                "missing": missing,
                "forbidden": forbidden,
                "actual": {
                    "answer": answer.answer,
                    "sources": list(answer.sources),
                    "retrieved": list(answer.retrieved),
                    "abstained": answer.abstained,
                },
            }
        )

    args.failures.parent.mkdir(parents=True, exist_ok=True)
    with args.failures.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            if not result["pass"]:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    by_category: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        by_category[str(result["category"])].append(bool(result["pass"]))
    abstention = [r for r in results if r["expected_abstained"]]
    supported = [r for r in results if not r["expected_abstained"]]
    exclusion_cases = [
        (row, result)
        for row, result in zip(rows, results)
        if isinstance(row.get("expected"), dict) and row["expected"].get("answer_excludes")
    ]
    forbidden_checks = [result for _, result in exclusion_cases if result["forbidden"]]

    def rate(values: list[dict[str, object]]) -> float:
        return round(sum(bool(value["pass"]) for value in values) / len(values), 4) if values else 0.0

    summary = {
        "n": len(results),
        "pass_rate": rate(results),
        "supported_answer_accuracy": rate(supported),
        "abstention_accuracy": rate(abstention),
        "forbidden_term_accuracy": round(
            1.0 - len(forbidden_checks) / len(exclusion_cases), 4
        ) if exclusion_cases else 1.0,
        "forbidden_term_violations": len(forbidden_checks),
        "by_category": {
            category: round(sum(values) / len(values), 4)
            for category, values in sorted(by_category.items())
        },
        "failures": sum(not bool(result["pass"]) for result in results),
        "failure_artifact": str(args.failures.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
