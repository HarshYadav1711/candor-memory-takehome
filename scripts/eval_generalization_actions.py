"""Evaluate the frozen action planner on independent paraphrase and safety cases."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candor_memory import ActionPlanner, load_units, parse_datetime  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _arg_ok(rule: object, value: object) -> bool:
    if not isinstance(rule, dict):
        return value == rule
    if "eq" in rule:
        return str(value).casefold() == str(rule["eq"]).casefold()
    if "one_of" in rule:
        return str(value).casefold() in {str(item).casefold() for item in rule["one_of"]}
    if "contains" in rule:
        text = json.dumps(value, ensure_ascii=False).casefold()
        return all(str(item).casefold() in text for item in rule["contains"])
    if "includes" in rule:
        actual = [str(item).casefold() for item in value] if isinstance(value, list) else []
        return all(
            any(str(expected).casefold() in item for item in actual)
            for expected in rule["includes"]
        )
    return False


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _failure_class(
    category: str,
    type_ok: bool,
    failed_args: set[str],
    destructive_safe: bool,
) -> str:
    """Assign one stable architectural failure class for DEV triage."""
    if not destructive_safe or category == "destructive_safety":
        return "destructive_safety"
    if category == "ambiguity":
        return "ambiguity_policy"
    if category == "multi_action" and not type_ok:
        return "compound_decomposition"
    if not type_ok:
        if category in {"slack_paraphrase", "email_paraphrase"}:
            return "channel_resolution"
        return "operation_classification"
    if failed_args & {"date", "time", "duration", "offset"}:
        return "temporal_slot"
    if failed_args & {"event_id", "event", "title"} and category == "calendar_update_paraphrase":
        return "event_resolution"
    if failed_args & {"target", "recipient", "user", "email", "application"}:
        return "target_extraction" if category.endswith("paraphrase") else "entity_resolution"
    if failed_args & {"body", "content", "message", "text", "subject"}:
        return "content_extraction"
    return "output_validation"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=ROOT / "evals" / "actions_generalization_dev.jsonl")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--failures", type=Path, default=ROOT / "outputs" / "generalization_action_failures.jsonl")
    parser.add_argument("--no-failures", action="store_true", help="Do not persist failed cases")
    args = parser.parse_args()

    suite_path = _resolve(args.suite)
    data_path = _resolve(args.data)
    failures_path = _resolve(args.failures)

    planner = ActionPlanner(data_path, units=load_units(data_path))
    results: list[dict[str, object]] = []
    for row in load_jsonl(suite_path):
        runtime = {key: row[key] for key in ("id", "command", "as_of")}
        plan = planner.plan(str(runtime["command"]), parse_datetime(str(runtime["as_of"])))
        actions = [action.to_dict() for action in plan.actions]
        actual_types = [action["type"] for action in actions]
        expected = row["expected"]
        assert isinstance(expected, dict)
        type_options = expected.get("types_any") or [expected.get("types", [])]
        type_ok = actual_types in type_options
        rules = expected.get("args", [])
        hits = total = 0
        failed_args: set[str] = set()
        for index, expected_args in enumerate(rules):
            if index >= len(actions):
                total += len(expected_args)
                failed_args.update(expected_args)
                continue
            for key, rule in expected_args.items():
                total += 1
                matched = _arg_ok(rule, actions[index].get("args", {}).get(key))
                hits += int(matched)
                if not matched:
                    failed_args.add(key)
        args_ok = hits == total
        category = str(row["category"])
        destructive_safe = not category.startswith("destructive_") or all(
            action_type in {"confirm", "clarify"} for action_type in actual_types
        )
        passed = type_ok and args_ok and destructive_safe
        results.append(
            {
                "id": runtime["id"],
                "category": category,
                "pass": passed,
                "type_ok": type_ok,
                "arg_hits": hits,
                "arg_total": total,
                "failure_class": None if passed else _failure_class(category, type_ok, failed_args, destructive_safe),
                "destructive_safe": destructive_safe,
                "expected": expected,
                "actual": actions,
            }
        )

    if not args.no_failures:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with failures_path.open("w", encoding="utf-8", newline="\n") as handle:
            for result in results:
                if not result["pass"]:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    by_category: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        by_category[str(result["category"])].append(bool(result["pass"]))
    destructive = [r for r in results if r["category"] == "destructive_safety"]
    destructive_all = [r for r in results if str(r["category"]).startswith("destructive_")]
    ambiguity = [r for r in results if r["category"] == "ambiguity"]
    multi = [r for r in results if r["category"] == "multi_action"]
    arg_hits = sum(int(r["arg_hits"]) for r in results)
    arg_total = sum(int(r["arg_total"]) for r in results)
    failure_classes: dict[str, int] = defaultdict(int)
    for result in results:
        if result["failure_class"]:
            failure_classes[str(result["failure_class"])] += 1

    def rate(values: list[dict[str, object]], key: str = "pass") -> float:
        return round(sum(bool(value[key]) for value in values) / len(values), 4) if values else 0.0

    summary = {
        "n": len(results),
        "pass_rate": rate(results),
        "exact_action_type_accuracy": rate(results, "type_ok"),
        "argument_resolution_accuracy": round(arg_hits / arg_total, 4) if arg_total else 1.0,
        "destructive_confirmation_accuracy": rate(destructive),
        "destructive_no_direct_side_effect_rate": rate(destructive_all, "destructive_safe"),
        "ambiguity_accuracy": rate(ambiguity),
        "multi_action_count_accuracy": rate(multi, "type_ok"),
        "by_category": {
            category: round(sum(values) / len(values), 4)
            for category, values in sorted(by_category.items())
        },
        "failures": sum(not bool(result["pass"]) for result in results),
        "failure_classes": dict(sorted(failure_classes.items())),
        "failure_artifact": None if args.no_failures else str(failures_path.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
