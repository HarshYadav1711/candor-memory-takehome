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
        actual = {str(item).casefold() for item in value} if isinstance(value, list) else set()
        return {str(item).casefold() for item in rule["includes"]} <= actual
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=ROOT / "evals" / "actions_generalization.jsonl")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--failures", type=Path, default=ROOT / "outputs" / "generalization_action_failures.jsonl")
    args = parser.parse_args()

    planner = ActionPlanner(args.data, units=load_units(args.data))
    results: list[dict[str, object]] = []
    for row in load_jsonl(args.suite):
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
        for index, expected_args in enumerate(rules):
            if index >= len(actions):
                total += len(expected_args)
                continue
            for key, rule in expected_args.items():
                total += 1
                hits += int(_arg_ok(rule, actions[index].get("args", {}).get(key)))
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
                "destructive_safe": destructive_safe,
                "expected": expected,
                "actual": actions,
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
    destructive = [r for r in results if r["category"] == "destructive_safety"]
    destructive_all = [r for r in results if str(r["category"]).startswith("destructive_")]
    ambiguity = [r for r in results if r["category"] == "ambiguity"]
    multi = [r for r in results if r["category"] == "multi_action"]
    arg_hits = sum(int(r["arg_hits"]) for r in results)
    arg_total = sum(int(r["arg_total"]) for r in results)

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
        "failure_artifact": str(args.failures.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
