"""Create deterministic, category-balanced action generalization splits."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evals" / "actions_generalization.jsonl"
DEV = ROOT / "evals" / "actions_generalization_dev.jsonl"
HOLDOUT = ROOT / "evals" / "actions_generalization_holdout.jsonl"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)

    holdout_ids: set[str] = set()
    for category_rows in by_category.values():
        count = max(1, round(len(category_rows) * 0.30))
        ranked = sorted(
            category_rows,
            key=lambda row: hashlib.sha256(str(row["id"]).encode("utf-8")).hexdigest(),
        )
        holdout_ids.update(str(row["id"]) for row in ranked[:count])
    dev = [row for row in rows if str(row["id"]) not in holdout_ids]
    holdout = [row for row in rows if str(row["id"]) in holdout_ids]
    return dev, holdout


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    dev, holdout = split(load_jsonl(SOURCE))
    write_jsonl(DEV, dev)
    write_jsonl(HOLDOUT, holdout)
    print(json.dumps({"dev": len(dev), "holdout": len(holdout)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
