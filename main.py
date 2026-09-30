"""Safe development CLI for Phase 1 ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from candor_memory import load_units  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Candor data and print a safe summary.")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    units = load_units(args.data)
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


if __name__ == "__main__":
    main()

