"""Exploratory synthetic checks. Run separately from the release test suite."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candor_memory import GroundedMemory, MemoryUnit  # noqa: E402
from candor_memory.actions import ActionPlanner  # noqa: E402


NOW = datetime.fromisoformat("2026-09-18T12:00:00-07:00")


def unit(unit_id: str, text: str, at: datetime = NOW) -> MemoryUnit:
    return MemoryUnit(unit_id, unit_id, "slack", at, text, text, "Morgan Lee", {})


class MemoryGeneralizationBaseline(unittest.TestCase):
    @unittest.expectedFailure
    def test_statement_like_missing_slot_abstains(self) -> None:
        memory = GroundedMemory((unit("U1", "Morgan owns Project Aurora."),))
        answer, _ = memory.answer("Tell me Morgan's salary.", NOW)
        self.assertTrue(answer.abstained)

    def test_future_evidence_remains_invisible(self) -> None:
        memory = GroundedMemory((unit("FUTURE", "Aurora launches Friday.", NOW + timedelta(days=1)),))
        answer, _ = memory.answer("Recall Aurora's launch day.", NOW)
        self.assertTrue(answer.abstained)


class ActionGeneralizationBaseline(unittest.TestCase):
    @unittest.expectedFailure
    def test_common_slack_paraphrase_maps_to_message(self) -> None:
        planner = ActionPlanner(ROOT / "data")
        plan = planner.plan("Ping Ben on Slack about the dashboard", NOW)
        self.assertEqual([action.type for action in plan.actions], ["slack.send_message"])

    @unittest.expectedFailure
    def test_supported_destructive_calendar_request_requires_confirmation(self) -> None:
        planner = ActionPlanner(ROOT / "data")
        plan = planner.plan("Cancel the board meeting", NOW)
        self.assertEqual([action.type for action in plan.actions], ["confirm"])


if __name__ == "__main__":
    unittest.main()
