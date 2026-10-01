from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from candor_memory.textos import TextOS, format_textos_response  # noqa: E402


AS_OF = "2026-09-18T09:00:00-07:00"


class TextOSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shell = TextOS(ROOT / "data")

    def test_memory_question_routes_to_grounded_answering(self) -> None:
        result = self.shell.handle("What's our launch date?", AS_OF)
        self.assertEqual(result["kind"], "memory")
        self.assertTrue(result["sources"])

    def test_memory_ask_plan_routes_to_grounded_answering(self) -> None:
        result = self.shell.handle("When is the board meeting", AS_OF)
        self.assertEqual(result["kind"], "memory")
        self.assertNotIn("actions", result)

    def test_slack_command_displays_dry_run(self) -> None:
        result = self.shell.handle("Message Sarah on Slack that the fix looks good", AS_OF)
        rendered = format_textos_response(result)
        self.assertIn("slack.send_message", rendered)
        self.assertIn("Dry run", rendered)

    def test_gmail_command_displays_dry_run(self) -> None:
        result = self.shell.handle("Email Sarah Patel about the proposal", AS_OF)
        rendered = format_textos_response(result)
        self.assertIn("gmail.send", rendered)
        self.assertIn("Dry run", rendered)

    def test_calendar_command_remains_dry_run(self) -> None:
        result = self.shell.handle("Move board deck prep to 3pm", AS_OF)
        self.assertEqual(result["actions"][0]["type"], "calendar.update_event")
        self.assertTrue(result["dry_run"])

    def test_reminder_command_remains_dry_run(self) -> None:
        result = self.shell.handle("Remind me an hour before the board meeting to print the deck", AS_OF)
        self.assertEqual(result["actions"][0]["type"], "reminder.create")
        self.assertTrue(result["dry_run"])

    def test_destructive_command_displays_confirmation_only(self) -> None:
        result = self.shell.handle("Delete all my emails from Marcus", AS_OF)
        self.assertEqual([action["type"] for action in result["actions"]], ["confirm"])

    def test_ambiguous_command_displays_clarification_only(self) -> None:
        result = self.shell.handle("Message Sarah about the proposal", AS_OF)
        self.assertEqual([action["type"] for action in result["actions"]], ["clarify"])

    def test_compound_command_displays_each_action(self) -> None:
        result = self.shell.handle(
            "Email John the corrected NRR and thank Ben on Slack",
            "2026-09-16T12:00:00-07:00",
        )
        rendered = format_textos_response(result)
        self.assertEqual(rendered.count("Dry-run action"), 2)
        self.assertIn("gmail.send", rendered)
        self.assertIn("slack.send_message", rendered)

    def test_textos_memory_path_never_calls_external_http(self) -> None:
        with patch("urllib.request.urlopen", side_effect=AssertionError("external call")):
            result = self.shell.handle("What's our launch date?", AS_OF)
        self.assertEqual(result["kind"], "memory")

    def test_once_works_non_interactively(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "textos", "--once", "Open Figma", "--as-of", AS_OF],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("app.open", completed.stdout)
        self.assertIn("nothing was executed", completed.stdout)

    def test_json_output_is_valid_json_only(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "textos", "--once", "Open Figma", "--as-of", AS_OF, "--json"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        parsed = json.loads(completed.stdout)
        self.assertEqual(parsed["actions"][0]["type"], "app.open")

    def test_explicit_as_of_is_respected(self) -> None:
        earlier = self.shell.handle("When is Route Planner v2 launching?", "2026-09-15T12:00:00-07:00")
        later = self.shell.handle("When is Route Planner v2 launching?", AS_OF)
        self.assertIn("14", earlier["answer"])
        self.assertIn("21", later["answer"])

    def test_unsafe_secret_marker_never_appears_in_display(self) -> None:
        result = self.shell.handle("What API credential was pasted?", AS_OF)
        rendered = format_textos_response(result)
        self.assertNotIn("[REDACTED]", rendered)
        self.assertNotIn("api key:", rendered.casefold())

    def test_planted_source_instruction_cannot_control_shell(self) -> None:
        result = self.shell.handle("Has Acme signed the agreement?", AS_OF)
        self.assertEqual(result["kind"], "memory")
        self.assertNotIn("actions", result)

    def test_repeated_once_calls_are_deterministic(self) -> None:
        first = json.dumps(self.shell.handle("Open Figma", AS_OF), sort_keys=True)
        second = json.dumps(self.shell.handle("Open Figma", AS_OF), sort_keys=True)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
