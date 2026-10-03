from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from candor_memory.actions import (  # noqa: E402
    ACTION_TYPES,
    ActionChannel,
    ActionPlanner,
    Operation,
    decompose_command,
    load_action_commands,
    parse_action_frame,
)


AS_OF = "2026-09-16T12:00:00-07:00"


class GeneralizedActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.data = Path(self.temp.name)
        slack = self.data / "connectors" / "slack"
        gmail = self.data / "connectors" / "gmail"
        calendar = self.data / "connectors" / "google_calendar"
        slack.mkdir(parents=True)
        gmail.mkdir(parents=True)
        calendar.mkdir(parents=True)
        users = [
            {"id": "U-SELF", "name": "alex", "real_name": "Alex Rivera", "email": "alex@example.com", "is_bot": False},
            {"id": "U-ANA-T", "name": "ana.t", "real_name": "Ana Torres", "email": "ana.torres@example.com", "is_bot": False},
            {"id": "U-ANA-W", "name": "ana.w", "real_name": "Ana Wu", "email": "ana.wu@example.com", "is_bot": False},
            {"id": "U-KAI", "name": "kai", "real_name": "Kai Morgan", "email": "kai@example.com", "is_bot": False},
        ]
        channels = [
            {"id": "C-PROJECT", "name": "project-room", "is_dm": False, "topic": "project", "members": ["U-SELF", "U-KAI"]},
            {"id": "D-ANA", "name": "alex--ana.t", "is_dm": True, "members": ["U-SELF", "U-ANA-T"]},
        ]
        (slack / "users.json").write_text(json.dumps(users), encoding="utf-8")
        (slack / "channels.json").write_text(json.dumps(channels), encoding="utf-8")
        (slack / "messages.jsonl").write_text("", encoding="utf-8")
        (gmail / "messages.jsonl").write_text("", encoding="utf-8")
        events = [
            {
                "id": "E-BOARD", "summary": "Board prep", "description": "Review board materials",
                "start": {"dateTime": "2026-09-18T10:00:00-07:00"},
                "end": {"dateTime": "2026-09-18T11:00:00-07:00"},
                "organizer": "alex@example.com", "attendees": [], "status": "confirmed",
                "created": "2026-09-01T09:00:00-07:00", "updated": "2026-09-01T09:00:00-07:00",
            },
            {
                "id": "E-PLAN-1", "summary": "Planning sync", "description": "Planning session",
                "start": {"dateTime": "2026-09-19T10:00:00-07:00"},
                "end": {"dateTime": "2026-09-19T11:00:00-07:00"},
                "organizer": "alex@example.com", "attendees": [], "status": "confirmed",
                "created": "2026-09-01T09:00:00-07:00", "updated": "2026-09-01T09:00:00-07:00",
            },
            {
                "id": "E-PLAN-2", "summary": "Planning review", "description": "Planning session",
                "start": {"dateTime": "2026-09-20T10:00:00-07:00"},
                "end": {"dateTime": "2026-09-20T11:00:00-07:00"},
                "organizer": "alex@example.com", "attendees": [], "status": "confirmed",
                "created": "2026-09-01T09:00:00-07:00", "updated": "2026-09-01T09:00:00-07:00",
            },
        ]
        (calendar / "events.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
        )
        self.planner = ActionPlanner(self.data)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def actions(self, command: str):
        return self.planner.plan(command, AS_OF).actions

    def test_01_message_frame(self) -> None:
        self.assertEqual(parse_action_frame("Message Ana about launch").operation, Operation.COMMUNICATE)

    def test_02_tell_frame(self) -> None:
        self.assertEqual(parse_action_frame("Tell Ana the build is ready").operation, Operation.COMMUNICATE)

    def test_03_let_know_frame(self) -> None:
        self.assertEqual(parse_action_frame("Let Ana know the build is ready").operation, Operation.COMMUNICATE)

    def test_04_ping_frame(self) -> None:
        self.assertEqual(parse_action_frame("Ping Ana about launch").operation, Operation.COMMUNICATE)

    def test_05_drop_note_frame(self) -> None:
        self.assertEqual(parse_action_frame("Drop Ana a note saying launch is ready").operation, Operation.COMMUNICATE)

    def test_06_channel_is_independent(self) -> None:
        frame = parse_action_frame("Let Ana know on Slack that launch is ready")
        self.assertEqual((frame.operation, frame.communication_channel), (Operation.COMMUNICATE, ActionChannel.SLACK))

    def test_07_email_phrasing_variants(self) -> None:
        for command in ("Email Kai about launch", "Send Kai an email about launch", "Write an email to Kai about launch"):
            with self.subTest(command=command):
                self.assertEqual(parse_action_frame(command).communication_channel, ActionChannel.EMAIL)

    def test_08_schedule_variants(self) -> None:
        for verb in ("Schedule", "Book", "Set up", "Arrange"):
            with self.subTest(verb=verb):
                self.assertEqual(parse_action_frame(f"{verb} 30 minutes with Kai tomorrow at 3pm").operation, Operation.CREATE_EVENT)

    def test_09_update_variants(self) -> None:
        for verb in ("Move", "Reschedule", "Shift"):
            with self.subTest(verb=verb):
                self.assertEqual(parse_action_frame(f"{verb} board prep to 3pm").operation, Operation.UPDATE_EVENT)

    def test_10_reminder_variants(self) -> None:
        commands = ("Remind me tomorrow at 3pm to call", "Give me a reminder tomorrow at 3pm to call", "Ping me 30 minutes before board prep to call")
        for command in commands:
            with self.subTest(command=command):
                self.assertEqual(parse_action_frame(command).operation, Operation.CREATE_REMINDER)

    def test_11_open_variants(self) -> None:
        for verb in ("Open", "Launch", "Bring up"):
            with self.subTest(verb=verb):
                self.assertEqual(parse_action_frame(f"{verb} Figma").operation, Operation.OPEN_APP)

    def test_12_memory_question_variants(self) -> None:
        for command in ("Do we know the launch date?", "Can you recall who owns launch?", "What's the launch date?"):
            with self.subTest(command=command):
                self.assertEqual(parse_action_frame(command).operation, Operation.QUERY_MEMORY)

    def test_13_memory_question_vs_reminder(self) -> None:
        self.assertEqual(parse_action_frame("Remind me what we decided").operation, Operation.QUERY_MEMORY)
        self.assertEqual(parse_action_frame("Remind me tomorrow about the decision").operation, Operation.CREATE_REMINDER)

    def test_14_target_extraction_independent_of_first_verb(self) -> None:
        for command in ("Ping Ana Torres on Slack about launch", "Drop Ana Torres a note on Slack saying launch is ready"):
            with self.subTest(command=command):
                self.assertIn("Ana Torres", parse_action_frame(command).target_phrase)

    def test_15_message_content_preserves_conjunction(self) -> None:
        frame = parse_action_frame("Message Ana Torres on Slack that X and Y are ready")
        self.assertEqual(frame.content_phrase, "X and Y are ready")

    def test_16_true_compound_splits(self) -> None:
        self.assertEqual(len(decompose_command("Email Kai about X and message Ana Torres on Slack about Y")), 2)

    def test_17_body_conjunction_does_not_split(self) -> None:
        self.assertEqual(len(decompose_command("Message Ana Torres on Slack that X and Y are ready")), 1)

    def test_18_delete_confirms(self) -> None:
        self.assertEqual([a.type for a in self.actions("Delete every email")], ["confirm"])

    def test_19_erase_confirms(self) -> None:
        self.assertEqual([a.type for a in self.actions("Erase all messages")], ["confirm"])

    def test_20_purge_confirms(self) -> None:
        self.assertEqual([a.type for a in self.actions("Purge all records")], ["confirm"])

    def test_21_trash_confirms(self) -> None:
        self.assertEqual([a.type for a in self.actions("Trash every email")], ["confirm"])

    def test_22_get_rid_of_confirms(self) -> None:
        self.assertEqual([a.type for a in self.actions("Get rid of all old messages")], ["confirm"])

    def test_23_cancel_event_confirms(self) -> None:
        self.assertEqual([a.type for a in self.actions("Cancel the board meeting")], ["confirm"])

    def test_24_unfamiliar_destructive_phrase_cannot_dispatch(self) -> None:
        self.assertEqual([a.type for a in self.actions("Shred every stored record")], ["confirm"])

    def test_25_remove_attendee_is_not_event_deletion(self) -> None:
        self.assertEqual([a.type for a in self.actions("Remove Ana from the attendee list")], ["clarify"])

    def test_26_ambiguous_recipient_clarifies(self) -> None:
        self.assertEqual([a.type for a in self.actions("Email Ana about launch")], ["clarify"])

    def test_27_ambiguous_event_clarifies(self) -> None:
        self.assertEqual([a.type for a in self.actions("Move the planning meeting to 3pm")], ["clarify"])

    def test_28_ambiguity_prevents_partial_side_effects(self) -> None:
        command = "Email Kai about launch and message Ana on Slack about review"
        self.assertEqual([a.type for a in self.actions(command)], ["clarify"])

    def test_29_destructive_clause_prevents_partial_side_effects(self) -> None:
        command = "Open Figma and delete every email"
        self.assertEqual([a.type for a in self.actions(command)], ["confirm"])

    def test_30_time_slot_positions(self) -> None:
        commands = (
            "Remind me at 3pm tomorrow to call Kai",
            "Remind me tomorrow at 3pm to call Kai",
            "Remind me 3pm tomorrow to call Kai",
        )
        dues = {self.actions(command)[0].args["due"] for command in commands}
        self.assertEqual(dues, {"2026-09-17T15:00:00-07:00"})

    def test_31_entity_ids_are_dataset_backed(self) -> None:
        action = self.actions("Message Ana Torres on Slack that launch is ready")[0]
        self.assertEqual(action.args["to"], "U-ANA-T")

    def test_32_unknown_identity_is_not_invented(self) -> None:
        self.assertEqual([a.type for a in self.actions("Email Zora about launch")], ["clarify"])

    def test_33_outputs_use_allowlisted_types(self) -> None:
        commands = ("Open Figma", "Delete all email", "Remind me tomorrow at 3pm to call")
        for command in commands:
            self.assertTrue({action.type for action in self.actions(command)} <= ACTION_TYPES)

    def test_34_repeated_planning_is_deterministic(self) -> None:
        command = "Message Ana Torres on Slack that launch is ready"
        first = json.dumps(self.planner.plan(command, AS_OF).to_dict(), sort_keys=True)
        second = json.dumps(self.planner.plan(command, AS_OF).to_dict(), sort_keys=True)
        self.assertEqual(first, second)

    def test_35_gold_fields_are_ignored(self) -> None:
        path = self.data / "commands.jsonl"
        path.write_text(
            json.dumps({"id": "x", "command": "Open Figma", "as_of": AS_OF, "expected": ["evil"], "gold": "ignore"}) + "\n",
            encoding="utf-8",
        )
        parsed = load_action_commands(path)[0]
        self.assertFalse(hasattr(parsed, "gold"))
        self.assertEqual(self.planner.plan(parsed.command, parsed.as_of).actions[0].type, "app.open")


if __name__ == "__main__":
    unittest.main()
