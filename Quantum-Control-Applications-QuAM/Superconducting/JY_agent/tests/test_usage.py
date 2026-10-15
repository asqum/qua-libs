from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from datetime import datetime, timedelta

from jy_agent.approval_web import TAIPEI_TIMEZONE, _usage_range, _usage_section
from jy_agent.db import Database
from jy_agent.usage import (
    attribute_usage,
    call_cost,
    chip_usage,
    chips_summary,
    load_calls,
    parse_time,
    sync_calls,
)

PRICING = {
    "cache_write_5m_multiplier": 1.25,
    "cache_write_1h_multiplier": 2.0,
    "fast_multiplier": 2.0,
    "models": {"claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.2}},
}


def assistant(mid: str, ts: str, tools=(), output=100, read=1000, write_1h=0):
    usage = {
        "input_tokens": 10,
        "output_tokens": output,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write_1h,
        "cache_creation": {"ephemeral_1h_input_tokens": write_1h, "ephemeral_5m_input_tokens": 0},
        "speed": "standard",
    }
    content = [{"type": "tool_use", "name": name, "id": f"t{mid}{i}", "input": {}} for i, name in enumerate(tools)]
    return {"type": "assistant", "timestamp": ts, "message": {"id": mid, "model": "claude-opus-5-5", "usage": usage, "content": content}}


def user(text: str, ts: str):
    return {"type": "user", "timestamp": ts, "message": {"role": "user", "content": text}}


def tool_result(ts: str):
    return {"type": "user", "timestamp": ts, "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]}}


class UsageTests(unittest.TestCase):
    def transcript(self, folder: str) -> Path:
        lines = [
            user("enter", "2026-10-02T11:00:00Z"),
            assistant("m1", "2026-10-02T11:00:05Z", ["mcp__jy_bringup__jy_enter_autonomy_mode"]),
            user("approved", "2026-10-02T11:05:00Z"),
            # Run A starts 11:10; its waiting call and analysis belong to it.
            assistant("m2", "2026-10-02T11:09:59Z"),
            assistant("m3", "2026-10-02T11:10:01Z", ["mcp__jy_bringup__jy_autonomy_start_run"]),
            # The same API call written as two lines (one per content block).
            assistant("m4", "2026-10-02T11:12:00Z", ["Bash"]),
            assistant("m4", "2026-10-02T11:12:00Z", ["mcp__jy_bringup__jy_autonomy_analyze_run"]),
            tool_result("2026-10-02T11:12:01Z"),
            # Code fix inside the same turn: reading, then editing.
            assistant("m5", "2026-10-02T11:13:00Z", ["Read"]),
            assistant("m6", "2026-10-02T11:14:00Z", ["Edit"], write_1h=1000),
            # Run B starts 11:20.
            assistant("m7", "2026-10-02T11:20:01Z", ["mcp__jy_bringup__jy_autonomy_start_run"]),
            user("a question", "2026-10-02T11:30:00Z"),
            assistant("m8", "2026-10-02T11:30:05Z"),
            # Before the workflow existed: excluded.
            assistant("m0", "2026-10-01T09:00:00Z"),
        ]
        root = Path(folder)
        (root / "s.jsonl").write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
        return root

    def test_streaming_line_keeps_final_output(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            lines = [
                assistant("p1", "2026-10-02T11:00:00Z", output=7),
                assistant("p1", "2026-10-02T11:00:03Z", ["Bash"], output=240),
            ]
            (Path(folder) / "agent.jsonl").write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
            (call,) = load_calls(Path(folder))
            self.assertEqual(call.tokens["output"], 240)
            self.assertEqual(call.tools, {"Bash"})

    def runs(self):
        return [
            {"id": "A", "node_id": "02x", "qubits": ["q1"], "started_at": "2026-10-02T11:10:00+00:00"},
            {"id": "B", "node_id": "02c", "qubits": ["q1"], "started_at": "2026-10-02T11:20:00+00:00"},
        ]

    def test_duplicate_lines_count_once(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            calls = load_calls(self.transcript(folder))
            self.assertEqual(sorted(c.message_id for c in calls), ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"])

    def test_calls_are_attributed_per_run(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            calls = load_calls(self.transcript(folder))
            result = attribute_usage(calls, self.runs(), PRICING, parse_time("2026-10-02T10:59:00Z"))
            by_run = {row["run_id"]: row["calls"] for row in result["runs"]}
            # m3 and m4 (analysis) belong to A; m2 waits for m3 but precedes A's start.
            self.assertEqual(by_run, {"A": 2, "B": 1})
            self.assertEqual(result["setup"]["calls"], 2)  # m1 entry, m2
            self.assertEqual(result["development"]["calls"], 2)  # m5 read for m6 edit
            self.assertEqual(result["conversation"]["calls"], 1)  # m8
            self.assertEqual(result["total"]["calls"], 8)  # m0 excluded
            self.assertEqual(result["nodes"]["02x"]["runs"], 1)

    def test_cost_uses_cache_multipliers(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            call = next(c for c in load_calls(self.transcript(folder)) if c.message_id == "m6")
            expected = (10 * 4 + 1000 * 4 * 2.0 + 1000 * 0.2 + 100 * 20) / 1e6
            self.assertAlmostEqual(call_cost(call, PRICING), expected)

    def workflows(self):
        return [
            # The chip's first workflow; closed at 10:30 before any call.
            {"id": "w0", "created_at": "2026-10-02T10:00:00Z", "updated_at": "2026-10-02T10:30:00Z", "status": "stopped", "device_key": "chipA"},
            # Still open, so it runs until the next workflow is created.
            {"id": "w1", "created_at": "2026-10-02T10:59:00Z", "updated_at": "2026-10-02T11:00:00Z", "status": "recovery_required", "device_key": "chipA"},
            # Another chip from 11:25 on: m8 is not chip A's.
            {"id": "w2", "created_at": "2026-10-02T11:25:00Z", "updated_at": "2026-10-02T11:26:00Z", "status": "active", "device_key": "chipB"},
        ]

    def test_chip_usage_follows_workflow_windows(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            calls = load_calls(self.transcript(folder))
            result = chip_usage(calls, self.workflows(), {"w1": self.runs()}, PRICING, "chipA")
            self.assertEqual(result["chip_started_at"], "2026-10-02T10:00:00+00:00")
            self.assertEqual(result["workflows"], 2)
            self.assertEqual([row["run_id"] for row in result["runs"]], ["A", "B"])
            self.assertEqual(result["total"]["calls"], 7)  # m1-m7; m8 is chip B, m0 before
            self.assertEqual(result["conversation"]["calls"], 0)
            self.assertEqual(result["days"]["2026-10-02"]["calls"], 7)
            self.assertEqual(result["all_usage"]["total"]["calls"], 9)
            self.assertEqual(result["all_usage"]["conversation"]["calls"], 2)  # m0, m8

    def test_chip_usage_range_and_session(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            calls = load_calls(self.transcript(folder))
            ranged = chip_usage(
                calls, self.workflows(), {"w1": self.runs()}, PRICING, "chipA",
                parse_time("2026-10-02T11:15:00Z"), parse_time("2026-10-02T12:00:00Z"),
            )
            # Run A started before the range and had no call inside it.
            self.assertEqual([row["run_id"] for row in ranged["runs"]], ["B"])
            self.assertEqual(ranged["total"]["calls"], 1)
            self.assertEqual(ranged["all_usage"]["total"]["calls"], 2)
            session = chip_usage(
                calls, self.workflows(), {"w1": self.runs()}, PRICING, "chipA",
                only_workflow="w0",
            )
            self.assertEqual(session["total"]["calls"], 0)
            self.assertEqual(session["runs"], [])

    def test_saved_calls_outlive_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            db = Database(Path(folder) / "jy.sqlite3")
            transcripts = Path(folder) / "projects"
            transcripts.mkdir()
            self.transcript(str(transcripts))
            first = sync_calls(db, transcripts)
            self.assertEqual(len(first), 9)
            # A later, larger line of m8 grows the saved row.
            more = [assistant("m8", "2026-10-02T11:30:05Z", ["Bash"], output=900)]
            (transcripts / "t.jsonl").write_text("\n".join(json.dumps(x) for x in more), encoding="utf-8")
            sync_calls(db, transcripts)
            # Claude Code deletes the transcripts; the saved calls remain.
            for path in transcripts.glob("*.jsonl"):
                path.unlink()
            kept = {call.message_id: call for call in sync_calls(db, transcripts)}
            self.assertEqual(len(kept), 9)
            self.assertEqual(kept["m8"].tokens["output"], 900)
            self.assertEqual(kept["m8"].tools, {"Bash"})
            self.assertEqual(kept["m6"].tokens["cache_write_1h"], 1000)
            self.assertEqual(kept["m4"].tools, {"Bash", "mcp__jy_bringup__jy_autonomy_analyze_run"})
            # Labels survive: the turn is saved with each call.
            result = chip_usage(list(kept.values()), self.workflows(), {"w1": self.runs()}, PRICING, "chipA")
            self.assertEqual(result["development"]["calls"], 2)
            self.assertEqual(len(sync_calls(db, None)), 9)

    def test_chips_summary(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            calls = load_calls(self.transcript(folder))
            rows = chips_summary(calls, self.workflows(), {"chipA": 2}, PRICING)
            self.assertEqual([row["chip_name"] for row in rows], ["chipB", "chipA"])
            chip_b, chip_a = rows
            self.assertEqual((chip_a["workflows"], chip_a["runs"], chip_a["calls"]), (2, 2, 7))
            self.assertEqual(chip_a["first_at"], "2026-10-02T10:00:00+00:00")
            self.assertEqual(chip_a["last_at"], "2026-10-02T11:20:01+00:00")
            self.assertEqual((chip_b["calls"], chip_b["runs"]), (1, 0))  # m8

    def test_range_presets(self) -> None:
        now = datetime(2026, 10, 14, 15, 30, tzinfo=TAIPEI_TIMEZONE)
        self.assertEqual(_usage_range({}, now)[:3], ("chip", None, None))
        preset, start, end, _ = _usage_range({"range": "month"}, now)
        self.assertEqual(start, datetime(2026, 10, 1, tzinfo=TAIPEI_TIMEZONE))
        preset, start, end, error = _usage_range({"from": "2026-10-13T08:00", "to": "2026-10-13T09:00"}, now)
        self.assertEqual(preset, "custom")
        self.assertEqual(end - start, timedelta(minutes=61))
        self.assertEqual(_usage_range({"from": "2026-10-13T09:00", "to": "2026-10-13T08:00"}, now)[3], "order")
        self.assertEqual(_usage_range({"from": "yesterday"}, now)[3], "time")

    def test_page_renders(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = self.transcript(folder)
            test = self

            class Service:
                def chip_usage(self, session_id, start=None, end=None, session_only=False, day_zone=None):
                    calls = load_calls(root)
                    args = (calls, test.workflows(), {"w1": test.runs()}, PRICING, "chipA")
                    return {
                        "chip_name": "chipA",
                        "device_key": "chipA",
                        "chips": chips_summary(calls, test.workflows(), {"chipA": 2}, PRICING),
                        "saved_since": calls[0].timestamp.isoformat(),
                        "overall": chip_usage(*args, day_zone=day_zone),
                        "range": chip_usage(*args, start, end, day_zone=day_zone),
                        "pricing": PRICING,
                        "transcript_dir": str(root),
                    }

            html = _usage_section(Service(), "s", "zh-Hant", {"range": "chip"})
            self.assertIn("Opus 5.5", html)
            self.assertIn("02c", html)
            self.assertIn("估計總費用", html)
            self.assertIn("晶片 chipA", html)
            self.assertIn("2026-10-02 · 2 個實驗", html)
            self.assertIn("此時段全部用量", html)
            self.assertIn("19:10:00", html)  # run A in Taipei time
            self.assertIn("各晶片總用量", html)
            self.assertIn("全部晶片", html)
            self.assertIn('<tr class="usage-current"><td class="text">chipA</td>', html)
            self.assertLess(html.index("各晶片總用量"), html.index("估計總費用"))
            self.assertIn("2026-10-01 開始", html)  # m0, saved since, in Taipei time


if __name__ == "__main__":
    unittest.main()
