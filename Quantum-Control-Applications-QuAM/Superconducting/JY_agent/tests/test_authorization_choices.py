"""Approval-page choices, Home re-authorization and full-auto judgement.

Operator decisions 2026-10-20:

* Every lease approval chooses 8/16/24 h (counted from the approval) and
  whether "full-auto judgement" is on.
* Home "停止核准" lets the current run finish, ends the lease and puts a
  renewal card on the Approval page; the waiting agent follows the new lease.
* A pause from Home continues through a simple card (hours, judgement); the
  re-plan form is folded away.
* After a Recover (not an operator end) entry continues with a simple card;
  an operator end brings back the full lease page.
* Full-auto judgement retries an unknown crash, marks a target incomplete
  after repeated failures, lifts an instrument pause once the QOP answers,
  and closes the chip's stale quantum machine first.
* The watchdog logs a failed run's event once, not on every sweep.
* Shadow mode records what the deterministic planner would have done.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path

from jy_agent.approval_web import (
    _authorization_choices_from_fields,
    _continuation_card,
    _replan_card,
    handle_session_dashboard,
)
from jy_agent.qm_cleanup import classify, close_stale_qms, config_ports, state_ports
from jy_agent.service import AgentService, AutonomyScopeError, ServiceError
from jy_agent.util import atomic_write_json, json_dumps, json_loads

from test_autonomy import activate_lease
from test_autonomy_continuity import UNKNOWN_TRACEBACK, insert_failed_run
from test_core import make_settings, sample_state, start_test_workflow
from test_dashboard_replan import pause, replan_proposal
from test_live_dashboard_telemetry import _request
from test_per_qubit_pass_rules import build


def control(service: AgentService, lease: dict, action: str) -> dict:
    return service.control_autonomy_from_browser(
        lease["id"], action, service.autonomy_csrf_token(lease["id"]), "operator"
    )


def hours_left(service: AgentService, lease_id: str) -> float:
    expires = datetime.fromisoformat(service._autonomy_row(lease_id)["expires_at"])
    return (expires - datetime.now(timezone.utc)).total_seconds() / 3600


def wait_for_cleanup(service: AgentService) -> None:
    deadline = time.monotonic() + 10
    while service._qm_cleanup_active.is_set() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not service._qm_cleanup_active.is_set()


class AuthorizationChoiceTests(unittest.TestCase):
    def test_choices_are_parsed_and_validated(self) -> None:
        self.assertIsNone(_authorization_choices_from_fields({}))
        self.assertEqual(
            _authorization_choices_from_fields(
                {"duration_hours": ["16"], "auto_judgement": ["yes"]}
            ),
            {"duration_hours": "16", "auto_judgement": True},
        )
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, _lease = build(folder, "04")
            self.assertEqual(service.authorization_duration_choices(), [8.0, 16.0, 24.0])
            self.assertEqual(
                service._resolve_authorization_choices({"duration_hours": "24"}),
                {"duration_hours": 24.0, "auto_judgement": False},
            )
            with self.assertRaises(ServiceError):
                service._resolve_authorization_choices({"duration_hours": "12"})

    def test_a_new_lease_takes_the_chosen_hours_and_judgement(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(make_settings(Path(folder), sample_state(0.2, 0.1)))
            workflow = start_test_workflow(service)
            proposal = service.request_autonomy_lease(
                workflow["id"], "unittest-agent", "進入 JY 自動量測模式", "test"
            )
            service._approve_pending_proposal(
                proposal["id"],
                "operator",
                "unit_test",
                authorization_choices=service._resolve_authorization_choices(
                    {"duration_hours": "16", "auto_judgement": True}
                ),
            )
            lease = service.autonomy_status(lease_id=proposal["autonomy_lease_id"])
            self.assertEqual(lease["status"], "active")
            self.assertEqual(lease["duration_hours"], 16.0)
            self.assertEqual(lease["auto_judgement"], 1)
            self.assertAlmostEqual(hours_left(service, lease["id"]), 16.0, delta=0.05)


class ReauthorizationTests(unittest.TestCase):
    def test_reauthorize_without_a_run_offers_a_renewal_at_once(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "04")
            control(service, lease, "reauthorize")
            old = service._autonomy_row(lease["id"])
            self.assertEqual(old["status"], "revoked")
            renewal = service.db.one(
                "SELECT * FROM autonomy_leases WHERE workflow_id = ? AND status = 'pending'",
                (workflow["id"],),
            )
            self.assertIsNotNone(renewal)
            proposal = service.proposal(renewal["proposal_id"])
            self.assertEqual(proposal["payload"]["renewal_of"], lease["id"])

            waited = service.wait_for_autonomy_status(lease["id"], timeout_seconds=0)
            self.assertEqual(waited["superseded_lease_id"], lease["id"])
            self.assertEqual(waited["authorization"]["id"], renewal["id"])
            self.assertEqual(len(waited["operator_handoff"]["lines"]), 3)

            with self.assertRaises(ServiceError):
                service.renew_lease_from_browser(
                    proposal["id"],
                    f"APPROVE {proposal['id']}",
                    service.approval_csrf_token(proposal["id"]),
                    "127.0.0.1",
                    actor="operator",
                    authorization_choices={"duration_hours": "12"},
                )
            service.renew_lease_from_browser(
                proposal["id"],
                f"APPROVE {proposal['id']}",
                service.approval_csrf_token(proposal["id"]),
                "127.0.0.1",
                actor="operator",
                authorization_choices={"duration_hours": "24", "auto_judgement": True},
            )
            renewed = service.autonomy_status(lease_id=renewal["id"])
            self.assertEqual(renewed["status"], "active")
            self.assertEqual(renewed["duration_hours"], 24.0)
            self.assertEqual(renewed["auto_judgement"], 1)
            self.assertAlmostEqual(hours_left(service, renewal["id"]), 24.0, delta=0.05)
            self.assertEqual(service._workflow(workflow["id"])["current_node"], "04")

    def test_reauthorize_waits_for_the_running_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "04")
            service.db.execute(
                "INSERT INTO proposals(id, workflow_id, kind, payload_json, status, "
                "created_at, expires_at, max_uses, uses, source_client) VALUES "
                "('p-run', ?, 'run', '{}', 'approved', '2026-10-20T00:00:00+00:00', "
                "'2026-10-20T01:00:00+00:00', 1, 1, 'test')",
                (workflow["id"],),
            )
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, parameters_json, "
                "status, autonomy_lease_id) VALUES ('r-live', ?, 'p-run', '04', ?, "
                "'running', ?)",
                (workflow["id"], json_dumps({"qubits": ["q1"]}), lease["id"]),
            )
            control(service, lease, "reauthorize")
            row = service._autonomy_row(lease["id"])
            self.assertEqual(row["status"], "paused")
            self.assertEqual(row["stopped_reason"], AgentService.REAUTHORIZE_REASON)
            waited = service.wait_for_autonomy_status(lease["id"], timeout_seconds=0)
            self.assertEqual(waited["authorization"]["id"], lease["id"])
            self.assertIn("operator_handoff", waited)

            service.db.execute("UPDATE runs SET status = 'completed' WHERE id = 'r-live'")
            service.autonomy_watchdog_sweep()
            self.assertEqual(service._autonomy_row(lease["id"])["status"], "revoked")
            self.assertIsNotNone(
                service.db.one(
                    "SELECT id FROM autonomy_leases WHERE workflow_id = ? "
                    "AND status = 'pending'",
                    (workflow["id"],),
                )
            )

    def test_home_shows_the_reauthorize_button(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "04")
            session = service._start_dashboard_session(
                workflow["id"], "autonomous", "unittest-agent", autonomy_lease_id=lease["id"]
            )
            path = f"/session/{session['id']}/home"
            page = asyncio.run(
                handle_session_dashboard(
                    _request(path, {"session_id": session["id"]}), service, view="home"
                )
            ).body.decode("utf-8")
            self.assertIn('value="reauthorize"', page)
            self.assertIn("停止核准", page)
            self.assertIn("全自動判斷模式：關</h2>", page)


class PauseCardTests(unittest.TestCase):
    def test_the_pause_card_is_simple_and_restarts_the_clock(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "02c")
            pause(service, lease)
            proposal = replan_proposal(service, workflow["id"])
            html = _replan_card(service, "session", proposal, "zh-Hant")
            self.assertIn('value="replan_continue"', html)
            self.assertIn('name="duration_hours"', html)
            self.assertIn("<details>", html)
            self.assertLess(html.index('value="replan_continue"'), html.index("<details>"))

            service.replan_continue_from_browser(
                proposal["id"],
                service.approval_csrf_token(proposal["id"]),
                "operator",
                {"duration_hours": "16", "auto_judgement": True},
            )
            resumed = service.autonomy_status(lease_id=lease["id"])
            self.assertEqual(resumed["status"], "active")
            self.assertEqual(resumed["duration_hours"], 16.0)
            self.assertEqual(resumed["auto_judgement"], 1)
            self.assertAlmostEqual(hours_left(service, lease["id"]), 16.0, delta=0.05)


class ContinuationAfterRecoveryTests(unittest.TestCase):
    def test_a_recovered_workflow_continues_with_the_simple_card(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "04")
            closed = service.prepare_recovery_close("operator")
            self.assertEqual(closed["status"], "ready")
            self.assertEqual(service._workflow(workflow["id"])["end_kind"], "recovery")

            result = service.enter_autonomy_mode("unittest-agent", "進入 JY 自動量測模式")
            self.assertEqual(result["continuation_of"], workflow["id"])
            self.assertEqual(result["next_action"], "human_approval")
            self.assertTrue(result["browser_url"])
            proposal = result["approval"]
            continuation = proposal["payload"]["continuation"]
            self.assertEqual(continuation["of_workflow_id"], workflow["id"])
            self.assertTrue(continuation["qubit_plan"])
            html = _continuation_card(service, "session", proposal, "en")
            self.assertIn('value="continuation_approve"', html)
            self.assertIn('name="duration_hours"', html)
            self.assertNotIn('value="preview"', html)

            service.approve_continuation_from_browser(
                proposal["id"],
                f"APPROVE {proposal['id']}",
                service.approval_csrf_token(proposal["id"]),
                "127.0.0.1",
                actor="operator",
                authorization_choices={"duration_hours": "8", "auto_judgement": True},
            )
            new_lease = service.autonomy_status(lease_id=result["authorization"]["id"])
            self.assertEqual(new_lease["status"], "active")
            self.assertEqual(new_lease["auto_judgement"], 1)
            self.assertNotEqual(new_lease["workflow_id"], workflow["id"])
            self.assertEqual(service._workflow(workflow["id"])["end_kind"], "continued")

    def test_the_continue_phrase_also_continues_after_a_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "04")
            service.prepare_recovery_close("operator")
            result = service.continue_autonomy_mode("unittest-agent", "繼續JY自動量測模式")
            self.assertEqual(result["continuation_of"], workflow["id"])

    def test_full_options_replace_the_continuation_card(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, _lease = build(folder, "04")
            service.prepare_recovery_close("operator")
            result = service.enter_autonomy_mode("unittest-agent", "進入 JY 自動量測模式")
            proposal_id = result["approval"]["id"]
            service.expand_continuation_from_browser(
                proposal_id, service.approval_csrf_token(proposal_id), "operator"
            )
            payload = service.proposal(proposal_id)["payload"]
            self.assertNotIn("continuation", payload)
            self.assertIn("workflow_options_offer", payload)

    def test_a_dashboard_shutdown_stays_an_operator_end_after_recover(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "04")
            service.db.event(
                "full_shutdown_requested",
                "authorized operator via public HTTPS dashboard",
                {"session_id": "s", "status": "waiting_for_run"},
                workflow["id"],
            )
            service.prepare_recovery_close("operator")
            self.assertEqual(service._workflow(workflow["id"])["end_kind"], "operator")
            result = service.enter_autonomy_mode("unittest-agent", "進入 JY 自動量測模式")
            self.assertNotIn("continuation_of", result)

    def test_a_mislabelled_dashboard_shutdown_is_not_continued(self) -> None:
        """2026-10-20: a Recover run by an interim build labelled the 06:22
        Dashboard shutdown `recovery`; the audit log still says who ended it."""

        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "04")
            service.db.event(
                "full_shutdown_requested", "operator", {"session_id": "s"}, workflow["id"]
            )
            service.prepare_recovery_close("operator")
            service.db.execute(
                "UPDATE workflows SET end_kind = 'recovery' WHERE id = ?", (workflow["id"],)
            )
            self.assertIsNone(service._recovered_workflow())

    def test_changed_active_qubits_bring_back_the_full_lease_page(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, _lease = build(folder, "04")
            service.prepare_recovery_close("operator")
            state = json_loads(service.settings.active_state.read_text(encoding="utf-8"), {})
            state["active_qubit_names"] = ["q2"]
            atomic_write_json(service.settings.active_state, state)
            result = service.enter_autonomy_mode("unittest-agent", "進入 JY 自動量測模式")
            self.assertNotIn("continuation_of", result)
            self.assertEqual(result["workflow"]["targets"], ["q2"])

    def test_workflows_ended_before_the_upgrade_count_as_operator_ends(self) -> None:
        import sqlite3

        from jy_agent.db import Database

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE workflows (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
                "updated_at TEXT NOT NULL, status TEXT NOT NULL, targets_json TEXT NOT NULL, "
                "initial_parameters_json TEXT NOT NULL, current_node TEXT, "
                "client_id TEXT NOT NULL, options_json TEXT)"
            )
            for workflow_id, status in (("old", "recovery_required"), ("open", "active")):
                connection.execute(
                    "INSERT INTO workflows VALUES (?, 't', 't', ?, '[]', '{}', '03a', 'c', NULL)",
                    (workflow_id, status),
                )
            connection.commit()
            connection.close()
            database = Database(path)
            kinds = {
                row["id"]: row["end_kind"]
                for row in database.all("SELECT id, end_kind FROM workflows")
            }
            self.assertEqual(kinds, {"old": "operator", "open": None})

    def test_an_operator_end_brings_back_the_full_lease_page(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "04")
            service.leave_measurement_mode(workflow["id"], "operator", "結束量測")
            self.assertEqual(service._workflow(workflow["id"])["end_kind"], "operator")
            result = service.enter_autonomy_mode("unittest-agent", "進入 JY 自動量測模式")
            self.assertNotIn("continuation_of", result)
            self.assertNotIn("continuation", result["approval"]["payload"])


class FullAutoJudgementTests(unittest.TestCase):
    def build(self, folder: str):
        service = AgentService(make_settings(Path(folder), sample_state(0.2, 0.1)))
        workflow = start_test_workflow(service)
        proposal, lease = activate_lease(service, workflow["id"])
        service.db.execute(
            "UPDATE autonomy_leases SET auto_judgement = 1 WHERE id = ?", (lease["id"],)
        )
        cleanups: list[bool] = []

        def fake_cleanup(dry_run: bool) -> dict:
            cleanups.append(dry_run)
            return {"ok": True, "closed": ["qm-1"], "machines": []}

        service._qm_cleanup_runner = fake_cleanup
        return service, workflow, proposal, lease, cleanups

    def crash(self, service, workflow, proposal, lease, run_id: str) -> None:
        insert_failed_run(
            service, workflow["id"], proposal["id"], lease["id"],
            run_id=run_id,
            node_id="07b",
            parameters={"qubits": ["q1"]},
            traceback_text=UNKNOWN_TRACEBACK,
        )
        service.autonomy_watchdog_sweep()
        wait_for_cleanup(service)

    def test_an_unknown_crash_is_retried_then_marked_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal, lease, cleanups = self.build(folder)
            self.crash(service, workflow, proposal, lease, "crash-1")
            self.assertEqual(service.autonomy_status(lease_id=lease["id"])["status"], "active")
            self.crash(service, workflow, proposal, lease, "crash-2")
            self.crash(service, workflow, proposal, lease, "crash-3")
            self.assertEqual(service.autonomy_status(lease_id=lease["id"])["status"], "active")
            actions = [entry["action"] for entry in reversed(service.auto_judgement_log(lease["id"]))]
            self.assertEqual(actions, ["retry", "retry", "mark_incomplete"])
            marks = service._scientifically_unmeasurable_targets_by_node(
                workflow["id"], lease_id=lease["id"]
            )
            self.assertEqual(marks.get("07b"), {"q1"})
            self.assertEqual(len(cleanups), 3)
            # Each crash is decided once, however often the watchdog sweeps.
            service.autonomy_watchdog_sweep()
            self.assertEqual(len(service.auto_judgement_log(lease["id"])), 3)

    def test_the_lease_budget_hands_back_to_the_operator(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal, lease, _cleanups = self.build(folder)
            service.policy.raw["autonomy"]["auto_judgement"]["max_per_lease"] = 1
            self.crash(service, workflow, proposal, lease, "crash-1")
            self.assertEqual(service.autonomy_status(lease_id=lease["id"])["status"], "active")
            self.crash(service, workflow, proposal, lease, "crash-2")
            self.assertEqual(service.autonomy_status(lease_id=lease["id"])["status"], "paused")

    def test_a_retained_hardware_lock_hands_back_to_the_operator(self) -> None:
        """Audit 2026-10-20: a retry cannot start while the lock is retained."""

        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal, lease, cleanups = self.build(folder)
            service.settings.lock_path.parent.mkdir(parents=True, exist_ok=True)
            service.settings.lock_path.write_text('{"run_id": "crash-1"}', encoding="utf-8")
            self.crash(service, workflow, proposal, lease, "crash-1")
            self.assertEqual(service.autonomy_status(lease_id=lease["id"])["status"], "paused")
            self.assertEqual(service.auto_judgement_log(lease["id"]), [])
            self.assertEqual(cleanups, [])
            self.pause_for_outage(service, workflow, proposal, lease, submission=True)
            self.assertFalse(service._auto_resume_paused_lease(service._autonomy_row(lease["id"])))

    def test_without_full_auto_an_unknown_crash_still_pauses(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal, lease, cleanups = self.build(folder)
            service.db.execute(
                "UPDATE autonomy_leases SET auto_judgement = 0 WHERE id = ?", (lease["id"],)
            )
            self.crash(service, workflow, proposal, lease, "crash-1")
            self.assertEqual(service.autonomy_status(lease_id=lease["id"])["status"], "paused")
            self.assertEqual(cleanups, [])

    def test_runs_wait_while_the_stale_machine_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _proposal, lease, _cleanups = self.build(folder)
            service._qm_cleanup_active.set()
            with self.assertRaises(AutonomyScopeError):
                service.start_authorized_run(
                    workflow["id"], lease["id"], "02c", {"qubits": ["q1"]}, "test", "agent"
                )
            self.assertEqual(service.autonomy_status(lease_id=lease["id"])["status"], "active")
            service._qm_cleanup_active.clear()

    def pause_for_outage(self, service, workflow, proposal, lease, *, submission: bool) -> None:
        analysis = {
            "analysis_status": "failed",
            "failure_category": (
                "program_submission_timeout" if submission else "instrument_unreachable"
            ),
            "measurement_paused": True,
        }
        service.db.execute(
            "INSERT INTO runs(id, workflow_id, proposal_id, node_id, parameters_json, "
            "status, analysis_status, analysis_json, termination_cause, autonomy_lease_id) "
            "VALUES ('outage', ?, ?, '03a', ?, 'failed', 'failed', ?, ?, ?)",
            (
                workflow["id"],
                proposal["id"],
                json_dumps({"qubits": ["q2"]}),
                json_dumps(analysis),
                "program_submission_timeout" if submission else "instrument_unreachable",
                lease["id"],
            ),
        )
        service.db.execute(
            "UPDATE autonomy_leases SET status = 'paused', stopped_reason = 'outage' "
            "WHERE id = ?",
            (lease["id"],),
        )
        service.db.event(
            "measurement_paused_for_instrument_error",
            "worker",
            {"run_id": "outage", "reason": "outage"},
            workflow["id"],
        )

    def test_an_outage_resumes_once_the_qop_answers(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal, lease, cleanups = self.build(folder)
            self.pause_for_outage(service, workflow, proposal, lease, submission=False)
            answers = [False, True]

            def probe(_wiring):
                return types.SimpleNamespace(
                    reachable=answers.pop(0), as_dict=lambda: {"stage": "rpc"}
                )

            service._instrument_probe = probe
            row = service._autonomy_row(lease["id"])
            self.assertFalse(service._auto_resume_paused_lease(row))
            self.assertEqual(service._autonomy_row(lease["id"])["status"], "paused")
            # The probe is throttled; the next one is allowed after the interval.
            service._auto_probe_at.clear()
            self.assertTrue(service._auto_resume_paused_lease(service._autonomy_row(lease["id"])))
            self.assertEqual(service.autonomy_status(lease_id=lease["id"])["status"], "active")
            self.assertEqual(cleanups, [False])
            self.assertEqual(
                service.auto_judgement_log(lease["id"])[0]["action"], "resume_after_outage"
            )

    def test_a_single_target_submission_timeout_is_marked_and_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal, lease, _cleanups = self.build(folder)
            self.pause_for_outage(service, workflow, proposal, lease, submission=True)
            service._instrument_probe = lambda _wiring: self.fail("no probe needed")
            self.assertTrue(service._auto_resume_paused_lease(service._autonomy_row(lease["id"])))
            self.assertEqual(service.autonomy_status(lease_id=lease["id"])["status"], "active")
            marks = service._scientifically_unmeasurable_targets_by_node(
                workflow["id"], lease_id=lease["id"]
            )
            self.assertEqual(marks.get("03a"), {"q2"})

    def test_a_manual_pause_is_never_lifted(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, _proposal, lease, _cleanups = self.build(folder)
            control(service, lease, "pause")
            self.assertFalse(service._auto_resume_paused_lease(service._autonomy_row(lease["id"])))
            self.assertEqual(service._autonomy_row(lease["id"])["status"], "paused")


class WatchdogEventTests(unittest.TestCase):
    def test_a_submission_timeout_is_logged_once(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(make_settings(Path(folder), sample_state(0.2, 0.1)))
            workflow = start_test_workflow(service)
            proposal, lease = activate_lease(service, workflow["id"])
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, parameters_json, "
                "status, analysis_status, analysis_json, termination_cause, autonomy_lease_id) "
                "VALUES ('big', ?, ?, '03a', ?, 'failed', 'failed', ?, "
                "'program_submission_timeout', ?)",
                (
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": ["q1", "q2"]}),
                    json_dumps({"failure_category": "program_submission_timeout"}),
                    lease["id"],
                ),
            )
            for _ in range(3):
                service.autonomy_watchdog_sweep()
            count = service.db.one(
                "SELECT COUNT(*) AS n FROM events "
                "WHERE event_type = 'autonomy_program_submission_timeout'"
            )["n"]
            self.assertEqual(count, 1)


class ShadowModeTests(unittest.TestCase):
    def test_runs_and_decisions_are_compared_with_the_planner(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "04")
            plan = service.next_action(workflow["id"])
            parameters = dict((plan.get("run") or {}).get("parameters") or {"qubits": ["q1"]})
            for candidate in (parameters, {"qubits": ["q9"]}):
                record = service._shadow_run_record(workflow["id"], lease["id"], "04", candidate)
                self.assertIsNotNone(record)
                service.db.event("autopilot_shadow", "test", record, workflow["id"])
            report = service.autopilot_shadow_report(workflow["id"])
            runs = report["stages"]["run"]
            self.assertEqual(runs["total"], 2)
            if plan.get("action") == "run":
                self.assertEqual(runs["matched"], 1)
            self.assertIn("timing", report)
            self.assertTrue(report["recent_mismatches"])


class HistorySiteTests(unittest.TestCase):
    """Operator request 2026-10-22: results stay readable after shutdown."""

    def session(self, folder: str):
        from test_per_qubit_pass_rules import fine_03a, insert_run

        service, workflow, lease = build(folder, "03a")
        session = service._start_dashboard_session(
            workflow["id"], "autonomous", "unittest-agent", autonomy_lease_id=lease["id"]
        )
        plot = service.settings.data_root / "2026-10-22" / "#1_03a" / "figure.png"
        plot.parent.mkdir(parents=True, exist_ok=True)
        plot.write_bytes(bytes([0x89]) + b"PNG" + b"0" * 32)
        document = fine_03a("q1", 3.02e9)
        document["plots"] = [str(plot)]
        run_id = insert_run(
            service, workflow["id"], "03a", "q1", document,
            lease_id=lease["id"], decision="repeat", next_node="03a",
        )
        return service, workflow, lease, session, run_id

    def test_the_site_lists_sessions_and_keeps_their_results(self) -> None:
        from jy_agent.history_site import site_root, write_history_site

        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease, session, run_id = self.session(folder)
            index = write_history_site(service, everything=True)
            root = site_root(service)
            page = index.read_text(encoding="utf-8")
            self.assertIn(f'sessions/{session["id"]}/index.html', page)
            # While the session runs the homepage links to the live Dashboard.
            self.assertIn(service.session_home_url(session["id"]), page)
            session_page = (root / "sessions" / session["id"] / "index.html").read_text(
                encoding="utf-8"
            )
            self.assertIn(run_id[:8], session_page)
            images = list((root / "sessions" / session["id"] / "images").iterdir())
            self.assertEqual(len(images), 1)
            self.assertIn(f"images/{images[0].name}", session_page)
            self.assertNotIn("/session/", session_page.replace("/sessions/", ""))

            # After the measurement ends the page stays, the live link goes.
            service.leave_measurement_mode(workflow["id"], "operator", "結束量測")
            page = index.read_text(encoding="utf-8")
            self.assertNotIn(service.session_home_url(session["id"]), page)
            self.assertIn(f'sessions/{session["id"]}/index.html', page)

    def test_an_analysis_refreshes_the_session_page(self) -> None:
        from jy_agent.history_site import site_root

        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease, session, _run_id = self.session(folder)
            service._refresh_history_site(workflow["id"])
            self.assertTrue(
                (site_root(service) / "sessions" / session["id"] / "index.html").is_file()
            )

    def test_the_refresh_runs_off_the_calling_thread(self) -> None:
        from jy_agent.history_site import site_root

        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease, session, _run_id = self.session(folder)
            service.HISTORY_SITE_ASYNC = True
            service._refresh_history_site(workflow["id"])
            thread = service._history_site_thread
            self.assertIsNotNone(thread)
            thread.join(timeout=60)
            self.assertTrue(
                (site_root(service) / "sessions" / session["id"] / "index.html").is_file()
            )

    def test_the_static_review_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease, session, _run_id = self.session(folder)
            service.db.execute(
                "INSERT INTO proposals(id, workflow_id, kind, payload_json, status, "
                "created_at, expires_at, max_uses, uses, source_client) VALUES "
                "('old', ?, 'run', '{}', 'pending', '2026-10-01T00:00:00+00:00', "
                "'2026-10-01T01:00:00+00:00', 1, 0, 'test')",
                (workflow["id"],),
            )
            service.dashboard_review(session["id"], maintain=False)
            self.assertEqual(
                service.db.one("SELECT status FROM proposals WHERE id = 'old'")["status"],
                "pending",
            )
            service.dashboard_review(session["id"])
            self.assertEqual(
                service.db.one("SELECT status FROM proposals WHERE id = 'old'")["status"],
                "expired",
            )


class QmCleanupTests(unittest.TestCase):
    STATE = {
        "ports": {
            "mw_outputs": {
                "con1": {"1": {"1": {"controller_id": "con1", "fem_id": 1, "port_id": 1}}}
            },
            "analog_outputs": {
                "con1": {"2": {"4": {"controller_id": "con1", "fem_id": 2, "port_id": 4}}}
            },
        }
    }

    def test_ports_and_ownership(self) -> None:
        chip = state_ports(self.STATE)
        self.assertEqual(chip, {("con1", 1, 1), ("con1", 2, 4)})
        ours = {"elements": {"q1.xy": {"MWInput": {"port": ("con1", 1, 1)}}}}
        shared = {
            "elements": {
                "a": {"MWInput": {"port": ["con1", 1, 1]}},
                "b": {"singleInput": {"port": ("con1", 3, 1)}},
            }
        }
        self.assertEqual(classify(config_ports(ours), chip), "ours")
        self.assertEqual(classify(config_ports(shared), chip), "overlapping")
        self.assertEqual(classify(set(), chip), "unrelated")

    def test_only_this_chips_machine_is_closed(self) -> None:
        events: list[str] = []

        class JobData:
            """qm-qua 1.4 on QOP 3 lists jobs as records with only an id."""

            id = "job-1"

        class Job:
            id = "job-1"

            def cancel(self) -> None:
                events.append("cancel job-1")

        class Machine:
            def __init__(self, name: str, config: dict) -> None:
                self.name, self.config = name, config

            def get_config(self) -> dict:
                return self.config

            def get_jobs(self, status=None):
                assert "Running" in status
                return [JobData()] if self.name == "mine" else []

            def get_job(self, job_id: str):
                assert job_id == "job-1"
                return Job()

            def close(self) -> None:
                events.append(f"close {self.name}")

        machines = {
            "mine": Machine("mine", {"elements": {"x": {"MWInput": {"port": ("con1", 1, 1)}}}}),
            "theirs": Machine(
                "theirs",
                {"elements": {"x": {"MWInput": {"port": ("con1", 1, 1)}}, "y": {"singleInput": {"port": ("con2", 1, 1)}}}},
            ),
        }

        class Manager:
            def __init__(self, **_settings) -> None:
                pass

            def list_open_qms(self):
                return list(machines)

            def get_qm(self, machine_id):
                return machines[machine_id]

        fake = types.ModuleType("qm")
        fake.QuantumMachinesManager = Manager
        previous = sys.modules.get("qm")
        sys.modules["qm"] = fake
        try:
            with tempfile.TemporaryDirectory() as folder:
                state = Path(folder) / "state.json"
                wiring = Path(folder) / "wiring.json"
                atomic_write_json(state, self.STATE)
                atomic_write_json(wiring, {"network": {"host": "127.0.0.1", "port": 80}})
                report = close_stale_qms(state, wiring)
        finally:
            if previous is None:
                sys.modules.pop("qm", None)
            else:
                sys.modules["qm"] = previous
        self.assertTrue(report["ok"])
        self.assertEqual(report["closed"], ["mine"])
        self.assertEqual(report["overlapping_not_closed"], ["theirs"])
        self.assertEqual(events, ["cancel job-1", "close mine"])

    def test_an_old_style_running_job_is_halted(self) -> None:
        from jy_agent.qm_cleanup import _halt_running_jobs

        events: list[str] = []

        class OldJob:
            id = "old"

            def halt(self) -> None:
                events.append("halt old")

        class OldMachine:
            def get_running_job(self):
                return OldJob()

        self.assertEqual(_halt_running_jobs(OldMachine()), ["halt old"])
        self.assertEqual(events, ["halt old"])

    def test_cleanup_is_refused_while_a_run_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "04")
            service._qm_cleanup_runner = lambda dry_run: {"ok": True}
            self.assertEqual(service.close_stale_qm("operator", "test"), {"ok": True})
            service.db.execute(
                "INSERT INTO proposals(id, workflow_id, kind, payload_json, status, "
                "created_at, expires_at, max_uses, uses, source_client) VALUES "
                "('p-live', ?, 'run', '{}', 'approved', '2026-10-20T00:00:00+00:00', "
                "'2026-10-20T01:00:00+00:00', 1, 1, 'test')",
                (workflow["id"],),
            )
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, parameters_json, "
                "status, autonomy_lease_id) VALUES ('live', ?, 'p-live', '04', '{}', "
                "'running', ?)",
                (workflow["id"], lease["id"]),
            )
            with self.assertRaises(ServiceError):
                service.close_stale_qm("operator", "test")


if __name__ == "__main__":
    unittest.main()
