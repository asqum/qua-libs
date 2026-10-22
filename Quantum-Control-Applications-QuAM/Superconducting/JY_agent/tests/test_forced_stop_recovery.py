"""Forced-stop hardware-lock recovery (operator decision 2026-10-20).

A worker terminated after the emergency-stop grace never records its
post-run state hash and may leave this chip's quantum machine open on the
shared QOP. Recovery accepts `forced_stop_checkpoint` evidence once a fresh
stale-QM cleanup shows no machine of this chip (or sharing its ports) is
open; other users' machines are ignored. All QM access is mocked.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlencode

from jy_agent.dashboard_access import DashboardAccessManager
from jy_agent.db import Database
from jy_agent.operator_web import handle_operator_console
from jy_agent.recovery import HardwareLockRecovery
from jy_agent.service import AgentService
from jy_agent.util import atomic_write_json, utc_now

from test_recovery import RUN_ID, WORKFLOW_ID, make_settings, seed_retained_lock
from test_shutdown_and_operator_web import _request


OURS_CLOSED = {"qm_id": "QM-ours", "verdict": "ours", "closed": True, "jobs": []}
OURS_OPEN = {"qm_id": "QM-ours", "verdict": "ours", "closed": False, "error": "busy"}
UNRELATED = {"qm_id": "QM-other-user", "verdict": "unrelated", "ports": []}
OVERLAPPING = {"qm_id": "QM-overlap", "verdict": "overlapping", "ports": []}


def _report(*machines: dict) -> dict:
    return {"ok": True, "closed": [], "machines": list(machines), "dry_run": False}


def _seed_forced_stop(settings, database) -> None:
    seed_retained_lock(settings, database)
    database.execute(
        "UPDATE runs SET status = 'force_stopped', active_state_hash_after = NULL, "
        "termination_cause = 'forced_after_grace', error = ? WHERE id = ?",
        (
            "Emergency stop grace period elapsed; the exact worker was terminated. "
            "The hardware lock was retained for manual safety inspection.",
            RUN_ID,
        ),
    )


def _record(recovery: HardwareLockRecovery, settings, report: dict) -> None:
    recovery.record_qm_cleanup(
        RUN_ID,
        report,
        state_path=settings.active_state,
        wiring_path=settings.wiring_path,
        actor="test",
        dry_run=False,
    )


class ForcedStopInspectionTests(unittest.TestCase):
    def _inspect(self, report: dict | None, *, mutate=None) -> dict:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder))
            database = Database(settings.database_path)
            _seed_forced_stop(settings, database)
            recovery = HardwareLockRecovery(settings, database)
            if report is not None:
                _record(recovery, settings, report)
            if mutate is not None:
                mutate(settings, database, recovery)
            with patch.object(recovery, "_active_worker_pids", return_value=(True, [])):
                return recovery.inspect(RUN_ID)

    def test_clean_cleanup_and_matching_hashes_are_ready(self) -> None:
        result = self._inspect(_report(OURS_CLOSED))
        self.assertTrue(result["ready"], result["failures"])
        self.assertEqual(result["checks"]["state_evidence_mode"], "forced_stop_checkpoint")
        self.assertTrue(result["checks"]["qm_cleanup_clean"])

    def test_open_machine_of_this_chip_blocks_with_its_id(self) -> None:
        result = self._inspect(_report(OURS_OPEN, UNRELATED))
        self.assertFalse(result["ready"])
        self.assertTrue(
            any(
                "this chip still has an open quantum machine qm-ours" in item.casefold()
                and "close it first" in item
                for item in result["failures"]
            ),
            result["failures"],
        )

    def test_overlapping_machine_blocks(self) -> None:
        result = self._inspect(_report(OVERLAPPING))
        self.assertFalse(result["ready"])
        self.assertTrue(any("QM-overlap" in item for item in result["failures"]))

    def test_only_other_users_machine_is_ignored(self) -> None:
        result = self._inspect(_report(UNRELATED))
        self.assertTrue(result["ready"], result["failures"])
        self.assertEqual(
            result["checks"]["qm_cleanup_summary"]["unrelated_ignored"],
            ["QM-other-user"],
        )

    def test_missing_cleanup_receipt_blocks(self) -> None:
        result = self._inspect(None)
        self.assertFalse(result["ready"])
        self.assertTrue(
            any("No quantum-machine cleanup check" in item for item in result["failures"])
        )

    def test_unreachable_qop_blocks(self) -> None:
        result = self._inspect({"ok": False, "error": "Could not reach the QOP", "machines": []})
        self.assertFalse(result["ready"])
        self.assertTrue(any("could not check the QOP" in item for item in result["failures"]))

    def test_state_hash_mismatch_blocks(self) -> None:
        def change_state(settings, database, recovery) -> None:
            settings.active_state.write_text('{"changed": true}\n', encoding="utf-8")

        result = self._inspect(_report(OURS_CLOSED), mutate=change_state)
        self.assertFalse(result["ready"])
        self.assertEqual(result["checks"]["state_evidence_mode"], "insufficient")
        self.assertTrue(
            any("state.json changed since the run started" in item for item in result["failures"]),
            result["failures"],
        )

    def test_old_receipt_blocks(self) -> None:
        def age_receipt(settings, database, recovery) -> None:
            old = datetime.now(timezone.utc) - timedelta(minutes=20)
            database.execute(
                "UPDATE runs SET finished_at = ? WHERE id = ?",
                ((old - timedelta(hours=1)).isoformat(timespec="seconds"), RUN_ID),
            )
            receipt = recovery.read_qm_receipt(RUN_ID)
            receipt["checked_at"] = old.isoformat(timespec="seconds")
            atomic_write_json(recovery.qm_receipt_path(RUN_ID), receipt)

        result = self._inspect(_report(OURS_CLOSED), mutate=age_receipt)
        self.assertFalse(result["ready"])
        self.assertTrue(any("older than 10 minutes" in item for item in result["failures"]))

    def test_receipt_before_termination_blocks(self) -> None:
        def later_finish(settings, database, recovery) -> None:
            later = datetime.now(timezone.utc) + timedelta(seconds=30)
            database.execute(
                "UPDATE runs SET finished_at = ? WHERE id = ?",
                (later.isoformat(timespec="seconds"), RUN_ID),
            )

        result = self._inspect(_report(OURS_CLOSED), mutate=later_finish)
        self.assertFalse(result["ready"])
        self.assertTrue(any("predates the worker termination" in item for item in result["failures"]))


class ForcedStopOperatorConsoleTests(unittest.TestCase):
    def test_console_offers_cleanup_and_marks_other_users_machine_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder))
            database = Database(settings.database_path)
            _seed_forced_stop(settings, database)
            service = AgentService(settings)
            access = DashboardAccessManager(service.db, instance_nonce="local-development")
            calls: list[tuple[Path, Path, bool]] = []

            def fake_cleanup(state: Path, wiring: Path, dry_run: bool) -> dict:
                calls.append((state, wiring, dry_run))
                return _report(OURS_CLOSED, UNRELATED)

            service._qm_cleanup_runner_for = fake_cleanup
            with patch.object(
                HardwareLockRecovery, "_active_worker_pids", return_value=(True, [])
            ):
                before = asyncio.run(
                    handle_operator_console(_request("/operator", port=8765), service, access)
                ).body.decode("utf-8")
                self.assertIn("Recovery is blocked", before)
                self.assertIn("Close this chip's stale quantum machine", before)
                self.assertIn('value="close_stale_qm"', before)

                form = urlencode(
                    {
                        "operation": "close_stale_qm",
                        "csrf_token": service.operator_csrf_token(),
                    }
                ).encode("ascii")
                response = asyncio.run(
                    handle_operator_console(
                        _request("/operator", method="POST", body=form, port=8765),
                        service,
                        access,
                    )
                )
            page = response.body.decode("utf-8")
            self.assertEqual(response.status_code, 200, page)
            self.assertEqual(calls, [(settings.active_state, settings.wiring_path, False)])
            self.assertIn("QM-other-user", page)
            self.assertIn("other user, ignored", page)
            self.assertIn(f"RECOVER HARDWARE LOCK {RUN_ID}", page)
            self.assertTrue(settings.lock_path.exists())

    def test_console_rejects_cleanup_without_csrf(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder))
            database = Database(settings.database_path)
            _seed_forced_stop(settings, database)
            service = AgentService(settings)
            access = DashboardAccessManager(service.db, instance_nonce="local-development")
            service._qm_cleanup_runner_for = lambda *args: self.fail("must not run")
            form = urlencode({"operation": "close_stale_qm", "csrf_token": "x"}).encode()
            with patch.object(
                HardwareLockRecovery, "_active_worker_pids", return_value=(True, [])
            ):
                response = asyncio.run(
                    handle_operator_console(
                        _request("/operator", method="POST", body=form, port=8765),
                        service,
                        access,
                    )
                )
            self.assertEqual(response.status_code, 400)


class ForcedStopAutoReleaseTests(unittest.TestCase):
    def _service(self, folder: str, cleanup) -> tuple[AgentService, object]:
        settings = make_settings(Path(folder))
        database = Database(settings.database_path)
        _seed_forced_stop(settings, database)
        service = AgentService(settings)
        service._qm_cleanup_runner_for = cleanup
        return service, settings

    def test_clean_cleanup_releases_lock_and_records_post_run_hash(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, settings = self._service(
                folder, lambda state, wiring, dry_run: _report(OURS_CLOSED, UNRELATED)
            )
            with patch.object(
                HardwareLockRecovery, "_active_worker_pids", return_value=(True, [])
            ):
                outcome = service.release_lock_after_forced_stop(RUN_ID, "test")
            self.assertTrue(outcome["released"], outcome)
            self.assertFalse(settings.lock_path.exists())
            recovery_dir = settings.runtime / "recovery"
            self.assertTrue((recovery_dir / f"{RUN_ID}-hardware-lock-recovery.json").is_file())
            self.assertTrue((recovery_dir / f"{RUN_ID}-hardware.lock.recovered.json").is_file())
            run = service.db.one("SELECT * FROM runs WHERE id = ?", (RUN_ID,))
            self.assertEqual(run["active_state_hash_after"], run["active_state_hash_before"])

    def test_open_machine_of_this_chip_keeps_lock(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, settings = self._service(
                folder, lambda state, wiring, dry_run: _report(OURS_OPEN)
            )
            with patch.object(
                HardwareLockRecovery, "_active_worker_pids", return_value=(True, [])
            ):
                outcome = service.release_lock_after_forced_stop(RUN_ID, "test")
            self.assertFalse(outcome["released"])
            self.assertIn("QM-ours", outcome["reason"])
            self.assertTrue(settings.lock_path.exists())

    def test_overlapping_machine_keeps_lock(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, settings = self._service(
                folder, lambda state, wiring, dry_run: _report(OVERLAPPING)
            )
            with patch.object(
                HardwareLockRecovery, "_active_worker_pids", return_value=(True, [])
            ):
                outcome = service.release_lock_after_forced_stop(RUN_ID, "test")
            self.assertFalse(outcome["released"])
            self.assertTrue(settings.lock_path.exists())

    def test_cleanup_exception_never_raises_and_keeps_lock(self) -> None:
        def boom(state, wiring, dry_run):
            raise RuntimeError("QOP gone")

        with tempfile.TemporaryDirectory() as folder:
            service, settings = self._service(folder, boom)
            with patch.object(
                HardwareLockRecovery, "_active_worker_pids", return_value=(True, [])
            ):
                outcome = service.release_lock_after_forced_stop(RUN_ID, "test")
            self.assertFalse(outcome["released"])
            self.assertIn("QOP gone", outcome["reason"])
            self.assertTrue(settings.lock_path.exists())

    def test_changed_state_keeps_lock(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, settings = self._service(
                folder, lambda state, wiring, dry_run: _report(OURS_CLOSED)
            )
            settings.active_state.write_text('{"changed": true}\n', encoding="utf-8")
            with patch.object(
                HardwareLockRecovery, "_active_worker_pids", return_value=(True, [])
            ):
                outcome = service.release_lock_after_forced_stop(RUN_ID, "test")
            self.assertFalse(outcome["released"])
            self.assertTrue(settings.lock_path.exists())
            run = service.db.one("SELECT * FROM runs WHERE id = ?", (RUN_ID,))
            self.assertIsNone(run["active_state_hash_after"])

    def _shutdown(self, cleanup) -> tuple[dict, dict, object]:
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        service, settings = self._service(folder, cleanup)
        session_id = "forced-stop-session"
        now = utc_now()
        service.db.execute(
            "INSERT INTO measurement_sessions(id, workflow_id, mode, status, "
            "started_at, updated_at, source_client) "
            "VALUES (?, ?, 'autonomous', 'halted', ?, ?, 'unittest')",
            (session_id, WORKFLOW_ID, now, now),
        )
        with patch.object(
            HardwareLockRecovery, "_active_worker_pids", return_value=(True, [])
        ):
            first = service.request_full_shutdown(session_id, "browser operator")
            service._forced_release_threads[RUN_ID].join(timeout=30)
            final = service.reconcile_full_shutdown_request()
        return first, final, settings

    def test_full_shutdown_auto_releases_instead_of_quarantine(self) -> None:
        first, final, settings = self._shutdown(
            lambda state, wiring, dry_run: _report(OURS_CLOSED, UNRELATED)
        )
        self.assertEqual(first["status"], "closing_stale_qm")
        self.assertEqual(final["status"], "ready", final)
        self.assertFalse(final["quarantine"])
        self.assertTrue(final["forced_stop_release"]["released"])
        self.assertFalse(settings.lock_path.exists())

    def test_full_shutdown_quarantines_when_cleanup_leaves_our_machine(self) -> None:
        first, final, settings = self._shutdown(
            lambda state, wiring, dry_run: _report(OURS_OPEN)
        )
        self.assertEqual(first["status"], "closing_stale_qm")
        self.assertEqual(final["status"], "quarantine_ready")
        self.assertFalse(final["forced_stop_release"]["released"])
        self.assertTrue(settings.lock_path.exists())
        self.assertTrue((settings.runtime / "recovery-required.json").is_file())


if __name__ == "__main__":
    unittest.main()
