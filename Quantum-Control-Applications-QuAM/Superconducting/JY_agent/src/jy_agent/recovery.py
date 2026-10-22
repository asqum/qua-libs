from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .db import Database
from .runner import ExperimentRunner
from .util import (
    atomic_write_json,
    exclusive_file_lock,
    parse_iso_datetime,
    sha256_file,
    utc_now,
)


class HardwareLockRecoveryError(RuntimeError):
    pass


def qm_cleanup_problems(report: Any) -> list[str]:
    """What, in a `qm_cleanup` report, still blocks this chip's recovery.

    Operator rule 2026-10-20: the OPX1000 is shared, so a machine with no
    port on this chip (`unrelated`) belongs to another user and is ignored.
    A machine of ours that is still open, a machine that shares some of this
    chip's ports, and a machine whose ports could not be read all block.
    """

    if not isinstance(report, dict):
        return ["The quantum-machine cleanup returned no report."]
    machines = report.get("machines")
    if not isinstance(machines, list) or (report.get("error") and not machines):
        return [
            "The quantum-machine cleanup could not check the QOP: "
            + str(report.get("error") or "no machine list was returned")
        ]
    problems: list[str] = []
    for machine in machines:
        if not isinstance(machine, dict):
            problems.append("The quantum-machine cleanup returned an unreadable entry.")
            continue
        qm_id = str(machine.get("qm_id") or "?")
        verdict = machine.get("verdict")
        if verdict == "unrelated":
            continue
        if verdict == "ours":
            if machine.get("closed") is not True:
                detail = f" ({machine['error']})" if machine.get("error") else ""
                problems.append(
                    f"This chip still has an open quantum machine {qm_id}{detail}; "
                    "close it first."
                )
        elif verdict == "overlapping":
            problems.append(
                f"Quantum machine {qm_id} shares some of this chip's ports and was not "
                "closed (it may belong to another user); resolve it on the QOP first."
            )
        else:
            problems.append(
                f"Quantum machine {qm_id} could not be read "
                f"({machine.get('error') or verdict}); its ports cannot be checked."
            )
    return problems


def qm_cleanup_summary(report: Any) -> dict[str, list[str]]:
    """Machine ids by class; `unrelated_ignored` are other users' machines."""

    summary: dict[str, list[str]] = {
        "closed": [],
        "ours_open": [],
        "overlapping": [],
        "unreadable": [],
        "unrelated_ignored": [],
    }
    machines = report.get("machines") if isinstance(report, dict) else None
    for machine in machines if isinstance(machines, list) else []:
        if not isinstance(machine, dict):
            continue
        qm_id = str(machine.get("qm_id") or "?")
        verdict = machine.get("verdict")
        if verdict == "unrelated":
            summary["unrelated_ignored"].append(qm_id)
        elif verdict == "ours":
            key = "closed" if machine.get("closed") is True else "ours_open"
            summary[key].append(qm_id)
        elif verdict == "overlapping":
            summary["overlapping"].append(qm_id)
        else:
            summary["unreadable"].append(qm_id)
    return summary


class HardwareLockRecovery:
    """Fail-closed recovery for a deliberately retained hardware lock.

    This API is intentionally exposed only by the local interactive CLI.  It is
    not an MCP or Dashboard operation because clearing the last safety barrier
    requires an operator who can inspect the connected hardware.
    """

    TERMINAL_RUN_STATUSES = {
        "failed",
        "stopped",
        "cancelled_by_shutdown",
        "force_stopped",
    }

    #: A recorded stale-QM cleanup proves the QOP state only briefly.
    QM_RECEIPT_MAX_AGE_SECONDS = 600

    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.db = database

    # -- Stale-QM evidence for a forced stop (operator decision 2026-10-20) --

    def qm_receipt_path(self, run_id: str) -> Path:
        return self.settings.runtime / "recovery" / f"{run_id}-qm-cleanup.json"

    def record_qm_cleanup(
        self,
        run_id: str,
        report: Any,
        *,
        state_path: Path,
        wiring_path: Path,
        actor: str,
        dry_run: bool,
    ) -> dict[str, Any]:
        """Persist a stale-QM cleanup report as recovery evidence for a run."""

        receipt = {
            "run_id": run_id,
            "checked_at": utc_now(),
            "actor": actor,
            "state_path": str(state_path),
            "wiring_path": str(wiring_path),
            "dry_run": bool(dry_run),
            "report": report,
            "summary": qm_cleanup_summary(report),
            "problems": qm_cleanup_problems(report),
        }
        atomic_write_json(self.qm_receipt_path(run_id), receipt)
        run = self.db.one("SELECT workflow_id FROM runs WHERE id = ?", (run_id,))
        self.db.event(
            "hardware_lock_qm_cleanup_recorded",
            actor,
            {
                "run_id": run_id,
                "summary": receipt["summary"],
                "problems": receipt["problems"],
                "dry_run": bool(dry_run),
            },
            run.get("workflow_id") if run else None,
        )
        return receipt

    def read_qm_receipt(self, run_id: str) -> dict[str, Any] | None:
        try:
            receipt = json.loads(
                self.qm_receipt_path(run_id).read_text(encoding="utf-8-sig")
            )
        except (OSError, ValueError):
            return None
        return receipt if isinstance(receipt, dict) else None

    def _qm_evidence_failures(
        self,
        run_id: str,
        run: dict[str, Any],
        state_path: Path,
        wiring_path: Path,
        checks: dict[str, Any],
    ) -> list[str]:
        receipt = self.read_qm_receipt(run_id)
        checks["qm_cleanup_receipt_path"] = str(self.qm_receipt_path(run_id))
        if receipt is None:
            checks["qm_cleanup_clean"] = False
            return [
                "No quantum-machine cleanup check is recorded for this forced-stop run; "
                "use \"Close this chip's stale quantum machine\" first."
            ]
        checks["qm_cleanup_checked_at"] = receipt.get("checked_at")
        checks["qm_cleanup_summary"] = qm_cleanup_summary(receipt.get("report"))
        failures: list[str] = []
        if receipt.get("run_id") != run_id:
            failures.append("The quantum-machine cleanup receipt is for another run.")
        if not (
            self._same_path(receipt.get("state_path"), state_path)
            and self._same_path(receipt.get("wiring_path"), wiring_path)
        ):
            failures.append(
                "The quantum-machine cleanup checked a different state.json/wiring.json "
                "than this run used."
            )
        checked_at = self._parse_utc(receipt.get("checked_at"))
        age = (
            None
            if checked_at is None
            else (datetime.now(timezone.utc) - checked_at).total_seconds()
        )
        checks["qm_cleanup_age_seconds"] = None if age is None else round(age)
        if age is None or age < -60 or age > self.QM_RECEIPT_MAX_AGE_SECONDS:
            failures.append(
                "The quantum-machine cleanup check is older than "
                f"{self.QM_RECEIPT_MAX_AGE_SECONDS // 60} minutes; run it again."
            )
        finished_at = self._parse_utc(run.get("finished_at"))
        if checked_at is not None and finished_at is not None and checked_at < finished_at:
            failures.append(
                "The quantum-machine cleanup check predates the worker termination; "
                "run it again."
            )
        failures.extend(qm_cleanup_problems(receipt.get("report")))
        checks["qm_cleanup_clean"] = not failures
        return failures

    @staticmethod
    def _parse_utc(value: Any) -> datetime | None:
        try:
            parsed = parse_iso_datetime(str(value or ""))
        except (TypeError, ValueError):
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)

    def inspect(self, run_id: str) -> dict[str, Any]:
        failures: list[str] = []
        checks: dict[str, Any] = {}
        lock = self._read_json(self.settings.lock_path, "hardware lock", failures)
        checks["lock_owner_matches"] = bool(
            isinstance(lock, dict) and lock.get("run_id") == run_id
        )
        if not checks["lock_owner_matches"]:
            failures.append("The hardware lock owner does not match the requested run.")

        run = self.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
        checks["run_exists"] = run is not None
        if run is None:
            failures.append("The retained-lock run is missing from the audit database.")
            return self._inspection_result(run_id, checks, failures)

        checks["run_status"] = run.get("status")
        checks["run_is_terminal"] = run.get("status") in self.TERMINAL_RUN_STATUSES
        if not checks["run_is_terminal"]:
            failures.append("The retained-lock run is not in a terminal failure state.")

        active_runs = self.db.all(
            "SELECT id FROM runs WHERE status IN ('starting', 'running', 'stopping')"
        )
        checks["active_run_count"] = len(active_runs)
        if active_runs:
            failures.append("At least one database run is still active.")

        pid = run.get("pid")
        pid_alive = isinstance(pid, int) and ExperimentRunner.process_is_alive(pid)
        checks["recorded_worker_pid"] = pid
        checks["recorded_worker_alive"] = pid_alive
        if pid_alive:
            failures.append("The recorded worker PID is still alive.")

        worker_scan_ok, worker_pids = self._active_worker_pids()
        checks["worker_process_scan_ok"] = worker_scan_ok
        checks["active_worker_pids"] = worker_pids
        if not worker_scan_ok:
            failures.append("Unable to verify that no JY worker process is active.")
        elif worker_pids:
            failures.append("At least one JY worker process is still active.")

        request_path = self.settings.runtime / "requests" / f"{run_id}.json"
        request = self._read_json(request_path, "run request", failures)
        request_active_state = (
            Path(str(request.get("active_state")))
            if isinstance(request, dict) and request.get("active_state")
            else self.settings.active_state
        )
        request_wiring = (
            Path(str(request.get("wiring_path")))
            if isinstance(request, dict) and request.get("wiring_path")
            else Path()
        )
        request_matches = isinstance(request, dict) and all(
            (
                request.get("run_id") == run_id,
                request.get("workflow_id") == run.get("workflow_id"),
                request.get("process_token") == run.get("process_token"),
                self._same_path(request.get("lock_path"), self.settings.lock_path),
                self._same_path(
                    request.get("database_path"), self.settings.database_path
                ),
                request_active_state.is_absolute(),
                request_active_state.name.casefold() == "state.json",
                request_active_state.is_file(),
                request_wiring.is_absolute(),
                request_wiring.name.casefold() == "wiring.json",
                request_wiring.is_file(),
                request_wiring.parent.resolve() == request_active_state.parent.resolve(),
                self._same_path(
                    request.get("recovery_state_path"),
                    self.settings.runtime / "recovery" / f"{run_id}-state.json",
                ),
            )
        )
        checks["request_identity_matches"] = bool(request_matches)
        if not request_matches:
            failures.append("The recorded request identity or protected paths do not match.")

        expected_hash = str(run.get("active_state_hash_before") or "")
        database_after_hash = str(run.get("active_state_hash_after") or "")
        active_hash = self._file_hash(request_active_state, "run active state", failures)
        configured_active_hash = self._file_hash(
            self.settings.active_state, "currently configured active state", failures
        )
        recovery_path = self.settings.runtime / "recovery" / f"{run_id}-state.json"
        recovery_hash = self._file_hash(recovery_path, "recovery state", failures)
        checks.update(
            {
                "expected_state_sha256": expected_hash or None,
                "active_state_sha256": active_hash,
                "recovery_state_sha256": recovery_hash,
                "database_after_sha256": database_after_hash or None,
                "configured_active_state_sha256": configured_active_hash,
                "configured_state_differs_from_run": (
                    self.settings.active_state.resolve() != request_active_state.resolve()
                ),
            }
        )
        checkpoint_hashes_match = bool(
            expected_hash
            and active_hash == expected_hash
            and recovery_hash == expected_hash
            and database_after_hash == expected_hash
        )
        abrupt_legacy_exit = bool(
            not expected_hash
            and not database_after_hash
            and "exited without recording completion"
            in str(run.get("error") or "")
        )
        legacy_backup_hashes_match = bool(
            abrupt_legacy_exit
            and active_hash
            and active_hash == recovery_hash
            and configured_active_hash == active_hash
            and not checks["configured_state_differs_from_run"]
        )
        # Operator decision 2026-10-20: a worker terminated after the
        # emergency-stop grace never writes its post-run hash.  Its pre-run
        # hash, the untouched active state, the protected backup and the
        # configured state together still prove the state is unchanged (a
        # post-run hash JY records itself after a verified termination equals
        # the pre-run hash and is accepted too).  Every forced-stop run also
        # needs a fresh, clean stale-QM cleanup receipt (below).
        forced_stop = bool(
            run.get("status") == "force_stopped"
            or run.get("termination_cause") == "forced_after_grace"
        )
        checks["forced_stop"] = forced_stop
        forced_stop_hashes_match = bool(
            forced_stop
            and expected_hash
            and database_after_hash in ("", expected_hash)
            and active_hash == expected_hash
            and recovery_hash == expected_hash
            and configured_active_hash == expected_hash
            and not checks["configured_state_differs_from_run"]
        )
        hashes_match = (
            checkpoint_hashes_match
            or legacy_backup_hashes_match
            or forced_stop_hashes_match
        )
        checks["state_hashes_match"] = hashes_match
        if forced_stop_hashes_match:
            checks["state_evidence_mode"] = "forced_stop_checkpoint"
        elif checkpoint_hashes_match:
            checks["state_evidence_mode"] = "worker_checkpoint"
        elif legacy_backup_hashes_match:
            checks["state_evidence_mode"] = "legacy_recovery_backup_match"
        else:
            checks["state_evidence_mode"] = "insufficient"
        if not hashes_match:
            if forced_stop:
                failures.append(
                    self._forced_stop_hash_failure(
                        expected_hash=expected_hash,
                        database_after_hash=database_after_hash,
                        active_hash=active_hash,
                        recovery_hash=recovery_hash,
                        configured_active_hash=configured_active_hash,
                        configured_differs=bool(
                            checks["configured_state_differs_from_run"]
                        ),
                    )
                )
            else:
                failures.append(
                    "State recovery evidence is insufficient: require either the worker's "
                    "pre-run/database hashes, or for a legacy abrupt exit an identical active "
                    "state, configured state, and protected recovery backup."
                )
        if forced_stop:
            failures.extend(
                self._qm_evidence_failures(
                    run_id, run, request_active_state, request_wiring, checks
                )
            )

        analysis = self._decode_json(run.get("analysis_json"))
        restore_error = analysis.get("active_state_restore_error")
        checks["active_state_restore_error"] = restore_error
        if restore_error:
            failures.append("The failed worker recorded an active-state restore error.")

        return self._inspection_result(run_id, checks, failures)

    def recover(
        self,
        run_id: str,
        *,
        actor: str,
        operator_confirmation: str,
    ) -> dict[str, Any]:
        expected = f"RECOVER HARDWARE LOCK {run_id}"
        if operator_confirmation != expected:
            raise HardwareLockRecoveryError("Operator confirmation did not match.")
        if not actor.strip():
            raise HardwareLockRecoveryError("An operator identity is required.")
        return self._release(
            run_id,
            actor=actor.strip(),
            attestation=(
                "Connected hardware was inspected locally and has no active job/output."
            ),
        )

    def release_after_forced_stop(self, run_id: str, *, actor: str) -> dict[str, Any]:
        """Automatic release right after a verified emergency-stop termination.

        Same checks, archive and receipt as the operator recovery, restricted
        to `forced_stop_checkpoint` evidence (which includes a fresh, clean
        stale-QM cleanup receipt).
        """

        return self._release(
            run_id,
            actor=actor,
            attestation=(
                "Automatic release after a forced stop: the worker was terminated, "
                "this chip's quantum machine was closed and verified absent, and the "
                "state hashes are unchanged."
            ),
            required_mode="forced_stop_checkpoint",
        )

    def _release(
        self,
        run_id: str,
        *,
        actor: str,
        attestation: str,
        required_mode: str | None = None,
    ) -> dict[str, Any]:
        transition_lock = self.settings.runtime / "hardware-transition.lock"
        with exclusive_file_lock(transition_lock, "operator hardware-lock recovery"):
            inspection = self.inspect(run_id)
            if not inspection["ready"]:
                raise HardwareLockRecoveryError(
                    "Hardware-lock recovery checks failed: "
                    + "; ".join(inspection["failures"])
                )
            if (
                required_mode is not None
                and inspection["checks"].get("state_evidence_mode") != required_mode
            ):
                raise HardwareLockRecoveryError(
                    f"Automatic release requires {required_mode} evidence."
                )
            actor = actor.strip()
            recovered_at = utc_now()
            recovery_dir = self.settings.runtime / "recovery"
            archived_lock = recovery_dir / f"{run_id}-hardware.lock.recovered.json"
            receipt_path = recovery_dir / f"{run_id}-hardware-lock-recovery.json"
            if archived_lock.exists() or receipt_path.exists():
                raise HardwareLockRecoveryError(
                    "A recovery archive or receipt already exists; refusing to overwrite it."
                )
            receipt = {
                "run_id": run_id,
                "workflow_id": inspection["workflow_id"],
                "operator": actor,
                "recovered_at": recovered_at,
                "operator_attestation": attestation,
                "checks": inspection["checks"],
            }
            atomic_write_json(receipt_path, receipt)
            self.db.event(
                "hardware_lock_recovery_authorized",
                actor,
                {
                    "run_id": run_id,
                    "receipt_path": str(receipt_path),
                    "state_sha256": inspection["checks"]["active_state_sha256"],
                    "state_evidence_mode": inspection["checks"].get(
                        "state_evidence_mode"
                    ),
                },
                inspection["workflow_id"],
            )
            try:
                self.settings.lock_path.rename(archived_lock)
            except Exception:
                receipt_path.unlink(missing_ok=True)
                raise
            (self.settings.runtime / "recovery-required.json").unlink(missing_ok=True)
            self.db.event(
                "hardware_lock_recovered",
                actor,
                {
                    "run_id": run_id,
                    "archived_lock_path": str(archived_lock),
                    "receipt_path": str(receipt_path),
                },
                inspection["workflow_id"],
            )
            return {
                "run_id": run_id,
                "workflow_id": inspection["workflow_id"],
                "status": "recovered",
                "hardware_lock_present": self.settings.lock_path.exists(),
                "archived_lock_path": str(archived_lock),
                "receipt_path": str(receipt_path),
            }

    @staticmethod
    def _forced_stop_hash_failure(
        *,
        expected_hash: str,
        database_after_hash: str,
        active_hash: str | None,
        recovery_hash: str | None,
        configured_active_hash: str | None,
        configured_differs: bool,
    ) -> str:
        prefix = "Forced-stop state evidence is insufficient: "
        if not expected_hash:
            return prefix + "the run recorded no pre-run state hash."
        problems: list[str] = []
        if database_after_hash and database_after_hash != expected_hash:
            problems.append("the recorded post-run hash differs from the pre-run hash")
        if active_hash != expected_hash:
            problems.append("the run's state.json changed since the run started")
        if recovery_hash != expected_hash:
            problems.append("the protected recovery backup differs from the pre-run state")
        if configured_differs:
            problems.append("the configured state.json path is not the run's state path")
        elif configured_active_hash != expected_hash:
            problems.append("the configured state.json differs from the pre-run state")
        return prefix + "; ".join(problems or ["the hashes do not match"]) + "."

    def _inspection_result(
        self,
        run_id: str,
        checks: dict[str, Any],
        failures: list[str],
    ) -> dict[str, Any]:
        run = self.db.one("SELECT workflow_id FROM runs WHERE id = ?", (run_id,))
        return {
            "run_id": run_id,
            "workflow_id": run.get("workflow_id") if run else None,
            "ready": not failures,
            "checks": checks,
            "failures": failures,
        }

    @staticmethod
    def _decode_json(value: Any) -> dict[str, Any]:
        try:
            decoded = json.loads(str(value or "{}"))
        except (TypeError, ValueError):
            return {}
        return decoded if isinstance(decoded, dict) else {}

    @staticmethod
    def _read_json(path: Path, label: str, failures: list[str]) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            failures.append(f"Unable to read {label}: {type(exc).__name__}: {exc}")
            return None

    @staticmethod
    def _file_hash(path: Path, label: str, failures: list[str]) -> str | None:
        try:
            return sha256_file(path)
        except OSError as exc:
            failures.append(f"Unable to hash {label}: {type(exc).__name__}: {exc}")
            return None

    @staticmethod
    def _same_path(value: Any, expected: Path) -> bool:
        try:
            candidate = Path(str(value))
            return candidate.is_absolute() and candidate.resolve() == expected.resolve()
        except (OSError, RuntimeError, ValueError):
            return False

    @staticmethod
    def _active_worker_pids() -> tuple[bool, list[int]]:
        if os.name == "nt":
            command = (
                "$items = Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -like 'python*' -and "
                "$_.CommandLine -match '(?i)-m\\s+jy_agent\\.worker(?:\\s|$)' } | "
                "Select-Object -ExpandProperty ProcessId; "
                "@($items) | ConvertTo-Json -Compress"
            )
            try:
                completed = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-Command", command],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if completed.returncode != 0:
                    return False, []
                decoded = json.loads(completed.stdout.strip() or "[]")
                values = decoded if isinstance(decoded, list) else [decoded]
                return True, sorted(int(value) for value in values if value is not None)
            except (OSError, subprocess.SubprocessError, TypeError, ValueError):
                return False, []

        try:
            worker_pids: list[int] = []
            for item in Path("/proc").iterdir():
                if not item.name.isdigit():
                    continue
                try:
                    command_line = (item / "cmdline").read_bytes().replace(b"\x00", b" ")
                except OSError:
                    continue
                if b"-m jy_agent.worker" in command_line:
                    worker_pids.append(int(item.name))
            return True, sorted(worker_pids)
        except OSError:
            return False, []
