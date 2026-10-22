from __future__ import annotations

import _thread
import argparse
import hashlib
import hmac
import json
import os
import signal
import shutil
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from .db import Database
from .failures import FailureClassification, classify_failure
from .instrument_probe import ProbeResult, probe_qop
from .runner import release_worker_lock
from .util import atomic_write_json, json_compatible, sha256_file, utc_now


def _inspect_node(script_path: Path) -> Any:
    from qualibrate import QualibrationNode
    from qualibrate.models.run_mode import RunModes
    from qualibrate.q_runnnable import run_modes_ctx

    nodes: dict[str, Any] = {}
    token = run_modes_ctx.set(RunModes(inspection=True))
    try:
        QualibrationNode.scan_node_file(script_path, nodes)
    finally:
        run_modes_ctx.reset(token)
    if not nodes:
        raise RuntimeError(f"No QualibrationNode found in {script_path}")
    return next(iter(nodes.values()))


def _assert_active_state_source(active_state: Path) -> None:
    from quam_libs.components.quam_root import QuAM

    quam_state_root = QuAM.get_quam_state_path()
    if quam_state_root is None:
        raise RuntimeError(
            "ACTIVE_STATE_SOURCE_MISSING: ~/.qualibrate/config.toml does not "
            "define quam.state_path"
        )
    expected_state = (Path(quam_state_root).expanduser() / "state.json").resolve()
    actual_state = active_state.expanduser().resolve()
    if actual_state != expected_state:
        raise RuntimeError(
            "ACTIVE_STATE_SOURCE_MISMATCH: JY active state "
            f"{actual_state} does not match the QuAM.load() source {expected_state}"
        )


def run_request(request_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    database = Database(Path(request["database_path"]))
    run_id = request["run_id"]
    active_state = Path(request["active_state"])
    wiring_path = Path(request["wiring_path"])
    lock_path = Path(request["lock_path"])
    recovery_path = Path(request["recovery_state_path"])
    stop_request_path = Path(
        request.get("stop_request_path")
        or request_path.with_name(f"{run_id}.stop.json")
    )
    exit_receipt_path = Path(
        request.get("exit_receipt_path")
        or request_path.with_name(f"{run_id}.exit.json")
    )
    for label, path in (
        ("active state", active_state),
        ("wiring", wiring_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Configured {label} file is missing: {path}")
    _assert_active_state_source(active_state)
    before_hash = sha256_file(active_state)
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(active_state, recovery_path)
    _persist_recovery_checkpoint(
        database,
        run_id=run_id,
        workflow_id=request["workflow_id"],
        state_hash=before_hash,
        recovery_path=recovery_path,
    )
    release_lock = False
    worker_done = threading.Event()
    stop_requested = threading.Event()
    stop_details: dict[str, Any] = {}
    exit_code = 1
    final_status = "failed"
    termination_cause = "worker_exception"
    cleanup_verified = False
    execution_phase = "initializing"

    def watch_stop_request() -> None:
        while not worker_done.wait(0.2):
            try:
                payload = json.loads(stop_request_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("run_id") != run_id or not hmac.compare_digest(
                str(payload.get("process_token") or ""),
                str(request.get("process_token") or ""),
            ):
                continue
            stop_details.update(payload)
            stop_requested.set()
            _thread.interrupt_main()
            return

    def handle_interrupt(signum: int, frame: Any) -> None:
        try:
            from qualibrate import QualibrationNode

            if QualibrationNode.active_node is not None:
                QualibrationNode.active_node.stop()
        finally:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_interrupt)
    if os.name == "nt" and hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handle_interrupt)

    threading.Thread(
        target=watch_stop_request,
        name=f"jy-stop-watch-{run_id[:8]}",
        daemon=True,
    ).start()

    try:
        from qualibrate.storage.local_storage_manager import LocalStorageManager

        inspected = _inspect_node(Path(request["script_path"]))
        actual_node_name = str(getattr(inspected, "name", ""))
        if actual_node_name != str(request["node_name"]):
            raise RuntimeError(
                "POLICY_NODE_MISMATCH: inspected QualibrationNode name "
                f"{actual_node_name!r} does not match {request['node_name']!r}"
            )
        parameters = {**inspected.parameters.model_dump(), **request["parameters"]}
        runnable = inspected.copy(**parameters)
        storage = LocalStorageManager(
            root_data_folder=Path(request["data_root"]),
            active_machine_path=None,
        )
        runnable.storage_manager = storage
        execution_phase = "hardware_execution"
        runnable.run(interactive=True)
        if stop_requested.is_set():
            raise KeyboardInterrupt
        state_updates_path = Path(request["state_updates_path"])
        atomic_write_json(
            state_updates_path,
            json_compatible(dict(runnable.state_updates)),
        )
        snapshot_path = (
            str(storage.data_handler.path)
            if storage.data_handler.path is not None
            else None
        )
        after_hash = sha256_file(active_state)
        if after_hash != before_hash:
            raise RuntimeError(
                "ACTIVE_STATE_GUARD_VIOLATION: the calibration node changed active state"
            )
        database.execute(
            """
            UPDATE runs
            SET status = 'completed', finished_at = ?, snapshot_id = ?,
                snapshot_path = ?, active_state_hash_before = ?,
                active_state_hash_after = ?
            WHERE id = ?
            """,
            (
                utc_now(),
                runnable.snapshot_idx,
                snapshot_path,
                before_hash,
                after_hash,
                run_id,
            ),
        )
        database.event(
            "run_completed",
            "worker",
            {"run_id": run_id, "snapshot_id": runnable.snapshot_idx},
            request["workflow_id"],
        )
        release_lock = True
        cleanup_verified = True
        exit_code = 0
        final_status = "completed"
        termination_cause = "normal_exit"
        recovery_path.unlink(missing_ok=True)
        return 0
    except KeyboardInterrupt:
        observed_hash, restored, restore_error = _restore_state_if_changed(
            active_state, recovery_path, before_hash
        )
        final_hash = sha256_file(active_state) if active_state.is_file() else observed_hash
        intent = str(stop_details.get("intent") or "interrupt")
        final_status = (
            "cancelled_by_shutdown" if intent == "full_shutdown" else "stopped"
        )
        termination_cause = (
            "cooperative_full_shutdown"
            if intent == "full_shutdown"
            else "cooperative_stop"
        )
        exit_code = 130
        database.execute(
            """
            UPDATE runs SET status = ?, finished_at = ?,
                active_state_hash_before = ?, active_state_hash_after = ?,
                termination_cause = ?
            WHERE id = ?
            """,
            (
                final_status,
                utc_now(),
                before_hash,
                final_hash,
                termination_cause,
                run_id,
            ),
        )
        cleanup_verified = final_hash == before_hash and not restore_error
        if cleanup_verified:
            release_lock = True
            recovery_path.unlink(missing_ok=True)
        elif restore_error:
            database.event(
                "active_state_restore_failed",
                "worker",
                {"run_id": run_id, "error": restore_error},
                request["workflow_id"],
            )
        database.event(
            "run_cancelled_by_shutdown"
            if final_status == "cancelled_by_shutdown"
            else "run_stopped",
            "worker",
            {
                "run_id": run_id,
                "intent": intent,
                "reason": stop_details.get("reason"),
                "active_state_restored": restored,
                "cleanup_verified": cleanup_verified,
            },
            request["workflow_id"],
        )
        return 130
    except Exception as exc:
        message = "".join(traceback.format_exception(exc))
        classification = classify_failure(exc)
        attempted_targets = _attempted_targets(request)
        classification, probe, submission_timeout, submission_escalated = (
            _resolve_submission_timeout(
                classification, request, attempted_targets
            )
        )
        instrument_unreachable = (
            classification.category == "instrument_unreachable"
        )
        qop_compile_failure = classification.category == "qop_compile_failure"
        analysis_failure = classification.category == "analysis_failure"
        # A submission timeout pauses only when a single target already failed
        # to be accepted; with more than one target the group is halved and the
        # lease keeps running.
        measurement_paused = instrument_unreachable or submission_escalated
        hardware_cleanup_verified = bool(
            (
                instrument_unreachable
                or qop_compile_failure
            )
            and classification.safe_to_release_lock
        )
        hardware_cleanup_error: str | None = None
        if (
            (instrument_unreachable or submission_timeout or analysis_failure)
            and not hardware_cleanup_verified
            and execution_phase == "hardware_execution"
        ):
            hardware_cleanup_verified, hardware_cleanup_error = (
                _stop_active_node_with_evidence()
            )
        observed_hash, restored, restore_error = _restore_state_if_changed(
            active_state, recovery_path, before_hash
        )
        after_hash = sha256_file(active_state) if active_state.is_file() else observed_hash
        cleanup_verified = after_hash == before_hash and not restore_error
        # Operator instruction: an instrument-connectivity failure must never
        # engage the hardware lock, so it releases on an unchanged state file
        # even when cleanup could not be confirmed -- the instrument being
        # unreachable is exactly why the evidence is missing.  Every other
        # releasable category still requires verified hardware cleanup.
        # A program that was never accepted never owned the hardware, so a
        # submission timeout releases on the same unchanged-state evidence.
        release_after_verified_cleanup = bool(
            cleanup_verified
            and (
                instrument_unreachable
                or submission_timeout
                # Operator instruction 2026-10-19: a fitting crash comes after
                # the data was returned, so it never retains the lock.
                or analysis_failure
                or (
                    hardware_cleanup_verified
                    and qop_compile_failure
                    and classification.safe_to_release_lock
                )
            )
        )
        if release_after_verified_cleanup:
            release_lock = True
            recovery_path.unlink(missing_ok=True)
        if instrument_unreachable:
            termination_cause = "instrument_unreachable"
        elif submission_timeout:
            termination_cause = "program_submission_timeout"
        else:
            termination_cause = "worker_exception"
        failure = {
            "analysis_status": "failed",
            "node_id": request["node_id"],
            "snapshot_id": None,
            "snapshot_path": None,
            "plots": [],
            "failure_reasons": [
                "Execution failed before a usable snapshot was produced."
            ],
            "warnings": [],
            "candidate_state_patch": [],
            "error": message,
            "active_state_hash": after_hash,
            "active_state_observed_hash": observed_hash,
            "active_state_restored": restored,
            "active_state_restore_error": restore_error,
            "failure_category": classification.category,
            "execution_phase": execution_phase,
            "measurement_paused": measurement_paused,
            "attempted_target_count": len(attempted_targets),
            "instrument_probe": probe.as_dict() if probe is not None else None,
            "hardware_cleanup_verified": hardware_cleanup_verified,
            "hardware_cleanup_error": hardware_cleanup_error,
            "hardware_lock_recovery_required": (
                instrument_unreachable and not release_after_verified_cleanup
            ),
            "operator_message": _failure_operator_message(
                instrument_unreachable=instrument_unreachable,
                submission_escalated=submission_escalated,
                probe=probe,
            ),
        }
        database.execute(
            """
            UPDATE runs SET status = 'failed', finished_at = ?, error = ?,
                active_state_hash_before = ?, active_state_hash_after = ?,
                analysis_status = 'failed', analysis_json = ?,
                termination_cause = ?
            WHERE id = ?
            """,
            (
                utc_now(), message, before_hash, after_hash,
                json.dumps(failure, ensure_ascii=False), termination_cause, run_id,
            ),
        )
        if measurement_paused:
            _pause_for_instrument_failure(
                database,
                run_id=run_id,
                workflow_id=request["workflow_id"],
                lock_retained=not release_after_verified_cleanup,
            )
        database.event(
            "instrument_error_paused" if measurement_paused else "run_failed",
            "worker",
            {
                "run_id": run_id,
                "error": str(exc),
                "active_state_restored": restored,
                "failure_category": classification.category,
                "attempted_target_count": len(attempted_targets),
                "instrument_probe": probe.as_dict() if probe is not None else None,
                "hardware_cleanup_verified": hardware_cleanup_verified,
                "hardware_lock_retained": not release_lock,
            },
            request["workflow_id"],
        )
        exit_code = 1
        final_status = "failed"
        return 1
    finally:
        worker_done.set()
        receipt = {
            "run_id": run_id,
            "status": final_status,
            "exit_code": exit_code,
            "termination_cause": termination_cause,
            "stop_intent": stop_details.get("intent"),
            "stop_requested_at": stop_details.get("requested_at"),
            "cleanup_verified": cleanup_verified,
            "hardware_cleanup_verified": locals().get(
                "hardware_cleanup_verified", cleanup_verified
            ),
            "hardware_cleanup_error": locals().get("hardware_cleanup_error"),
            "lock_release_authorized": release_lock,
            "lock_released": False,
            "finished_at": utc_now(),
        }
        _write_signed_exit_receipt(
            exit_receipt_path,
            receipt,
            str(request.get("process_token") or ""),
        )
        if release_lock:
            release_worker_lock(lock_path, run_id)
        receipt["lock_released"] = not lock_path.exists()
        receipt["finished_at"] = utc_now()
        _write_signed_exit_receipt(
            exit_receipt_path,
            receipt,
            str(request.get("process_token") or ""),
        )


def _attempted_targets(request: dict[str, Any]) -> list[str]:
    parameters = request.get("parameters")
    targets = parameters.get("qubits") if isinstance(parameters, dict) else None
    if not isinstance(targets, list):
        return []
    return [str(name) for name in targets]


def _resolve_submission_timeout(
    classification: FailureClassification,
    request: dict[str, Any],
    attempted_targets: list[str],
) -> tuple[FailureClassification, ProbeResult | None, bool, bool]:
    """Confirm a submission timeout against the live instrument.

    The traceback proves the program was being submitted rather than the
    connection being opened, but it cannot tell a program that is too large
    from a QOP that is wedged.  Ask the instrument.  If it answers, the
    program was too large and the lease keeps running so the caller can halve
    the multiplex group; a single target that still times out is out of
    halving room and goes to the operator.  If it does not answer, this was an
    outage after all and it is reclassified.

    Returns the classification to use, the probe result, whether this is a
    confirmed submission timeout, and whether it must pause anyway.
    """

    if classification.category != "program_submission_timeout":
        return classification, None, False, False

    # This runs inside the failure handler, so a probe that raises would lose
    # the original error. Any surprise means reachability is unconfirmed, which
    # is the conservative verdict: pause instead of halving.
    try:
        probe = probe_qop(Path(str(request.get("wiring_path") or "")))
    except Exception as probe_error:  # pragma: no cover - defensive
        probe = ProbeResult(
            reachable=False,
            stage="probe",
            cause="unknown",
            detail=f"Reachability probe failed: {probe_error}",
        )
    if not probe.reachable:
        return (
            FailureClassification(
                "instrument_unreachable",
                safe_to_release_lock=probe.stage in {"dns", "tcp"},
            ),
            probe,
            False,
            False,
        )
    return classification, probe, True, len(attempted_targets) <= 1


def _failure_operator_message(
    *,
    instrument_unreachable: bool,
    submission_escalated: bool,
    probe: ProbeResult | None,
) -> dict[str, str] | None:
    if submission_escalated:
        detail = probe.detail if probe is not None else ""
        return {
            "zh-Hant": (
                "單一 qubit 的程式仍然無法送進 QOP，已無法再拆分，本次實驗與"
                "後續排程已暫停。儀器有回應健康檢查，所以這不是網路問題，"
                "請檢查 QOP 的工作佇列或重啟 QOP；確認後回 AI 對話輸入"
                "「恢復量測」。若已開啟全自動判斷模式，JY 會把這顆 qubit 在"
                "此節點標為 incomplete 並自動繼續，不需輸入。"
            ),
            "en": (
                "A single-qubit program still could not be submitted to the "
                "QOP and there is no group left to halve, so this experiment "
                "and new scheduling were paused. The instrument answered a "
                "health check, so this is not a network fault: check the QOP "
                "job queue or restart the QOP, then enter 'Resume measurement' "
                "in the AI conversation. With full-auto judgement on, JY marks "
                "this qubit incomplete at this node and resumes by itself. "
                f"{detail}"
            ).strip(),
        }
    if not instrument_unreachable:
        return None
    detail = probe.detail if probe is not None else ""
    if probe is not None and probe.cause == "network":
        return {
            "zh-Hant": (
                "無法連線到 QOP（網路層級），本次實驗與後續排程已暫停。"
                "請檢查實驗室網路、線路與主機位址；確認後回 AI 對話輸入"
                "「恢復量測」。"
            ),
            "en": (
                "The QOP could not be reached at the network level, so this "
                "experiment and new scheduling were paused. Check the lab "
                "network, cabling, and host address, then enter 'Resume "
                f"measurement' in the AI conversation. {detail}"
            ).strip(),
        }
    if probe is not None and probe.cause == "instrument":
        return {
            "zh-Hant": (
                "網路可達但 QOP 沒有回應，本次實驗與後續排程已暫停。"
                "請檢查 QOP/OPX 電源與服務狀態，必要時重啟；確認後回 AI "
                "對話輸入「恢復量測」。"
            ),
            "en": (
                "The network is reachable but the QOP did not respond, so "
                "this experiment and new scheduling were paused. Check QOP/OPX "
                "power and service state, restart if needed, then enter "
                f"'Resume measurement' in the AI conversation. {detail}"
            ).strip(),
        }
    return {
        "zh-Hant": (
            "儀器連線失敗，本次實驗與後續排程已暫停。請檢查 QOP/OPX、"
            "儀器電源與實驗室網路；確認後回 AI 對話輸入「恢復量測」。"
        ),
        "en": (
            "Instrument connectivity failed, so this experiment and new "
            "scheduling were paused. Check QOP/OPX, instrument power, and "
            "the lab network, then enter 'Resume measurement' in the AI conversation."
        ),
    }


def _pause_for_instrument_failure(
    database: Database,
    *,
    run_id: str,
    workflow_id: str,
    lock_retained: bool,
) -> None:
    """Pause new scheduling while preserving the current authorization scope."""

    now = utc_now()
    reason = (
        f"Instrument connectivity error paused run {run_id}; no new experiment "
        "will be scheduled until the operator resumes the measurement mode."
    )
    try:
        with database.transaction(immediate=True) as connection:
            run = connection.execute(
                "SELECT autonomy_lease_id FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            lease_id = run["autonomy_lease_id"] if run is not None else None
            if lease_id:
                connection.execute(
                    "UPDATE autonomy_leases SET status = 'paused', stopped_reason = ?, "
                    "updated_at = ? WHERE id = ? AND status = 'active'",
                    (reason, now, lease_id),
                )
            else:
                connection.execute(
                    "UPDATE workflows SET status = 'paused', updated_at = ? "
                    "WHERE id = ? AND status = 'active'",
                    (now, workflow_id),
                )
            connection.execute(
                "UPDATE measurement_sessions SET status = 'paused', updated_at = ? "
                "WHERE workflow_id = ? AND status IN ('active', 'halted')",
                (now, workflow_id),
            )
            database.event(
                "measurement_paused_for_instrument_error",
                "worker",
                {
                    "run_id": run_id,
                    "reason": reason,
                    "hardware_lock_retained": lock_retained,
                },
                workflow_id,
                connection=connection,
            )
    except Exception as exc:
        database.event(
            "instrument_pause_record_failed",
            "worker",
            {"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"},
            workflow_id,
        )


def _persist_recovery_checkpoint(
    database: Database,
    *,
    run_id: str,
    workflow_id: str,
    state_hash: str,
    recovery_path: Path,
) -> None:
    """Record pre-hardware recovery evidence before importing/running a node."""
    database.execute(
        "UPDATE runs SET active_state_hash_before = ? WHERE id = ?",
        (state_hash, run_id),
    )
    database.event(
        "run_recovery_checkpointed",
        "worker",
        {
            "run_id": run_id,
            "state_sha256": state_hash,
            "recovery_path": str(recovery_path),
        },
        workflow_id,
    )


def _stop_active_node_with_evidence() -> tuple[bool, str | None]:
    """Stop the recorded active node; absence is not accepted as cleanup proof."""

    try:
        from qualibrate import QualibrationNode

        active_node = QualibrationNode.active_node
        if active_node is None:
            return False, "QualibrationNode.active_node is unavailable"
        active_node.stop()
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _write_signed_exit_receipt(
    path: Path,
    payload: dict[str, Any],
    process_token: str,
) -> None:
    if not process_token:
        raise RuntimeError("Worker cannot sign an exit receipt without its process token")
    unsigned = dict(payload)
    unsigned.pop("worker_signature", None)
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signed = {
        **unsigned,
        "worker_signature": hmac.new(
            process_token.encode("utf-8"),
            canonical,
            hashlib.sha256,
        ).hexdigest(),
    }
    atomic_write_json(path, signed)


def _restore_state_if_changed(
    active_state: Path, recovery_path: Path, before_hash: str
) -> tuple[str | None, bool, str | None]:
    """Restore the exact pre-run state after any guarded worker failure."""
    try:
        observed_hash = sha256_file(active_state)
    except OSError:
        observed_hash = None
    if observed_hash == before_hash:
        return observed_hash, False, None
    temporary = active_state.with_name(f".{active_state.name}.{os.getpid()}.restore.tmp")
    try:
        shutil.copy2(recovery_path, temporary)
        os.replace(temporary, active_state)
        restored_hash = sha256_file(active_state)
        if restored_hash != before_hash:
            raise RuntimeError("restored state hash does not match pre-run state")
        return observed_hash, True, None
    except Exception as exc:
        return observed_hash, False, f"{type(exc).__name__}: {exc}"
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--process-token", required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if not hmac.compare_digest(
        str(args.process_token), str(request.get("process_token") or "")
    ):
        raise SystemExit("Worker process token does not match its request")
    raise SystemExit(run_request(args.request))


if __name__ == "__main__":
    main()
