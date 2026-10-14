from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jy_agent.config import Settings
from jy_agent.db import Database
from jy_agent.recovery import HardwareLockRecovery, HardwareLockRecoveryError
from jy_agent.util import atomic_write_json, sha256_file, utc_now


AGENT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "a" * 32
WORKFLOW_ID = "b" * 32
PROPOSAL_ID = "c" * 32
PROCESS_TOKEN = "d" * 32


def make_settings(root: Path) -> Settings:
    active_state = root / "state.json"
    atomic_write_json(active_state, {"qubits": {"q1": {}}})
    wiring_path = root / "wiring.json"
    atomic_write_json(wiring_path, {})
    config_path = root / "config.toml"
    config_path.write_text("# test\n", encoding="utf-8")
    data_root = root / "Data"
    data_root.mkdir()
    return Settings(
        agent_root=AGENT_ROOT,
        superconducting_root=AGENT_ROOT.parent,
        calibration_graph=AGENT_ROOT.parent / "calibration_graph",
        data_root=data_root,
        quam_state_root=root,
        active_state=active_state,
        wiring_path=wiring_path,
        qualibrate_config_path=config_path,
        qualibrate_project="unittest",
        runtime=root / "runtime",
        policy_path=AGENT_ROOT / "rules" / "policies.yaml",
        playbook_path=AGENT_ROOT / "rules" / "PLAYBOOK.md",
        host="127.0.0.1",
        port=8765,
        mcp_path="/mcp",
        qualibrate_python=Path(sys.executable),
        workflow_sequence=("02c",),
        require_explicit_qubits=True,
        measurement_mode_entry_phrase="進入 JY 量測模式",
        measurement_mode_exit_phrase="退出 JY 量測模式",
        fresh_qubit_defaults=False,
    )


def seed_retained_lock(settings: Settings, database: Database) -> None:
    now = utc_now()
    state_hash = sha256_file(settings.active_state)
    database.execute(
        "INSERT INTO workflows(id, created_at, updated_at, status, targets_json, "
        "initial_parameters_json, current_node, client_id) "
        "VALUES (?, ?, ?, 'active', '[\"q1\"]', '{}', '02c', 'test')",
        (WORKFLOW_ID, now, now),
    )
    database.execute(
        "INSERT INTO proposals(id, workflow_id, kind, payload_json, status, "
        "created_at, expires_at, max_uses, uses, source_client) "
        "VALUES (?, ?, 'run', '{}', 'approved', ?, ?, 1, 1, 'test')",
        (PROPOSAL_ID, WORKFLOW_ID, now, "2099-01-01T00:00:00+00:00"),
    )
    database.execute(
        "INSERT INTO runs(id, workflow_id, proposal_id, node_id, parameters_json, "
        "status, pid, started_at, finished_at, active_state_hash_before, "
        "active_state_hash_after, analysis_status, analysis_json, process_token) "
        "VALUES (?, ?, ?, '02c', '{}', 'failed', 99999999, ?, ?, ?, ?, "
        "'failed', ?, ?)",
        (
            RUN_ID,
            WORKFLOW_ID,
            PROPOSAL_ID,
            now,
            now,
            state_hash,
            state_hash,
            json.dumps({"active_state_restore_error": None}),
            PROCESS_TOKEN,
        ),
    )
    atomic_write_json(
        settings.lock_path,
        {"run_id": RUN_ID, "created_at": now},
    )
    recovery_path = settings.runtime / "recovery" / f"{RUN_ID}-state.json"
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_path.write_bytes(settings.active_state.read_bytes())
    atomic_write_json(
        settings.runtime / "requests" / f"{RUN_ID}.json",
        {
            "run_id": RUN_ID,
            "workflow_id": WORKFLOW_ID,
            "process_token": PROCESS_TOKEN,
            "database_path": str(settings.database_path),
            "lock_path": str(settings.lock_path),
            "active_state": str(settings.active_state),
            "wiring_path": str(settings.wiring_path),
            "recovery_state_path": str(recovery_path),
        },
    )


class HardwareLockRecoveryTests(unittest.TestCase):
    def test_matching_terminal_run_is_ready_for_operator_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder))
            database = Database(settings.database_path)
            seed_retained_lock(settings, database)
            recovery = HardwareLockRecovery(settings, database)

            with patch.object(recovery, "_active_worker_pids", return_value=(True, [])):
                result = recovery.inspect(RUN_ID)

            self.assertTrue(result["ready"])
            self.assertTrue(result["checks"]["state_hashes_match"])
            self.assertFalse(result["checks"]["recorded_worker_alive"])

    def test_state_hash_mismatch_refuses_recovery_and_retains_lock(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder))
            database = Database(settings.database_path)
            seed_retained_lock(settings, database)
            settings.active_state.write_text('{"changed": true}\n', encoding="utf-8")
            recovery = HardwareLockRecovery(settings, database)

            with patch.object(recovery, "_active_worker_pids", return_value=(True, [])):
                inspection = recovery.inspect(RUN_ID)
                with self.assertRaises(HardwareLockRecoveryError):
                    recovery.recover(
                        RUN_ID,
                        actor="operator",
                        operator_confirmation=f"RECOVER HARDWARE LOCK {RUN_ID}",
                    )

            self.assertFalse(inspection["ready"])
            self.assertTrue(settings.lock_path.exists())

    def test_legacy_abrupt_exit_uses_identical_protected_backup_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder))
            database = Database(settings.database_path)
            seed_retained_lock(settings, database)
            database.execute(
                "UPDATE runs SET active_state_hash_before = NULL, "
                "active_state_hash_after = NULL, error = ? WHERE id = ?",
                (
                    "Worker process 99999999 exited without recording completion "
                    f"for run {RUN_ID}; hardware lock retained for inspection.",
                    RUN_ID,
                ),
            )
            recovery = HardwareLockRecovery(settings, database)

            with patch.object(recovery, "_active_worker_pids", return_value=(True, [])):
                result = recovery.inspect(RUN_ID)

            self.assertTrue(result["ready"])
            self.assertTrue(result["checks"]["state_hashes_match"])
            self.assertEqual(
                result["checks"]["state_evidence_mode"],
                "legacy_recovery_backup_match",
            )

    def test_missing_checkpoint_does_not_relax_non_abrupt_failure(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder))
            database = Database(settings.database_path)
            seed_retained_lock(settings, database)
            database.execute(
                "UPDATE runs SET active_state_hash_before = NULL, "
                "active_state_hash_after = NULL, error = 'ordinary worker error' "
                "WHERE id = ?",
                (RUN_ID,),
            )
            recovery = HardwareLockRecovery(settings, database)

            with patch.object(recovery, "_active_worker_pids", return_value=(True, [])):
                result = recovery.inspect(RUN_ID)

            self.assertFalse(result["ready"])
            self.assertEqual(result["checks"]["state_evidence_mode"], "insufficient")

    def test_recovery_archives_lock_and_writes_receipt_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder))
            database = Database(settings.database_path)
            seed_retained_lock(settings, database)
            recovery = HardwareLockRecovery(settings, database)

            with patch.object(recovery, "_active_worker_pids", return_value=(True, [])):
                result = recovery.recover(
                    RUN_ID,
                    actor="operator",
                    operator_confirmation=f"RECOVER HARDWARE LOCK {RUN_ID}",
                )

            self.assertEqual(result["status"], "recovered")
            self.assertFalse(settings.lock_path.exists())
            self.assertTrue(Path(result["archived_lock_path"]).is_file())
            self.assertTrue(Path(result["receipt_path"]).is_file())
            events = database.all(
                "SELECT event_type FROM events WHERE workflow_id = ? ORDER BY id",
                (WORKFLOW_ID,),
            )
            self.assertEqual(
                [event["event_type"] for event in events],
                ["hardware_lock_recovery_authorized", "hardware_lock_recovered"],
            )


if __name__ == "__main__":
    unittest.main()
