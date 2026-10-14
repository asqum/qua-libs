from __future__ import annotations

import getpass
import hashlib
import hmac
import json
import math
import os
import re
import sqlite3
import statistics
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Collection
from urllib.parse import urlsplit

from .analysis import AnalysisError, SnapshotAnalyzer, evidence_digest
from .config import Settings
from .db import Database
from .policy import PolicyEngine, PolicyError
from .runner import ExperimentRunner, RunnerError
from .state import (
    bootstrap_patch,
    channel_kind,
    commit_state,
    drive_lo_recenter_patch,
    filter_patch,
    fresh_qubit_defaults_patch,
    json_diff,
    load_state,
    operation_amplitude,
    operation_backing_name,
    patch_qubit_targets,
    pointer_get,
    snapshot_state_path,
    StateError,
)
from .util import (
    atomic_write_json,
    exclusive_file_lock,
    json_dumps,
    json_loads,
    parse_iso_datetime,
    sha256_file,
    utc_now,
)
from .state_patch_02c import _output_reference as _readout_output_reference
from .state_patch_02c import _readout_limits as _readout_power_limits
from .state_patch_02c import _readout_line as _readout_line_of
from .setup_02x import (
    probe_02x_headroom_patch,
    unique_grid_location_patch,
    unique_thread_patch,
)
from .state_patch_02c import _smallest_full_scale
from .state_patch_02c import readout_port_members
from .xy_lo import xy_lo_plan_patch, xy_output as _xy_output_of
from .workflow_02c import resolve_02c_targets
from .workflow_subgroups import (
    FLUX_NODES,
    SHARED_FIRST_BATCH_NODES,
    SUBGROUP_NODES,
    resolve_02b_tunable_targets,
    resolve_03a_candidate_targets,
    resolve_03a_shift_ready_targets,
    resolve_node_targets,
)


class ServiceError(RuntimeError):
    pass


class AutonomyScopeError(ServiceError):
    pass


class AutonomyQuotaError(ServiceError):
    """A per-target attempt budget is exhausted without invalidating the lease."""


class AutonomyEvidenceError(ServiceError):
    pass


SHUTDOWN_HINT = "如需結束量測，請於網頁首頁結束量測後再對話輸入「結束量測」。"

# The documented resume phrases for a paused autonomy lease (Command.md).
AUTONOMY_RESUME_PHRASES = (
    "繼續 JY 自動量測",
    "繼續JY自動量測",
    "Resume JY automatic measurement",
)


def operator_handoff(do: str, reply: str) -> dict[str, Any]:
    """Fixed chat footer: what the operator must do, then which phrase to send."""
    lines = [
        f"需要使用者做什麼: {do}",
        f"完成後回傳: {reply}",
        SHUTDOWN_HINT,
    ]
    return {
        "do": do,
        "reply": reply,
        "shutdown_hint": SHUTDOWN_HINT,
        "lines": lines,
        "chat": "\n".join(f"- {line}" for line in lines),
    }


def approval_operator_handoff() -> dict[str, str]:
    return operator_handoff(
        "開啟對話中的 Dashboard 網址並登入，到 Approval 頁核准。",
        "已核准",
    )


def instrument_operator_handoff() -> dict[str, str]:
    return operator_handoff(
        "現場檢查 QOP/OPX、儀器電源與實驗室網路，確認儀器可連線。",
        "恢復量測",
    )


def submission_escalation_operator_handoff() -> dict[str, str]:
    """Handoff for a single-target program the QOP still would not accept."""

    return operator_handoff(
        "單一 qubit 的程式仍送不進 QOP，且儀器有回應健康檢查，"
        "請檢查 QOP 工作佇列或重啟 QOP。",
        "恢復量測",
    )


def recover_operator_handoff() -> dict[str, str]:
    return operator_handoff(
        "讓系統安全收尾殘留的 workflow、process 與 JY 服務。",
        "恢復",
    )


def recovery_console_operator_handoff() -> dict[str, str]:
    return operator_handoff(
        "到量測電腦本機 loopback operator console 完成硬體鎖現場確認。",
        "恢復",
    )


def host_pause_operator_handoff() -> dict[str, str]:
    return operator_handoff(
        "無需新操作。對話若已被平台暫停，回原對話喚醒即可。",
        "已核准",
    )


def resume_autonomy_operator_handoff() -> dict[str, str]:
    return operator_handoff(
        "確認可繼續後，恢復同一份未過期的自動授權。",
        "繼續 JY 自動量測",
    )


# Operator choice 2026-10-04: at these nodes a target that passed analysis but
# was deliberately left out of the decision patch (a suspect fit, such as
# q9/q10's sub-microsecond Ramsey at 3.3 samples per lifetime) is not resolved,
# so it can be measured again. Operator instruction 2026-10-10 extended this
# beyond 06/06b: a qubit the agent did not commit is never a success (03a q9
# edge spike in run #374, 03b q8 neighbour line in run #381). 02a/02b/02c keep
# their resolution because they withhold by design (02b flat qubits, the 02a
# vs 02c split).
WITHHOLD_AWARE_NODES = frozenset(
    {"03a", "03b", "04", "05", "06", "06a", "06b", "10a"}
)


def _withheld_targets(rows: list[dict[str, Any]]) -> set[str]:
    """Passing targets the decision left out of the state patch.

    A target counts as withheld when the analyzer proposed values for it but
    the recorded decision's patch omits them, or when it passed and a
    non-empty patch omits it. Committing it in any run clears that.
    """
    committed: set[str] = set()
    withheld: set[str] = set()
    for row in rows:
        if row.get("decision") is None and row.get("state_patch_json") is None:
            continue
        patch = json_loads(row.get("state_patch_json"), [])
        if not isinstance(patch, list):
            patch = []
        analysis = json_loads(row.get("analysis_json"), {})
        passing = {str(name) for name in analysis.get("passing_targets") or []}
        proposed = patch_qubit_targets(analysis.get("candidate_state_patch") or [])
        in_patch = patch_qubit_targets(patch)
        committed |= passing & in_patch
        withheld |= (passing & proposed) - in_patch
        if patch:
            withheld |= passing - in_patch
    return withheld - committed


def _asset_token(path: Path) -> str:
    """Short version stamp for a result image: changes when the file does."""
    try:
        stat = path.stat()
        seed = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
    except OSError:
        seed = str(path)
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


# Nodes whose earlier runs stop counting after a 07d reopen (2026-10-02).
# A 07d reopen restarts the bring-up from 07d: runs of 07d and every later
# node before the reopen no longer count (2026-10-04).
READOUT_REOPEN_NODES = frozenset(
    {"07d", "07b", "06", "06a", "06b", "10a", "05st", "06st_t2star", "06st_t2e"}
)


# Operator request 2026-10-04: measurement modes the lease page offers from the
# serialization node on.
DRIVE_MODES = ("per_qubit", "multiplex")


def _natural_qubit_key(name: str) -> tuple[Any, ...]:
    """Sort q2 before q10."""
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(name))
    )


# Operator request 2026-10-07: run parameters the operator supplies on the
# lease page while ticking an optional node. Every run of that node in the
# workflow uses them.
OPERATOR_NODE_PARAMETERS = {"02": frozenset({"res_num", "res_design_freq"})}

# Nodes that only use the readout line. Everything after the last of them in
# the sequence works on the targets 02c resolved as present.
READOUT_LINE_NODES = frozenset({"02", "02x", "02c", "02a"})


def _after_readout_line_nodes(sequence: list[str] | tuple[str, ...]) -> list[str]:
    """The part of the sequence after its last readout-line node."""

    sequence = list(sequence)
    last = max(
        (index for index, node in enumerate(sequence) if node in READOUT_LINE_NODES),
        default=-1,
    )
    return sequence[last + 1:]


class AgentService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self.settings.ensure_runtime()
        self.db = Database(self.settings.database_path)
        self.policy = PolicyEngine(self.settings)
        self.runner = ExperimentRunner(self.settings, self.db)
        self.analyzer = SnapshotAnalyzer(self.settings, self.policy)
        self._approval_secret = self._load_or_create_approval_secret()
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        # session id -> (Dashboard event revision, result plot paths)
        self._dashboard_plot_cache: dict[str, tuple[int, list[str]]] = {}
        self._backfill_device_keys()
        self._backfill_inherited_marks()

    def _device_key(self) -> str:
        """The chip a workflow measures: the QuAM state folder config.toml names."""
        return str(Path(self.settings.quam_state_root).resolve())

    def _backfill_device_keys(self) -> None:
        """Give older workflows their chip from a recorded state-commit path.

        Operator choice 2026-10-10: a workflow whose chip cannot be told from
        its records stays without one and is never offered as history.
        """
        for row in self.db.all("SELECT id FROM workflows WHERE device_key IS NULL"):
            proposals = self.db.all(
                "SELECT payload_json FROM proposals WHERE workflow_id = ? "
                "AND payload_json LIKE '%\"active_state\"%' ORDER BY created_at",
                (row["id"],),
            )
            for proposal in proposals:
                path = json_loads(proposal["payload_json"], {}).get("active_state")
                if isinstance(path, str) and path.strip():
                    self.db.execute(
                        "UPDATE workflows SET device_key = ? "
                        "WHERE id = ? AND device_key IS NULL",
                        (str(Path(path).parent.resolve()), row["id"]),
                    )
                    break

    def idempotent_call(
        self,
        operation: str,
        operation_id: str | None,
        request: dict[str, Any],
        callback: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Replay one completed mutation and reject conflicting/concurrent reuse."""
        if operation_id is None:
            return callback()
        key = operation_id.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", key):
            raise ServiceError(
                "operation_id must be 8-128 stable URL-safe characters"
            )
        request_hash = hashlib.sha256(
            json_dumps(request).encode("utf-8")
        ).hexdigest()
        with self.db.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM idempotency_records "
                "WHERE operation = ? AND operation_id = ?",
                (operation, key),
            ).fetchone()
            if row is not None:
                record = dict(row)
                if not hmac.compare_digest(record["request_sha256"], request_hash):
                    raise ServiceError(
                        "operation_id was already used with a different request"
                    )
                if record["status"] == "completed":
                    return json_loads(record["response_json"], {})
                raise ServiceError(
                    "The same operation_id is already in progress; inspect status "
                    "before deciding whether recovery is required"
                )
            connection.execute(
                "INSERT INTO idempotency_records(operation, operation_id, "
                "request_sha256, status, created_at) VALUES (?, ?, ?, 'in_progress', ?)",
                (operation, key, request_hash, utc_now()),
            )
        try:
            response = callback()
        except Exception:
            self.db.execute(
                "DELETE FROM idempotency_records WHERE operation = ? "
                "AND operation_id = ? AND status = 'in_progress'",
                (operation, key),
            )
            raise
        self.db.execute(
            "UPDATE idempotency_records SET status = 'completed', response_json = ?, "
            "completed_at = ? WHERE operation = ? AND operation_id = ? "
            "AND status = 'in_progress'",
            (json_dumps(response), utc_now(), operation, key),
        )
        return response

    def status(self, workflow_id: str | None = None) -> dict[str, Any]:
        workflow = (
            self.db.one("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
            if workflow_id
            else self.db.one("SELECT * FROM workflows ORDER BY created_at DESC LIMIT 1")
        )
        active_run = self.db.one(
            """
            SELECT * FROM runs
            WHERE status IN ('starting', 'running', 'stopping')
            ORDER BY started_at DESC LIMIT 1
            """
        )
        lock = None
        if self.settings.lock_path.is_file():
            try:
                lock = json.loads(self.settings.lock_path.read_text(encoding="utf-8"))
            except Exception:
                lock = {"status": "unreadable"}
        result: dict[str, Any] = {
            "server": {
                "endpoint": f"http://{self.settings.host}:{self.settings.port}{self.settings.mcp_path}",
                "approval_endpoint": self.browser_approval_origin,
                "approval_transport": self.approval_transport,
                "remote_approval_enabled": self.remote_approval_enabled,
                "recovery_only": os.getenv("JY_RECOVERY_ONLY", "0") == "1",
                "qualibrate_config": str(self.settings.qualibrate_config_path),
                "qualibrate_project": self.settings.qualibrate_project,
                "active_state": str(self.settings.active_state),
                "wiring": str(self.settings.wiring_path),
                "data_root": str(self.settings.data_root),
                "operator_console": self.operator_console_url,
            },
            "measurement_mode": {
                "entry_phrase": self.settings.measurement_mode_entry_phrase,
                "entry_phrases": list(
                    self.settings.measurement_mode_entry_phrases
                ),
                "exit_phrase": self.settings.measurement_mode_exit_phrase,
                "approval_model": "per_run_and_state_commit",
                "workflow_active": bool(
                    workflow and workflow["status"] == "active"
                ),
            },
            "autonomy_mode": {
                "entry_phrase": self.settings.autonomy_mode_entry_phrase,
                "entry_phrases": list(
                    self.settings.autonomy_mode_entry_phrases
                ),
                "approval_model": "bounded_lease",
            },
            "recovery_phrases": list(self.settings.recovery_phrases),
            "measurement_resume_phrases": list(
                self.settings.measurement_resume_phrases
            ),
            "hardware_lock": lock,
            "active_run": self._decode_run(active_run) if active_run else None,
        }
        if workflow:
            result["workflow"] = self._decode_workflow(workflow)
            result["recent_runs"] = [
                self._decode_run(item)
                for item in self.db.all(
                    "SELECT * FROM runs WHERE workflow_id = ? AND inherited_from_run_id IS NULL "
                    "ORDER BY started_at DESC LIMIT 10",
                    (workflow["id"],),
                )
            ]
            lease = self.db.one(
                "SELECT * FROM autonomy_leases WHERE workflow_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (workflow["id"],),
            )
            result["autonomy"] = (
                self.autonomy_status(lease_id=lease["id"]) if lease else None
            )
        else:
            result["workflow"] = None
            result["autonomy"] = None
        return result

    def run_telemetry(
        self,
        node_id: str | None = None,
        targets: list[str] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Return persisted run parameters/timing and a robust duration estimate."""
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
            raise ServiceError("limit must be between 1 and 500")
        query = "SELECT * FROM runs"
        params: tuple[Any, ...] = ()
        if node_id is not None:
            self.policy.node_definition(node_id)
            query += " WHERE node_id = ?"
            params = (node_id,)
        rows = self.db.all(query + " ORDER BY started_at DESC LIMIT ?", (*params, limit))
        requested = set(targets or [])
        history: list[dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_run(row)
            run_targets = set(decoded["parameters"].get("qubits", []))
            if requested and run_targets != requested:
                continue
            history.append(
                {
                    "run_id": decoded["id"],
                    "node_id": decoded["node_id"],
                    "targets": sorted(run_targets),
                    "parameters": decoded["parameters"],
                    "status": decoded["status"],
                    "started_at": decoded.get("started_at"),
                    "finished_at": decoded.get("finished_at"),
                    "elapsed_seconds": decoded.get("elapsed_seconds"),
                    "snapshot_id": decoded.get("snapshot_id"),
                    "analysis_status": decoded.get("analysis_status"),
                }
            )
        durations = [
            float(item["elapsed_seconds"])
            for item in history
            if item["status"] == "completed"
            and isinstance(item.get("elapsed_seconds"), (int, float))
        ]
        prediction = None
        if durations:
            ordered = sorted(durations)
            prediction = {
                "basis": "completed runs with the same node and exact targets"
                if requested
                else "completed runs for the selected node",
                "sample_size": len(ordered),
                "median_seconds": statistics.median(ordered),
                "p80_seconds": ordered[min(len(ordered) - 1, math.ceil(0.8 * len(ordered)) - 1)],
                "minimum_seconds": ordered[0],
                "maximum_seconds": ordered[-1],
            }
        return {"prediction": prediction, "history": history}

    def decision_experience(
        self,
        node_id: str | None = None,
        targets: list[str] | None = None,
        limit: int = 10,
        include_analysis: bool = False,
    ) -> list[dict[str, Any]]:
        """Expose recorded scientific decisions as reusable audit experience.

        Operator instruction 2026-09-20: each entry used to carry the run's
        whole analysis document, and the default asked for fifty of them, which
        is far more context than choosing the next parameters needs. Each entry
        now carries `evidence`, a per-target digest of the numbers the playbook
        actually gates on. Pass `include_analysis=True` for the full document.
        """
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
            raise ServiceError("limit must be between 1 and 200")
        query = (
            "SELECT d.*, r.node_id, r.parameters_json, r.analysis_json, "
            "r.analysis_status, r.snapshot_id, r.snapshot_path, r.started_at, "
            "r.finished_at FROM decisions d JOIN runs r ON r.id = d.run_id"
        )
        params: tuple[Any, ...] = ()
        if node_id is not None:
            self.policy.node_definition(node_id)
            query += " WHERE r.node_id = ?"
            params = (node_id,)
        rows = self.db.all(query + " ORDER BY d.created_at DESC LIMIT ?", (*params, limit))
        requested = set(targets or [])
        result: list[dict[str, Any]] = []
        for row in rows:
            parameters = json_loads(row["parameters_json"], {})
            run_targets = set(parameters.get("qubits", []))
            if requested and run_targets != requested:
                continue
            analysis = json_loads(row.get("analysis_json"), None)
            entry = {
                "decision_id": row["id"],
                "workflow_id": row["workflow_id"],
                "run_id": row["run_id"],
                "node_id": row["node_id"],
                "targets": sorted(run_targets),
                "parameters": parameters,
                "elapsed_seconds": _elapsed_seconds(row),
                "snapshot_id": row.get("snapshot_id"),
                "snapshot_path": row.get("snapshot_path"),
                "analysis_status": row.get("analysis_status"),
                "evidence": evidence_digest(analysis),
                "Decision": row["decision"],
                "Reason": row["reason"],
                "next_action": {
                    "next_node": row.get("next_node"),
                    "new_parameters": json_loads(row["next_parameters_json"], {}),
                },
                "state_patch": json_loads(row["state_patch_json"], []),
                "created_at": row["created_at"],
            }
            if include_analysis:
                entry["analysis"] = analysis
            result.append(entry)
        return result

    def start_workflow(
        self,
        targets: list[str],
        initial_parameters: dict[str, Any] | None,
        client_id: str,
        activation_phrase: str,
    ) -> dict[str, Any]:
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        if activation_phrase not in {
            *self.settings.measurement_mode_entry_phrases,
            *self.settings.autonomy_mode_entry_phrases,
        }:
            raise ServiceError(
                "Starting a workflow requires the exact measurement-mode or "
                "autonomy-mode entry phrase"
            )
        state = load_state(self.settings.active_state)
        if not targets or any(name not in state.get("qubits", {}) for name in targets):
            raise ServiceError("targets must be explicit qubits present in active state")
        if len(set(targets)) != len(targets):
            raise ServiceError("targets must not contain duplicates")
        workflow_parameters = initial_parameters or {}
        if not isinstance(workflow_parameters, dict):
            raise ServiceError("initial_parameters must be an object")
        multiplexed = workflow_parameters.get("multiplexed", False)
        if not isinstance(multiplexed, bool):
            raise ServiceError("workflow multiplexed must be a boolean")
        if multiplexed and len(targets) > 1:
            # Unset or duplicate readout IFs do not block entry; the workflow
            # then has to start with 02 (operator rule 2026-10-07).
            self.policy.validate_multiplex_targets(
                targets,
                state,
                allow_unset_readout=bool(
                    self.policy.readout_frequencies_unset(targets, state)
                ),
                # An over-budget line is scaled by a required setup before
                # the first run (operator choice 2026-10-07).
                allow_over_budget=bool(
                    self.policy.readout_amplitude_budget_patch(targets, state)[0]
                ),
            )
        workflow_id = uuid.uuid4().hex
        now = utc_now()
        first_node = self.settings.workflow_sequence[0]
        try:
            with self.db.transaction(immediate=True) as connection:
                existing = connection.execute(
                    "SELECT id FROM workflows WHERE status IN ('active', 'paused') "
                    "LIMIT 1"
                ).fetchone()
                if existing is not None:
                    raise ServiceError(
                        "An active or paused workflow already exists; resume or "
                        "finish it before starting another. If it is stale, issue "
                        "'恢復' or 'Recover' to close the prior JY state safely"
                    )
                connection.execute(
                    """
                    INSERT INTO workflows(
                        id, created_at, updated_at, status, targets_json,
                        initial_parameters_json, current_node, client_id
                    ) VALUES (?, ?, ?, 'active', ?, ?, ?, ?)
                    """,
                    (
                        workflow_id,
                        now,
                        now,
                        json_dumps(targets),
                        json_dumps(workflow_parameters),
                        first_node,
                        client_id,
                    ),
                )
                connection.execute(
                    "UPDATE workflows SET device_key = ? WHERE id = ?",
                    (self._device_key(), workflow_id),
                )
                self.db.event(
                    "workflow_started",
                    client_id,
                    {
                        "targets": targets,
                        "current_node": first_node,
                        "multiplexed": multiplexed,
                    },
                    workflow_id,
                    connection=connection,
                )
        except sqlite3.IntegrityError as exc:
            raise ServiceError(
                "Another client started an active workflow concurrently"
            ) from exc
        return self._decode_workflow(
            self.db.one("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
        )

    def request_autonomy_lease(
        self,
        workflow_id: str,
        client_id: str,
        activation_phrase: str,
        reason: str,
        targets: list[str] | None = None,
        allowed_nodes: list[str] | None = None,
        duration_hours: float | None = None,
    ) -> dict[str, Any]:
        """Create the one human approval that bounds unattended JY actions."""
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        if activation_phrase not in self.settings.autonomy_mode_entry_phrases:
            raise ServiceError(
                "Autonomy requires the exact autonomous-measurement entry phrase"
            )
        if not reason.strip():
            raise ServiceError("An autonomy authorization reason is required")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active":
            raise ServiceError("Workflow must be active before requesting autonomy")
        existing = self.db.one(
            "SELECT * FROM autonomy_leases WHERE workflow_id = ? "
            "AND status IN ('pending', 'active', 'paused') "
            "ORDER BY created_at DESC LIMIT 1",
            (workflow_id,),
        )
        if existing is not None:
            existing = self._expire_autonomy_if_needed(existing)
            if existing["status"] in {"pending", "active", "paused"}:
                raise ServiceError(
                    "Workflow already has a pending or usable autonomy lease: "
                    f"{existing['id']} ({existing['status']})"
                )

        workflow = self._rewind_invalidated_subgroup_evidence(
            workflow, client_id
        )

        workflow_targets = json_loads(workflow["targets_json"], [])
        resolved_targets = targets or list(workflow_targets)
        if (
            not resolved_targets
            or len(set(resolved_targets)) != len(resolved_targets)
            or not set(resolved_targets).issubset(set(workflow_targets))
        ):
            raise ServiceError(
                "Autonomy targets must be a non-empty subset of workflow targets"
            )
        sequence = self._workflow_sequence(workflow, include_undecided_flux=True)
        current_index = sequence.index(workflow["current_node"])
        serial_node = self._serialize_from_node(workflow)
        if serial_node is not None:
            # Each new qubit restarts the tail at the serialization node, so a
            # lease taken mid-pass still has to cover it.
            current_index = min(current_index, sequence.index(serial_node))
        remaining_nodes = sequence[current_index:]
        resolved_nodes = allowed_nodes or remaining_nodes
        if (
            not resolved_nodes
            or len(set(resolved_nodes)) != len(resolved_nodes)
            or resolved_nodes[0] != remaining_nodes[0]
            or any(node not in remaining_nodes for node in resolved_nodes)
            or [node for node in remaining_nodes if node in resolved_nodes]
            != resolved_nodes
        ):
            raise ServiceError(
                "Autonomy nodes must be an ordered subset of the remaining workflow "
                "and start at the current node"
            )
        config = self.policy.raw["autonomy"]
        duration_hours = self._resolve_autonomy_duration(duration_hours)
        max_total_runs = config.get("max_total_runs")
        max_attempts = int(config["max_attempts_per_node_qubit"])
        lease_id = uuid.uuid4().hex
        payload = {
            "lease_id": lease_id,
            "reason": reason.strip(),
            "targets": resolved_targets,
            "allowed_nodes": resolved_nodes,
            "duration_hours": duration_hours,
            "max_total_runs": max_total_runs,
            "max_attempts_per_node_qubit": max_attempts,
            "auto_state_commit_analysis_statuses": list(
                config["auto_state_commit_analysis_statuses"]
            ),
            "pause_on_analysis_statuses": list(
                config.get("pause_on_analysis_statuses", [])
            ),
            "halt_conditions": list(config["halt_conditions"]),
            "manual_controls": ["pause", "stop", "emergency_stop"],
            "workflow_options_offer": self._lease_option_offer(workflow),
        }
        proposal = self._create_proposal(
            workflow_id,
            "autonomy_lease",
            payload,
            client_id,
            int(self.policy.raw["approval"]["run_ttl_minutes"]),
            1,
            autonomy_lease_id=lease_id,
        )
        now = utc_now()
        try:
            self.db.execute(
                """
                INSERT INTO autonomy_leases(
                    id, workflow_id, proposal_id, status, targets_json,
                    allowed_nodes_json, duration_hours, max_total_runs,
                    max_attempts_per_node_qubit, auto_state_commit_statuses_json,
                    halt_conditions_json, created_at, updated_at, source_client
                ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lease_id,
                    workflow_id,
                    proposal["id"],
                    json_dumps(resolved_targets),
                    json_dumps(resolved_nodes),
                    duration_hours,
                    max_total_runs,
                    max_attempts,
                    json_dumps(config["auto_state_commit_analysis_statuses"]),
                    json_dumps(config["halt_conditions"]),
                    now,
                    now,
                    client_id,
                ),
            )
        except sqlite3.IntegrityError as exc:
            self.db.execute(
                "UPDATE proposals SET status = 'cancelled' WHERE id = ? "
                "AND status = 'pending'",
                (proposal["id"],),
            )
            raise ServiceError(
                "Another client created a pending or active autonomy lease concurrently"
            ) from exc
        self.db.event(
            "autonomy_lease_requested",
            client_id,
            payload,
            workflow_id,
        )
        dashboard = self._start_dashboard_session(
            workflow_id,
            "autonomous",
            client_id,
            autonomy_lease_id=lease_id,
        )
        # Decode the proposal only after the workflow's stable Dashboard session
        # is bound to this lease.  Direct MCP callers of request_autonomy_lease
        # must receive the same canonical browser page as enter_autonomy_mode.
        proposal = self.proposal(proposal["id"])
        proposal["autonomy"] = self.autonomy_status(lease_id=lease_id)
        proposal["dashboard"] = dashboard
        proposal["browser_url"] = dashboard["browser_url"]
        return proposal

    def enter_measurement_mode(
        self,
        client_id: str,
        activation_phrase: str,
        targets: list[str] | None = None,
        *,
        multiplexed: bool | None = None,
    ) -> dict[str, Any]:
        """Start or resume conversational, one-experiment-at-a-time mode."""
        if activation_phrase not in self.settings.measurement_mode_entry_phrases:
            raise ServiceError(
                "Entry requires the exact JY measurement-mode phrase"
            )
        self._reject_retained_hardware_lock()
        workflow = self._enter_or_resume_workflow(
            client_id,
            activation_phrase,
            targets,
            multiplexed=multiplexed,
            allow_live_autonomy=False,
        )
        dashboard = self._start_dashboard_session(
            workflow["id"], "conversational", client_id
        )
        return {
            "workflow": workflow,
            "mode": "conversational",
            "approval_model": "per_run_and_state_commit",
            "next_action": "discuss_and_propose_one_experiment",
            "current_node": workflow["current_node"],
            "dashboard": dashboard,
            "browser_url": dashboard["browser_url"],
            "operator_handoff": approval_operator_handoff(),
        }

    def experiment_catalog(self) -> dict[str, Any]:
        """List policy-runnable experiments and scripts awaiting safe onboarding."""
        registered: list[dict[str, Any]] = []
        registered_paths: set[Path] = set()
        sequence = set(self.settings.workflow_sequence)
        for node_id, definition in sorted(self.policy.nodes.items()):
            script = self.policy.node_script(node_id)
            registered_paths.add(script.resolve())
            registered.append(
                {
                    "node_id": node_id,
                    "name": definition["name"],
                    "script": str(script),
                    "automatic_sequence": node_id in sequence,
                    "conversational_available": True,
                    "allowed_parameters": definition.get(
                        "allowed_parameters", {}
                    ),
                    "defaults": definition.get("defaults", {}),
                }
            )

        roots = {
            "calibration_graph": self.settings.calibration_graph,
            "Script": self.settings.superconducting_root / "Script",
            "side_project": self.settings.superconducting_root / "side_project",
        }
        awaiting: list[dict[str, str]] = []
        for root_name, root in roots.items():
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.py")):
                if "__pycache__" in path.parts or path.resolve() in registered_paths:
                    continue
                awaiting.append(
                    {
                        "root": root_name,
                        "script": path.relative_to(root).as_posix(),
                        "status": "policy_onboarding_required",
                    }
                )
        return {
            "registered": registered,
            "registered_count": len(registered),
            "awaiting_policy_onboarding": awaiting,
            "awaiting_count": len(awaiting),
            "safety_note": (
                "Conversational mode can run every registered node, but never "
                "executes an unregistered Python file. Add a policy schema, limits, "
                "and analyzer before a discovered script becomes runnable."
            ),
        }

    def enter_autonomy_mode(
        self,
        client_id: str,
        activation_phrase: str,
        targets: list[str] | None = None,
        *,
        multiplexed: bool | None = None,
        duration_hours: float | None = None,
        reason: str = "Operator entered JY autonomous measurement mode.",
    ) -> dict[str, Any]:
        """Start/resume a workflow and obtain its bounded autonomy lease.

        `duration_hours` sets this session's budget; omitting it uses the
        configured default. It applies only when a new lease is created -- an
        existing pending, active, or paused lease keeps the hours it was
        approved with.
        """
        if activation_phrase not in self.settings.autonomy_mode_entry_phrases:
            raise ServiceError(
                "Entry requires the exact JY autonomous-measurement phrase"
            )
        # Reject an out-of-policy budget before a workflow is created.
        self._resolve_autonomy_duration(duration_hours)
        self._reject_retained_hardware_lock()
        workflow = self._enter_or_resume_workflow(
            client_id,
            activation_phrase,
            targets,
            multiplexed=multiplexed,
            allow_live_autonomy=True,
        )
        now = utc_now()
        self.db.execute(
            "UPDATE proposals SET status = 'expired' "
            "WHERE workflow_id = ? AND status = 'pending' "
            "AND autonomy_lease_id IS NULL AND expires_at <= ?",
            (workflow["id"], now),
        )
        pending_manual = self.db.one(
            "SELECT id FROM proposals WHERE workflow_id = ? AND status = 'pending' "
            "AND autonomy_lease_id IS NULL AND kind != 'autonomy_lease' "
            "ORDER BY created_at DESC LIMIT 1",
            (workflow["id"],),
        )
        if pending_manual is not None:
            raise ServiceError(
                "Automatic mode cannot start while a conversational proposal is "
                "pending; approve, expire, or cancel that proposal first"
            )

        lease_row = self.db.one(
            "SELECT * FROM autonomy_leases WHERE workflow_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (workflow["id"],),
        )
        if lease_row is not None:
            lease = self.autonomy_status(lease_id=lease_row["id"])
            if lease["status"] == "pending":
                result = {
                    "workflow": workflow,
                    "mode": "autonomous",
                    "authorization": lease,
                    "approval": self.proposal(lease["proposal_id"]),
                    "next_action": "human_approval",
                }
                return self._with_autonomy_dashboard(result, client_id)
            if lease["status"] == "active":
                result = {
                    "workflow": workflow,
                    "mode": "autonomous",
                    "authorization": lease,
                    "next_action": "continue_authorized_workflow",
                }
                return self._with_autonomy_dashboard(result, client_id)
            if lease["status"] == "paused":
                lease = self.resume_autonomy(
                    lease["id"], client_id, activation_phrase
                )
                result = {
                    "workflow": workflow,
                    "mode": "autonomous",
                    "authorization": lease,
                    "next_action": "continue_authorized_workflow",
                }
                return self._with_autonomy_dashboard(result, client_id)

        proposal = self.request_autonomy_lease(
            workflow["id"],
            client_id,
            activation_phrase,
            reason or "Operator entered JY autonomous measurement mode.",
            duration_hours=duration_hours,
        )
        result = {
            "workflow": workflow,
            "mode": "autonomous",
            "authorization": proposal["autonomy"],
            "approval": proposal,
            "next_action": "human_approval",
        }
        return self._with_autonomy_dashboard(result, client_id)

    def _with_autonomy_dashboard(
        self, result: dict[str, Any], client_id: str
    ) -> dict[str, Any]:
        lease_id = str(result["authorization"]["id"])
        dashboard = self._start_dashboard_session(
            str(result["authorization"]["workflow_id"]),
            "autonomous",
            client_id,
            autonomy_lease_id=lease_id,
        )
        result["authorization"] = self.autonomy_status(lease_id=lease_id)
        approval_id = str((result.get("approval") or {}).get("id") or "")
        if approval_id:
            result["approval"] = self.proposal(approval_id)
        result["dashboard"] = dashboard
        result["browser_url"] = dashboard["browser_url"]
        if result.get("next_action") == "human_approval":
            result["operator_handoff"] = approval_operator_handoff()
        return result

    def _active_qubit_names_from_state(self) -> list[str]:
        state = load_state(self.settings.active_state)
        names = state.get("active_qubit_names")
        if not isinstance(names, list) or not names:
            raise ServiceError(
                "A new workflow needs targets; none were given and "
                "state.json active_qubit_names is missing or empty"
            )
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ServiceError("active_qubit_names must be a list of qubit names")
        if len(set(names)) != len(names):
            raise ServiceError("active_qubit_names must not contain duplicates")
        return list(names)

    def _enter_or_resume_workflow(
        self,
        client_id: str,
        activation_phrase: str,
        targets: list[str] | None,
        *,
        multiplexed: bool | None,
        allow_live_autonomy: bool,
    ) -> dict[str, Any]:
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")

        row = self.db.one(
            "SELECT * FROM workflows WHERE status IN ('active', 'paused') "
            "ORDER BY created_at DESC LIMIT 1"
        )
        if row is None:
            resolved_targets = list(targets) if targets else self._active_qubit_names_from_state()
            resolved_multiplexed = True if multiplexed is None else bool(multiplexed)
            workflow = self.start_workflow(
                resolved_targets,
                {"multiplexed": resolved_multiplexed},
                client_id,
                activation_phrase,
            )
        else:
            workflow = self._decode_workflow(row)
            if targets is not None and list(targets) != workflow["targets"]:
                raise ServiceError(
                    "Requested targets differ from the existing workflow; stop or "
                    "finish it before creating a different authorization scope. "
                    "If the prior workflow is stale, issue '恢復' or 'Recover'"
                )
            existing_multiplexed = bool(
                workflow["initial_parameters"].get("multiplexed", False)
            )
            if multiplexed is True and not existing_multiplexed:
                raise ServiceError(
                    "The existing workflow was not created as multiplexed"
                )
            if not allow_live_autonomy:
                lease_row = self.db.one(
                    "SELECT * FROM autonomy_leases WHERE workflow_id = ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (workflow["id"],),
                )
                if lease_row is not None:
                    lease = self.autonomy_status(lease_id=lease_row["id"])
                    if lease["status"] in {"pending", "active", "paused"}:
                        raise ServiceError(
                            "Conversational mode cannot share a live autonomy lease; "
                            "stop or revoke that lease first"
                        )
            if workflow["status"] == "paused":
                workflow = self.resume_measurement_mode(
                    workflow["id"], client_id, activation_phrase
                )
        return workflow

    def autonomy_status(
        self,
        lease_id: str | None = None,
        workflow_id: str | None = None,
    ) -> dict[str, Any]:
        if not lease_id and not workflow_id:
            raise ServiceError("lease_id or workflow_id is required")
        if lease_id:
            row = self.db.one("SELECT * FROM autonomy_leases WHERE id = ?", (lease_id,))
        else:
            row = self.db.one(
                "SELECT * FROM autonomy_leases WHERE workflow_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (workflow_id,),
            )
        if row is None:
            raise ServiceError("Unknown autonomy lease")
        row = self._expire_autonomy_if_needed(row)
        return self._decode_autonomy(row)

    def wait_for_autonomy_status(
        self,
        lease_id: str,
        *,
        after_event_id: int = 0,
        timeout_seconds: float = 45.0,
    ) -> dict[str, Any]:
        """Wait briefly for approval, a result/control event, or terminal lease state."""
        timeout = max(0.0, min(float(timeout_seconds), 55.0))
        cursor = max(0, int(after_event_id))
        deadline = time.monotonic() + timeout
        initial_status = self.autonomy_status(lease_id=lease_id)["status"]
        while True:
            status = self.autonomy_status(lease_id=lease_id)
            events = self.autonomy_ui_events_after(lease_id, cursor)
            if (
                events["events"]
                or status["status"] != initial_status
                or status["status"] not in {"pending", "active"}
            ):
                return self._with_wait_operator_handoff(
                    {
                        "timed_out": False,
                        "authorization": status,
                        "cursor": events["cursor"],
                        "events": events["events"],
                    }
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._with_wait_operator_handoff(
                    {
                        "timed_out": True,
                        "authorization": status,
                        "cursor": events["cursor"],
                        "events": [],
                    }
                )
            time.sleep(min(0.25, remaining))

    def _with_wait_operator_handoff(self, result: dict[str, Any]) -> dict[str, Any]:
        status = str((result.get("authorization") or {}).get("status") or "")
        if status == "pending":
            result["operator_handoff"] = approval_operator_handoff()
        elif status == "paused":
            result["operator_handoff"] = resume_autonomy_operator_handoff()
        return result

    def mark_scientifically_unmeasurable(
        self,
        workflow_id: str,
        lease_id: str,
        run_id: str,
        targets: list[str],
        reason: str,
        client_id: str,
    ) -> dict[str, Any]:
        """Mark evidence-reviewed targets incomplete before their attempt quota."""
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        if not reason.strip():
            raise ServiceError("A scientific boundary reason is required")
        lease = self._active_autonomy(lease_id, workflow_id)
        workflow = self._workflow(workflow_id)
        run = self.db.one(
            "SELECT * FROM runs WHERE id = ? AND workflow_id = ?",
            (run_id, workflow_id),
        )
        if run is None or run.get("autonomy_lease_id") != lease["id"]:
            raise ServiceError("Run is not associated with this autonomy lease")
        if run["node_id"] != workflow["current_node"]:
            raise ServiceError("Scientific boundary evidence must belong to the current node")
        # Operator choice 2026-10-14 (q3 06 #a326ea36): a node whose own fit
        # crashes after measuring (worker exception in the hardware phase)
        # leaves no completed run; that crash is the scientific evidence.
        node_fit_crash = (
            run["status"] == "failed"
            and run.get("termination_cause") == "worker_exception"
            and json_loads(run.get("analysis_json"), {}).get("execution_phase")
            == "hardware_execution"
        )
        if not node_fit_crash and (
            run["status"] != "completed"
            or run["analysis_status"] not in {"needs_review", "failed"}
        ):
            raise ServiceError(
                "Scientific incompleteness requires a completed non-passing analysis"
            )
        decision = self.db.one(
            "SELECT decision FROM decisions WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
            (run_id,),
        )
        # Either order is allowed: review the evidence run first, or mark the
        # target here and let the `advance` that closes the node be this run's
        # one decision. A per-qubit pass needs the second form -- when its only
        # qubit is unmeasurable there is no other run left to carry the
        # advance, and every run may hold exactly one decision.
        if decision is not None and decision["decision"] != "manual_review":
            raise ServiceError(
                "The evidence run already recorded "
                f"{decision['decision']!r}; a target can only be marked "
                "scientifically unmeasurable against an unreviewed run or one "
                "held at manual_review"
            )
        normalized_targets = {str(target) for target in targets if str(target)}
        if not normalized_targets:
            raise ServiceError("At least one scientifically unmeasurable target is required")
        run_targets = set(
            json_loads(run.get("parameters_json"), {}).get("qubits", [])
        )
        if not normalized_targets.issubset(run_targets):
            raise ServiceError("Targets must be a subset of the evidence run targets")
        active_targets = self._active_targets_for_node(workflow, run["node_id"])
        if not normalized_targets.issubset(active_targets):
            raise ServiceError("Targets are outside the active workflow target set")
        resolved = self._node_target_resolution(workflow_id, run["node_id"])
        already_resolved = normalized_targets & resolved
        if already_resolved:
            raise ServiceError(
                f"Resolved targets cannot be marked incomplete: {sorted(already_resolved)}"
            )
        self.db.event(
            "autonomy_targets_scientifically_unmeasurable",
            client_id,
            {
                "lease_id": lease_id,
                "run_id": run_id,
                "node_id": run["node_id"],
                "targets": sorted(normalized_targets),
                "reason": reason.strip(),
            },
            workflow_id,
        )
        return {
            "workflow_id": workflow_id,
            "lease_id": lease_id,
            "run_id": run_id,
            "node_id": run["node_id"],
            "targets": sorted(normalized_targets),
            "status": "scientifically_unmeasurable",
            "reason": reason.strip(),
            "incomplete_targets_by_node": self.autonomy_status(
                lease_id=lease_id
            )["incomplete_targets_by_node"],
        }

    def pause_autonomy(
        self, lease_id: str, client_id: str, reason: str
    ) -> dict[str, Any]:
        return self._set_autonomy_control(
            lease_id, client_id, reason, action="pause"
        )

    def resume_autonomy(
        self,
        lease_id: str,
        client_id: str,
        activation_phrase: str,
    ) -> dict[str, Any]:
        # 2026-10-12: Command.md documents 「繼續 JY 自動量測」 as the
        # resume phrase for a paused lease, so it is accepted here too.
        if activation_phrase not in {
            *self.settings.autonomy_mode_entry_phrases,
            *self.settings.measurement_resume_phrases,
            *AUTONOMY_RESUME_PHRASES,
        }:
            raise ServiceError(
                "Resuming autonomy requires an exact resume or "
                "autonomous-measurement phrase"
            )
        self._reject_retained_hardware_lock()
        row = self._autonomy_row(lease_id)
        row = self._expire_autonomy_if_needed(row)
        if row["status"] != "paused":
            raise ServiceError("Only a paused, unexpired autonomy lease can resume")
        self.db.execute(
            "UPDATE autonomy_leases SET status = 'active', stopped_reason = NULL, "
            "updated_at = ? WHERE id = ?",
            (utc_now(), lease_id),
        )
        self.db.execute(
            "UPDATE measurement_sessions SET status = 'active', updated_at = ? "
            "WHERE autonomy_lease_id = ? AND status = 'paused'",
            (utc_now(), lease_id),
        )
        self.db.event(
            "autonomy_resumed", client_id, {"lease_id": lease_id}, row["workflow_id"]
        )
        return self.autonomy_status(lease_id=lease_id)

    def stop_autonomy(
        self,
        lease_id: str,
        client_id: str,
        reason: str,
        *,
        emergency: bool = False,
    ) -> dict[str, Any]:
        return self._set_autonomy_control(
            lease_id,
            client_id,
            reason,
            action="emergency_stop" if emergency else "stop",
        )

    def leave_measurement_mode(
        self,
        workflow_id: str,
        client_id: str,
        exit_phrase: str,
    ) -> dict[str, Any]:
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        if exit_phrase not in self.settings.measurement_mode_shutdown_phrases:
            raise ServiceError(
                "Leaving measurement mode requires a configured shutdown phrase"
            )
        return self.stop_workflow(
            workflow_id,
            client_id,
            f"Operator issued shutdown phrase: {exit_phrase}",
        )

    def resume_measurement_mode(
        self,
        workflow_id: str,
        client_id: str,
        activation_phrase: str,
    ) -> dict[str, Any]:
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        if activation_phrase not in {
            *self.settings.measurement_resume_phrases,
            *self.settings.measurement_mode_entry_phrases,
            *self.settings.autonomy_mode_entry_phrases,
        }:
            raise ServiceError(
                "Resuming requires an exact measurement-resume or mode-entry phrase"
            )
        self._reject_retained_hardware_lock()
        active_run = self.db.one(
            "SELECT id FROM runs WHERE workflow_id = ? "
            "AND status IN ('starting', 'running', 'stopping') LIMIT 1",
            (workflow_id,),
        )
        if active_run is not None:
            raise ServiceError("The workflow already has an active worker")
        workflow = self._workflow(workflow_id)
        lease_row = self.db.one(
            "SELECT * FROM autonomy_leases WHERE workflow_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (workflow_id,),
        )
        lease = (
            self._expire_autonomy_if_needed(lease_row)
            if lease_row is not None
            else None
        )
        paused_lease = lease if lease and lease["status"] == "paused" else None
        if workflow["status"] != "paused" and paused_lease is None:
            raise ServiceError("Only a paused workflow or authorization can be resumed")
        now = utc_now()
        with self.db.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE workflows SET status = 'active', updated_at = ? "
                "WHERE id = ? AND status = 'paused'",
                (now, workflow_id),
            )
            if paused_lease is not None:
                connection.execute(
                    "UPDATE autonomy_leases SET status = 'active', "
                    "stopped_reason = NULL, updated_at = ? "
                    "WHERE id = ? AND status = 'paused'",
                    (now, paused_lease["id"]),
                )
            connection.execute(
                "UPDATE measurement_sessions SET status = 'active', updated_at = ? "
                "WHERE workflow_id = ? AND status = 'paused'",
                (now, workflow_id),
            )
            self.db.event(
                "measurement_mode_resumed",
                client_id,
                {
                    "workflow_id": workflow_id,
                    "activation_phrase": activation_phrase,
                    "autonomy_lease_id": (
                        paused_lease["id"] if paused_lease is not None else None
                    ),
                    "resume_policy": "restart_current_node_as_new_run",
                },
                workflow_id,
                connection=connection,
            )
        result = self._decode_workflow(self._workflow(workflow_id))
        result["resume_policy"] = "restart_current_node_as_new_run"
        result["authorization"] = (
            self.autonomy_status(lease_id=str(paused_lease["id"]))
            if paused_lease is not None
            else None
        )
        session_id = self.dashboard_session_id_for_workflow(workflow_id)
        result["dashboard"] = (
            self.dashboard_status(session_id) if session_id is not None else None
        )
        return result

    def reopen_07b_after_morphology_rule_change(
        self,
        workflow_id: str,
        run_id: str,
        reason: str,
        client_id: str,
    ) -> dict[str, Any]:
        """Backtrack only one node when accepted 07b evidence is now invalid."""
        if not client_id.strip() or not reason.strip():
            raise ServiceError("07b reopening requires actor and reason")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active" or workflow["current_node"] != "06":
            raise ServiceError("07b may be reopened only from the immediately following 06")
        active_run = self.db.one(
            "SELECT id FROM runs WHERE workflow_id = ? "
            "AND status IN ('starting', 'running', 'stopping') LIMIT 1",
            (workflow_id,),
        )
        if active_run is not None:
            raise ServiceError("Wait for or stop the active run before reopening 07b")
        run = self.db.one(
            "SELECT * FROM runs WHERE id = ? AND workflow_id = ? AND node_id = '07b'",
            (run_id, workflow_id),
        )
        if (
            run is None
            or run["status"] != "completed"
            or run["analysis_status"] == "pass"
        ):
            raise ServiceError(
                "Reopening requires a completed 07b run reanalyzed as non-passing"
            )
        analysis = json_loads(run.get("analysis_json"), {})
        morphology = analysis.get("dataset_metrics", {}).get("qubits", {})
        targets = json_loads(run.get("parameters_json"), {}).get("qubits", [])
        invalid = [
            name
            for name in targets
            if isinstance(morphology.get(name), dict)
            and morphology[name].get("morphology_pass") is False
        ]
        if not invalid:
            raise ServiceError("Reanalyzed 07b run has no failed cloud morphology evidence")

        leases = self.db.all(
            "SELECT * FROM autonomy_leases WHERE workflow_id = ? "
            "AND status IN ('pending', 'active', 'paused')",
            (workflow_id,),
        )
        for lease in leases:
            lease = self._expire_autonomy_if_needed(lease)
            if lease["status"] in {"active", "paused"}:
                raise ServiceError(
                    "Stop the active/paused autonomy lease before reopening 07b"
                )
            if lease["status"] == "pending":
                self.db.execute(
                    "UPDATE autonomy_leases SET status = 'revoked', stopped_reason = ?, "
                    "updated_at = ? WHERE id = ?",
                    (reason.strip(), utc_now(), lease["id"]),
                )
                self.db.execute(
                    "UPDATE proposals SET status = 'cancelled' WHERE id = ? "
                    "AND status = 'pending'",
                    (lease["proposal_id"],),
                )
        self.db.execute(
            "UPDATE workflows SET current_node = '07b', updated_at = ? WHERE id = ?",
            (utc_now(), workflow_id),
        )
        result = {
            "workflow_id": workflow_id,
            "reopened_node": "07b",
            "invalidated_run_id": run_id,
            "morphology_failed_targets": invalid,
            "reason": reason.strip(),
        }
        self.db.event("07b_reopened", client_id, result, workflow_id)
        result["workflow"] = self._decode_workflow(self._workflow(workflow_id))
        return result

    def reopen_07d_after_readout_rule_change(
        self,
        workflow_id: str,
        reason: str,
        client_id: str,
    ) -> dict[str, Any]:
        """Return to 07d after the readout-selection or 07b blob rule changed.

        Operator instruction 2026-10-02. Allowed only from 07b or 06, with no
        run active and no active or paused autonomy lease, so a new lease is
        approved for the repeated readout nodes. 07d and 07b runs from before
        the reopen no longer count toward single pass or target resolution.
        """
        if not client_id.strip() or not reason.strip():
            raise ServiceError("07d reopening requires actor and reason")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active" or workflow["current_node"] not in {"07b", "06"}:
            raise ServiceError("07d may be reopened only from 07b or 06")
        active_run = self.db.one(
            "SELECT id FROM runs WHERE workflow_id = ? "
            "AND status IN ('starting', 'running', 'stopping') LIMIT 1",
            (workflow_id,),
        )
        if active_run is not None:
            raise ServiceError("Wait for or stop the active run before reopening 07d")
        for lease in self.db.all(
            "SELECT * FROM autonomy_leases WHERE workflow_id = ? "
            "AND status IN ('pending', 'active', 'paused')",
            (workflow_id,),
        ):
            lease = self._expire_autonomy_if_needed(lease)
            if lease["status"] in {"active", "paused"}:
                raise ServiceError(
                    "Stop the active/paused autonomy lease before reopening 07d"
                )
        previous = workflow["current_node"]
        self.db.execute(
            "UPDATE workflows SET current_node = '07d', updated_at = ? WHERE id = ?",
            (utc_now(), workflow_id),
        )
        result = {
            "workflow_id": workflow_id,
            "reopened_node": "07d",
            "previous_node": previous,
            "reason": reason.strip(),
        }
        self.db.event("07d_reopened", client_id, result, workflow_id)
        result["workflow"] = self._decode_workflow(self._workflow(workflow_id))
        return result

    def stop_workflow(
        self,
        workflow_id: str,
        client_id: str,
        reason: str,
    ) -> dict[str, Any]:
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        if not reason.strip():
            raise ServiceError("A stop reason is required")
        workflow = self._workflow(workflow_id)
        if workflow["status"] not in {"active", "paused"}:
            raise ServiceError("Only an active or paused workflow can be stopped")
        active_run = self.db.one(
            """
            SELECT id FROM runs
            WHERE workflow_id = ?
              AND status IN ('starting', 'running', 'stopping')
            LIMIT 1
            """,
            (workflow_id,),
        )
        if active_run is not None:
            raise ServiceError(
                "Wait for or stop the active run before stopping the workflow"
            )
        recovery_required = self.settings.lock_path.exists()
        final_status = "recovery_required" if recovery_required else "stopped"
        self.db.execute(
            """
            UPDATE workflows SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (final_status, utc_now(), workflow_id),
        )
        self.db.execute(
            "UPDATE autonomy_leases SET status = 'revoked', stopped_reason = ?, "
            "updated_at = ? WHERE workflow_id = ? "
            "AND status IN ('pending', 'active', 'paused')",
            (reason.strip(), utc_now(), workflow_id),
        )
        self.db.execute(
            "UPDATE proposals SET status = 'cancelled' WHERE workflow_id = ? "
            "AND status = 'pending'",
            (workflow_id,),
        )
        self.db.execute(
            "UPDATE measurement_sessions SET status = ?, updated_at = ? "
            "WHERE workflow_id = ? AND status != 'stopped'",
            (final_status, utc_now(), workflow_id),
        )
        self.db.event(
            "workflow_quarantined" if recovery_required else "workflow_stopped",
            client_id,
            {
                "workflow_id": workflow_id,
                "reason": reason.strip(),
                "recovery_required": recovery_required,
            },
            workflow_id,
        )
        return self._decode_workflow(self._workflow(workflow_id))

    def prepare_recovery_close(self, actor: str) -> dict[str, Any]:
        """Safely converge stale JY state before the lifecycle wrapper stops services.

        This operation never clears a retained hardware lock.  It requests a
        cooperative stop from a live, authenticated worker and otherwise closes
        orphaned workflows, leases, proposals, and Dashboard sessions so a
        subsequent service stop is deterministic.
        """

        if not actor.strip():
            raise ServiceError("Recovery close requires an audit actor")
        active_run = self.db.one(
            "SELECT * FROM runs WHERE status IN ('starting','running','stopping') "
            "ORDER BY started_at DESC LIMIT 1"
        )
        if active_run is not None:
            pid = active_run.get("pid")
            if isinstance(pid, int) and self.runner.process_is_alive(pid):
                try:
                    if active_run["status"] != "stopping":
                        self.runner.stop(
                            str(active_run["id"]),
                            actor,
                            intent="full_shutdown",
                            reason="Recover command requested a fully closed JY state.",
                        )
                except Exception as exc:
                    return {
                        "status": "blocked",
                        "reason": (
                            "The active worker could not receive a verified cooperative "
                            f"stop request: {exc}"
                        ),
                        "run_id": active_run["id"],
                        "services_may_stop": False,
                    }
                return {
                    "status": "waiting_for_worker",
                    "reason": "A verified worker is stopping cooperatively.",
                    "run_id": active_run["id"],
                    "services_may_stop": False,
                }

            evidence = self.runner.exit_evidence(str(active_run["id"]))
            stop_intent = str(
                active_run.get("stop_intent")
                or evidence.get("stop_request", {}).get("intent")
                or ""
            )
            terminal_status = (
                "cancelled_by_shutdown" if stop_intent == "full_shutdown" else "failed"
            )
            termination_cause = str(
                evidence.get("exit_receipt", {}).get("termination_cause")
                or "worker_missing_during_recovery_close"
            )
            error = active_run.get("error") or (
                "The recorded worker is no longer alive; recovery close converted "
                "the stale run record to a terminal state."
            )
            self.db.execute(
                "UPDATE runs SET status = ?, finished_at = COALESCE(finished_at, ?), "
                "error = COALESCE(error, ?), termination_cause = ? WHERE id = ?",
                (
                    terminal_status,
                    utc_now(),
                    error,
                    termination_cause,
                    active_run["id"],
                ),
            )
            self.db.event(
                "stale_run_closed_by_recovery",
                actor,
                {
                    "run_id": active_run["id"],
                    "status": terminal_status,
                    "termination_cause": termination_cause,
                },
                active_run["workflow_id"],
            )

        recovery_required = self.settings.lock_path.exists()
        final_status = "recovery_required" if recovery_required else "stopped"
        now = utc_now()
        workflows = self.db.all(
            "SELECT id FROM workflows WHERE status IN "
            "('active','paused','recovery_required')"
        )
        with self.db.transaction(immediate=True) as connection:
            for workflow in workflows:
                workflow_id = str(workflow["id"])
                connection.execute(
                    "UPDATE workflows SET status = ?, updated_at = ? WHERE id = ?",
                    (final_status, now, workflow_id),
                )
                connection.execute(
                    "UPDATE measurement_sessions SET status = ?, updated_at = ? "
                    "WHERE workflow_id = ? AND status != 'stopped'",
                    (final_status, now, workflow_id),
                )
                connection.execute(
                    "UPDATE autonomy_leases SET status = 'revoked', stopped_reason = ?, "
                    "updated_at = ? WHERE workflow_id = ? "
                    "AND status IN ('pending','active','paused')",
                    (
                        "Recover command closed the JY measurement environment.",
                        now,
                        workflow_id,
                    ),
                )
                connection.execute(
                    "UPDATE proposals SET status = 'cancelled' WHERE workflow_id = ? "
                    "AND status = 'pending'",
                    (workflow_id,),
                )
                self.db.event(
                    "workflow_quarantined"
                    if recovery_required
                    else "workflow_stopped",
                    actor,
                    {
                        "workflow_id": workflow_id,
                        "reason": "Recover command requested a fully closed state.",
                        "recovery_required": recovery_required,
                    },
                    workflow_id,
                    connection=connection,
                )
        return {
            "status": "recovery_required" if recovery_required else "ready",
            "reason": (
                "No worker is alive. Services may close; the retained hardware "
                "lock remains in local safety quarantine."
                if recovery_required
                else "No worker is alive and all open JY workflow state is closed."
            ),
            "services_may_stop": True,
            "hardware_lock_retained": recovery_required,
            "closed_workflow_ids": [str(item["id"]) for item in workflows],
        }

    def request_run(
        self,
        workflow_id: str,
        node_id: str,
        parameters: dict[str, Any],
        reason: str,
        client_id: str,
        auto_retry_count: int = 0,
        autonomy_lease_id: str | None = None,
        *,
        conversational: bool = False,
    ) -> dict[str, Any]:
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        if not reason.strip():
            raise ServiceError("A run proposal reason is required")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active":
            raise ServiceError("Workflow is not active")
        autonomy_lease = (
            self._active_autonomy(autonomy_lease_id, workflow_id)
            if autonomy_lease_id is not None
            else None
        )
        if conversational and autonomy_lease_id is not None:
            raise ServiceError(
                "A conversational run cannot be delegated to an autonomy lease"
            )
        if conversational:
            lease_row = self.db.one(
                "SELECT * FROM autonomy_leases WHERE workflow_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (workflow_id,),
            )
            if lease_row is not None:
                lease = self.autonomy_status(lease_id=lease_row["id"])
                if lease["status"] in {"pending", "active", "paused"}:
                    raise ServiceError(
                        "Conversational runs cannot share a live autonomy lease"
                    )
        # 2026-10-14 (q3 05 -> 06): requesting the pass's next node on a
        # finished node is the advance; check it before the active-reset
        # verification path, which also matches 05/06/06b.
        advances_completed_node = (
            not conversational
            and node_id != workflow["current_node"]
            and self._node_is_finished_and_fully_decided(workflow, node_id)
        )
        prerequisite_verification = (
            not conversational
            and not advances_completed_node
            and node_id != workflow["current_node"]
            and self._is_active_reset_verification_node(workflow, node_id)
        )
        if (
            not conversational
            and node_id != workflow["current_node"]
            and not prerequisite_verification
            and not advances_completed_node
        ):
            raise ServiceError(
                f"Current workflow node is {workflow['current_node']}; record a decision first"
            )
        if advances_completed_node:
            previous_node = str(workflow["current_node"])
            serial = self._serial_pass_state(workflow)
            if serial is not None and serial["current"] is not None:
                transition = self._pass_next_node(
                    workflow, previous_node, str(serial["current"])
                )[1]
                last_run = self.db.one(
                    "SELECT * FROM runs WHERE workflow_id = ? AND node_id = ? "
                    "ORDER BY rowid DESC LIMIT 1",
                    (workflow_id, previous_node),
                )
                if transition in {"return", "end"} and last_run is not None:
                    self._record_pass_transition(
                        workflow, last_run, str(serial["current"]), transition, client_id
                    )
            self.db.execute(
                "UPDATE workflows SET current_node = ?, updated_at = ? "
                "WHERE id = ? AND status = 'active'",
                (node_id, utc_now(), workflow_id),
            )
            self.db.event(
                "workflow_node_advanced_after_boundary",
                client_id,
                {"from_node": previous_node, "to_node": node_id},
                workflow_id,
            )
            workflow = self._workflow(workflow_id)
        maximum_retries = int(
            self.policy.raw["approval"]["max_auto_retries_per_approval"]
        )
        if (
            not isinstance(auto_retry_count, int)
            or isinstance(auto_retry_count, bool)
            or not 0 <= auto_retry_count <= maximum_retries
        ):
            raise ServiceError(
                f"auto_retry_count must be between 0 and {maximum_retries}"
            )
        workflow_parameters = json_loads(
            workflow["initial_parameters_json"], {}
        )
        workflow_targets = set(json_loads(workflow["targets_json"], []))
        definition = self.policy.node_definition(node_id)
        supports_multiplexed = "multiplexed" in definition.get(
            "allowed_parameters", {}
        )
        if workflow_parameters.get("multiplexed") is True and supports_multiplexed:
            if parameters.get("multiplexed") is False:
                raise ServiceError(
                    "This workflow requires multiplexed=true for every node"
                )
            parameters = {**parameters, "multiplexed": True}
        # Operator request 2026-10-07: values typed on the lease page (02:
        # res_num, res_design_freq) win over whatever the agent proposed.
        operator_parameters = (
            self._workflow_options(workflow).get("node_parameters") or {}
        ).get(node_id) or {}
        parameters = {**parameters, **operator_parameters}
        try:
            merged, warnings = self.policy.validate_run(
                node_id, parameters, drive_mode=self._drive_mode(workflow)
            )
        except PolicyError as exc:
            # Refused before anything reached the instrument, so the refusal is
            # itself the protection. Record it and keep the lease usable; the
            # agent corrects the parameter and proposes again.
            if autonomy_lease_id is not None:
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "parameter_policy_violation",
                    f"{node_id}: {exc}",
                )
            raise
        if prerequisite_verification and not _is_active_reset(merged):
            raise ServiceError(
                "An off-sequence 05/06/06b verification run is allowed only "
                "for active-reset revalidation after 07b qualification"
            )
        requested_targets = set(merged["qubits"])
        try:
            if conversational:
                workflow_targets = set(json_loads(workflow["targets_json"], []))
                if not requested_targets or not requested_targets <= workflow_targets:
                    raise ServiceError(
                        "Conversational run qubits must be a non-empty subset of "
                        "the workflow targets"
                    )
            elif node_id in SUBGROUP_NODES:
                eligible_targets = self._active_targets_for_node(workflow, node_id)
                if not requested_targets or not requested_targets <= eligible_targets:
                    raise AutonomyScopeError(
                        f"{node_id} run qubits must be a non-empty subset of active "
                        f"workflow targets {sorted(eligible_targets)}"
                    )
            elif node_id in _after_readout_line_nodes(self.settings.workflow_sequence):
                expected_targets = self._active_targets_for_node(workflow, node_id)
                if (
                    self._node_is_single_pass(node_id)
                    and self._node_multiplex_cap(node_id) is not None
                ):
                    # Operator instruction 2026-09-29: a capped single-pass node
                    # covers its targets in groups. It stays out of
                    # SUBGROUP_NODES so the targets carried downstream are not
                    # narrowed to its per-target evidence.
                    if not requested_targets or not (
                        requested_targets <= expected_targets
                    ):
                        raise AutonomyScopeError(
                            f"{node_id} run qubits must be a non-empty subset of "
                            f"the active targets {sorted(expected_targets)}"
                        )
                elif requested_targets != expected_targets:
                    raise ServiceError(
                        "Run qubits must exactly match the resolved active targets "
                        "carried into this node"
                    )
            elif requested_targets != workflow_targets:
                raise ServiceError(
                    "Run qubits must exactly match the workflow targets; start a "
                    "separate workflow for a different target set"
                )
            if not conversational and not prerequisite_verification:
                self._validate_single_pass_node(workflow, node_id, merged)
                self._validate_reset_group(workflow, node_id, merged)
            if (
                workflow_parameters.get("multiplexed") is True
                and not prerequisite_verification
            ):
                self._validate_multiplex_cap(node_id, requested_targets)
                self._validate_shared_parameter_first_batch(
                    workflow, node_id, requested_targets
                )
                # The registered 07b active-reset repeat deliberately remeasures
                # targets that passed the thermal run (_validate_single_pass_node
                # already admits it); 2026-10-03 fix for the planner/guard clash.
                if not self._is_active_reset_repeat(workflow, node_id, merged):
                    self._validate_unresolved_retry_targets(
                        workflow, node_id, requested_targets
                    )
                warnings.extend(
                    self._shared_parameter_retry_warnings(
                        workflow, node_id, requested_targets
                    )
                )
            if not prerequisite_verification:
                fresh = sorted(
                    set(self._fresh_qubits(workflow)) & set(merged["qubits"])
                )
                if fresh:
                    raise AutonomyScopeError(
                        f"{', '.join(fresh)} have no measurement record on this "
                        "chip; apply jy_request_fresh_qubit_defaults first"
                    )
            self._validate_active_reset_prerequisite(workflow, node_id, merged)
            self._validate_node_prerequisites(workflow, node_id, merged)
            if not conversational and not prerequisite_verification:
                lo_patch, lo_details = self._xy_lo_plan(workflow, node_id)
                if lo_patch:
                    raise AutonomyScopeError(
                        "Apply jy_request_xy_lo_plan first: the shared XY LO is "
                        f"not yet placed for {lo_details.get('focus')} "
                        f"({float(lo_details.get('lo_hz', 0.0)) / 1e9:g} GHz)."
                    )
            if node_id == "03a":
                self._validate_03a_run_request(workflow, merged)
            if autonomy_lease_id is not None:
                self._validate_autonomy_scope(
                    autonomy_lease, node_id=node_id, targets=list(merged["qubits"])
                )
                self._assert_autonomy_attempt_quota(
                    autonomy_lease, node_id, list(merged["qubits"])
                )
        except AutonomyQuotaError:
            raise
        except AutonomyScopeError as exc:
            # Target-set / stage planning mistakes have not touched hardware.
            # Keep the lease usable so the agent can resubmit the unresolved
            # subgroup instead of converting a recoverable error into halt.
            if autonomy_lease_id is not None:
                self.db.event(
                    "autonomy_action_rejected",
                    client_id,
                    {
                        "lease_id": autonomy_lease_id,
                        "node_id": node_id,
                        "targets": sorted(requested_targets),
                        "reason": str(exc),
                    },
                    workflow_id,
                )
            raise
        script = self.policy.node_script(node_id)
        definition = self.policy.node_definition(node_id)
        payload = {
            "node_id": node_id,
            "node_name": definition["name"],
            "script_path": str(script),
            "parameters": merged,
            "reason": reason.strip(),
            "warnings": warnings,
            "conversational": conversational,
            "prerequisite_verification": prerequisite_verification,
        }
        payload["duration_prediction"] = self.run_telemetry(
            node_id, list(merged["qubits"]), limit=100
        )["prediction"]
        payload["prior_decision_experience"] = self.decision_experience(
            node_id, list(merged["qubits"]), limit=5
        )
        review_run_id = self._latest_review_run_id(workflow_id, node_id)
        if review_run_id is not None:
            payload["review_run_id"] = review_run_id
        proposal = self._create_proposal(
            workflow_id=workflow_id,
            kind="run",
            payload=payload,
            source_client=client_id,
            ttl_minutes=int(self.policy.raw["approval"]["run_ttl_minutes"]),
            max_uses=1 + auto_retry_count,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id is not None:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        return proposal

    def request_conversational_run(
        self,
        workflow_id: str,
        node_id: str,
        parameters: dict[str, Any],
        reason: str,
        client_id: str,
        auto_retry_count: int = 0,
    ) -> dict[str, Any]:
        """Propose one policy-registered experiment outside automatic sequencing."""
        return self.request_run(
            workflow_id,
            node_id,
            parameters,
            reason,
            client_id,
            auto_retry_count,
            conversational=True,
        )

    def start_authorized_run(
        self,
        workflow_id: str,
        lease_id: str,
        node_id: str,
        parameters: dict[str, Any],
        reason: str,
        client_id: str,
    ) -> dict[str, Any]:
        """Validate, delegate, and launch one run under a human-approved lease."""
        try:
            proposal = self.request_run(
                workflow_id,
                node_id,
                parameters,
                reason,
                client_id,
                auto_retry_count=0,
                autonomy_lease_id=lease_id,
            )
            result = self.execute_run(proposal["id"], client_id)
            return {**result, "proposal_id": proposal["id"], "lease_id": lease_id}
        except AutonomyQuotaError:
            # Exhausting one node/qubit budget marks only that target incomplete.
            # The lease remains usable for other targets and downstream nodes.
            raise
        except AutonomyScopeError:
            # Recoverable target-set / stage planning error; lease stays active.
            raise
        except (PolicyError, RunnerError, ServiceError, ValueError) as exc:
            if self._is_recoverable_scheduling_block(exc):
                raise
            self._halt_autonomy(
                lease_id,
                f"Authorized run was refused or failed to launch: {exc}",
                "autonomy_guard",
                stop_active=True,
            )
            raise

    def execute_run(self, proposal_id: str, client_id: str) -> dict[str, Any]:
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        proposal = self.db.one("SELECT * FROM proposals WHERE id = ?", (proposal_id,))
        if proposal is None:
            raise ServiceError(f"Unknown proposal: {proposal_id}")
        if self._workflow(proposal["workflow_id"])["status"] != "active":
            raise ServiceError("Resume measurement mode before executing a run")
        lease_id = proposal.get("autonomy_lease_id")
        if lease_id:
            lease = self._active_autonomy(str(lease_id), proposal["workflow_id"])
            payload = json_loads(proposal["payload_json"], {})
            parameters = payload.get("parameters", {})
            targets = list(parameters.get("qubits", []))
            try:
                self._validate_autonomy_scope(
                    lease, node_id=str(payload.get("node_id", "")), targets=targets
                )
                self._assert_autonomy_attempt_quota(
                    lease, str(payload.get("node_id", "")), targets
                )
            except AutonomyQuotaError:
                raise
            except AutonomyScopeError as exc:
                # Same rule as `start_authorized_run`: the run is refused before
                # the worker launches, so the lease stays usable.
                self._record_refused_autonomy_action(
                    str(lease_id),
                    proposal["workflow_id"],
                    "autonomy_scope",
                    str(exc),
                )
                raise
        try:
            return self.runner.start_approved_run(proposal_id, client_id)
        except Exception as exc:
            if lease_id:
                self._halt_autonomy(
                    str(lease_id),
                    f"Worker launch or hardware lock failure: {exc}",
                    "autonomy_guard",
                    stop_active=True,
                )
            raise

    def poll_run(self, run_id: str) -> dict[str, Any]:
        run = self.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
        if run is None:
            raise ServiceError(f"Unknown run: {run_id}")
        if run["status"] == "failed" and run["analysis_status"] == "not_started":
            failure = {
                "analysis_status": "failed",
                "node_id": run["node_id"],
                "snapshot_id": run.get("snapshot_id"),
                "snapshot_path": run.get("snapshot_path"),
                "plots": [],
                "failure_reasons": [
                    "Execution failed before a usable snapshot was produced."
                ],
                "warnings": [],
                "candidate_state_patch": [],
                "error": run.get("error"),
                "active_state_hash": run.get("active_state_hash_after"),
            }
            self.db.execute(
                "UPDATE runs SET analysis_status = 'failed', analysis_json = ? "
                "WHERE id = ? AND analysis_status = 'not_started'",
                (json_dumps(failure), run_id),
            )
            run = self.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
        if self._is_instrument_connectivity_failure(run):
            self._pause_for_instrument_connectivity(run)
            run = self.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
        if run.get("autonomy_lease_id"):
            self._enforce_run_hard_stops(run)
            run = self.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
        decoded = self._decode_run(run)
        if self._is_instrument_connectivity_failure(run):
            decoded["operator_handoff"] = instrument_operator_handoff()
        else:
            submission = self._program_submission_timeout(run)
            if submission is not None and submission.get("measurement_paused"):
                decoded["operator_handoff"] = (
                    submission_escalation_operator_handoff()
                )
        return decoded

    def stop_run(self, run_id: str, client_id: str) -> dict[str, Any]:
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        return self.runner.stop(run_id, client_id)

    def analyze_run(self, run_id: str) -> dict[str, Any]:
        run = self.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
        if run is None:
            raise ServiceError(f"Unknown run: {run_id}")
        if run["status"] != "completed":
            raise ServiceError("Only a completed run can be analyzed")
        if run["node_id"] == "07d":
            run = {
                **run,
                "readout_power_limits_dbm": self._dressed_power_limits(
                    run["workflow_id"]
                ),
            }
        if run["node_id"] == "03a":
            # Operator rule 2026-10-12: lines 04 rejected stay masked.
            run = {
                **run,
                "rejected_03a_lines": self._rejected_03a_lines(run["workflow_id"]),
                # 2026-10-12: lines two LOs confirmed get the cross-LO fine floor.
                "cross_lo_03a_lines": self._03a_cross_lo_candidates(run["workflow_id"]),
            }
        result = self.analyzer.analyze_run(run)
        if run["node_id"] == "03a":
            result = self._require_reproduced_03a_line(run, result)
        self.db.execute(
            """
            UPDATE runs SET analysis_status = ?, analysis_json = ? WHERE id = ?
            """,
            (result["analysis_status"], json_dumps(result), run_id),
        )
        self.db.event(
            "run_analyzed",
            "deterministic_analyzer",
            {"run_id": run_id, "analysis_status": result["analysis_status"]},
            run["workflow_id"],
        )
        if result.get("passing_targets"):
            # Operator request 2026-10-13: the static chip history follows
            # every success, so it is current even after shutdown.
            self._refresh_static_chip_history(str(run["workflow_id"]))
        return result

    def _require_reproduced_03a_line(
        self, run: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, Any]:
        """A fine 03a line commits only if an earlier 03a run showed it too.

        Operator instruction 2026-10-12: the 03a peak must be reproducible.
        Another 03a run of the same qubit (coarse, refinement or fine, after
        its latest return from 04) must have put a feature with at least
        `reproduce_min_robust_snr` within `reproduce_tolerance_mhz` (or this
        fine FWHM, if wider) of the fine line.
        """

        if result.get("03a_stage") != "fine":
            return result
        rules = self.policy.raw["analysis"]["03a"]
        tolerance = 1e6 * float(rules.get("reproduce_tolerance_mhz", 5.0))
        min_snr = float(rules.get("reproduce_min_robust_snr", 4.5))
        results = result.get("fit_quality", {}).get("results", {})
        metrics_by_qubit = result.get("dataset_metrics", {}).get("qubits", {})
        unconfirmed: dict[str, str] = {}
        for name in list(result.get("passing_targets") or []):
            fit = results.get(name, {}) if isinstance(results, dict) else {}
            metrics = metrics_by_qubit.get(name, {}) if isinstance(metrics_by_qubit, dict) else {}
            line = fit.get("drive_freq") if isinstance(fit, dict) else None
            if not _finite_number(line):
                continue
            width = metrics.get("feature_fwhm_hz") if isinstance(metrics, dict) else None
            window = max(tolerance, float(width) if _finite_number(width) else 0.0)
            seen = []
            for row in self._qubit_runs(str(run["workflow_id"]), "03a", str(name)):
                if row["id"] == run["id"]:
                    continue
                earlier = json_loads(row.get("analysis_json"), {})
                earlier_fit = earlier.get("fit_quality", {}).get("results", {}).get(name, {})
                earlier_metrics = earlier.get("dataset_metrics", {}).get("qubits", {}).get(name, {})
                frequency = earlier_fit.get("drive_freq") if isinstance(earlier_fit, dict) else None
                snr = earlier_metrics.get("robust_snr") if isinstance(earlier_metrics, dict) else None
                if _finite_number(frequency):
                    seen.append(float(frequency))
                    if abs(float(frequency) - float(line)) <= window and (
                        _finite_number(snr) and float(snr) >= min_snr
                    ):
                        break
            else:
                unconfirmed[str(name)] = (
                    f"{name} line {float(line) / 1e9:.6g} GHz is not reproduced: no "
                    f"earlier 03a run of {name} showed a feature within "
                    f"{window / 1e6:.3g} MHz with robust SNR >= {min_snr:g}"
                    + (
                        f" (earlier: {', '.join(f'{value / 1e9:.6g}' for value in seen)} GHz)"
                        if seen
                        else ""
                    )
                    + ". Repeat the fine scan to confirm it."
                )
        if not unconfirmed:
            return result
        result = dict(result)
        result["passing_targets"] = [
            name for name in result.get("passing_targets") or [] if name not in unconfirmed
        ]
        for name, message in unconfirmed.items():
            fit = results.get(name)
            if isinstance(fit, dict):
                fit["fit_successful"] = False
                fit["reproducibility"] = "unconfirmed"
            result.setdefault("failure_reasons", []).append(message)
            result.setdefault("failure_reasons_by_target", {}).setdefault(name, []).append(message)
        result["candidate_state_patch"] = [
            item
            for item in result.get("candidate_state_patch") or []
            if not any(
                str(item.get("path", "")).startswith(f"/qubits/{name}/")
                for name in unconfirmed
            )
        ]
        if result.get("analysis_status") == "pass":
            result["analysis_status"] = "needs_review"
        return result

    def analyze_authorized_run(
        self, run_id: str, lease_id: str, client_id: str
    ) -> dict[str, Any]:
        run = self.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
        if run is None or run.get("autonomy_lease_id") != lease_id:
            raise ServiceError("Run is not associated with this autonomy lease")
        self._active_autonomy(lease_id, run["workflow_id"])
        if run["status"] != "completed":
            # Refuse a premature analyze before the watchdog runs. Enforcing
            # hardware-anomaly rules as a side effect of a timing mistake used
            # to end the lease over a call that did nothing.
            self._record_refused_autonomy_action(
                lease_id,
                run["workflow_id"],
                "analysis_precondition",
                f"Run {run_id} is {run['status']}, not completed",
            )
            raise ServiceError(
                f"Only a completed run can be analyzed; run {run_id} is "
                f"{run['status']}"
            )
        self._enforce_run_hard_stops(run)
        try:
            return self.analyze_run(run_id)
        except ServiceError as exc:
            # A precondition the agent got wrong -- most often analyzing a run
            # that has not reached a terminal state yet. Nothing was consumed,
            # so refuse it and let the agent wait and call again.
            self._record_refused_autonomy_action(
                lease_id, run["workflow_id"], "analysis_precondition", str(exc)
            )
            raise
        except Exception as exc:
            # An unreadable or unusable snapshot is the configured
            # `snapshot_missing` hard stop; anything else is an unknown
            # analyzer failure, which pauses for the operator.
            if isinstance(exc, AnalysisError):
                self._halt_autonomy(
                    lease_id,
                    f"Snapshot analysis could not read run {run_id}: {exc}",
                    client_id,
                    stop_active=False,
                )
            else:
                self._pause_autonomy_for_review(
                    run,
                    f"Snapshot analysis raised an unexpected exception for run "
                    f"{run_id}: {exc}",
                    "autonomy_paused_for_analysis_failure",
                    client_id,
                )
            raise

    def record_decision(
        self,
        workflow_id: str,
        run_id: str,
        decision: str,
        reason: str,
        next_node: str | None,
        new_parameters: dict[str, Any] | None,
        state_patch: list[dict[str, Any]] | None,
        client_id: str,
    ) -> dict[str, Any]:
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active":
            raise ServiceError("Resume measurement mode before recording a decision")
        run = self.db.one(
            "SELECT * FROM runs WHERE id = ? AND workflow_id = ?",
            (run_id, workflow_id),
        )
        if run is None:
            raise ServiceError("Run does not belong to this workflow")
        proposal = self.db.one(
            "SELECT payload_json FROM proposals WHERE id = ?",
            (run["proposal_id"],),
        )
        proposal_payload = json_loads(
            proposal["payload_json"] if proposal is not None else None, {}
        )
        conversational = bool(proposal_payload.get("conversational", False))
        prerequisite_verification = bool(
            proposal_payload.get("prerequisite_verification", False)
        )
        if (
            not conversational
            and not prerequisite_verification
            and run["node_id"] != workflow["current_node"]
        ):
            raise ServiceError(
                f"Run node {run['node_id']} does not match current workflow node "
                f"{workflow['current_node']}"
            )
        if self.db.one("SELECT id FROM decisions WHERE run_id = ?", (run_id,)):
            raise ServiceError("A decision has already been recorded for this run")
        if run["analysis_status"] not in {"pass", "needs_review", "failed"}:
            raise ServiceError("Analyze the run before recording a decision")
        allowed_decisions = {"advance", "repeat", "manual_review", "stop"}
        if decision not in allowed_decisions:
            raise ServiceError(f"Decision must be one of {sorted(allowed_decisions)}")
        if not reason.strip():
            raise ServiceError("Decision reason is required")

        current = workflow["current_node"]
        sequence = self._workflow_sequence(workflow)
        resolved_next: str | None
        pass_transition: str | None = None
        if conversational:
            if next_node is not None:
                self.policy.node_definition(next_node)
            resolved_next = next_node
        elif prerequisite_verification:
            if run["node_id"] not in self._active_reset_verification_nodes():
                raise ServiceError("Invalid active-reset prerequisite verification node")
            if not _is_active_reset(json_loads(run.get("parameters_json"), {})):
                raise ServiceError("Prerequisite verification must use active reset")
            if decision == "advance":
                rows = [
                    {
                        **run,
                        "decision": "advance",
                    }
                ]
                resolved = self._resolve_targets_from_rows(run["node_id"], rows)
                if not resolved:
                    raise ServiceError(
                        "Cannot accept active-reset verification without at least "
                        "one usable target result"
                    )
            if next_node not in {None, current}:
                raise ServiceError(
                    f"Prerequisite verification keeps the workflow at {current}"
                )
            resolved_next = None if decision == "stop" else current
        elif decision == "repeat":
            resolved_next = current
        elif decision == "advance":
            serial = self._serial_pass_state(workflow)
            in_serial_tail = serial is not None and sequence.index(
                current
            ) >= sequence.index(serial["node"])
            if current in SUBGROUP_NODES:
                resolved_targets = self._node_target_resolution(
                    workflow_id, current
                )
                eligible_targets = self._active_targets_for_node(workflow, current)
                incomplete_targets = self._autonomy_incomplete_targets_for_node(
                    workflow_id, current
                )
                missing_targets = (
                    eligible_targets - resolved_targets - incomplete_targets
                )
                if self._node_is_single_pass(current):
                    # Operator choice 2026-10-03: a single-pass node runs once
                    # per target; a target measured there without passing
                    # carries no update and does not hold the workflow up.
                    missing_targets -= self._single_pass_measured_targets(
                        workflow, current
                    )
                if in_serial_tail and serial["current"] in missing_targets:
                    # Operator rules 2026-10-12: 05 may be left for its revisit
                    # after 07b, and a 04 without any Rabi may return to 03a.
                    pass_qubit = str(serial["current"])
                    if self._deferral_pending(workflow, current, pass_qubit) or (
                        current == "04" and self._04_return_check(workflow, pass_qubit)[0]
                    ):
                        missing_targets.discard(pass_qubit)
                if missing_targets:
                    raise ServiceError(
                        f"Cannot advance from {current} until every active workflow "
                        "target is resolved, is marked scientifically unmeasurable, "
                        "or reaches its autonomy attempt limit; "
                        f"missing {sorted(missing_targets)}"
                    )
                if (
                    not (eligible_targets & resolved_targets)
                    and not in_serial_tail
                    and not self._node_is_advisory(current)
                    and current not in FLUX_NODES
                ):
                    raise ServiceError(
                        f"Cannot advance from {current}: no target has usable "
                        "evidence for the next node"
                    )
                statistics_dependency = {
                    "05": "05st",
                    "06": "06st_t2star",
                    "06b": "06st_t2e",
                }.get(current)
                if statistics_dependency in sequence:
                    insufficient = {
                        target
                        for target in eligible_targets & resolved_targets
                        if not self._target_has_lifetime_coverage(
                            workflow_id, current, target, 3.5
                        )
                    }
                    if insufficient:
                        raise ServiceError(
                            f"Cannot advance from {current}: downstream 100-run "
                            "statistics require decay-to-equilibrium evidence covering "
                            f"at least 3.5 lifetimes for {sorted(insufficient)}"
                        )
            expected = self._expected_next_node(workflow, current)
            pass_transition = (
                self._pass_next_node(workflow, current, str(serial["current"]))[1]
                if in_serial_tail and serial["current"] is not None
                else None
            )
            if next_node != expected:
                raise ServiceError(
                    f"Next node after {current} must be {expected}"
                    if not in_serial_tail
                    else f"Next node after {current} must be {expected}: this "
                    f"chip runs one qubit at a time and {serial['current']} is "
                    + (
                        "still mid-pass"
                        if expected not in (None, serial["node"])
                        else "at the end of its pass"
                    )
                )
            resolved_next = expected
        elif decision == "manual_review":
            resolved_next = current
        else:
            resolved_next = None

        resolved_parameters = dict(new_parameters or {})
        parameter_adjustments: list[str] = []
        if resolved_next is not None and resolved_parameters:
            targets = (
                set(json_loads(workflow["targets_json"], []))
                if conversational
                else self._active_targets_for_node(workflow, resolved_next)
            )
            if (
                not conversational
                and not prerequisite_verification
                and decision == "advance"
                and current in SUBGROUP_NODES
            ):
                targets &= self._node_target_resolution(workflow_id, current)
            # A look-ahead parameter check, not a run request: a serialized
            # node will carry these targets one run at a time, so validate the
            # first of them rather than a batch the node would refuse.
            lookahead = sorted(targets)
            cap = self._shared_xy_drive_cap(resolved_next)
            if cap is not None and len(lookahead) > cap:
                lookahead = lookahead[:cap]
            self.policy.validate_run(
                resolved_next,
                {"qubits": lookahead, **resolved_parameters},
                drive_mode=self._drive_mode(workflow),
            )
        patch = state_patch or []
        if patch:
            self.policy.validate_state_patch(patch)

        decision_id = uuid.uuid4().hex
        terminal_advance = (
            not conversational
            and not prerequisite_verification
            and decision == "advance"
            and resolved_next is None
        )
        workflow_status = "stopped" if decision == "stop" else (
            "completed" if terminal_advance and not patch else "active"
        )
        workflow_next = (
            current
            if conversational or (prerequisite_verification and decision != "stop")
            else resolved_next
        )
        try:
            with self.db.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO decisions(
                        id, workflow_id, run_id, decision, reason, next_node,
                        next_parameters_json, state_patch_json, client_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision_id,
                        workflow_id,
                        run_id,
                        decision,
                        reason.strip(),
                        resolved_next,
                        json_dumps(resolved_parameters),
                        json_dumps(patch),
                        client_id,
                        utc_now(),
                    ),
                )
                connection.execute(
                    "UPDATE workflows SET current_node = ?, status = ?, "
                    "updated_at = ? WHERE id = ? AND status = 'active'",
                    (workflow_next, workflow_status, utc_now(), workflow_id),
                )
        except sqlite3.IntegrityError as exc:
            raise ServiceError("A decision has already been recorded for this run") from exc
        if decision == "advance" and pass_transition in {"return", "end"}:
            self._record_pass_transition(
                workflow, run, str(serial["current"]), pass_transition, client_id
            )
        if (
            not conversational
            and not prerequisite_verification
            and decision == "advance"
            and current in SUBGROUP_NODES
        ):
            incomplete_targets = self._autonomy_incomplete_targets_for_node(
                workflow_id, current
            )
            if incomplete_targets:
                self.db.event(
                    "autonomy_targets_incomplete",
                    client_id,
                    {
                        "node_id": current,
                        "targets": sorted(incomplete_targets),
                        "continued_targets": sorted(
                            self._node_target_resolution(workflow_id, current)
                        ),
                    },
                    workflow_id,
                )
        if workflow_status in {"completed", "stopped"}:
            lease_status = "completed" if workflow_status == "completed" else "revoked"
            self.db.execute(
                "UPDATE autonomy_leases SET status = ?, stopped_reason = ?, "
                "updated_at = ? WHERE workflow_id = ? "
                "AND status IN ('active', 'paused')",
                (
                    lease_status,
                    f"Workflow {workflow_status} by recorded decision.",
                    utc_now(),
                    workflow_id,
                ),
            )
        elif (
            not conversational
            and not prerequisite_verification
            and decision == "advance"
            and not patch
        ):
            self._complete_autonomy_scope_if_needed(
                workflow_id, resolved_next, client_id
            )
        result = {
            "decision_id": decision_id,
            "workflow_id": workflow_id,
            "run_id": run_id,
            "Decision": decision,
            "Reason": reason.strip(),
            "next_action": {
                "next_node": resolved_next,
                "new_parameters": resolved_parameters,
            },
            "candidate_state_patch": patch,
            "state_commit_required": bool(patch),
            "parameter_adjustments": parameter_adjustments,
            "mode": (
                "conversational"
                if conversational
                else "prerequisite_verification"
                if prerequisite_verification
                else "sequenced"
            ),
        }
        self.db.event("decision_recorded", client_id, result, workflow_id)
        return result

    def request_state_commit(
        self,
        workflow_id: str,
        patch: list[dict[str, Any]],
        reason: str,
        client_id: str,
        run_id: str | None = None,
        autonomy_lease_id: str | None = None,
        _trusted_setup: bool = False,
    ) -> dict[str, Any]:
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        if not reason.strip():
            raise ServiceError("A state-commit reason is required")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active":
            raise ServiceError("Resume measurement mode before proposing a state update")
        autonomy_lease = (
            self._active_autonomy(autonomy_lease_id, workflow_id)
            if autonomy_lease_id is not None
            else None
        )
        try:
            self.policy.validate_state_patch(patch)
        except PolicyError as exc:
            # The patch is rejected before any proposal exists, so `state.json`
            # is untouched. Refuse and keep the lease.
            if autonomy_lease_id is not None:
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "state_patch_policy_violation",
                    str(exc),
                )
            raise
        run = (
            self.db.one(
                "SELECT * FROM runs WHERE id = ? AND workflow_id = ?",
                (run_id, workflow_id),
            )
            if run_id is not None
            else None
        )
        workflow_targets = set(json_loads(workflow["targets_json"], []))
        preserved_paths = (
            self._power_preserving_line_rescale_paths(patch, workflow_targets)
            if run is not None and run["node_id"] == "02c" and autonomy_lease_id is None
            else set()
        )
        for item in patch:
            path = str(item.get("path", ""))
            match = re.fullmatch(r"/qubits/(q[0-9]+)/.*", path)
            if match is not None and match.group(1) in workflow_targets:
                continue
            if path in preserved_paths:
                continue
            if (
                run is not None
                and run["node_id"] == "02c"
                and self._readout_full_scale_owned_by(path, workflow_targets)
            ):
                continue
            raise ServiceError(
                "Every state patch path must belong to a workflow target"
            )
        if run_id is not None:
            if run is None:
                raise ServiceError("State proposal run does not belong to this workflow")
            if run["analysis_status"] not in {"pass", "needs_review", "failed"}:
                raise ServiceError("Analyze the run before proposing its state update")
        if autonomy_lease_id is not None:
            try:
                if _trusted_setup:
                    self._validate_autonomy_patch_targets(autonomy_lease, patch)
                else:
                    self._validate_autonomous_scientific_state_commit(
                        autonomy_lease, run_id, patch
                    )
            except AutonomyScopeError as exc:
                # Out-of-scope commit, refused before the proposal is created.
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "state_commit_scope",
                    str(exc),
                )
                raise
        payload = {
            "patch": patch,
            "reason": reason.strip(),
            "run_id": run_id,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "authorization_evidence": (
                "deterministic_setup"
                if _trusted_setup and autonomy_lease_id
                else "accepted_run"
                if autonomy_lease_id
                else "human_proposal"
            ),
        }
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id is not None:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        return proposal

    def commit_authorized_state(
        self,
        workflow_id: str,
        lease_id: str,
        patch: list[dict[str, Any]],
        reason: str,
        client_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Create and atomically apply a pass-backed state patch under a lease."""
        try:
            proposal = self.request_state_commit(
                workflow_id,
                patch,
                reason,
                client_id,
                run_id,
                autonomy_lease_id=lease_id,
            )
            result = self.apply_state_commit(proposal["id"], client_id)
            return {**result, "lease_id": lease_id}
        except (AutonomyEvidenceError, AutonomyScopeError, PolicyError):
            # Refused before `state.json` was written; the inner layer has
            # already recorded it and the lease stays usable.
            raise
        except (StateError, ValueError) as exc:
            # A conflict while applying: the state file may no longer match what
            # the evidence was derived from. `apply_state_commit` halts on its
            # own for a claimed proposal; halt here for the rest.
            self._halt_autonomy(
                lease_id,
                f"Authorized state commit conflicted while applying: {exc}",
                "autonomy_guard",
                stop_active=True,
            )
            raise

    def apply_authorized_setup_state(
        self, proposal_id: str, lease_id: str, client_id: str
    ) -> dict[str, Any]:
        """Apply a trusted bootstrap/LO setup proposal delegated by the lease."""
        proposal = self.db.one("SELECT * FROM proposals WHERE id = ?", (proposal_id,))
        if (
            proposal is None
            or proposal.get("autonomy_lease_id") != lease_id
            or proposal["kind"] != "state_commit"
        ):
            raise AutonomyScopeError(
                "Setup proposal is not bound to this autonomy lease"
            )
        payload = json_loads(proposal["payload_json"], {})
        if payload.get("authorization_evidence") != "deterministic_setup":
            raise AutonomyEvidenceError(
                "Only a server-generated deterministic setup may use this tool"
            )
        self._active_autonomy(lease_id, proposal["workflow_id"])
        try:
            result = self.apply_state_commit(proposal_id, client_id)
            return {**result, "lease_id": lease_id}
        except (PolicyError, AutonomyScopeError, AutonomyEvidenceError) as exc:
            self._record_refused_autonomy_action(
                lease_id,
                proposal["workflow_id"],
                "deterministic_setup_refused",
                str(exc),
            )
            raise
        except Exception as exc:
            # Anything else here means the write itself failed or conflicted.
            self._halt_autonomy(
                lease_id,
                f"Authorized setup commit failed: {exc}",
                "autonomy_guard",
                stop_active=True,
            )
            raise

    def request_bootstrap(
        self,
        workflow_id: str,
        qubits: list[str],
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active":
            raise ServiceError("Resume measurement mode before bootstrap")
        targets = set(json_loads(workflow["targets_json"], []))
        if not qubits or not set(qubits).issubset(targets):
            raise ServiceError("Bootstrap qubits must be workflow targets")
        state = load_state(self.settings.active_state)
        patch, details = bootstrap_patch(state, qubits, self.policy.raw)
        if not patch:
            return {
                "proposal_created": False,
                "message": "All selected x180/x90 amplitudes are already nonzero.",
                "details": details,
            }
        proposal = self.request_state_commit(
            workflow_id,
            patch,
            "Bootstrap missing/zero x180 at 50% of max_x180_wf_amplitude; "
            "set missing/zero x90 from x180/2.",
            client_id,
            autonomy_lease_id=autonomy_lease_id,
            _trusted_setup=bool(autonomy_lease_id),
        )
        proposal["bootstrap_details"] = details
        return proposal

    def request_07b_prerequisites(
        self,
        workflow_id: str,
        qubits: list[str],
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Initialize missing IQ-blob output keys before Qualibrate records them.

        Qualibrate's state-update recorder must read the old mapping value before
        07b assigns a measured fidelity. A zero sentinel is safe because it cannot
        pass 07b analysis and a successful run replaces it with measured evidence.
        """
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active" or workflow["current_node"] != "07b":
            raise ServiceError("07b prerequisites require an active 07b workflow")
        active_targets = self._active_targets_for_node(workflow, "07b")
        selected = set(qubits)
        if (
            not selected
            or len(selected) != len(qubits)
            or not selected.issubset(active_targets)
        ):
            raise ServiceError(
                "07b prerequisite qubits must be unique active workflow targets"
            )

        state = load_state(self.settings.active_state)
        patch: list[dict[str, Any]] = []
        details: dict[str, dict[str, Any]] = {}
        for name in qubits:
            qubit = state.get("qubits", {}).get(name)
            if not isinstance(qubit, dict):
                raise ServiceError(f"Active state has no qubit {name}")
            extras = qubit.get("extras")
            if extras is None:
                extras = {}
            if not isinstance(extras, dict):
                raise ServiceError(f"{name} extras must be a mapping")
            already_present = "readout_fidelity" in extras
            details[name] = {
                "readout_fidelity_already_present": already_present,
                "placeholder": None if already_present else 0.0,
            }
            if not already_present:
                patch.append(
                    {
                        "op": "add",
                        "path": f"/qubits/{name}/extras/readout_fidelity",
                        "value": 0.0,
                    }
                )

        if not patch:
            return {
                "proposal_created": False,
                "message": "All selected 07b readout-fidelity keys already exist.",
                "07b_prerequisite_details": details,
            }
        proposal = self.request_state_commit(
            workflow_id,
            patch,
            "Initialize only missing readout_fidelity mapping keys to a non-passing "
            "zero sentinel so 07b can record and later replace measured values.",
            client_id,
            autonomy_lease_id=autonomy_lease_id,
            _trusted_setup=bool(autonomy_lease_id),
        )
        proposal["07b_prerequisite_details"] = details
        return proposal

    def request_07b_tail_power_reduction(
        self,
        workflow_id: str,
        qubits: list[str],
        evidence_run_id: str,
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Reduce readout amplitude one fixed step after long-tail evidence."""
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active" or workflow["current_node"] != "07b":
            raise ServiceError("07b power reduction requires an active 07b workflow")
        active_targets = self._active_targets_for_node(workflow, "07b")
        selected = set(qubits)
        if (
            not selected
            or len(selected) != len(qubits)
            or not selected.issubset(active_targets)
        ):
            raise ServiceError("Power-reduction qubits must be unique active targets")
        evidence = self.db.one(
            "SELECT * FROM runs WHERE id = ? AND workflow_id = ? AND node_id = '07b'",
            (evidence_run_id, workflow_id),
        )
        if (
            evidence is None
            or evidence["status"] != "completed"
            or evidence["analysis_status"] == "pass"
        ):
            raise ServiceError("Power reduction requires non-passing saved 07b evidence")
        analysis = json_loads(evidence.get("analysis_json"), {})
        metrics = analysis.get("dataset_metrics", {}).get("qubits", {})
        unsupported = [
            name
            for name in qubits
            if not isinstance(metrics.get(name), dict)
            or metrics[name].get("morphology_pass") is not False
        ]
        if unsupported:
            raise ServiceError(
                "Selected targets lack failed long-tail morphology evidence: "
                f"{sorted(unsupported)}"
            )

        rules = self.policy.raw["analysis"]["07b"]
        factor = float(rules["tail_retry_amplitude_factor"])
        minimum = float(rules["min_readout_amplitude"])
        state = load_state(self.settings.active_state)
        patch: list[dict[str, Any]] = []
        details: dict[str, dict[str, float]] = {}
        for name in qubits:
            readout = (
                state.get("qubits", {})
                .get(name, {})
                .get("resonator", {})
                .get("operations", {})
                .get("readout")
            )
            if not isinstance(readout, dict):
                raise ServiceError(f"{name} readout operation must be an inline mapping")
            amplitude = readout.get("amplitude")
            if (
                not isinstance(amplitude, (int, float))
                or isinstance(amplitude, bool)
                or not math.isfinite(float(amplitude))
                or float(amplitude) <= minimum
            ):
                raise ServiceError(
                    f"{name} readout amplitude cannot be reduced safely below {minimum:g}"
                )
            reduced = max(minimum, float(amplitude) * factor)
            patch.append(
                {
                    "op": "replace",
                    "path": f"/qubits/{name}/resonator/operations/readout/amplitude",
                    "value": reduced,
                }
            )
            details[name] = {
                "old_amplitude": float(amplitude),
                "new_amplitude": reduced,
                "amplitude_factor": factor,
                "power_change_db": 20.0 * math.log10(factor),
            }
        proposal = self.request_state_commit(
            workflow_id,
            patch,
            "Reduce readout amplitude by the fixed 07b tail-recovery step after "
            f"snapshot {evidence_run_id} showed elongated/tail morphology.",
            client_id,
            autonomy_lease_id=autonomy_lease_id,
            _trusted_setup=bool(autonomy_lease_id),
        )
        proposal["07b_tail_power_reduction"] = {
            "evidence_run_id": evidence_run_id,
            "targets": details,
        }
        return proposal

    def request_drive_lo_recenter(
        self,
        workflow_id: str,
        qubits: list[str],
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Propose preserving drive RF while moving private XY LOs to a grid."""
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active":
            raise ServiceError("Resume measurement mode before LO recentering")
        if workflow["current_node"] != "03a":
            raise ServiceError("Drive LO recentering is available only at node 03a")
        targets = self._active_targets_for_node(workflow, "03a")
        if not qubits or not set(qubits).issubset(targets):
            raise ServiceError("LO recenter qubits must be active workflow targets")

        state = load_state(self.settings.active_state)
        wiring = load_state(self.settings.wiring_path)
        try:
            lo_grid_hz = float(
                self.policy.raw["instrument_limits"]["drive_lo_grid_hz"]
            )
            patch, details = drive_lo_recenter_patch(
                state,
                wiring,
                qubits,
                lo_grid_hz,
                **self._drive_lo_recenter_kwargs(),
            )
            self.policy.validate_state_patch(patch, state)
        except (PolicyError, ValueError) as exc:
            if autonomy_lease_id:
                # Refused before the setup proposal exists; nothing was
                # written, so the lease stays usable.
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "deterministic_setup_policy_violation",
                    f"LO recenter setup: {exc}",
                )
            raise ServiceError(str(exc)) from exc
        payload = {
            "patch": patch,
            "reason": (
                "Preserve each selected qubit's current drive RF frequency while "
                "moving its private XY LO to the nearest 100 MHz grid point and "
                "keeping only the residual IF before a wide 03a spectroscopy sweep."
            ),
            "run_id": None,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "drive_lo_recenter": details,
            "authorization_evidence": (
                "deterministic_setup" if autonomy_lease_id else "human_proposal"
            ),
        }
        if autonomy_lease_id:
            lease = self._active_autonomy(autonomy_lease_id, workflow_id)
            self._validate_autonomy_scope(lease, node_id="03a", targets=qubits)
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        proposal["drive_lo_recenter"] = details
        return proposal

    def _fresh_qubits(self, workflow: dict[str, Any]) -> list[str]:
        """Targets this chip has never measured and not yet given defaults.

        Operator instruction 2026-10-14: any JY run of a qubit on the chip
        config.toml names is a measurement record. A target without one -- a
        copied state folder or a newly added qubit -- is measured from scratch
        and needs jy_request_fresh_qubit_defaults before its first run.
        """

        if not self.settings.fresh_qubit_defaults:
            return []
        device = str(workflow.get("device_key") or self._device_key())
        measured: set[str] = set()
        for row in self.db.all(
            "SELECT r.parameters_json FROM runs r JOIN workflows w "
            "ON w.id = r.workflow_id WHERE w.device_key = ?",
            (device,),
        ):
            measured.update(
                str(name)
                for name in json_loads(row.get("parameters_json"), {}).get("qubits") or []
            )
        for row in self.db.all(
            "SELECT p.payload_json FROM proposals p JOIN workflows w "
            "ON w.id = p.workflow_id WHERE w.device_key = ? "
            "AND p.status = 'consumed' AND p.payload_json LIKE ?",
            (device, '%"fresh_qubit_defaults"%'),
        ):
            details = json_loads(row.get("payload_json"), {}).get("fresh_qubit_defaults")
            if isinstance(details, dict):
                measured.update(str(name) for name in details)
        return [
            name
            for name in json_loads(workflow["targets_json"], [])
            if name not in measured
        ]

    def request_fresh_qubit_defaults(
        self,
        workflow_id: str,
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Empty extras and write starting values for qubits measured from scratch."""
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active":
            raise ServiceError("Resume measurement mode before the fresh-qubit defaults")
        qubits = self._fresh_qubits(workflow)
        if not qubits:
            raise ServiceError(
                "Every target already has a measurement record on this chip; "
                "its state is kept as it is"
            )
        state = load_state(self.settings.active_state)
        shared = {
            name
            for names in self.policy.shared_xy_drive_groups().values()
            for name in names
        }
        try:
            patch, details = fresh_qubit_defaults_patch(
                state, qubits, self.policy.raw["fresh_qubit_defaults"], shared
            )
            self.policy.validate_state_patch(patch, state)
        except (PolicyError, StateError, KeyError) as exc:
            if autonomy_lease_id:
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "deterministic_setup_policy_violation",
                    f"fresh-qubit defaults: {exc}",
                )
            raise ServiceError(str(exc)) from exc
        payload = {
            "patch": patch,
            "reason": (
                f"{', '.join(qubits)} have no measurement record on this chip: "
                "empty their extras and write the fresh-qubit starting values "
                "before the first run."
            ),
            "run_id": None,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "fresh_qubit_defaults": details,
            "authorization_evidence": (
                "deterministic_setup" if autonomy_lease_id else "human_proposal"
            ),
        }
        if autonomy_lease_id:
            lease = self._active_autonomy(autonomy_lease_id, workflow_id)
            self._validate_autonomy_patch_targets(lease, patch)
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        proposal["fresh_qubit_defaults"] = details
        return proposal

    def request_initial_03a_zero_if(
        self,
        workflow_id: str,
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Propose IF=0 for every active target before the first 03a coarse run."""
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active" or workflow["current_node"] != "03a":
            raise ServiceError("Initial 03a LO setup requires an active 03a workflow")
        if self._initial_03a_setup_taken(workflow):
            raise ServiceError(
                "Initial 03a LO setup is available only before the first 03a "
                "run of the pass it belongs to"
            )
        active = self._active_targets_for_node(workflow, "03a")
        qubits = [
            name
            for name in json_loads(workflow["targets_json"], [])
            if name in active
        ]
        state = load_state(self.settings.active_state)
        wiring = load_state(self.settings.wiring_path)
        max_if = float(self.policy.limits["qubit_if_abs_hz"])
        focus_rf: dict[str, float] = {}
        for name in qubits:
            # A qubit parked at the IF limit by another qubit's LO move starts
            # its search at its stored frequency (operator rule 2026-10-12).
            qubit = state["qubits"].get(name, {})
            intermediate = (qubit.get("xy") or {}).get("intermediate_frequency")
            idle = (qubit.get("extras") or {}).get("idle_freq")
            if (
                _finite_number(intermediate)
                and abs(float(intermediate)) >= max_if - 1.0
                and _finite_number(idle)
            ):
                focus_rf[name] = float(idle)
        try:
            lo_grid_hz = float(
                self.policy.raw["instrument_limits"]["drive_lo_grid_hz"]
            )
            patch, details = drive_lo_recenter_patch(
                state,
                wiring,
                qubits,
                lo_grid_hz,
                force_zero_if=True,
                focus_rf_hz=focus_rf or None,
                **self._drive_lo_recenter_kwargs(),
            )
            self.policy.validate_state_patch(patch, state)
        except (PolicyError, ValueError) as exc:
            if autonomy_lease_id:
                # Refused before the setup proposal exists; nothing was
                # written, so the lease stays usable.
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "deterministic_setup_policy_violation",
                    f"initial 03a setup: {exc}",
                )
            raise ServiceError(str(exc)) from exc
        payload = {
            "patch": patch,
            "reason": (
                "Before the first multiplexed 03a coarse search, preserve every "
                "active target's current RF by moving its private LO to that RF "
                "and setting every XY IF to zero, enabling an 800 MHz full span."
            ),
            "run_id": None,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "initial_03a_zero_if": details,
            "authorization_evidence": (
                "deterministic_setup" if autonomy_lease_id else "human_proposal"
            ),
        }
        if autonomy_lease_id:
            lease = self._active_autonomy(autonomy_lease_id, workflow_id)
            self._validate_autonomy_scope(lease, node_id="03a", targets=qubits)
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        proposal["initial_03a_zero_if"] = details
        return proposal

    def _readout_full_scale_plan(
        self, workflow: dict[str, Any], state: dict[str, Any] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Lower a readout line's full scale once its measurable qubits passed 02c.

        Operator choice 2026-10-07: qubits marked scientifically unmeasurable no
        longer hold a line at its template full scale. Each remaining member
        keeps its delivered power; the full scale becomes the smallest integer
        that keeps every pulse <= max_readout_pulse_amplitude and the line sum
        <= max_aggregate_readout_amplitude.
        """
        import math as _math

        state = state or load_state(self.settings.active_state)
        wiring = (
            load_state(self.settings.wiring_path)
            if self.settings.wiring_path.is_file()
            else {}
        )
        limits = _readout_power_limits(self.settings, self.policy.raw)
        unmeasurable = set().union(
            *self._scientifically_unmeasurable_targets_by_node(workflow["id"]).values()
        )
        resolved, _absent = self._02c_target_resolution(workflow["id"])
        targets = set(json_loads(workflow.get("targets_json"), []) or [])
        lines: dict[str, dict[str, Any]] = {}
        for name in sorted(targets - unmeasurable):
            try:
                line = _readout_line_of(state, wiring, name)
            except Exception:
                continue
            if line.get("path") is None:
                continue
            lines.setdefault(line["key"], line)
        patch: list[dict[str, Any]] = []
        details: dict[str, Any] = {}
        for key, line in sorted(lines.items()):
            members = [name for name in line["members"] if name not in unmeasurable]
            if not members or not set(members) <= resolved:
                continue
            old = float(line["full_scale"])
            powers: dict[str, float] = {}
            paths: dict[str, str] = {}
            for name in members:
                operations = state["qubits"][name]["resonator"]["operations"]
                op = "readout"
                readout = operations.get(op)
                if isinstance(readout, str) and readout.startswith("#./"):
                    op = readout[3:]
                    readout = operations.get(op)
                amplitude = float(readout["amplitude"])
                if amplitude <= 0:
                    break
                powers[name] = old + 20.0 * _math.log10(amplitude)
                paths[name] = f"/qubits/{name}/resonator/operations/{op}/amplitude"
            else:
                chosen = _smallest_full_scale(powers, limits)
                if chosen is None or float(chosen) == old:
                    continue
                patch.append({"op": "replace", "path": line["path"], "value": int(chosen)})
                amplitudes = {
                    name: 10.0 ** ((power - chosen) / 20.0) for name, power in powers.items()
                }
                for name, amplitude in amplitudes.items():
                    patch.append({"op": "replace", "path": paths[name], "value": amplitude})
                details[key] = {
                    "old_full_scale_dbm": old,
                    "new_full_scale_dbm": int(chosen),
                    "excluded_unmeasurable": sorted(set(line["members"]) & unmeasurable),
                    "power_dbm": powers,
                    "amplitudes": amplitudes,
                }
        return patch, details

    def request_readout_full_scale_rescale(
        self,
        workflow_id: str,
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Lower readout full scales ignoring unmeasurable line members."""
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] not in {"active", "paused"}:
            raise ServiceError("Full-scale rescale requires an open workflow")
        state = load_state(self.settings.active_state)
        patch, details = self._readout_full_scale_plan(workflow, state)
        if not patch:
            return {
                "proposal_created": False,
                "message": "No readout line can lower its full scale yet.",
            }
        try:
            self.policy.validate_state_patch(patch, state)
        except PolicyError as exc:
            if autonomy_lease_id:
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "deterministic_setup_policy_violation",
                    f"readout full-scale rescale: {exc}",
                )
            raise ServiceError(str(exc)) from exc
        payload = {
            "patch": patch,
            "reason": (
                "Lower each readout line's full scale to the smallest integer that "
                "fits its measurable qubits, keeping their delivered power; "
                "unmeasurable line members no longer hold it."
            ),
            "run_id": None,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "readout_full_scale_rescale": details,
            "authorization_evidence": (
                "deterministic_setup" if autonomy_lease_id else "human_proposal"
            ),
        }
        if autonomy_lease_id:
            lease = self._active_autonomy(autonomy_lease_id, workflow_id)
            self._validate_autonomy_patch_targets(lease, patch)
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        proposal["readout_full_scale_rescale"] = details
        return proposal

    def _readout_length_normalize_patch(
        self, workflow: dict[str, Any], state: dict[str, Any] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Cap readout length and depletion time (operator choice 2026-10-07)."""
        config = self.policy.raw.get("autonomy", {})
        length_cap = config.get("readout_length_normalize_ns")
        depletion_cap = config.get("readout_depletion_normalize_ns")
        if length_cap is None:
            return [], {}
        length_cap = int(length_cap)
        if length_cap < 16 or length_cap % 4:
            raise ServiceError("readout_length_normalize_ns must be a multiple of 4, >= 16")
        targets = json_loads(workflow.get("targets_json"), None)
        if not isinstance(targets, list):
            targets = list(workflow.get("targets") or [])
        unmeasurable = set().union(
            *self._scientifically_unmeasurable_targets_by_node(workflow["id"]).values()
        ) if workflow.get("id") else set()
        state = state or load_state(self.settings.active_state)
        patch: list[dict[str, Any]] = []
        details: dict[str, Any] = {}
        for name in sorted(set(targets) - unmeasurable):
            resonator = state.get("qubits", {}).get(name, {}).get("resonator") or {}
            readout = (resonator.get("operations") or {}).get("readout")
            if not isinstance(readout, dict):
                continue  # a referenced readout operation is left alone
            change: dict[str, Any] = {}
            length = readout.get("length")
            if isinstance(length, (int, float)) and length > length_cap:
                patch.append(
                    {
                        "op": "replace",
                        "path": f"/qubits/{name}/resonator/operations/readout/length",
                        "value": length_cap,
                    }
                )
                change["length_ns"] = [length, length_cap]
            depletion = resonator.get("depletion_time")
            if (
                depletion_cap is not None
                and isinstance(depletion, (int, float))
                and depletion > int(depletion_cap)
            ):
                patch.append(
                    {
                        "op": "replace",
                        "path": f"/qubits/{name}/resonator/depletion_time",
                        "value": int(depletion_cap),
                    }
                )
                change["depletion_time_ns"] = [depletion, int(depletion_cap)]
            if change:
                details[name] = change
        return patch, details

    def _02x_probe_setup_plan(
        self, workflow: dict[str, Any], state: dict[str, Any] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Line headroom for 02x's fixed probe and unique plot cells."""
        state = state or load_state(self.settings.active_state)
        wiring = (
            load_state(self.settings.wiring_path)
            if self.settings.wiring_path.is_file()
            else {}
        )
        details: dict[str, Any] = {}
        grid_patch, grid = unique_grid_location_patch(state)
        if grid:
            details["grid_location"] = grid
        patch = list(grid_patch)
        thread_patch, threads = unique_thread_patch(state)
        if threads:
            details["thread"] = threads
        patch.extend(thread_patch)
        if str(workflow.get("current_node")) == "02x":
            unmeasurable = set().union(
                *self._scientifically_unmeasurable_targets_by_node(workflow["id"]).values()
            )
            targets = sorted(
                set(json_loads(workflow.get("targets_json"), []) or []) - unmeasurable
            )
            minimum = float(
                self.policy.limits["opx1000_full_scale_power_dbm"].get("min", -11)
            )
            headroom_patch, headroom = probe_02x_headroom_patch(
                state, wiring, targets, min_full_scale_dbm=minimum
            )
            if headroom:
                details["probe_headroom"] = headroom
            patch.extend(headroom_patch)
        return patch, details

    def request_02x_probe_setup(
        self,
        workflow_id: str,
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Prepare the state so 02x's probe stays linear and every plot is visible."""
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] not in {"active", "paused"}:
            raise ServiceError("02x probe setup requires an open workflow")
        state = load_state(self.settings.active_state)
        patch, details = self._02x_probe_setup_plan(workflow, state)
        if not patch:
            return {
                "proposal_created": False,
                "message": "Every qubit has its own grid cell and 02x's probe fits each line.",
            }
        try:
            self.policy.validate_state_patch(patch, state)
        except PolicyError as exc:
            if autonomy_lease_id:
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "deterministic_setup_policy_violation",
                    f"02x probe setup: {exc}",
                )
            raise ServiceError(str(exc)) from exc
        payload = {
            "patch": patch,
            "reason": (
                "Give each qubit its own QUA thread (resonator and XY) when "
                "elements share one; raise each readout line's full scale so "
                "02x's fixed -10 dBm "
                "probe tones sum to at most 1 (no clipping), keeping every "
                "qubit's delivered readout power; give each qubit its own "
                "grid_location so node plots no longer overlap."
            ),
            "run_id": None,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "02x_probe_setup": details,
            "authorization_evidence": (
                "deterministic_setup" if autonomy_lease_id else "human_proposal"
            ),
        }
        if autonomy_lease_id:
            lease = self._active_autonomy(autonomy_lease_id, workflow_id)
            self._validate_autonomy_patch_targets(lease, patch)
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        proposal["02x_probe_setup"] = details
        return proposal

    def request_readout_length_normalize(
        self,
        workflow_id: str,
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Cap long template readout pulses before the readout nodes."""
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] not in {"active", "paused"}:
            raise ServiceError("Readout length normalize requires an open workflow")
        state = load_state(self.settings.active_state)
        patch, details = self._readout_length_normalize_patch(workflow, state)
        if not patch:
            return {
                "proposal_created": False,
                "message": "Every readout length and depletion time is within the cap.",
            }
        try:
            self.policy.validate_state_patch(patch, state)
        except PolicyError as exc:
            if autonomy_lease_id:
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "deterministic_setup_policy_violation",
                    f"readout length normalize: {exc}",
                )
            raise ServiceError(str(exc)) from exc
        payload = {
            "patch": patch,
            "reason": (
                "Cap readout length and depletion time at the configured values "
                "so multiplexed points no longer wait for a long template pulse; "
                "07d later selects each qubit's readout length."
            ),
            "run_id": None,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "readout_length_normalize": details,
            "authorization_evidence": (
                "deterministic_setup" if autonomy_lease_id else "human_proposal"
            ),
        }
        if autonomy_lease_id:
            lease = self._active_autonomy(autonomy_lease_id, workflow_id)
            self._validate_autonomy_patch_targets(lease, patch)
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        proposal["readout_length_normalize"] = details
        return proposal

    def _missing_node_extras(
        self, workflow: dict[str, Any], state: dict[str, Any] | None = None
    ) -> dict[str, list[str]]:
        """Extras keys the remaining nodes record but the state lacks.

        Per qubit, for every node from the current one on (2026-10-07).
        """
        config = self.policy.raw.get("autonomy", {}).get("node_extras_prerequisites") or {}
        if not config:
            return {}
        sequence = self._workflow_sequence(workflow)
        current = str(workflow.get("current_node"))
        remaining = sequence[sequence.index(current):] if current in sequence else sequence
        keys = sorted({str(key) for node in remaining for key in config.get(node, [])})
        targets = json_loads(workflow.get("targets_json"), None)
        if not isinstance(targets, list):
            targets = list(workflow.get("targets") or [])
        # 2026-10-12: only a qubit that cannot reach the remaining nodes is
        # skipped -- no resonator (02/02c) or no frequency / pi pulse. A
        # qubit incomplete at 06 still runs 10a and needs its extras.
        marks = (
            self._scientifically_unmeasurable_targets_by_node(workflow["id"])
            if workflow.get("id")
            else {}
        )
        unmeasurable = set().union(
            *(
                targets_at
                for node, targets_at in marks.items()
                if node in {"02", "02c"} | self._pass_prerequisite_nodes()
            )
        )
        state = state or load_state(self.settings.active_state)
        missing: dict[str, list[str]] = {}
        for name in sorted(set(targets) - unmeasurable):
            extras = state.get("qubits", {}).get(name, {}).get("extras")
            extras = extras if isinstance(extras, dict) else {}
            absent = [key for key in keys if key not in extras]
            if absent:
                missing[name] = absent
        return missing

    def request_node_extras_prerequisites(
        self,
        workflow_id: str,
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Add every missing node-output extras key with a 0.0 placeholder."""
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] not in {"active", "paused"}:
            raise ServiceError("Extras prerequisites require an open workflow")
        state = load_state(self.settings.active_state)
        missing = self._missing_node_extras(workflow, state)
        wiring = (
            load_state(self.settings.wiring_path)
            if self.settings.wiring_path.is_file()
            else {}
        )
        wiring_qubits = wiring.get("wiring", wiring).get("qubits", {})

        def current_rf(name: str, element: str) -> float | None:
            rr = "rr" if element == "resonator" else "xy"
            output = wiring_qubits.get(name, {}).get(rr, {}).get("opx_output")
            try:
                lo = pointer_get(state, str(output).lstrip("#") + "/upconverter_frequency")
                if_ = state["qubits"][name][element]["intermediate_frequency"]
                return float(lo) + float(if_)
            except Exception:
                return None

        def placeholder(name: str, key: str) -> float | None:
            # Frequencies take the current RF (positive, replaced by the
            # node); everything else a 0.0 sentinel, as 07b does.
            if key in {"bare_resonator_freq", "dressed_resonator_freq"}:
                return current_rf(name, "resonator")
            if key in {"idle_freq", "sweetspot_freq"}:
                return current_rf(name, "xy")
            # 2026-10-13: lifetimes must stay positive (policy), so the
            # statistics keys start from the qubit's single-run value.
            lifetime_source = {
                "T1": "T1", "T1_dev": "T1",
                "T2": "T2echo", "T2_dev": "T2echo",
                "T2ramsey": "T2ramsey", "T2ramsey_dev": "T2ramsey",
            }.get(key)
            if lifetime_source is not None:
                value = state["qubits"].get(name, {}).get(lifetime_source)
                return float(value) if _finite_positive(value) else 1e-6
            return 0.0

        patch = []
        for name, keys in sorted(missing.items()):
            for key in keys:
                value = placeholder(name, key)
                if value is None or value <= 0 and key.endswith("freq"):
                    raise ServiceError(f"No current RF for {name} extras/{key}")
                patch.append(
                    {"op": "add", "path": f"/qubits/{name}/extras/{key}", "value": value}
                )
        if not patch:
            return {
                "proposal_created": False,
                "message": "Every extras key the remaining nodes record already exists.",
            }
        try:
            self.policy.validate_state_patch(patch, state)
        except PolicyError as exc:
            if autonomy_lease_id:
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "deterministic_setup_policy_violation",
                    f"node extras prerequisites: {exc}",
                )
            raise ServiceError(str(exc)) from exc
        payload = {
            "patch": patch,
            "reason": (
                "Add the extras keys later nodes record, with a 0.0 placeholder, "
                "so Qualibrate's state recorder can read an old value."
            ),
            "run_id": None,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "node_extras_prerequisites": missing,
            "authorization_evidence": (
                "deterministic_setup" if autonomy_lease_id else "human_proposal"
            ),
        }
        if autonomy_lease_id:
            lease = self._active_autonomy(autonomy_lease_id, workflow_id)
            self._validate_autonomy_patch_targets(lease, patch)
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        proposal["node_extras_prerequisites"] = missing
        return proposal

    def _readout_lo_recenter_plan(
        self, workflow: dict[str, Any], state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Readout LO per line that covers every operator design frequency.

        Operator request 2026-10-07: 02 sweeps each readout line over its LO
        +- `max_if_in_mhz`. When the design frequencies typed on the lease page
        fall outside that window (or inside the LO guard), the line's LO is
        moved on the LO grid. Assigned qubits keep their RF (their IF is
        recomputed); unassigned ones stay unassigned.
        """
        options = self._workflow_options(workflow)
        values = (options.get("node_parameters") or {}).get("02") or {}
        design = values.get("res_design_freq") or []
        targets = json_loads(workflow.get("targets_json"), None)
        if not isinstance(targets, list):
            targets = list(workflow.get("targets") or [])
        ordered = sorted(targets, key=lambda name: (len(name), name))
        if not design or len(design) != len(ordered):
            return {}
        design_hz = {
            name: float(value) * 1e9 if float(value) < 1e6 else float(value)
            for name, value in zip(ordered, design)
        }
        defaults = self.policy.node_definition("02").get("defaults", {})
        max_if = float(defaults.get("max_if_in_mhz", 400.0)) * 1e6
        guard = float(defaults.get("lo_guard_in_mhz", 10.0)) * 1e6
        tolerance = float(defaults.get("design_match_tolerance_in_mhz", 10.0)) * 1e6
        margin = 20e6 + tolerance
        grid = float(self.policy.raw["instrument_limits"].get("drive_lo_grid_hz", 1e8))
        state = state or load_state(self.settings.active_state)
        wiring = (
            load_state(self.settings.wiring_path)
            if self.settings.wiring_path.is_file()
            else {}
        )
        wiring_qubits = wiring.get("wiring", wiring).get("qubits", {})
        unset = set(self.policy.readout_frequencies_unset(ordered, state))
        lines: dict[tuple[str, str], list[str]] = {}
        for name in ordered:
            rr = wiring_qubits.get(name, {}).get("rr", {})
            output, input_ = rr.get("opx_output"), rr.get("opx_input")
            if isinstance(output, str) and isinstance(input_, str):
                lines.setdefault((output, input_), []).append(name)
        patch: list[dict[str, Any]] = []
        details: dict[str, Any] = {}
        moved: list[str] = []
        for (output, input_), names in sorted(lines.items()):
            out_path = output.lstrip("#") + "/upconverter_frequency"
            in_path = input_.lstrip("#") + "/downconverter_frequency"
            old_lo = pointer_get(state, out_path)
            freqs = [design_hz[name] for name in names]

            def covers(lo: float) -> bool:
                return all(
                    guard + margin <= abs(f - lo) <= max_if - margin for f in freqs
                )

            if covers(float(old_lo)):
                continue
            centre = (min(freqs) + max(freqs)) / 2.0
            base = round(centre / grid) * grid
            candidates = sorted(
                (base + step * grid for step in range(-3, 4)),
                key=lambda lo: abs(lo - centre),
            )
            new_lo = next((lo for lo in candidates if covers(lo)), None)
            if new_lo is None:
                raise ServiceError(
                    f"No readout LO on {output} covers design frequencies "
                    f"{[round(f / 1e9, 4) for f in freqs]} GHz within "
                    f"+-{(max_if - margin) / 1e6:g} MHz"
                )
            patch.append({"op": "replace", "path": out_path, "value": new_lo})
            patch.append({"op": "replace", "path": in_path, "value": new_lo})
            ifs: dict[str, float] = {}
            for name in names:
                path = f"/qubits/{name}/resonator/intermediate_frequency"
                old_if = float(pointer_get(state, path))
                new_if = 0.0 if name in unset else float(old_lo) + old_if - new_lo
                # Always listed, so the patch names its qubits for the lease.
                patch.append({"op": "replace", "path": path, "value": new_if})
                ifs[name] = new_if
            moved.extend(names)
            details[output] = {
                "old_lo_hz": float(old_lo),
                "new_lo_hz": new_lo,
                "design_frequencies_hz": {name: design_hz[name] for name in names},
                "new_if_hz": ifs,
            }
        return {"patch": patch, "details": details, "qubits": moved}

    def request_readout_lo_recenter(
        self,
        workflow_id: str,
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Move readout LOs so 02 covers every design frequency (2026-10-07)."""
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active" or workflow["current_node"] != "02":
            raise ServiceError("Readout LO recentring requires an active 02 workflow")
        state = load_state(self.settings.active_state)
        plan = self._readout_lo_recenter_plan(workflow, state)
        patch = plan.get("patch") or []
        if not patch:
            return {
                "proposal_created": False,
                "message": "Every readout LO already covers its design frequencies.",
            }
        try:
            self.policy.validate_state_patch(patch, state)
        except PolicyError as exc:
            if autonomy_lease_id:
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "deterministic_setup_policy_violation",
                    f"readout LO recentre: {exc}",
                )
            raise ServiceError(str(exc)) from exc
        payload = {
            "patch": patch,
            "reason": (
                "Move each readout LO so 02 covers every operator design "
                "frequency; assigned qubits keep their RF, and the earlier 02 "
                "pass no longer counts."
            ),
            "run_id": None,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "readout_lo_recenter": plan["details"],
            "authorization_evidence": (
                "deterministic_setup" if autonomy_lease_id else "human_proposal"
            ),
        }
        if autonomy_lease_id:
            lease = self._active_autonomy(autonomy_lease_id, workflow_id)
            self._validate_autonomy_patch_targets(lease, patch)
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        proposal["readout_lo_recenter"] = plan["details"]
        return proposal

    def request_readout_amplitude_budget(
        self,
        workflow_id: str,
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Scale over-budget readout lines down (operator choice 2026-10-07)."""
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active":
            raise ServiceError("Readout amplitude scaling requires an active workflow")
        targets = sorted(json_loads(workflow["targets_json"], []))
        state = load_state(self.settings.active_state)
        patch, details = self.policy.readout_amplitude_budget_patch(targets, state)
        if not patch:
            return {
                "proposal_created": False,
                "message": "Every readout line is already within its amplitude budget.",
            }
        try:
            self.policy.validate_state_patch(patch, state)
        except PolicyError as exc:
            if autonomy_lease_id:
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "deterministic_setup_policy_violation",
                    f"readout amplitude budget: {exc}",
                )
            raise ServiceError(str(exc)) from exc
        payload = {
            "patch": patch,
            "reason": (
                "Scale every amplitude on an over-budget readout line by one "
                "factor so the line sums to readout_amplitude_budget_target."
            ),
            "run_id": None,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "readout_amplitude_budget": details,
            "authorization_evidence": (
                "deterministic_setup" if autonomy_lease_id else "human_proposal"
            ),
        }
        if autonomy_lease_id:
            lease = self._active_autonomy(autonomy_lease_id, workflow_id)
            self._validate_autonomy_patch_targets(lease, patch)
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        proposal["readout_amplitude_budget"] = details
        return proposal

    def request_02c_window_recenter(
        self,
        workflow_id: str,
        qubits: list[str],
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Centre each readout IF between the latest 02c dressed and bare lines.

        Operator rule 2026-10-05: a 02c pass needs a window about 2-5x the
        bare-dressed separation. A narrow retry must be centred between the
        two lines, so this setup moves the resonator IF to their midpoint and
        reports the recommended span.
        """
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active" or workflow["current_node"] != "02c":
            raise ServiceError("02c window recentring requires an active 02c workflow")
        active = self._active_targets_for_node(workflow, "02c")
        unresolved = active - self._node_target_resolution(workflow_id, "02c")
        if not qubits or not set(qubits) <= unresolved:
            raise ServiceError(
                f"02c recentring targets must be unresolved 02c targets: {sorted(unresolved)}"
            )
        state = load_state(self.settings.active_state)
        wiring = (
            load_state(self.settings.wiring_path)
            if self.settings.wiring_path.is_file()
            else {}
        )
        transitions: dict[str, dict[str, Any]] = {}
        for row in self.db.all(
            "SELECT parameters_json, analysis_json FROM runs WHERE workflow_id = ? "
            "AND node_id = '02c' AND status = 'completed' ORDER BY rowid",
            (workflow_id,),
        ):
            analysis = json_loads(row.get("analysis_json"), {})
            metrics = (analysis.get("dataset_metrics") or {}).get("qubits") or {}
            for name in json_loads(row.get("parameters_json"), {}).get("qubits", []):
                entry = metrics.get(name)
                transition = entry.get("02c_transition") if isinstance(entry, dict) else None
                if isinstance(transition, dict):
                    transitions[str(name)] = transition
        patch: list[dict[str, Any]] = []
        details: dict[str, Any] = {}
        for name in qubits:
            transition = transitions.get(name, {})
            dressed = transition.get("dressed_frequency_hz")
            bare = transition.get("bare_frequency_hz")
            if not (_finite_number(dressed) and _finite_number(bare)):
                raise ServiceError(f"{name} has no 02c dressed/bare frequencies to centre on")
            reference = _readout_output_reference(state, wiring, name)
            port = pointer_get(state, reference) if reference else None
            lo = port.get("upconverter_frequency") if isinstance(port, dict) else None
            if not _finite_number(lo):
                raise ServiceError(f"{name} readout upconverter frequency is unavailable")
            midpoint = (float(dressed) + float(bare)) / 2.0
            new_if = midpoint - float(lo)
            path = f"/qubits/{name}/resonator/intermediate_frequency"
            patch.append({"op": "replace", "path": path, "value": new_if})
            details[name] = {
                "dressed_frequency_hz": float(dressed),
                "bare_frequency_hz": float(bare),
                "midpoint_frequency_hz": midpoint,
                "old_if_hz": pointer_get(state, path),
                "new_if_hz": new_if,
                "separation_hz": abs(float(bare) - float(dressed)),
                "recommended_frequency_span_mhz": transition.get(
                    "recommended_frequency_span_mhz"
                ),
            }
        try:
            self.policy.validate_state_patch(patch, state)
        except PolicyError as exc:
            if autonomy_lease_id:
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "deterministic_setup_policy_violation",
                    f"02c window recentre: {exc}",
                )
            raise ServiceError(str(exc)) from exc
        payload = {
            "patch": patch,
            "reason": (
                "Centre each readout IF between its latest 02c dressed and bare "
                "frequencies so the retry window can be 2-5x their separation."
            ),
            "run_id": None,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "02c_window_recenter": details,
            "authorization_evidence": (
                "deterministic_setup" if autonomy_lease_id else "human_proposal"
            ),
        }
        if autonomy_lease_id:
            lease = self._active_autonomy(autonomy_lease_id, workflow_id)
            self._validate_autonomy_scope(lease, node_id="02c", targets=qubits)
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        proposal["02c_window_recenter"] = details
        return proposal

    def request_03a_window_shift(
        self,
        workflow_id: str,
        lo_centers_in_ghz: dict[str, float],
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Propose 100 MHz-grid, IF=0 coarse windows for unresolved targets."""
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active" or workflow["current_node"] != "03a":
            raise ServiceError("03a window shifting requires an active 03a workflow")
        shift_ready = self._03a_shift_ready_targets(workflow_id)
        selected = set(lo_centers_in_ghz)
        if not selected or not selected.issubset(shift_ready):
            raise ServiceError(
                "Window-shift qubits must either be edge-limited or remain "
                "unresolved after 2000 averages; eligible targets are "
                f"{sorted(shift_ready)}"
            )
        centers_hz = {
            name: float(value) * 1e9
            for name, value in lo_centers_in_ghz.items()
        }
        state = load_state(self.settings.active_state)
        wiring = load_state(self.settings.wiring_path)
        try:
            lo_grid_hz = float(
                self.policy.raw["instrument_limits"]["drive_lo_grid_hz"]
            )
            patch, details = drive_lo_recenter_patch(
                state,
                wiring,
                list(lo_centers_in_ghz),
                lo_grid_hz,
                target_lo_hz=centers_hz,
                **self._drive_lo_recenter_kwargs(),
            )
            self.policy.validate_state_patch(patch, state)
        except (PolicyError, ValueError) as exc:
            if autonomy_lease_id:
                # Refused before the setup proposal exists; nothing was
                # written, so the lease stays usable.
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "deterministic_setup_policy_violation",
                    f"03a window shift: {exc}",
                )
            raise ServiceError(str(exc)) from exc
        payload = {
            "patch": patch,
            "reason": (
                "Move only unresolved 03a targets to explicitly selected "
                "100 MHz-grid LO coarse-search windows with IF=0."
            ),
            "run_id": None,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "03a_window_shift": details,
            "authorization_evidence": (
                "deterministic_setup" if autonomy_lease_id else "human_proposal"
            ),
        }
        if autonomy_lease_id:
            lease = self._active_autonomy(autonomy_lease_id, workflow_id)
            self._validate_autonomy_scope(
                lease, node_id="03a", targets=list(lo_centers_in_ghz)
            )
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        proposal["03a_window_shift"] = details
        return proposal

    def request_03a_candidate_center(
        self,
        workflow_id: str,
        qubits: list[str],
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Place credible unresolved 03a candidates at safe LO/IF centers."""
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active" or workflow["current_node"] != "03a":
            raise ServiceError("03a candidate centering requires an active 03a workflow")
        selected = set(qubits)
        eligible = (
            self._03a_candidate_targets(workflow_id)
            - self._node_target_resolution(workflow_id, "03a")
        )
        if not selected or len(selected) != len(qubits) or not selected.issubset(eligible):
            raise ServiceError(
                "Candidate-center qubits must be credible unresolved 03a targets; "
                f"eligible targets are {sorted(eligible)}"
            )

        min_snr = float(self.policy.raw["analysis"]["03a"]["min_robust_snr"])
        target_rf_hz: dict[str, float] = {}
        evidence: dict[str, dict[str, Any]] = {}
        for name in qubits:
            # After a 04 -> 03a return only the qubit's new runs count.
            for row in reversed(self._qubit_runs(workflow_id, "03a", name)):
                if name not in resolve_03a_candidate_targets([row], min_snr):
                    continue
                analysis = json_loads(row.get("analysis_json"), {})
                fit = analysis.get("fit_quality", {}).get("results", {}).get(name, {})
                drive_freq = fit.get("drive_freq") if isinstance(fit, dict) else None
                if (
                    isinstance(drive_freq, (int, float))
                    and not isinstance(drive_freq, bool)
                    and math.isfinite(float(drive_freq))
                ):
                    target_rf_hz[name] = float(drive_freq)
                    metrics = analysis.get("dataset_metrics", {}).get("qubits", {}).get(name, {})
                    evidence[name] = {
                        "run_id": row["id"],
                        "candidate_rf_hz": float(drive_freq),
                        "robust_snr": metrics.get("robust_snr") if isinstance(metrics, dict) else None,
                        "edge_fraction": metrics.get("edge_fraction") if isinstance(metrics, dict) else None,
                        "03a_stage": analysis.get("03a_stage"),
                    }
                    break
            if name not in target_rf_hz:
                # 2026-10-12: a sub-floor line confirmed under two LOs.
                cross_lo = self._03a_cross_lo_candidates(workflow_id)
                if name in cross_lo:
                    target_rf_hz[name] = cross_lo[name]
                    evidence[name] = {
                        "candidate_rf_hz": cross_lo[name],
                        "03a_stage": "cross_lo_confirmed",
                    }
            if name not in target_rf_hz:
                raise ServiceError(f"No finite credible 03a candidate frequency for {name}")

        state = load_state(self.settings.active_state)
        wiring = load_state(self.settings.wiring_path)
        try:
            lo_grid_hz = float(
                self.policy.raw["instrument_limits"]["drive_lo_grid_hz"]
            )
            patch, details = drive_lo_recenter_patch(
                state,
                wiring,
                qubits,
                lo_grid_hz,
                target_rf_hz=target_rf_hz,
                **self._drive_lo_recenter_kwargs(),
            )
            self.policy.validate_state_patch(patch, state)
        except (PolicyError, ValueError) as exc:
            if autonomy_lease_id:
                # Refused before the setup proposal exists; nothing was
                # written, so the lease stays usable.
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "deterministic_setup_policy_violation",
                    f"03a candidate center: {exc}",
                )
            raise ServiceError(str(exc)) from exc
        for name, item in details.items():
            item["candidate_evidence"] = evidence[name]

        payload = {
            "patch": patch,
            "reason": (
                "Center each credible unresolved 03a candidate using the nearest "
                "100 MHz-grid private XY LO and the residual IF, preserving the "
                "evidence-backed candidate RF before a low-power fine scan."
            ),
            "run_id": None,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "03a_candidate_center": details,
            "authorization_evidence": (
                "deterministic_setup" if autonomy_lease_id else "human_proposal"
            ),
        }
        if autonomy_lease_id:
            lease = self._active_autonomy(autonomy_lease_id, workflow_id)
            self._validate_autonomy_scope(lease, node_id="03a", targets=qubits)
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        proposal["03a_candidate_center"] = details
        return proposal

    def apply_state_commit(self, proposal_id: str, client_id: str) -> dict[str, Any]:
        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        proposal = self._approved_proposal(proposal_id, "state_commit")
        if self._workflow(proposal["workflow_id"])["status"] != "active":
            raise ServiceError("Resume measurement mode before applying a state update")
        payload = json_loads(proposal["payload_json"], {})
        lease_id = proposal.get("autonomy_lease_id")
        if lease_id:
            lease = self._active_autonomy(str(lease_id), proposal["workflow_id"])
            self._validate_autonomy_patch_targets(lease, payload["patch"])
        self.policy.validate_state_patch(payload["patch"])
        claimed = False
        try:
            with exclusive_file_lock(
                self.settings.runtime / "hardware-transition.lock",
                "state commit hardware exclusion",
            ):
                self._assert_state_commit_idle()
                changed = self.db.execute(
                    "UPDATE proposals SET status = 'applying' WHERE id = ? "
                    "AND status = 'approved' AND uses < max_uses",
                    (proposal_id,),
                )
                if changed != 1:
                    raise ServiceError(
                        "State proposal was consumed or claimed concurrently"
                    )
                claimed = True
                result = commit_state(
                    self.settings.active_state,
                    payload["patch"],
                    self.settings.runtime / "backups",
                    payload["base_hash"],
                )
                self.db.execute(
                    "UPDATE proposals SET uses = uses + 1, status = 'consumed' "
                    "WHERE id = ? AND status = 'applying'",
                    (proposal_id,),
                )
        except Exception as exc:
            if claimed:
                self.db.execute(
                    "UPDATE proposals SET status = 'approved' WHERE id = ? "
                    "AND status = 'applying'",
                    (proposal_id,),
                )
            if lease_id:
                self._halt_autonomy(
                    str(lease_id),
                    f"State hash conflict or commit failure: {exc}",
                    "autonomy_guard",
                    stop_active=True,
                )
            raise
        output = {
            "proposal_id": proposal_id,
            "status": "committed",
            **result,
            "patch": payload["patch"],
        }
        self.db.event(
            "state_committed", client_id, output, proposal["workflow_id"]
        )
        run_id = payload.get("run_id")
        if run_id:
            terminal = self.db.one(
                "SELECT id FROM decisions WHERE run_id = ? AND decision = 'advance' "
                "AND next_node IS NULL ORDER BY created_at DESC LIMIT 1",
                (run_id,),
            )
            if terminal is not None:
                self.db.execute(
                    "UPDATE workflows SET status = 'completed', updated_at = ? "
                    "WHERE id = ? AND current_node IS NULL",
                    (utc_now(), proposal["workflow_id"]),
                )
                self.db.execute(
                    "UPDATE autonomy_leases SET status = 'completed', "
                    "stopped_reason = ?, updated_at = ? WHERE workflow_id = ? "
                    "AND status IN ('active', 'paused')",
                    (
                        "Workflow completed after its terminal state commit.",
                        utc_now(),
                        proposal["workflow_id"],
                    ),
                )
            else:
                workflow = self._workflow(proposal["workflow_id"])
                self._complete_autonomy_scope_if_needed(
                    proposal["workflow_id"], workflow.get("current_node"), client_id
                )
        return output

    def proposal(self, proposal_id: str) -> dict[str, Any]:
        proposal = self.db.one("SELECT * FROM proposals WHERE id = ?", (proposal_id,))
        if proposal is None:
            raise ServiceError(f"Unknown proposal: {proposal_id}")
        return self._decode_proposal(proposal)

    def approve_interactively(self, proposal_id: str) -> dict[str, Any]:
        return self._approve_pending_proposal(
            proposal_id,
            getpass.getuser(),
            "interactive_terminal",
        )

    @property
    def approval_local_origin(self) -> str:
        runtime = self._runtime_approval_configuration()
        host = str(runtime["host"])
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{runtime['port']}"

    @property
    def browser_approval_origin(self) -> str:
        runtime = self._runtime_approval_configuration()
        return str(runtime["public_base_url"] or self.approval_local_origin)

    @property
    def approval_transport(self) -> str:
        return str(self._runtime_approval_configuration()["transport"])

    @property
    def remote_approval_enabled(self) -> bool:
        runtime = self._runtime_approval_configuration()
        return bool(
            runtime["transport"] == "public"
            and runtime["public_base_url"]
            and runtime.get("access_secret_present")
            and runtime.get("password_present")
        )

    @property
    def approval_allowed_origins(self) -> tuple[str, ...]:
        runtime = self._runtime_approval_configuration()
        origins = {
            self.approval_local_origin,
            *self.settings.approval_trusted_origins,
        }
        if runtime["public_base_url"]:
            origins.add(str(runtime["public_base_url"]))
        return tuple(sorted(origins))

    def _runtime_approval_configuration(self) -> dict[str, Any]:
        """Resolve current Dashboard metadata without coupling STDIO startup to it."""
        fallback: dict[str, Any] = {
            "transport": self.settings.approval_transport,
            "host": self.settings.approval_host,
            "port": self.settings.approval_port,
            "public_base_url": self.settings.approval_public_base_url,
            "access_secret_present": bool(self.settings.approval_access_token),
            "password_present": self._dashboard_password_present(
                self.settings.runtime / "dashboard-password.txt"
            ),
        }
        bootstrap_path = self.settings.runtime / "server-bootstrap.json"
        try:
            raw = json.loads(bootstrap_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, TypeError):
            return fallback
        if not isinstance(raw, dict) or raw.get("status") == "stopped":
            return fallback

        host = str(raw.get("approval_bind_host") or "127.0.0.1").strip()
        if host not in {"127.0.0.1", "localhost", "::1"}:
            return fallback
        try:
            port = int(raw.get("approval_port") or self.settings.approval_port)
        except (TypeError, ValueError):
            return fallback
        if not 1 <= port <= 65535:
            return fallback

        provider = str(raw.get("remote_provider") or "local").casefold()
        if provider != "public":
            return {
                "transport": "local",
                "host": host,
                "port": port,
                "public_base_url": None,
                "access_secret_present": False,
                "password_present": False,
            }
        public_base_url = str(raw.get("public_base_url") or "").strip().rstrip("/")
        parsed = urlsplit(public_base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            return fallback

        secret_path_value = str(raw.get("approval_access_token_path") or "").strip()
        if not secret_path_value:
            return fallback
        try:
            secret_path = Path(secret_path_value).expanduser().resolve()
            runtime = self.settings.runtime.resolve()
            if secret_path != runtime and runtime not in secret_path.parents:
                return fallback
            secret = secret_path.read_text(encoding="ascii").strip()
        except (OSError, ValueError):
            return fallback
        if len(secret) < 64:
            return fallback
        # The long-lived secret authenticates service health only.
        # It must never be returned in a browser URL.
        password_path_value = str(raw.get("dashboard_password_path") or "").strip()
        if not password_path_value:
            return fallback
        try:
            password_path = Path(password_path_value).expanduser().resolve()
            if password_path != runtime and runtime not in password_path.parents:
                return fallback
        except (OSError, ValueError):
            return fallback
        return {
            "transport": "public",
            "host": host,
            "port": port,
            "public_base_url": public_base_url,
            "access_secret_present": True,
            "password_present": self._dashboard_password_present(password_path),
        }

    @staticmethod
    def _dashboard_password_present(path: Path) -> bool:
        try:
            password = path.read_text(encoding="utf-8-sig").strip()
        except OSError:
            return False
        return 12 <= len(password) <= 256 and "\n" not in password and "\r" not in password

    def browser_approval_url(self, proposal_id: str) -> str:
        return self._external_browser_url(f"/approve/{proposal_id}")

    def session_dashboard_url(self, session_id: str) -> str:
        """Return the single public entry URL shared by phones and computers."""
        return self._external_browser_url("/")

    def session_home_url(self, session_id: str) -> str:
        return self._external_browser_url(f"/session/{session_id}/home")

    def session_approval_url(self, session_id: str) -> str:
        return self._external_browser_url(f"/session/{session_id}/approval")

    def session_results_url(self, session_id: str) -> str:
        return self._external_browser_url(f"/session/{session_id}/results")

    def current_dashboard_session_id(self) -> str | None:
        row = self.db.one(
            "SELECT id FROM measurement_sessions "
            "WHERE status IN ('active','paused','halted','stopping',"
            "'recovery_required') ORDER BY updated_at DESC LIMIT 1"
        )
        return str(row["id"]) if row is not None else None

    def dashboard_session_id_for_workflow(self, workflow_id: str) -> str | None:
        row = self.db.one(
            "SELECT id FROM measurement_sessions WHERE workflow_id = ? "
            "ORDER BY started_at DESC LIMIT 1",
            (workflow_id,),
        )
        return str(row["id"]) if row is not None else None

    @property
    def operator_console_url(self) -> str:
        return f"http://127.0.0.1:{self.settings.port}/operator"

    def _external_browser_url(self, path: str) -> str:
        return f"{self.browser_approval_origin}{path}"

    def _start_dashboard_session(
        self,
        workflow_id: str,
        mode: str,
        source_client: str,
        *,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        if mode not in {"conversational", "autonomous"}:
            raise ServiceError("Unknown dashboard mode")
        existing = self.db.one(
            "SELECT * FROM measurement_sessions WHERE workflow_id = ? "
            "ORDER BY started_at DESC LIMIT 1",
            (workflow_id,),
        )
        created = False
        if existing is None:
            now = utc_now()
            session_id = uuid.uuid4().hex
            created = self.db.execute(
                "INSERT OR IGNORE INTO measurement_sessions("
                "id, workflow_id, mode, status, started_at, updated_at, "
                "source_client, autonomy_lease_id) "
                "VALUES (?, ?, ?, 'active', ?, ?, ?, ?)",
                (
                    session_id,
                    workflow_id,
                    mode,
                    now,
                    now,
                    source_client,
                    autonomy_lease_id,
                ),
            ) == 1
            existing = self.db.one(
                "SELECT * FROM measurement_sessions WHERE workflow_id = ?",
                (workflow_id,),
            )
            if created:
                self.db.event(
                    "dashboard_session_started",
                    source_client,
                    {
                        "session_id": session_id,
                        "mode": mode,
                        "autonomy_lease_id": autonomy_lease_id,
                    },
                    workflow_id,
                )
        if existing is None:
            raise ServiceError("Dashboard session could not be created")
        if not created:
            self.db.execute(
                "UPDATE measurement_sessions SET mode = ?, status = 'active', "
                "updated_at = ?, source_client = ?, autonomy_lease_id = ? "
                "WHERE id = ?",
                (
                    mode,
                    utc_now(),
                    source_client,
                    autonomy_lease_id,
                    existing["id"],
                ),
            )
            existing = self.db.one(
                "SELECT * FROM measurement_sessions WHERE id = ?", (existing["id"],)
            )
        return self._decode_dashboard_session(existing)

    def dashboard_status(self, session_id: str) -> dict[str, Any]:
        row = self.db.one(
            "SELECT * FROM measurement_sessions WHERE id = ?", (session_id,)
        )
        return self._decode_dashboard_session(row)

    def _decode_dashboard_session(
        self, row: dict[str, Any] | None
    ) -> dict[str, Any]:
        if row is None:
            raise ServiceError("Unknown measurement dashboard session")
        result = dict(row)
        result["browser_url"] = self.session_dashboard_url(str(result["id"]))
        result["home_url"] = self.session_home_url(str(result["id"]))
        result["approval_url"] = self.session_approval_url(str(result["id"]))
        result["results_url"] = self.session_results_url(str(result["id"]))
        result["workflow"] = self._decode_workflow(
            self._workflow(str(result["workflow_id"]))
        )
        lease_id = result.get("autonomy_lease_id")
        result["authorization"] = (
            self.autonomy_status(lease_id=str(lease_id)) if lease_id else None
        )
        result["operator_console_url"] = self.operator_console_url
        result["shutdown"] = self._shutdown_request_for_session(str(result["id"]))
        return result

    def dashboard_review(self, session_id: str) -> dict[str, Any]:
        raw_session = self.db.one(
            "SELECT workflow_id FROM measurement_sessions WHERE id = ?",
            (session_id,),
        )
        if raw_session is None:
            raise ServiceError("Unknown measurement dashboard session")
        failed_runs = self.db.all(
            "SELECT * FROM runs WHERE workflow_id = ? AND status = 'failed' "
            "ORDER BY started_at DESC LIMIT 20",
            (raw_session["workflow_id"],),
        )
        for failed_run in failed_runs:
            if self._is_instrument_connectivity_failure(failed_run):
                self._pause_for_instrument_connectivity(failed_run)
                break
        session = self.dashboard_status(session_id)
        workflow_id = str(session["workflow_id"])
        now = utc_now()
        self.db.execute(
            "UPDATE proposals SET status = 'expired' WHERE workflow_id = ? "
            "AND status = 'pending' AND expires_at <= ?",
            (workflow_id, now),
        )
        run_rows = self.db.all(
            # Runs carried over from an earlier workflow on this chip belong to
            # that workflow's results, not this session's.
            "SELECT id FROM runs WHERE workflow_id = ? "
            "AND inherited_from_run_id IS NULL ORDER BY rowid",
            (workflow_id,),
        )
        pending = None
        autonomy_lease_id = session.get("autonomy_lease_id")
        if autonomy_lease_id:
            # The initial autonomy proposal is created immediately before its
            # stable Dashboard session is bound.  Find that proposal through
            # the session's lease instead of excluding it with started_at.
            pending = self.db.one(
                "SELECT p.id FROM proposals p "
                "JOIN autonomy_leases a ON a.proposal_id = p.id "
                "WHERE a.id = ? AND a.workflow_id = ? "
                "AND p.workflow_id = ? AND p.status = 'pending'",
                (autonomy_lease_id, workflow_id, workflow_id),
            )
        if pending is None:
            pending = self.db.one(
                "SELECT id FROM proposals WHERE workflow_id = ? "
                "AND created_at >= ? AND status = 'pending' "
                "ORDER BY created_at DESC LIMIT 1",
                (workflow_id, session["started_at"]),
            )
        history: list[dict[str, Any]] = []
        all_plots: list[str] = []
        for row in run_rows:
            review = self._run_review(str(row["id"]), workflow_id)
            run_info = review.get("run") or {}
            parameters = run_info.get("parameters", {})
            review["duration_prediction"] = self.run_telemetry(
                str(run_info.get("node_id")),
                list(parameters.get("qubits", [])),
                limit=100,
            )["prediction"]
            item_plots = list(review.get("plots", []))
            review["asset_indices"] = list(
                range(len(all_plots), len(all_plots) + len(item_plots))
            )
            review["asset_tokens"] = [_asset_token(Path(p)) for p in item_plots]
            all_plots.extend(item_plots)
            history.append(review)
        revision = self.dashboard_ui_revision(session_id)
        self._dashboard_plot_cache[session_id] = (revision, list(all_plots))
        return {
            "session": session,
            "pending_proposal": (
                self.proposal(str(pending["id"])) if pending is not None else None
            ),
            "history": history,
            "plots": all_plots,
            "event_revision": revision,
        }

    def chip_usage(
        self,
        session_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        session_only: bool = False,
        day_zone: Any = timezone.utc,
    ) -> dict[str, Any]:
        """Model, token and estimated-cost usage of this session's chip.

        Operator request 2026-10-03, widened 2026-10-14 from the current
        workflow to every workflow on the chip, with an optional time range.
        ``overall`` is the chip since its first workflow; ``range`` is cut to
        [start, end) and, with ``session_only``, to this session's workflow.
        Read-only: parses the Claude Code transcripts; see ``usage.py``.
        """
        from .usage import chip_usage, default_transcript_dir, load_calls, parse_time

        config = self.policy.raw.get("usage", {})
        pricing = config.get("pricing", {})
        session = self.dashboard_status(session_id)
        workflow = self._workflow(str(session["workflow_id"]))
        device_key = str(workflow.get("device_key") or self._device_key())
        configured = config.get("transcript_dir")
        root = (
            Path(str(configured)).expanduser()
            if configured
            else default_transcript_dir(self.settings.agent_root)
        )
        result: dict[str, Any] = {
            "workflow_id": str(workflow["id"]),
            "device_key": device_key,
            "chip_name": Path(device_key).name,
            "session_started_at": workflow["created_at"],
            "transcript_dir": str(root) if root else None,
            "pricing": pricing,
        }
        if root is None or not root.is_dir():
            result["error"] = "Claude Code transcript folder was not found."
            return result
        workflows = self.db.all(
            "SELECT id, created_at, updated_at, status, device_key FROM workflows"
        )
        runs_by_workflow: dict[str, list[dict[str, Any]]] = {}
        # Runs carried over from an earlier workflow were never measured again.
        for row in self.db.all(
            "SELECT r.id, r.workflow_id, r.node_id, r.parameters_json, r.started_at "
            "FROM runs r JOIN workflows w ON w.id = r.workflow_id "
            "WHERE w.device_key = ? AND r.inherited_from_run_id IS NULL "
            "ORDER BY r.rowid",
            (device_key,),
        ):
            qubits = json_loads(row.get("parameters_json"), {}).get("qubits", [])
            runs_by_workflow.setdefault(str(row["workflow_id"]), []).append(
                {**row, "qubits": qubits if isinstance(qubits, list) else []}
            )
        calls = load_calls(root)
        overall = chip_usage(
            calls, workflows, runs_by_workflow, pricing, device_key, day_zone=day_zone
        )
        if session_only:
            opened = parse_time(workflow["created_at"])
            start = opened if start is None or opened is None else max(start, opened)
        ranged = (
            overall
            if start is None and end is None and not session_only
            else chip_usage(
                calls, workflows, runs_by_workflow, pricing, device_key,
                start, end, day_zone=day_zone,
                only_workflow=str(workflow["id"]) if session_only else None,
            )
        )
        result.update({"overall": overall, "range": ranged})
        return result

    def dashboard_review_asset(self, session_id: str, index: int) -> Path:
        # 2026-10-04: a page with 50 images used to rebuild the whole review
        # (every run plus its telemetry) once per image. Reuse the plot list
        # of the last review while no Dashboard event has happened since.
        cached = self._dashboard_plot_cache.get(session_id)
        if cached is not None and cached[0] == self.dashboard_ui_revision(session_id):
            plots = cached[1]
        else:
            plots = self.dashboard_review(session_id)["plots"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(plots)
        ):
            raise ServiceError("Unknown dashboard review asset")
        path = Path(plots[index]).resolve()
        if not self._is_allowed_review_asset(path) or not path.is_file():
            raise ServiceError("Dashboard review asset is unavailable")
        return path

    def _session_device_key(self, session_id: str) -> str:
        session = self.dashboard_status(session_id)
        workflow = self._workflow(str(session["workflow_id"]))
        return str(workflow.get("device_key") or self._device_key())

    def chip_history_for_session(self, session_id: str) -> dict[str, Any]:
        """Every successful result of this session's chip (operator 2026-10-13)."""
        from .chip_history import collect_chip_history

        return collect_chip_history(self, self._session_device_key(session_id))

    def chip_history_asset(self, session_id: str, run_id: str, position: int) -> Path:
        from .chip_history import chip_history_asset

        try:
            return chip_history_asset(
                self, self._session_device_key(session_id), run_id, position
            )
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc

    def _refresh_static_chip_history(self, workflow_id: str) -> None:
        """Rewrite runtime/chip_history/<chip>/ after an analysis; never fails it."""
        from .chip_history import write_static_chip_history

        try:
            workflow = self._workflow(workflow_id)
            write_static_chip_history(
                self, str(workflow.get("device_key") or self._device_key())
            )
        except Exception as exc:  # pragma: no cover - a report must not block analysis
            self.db.event(
                "chip_history_report_failed",
                "deterministic_analyzer",
                {"error": str(exc)},
                workflow_id,
            )

    def dashboard_ui_revision(self, session_id: str) -> int:
        session = self.dashboard_status(session_id)
        row = self.db.one(
            "SELECT MAX(id) AS id FROM events WHERE workflow_id = ? "
            "AND created_at >= ? AND event_type IN ("
            "'proposal_created','proposal_approved','run_started','run_analyzed',"
            "'run_failed','run_stopped','run_cancelled_by_shutdown','decision_recorded','autonomy_pause','autonomy_stop',"
            "'autonomy_emergency_stop','autonomy_halted','autonomy_lease_expired',"
            "'autonomy_scope_completed','autonomy_resumed','measurement_mode_resumed',"
            "'measurement_paused_for_instrument_error','instrument_error_paused',"
            "'workflow_stopped','workflow_quarantined',"
            "'full_shutdown_requested','hardware_lock_recovered')",
            (session["workflow_id"], session["started_at"]),
        )
        return int(row["id"] or 0) if row is not None else 0

    def dashboard_ui_events_after(
        self,
        session_id: str,
        after_id: int,
        event_types: Collection[str] | None = None,
    ) -> dict[str, Any]:
        session = self.dashboard_status(session_id)
        rows = self.db.all(
            "SELECT id, event_type FROM events WHERE workflow_id = ? "
            "AND created_at >= ? AND id > ? AND event_type IN ("
            "'proposal_created','proposal_approved','run_started','run_analyzed',"
            "'run_failed','run_stopped','run_cancelled_by_shutdown','decision_recorded','autonomy_pause','autonomy_stop',"
            "'autonomy_emergency_stop','autonomy_halted','autonomy_lease_expired',"
            "'autonomy_scope_completed','autonomy_resumed','measurement_mode_resumed',"
            "'measurement_paused_for_instrument_error','instrument_error_paused',"
            "'workflow_stopped','workflow_quarantined',"
            "'full_shutdown_requested','hardware_lock_recovered') "
            "ORDER BY id LIMIT 250",
            (session["workflow_id"], session["started_at"], int(after_id)),
        )
        selected = set(event_types) if event_types is not None else None
        return {
            "cursor": max(
                (int(row["id"]) for row in rows), default=int(after_id)
            ),
            "events": [
                {"id": int(row["id"]), "event_type": row["event_type"]}
                for row in rows
                if selected is None or row["event_type"] in selected
            ],
        }

    def approval_csrf_token(self, proposal_id: str) -> str:
        proposal = self.db.one("SELECT * FROM proposals WHERE id = ?", (proposal_id,))
        if proposal is None:
            raise ServiceError(f"Unknown proposal: {proposal_id}")
        message = "\0".join(
            (
                proposal["id"],
                proposal["workflow_id"],
                proposal["created_at"],
                proposal["expires_at"],
            )
        ).encode("utf-8")
        return hmac.new(self._approval_secret, message, hashlib.sha256).hexdigest()

    def preview_lease_options(
        self, proposal_id: str, lease_options: dict[str, Any]
    ) -> dict[str, Any]:
        """The nodes a lease would run with these choices, without applying them.

        Operator request 2026-10-10: the approval page shows this list before
        the agree box and the approval code. The choices are applied inside a
        transaction that is always rolled back, so the preview is exactly what
        the approval would do.
        """

        class _Rollback(Exception):
            pass

        preview: dict[str, Any] = {}
        try:
            with self.db.transaction(immediate=True) as connection:
                row = connection.execute(
                    "SELECT * FROM proposals WHERE id = ?", (proposal_id,)
                ).fetchone()
                if row is None or row["kind"] != "autonomy_lease":
                    raise ServiceError("Only an automatic-measurement approval has a preview.")
                if row["status"] != "pending":
                    raise ServiceError(f"Proposal status is {row['status']}, not pending")
                proposal = dict(row)
                self._apply_lease_options(
                    connection, proposal, lease_options, "preview", "preview"
                )
                workflow = dict(
                    connection.execute(
                        "SELECT * FROM workflows WHERE id = ?", (proposal["workflow_id"],)
                    ).fetchone()
                )
                lease = connection.execute(
                    "SELECT allowed_nodes_json FROM autonomy_leases WHERE proposal_id = ?",
                    (proposal_id,),
                ).fetchone()
                allowed = (
                    set(json_loads(lease["allowed_nodes_json"], []))
                    if lease is not None
                    else None
                )
                options = self._workflow_options(workflow)
                sequence = self._workflow_sequence(workflow)
                full = list(self.settings.workflow_sequence)
                current = str(workflow["current_node"])
                start_index = full.index(current) if current in full else 0
                flux_nodes = set(self._flux_nodes())
                nodes = []
                for index, node in enumerate(full):
                    if node in flux_nodes and options.get("flux_calibration") is not True:
                        status = "skipped_flux"
                    elif index < start_index:
                        status = "inherited" if workflow.get("start_node") else "not_run"
                    elif node in sequence and (allowed is None or node in allowed):
                        status = "run"
                    else:
                        status = "skipped"
                    nodes.append({"node": node, "status": status})
                history = (
                    json_loads(proposal.get("payload_json"), {})
                    .get("workflow_options_offer", {})
                    .get("start_node")
                ) or {}
                qubit_plan = [
                    {"qubit": name, "start_node": self._qubit_start_node(workflow, name)}
                    for name in (history.get("qubit_start_nodes") or {})
                    if workflow.get("start_node") == history.get("serialize_from_node")
                ]
                preview = {
                    "nodes": nodes,
                    "qubit_starts": qubit_plan,
                    "start_node": workflow.get("start_node"),
                    "current_node": current,
                    "flux_calibration": options.get("flux_calibration"),
                    "drive_mode": options.get("drive_mode"),
                }
                raise _Rollback()
        except _Rollback:
            pass
        return preview

    def lease_preview_digest(self, proposal_id: str, lease_options: dict[str, Any]) -> str:
        """Binds an approval to the exact choices its preview showed."""
        message = proposal_id + "\0" + json_dumps(lease_options)
        return hmac.new(
            self._approval_secret, message.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def approve_from_browser(
        self,
        proposal_id: str,
        confirmation: str,
        csrf_token: str,
        remote_host: str,
        *,
        actor: str | None = None,
        approval_method: str = "local_browser",
        lease_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        expected_confirmation = f"APPROVE {proposal_id}"
        if not hmac.compare_digest(confirmation, expected_confirmation):
            raise ServiceError("The confirmation text did not match exactly.")
        expected_token = self.approval_csrf_token(proposal_id)
        if not hmac.compare_digest(csrf_token, expected_token):
            raise ServiceError("The browser approval token was invalid.")
        resolved_actor = actor or (
            f"{getpass.getuser()} via local browser ({remote_host})"
        )
        if lease_options is None:
            pending = self.db.one(
                "SELECT kind, payload_json FROM proposals WHERE id = ?", (proposal_id,)
            )
            if (
                pending is not None
                and pending["kind"] == "autonomy_lease"
                and isinstance(
                    json_loads(pending["payload_json"], {}).get("workflow_options_offer"),
                    dict,
                )
            ):
                # Operator request 2026-10-04: automatic measurement starts
                # only after the choices on the approval page.
                raise ServiceError(
                    "Choose the optional experiments and the measurement mode "
                    "on the approval page before approving automatic measurement."
                )
        return self._approve_pending_proposal(
            proposal_id,
            resolved_actor,
            approval_method,
            lease_options=lease_options,
        )

    def proposal_review(self, proposal_id: str) -> dict[str, Any]:
        """Return prior run evidence that explains the proposed next action."""
        proposal = self.db.one("SELECT * FROM proposals WHERE id = ?", (proposal_id,))
        if proposal is None:
            raise ServiceError(f"Unknown proposal: {proposal_id}")
        payload = json_loads(proposal["payload_json"], {})
        run_id = payload.get("run_id") or payload.get("review_run_id")
        if not run_id and proposal["kind"] == "run":
            run_id = self._latest_review_run_id(
                proposal["workflow_id"], str(payload.get("node_id", ""))
            )
        if not run_id:
            run_id = self._latest_workflow_decision_run_id(
                proposal["workflow_id"]
            )
        if not isinstance(run_id, str) or not run_id:
            return {
                "available": False,
                "run": None,
                "analysis": None,
                "decision": None,
                "plots": [],
            }
        return self._run_review(str(run_id), proposal["workflow_id"])

    def autonomy_review(self, lease_id: str) -> dict[str, Any]:
        """Return every run/result/decision accumulated under one authorization."""
        lease = self._autonomy_row(lease_id)
        event_revision = self.autonomy_ui_revision(lease_id)
        runs = self.db.all(
            "SELECT id FROM runs WHERE autonomy_lease_id = ? "
            "ORDER BY rowid",
            (lease_id,),
        )
        if not runs:
            return {
                "available": False,
                "run": None,
                "analysis": None,
                "decision": None,
                "plots": [],
                "history": [],
                "duration_prediction": None,
                "event_revision": event_revision,
            }
        history: list[dict[str, Any]] = []
        all_plots: list[str] = []
        for row in runs:
            review = self._run_review(str(row["id"]), lease["workflow_id"])
            run_info = review.get("run") or {}
            parameters = run_info.get("parameters", {})
            review["duration_prediction"] = self.run_telemetry(
                str(run_info.get("node_id")),
                list(parameters.get("qubits", [])),
                limit=100,
            )["prediction"]
            item_plots = list(review.get("plots", []))
            review["asset_indices"] = list(
                range(len(all_plots), len(all_plots) + len(item_plots))
            )
            all_plots.extend(item_plots)
            history.append(review)
        latest = dict(history[-1])
        latest["history"] = history
        latest["plots"] = all_plots
        latest["event_revision"] = event_revision
        return latest

    def autonomy_review_asset(self, lease_id: str, index: int) -> Path:
        review = self.autonomy_review(lease_id)
        plots = review["plots"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(plots)
        ):
            raise ServiceError("Unknown autonomy review asset")
        path = Path(plots[index]).resolve()
        if not self._is_allowed_review_asset(path) or not path.is_file():
            raise ServiceError("Autonomy review asset is unavailable")
        return path

    def autonomy_ui_revision(self, lease_id: str) -> int:
        """Return an event cursor captured before rendering an autonomy page."""
        lease = self._autonomy_row(lease_id)
        event_types = tuple(self._autonomy_ui_event_types())
        placeholders = ", ".join("?" for _ in event_types)
        row = self.db.one(
            f"SELECT MAX(id) AS id FROM events WHERE workflow_id = ? "
            f"AND event_type IN ({placeholders})",
            (lease["workflow_id"], *event_types),
        )
        return int(row["id"] or 0) if row is not None else 0

    def autonomy_ui_events_after(
        self, lease_id: str, after_id: int
    ) -> dict[str, Any]:
        """Return result/control events relevant to one autonomy lease."""
        lease = self._autonomy_row(lease_id)
        event_types = tuple(self._autonomy_ui_event_types())
        placeholders = ", ".join("?" for _ in event_types)
        rows = self.db.all(
            f"SELECT id, event_type, payload_json FROM events "
            f"WHERE workflow_id = ? AND id > ? "
            f"AND event_type IN ({placeholders}) ORDER BY id LIMIT 250",
            (lease["workflow_id"], int(after_id), *event_types),
        )
        cursor = max((int(row["id"]) for row in rows), default=int(after_id))
        relevant: list[dict[str, Any]] = []
        for row in rows:
            payload = json_loads(row.get("payload_json"), {})
            if payload.get("lease_id") == lease_id:
                relevant.append(
                    {"id": int(row["id"]), "event_type": row["event_type"]}
                )
                continue
            run_id = payload.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                continue
            run = self.db.one(
                "SELECT autonomy_lease_id FROM runs WHERE id = ?", (run_id,)
            )
            if run is not None and run.get("autonomy_lease_id") == lease_id:
                relevant.append(
                    {"id": int(row["id"]), "event_type": row["event_type"]}
                )
        return {"cursor": cursor, "events": relevant}

    @staticmethod
    def _autonomy_ui_event_types() -> frozenset[str]:
        return frozenset(
            {
                "run_analyzed",
                "run_failed",
                "run_stopped",
                "run_cancelled_by_shutdown",
                "run_force_stopped",
                "decision_recorded",
                "autonomy_pause",
                "autonomy_stop",
                "autonomy_emergency_stop",
                "autonomy_halted",
                "autonomy_lease_expired",
                "autonomy_scope_completed",
                "autonomy_resumed",
                "autonomy_targets_scientifically_unmeasurable",
            }
        )

    #: Nodes whose playbook recovery for a noisy-but-interior trace is the
    #: configured averaging ladder rather than a new window or power.
    SNR_LADDER_NODES = ("03a", "04", "05")

    def next_action(self, workflow_id: str) -> dict[str, Any]:
        result = self._next_action_plan(workflow_id)
        run = result.get("run")
        if result.get("action") == "run" and isinstance(run, dict):
            node_id = str(run.get("node_id"))
            parameters = run.get("parameters")
            if (
                isinstance(parameters, dict)
                and self.policy.shared_xy_drive.get("statistics_readout_node")
                and self._uses_state_discrimination(node_id, parameters)
            ):
                requested = {str(name) for name in parameters.get("qubits", [])}
                if requested - self._state_discrimination_qualified_targets(workflow_id):
                    # Operator decision 2026-10-14: one unqualified readout
                    # puts the whole multiplexed run on the raw I/Q signal.
                    parameters["use_state_discrimination"] = False
                    result.setdefault("rules", []).append(
                        "PLAYBOOK: 07b readout gates state discrimination"
                    )
        return result

    def _next_action_plan(self, workflow_id: str) -> dict[str, Any]:
        """Say deterministically what the workflow needs next, and why.

        This is an aggregator, not a second opinion: every field comes from the
        same server state the guards already enforce -- current node, per-target
        resolution, attempt quota, the shared-first-batch rule, the registered
        deterministic setup tools, and the configured averaging ladder. When no
        registered rule covers the situation it says so and cites the playbook
        section to read, rather than inventing a parameter.
        """

        workflow = self._workflow(workflow_id)
        node_id = str(workflow["current_node"])
        sequence = self._workflow_sequence(workflow)
        active = self._active_targets_for_node(workflow, node_id)
        resolved = self._node_target_resolution(workflow_id, node_id) & active
        incomplete = self._autonomy_incomplete_targets_for_node(workflow_id, node_id)
        unresolved = active - resolved - incomplete
        if self._node_is_single_pass(node_id):
            # One pass per target, whatever its fit says. Without this a
            # per-qubit split of 07d/07b would keep proposing the first target,
            # because these nodes register no per-target resolution rule.
            unresolved -= self._single_pass_measured_targets(workflow, node_id)
        next_node = (
            self._expected_next_node(workflow, node_id)
            if node_id in sequence
            else None
        )
        serial = self._serial_pass_state(workflow)
        latest = self.db.one(
            "SELECT r.*, d.id AS decision_id FROM runs r "
            "LEFT JOIN decisions d ON d.run_id = r.id "
            "WHERE r.workflow_id = ? AND r.node_id = ? ORDER BY r.rowid DESC LIMIT 1",
            (workflow_id, node_id),
        )
        result: dict[str, Any] = {
            "workflow_id": workflow_id,
            "workflow_status": workflow["status"],
            "current_node": node_id,
            "next_node_in_sequence": next_node,
            "targets": {
                "active": sorted(active),
                "resolved": sorted(resolved),
                "incomplete": sorted(incomplete),
                "unresolved": sorted(unresolved),
            },
            "latest_run": (
                {
                    "run_id": latest["id"],
                    "status": latest["status"],
                    "analysis_status": latest.get("analysis_status"),
                    "parameters": json_loads(latest.get("parameters_json"), {}),
                    "has_decision": latest.get("decision_id") is not None,
                    "evidence": evidence_digest(
                        json_loads(latest.get("analysis_json"), None)
                    ),
                }
                if latest is not None
                else None
            ),
            "rules": [],
            "required_setup": [],
            "notes": [],
        }
        operator_parameters = (
            self._workflow_options(workflow).get("node_parameters") or {}
        ).get(node_id)
        if operator_parameters:
            result["operator_parameters"] = operator_parameters
            result["notes"].append(
                f"The operator set {sorted(operator_parameters)} for {node_id} on "
                "the lease page; every run of this node uses them automatically."
            )
        if serial is not None:
            result["serial_pass"] = {
                "qubit": serial["current"],
                "position": serial["index"],
                "total": serial["total"],
                "finished": serial["finished"],
                "remaining": serial["remaining"],
                "restart_node": serial["node"],
            }
            result["rules"].append(
                "PLAYBOOK: A shared XY drive output runs one qubit at a time"
            )
        result["attempts"] = self._attempts_for_node(workflow_id, node_id, active)

        if workflow["status"] != "active":
            result["action"] = "blocked"
            result["reason"] = (
                f"The workflow is {workflow['status']}; resume it before scheduling."
            )
            return result
        if latest is not None and latest["status"] in {
            "starting",
            "running",
            "stopping",
        }:
            result["action"] = "wait"
            result["reason"] = (
                f"Run {latest['id']} is {latest['status']}. Wait for a terminal "
                "status before analyzing."
            )
            return result
        if (
            latest is not None
            and latest["status"] == "completed"
            and latest.get("analysis_status")
            not in {"pass", "needs_review", "failed"}
        ):
            # `analysis_status` defaults to 'not_started', so test the three
            # analyzed values rather than NULL.
            result["action"] = "analyze"
            result["reason"] = f"Run {latest['id']} completed and is not analyzed yet."
            return result
        if latest is not None and not latest.get("decision_id"):
            result["action"] = "record_decision"
            result["reason"] = (
                f"Run {latest['id']} has no recorded decision. A node cannot be "
                "left without one; see 'Record `advance`, not `manual_review`, "
                "on the run that finishes a node'."
            )
            result["rules"].append(
                "PLAYBOOK: Record `advance`, not `manual_review`, on the run "
                "that finishes a node"
            )
            return result

        single_pass_done = self._node_is_single_pass(node_id) and not (
            active - self._single_pass_measured_targets(workflow, node_id)
        ) and self._node_has_completed_run(workflow_id, node_id)
        if single_pass_done:
            repeat_targets = self._active_reset_repeat_targets(workflow)
            if repeat_targets:
                repeat_targets = (
                    repeat_targets
                    - self._active_reset_repeated_targets(workflow)
                ) & active
            if repeat_targets and self._is_active_reset_repeat(
                workflow,
                node_id,
                {
                    "reset_type_thermal_or_active": "active",
                    "qubits": sorted(repeat_targets),
                },
            ):
                batch = sorted(repeat_targets)
                repeat_cap = self._node_multiplex_cap(node_id)
                if repeat_cap is not None and len(batch) > repeat_cap:
                    batch = batch[:repeat_cap]
                result["action"] = "run"
                result["run"] = {
                    "node_id": node_id,
                    "qubits": batch,
                    "parameters": {
                        "qubits": batch,
                        "reset_type_thermal_or_active": "active",
                    },
                }
                result["reason"] = (
                    f"{len(batch)} target(s) cleared the active-reset trigger "
                    "fidelity on the thermal run, so repeat 07b once with "
                    "active reset. That repeat is their accepted result and "
                    "qualifies them to use active reset downstream."
                )
                result["rules"].append(
                    "PLAYBOOK: 07d and 07b run once on defaults"
                )
                return result
            result["action"] = "advance"
            result["reason"] = (
                f"{node_id} runs once on the node defaults and then advances. "
                "Its run is complete and analyzed, so record `advance` and move "
                "to the next node; do not retry or adjust parameters."
            )
            result["rules"].append(
                "PLAYBOOK: 07d and 07b run once on defaults"
            )
            result["run"] = None
            return result

        try:
            missing_extras = self._missing_node_extras(self._workflow(workflow_id), None)
        except Exception:  # pragma: no cover - advisory only
            missing_extras = {}
        try:
            long_readout = self._readout_length_normalize_patch(
                self._workflow(workflow_id), None
            )[1]
        except Exception:  # pragma: no cover - advisory only
            long_readout = {}
        try:
            rescale = self._readout_full_scale_plan(self._workflow(workflow_id), None)[1]
        except Exception:  # pragma: no cover - advisory only
            rescale = {}
        if rescale:
            result.setdefault("notes", []).append(
                f"Readout full scale can drop on {sorted(rescale)} now that its "
                "measurable qubits passed 02c; call "
                "jy_request_readout_full_scale_rescale (with the lease id) and "
                "apply it before the next run."
            )
        if long_readout:
            result.setdefault("notes", []).append(
                f"Readout length or depletion time above the cap for {sorted(long_readout)}; "
                "call jy_request_readout_length_normalize (with the lease id) and "
                "apply it before the next run."
            )
        if missing_extras:
            result.setdefault("notes", []).append(
                "Extras keys later nodes record are missing for "
                f"{sorted(missing_extras)}; call jy_request_node_extras_prerequisites "
                "(with the lease id) and apply it before the next run, or "
                "Qualibrate's state recorder raises KeyError."
            )
        if serial is not None and serial["current"] in unresolved:
            pass_qubit = str(serial["current"])
            if self._deferral_pending(workflow, node_id, pass_qubit):
                result["notes"].append(
                    f"Operator rule 2026-10-12: {node_id} may be deferred for "
                    f"{pass_qubit}. If the averaging ladder cannot reach the floor, "
                    f"record `advance` to the next node; {node_id} is revisited "
                    "after 07b with the optimised readout."
                )
            if node_id == "04":
                allowed, why = self._04_return_check(workflow, pass_qubit)
                if allowed:
                    result["notes"].append(
                        f"Operator rule 2026-10-12: {pass_qubit} may return to 03a "
                        f"({why}). Record `advance` to 03a; its current line is "
                        "then masked and 03a searches for another."
                    )
                else:
                    cap = self._xy_gate_length_cap(pass_qubit)
                    result["notes"].append(
                        "Operator rule 2026-10-12: if the amplitude sweep is at its "
                        "limit and the signal changes without turning over, call "
                        "jy_request_04_gate_time to lengthen the XY gate"
                        + (f" (at most {cap:g} ns)" if cap is not None else "")
                        + f". Returning to 03a is not yet allowed: {why}."
                    )
        result["required_setup"] = self._required_setup_for_node(
            node_id, sorted(active), workflow_id
        )
        if result["required_setup"]:
            result["action"] = "setup"
            result["reason"] = (
                "A registered deterministic setup must be applied before this "
                "node can run."
            )
            return result

        if not unresolved:
            result["action"] = "advance"
            result["reason"] = (
                f"Every active {node_id} target is resolved or incomplete at its "
                "scientific boundary."
                if incomplete
                else f"Every active {node_id} target is resolved."
            )
            if serial is not None and next_node == serial["node"]:
                following = [
                    name
                    for name in serial["remaining"]
                    if name != serial["current"]
                ]
                result["reason"] = (
                    f"{serial['current']} has finished its pass at {node_id}. "
                    f"Advance back to {serial['node']} and start "
                    f"{following[0] if following else 'the next qubit'}."
                )
            elif serial is not None and next_node is None:
                result["reason"] = (
                    f"{serial['current']} was the last qubit of the per-qubit "
                    "pass, so this advance completes the workflow."
                )
            result["run"] = None
            return result

        cap = self._node_multiplex_cap(node_id)
        groups = self._reset_groups(workflow, node_id)
        if groups:
            reset_name = self._reset_parameter_name(node_id)
            for group in ("active", "thermal"):
                members = groups.get(group, set())
                pending = members & unresolved
                if not pending:
                    continue
                first = not self._group_has_completed_run(
                    workflow, node_id, group, members
                )
                batch = sorted(members if first else pending)
                if cap is not None and len(batch) > cap:
                    batch = batch[:cap]
                result["action"] = "run"
                result["run"] = {
                    "node_id": node_id,
                    "qubits": batch,
                    "parameters": {"qubits": batch, reset_name: group},
                }
                result["reason"] = (
                    f"{node_id} runs its {group}-reset group separately "
                    f"({'first shared batch' if first else 'retry of unresolved targets'}); "
                    "qubits qualified by the active-reset 07b use active reset, "
                    "the rest thermal."
                )
                result["rules"].append("PLAYBOOK: Reset groups after 07b")
                return result
        first_batch = node_id in SHARED_FIRST_BATCH_NODES and not (
            self._node_has_completed_run(workflow_id, node_id)
        )
        if first_batch and not (cap is not None and len(active) > cap):
            result["action"] = "run"
            result["run"] = {
                "node_id": node_id,
                "qubits": sorted(active),
                "parameters": {"qubits": sorted(active)},
            }
            result["reason"] = (
                f"Start {node_id} for {serial['current']}, qubit "
                f"{serial['index']} of {serial['total']} in the per-qubit pass; "
                "node defaults apply to everything else."
                if serial is not None
                else f"The first {node_id} run must multiplex every active "
                "target with one shared parameter set; node defaults apply to "
                "the rest."
            )
            result["rules"].append("PLAYBOOK: Multiplex workflows")
            return result

        batch = sorted(unresolved)
        capped = cap is not None and len(batch) > cap
        if capped:
            batch = batch[:cap]
        result["action"] = "run"
        result["run"] = {
            "node_id": node_id,
            "qubits": batch,
            "parameters": {"qubits": batch},
        }
        if capped:
            result["reason"] = (
                f"{node_id} multiplexes at most {cap} targets per run; take the "
                f"next group of {len(batch)} and repeat the node for the rest."
            )
        elif serial is not None:
            result["reason"] = (
                f"Retry {node_id} for {batch[0]}, the qubit this pass is on. "
                "Its pass holds the node until it resolves, is marked "
                "scientifically unmeasurable, or runs out of attempts."
            )
        else:
            result["reason"] = (
                f"Retry the unresolved {node_id} targets together; omit the "
                "resolved ones."
            )
        result["rules"].append("PLAYBOOK: Multiplex workflows")
        ladder = self._ladder_recommendation(latest, node_id, unresolved)
        if ladder is not None:
            result["run"]["parameters"] = ladder["parameters"]
            result["run"]["qubits"] = ladder["parameters"]["qubits"]
            result["reason"] = ladder["reason"]
            result["rules"].append(ladder["rule"])
        else:
            result["notes"].append(
                "No registered deterministic parameter change covers this "
                f"situation. Read the '{node_id}' section of the playbook and "
                f"`rules/experiences/{node_id}.md` before choosing parameters."
            )
        return result

    def _attempts_for_node(
        self, workflow_id: str, node_id: str, targets: Collection[str]
    ) -> dict[str, Any]:
        """Per-target attempts used and left under the current lease."""

        lease = self.db.one(
            "SELECT * FROM autonomy_leases WHERE workflow_id = ? "
            "AND status IN ('active', 'paused') ORDER BY created_at DESC LIMIT 1",
            (workflow_id,),
        )
        if lease is None:
            return {"lease_id": None, "used": {}, "remaining": {}, "maximum": None}
        maximum = int(lease["max_attempts_per_node_qubit"])
        used = {str(name): 0 for name in targets}
        for run in self.db.all(
            "SELECT parameters_json FROM runs WHERE autonomy_lease_id = ? "
            "AND node_id = ?",
            (lease["id"], node_id),
        ):
            prior = set(json_loads(run.get("parameters_json"), {}).get("qubits", []))
            for name in used:
                if name in prior:
                    used[name] += 1
        return {
            "lease_id": lease["id"],
            "maximum": maximum,
            "used": used,
            "remaining": {
                name: max(0, maximum - count) for name, count in used.items()
            },
        }

    def _required_setup_for_node(
        self, node_id: str, targets: list[str], workflow_id: str
    ) -> list[dict[str, str]]:
        """The deterministic setup tools the playbook requires before a node."""

        required: list[dict[str, str]] = []
        try:
            fresh = self._fresh_qubits(self._workflow(workflow_id))
        except ServiceError:
            fresh = []
        if fresh:
            required.append(
                {
                    "tool": "jy_request_fresh_qubit_defaults",
                    "targets": ", ".join(fresh),
                    "why": (
                        "These qubits have no measurement record on this chip; "
                        "empty their extras and write the starting values "
                        "before any run."
                    ),
                }
            )
        try:
            state = load_state(self.settings.active_state)
        except Exception:
            return required
        qubits = state.get("qubits", {}) if isinstance(state, dict) else {}
        try:
            workflow_targets = json_loads(
                self._workflow(workflow_id).get("targets_json"), []
            )
            over_budget = self.policy.readout_amplitude_budget_patch(
                sorted(workflow_targets), state
            )[1]
        except Exception:  # pragma: no cover - wiring problems surface elsewhere
            over_budget = {}
        if over_budget:
            required.append(
                {
                    "tool": "jy_request_readout_amplitude_budget",
                    "targets": ", ".join(
                        sorted(
                            name
                            for line in over_budget.values()
                            for name in line["qubits"]
                        )
                    ),
                    "why": (
                        "A readout line's amplitudes sum above "
                        "max_aggregate_readout_amplitude; scale them down before "
                        "any run."
                    ),
                }
            )
        try:
            probe_plan = self._02x_probe_setup_plan(self._workflow(workflow_id), state)[1]
        except Exception:  # pragma: no cover - wiring problems surface elsewhere
            probe_plan = {}
        if probe_plan:
            required.append(
                {
                    "tool": "jy_request_02x_probe_setup",
                    "targets": ", ".join(
                        sorted(
                            set(probe_plan.get("grid_location", {}))
                            | set(probe_plan.get("thread", {}))
                            | {
                                name
                                for line in probe_plan.get("probe_headroom", {}).values()
                                for name in line["probed_qubits"]
                            }
                        )
                    ),
                    "why": (
                        "Qubits share a QUA thread (multiplexed readout then "
                        "reports one element on every channel) or a "
                        "grid_location (their plots overlap), or 02x's fixed "
                        "-10 dBm probe tones would sum above 1 on a readout line "
                        "and clip; apply this setup first."
                    ),
                }
            )
        if node_id == "02":
            try:
                plan = self._readout_lo_recenter_plan(self._workflow(workflow_id), state)
            except ServiceError:
                plan = {}
            if plan.get("patch"):
                required.append(
                    {
                        "tool": "jy_request_readout_lo_recenter",
                        "targets": ", ".join(sorted(plan.get("qubits", []))),
                        "why": (
                            "Some design resonator frequencies lie outside the "
                            "readout LO +- max IF window of 02; move the readout "
                            "LO so every design frequency is covered."
                        ),
                    }
                )
        if node_id in {"03a", "04", "05"}:
            missing_x180 = [
                name
                for name in targets
                if isinstance(qubits.get(name), dict)
                and operation_amplitude(qubits[name], "x180") == 0
            ]
            if missing_x180:
                required.append(
                    {
                        "tool": "jy_request_bootstrap",
                        "targets": ", ".join(missing_x180),
                        "why": (
                            "x180 amplitude is zero; 03a/04/05 cannot drive the "
                            "qubit until bootstrap is applied."
                        ),
                    }
                )
        if node_id == "03a" and not self._initial_03a_setup_taken(
            self._workflow(workflow_id)
        ):
            required.append(
                {
                    "tool": "jy_request_initial_03a_zero_if",
                    "targets": ", ".join(targets),
                    "why": (
                        "The first 03a coarse search requires every target's "
                        "private LO on its RF with XY IF at zero."
                    ),
                }
            )
        try:
            lo_patch, lo_details = self._xy_lo_plan(
                self._workflow(workflow_id), node_id, state
            )
        except Exception:  # pragma: no cover - surfaced by the request itself
            lo_patch, lo_details = [], {}
        if lo_patch:
            required.append(
                {
                    "tool": "jy_request_xy_lo_plan",
                    "targets": str(lo_details.get("focus", "")),
                    "why": (
                        "Operator rule 2026-10-12: before each node after 03a the "
                        "shared XY LO is placed for the qubit whose pass is running "
                        f"({lo_details.get('mode')} LO "
                        f"{float(lo_details.get('lo_hz', 0.0)) / 1e9:g} GHz) and "
                        "recorded in extras.xy_lo_freq."
                    ),
                }
            )
        if node_id == "07b":
            missing_fidelity = [
                name
                for name in targets
                if "readout_fidelity"
                not in (qubits.get(name, {}).get("extras", {}) or {})
            ]
            if missing_fidelity:
                required.append(
                    {
                        "tool": "jy_request_07b_prerequisites",
                        "targets": ", ".join(missing_fidelity),
                        "why": (
                            "extras/readout_fidelity is missing, so Qualibrate's "
                            "state recorder cannot observe an old value."
                        ),
                    }
                )
        return required

    def _ladder_recommendation(
        self, latest: dict[str, Any] | None, node_id: str, unresolved: set[str]
    ) -> dict[str, Any] | None:
        """The configured averaging ladder, when it applies to this retry."""

        if latest is None or node_id not in self.SNR_LADDER_NODES:
            return None
        if latest["status"] != "completed" or not latest.get("analysis_json"):
            return None
        try:
            guidance = self.snr_retry_recommendation(str(latest["id"]))
        except ServiceError:
            return None
        if not guidance.get("retry_recommended"):
            return None
        low_snr = set(guidance.get("low_snr_targets", []))
        if not low_snr or not low_snr <= unresolved:
            return None
        parameters = dict(guidance["retry_parameters"])
        parameters["qubits"] = sorted(low_snr)
        return {
            "parameters": parameters,
            "reason": guidance["reason"],
            "rule": (
                f"policies.yaml analysis.{node_id}.noise_confirmation_num_averages"
            ),
        }

    def snr_retry_recommendation(self, run_id: str) -> dict[str, Any]:
        """Recommend the next safe averaging step for a noisy 03a/04/05 result."""
        run = self.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
        if run is None:
            raise ServiceError(f"Unknown run: {run_id}")
        node_id = str(run["node_id"])
        if node_id not in self.SNR_LADDER_NODES:
            raise ServiceError(
                "SNR retry guidance is available only for nodes "
                f"{', '.join(self.SNR_LADDER_NODES)}"
            )
        analysis = json_loads(run.get("analysis_json"), {})
        if run["status"] != "completed" or not analysis:
            raise ServiceError(
                f"Analyze a completed {node_id} run before requesting guidance"
            )
        parameters = json_loads(run.get("parameters_json"), {})
        rules = self.policy.raw["analysis"][node_id]
        minimum = float(rules["min_robust_snr"])
        metrics = analysis.get("dataset_metrics", {}).get("qubits", {})
        requested = [str(name) for name in parameters.get("qubits", [])]
        low_snr: list[str] = []
        observed: dict[str, float | None] = {}
        for name in requested:
            item = metrics.get(name, {}) if isinstance(metrics, dict) else {}
            value = item.get("robust_snr") if isinstance(item, dict) else None
            observed[name] = float(value) if _finite_number(value) else None
            if not _finite_number(value) or float(value) < minimum:
                low_snr.append(name)
        current = parameters.get("num_averages")
        current_averages = (
            int(current)
            if isinstance(current, int) and not isinstance(current, bool)
            else 0
        )
        ladder = sorted(
            {
                int(value)
                for value in rules["noise_confirmation_num_averages"]
                if int(value) <= int(rules["max_noise_confirmation_num_averages"])
            }
        )
        next_averages = next(
            (value for value in ladder if value > current_averages), None
        )
        retry_parameters = dict(parameters)
        retry_parameters["qubits"] = low_snr
        if next_averages is not None:
            retry_parameters["num_averages"] = next_averages
            retry_parameters["load_data_id"] = None
            retry_parameters["simulate"] = False
        retry_recommended = bool(low_snr and next_averages is not None)
        return {
            "run_id": run_id,
            "node_id": node_id,
            "minimum_robust_snr": minimum,
            "observed_robust_snr": observed,
            "low_snr_targets": low_snr,
            "current_num_averages": current_averages,
            "next_num_averages": next_averages,
            "retry_recommended": retry_recommended,
            "retry_parameters": retry_parameters if retry_recommended else {},
            "reason": (
                f"Repeat only {low_snr} with num_averages={next_averages}; "
                f"the {node_id} completion floor is robust_snr >= {minimum:g}."
                if retry_recommended
                else "All requested targets meet the SNR floor."
                if not low_snr
                else "The configured averaging confirmation ceiling is reached; "
                "do not mark these targets complete from SNR evidence."
            ),
        }

    def _run_review(self, run_id: str, workflow_id: str) -> dict[str, Any]:
        run = self.db.one(
            "SELECT * FROM runs WHERE id = ? AND workflow_id = ?",
            (run_id, workflow_id),
        )
        if run is None:
            return {
                "available": False,
                "run": None,
                "analysis": None,
                "decision": None,
                "plots": [],
            }
        analysis = json_loads(run.get("analysis_json"), None)
        decision_row = self.db.one(
            "SELECT * FROM decisions WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
            (run_id,),
        )
        decision = None
        if decision_row is not None:
            decision = dict(decision_row)
            decision["next_parameters"] = json_loads(
                decision.pop("next_parameters_json"), {}
            )
            decision["state_patch"] = json_loads(
                decision.pop("state_patch_json"), []
            )
        plots: list[str] = []
        if isinstance(analysis, dict):
            for value in analysis.get("plots", []):
                if isinstance(value, str) and self._is_allowed_review_asset(
                    Path(value)
                ):
                    plots.append(str(Path(value).resolve()))
        return {
            "available": bool(analysis or decision or plots),
            "run": {
                "id": run["id"],
                "node_id": run["node_id"],
                "status": run["status"],
                "parameters": json_loads(run.get("parameters_json"), {}),
                "started_at": run.get("started_at"),
                "finished_at": run.get("finished_at"),
                "elapsed_seconds": _elapsed_seconds(run),
                "snapshot_id": run.get("snapshot_id"),
                "snapshot_path": run.get("snapshot_path"),
                "analysis_status": run.get("analysis_status"),
                # The Dashboard has no policy access, so tell it whether this
                # node is one that runs once on defaults; its result summary
                # must not report such a run as unsuccessful.
                "single_pass": self._node_is_single_pass(str(run["node_id"])),
            },
            "analysis": analysis,
            "decision": decision,
            "plots": plots,
        }

    def proposal_review_asset(self, proposal_id: str, index: int) -> Path:
        review = self.proposal_review(proposal_id)
        plots = review["plots"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(plots)
        ):
            raise ServiceError("Unknown proposal review asset")
        path = Path(plots[index]).resolve()
        if not self._is_allowed_review_asset(path) or not path.is_file():
            raise ServiceError("Proposal review asset is unavailable")
        return path

    def dashboard_asset_thumbnail(self, path: Path, width: int = 360) -> Path:
        """A small cached copy of a result plot for the summary rows."""
        target = (
            self.settings.runtime / "report_thumbnails" / f"{_asset_token(path)}-{width}.png"
        )
        if target.is_file():
            return target
        from PIL import Image

        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            image.thumbnail((width, width * 4))
            temporary = target.with_suffix(".tmp.png")
            image.save(temporary, format="PNG", optimize=True)
        temporary.replace(target)
        return target

    def _is_allowed_review_asset(self, path: Path) -> bool:
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            return False
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return False
        allowed_roots = (
            self.settings.data_root.resolve(),
            (self.settings.runtime / "report_assets").resolve(),
        )
        return any(
            resolved == root or root in resolved.parents for root in allowed_roots
        )

    def _approve_pending_proposal(
        self,
        proposal_id: str,
        actor: str,
        approval_method: str,
        *,
        lease_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.db.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM proposals WHERE id = ?", (proposal_id,)
            ).fetchone()
            if row is None:
                raise ServiceError(f"Unknown proposal: {proposal_id}")
            proposal = dict(row)
            if proposal["status"] != "pending":
                raise ServiceError(
                    f"Proposal status is {proposal['status']}, not pending"
                )
            if datetime.fromisoformat(proposal["expires_at"]) <= datetime.now(
                timezone.utc
            ):
                connection.execute(
                    "UPDATE proposals SET status = 'expired' "
                    "WHERE id = ? AND status = 'pending'",
                    (proposal_id,),
                )
                raise ServiceError("Proposal has expired")
            changed = connection.execute(
                "UPDATE proposals SET status = 'approved', approved_at = ?, "
                "approved_by = ? WHERE id = ? AND status = 'pending'",
                (utc_now(), actor, proposal_id),
            ).rowcount
            if changed != 1:
                raise ServiceError("Proposal approval was resolved concurrently")
            if proposal["kind"] == "autonomy_lease":
                self._apply_lease_options(
                    connection, proposal, lease_options, approval_method, actor
                )
                self._activate_autonomy_lease(
                    proposal, actor, connection=connection
                )
            self.db.event(
                "proposal_approved",
                actor,
                {
                    "proposal_id": proposal_id,
                    "kind": proposal["kind"],
                    "approval_method": approval_method,
                },
                proposal["workflow_id"],
                connection=connection,
            )
        return self.proposal(proposal_id)

    def _load_or_create_approval_secret(self) -> bytes:
        secret_path = self.settings.runtime / "browser_approval.secret"
        try:
            with secret_path.open("xb") as handle:
                handle.write(os.urandom(32))
            try:
                os.chmod(secret_path, 0o600)
            except OSError:
                pass
        except FileExistsError:
            pass
        secret = secret_path.read_bytes()
        if len(secret) < 32:
            raise ServiceError("Browser approval secret is invalid")
        return secret

    def list_historical_runs(
        self, node_id: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        if node_id is not None:
            definition = self.policy.node_definition(node_id)
            needle = definition["name"]
        else:
            needle = ""
        candidates: list[dict[str, Any]] = []
        for node_file in self.settings.data_root.rglob("node.json"):
            if needle and needle.lower() not in node_file.parent.name.lower():
                continue
            try:
                raw = json.loads(node_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            candidates.append(
                {
                    "snapshot_id": raw.get("id"),
                    "name": raw.get("metadata", {}).get("name"),
                    "created_at": raw.get("created_at"),
                    "path": str(node_file.parent),
                    "outcomes": raw.get("data", {}).get("outcomes", {}),
                }
            )
        candidates.sort(key=lambda item: item.get("snapshot_id") or -1, reverse=True)
        return candidates[: max(1, min(int(limit), 100))]

    def _assert_state_commit_idle(self) -> None:
        active = self.db.one(
            "SELECT id FROM runs WHERE status IN ('starting', 'running', 'stopping') "
            "ORDER BY started_at DESC LIMIT 1"
        )
        if active is not None:
            raise ServiceError(
                f"State commits are blocked while run {active['id']} is active"
            )
        if self.settings.lock_path.exists():
            raise ServiceError(
                "State commits are blocked while the JY hardware lock exists"
            )

    def _create_proposal(
        self,
        workflow_id: str,
        kind: str,
        payload: dict[str, Any],
        source_client: str,
        ttl_minutes: int,
        max_uses: int,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        if kind == "state_commit":
            self._assert_state_commit_idle()
        proposal_id = uuid.uuid4().hex
        created = datetime.now(timezone.utc)
        expires = created + timedelta(minutes=ttl_minutes)
        self.db.execute(
            """
            INSERT INTO proposals(
                id, workflow_id, kind, payload_json, status, created_at,
                expires_at, max_uses, source_client, autonomy_lease_id
            ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
            """,
            (
                proposal_id,
                workflow_id,
                kind,
                json_dumps(payload),
                created.isoformat(timespec="seconds"),
                expires.isoformat(timespec="seconds"),
                max_uses,
                source_client,
                autonomy_lease_id,
            ),
        )
        self.db.event(
            "proposal_created",
            source_client,
            {
                "proposal_id": proposal_id,
                "kind": kind,
                "max_uses": max_uses,
                "autonomy_lease_id": autonomy_lease_id,
            },
            workflow_id,
        )
        return self.proposal(proposal_id)

    @property
    def autonomy_mode_entry_phrase(self) -> str:
        return self.settings.autonomy_mode_entry_phrase

    def autonomy_control_url(self, lease_id: str) -> str:
        session = self.db.one(
            "SELECT id FROM measurement_sessions WHERE autonomy_lease_id = ? "
            "ORDER BY started_at DESC LIMIT 1",
            (lease_id,),
        )
        if session is not None:
            return self.session_dashboard_url(str(session["id"]))
        # Compatibility fallback for databases created before every lease was
        # guaranteed to have a Dashboard session.  The legacy route redirects
        # to the canonical session whenever a binding exists.
        return self._external_browser_url(f"/autonomy/{lease_id}")

    def autonomy_csrf_token(self, lease_id: str) -> str:
        lease = self._autonomy_row(lease_id)
        message = "\0".join(
            (
                lease["id"],
                lease["workflow_id"],
                lease["proposal_id"],
                lease["created_at"],
            )
        ).encode("utf-8")
        return hmac.new(self._approval_secret, message, hashlib.sha256).hexdigest()

    def dashboard_shutdown_csrf_token(self, session_id: str) -> str:
        session = self.db.one(
            "SELECT * FROM measurement_sessions WHERE id = ?", (session_id,)
        )
        if session is None:
            raise ServiceError("Unknown measurement dashboard session")
        message = "\0".join(
            (
                "full-shutdown",
                session["id"],
                session["workflow_id"],
                session["started_at"],
            )
        ).encode("utf-8")
        return hmac.new(self._approval_secret, message, hashlib.sha256).hexdigest()

    def dashboard_shutdown_confirmation(self, session_id: str) -> str:
        self.dashboard_status(session_id)
        return f"SHUTDOWN {session_id}"

    def operator_csrf_token(self) -> str:
        nonce = os.getenv("JY_SERVICE_INSTANCE_NONCE", "local-development")
        return hmac.new(
            self._approval_secret,
            f"operator-console\0{nonce}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def request_full_shutdown_from_browser(
        self,
        session_id: str,
        confirmation: str,
        csrf_token: str,
        actor: str,
    ) -> dict[str, Any]:
        if not hmac.compare_digest(
            confirmation, self.dashboard_shutdown_confirmation(session_id)
        ):
            raise ServiceError("The full-shutdown confirmation did not match exactly.")
        if not hmac.compare_digest(
            csrf_token, self.dashboard_shutdown_csrf_token(session_id)
        ):
            raise ServiceError("The full-shutdown control token was invalid.")
        return self.request_full_shutdown(session_id, actor)

    def request_full_shutdown(
        self,
        session_id: str,
        actor: str,
    ) -> dict[str, Any]:
        if not actor.strip():
            raise ServiceError("Shutdown actor is required")
        session = self.db.one(
            "SELECT * FROM measurement_sessions WHERE id = ?", (session_id,)
        )
        if session is None:
            raise ServiceError("Unknown measurement dashboard session")
        workflow = self._workflow(str(session["workflow_id"]))
        other_open = self.db.one(
            "SELECT id FROM workflows WHERE id != ? AND status IN ('active','paused') "
            "LIMIT 1",
            (workflow["id"],),
        )
        if other_open is not None:
            raise ServiceError(
                "This is an older Dashboard session; refuse to shut down services "
                "while another workflow is open."
            )
        active_run = self.db.one(
            "SELECT * FROM runs WHERE workflow_id = ? "
            "AND status IN ('starting','running','stopping') "
            "ORDER BY started_at DESC LIMIT 1",
            (workflow["id"],),
        )
        status = "waiting_for_run" if active_run is not None else "requested"
        now = datetime.now(timezone.utc)
        grace = int(
            self.policy.raw.get("autonomy", {}).get(
                "emergency_stop_grace_seconds", 30
            )
        )
        request = {
            "session_id": session_id,
            "workflow_id": workflow["id"],
            "requested_by": actor.strip(),
            "requested_at": now.isoformat(timespec="seconds"),
            "dispatch_after": (now + timedelta(seconds=3)).isoformat(
                timespec="seconds"
            ),
            "force_after": (now + timedelta(seconds=max(0, grace))).isoformat(
                timespec="seconds"
            ),
            "instance_nonce": os.getenv(
                "JY_SERVICE_INSTANCE_NONCE", "local-development"
            ),
            "status": status,
            "active_run_id": active_run.get("id") if active_run else None,
            "operator_console_url": self.operator_console_url,
        }
        self.db.execute(
            "UPDATE measurement_sessions SET status = 'stopping', updated_at = ? "
            "WHERE id = ?",
            (utc_now(), session_id),
        )
        self.db.event(
            "full_shutdown_requested",
            actor.strip(),
            {
                "session_id": session_id,
                "active_run_id": request["active_run_id"],
                "status": status,
            },
            workflow["id"],
        )
        # Persist the operator's intent before touching the worker.  A delivery
        # failure must never erase a valid request to close the workflow/services.
        self._save_shutdown_request(request)
        if active_run is not None:
            try:
                request["stop_result"] = self.runner.stop(
                    str(active_run["id"]),
                    actor.strip(),
                    intent="full_shutdown",
                    reason="Full measurement shutdown requested from Dashboard.",
                )
            except Exception as exc:
                request["stop_error"] = f"{type(exc).__name__}: {exc}"
            self._save_shutdown_request(request)
        return self._reconcile_full_shutdown_request(session_id) or request

    @property
    def _full_shutdown_request_path(self) -> Path:
        """Legacy latest-request mirror retained for diagnostics and upgrades."""
        return self.settings.runtime / "full-shutdown-request.json"

    def _shutdown_request_path(self, session_id: str) -> Path:
        return self.settings.runtime / "shutdown_requests" / f"{session_id}.json"

    def _save_shutdown_request(self, request: dict[str, Any]) -> None:
        session_id = str(request.get("session_id") or "")
        workflow_id = str(request.get("workflow_id") or "")
        if not session_id or not workflow_id:
            raise ServiceError("Shutdown request is missing its session/workflow id")
        now = utc_now()
        payload = dict(request)
        atomic_write_json(self._shutdown_request_path(session_id), payload)
        # Keep one human-readable latest mirror for older diagnostics.  Reconcile
        # never relies on it, so one session cannot consume another's request.
        atomic_write_json(self._full_shutdown_request_path, payload)
        self.db.execute(
            """
            INSERT INTO shutdown_requests(
                session_id, workflow_id, status, requested_by, requested_at,
                updated_at, active_run_id, stop_error, quarantine, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                status = excluded.status,
                updated_at = excluded.updated_at,
                active_run_id = excluded.active_run_id,
                stop_error = excluded.stop_error,
                quarantine = excluded.quarantine,
                payload_json = excluded.payload_json
            """,
            (
                session_id,
                workflow_id,
                str(payload.get("status") or "requested"),
                str(payload.get("requested_by") or "unknown"),
                str(payload.get("requested_at") or now),
                now,
                payload.get("active_run_id"),
                payload.get("stop_error"),
                int(bool(payload.get("quarantine"))),
                json_dumps(payload),
            ),
        )

    def _load_shutdown_request(
        self, session_id: str | None = None
    ) -> dict[str, Any] | None:
        if session_id:
            row = self.db.one(
                "SELECT payload_json FROM shutdown_requests WHERE session_id = ?",
                (session_id,),
            )
        else:
            row = self.db.one(
                "SELECT payload_json FROM shutdown_requests "
                "ORDER BY updated_at DESC LIMIT 1"
            )
        if row is None:
            return None
        request = json_loads(row.get("payload_json"), None)
        return request if isinstance(request, dict) else None

    def _shutdown_request_for_session(self, session_id: str) -> dict[str, Any] | None:
        request = self._load_shutdown_request(session_id)
        if request is None:
            return None
        request = dict(request)
        request.pop("instance_nonce", None)
        return request

    def _reconcile_full_shutdown_request(
        self, session_id: str | None = None
    ) -> dict[str, Any] | None:
        request = self._load_shutdown_request(session_id)
        if request is None:
            return None
        expected_nonce = os.getenv("JY_SERVICE_INSTANCE_NONCE", "local-development")
        if request.get("instance_nonce") != expected_nonce:
            if (
                bool(request.get("quarantine"))
                and request.get("status") == "dispatching"
                and not self.settings.lock_path.exists()
            ):
                # A prior service instance closed successfully with a retained
                # lock.  A later recovery-only instance may finish that exact
                # session after formal operator recovery.
                workflow_id = str(request.get("workflow_id") or "")
                recovered_session_id = str(request.get("session_id") or "")
                self._finalize_workflow_shutdown(
                    workflow_id,
                    recovered_session_id,
                    "hardware-lock-recovery",
                    recovery_required=False,
                )
                request["instance_nonce"] = expected_nonce
                request["status"] = "ready"
                request["recovered_at"] = utc_now()
                request["dispatch_after"] = (
                    datetime.now(timezone.utc) + timedelta(seconds=5)
                ).isoformat(timespec="seconds")
                self._save_shutdown_request(request)
                return request
            return None
        if request.get("status") == "dispatching":
            failure_path = (
                self.settings.runtime
                / "shutdown-stop-failures"
                / f"{request.get('session_id')}.json"
            )
            try:
                failure = json.loads(failure_path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError, TypeError):
                return request
            request["status"] = "failed"
            request["error"] = str(
                failure.get("error") or "The stop helper failed without details."
            )
            request["stop_helper_failure"] = failure
            self._save_shutdown_request(request)
            return request
        if request.get("status") == "failed":
            return request

        workflow_id = str(request.get("workflow_id") or "")
        session_id = str(request.get("session_id") or "")
        active_run = self.db.one(
            "SELECT * FROM runs WHERE workflow_id = ? "
            "AND status IN ('starting','running','stopping') "
            "ORDER BY started_at DESC LIMIT 1",
            (workflow_id,),
        )
        if active_run is None and request.get("active_run_id"):
            prior = self.db.one(
                "SELECT * FROM runs WHERE id = ?",
                (str(request["active_run_id"]),),
            )
            prior_pid = prior.get("pid") if prior else None
            if (
                prior is not None
                and isinstance(prior_pid, int)
                and self.runner.process_is_alive(prior_pid)
            ):
                active_run = prior
        if active_run is not None:
            request["status"] = "waiting_for_run"
            request["active_run_id"] = active_run["id"]
            pid = active_run.get("pid")
            if not isinstance(pid, int):
                # A just-created run may be between its DB insert and Popen PID
                # update.  Keep the shutdown request durable and let the next
                # sweep observe the exact process identity.
                try:
                    missing_pid_deadline = parse_iso_datetime(
                        str(request.get("force_after") or "")
                    )
                    if missing_pid_deadline.tzinfo is None:
                        missing_pid_deadline = missing_pid_deadline.replace(
                            tzinfo=timezone.utc
                        )
                except (TypeError, ValueError):
                    missing_pid_deadline = datetime.now(timezone.utc) + timedelta(
                        seconds=30
                    )
                if missing_pid_deadline > datetime.now(timezone.utc):
                    self._save_shutdown_request(request)
                    return request
                self.db.execute(
                    "UPDATE runs SET status = 'failed', finished_at = ?, error = ?, "
                    "termination_cause = 'missing_process_identity_during_shutdown' "
                    "WHERE id = ?",
                    (
                        utc_now(),
                        "Shutdown grace elapsed before the worker PID was recorded.",
                        active_run["id"],
                    ),
                )
            if isinstance(pid, int) and self.runner.process_is_alive(pid):
                try:
                    force_after = parse_iso_datetime(str(request.get("force_after") or ""))
                    if force_after.tzinfo is None:
                        force_after = force_after.replace(tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    force_after = datetime.now(timezone.utc) + timedelta(seconds=30)
                if force_after <= datetime.now(timezone.utc):
                    try:
                        request["force_stop_result"] = self.runner.force_stop(
                            str(active_run["id"]), "full-shutdown-coordinator"
                        )
                        request["status"] = "waiting_for_force_exit"
                    except Exception as exc:
                        request["stop_error"] = f"{type(exc).__name__}: {exc}"
                self._save_shutdown_request(request)
                return request

            evidence = self.runner.exit_evidence(str(active_run["id"]))
            stop_intent = str(
                active_run.get("stop_intent")
                or evidence.get("stop_request", {}).get("intent")
                or ""
            )
            refreshed = self.db.one(
                "SELECT * FROM runs WHERE id = ?", (active_run["id"],)
            )
            if refreshed and refreshed["status"] in {"starting", "running", "stopping"}:
                terminal_status = (
                    "cancelled_by_shutdown"
                    if stop_intent == "full_shutdown"
                    else "stopped"
                )
                cause = str(
                    evidence.get("exit_receipt", {}).get("termination_cause")
                    or "worker_exit_during_shutdown"
                )
                message = (
                    "Worker exited while full shutdown was in progress; "
                    "cleanup is determined from the retained lock and exit receipt."
                )
                self.db.execute(
                    "UPDATE runs SET status = ?, finished_at = ?, error = ?, "
                    "termination_cause = ? WHERE id = ?",
                    (terminal_status, utc_now(), message, cause, active_run["id"]),
                )
                self.db.event(
                    "run_cancelled_by_shutdown",
                    "full-shutdown-coordinator",
                    {
                        "run_id": active_run["id"],
                        "exit_evidence": evidence,
                        "hardware_lock_retained": self.settings.lock_path.exists(),
                    },
                    workflow_id,
                )
        if request.get("active_run_id"):
            request["last_run_id"] = request["active_run_id"]
        request["active_run_id"] = None
        previous_status = str(request.get("status") or "")
        if self.settings.lock_path.exists():
            request["status"] = "quarantine_ready"
            request["quarantine"] = True
            request["operator_console_url"] = self.operator_console_url
            self._finalize_workflow_shutdown(
                workflow_id,
                session_id,
                str(request.get("requested_by") or "full-shutdown-coordinator"),
                recovery_required=True,
            )
            atomic_write_json(
                self.settings.runtime / "recovery-required.json",
                {
                    "session_id": session_id,
                    "workflow_id": workflow_id,
                    "run_id": request.get("last_run_id"),
                    "reason": "Hardware lock retained after workflow shutdown.",
                    "created_at": utc_now(),
                },
            )
        else:
            self._finalize_workflow_shutdown(
                workflow_id,
                session_id,
                str(request.get("requested_by") or "full-shutdown-coordinator"),
                recovery_required=False,
            )
            request["status"] = "ready"
            request["quarantine"] = False
        if previous_status not in {"ready", "quarantine_ready"}:
            # Give the browser/operator response time to reach the user before
            # the helper stops the very HTTP services that delivered it.  The
            # MCP watchdog will reconcile the ready request again.
            ready_at = datetime.now(timezone.utc)
            request["ready_at"] = ready_at.isoformat(timespec="seconds")
            request["dispatch_after"] = (
                ready_at + timedelta(seconds=5)
            ).isoformat(timespec="seconds")
        self._save_shutdown_request(request)
        try:
            dispatch_after = parse_iso_datetime(request.get("dispatch_after", ""))
            if dispatch_after.tzinfo is None:
                dispatch_after = dispatch_after.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            dispatch_after = datetime.now(timezone.utc)
        if dispatch_after > datetime.now(timezone.utc):
            return request

        dispatch_lock = self.settings.runtime / "full-shutdown-dispatch.lock"
        try:
            with exclusive_file_lock(dispatch_lock, "full service shutdown dispatch"):
                current = self._load_shutdown_request(session_id) or request
                if current.get("status") not in {"ready", "quarantine_ready"}:
                    return current
                dispatch_mode = str(current["status"])
                current["status"] = "dispatching"
                current["dispatch_mode"] = dispatch_mode
                current["dispatched_at"] = utc_now()
                self._save_shutdown_request(current)
                self.db.event(
                    "full_shutdown_dispatched",
                    "full-shutdown-coordinator",
                    {"session_id": session_id},
                    workflow_id,
                )
                command = [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(self.settings.agent_root / "stop_server.ps1"),
                    "-SessionId",
                    session_id,
                ]
                (
                    self.settings.runtime
                    / "shutdown-stop-failures"
                    / f"{session_id}.json"
                ).unlink(missing_ok=True)
                creation_flags = 0
                if os.name == "nt":
                    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                subprocess.Popen(
                    command,
                    cwd=self.settings.agent_root,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creation_flags,
                )
                return current
        except Exception as exc:
            request["status"] = "failed"
            request["error"] = f"{type(exc).__name__}: {exc}"
            self._save_shutdown_request(request)
            self.db.event(
                "full_shutdown_dispatch_failed",
                "full-shutdown-coordinator",
                {"session_id": session_id, "error": request["error"]},
                workflow_id,
            )
            return request

    def reconcile_full_shutdown_request(
        self, session_id: str | None = None
    ) -> dict[str, Any] | None:
        """Re-check a queued Dashboard shutdown after local operator recovery."""
        return self._reconcile_full_shutdown_request(session_id)

    def _finalize_workflow_shutdown(
        self,
        workflow_id: str,
        session_id: str,
        actor: str,
        *,
        recovery_required: bool = False,
    ) -> None:
        now = utc_now()
        final_status = "recovery_required" if recovery_required else "stopped"
        event_type = "workflow_quarantined" if recovery_required else "workflow_stopped"
        with self.db.transaction(immediate=True) as connection:
            workflow = connection.execute(
                "SELECT status FROM workflows WHERE id = ?", (workflow_id,)
            ).fetchone()
            if workflow is None:
                raise ServiceError("Shutdown workflow no longer exists")
            changed = connection.execute(
                "UPDATE workflows SET status = ?, updated_at = ? "
                "WHERE id = ? AND status IN ('active','paused','recovery_required')",
                (final_status, now, workflow_id),
            ).rowcount
            connection.execute(
                "UPDATE measurement_sessions SET status = ?, updated_at = ? "
                "WHERE id = ?",
                (final_status, now, session_id),
            )
            connection.execute(
                "UPDATE autonomy_leases SET status = 'revoked', stopped_reason = ?, "
                "updated_at = ? WHERE workflow_id = ? "
                "AND status IN ('pending','active','paused')",
                ("Full measurement shutdown requested from Dashboard.", now, workflow_id),
            )
            connection.execute(
                "UPDATE proposals SET status = 'cancelled' WHERE workflow_id = ? "
                "AND status = 'pending'",
                (workflow_id,),
            )
            if changed:
                self.db.event(
                    event_type,
                    actor,
                    {
                        "workflow_id": workflow_id,
                        "reason": "Full measurement shutdown requested from Dashboard.",
                        "recovery_required": recovery_required,
                    },
                    workflow_id,
                    connection=connection,
                )

    def control_autonomy_from_browser(
        self,
        lease_id: str,
        action: str,
        csrf_token: str,
        actor: str,
    ) -> dict[str, Any]:
        if not hmac.compare_digest(csrf_token, self.autonomy_csrf_token(lease_id)):
            raise ServiceError("The autonomy control token was invalid.")
        reason = f"Manual {action.replace('_', ' ')} requested from control page."
        if action == "pause":
            return self.pause_autonomy(lease_id, actor, reason)
        if action == "resume":
            return self.resume_autonomy(
                lease_id,
                actor,
                self.settings.autonomy_mode_entry_phrase,
            )
        if action == "stop":
            return self.stop_autonomy(lease_id, actor, reason)
        if action == "emergency_stop":
            return self.stop_autonomy(lease_id, actor, reason, emergency=True)
        raise ServiceError("Unknown autonomy control action")

    def start_autonomy_watchdog(self) -> None:
        """Start an idempotent daemon that enforces expiry and hard-stop rules."""
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()

        def watch() -> None:
            interval = float(
                self.policy.raw["autonomy"].get("watchdog_interval_seconds", 2)
            )
            while not self._watchdog_stop.wait(max(0.25, interval)):
                try:
                    self.autonomy_watchdog_sweep()
                except Exception as exc:
                    self.db.event(
                        "autonomy_watchdog_error",
                        "autonomy_watchdog",
                        {"error": str(exc)},
                    )

        self._watchdog_thread = threading.Thread(
            target=watch,
            name="jy-autonomy-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def stop_autonomy_watchdog(self) -> None:
        self._watchdog_stop.set()

    def autonomy_watchdog_sweep(self) -> dict[str, int]:
        expired = 0
        halted = 0
        emergency_forced = self._enforce_pending_emergency_stops()
        for lease in self.db.all(
            "SELECT * FROM autonomy_leases WHERE status IN ('active', 'paused')"
        ):
            refreshed = self._expire_autonomy_if_needed(lease)
            if refreshed["status"] == "expired":
                expired += 1
                continue
            runs = self.db.all(
                "SELECT * FROM runs WHERE autonomy_lease_id = ? "
                "ORDER BY started_at DESC LIMIT 20",
                (lease["id"],),
            )
            for run in runs:
                before = self._autonomy_row(lease["id"])["status"]
                self._enforce_run_hard_stops(run)
                after = self._autonomy_row(lease["id"])["status"]
                if before != "halted" and after == "halted":
                    halted += 1
                    break
        shutdown = self._reconcile_full_shutdown_request()
        return {
            "expired": expired,
            "halted": halted,
            "emergency_forced": emergency_forced,
            "shutdown_dispatched": int(
                bool(shutdown and shutdown.get("status") == "dispatching")
            ),
        }

    def _activate_autonomy_lease(
        self,
        proposal: dict[str, Any],
        actor: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        lease_id = proposal.get("autonomy_lease_id")
        if not lease_id:
            raise ServiceError("Autonomy proposal is missing its lease id")
        if connection is None:
            lease = self._autonomy_row(str(lease_id))
        else:
            row = connection.execute(
                "SELECT * FROM autonomy_leases WHERE id = ?", (str(lease_id),)
            ).fetchone()
            if row is None:
                raise ServiceError(f"Unknown autonomy lease: {lease_id}")
            lease = dict(row)
        if lease["status"] != "pending":
            raise ServiceError(f"Autonomy lease status is {lease['status']}, not pending")
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=float(lease["duration_hours"]))
        query = (
            "UPDATE autonomy_leases SET status = 'active', activated_at = ?, "
            "expires_at = ?, updated_at = ?, approved_by = ?, stopped_reason = NULL "
            "WHERE id = ? AND status = 'pending'"
        )
        params = (
            now.isoformat(timespec="seconds"),
            expires.isoformat(timespec="seconds"),
            now.isoformat(timespec="seconds"),
            actor,
            lease_id,
        )
        if connection is None:
            changed = self.db.execute(query, params)
        else:
            changed = connection.execute(query, params).rowcount
        if changed != 1:
            raise ServiceError("Autonomy lease activation was resolved concurrently")
        self.db.event(
            "autonomy_lease_activated",
            actor,
            {
                "lease_id": lease_id,
                "expires_at": expires.isoformat(timespec="seconds"),
                "duration_hours": float(lease["duration_hours"]),
            },
            proposal["workflow_id"],
            connection=connection,
        )

    def _delegate_proposal_to_autonomy(
        self, proposal_id: str, lease_id: str, actor: str
    ) -> dict[str, Any]:
        lease = self._active_autonomy(lease_id)
        proposal = self.db.one("SELECT * FROM proposals WHERE id = ?", (proposal_id,))
        if proposal is None or proposal.get("autonomy_lease_id") != lease_id:
            raise ServiceError("Proposal is not bound to this autonomy lease")
        if proposal["workflow_id"] != lease["workflow_id"]:
            raise ServiceError("Proposal workflow does not match autonomy lease")
        if proposal["status"] != "pending":
            raise ServiceError("Only a pending proposal can receive delegated approval")
        approved_by = f"delegated autonomy lease {lease_id}"
        changed = self.db.execute(
            "UPDATE proposals SET status = 'approved', approved_at = ?, approved_by = ? "
            "WHERE id = ? AND status = 'pending'",
            (utc_now(), approved_by, proposal_id),
        )
        if changed != 1:
            raise ServiceError("Delegated approval was resolved concurrently")
        self.db.event(
            "proposal_delegated",
            actor,
            {
                "proposal_id": proposal_id,
                "kind": proposal["kind"],
                "autonomy_lease_id": lease_id,
            },
            proposal["workflow_id"],
        )
        return self.proposal(proposal_id)

    def _autonomy_row(self, lease_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM autonomy_leases WHERE id = ?", (lease_id,))
        if row is None:
            raise ServiceError(f"Unknown autonomy lease: {lease_id}")
        return row

    def _expire_autonomy_if_needed(self, row: dict[str, Any]) -> dict[str, Any]:
        if row["status"] == "pending":
            proposal = self.db.one(
                "SELECT status, expires_at FROM proposals WHERE id = ?",
                (row["proposal_id"],),
            )
            proposal_expired = False
            if proposal is not None:
                proposal_expired = proposal["status"] == "expired"
                if (
                    proposal["status"] == "pending"
                    and datetime.fromisoformat(str(proposal["expires_at"]))
                    <= datetime.now(timezone.utc)
                ):
                    self.db.execute(
                        "UPDATE proposals SET status = 'expired' WHERE id = ?",
                        (row["proposal_id"],),
                    )
                    proposal_expired = True
            if proposal_expired:
                now = utc_now()
                reason = "Approval proposal expired before lease activation."
                self.db.execute(
                    "UPDATE autonomy_leases SET status = 'expired', updated_at = ?, "
                    "stopped_reason = ? WHERE id = ?",
                    (now, reason, row["id"]),
                )
                self.db.execute(
                    "UPDATE measurement_sessions SET status = 'expired', "
                    "updated_at = ? WHERE autonomy_lease_id = ?",
                    (now, row["id"]),
                )
                self.db.event(
                    "autonomy_lease_expired",
                    "approval_expiry",
                    {
                        "lease_id": row["id"],
                        "proposal_id": row["proposal_id"],
                        "reason": reason,
                    },
                    row["workflow_id"],
                )
                row = self._autonomy_row(row["id"])

        expires_at = row.get("expires_at")
        if row["status"] in {"active", "paused"} and expires_at:
            if datetime.fromisoformat(str(expires_at)) <= datetime.now(timezone.utc):
                now = utc_now()
                self.db.execute(
                    "UPDATE autonomy_leases SET status = 'expired', updated_at = ?, "
                    "stopped_reason = ? WHERE id = ?",
                    (now, "Eight-hour authorization window expired.", row["id"]),
                )
                self.db.execute(
                    "UPDATE measurement_sessions SET status = 'expired', "
                    "updated_at = ? WHERE autonomy_lease_id = ?",
                    (now, row["id"]),
                )
                self.db.event(
                    "autonomy_lease_expired",
                    "autonomy_watchdog",
                    {"lease_id": row["id"]},
                    row["workflow_id"],
                )
                row = self._autonomy_row(row["id"])
        return row

    def _active_autonomy(
        self, lease_id: str, workflow_id: str | None = None
    ) -> dict[str, Any]:
        row = self._expire_autonomy_if_needed(self._autonomy_row(lease_id))
        if row["status"] != "active":
            raise ServiceError(
                f"Autonomy lease {lease_id} is {row['status']}; new actions are blocked"
            )
        if workflow_id is not None and row["workflow_id"] != workflow_id:
            raise ServiceError("Autonomy lease belongs to a different workflow")
        workflow = self._workflow(row["workflow_id"])
        if workflow["status"] != "active":
            raise ServiceError("Workflow is not active")
        return row

    def _validate_autonomy_scope(
        self, lease: dict[str, Any], *, node_id: str, targets: list[str]
    ) -> None:
        nodes = set(json_loads(lease["allowed_nodes_json"], []))
        allowed_targets = set(json_loads(lease["targets_json"], []))
        if node_id not in nodes:
            raise AutonomyScopeError(
                f"Node {node_id} is outside the autonomy lease; new human approval is required"
            )
        if not targets or not set(targets).issubset(allowed_targets):
            raise AutonomyScopeError(
                "Run targets are outside the autonomy lease; new human approval is required"
            )

    def _assert_autonomy_attempt_quota(
        self, lease: dict[str, Any], node_id: str, targets: list[str]
    ) -> None:
        max_attempts = int(lease["max_attempts_per_node_qubit"])
        counts = {target: 0 for target in targets}
        for run in self.db.all(
            "SELECT parameters_json FROM runs WHERE autonomy_lease_id = ? "
            "AND node_id = ?",
            (lease["id"], node_id),
        ):
            prior_targets = set(
                json_loads(run.get("parameters_json"), {}).get("qubits", [])
            )
            for target in counts:
                if target in prior_targets:
                    counts[target] += 1
        exhausted = {
            target: count for target, count in counts.items() if count >= max_attempts
        }
        if exhausted:
            raise AutonomyQuotaError(
                "Autonomy attempt limit reached for node/qubit pairs "
                f"{exhausted}; those targets are incomplete for this node. "
                "Schedule another qubit or advance the resolved targets; the "
                "autonomy lease remains active."
            )
        max_total = lease.get("max_total_runs")
        if max_total is not None:
            total = self.db.one(
                "SELECT COUNT(*) AS count FROM runs WHERE autonomy_lease_id = ?",
                (lease["id"],),
            )
            if int(total["count"]) >= int(max_total):
                raise AutonomyScopeError("Autonomy total-run limit reached")

    def _readout_full_scale_owned_by(self, path: str, targets: set[str]) -> bool:
        """Whether a readout port's full scale belongs to these targets alone.

        Operator instruction 2026-10-02: 02c sets each readout line's full
        scale. That port is shared, so it may change only when every qubit
        wired to it is a workflow target.
        """
        if not re.fullmatch(
            r"/ports/mw_outputs/[^/]+/[0-9]+/[0-9]+/full_scale_power_dbm", path
        ):
            return False
        wiring = (
            load_state(self.settings.wiring_path)
            if self.settings.wiring_path.is_file()
            else {}
        )
        members = readout_port_members(
            load_state(self.settings.active_state), wiring, path
        )
        return bool(members) and set(members) <= targets

    def _power_preserving_line_rescale_paths(
        self, patch: list[dict[str, Any]], targets: set[str]
    ) -> set[str]:
        """Shared-line paths a human-approved 02c commit may touch.

        Operator choice 2026-10-14 (CRSQ_26009_die(5,-1) q3): with the 3 dB
        backoff a target's readout may not fit its line's full scale. The
        line may then change full scale when the same patch rescales every
        non-target member's readout amplitude so its delivered power is
        unchanged (amp x 10^((old - new) / 20)). Returns the port and
        non-target amplitude paths that qualify, or an empty set.
        """
        state = load_state(self.settings.active_state)
        wiring = (
            load_state(self.settings.wiring_path)
            if self.settings.wiring_path.is_file()
            else {}
        )
        values = {str(item.get("path", "")): item.get("value") for item in patch}
        allowed: set[str] = set()
        for path, new_scale in values.items():
            if not re.fullmatch(
                r"/ports/mw_outputs/[^/]+/[0-9]+/[0-9]+/full_scale_power_dbm", path
            ):
                continue
            members = readout_port_members(state, wiring, path)
            if not members or not set(members) & targets:
                continue
            try:
                old_scale = float(pointer_get(state, path))
                factor = 10 ** ((old_scale - float(new_scale)) / 20)
            except (KeyError, IndexError, TypeError, ValueError, StateError):
                continue
            line_paths = {path}
            for name in set(members) - targets:
                amp_path = f"/qubits/{name}/resonator/operations/readout/amplitude"
                try:
                    old_amp = float(pointer_get(state, amp_path))
                    new_amp = float(values[amp_path])
                except (KeyError, IndexError, TypeError, ValueError, StateError):
                    line_paths = set()
                    break
                if not math.isclose(new_amp, old_amp * factor, rel_tol=1e-6):
                    line_paths = set()
                    break
                line_paths.add(amp_path)
            allowed |= line_paths
        return allowed

    def _validate_autonomy_patch_targets(
        self, lease: dict[str, Any], patch: list[dict[str, Any]]
    ) -> None:
        allowed_targets = set(json_loads(lease["targets_json"], []))
        companion_paths = self._shared_lo_companion_paths(patch, allowed_targets)
        patch_targets = patch_qubit_targets(
            [item for item in patch if str(item.get("path", "")) not in companion_paths]
        )
        if not patch_targets or not patch_targets.issubset(allowed_targets):
            raise AutonomyScopeError(
                "State patch targets are outside the autonomy lease; new approval is required"
            )

    def _shared_lo_companion_paths(
        self, patch: list[dict[str, Any]], targets: set[str]
    ) -> set[str]:
        """Non-target XY IF paths a lease may write while moving a shared LO.

        Operator choice 2026-10-14 (CRSQ_26009_die(5,-1), automatic mode): an
        LO setup for a lease target may shift the XY IF of a qubit outside the
        lease that shares that output, if it keeps the qubit's RF
        (old LO + old IF == new LO + new IF) or parks it at the IF limit
        (operator rule 2026-10-12). Anything else still needs approval.
        """
        values = {str(item.get("path", "")): item.get("value") for item in patch}
        lo_paths = [
            path
            for path in values
            if re.fullmatch(r"/ports/mw_outputs/[^/]+/[0-9]+/[0-9]+/upconverter_frequency", path)
        ]
        if not lo_paths:
            return set()
        state = load_state(self.settings.active_state)
        wiring = (
            load_state(self.settings.wiring_path)
            if self.settings.wiring_path.is_file()
            else {}
        )
        wiring_qubits = wiring.get("wiring", {}).get("qubits", {})
        if not isinstance(wiring_qubits, dict):
            return set()
        limit = float(self.policy.limits["qubit_if_abs_hz"])
        allowed: set[str] = set()
        for lo_path in lo_paths:
            reference = "#" + lo_path.rsplit("/", 1)[0]
            owners = {
                str(name)
                for name, item in wiring_qubits.items()
                if isinstance(item, dict)
                and item.get("xy", {}).get("opx_output") == reference
            }
            if not owners & targets:
                continue
            try:
                old_lo = float(pointer_get(state, lo_path))
                new_lo = float(values[lo_path])
            except (KeyError, IndexError, TypeError, ValueError, StateError):
                continue
            for name in owners - targets:
                if_path = f"/qubits/{name}/xy/intermediate_frequency"
                if if_path not in values:
                    continue
                try:
                    old_if = float(pointer_get(state, if_path))
                    new_if = float(values[if_path])
                except (KeyError, IndexError, TypeError, ValueError, StateError):
                    continue
                preserved = math.isclose(old_lo + old_if, new_lo + new_if, abs_tol=1.0)
                parked = math.isclose(abs(new_if), limit, abs_tol=1.0)
                if preserved or parked:
                    allowed.add(if_path)
        return allowed

    def _assert_partial_pass_state_commit(
        self,
        run: dict[str, Any],
        patch: list[dict[str, Any]],
        allowed_statuses: set[str],
    ) -> None:
        """Allow a `needs_review` run to commit only its passing targets.

        Operator instruction 2026-09-20. Suppressing the whole patch lost
        calibrated values permanently: the node refuses a further run once every
        target is resolved, so a qubit that passed inside a `needs_review` run
        had no remaining way to obtain a pass-backed commit.
        """

        refusal = AutonomyEvidenceError(
            "Autonomous state commit requires a completed run with an allowed "
            f"analysis status: {sorted(allowed_statuses)}"
        )
        if not bool(
            self.policy.raw["autonomy"].get("partial_pass_state_commit", False)
        ):
            raise refusal
        if run["analysis_status"] != "needs_review":
            raise refusal
        analysis = json_loads(run.get("analysis_json"), {})
        if not isinstance(analysis, dict) or "passing_targets" not in analysis:
            raise AutonomyEvidenceError(
                "This run was analyzed before per-target evidence was recorded; "
                "re-analyze it before committing a partial-pass state patch"
            )
        passing = {str(name) for name in analysis.get("passing_targets", [])}
        patch_targets = patch_qubit_targets(patch)
        if not patch_targets:
            raise AutonomyEvidenceError(
                "A needs_review run may commit only per-qubit calibration values"
            )
        # Operator instruction 2026-10-10: a 06 qubit whose fringes missed the
        # detuning may commit exactly its analyzer-built frequency correction
        # (IF and idle_freq) while it stays unresolved.
        corrections = [
            item
            for item in analysis.get("frequency_correction_patch") or []
            if isinstance(item, dict)
        ]
        correction_targets = patch_qubit_targets(corrections)
        for target in sorted((patch_targets - passing) & correction_targets):
            requested = [
                item
                for item in patch
                if str(item.get("path", "")).startswith(f"/qubits/{target}/")
            ]
            allowed = [
                item
                for item in corrections
                if str(item.get("path", "")).startswith(f"/qubits/{target}/")
            ]
            if requested != allowed:
                raise AutonomyEvidenceError(
                    f"{target} did not pass this run; it may commit only its "
                    f"exact frequency correction {allowed}"
                )
            passing = passing | {target}
        outside = sorted(patch_targets - passing)
        if outside:
            raise AutonomyEvidenceError(
                f"{outside} did not pass every check in this needs_review run; "
                f"only {sorted(passing)} may be committed from it"
            )

    def _validate_autonomous_scientific_state_commit(
        self,
        lease: dict[str, Any],
        run_id: str | None,
        patch: list[dict[str, Any]],
    ) -> None:
        if not run_id:
            raise AutonomyEvidenceError(
                "Autonomous scientific state commits require an accepted run"
            )
        run = self.db.one(
            "SELECT * FROM runs WHERE id = ? AND workflow_id = ?",
            (run_id, lease["workflow_id"]),
        )
        if run is None or run.get("autonomy_lease_id") != lease["id"]:
            raise AutonomyScopeError(
                "Accepted run is not bound to this autonomy lease"
            )
        self._validate_autonomy_scope(
            lease,
            node_id=run["node_id"],
            targets=json_loads(run["parameters_json"], {}).get("qubits", []),
        )
        allowed_statuses = set(
            json_loads(lease["auto_state_commit_statuses_json"], [])
        )
        if run["status"] != "completed":
            raise AutonomyEvidenceError(
                "Autonomous state commit requires a completed run"
            )
        if run["analysis_status"] not in allowed_statuses:
            self._assert_partial_pass_state_commit(run, patch, allowed_statuses)
        decision = self.db.one(
            "SELECT * FROM decisions WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
            (run_id,),
        )
        if decision is None or decision["decision"] not in {"advance", "repeat"}:
            raise AutonomyEvidenceError(
                "Autonomous state commit requires a recorded advance/repeat decision"
            )
        if json_loads(decision["state_patch_json"], []) != patch:
            raise AutonomyEvidenceError(
                "State patch must exactly match the patch recorded in the decision"
            )
        self._validate_autonomy_patch_targets(lease, patch)

    def _set_autonomy_control(
        self,
        lease_id: str,
        client_id: str,
        reason: str,
        *,
        action: str,
    ) -> dict[str, Any]:
        if not client_id.strip() or not reason.strip():
            raise ServiceError("Autonomy control requires actor and reason")
        lease = self._expire_autonomy_if_needed(self._autonomy_row(lease_id))
        if action == "pause":
            if lease["status"] != "active":
                raise ServiceError("Only active autonomy can be paused")
            new_status = "paused"
        elif action in {"stop", "emergency_stop"}:
            if lease["status"] not in {"pending", "active", "paused"}:
                raise ServiceError(f"Autonomy lease is already {lease['status']}")
            new_status = "emergency_stopped" if action == "emergency_stop" else "revoked"
        else:
            raise ServiceError("Unknown autonomy control action")
        now = utc_now()
        self.db.execute(
            "UPDATE autonomy_leases SET status = ?, stopped_reason = ?, updated_at = ? "
            "WHERE id = ?",
            (new_status, reason.strip(), now, lease_id),
        )
        self.db.execute(
            "UPDATE measurement_sessions SET status = ?, updated_at = ? "
            "WHERE autonomy_lease_id = ?",
            (new_status, now, lease_id),
        )
        self.db.execute(
            "UPDATE proposals SET status = 'cancelled' WHERE autonomy_lease_id = ? "
            "AND status IN ('pending', 'approved') AND uses = 0",
            (lease_id,),
        )
        active_run = self.db.one(
            "SELECT * FROM runs WHERE autonomy_lease_id = ? "
            "AND status IN ('starting', 'running', 'stopping') "
            "ORDER BY started_at DESC LIMIT 1",
            (lease_id,),
        )
        stop_result = None
        if action in {"stop", "emergency_stop"} and active_run is not None:
            stop_result = self.runner.stop(active_run["id"], client_id)
            if action == "emergency_stop":
                self._schedule_force_stop(active_run["id"], client_id)
        self.db.event(
            f"autonomy_{action}",
            client_id,
            {"lease_id": lease_id, "reason": reason.strip(), "run": stop_result},
            lease["workflow_id"],
        )
        result = self.autonomy_status(lease_id=lease_id)
        result["active_run_control"] = stop_result
        return result

    def _schedule_force_stop(self, run_id: str, actor: str) -> None:
        grace = int(
            self.policy.raw["autonomy"].get("emergency_stop_grace_seconds", 30)
        )
        run = self.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
        if run is None or not run.get("autonomy_lease_id"):
            raise ServiceError("Emergency stop run is not bound to autonomy")
        now = datetime.now(timezone.utc)
        force_after = now + timedelta(seconds=max(0, grace))
        self.db.execute(
            """
            INSERT OR REPLACE INTO emergency_stops(
                run_id, autonomy_lease_id, requested_at, force_after, actor, status
            ) VALUES (?, ?, ?, ?, ?, 'pending')
            """,
            (
                run_id,
                run["autonomy_lease_id"],
                now.isoformat(timespec="seconds"),
                force_after.isoformat(timespec="seconds"),
                actor,
            ),
        )

        def force_after_grace() -> None:
            time.sleep(max(0, grace))
            self._force_pending_emergency_stop(run_id)

        threading.Thread(
            target=force_after_grace,
            name=f"jy-emergency-stop-{run_id[:8]}",
            daemon=True,
        ).start()

    def _enforce_pending_emergency_stops(self) -> int:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = self.db.all(
            "SELECT run_id FROM emergency_stops "
            "WHERE status = 'pending' AND force_after <= ?",
            (now,),
        )
        return sum(
            1 for row in rows if self._force_pending_emergency_stop(row["run_id"])
        )

    def _force_pending_emergency_stop(self, run_id: str) -> bool:
        pending = self.db.one(
            "SELECT * FROM emergency_stops WHERE run_id = ? AND status = 'pending'",
            (run_id,),
        )
        if pending is None:
            return False
        run = self.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
        try:
            if run and run["status"] in {"starting", "running", "stopping"}:
                result = self.runner.force_stop(run_id, pending["actor"])
            else:
                result = {
                    "run_id": run_id,
                    "forced": False,
                    "status": run["status"] if run else "missing",
                }
            status = "completed"
        except Exception as exc:
            result = {"run_id": run_id, "error": str(exc)}
            status = "failed"
            if run:
                self.db.event(
                    "emergency_force_stop_failed",
                    pending["actor"],
                    result,
                    run["workflow_id"],
                )
        self.db.execute(
            "UPDATE emergency_stops SET status = ?, completed_at = ?, result_json = ? "
            "WHERE run_id = ?",
            (status, utc_now(), json_dumps(result), run_id),
        )
        return bool(result.get("forced"))

    def _halt_autonomy(
        self,
        lease_id: str,
        reason: str,
        actor: str,
        *,
        stop_active: bool,
    ) -> dict[str, Any]:
        lease = self._autonomy_row(lease_id)
        if lease["status"] not in {"pending", "active", "paused"}:
            return self._decode_autonomy(lease)
        now = utc_now()
        self.db.execute(
            "UPDATE autonomy_leases SET status = 'halted', stopped_reason = ?, "
            "updated_at = ? WHERE id = ?",
            (reason, now, lease_id),
        )
        self.db.execute(
            "UPDATE measurement_sessions SET status = 'halted', updated_at = ? "
            "WHERE autonomy_lease_id = ?",
            (now, lease_id),
        )
        self.db.execute(
            "UPDATE proposals SET status = 'cancelled' WHERE autonomy_lease_id = ? "
            "AND status IN ('pending', 'approved') AND uses = 0",
            (lease_id,),
        )
        active_run = self.db.one(
            "SELECT id FROM runs WHERE autonomy_lease_id = ? "
            "AND status IN ('starting', 'running', 'stopping') LIMIT 1",
            (lease_id,),
        )
        if stop_active and active_run is not None:
            try:
                self.runner.stop(active_run["id"], actor)
            except Exception:
                pass
        self.db.event(
            "autonomy_halted",
            actor,
            {"lease_id": lease_id, "reason": reason},
            lease["workflow_id"],
        )
        return self.autonomy_status(lease_id=lease_id)

    @staticmethod
    def _is_recoverable_scheduling_block(exc: BaseException) -> bool:
        """True when the refusal happened before the worker reached hardware.

        Operator instruction 2026-09-20: this used to compare three exact
        message strings, so rewording any one of them silently turned a
        recoverable refusal into a lease-ending halt. Scope, quota, and policy
        refusals are now recognized by type, and the remaining message tests
        only cover the paused/stopped-workflow wording.
        """

        if isinstance(
            exc, (AutonomyScopeError, AutonomyQuotaError, PolicyError)
        ):
            return True
        message = str(exc)
        return (
            "; new actions are blocked" in message
            or message == "Workflow is not active"
            or message.startswith("Resume measurement mode before")
        )

    @staticmethod
    def _is_instrument_connectivity_failure(run: dict[str, Any]) -> bool:
        if str(run.get("termination_cause") or "") == "instrument_unreachable":
            return True
        analysis = json_loads(run.get("analysis_json"), {})
        return analysis.get("failure_category") == "instrument_unreachable"

    @staticmethod
    def _program_submission_timeout(run: dict[str, Any]) -> dict[str, Any] | None:
        """Return the analysis of a probe-confirmed submission timeout.

        The worker records this category only after the QOP answered a health
        check, so it is a program-size failure and not an outage.
        """

        analysis = json_loads(run.get("analysis_json"), {})
        cause = str(run.get("termination_cause") or "")
        if (
            cause == "program_submission_timeout"
            or analysis.get("failure_category") == "program_submission_timeout"
        ):
            return analysis if isinstance(analysis, dict) else {}
        return None

    def _pause_already_resumed(
        self, run: dict[str, Any], event_type: str
    ) -> bool:
        """Skip re-pausing a failure the operator has already resumed past.

        The watchdog re-examines the same failed run on every sweep, so
        without this the resume is undone in the same second it is granted and
        `恢復量測` can never take effect.
        """

        pause = self.db.one(
            "SELECT created_at FROM events "
            "WHERE workflow_id = ? AND event_type = ? "
            "AND json_extract(payload_json, '$.run_id') = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (run["workflow_id"], event_type, run["id"]),
        )
        if pause is None:
            return False
        resume = self.db.one(
            "SELECT created_at FROM events "
            "WHERE workflow_id = ? "
            "AND event_type IN ('measurement_mode_resumed', 'autonomy_resumed') "
            "AND created_at >= ? "
            "ORDER BY created_at DESC LIMIT 1",
            (run["workflow_id"], pause["created_at"]),
        )
        return resume is not None

    def _instrument_outage_already_resumed(self, run: dict[str, Any]) -> bool:
        """Skip re-pausing an outage the operator has already resumed past."""

        return self._pause_already_resumed(
            run, "measurement_paused_for_instrument_error"
        )

    def _pause_for_instrument_connectivity(
        self, run: dict[str, Any]
    ) -> None:
        """Idempotently pause scheduling after a recognized instrument outage."""

        if not self._is_instrument_connectivity_failure(run):
            return
        if self._instrument_outage_already_resumed(run):
            return
        self._pause_autonomy_for_review(
            run,
            f"Instrument connectivity error paused run {run['id']}; no new "
            "experiment will be scheduled until the operator resumes.",
            "measurement_paused_for_instrument_error",
            "instrument_failure_guard",
        )

    def _resolve_autonomy_duration(self, requested: float | None) -> float:
        """Clamp-check an entry-supplied lease budget against policy.

        Operator instruction 2026-09-20: automatic-mode entry may name the hours
        for that session instead of always taking the configured default. The
        human still approves the number shown on the Dashboard.
        """

        config = self.policy.raw["autonomy"]
        default_hours = float(config["duration_hours"])
        if requested is None:
            return default_hours
        if isinstance(requested, bool) or not isinstance(requested, (int, float)):
            raise ServiceError("duration_hours must be a number of hours")
        hours = float(requested)
        maximum = float(config.get("max_duration_hours", default_hours))
        if not math.isfinite(hours) or hours <= 0:
            raise ServiceError("duration_hours must be a positive number of hours")
        if hours > maximum:
            raise ServiceError(
                f"duration_hours {hours:g} exceeds the policy maximum of "
                f"{maximum:g} hours"
            )
        return hours

    def _record_refused_autonomy_action(
        self,
        lease_id: str,
        workflow_id: str,
        category: str,
        detail: str,
    ) -> None:
        """Audit an in-scope action that was refused before it reached hardware.

        Operator instruction 2026-09-20. Refusing the action is the protection;
        ending the lease on top of that only costs a new human approval and
        stops a bring-up that could have continued. The refusal stays visible on
        the Dashboard and in the audit log.
        """

        self.db.event(
            "autonomy_action_refused",
            "autonomy_guard",
            {"lease_id": lease_id, "category": category, "detail": detail},
            workflow_id,
        )

    def recoverable_worker_exception(
        self, run: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Match a worker crash against the registered recoverable signatures.

        Operator instruction 2026-09-20: a crash that is a parameter the
        protected node itself rejects is deterministic and correctable, so it
        leaves the lease usable. Everything unrecognized pauses for a human.
        """

        traceback_text = str(run.get("error") or "")
        if not traceback_text:
            return None
        node_id = str(run.get("node_id") or "")
        parameters = json_loads(run.get("parameters_json"), {})
        parameter_names = set(parameters) if isinstance(parameters, dict) else set()
        folded = traceback_text.casefold()
        signatures = self.policy.raw["autonomy"].get(
            "recoverable_worker_exceptions", []
        )
        for signature in signatures if isinstance(signatures, list) else []:
            if not isinstance(signature, dict):
                continue
            nodes = signature.get("nodes") or []
            if nodes and node_id not in {str(item) for item in nodes}:
                continue
            types = [str(item).casefold() for item in signature.get("exception_types", [])]
            if types and not any(f"{item}:" in folded or f"{item} " in folded for item in types):
                continue
            markers = [str(item).casefold() for item in signature.get("message_markers", [])]
            if markers and not any(marker in folded for marker in markers):
                continue
            required = {str(item) for item in signature.get("parameters", [])}
            if required and not (required & parameter_names):
                continue
            return {
                "id": str(signature.get("id") or "unnamed_signature"),
                "remedy": str(signature.get("remedy") or "").strip(),
                "parameters": sorted(required & parameter_names),
            }
        return None

    def _pause_autonomy_for_review(
        self,
        run: dict[str, Any],
        reason: str,
        event_type: str,
        actor: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Pause the lease and scheduling without ending the authorization.

        The operator inspects the Dashboard and resumes the same lease with
        `恢復量測`; no new human approval is needed.
        """

        now = utc_now()
        with self.db.transaction(immediate=True) as connection:
            changed = 0
            lease_id = run.get("autonomy_lease_id")
            if lease_id:
                changed += connection.execute(
                    "UPDATE autonomy_leases SET status = 'paused', stopped_reason = ?, "
                    "updated_at = ? WHERE id = ? AND status = 'active'",
                    (reason, now, lease_id),
                ).rowcount
            else:
                changed += connection.execute(
                    "UPDATE workflows SET status = 'paused', updated_at = ? "
                    "WHERE id = ? AND status = 'active'",
                    (now, run["workflow_id"]),
                ).rowcount
            changed += connection.execute(
                "UPDATE measurement_sessions SET status = 'paused', updated_at = ? "
                "WHERE workflow_id = ? AND status IN ('active', 'halted')",
                (now, run["workflow_id"]),
            ).rowcount
            if changed:
                self.db.event(
                    event_type,
                    actor,
                    {
                        "run_id": run["id"],
                        "reason": reason,
                        "hardware_lock_retained": self.settings.lock_path.exists(),
                        **(detail or {}),
                    },
                    run["workflow_id"],
                    connection=connection,
                )

    def _enforce_run_hard_stops(self, run: dict[str, Any]) -> None:
        lease_id = run.get("autonomy_lease_id")
        if not lease_id:
            return
        lease = self._autonomy_row(str(lease_id))
        if lease["status"] not in {"active", "paused"}:
            return
        if run["status"] in {"stopped", "cancelled_by_shutdown"}:
            return
        if run["status"] in {"failed", "force_stopped"}:
            if self._is_instrument_connectivity_failure(run):
                self._pause_for_instrument_connectivity(run)
                return
            submission = self._program_submission_timeout(run)
            if submission is not None:
                # A live probe answered after the timeout, so the instrument is
                # healthy and the program was simply too large to submit. The
                # remedy is a smaller multiplex group, which the agent can do
                # without an operator, so the lease stays active. The worker
                # already paused if a single target had left no room to halve.
                if not submission.get("measurement_paused"):
                    self.db.event(
                        "autonomy_program_submission_timeout",
                        "autonomy_watchdog",
                        {
                            "run_id": run["id"],
                            "attempted_target_count": submission.get(
                                "attempted_target_count"
                            ),
                            "instrument_probe": submission.get(
                                "instrument_probe"
                            ),
                            "remedy": (
                                "Halve the multiplex group and repeat the node."
                            ),
                        },
                        run["workflow_id"],
                    )
                return
            recoverable = self.recoverable_worker_exception(run)
            if recoverable is not None:
                # Registered node-parameter rejection: nothing ran, the cause is
                # known, and the remedy is a different parameter. Keep the lease.
                self.db.event(
                    "autonomy_recoverable_worker_exception",
                    "autonomy_watchdog",
                    {
                        "run_id": run["id"],
                        "signature": recoverable["id"],
                        "parameters": recoverable["parameters"],
                        "remedy": recoverable["remedy"],
                    },
                    run["workflow_id"],
                )
                return
            if self._pause_already_resumed(
                run, "autonomy_paused_for_worker_failure"
            ):
                # The operator has seen this failure and resumed past it.
                # Re-pausing here would undo the resume on the next sweep and
                # leave the lease permanently stuck.
                return
            self._pause_autonomy_for_review(
                run,
                f"Unrecognized worker failure for run {run['id']} paused "
                "scheduling for operator review: "
                f"{run.get('error') or run['status']}",
                "autonomy_paused_for_worker_failure",
                "autonomy_watchdog",
                {"termination_cause": run.get("termination_cause")},
            )
            return
        if run["status"] == "completed":
            snapshot_path = run.get("snapshot_path")
            if (
                run.get("snapshot_id") is None
                or not snapshot_path
                or not Path(str(snapshot_path)).is_dir()
            ):
                self._halt_autonomy(
                    str(lease_id),
                    f"Snapshot is missing for completed run {run['id']}",
                    "autonomy_watchdog",
                    stop_active=False,
                )
            return
        if run["status"] in {"starting", "running", "stopping"}:
            pid = run.get("pid")
            if isinstance(pid, int) and not self.runner.process_is_alive(pid):
                evidence = self.runner.exit_evidence(str(run["id"]))
                stop_intent = str(
                    run.get("stop_intent")
                    or evidence.get("stop_request", {}).get("intent")
                    or ""
                )
                if stop_intent:
                    terminal_status = (
                        "cancelled_by_shutdown"
                        if stop_intent == "full_shutdown"
                        else "stopped"
                    )
                    cause = str(
                        evidence.get("exit_receipt", {}).get("termination_cause")
                        or "intentional_worker_exit"
                    )
                    self.db.execute(
                        "UPDATE runs SET status = ?, finished_at = ?, "
                        "termination_cause = ? WHERE id = ?",
                        (terminal_status, utc_now(), cause, run["id"]),
                    )
                    self.db.event(
                        "run_cancelled_by_shutdown"
                        if terminal_status == "cancelled_by_shutdown"
                        else "run_stopped",
                        "autonomy_watchdog",
                        {
                            "run_id": run["id"],
                            "intent": stop_intent,
                            "exit_evidence": evidence,
                            "hardware_lock_retained": self.settings.lock_path.exists(),
                        },
                        run["workflow_id"],
                    )
                    return
                message = (
                    f"Worker process {pid} exited without recording completion for "
                    f"run {run['id']}; hardware lock retained for inspection."
                )
                self.db.execute(
                    "UPDATE runs SET status = 'failed', finished_at = ?, error = ? "
                    "WHERE id = ?",
                    (utc_now(), message, run["id"]),
                )
                self._halt_autonomy(
                    str(lease_id),
                    message,
                    "autonomy_watchdog",
                    stop_active=False,
                )
                return
            try:
                lock = json.loads(
                    self.settings.lock_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                self._halt_autonomy(
                    str(lease_id),
                    f"Hardware lock is missing or unreadable for run {run['id']}: {exc}",
                    "autonomy_watchdog",
                    stop_active=True,
                )
                return
            if lock.get("run_id") != run["id"]:
                self._halt_autonomy(
                    str(lease_id),
                    f"Hardware lock owner does not match run {run['id']}",
                    "autonomy_watchdog",
                    stop_active=True,
                )

    def _decode_autonomy(self, row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["targets"] = json_loads(result.pop("targets_json"), [])
        result["allowed_nodes"] = json_loads(result.pop("allowed_nodes_json"), [])
        result["auto_state_commit_analysis_statuses"] = json_loads(
            result.pop("auto_state_commit_statuses_json"), []
        )
        result["halt_conditions"] = json_loads(
            result.pop("halt_conditions_json"), []
        )
        attempts: dict[str, dict[str, int]] = {}
        for run in self.db.all(
            "SELECT node_id, parameters_json FROM runs WHERE autonomy_lease_id = ?",
            (result["id"],),
        ):
            node_counts = attempts.setdefault(run["node_id"], {})
            for target in json_loads(run["parameters_json"], {}).get("qubits", []):
                node_counts[target] = node_counts.get(target, 0) + 1
        result["attempts_by_node_qubit"] = attempts
        quota_reached: dict[str, list[str]] = {}
        incomplete: dict[str, list[str]] = {}
        resolved_by_node: dict[str, list[str]] = {}
        scientifically_unmeasurable = (
            self._scientifically_unmeasurable_targets_by_node(
                result["workflow_id"], lease_id=result["id"]
            )
        )
        max_attempts = int(result["max_attempts_per_node_qubit"])
        for node_id in set(attempts) | set(scientifically_unmeasurable):
            node_counts = attempts.get(node_id, {})
            reached = {
                target
                for target, count in node_counts.items()
                if count >= max_attempts
            }
            resolved = self._node_target_resolution(
                result["workflow_id"], node_id
            )
            if resolved:
                resolved_by_node[node_id] = sorted(resolved)
            if reached:
                quota_reached[node_id] = sorted(reached)
            unresolved = (
                reached | scientifically_unmeasurable.get(node_id, set())
            ) - resolved
            if unresolved:
                incomplete[node_id] = sorted(unresolved)
        result["resolved_targets_by_node"] = resolved_by_node
        result["quota_reached_by_node"] = quota_reached
        result["scientifically_unmeasurable_targets_by_node"] = {
            node_id: sorted(targets)
            for node_id, targets in scientifically_unmeasurable.items()
            if targets
        }
        result["incomplete_targets_by_node"] = incomplete
        result["total_runs"] = sum(
            1
            for _ in self.db.all(
                "SELECT id FROM runs WHERE autonomy_lease_id = ?", (result["id"],)
            )
        )
        result["control_url"] = self.autonomy_control_url(result["id"])
        result["manual_controls"] = {
            "pause": "Finish the active run but schedule nothing new.",
            "stop": "Revoke the lease and gracefully stop the active worker.",
            "emergency_stop": (
                "Revoke, request graceful stop, then terminate the exact worker "
                "after the grace period while retaining the hardware lock."
            ),
        }
        result["continuation_policy"] = {
            "mode": "continue_until_workflow_complete",
            "continue_after_needs_review": True,
            "continue_other_resolved_qubits_after_quota": True,
            "terminal_conditions": [
                "scientifically_unmeasurable_target",
                "per_node_qubit_attempt_limit",
                "authorization_time_limit",
                "workflow_complete",
                "operator_stop",
                *result["halt_conditions"],
            ],
        }
        return result

    def _latest_review_run_id(
        self, workflow_id: str, proposed_node: str
    ) -> str | None:
        if not proposed_node:
            return None
        row = self.db.one(
            """
            SELECT d.run_id
            FROM decisions d
            JOIN runs r ON r.id = d.run_id
            WHERE d.workflow_id = ?
              AND (d.next_node = ? OR (r.node_id = ? AND d.decision IN ('repeat', 'manual_review')))
            ORDER BY d.created_at DESC
            LIMIT 1
            """,
            (workflow_id, proposed_node, proposed_node),
        )
        return str(row["run_id"]) if row is not None else None

    def _latest_workflow_decision_run_id(self, workflow_id: str) -> str | None:
        row = self.db.one(
            "SELECT run_id FROM decisions WHERE workflow_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (workflow_id,),
        )
        return str(row["run_id"]) if row is not None else None

    def _approved_proposal(self, proposal_id: str, kind: str) -> dict[str, Any]:
        proposal = self.db.one("SELECT * FROM proposals WHERE id = ?", (proposal_id,))
        if proposal is None:
            raise ServiceError(f"Unknown proposal: {proposal_id}")
        if proposal["kind"] != kind or proposal["status"] != "approved":
            raise ServiceError(f"An approved {kind} proposal is required")
        if proposal["uses"] >= proposal["max_uses"]:
            raise ServiceError("Proposal has already been consumed")
        if datetime.fromisoformat(proposal["expires_at"]) <= datetime.now(timezone.utc):
            raise ServiceError("Proposal has expired")
        return proposal

    def _02c_target_resolution(self, workflow_id: str) -> tuple[set[str], set[str]]:
        rows = self.db.all(
            "SELECT analysis_status, analysis_json FROM runs "
            "WHERE workflow_id = ? AND node_id = '02c'",
            (workflow_id,),
        )
        return resolve_02c_targets(rows)

    def _node_target_resolution(
        self, workflow_id: str, node_id: str
    ) -> set[str]:
        rows = self.db.all(
            "SELECT r.id AS run_id, r.status, r.analysis_status, r.analysis_json, "
            "r.parameters_json, r.started_at, r.rowid AS run_rowid, "
            "d.decision, d.state_patch_json "
            "FROM runs r LEFT JOIN decisions d ON d.run_id = r.id "
            "WHERE r.workflow_id = ? AND r.node_id = ? ORDER BY r.rowid",
            (workflow_id, node_id),
        )
        cutoff = self._readout_reopen_cutoff(workflow_id, node_id)
        if cutoff is not None:
            rows = [row for row in rows if str(row.get("started_at") or "") >= cutoff]

        def resolve(selected: list[dict[str, Any]]) -> set[str]:
            found = self._resolve_targets_from_rows(node_id, selected)
            if node_id in WITHHOLD_AWARE_NODES:
                found -= _withheld_targets(selected)
            return found

        resolved = resolve(rows)
        if node_id in {"03a", "04"}:
            resolved = self._drop_returned_targets(workflow_id, rows, resolved, resolve)
        if node_id == "02c":
            resolved = self._02c_committable_targets(workflow_id, rows, resolved)
        if node_id != "02c":
            resolved |= self._operator_committed_targets(workflow_id, rows, node_id)
        if node_id in SUBGROUP_NODES:
            return resolved
        for row in rows:
            if (
                row.get("status") == "completed"
                and row.get("analysis_status") == "pass"
            ):
                parameters = json_loads(row.get("parameters_json"), {})
                names = parameters.get("qubits", [])
                if isinstance(names, list):
                    resolved.update(str(name) for name in names)
        return resolved

    def _operator_committed_targets(
        self, workflow_id: str, rows: list[dict[str, Any]], node_id: str
    ) -> set[str]:
        """Qubits an operator-approved commit from one of these runs wrote.

        Operator choice 2026-10-14 (CRSQ_26009_die(5,-1) q3 at 04): a result
        below the SNR floor at the averaging ceiling but reproduced across
        runs may be committed with a human approval; that commit resolves the
        qubit at the node. Lease-delegated commits do not count here, and
        only paths the node itself writes count (a readout-angle fix tied to
        a 05 run is not a T1 result).
        """
        run_ids = {str(row.get("run_id")) for row in rows if row.get("run_id")}
        found: set[str] = set()
        for proposal in self.db.all(
            "SELECT payload_json FROM proposals WHERE workflow_id = ? "
            "AND kind = 'state_commit' AND status = 'consumed' "
            "AND autonomy_lease_id IS NULL",
            (workflow_id,),
        ):
            payload = json_loads(proposal.get("payload_json"), {})
            if str(payload.get("run_id")) not in run_ids:
                continue
            patch = payload.get("patch") or []
            names = sorted(patch_qubit_targets(patch))
            node_changes, _ = self.analyzer._target_node_changes(patch, node_id, names)
            found |= patch_qubit_targets(node_changes)
        return found

    def _02c_committable_targets(
        self, workflow_id: str, rows: list[dict[str, Any]], resolved: set[str]
    ) -> set[str]:
        """Drop 02c targets whose transition was found but not committable.

        Operator choice 2026-10-14 (q3, #507): a transition counts only when
        its run passed the target (e.g. the readout fits the line full scale)
        or a committed state proposal for one of its runs wrote that qubit.
        A confirmed-absent (bare-only) target stays resolved.
        """
        committed: dict[str, set[str]] = {}
        for proposal in self.db.all(
            "SELECT payload_json FROM proposals WHERE workflow_id = ? "
            "AND kind = 'state_commit' AND status = 'consumed'",
            (workflow_id,),
        ):
            payload = json_loads(proposal.get("payload_json"), {})
            run_id = payload.get("run_id")
            if not run_id:
                continue
            committed.setdefault(str(run_id), set()).update(
                patch_qubit_targets(payload.get("patch") or [])
            )
        _, absent = resolve_02c_targets(rows)
        keep = resolved & absent
        for row in rows:
            if row.get("analysis_status") not in {"pass", "needs_review"}:
                continue
            analysis = json_loads(row.get("analysis_json"), {})
            if row.get("analysis_status") == "pass":
                keep |= resolved & set(
                    analysis.get("dataset_metrics", {}).get("qubits", {})
                )
            keep |= resolved & {str(n) for n in analysis.get("passing_targets") or []}
            keep |= resolved & committed.get(str(row.get("run_id")), set())
        return keep

    def _resolve_targets_from_rows(
        self, node_id: str, rows: list[dict[str, Any]]
    ) -> set[str]:
        analysis_rules = self.policy.raw["analysis"]
        rules_03a = analysis_rules["03a"]

        def resolve(selected: list[dict[str, Any]], min_03a_snr: float) -> set[str]:
            return resolve_node_targets(
                node_id,
                selected,
                min_03a_snr,
                float(analysis_rules["04"]["min_robust_snr"]),
                float(analysis_rules["05"]["min_robust_snr"]),
                1e6 * float(rules_03a["final_min_feature_fwhm_mhz"]),
                1e6 * float(rules_03a["final_max_feature_fwhm_mhz"]),
                float(analysis_rules.get("02x", {}).get("min_robust_snr", 3.0)),
                float(analysis_rules.get("02a", {}).get("min_robust_snr", 3.0)),
            )

        found = resolve(rows, float(rules_03a["min_robust_snr"]))
        cross_lo_floor = rules_03a.get("cross_lo_fine_min_robust_snr")
        if node_id == "03a" and cross_lo_floor is not None:
            # 2026-10-12: a fine line the analyzer passed under the cross-LO
            # floor resolves at that floor (only the marked qubits of that run).
            for row in rows:
                results = (
                    json_loads(row.get("analysis_json"), {})
                    .get("fit_quality", {})
                    .get("results", {})
                )
                marked = {
                    str(name)
                    for name, fit in (results.items() if isinstance(results, dict) else [])
                    if isinstance(fit, dict) and fit.get("cross_lo_confirmed")
                }
                if marked:
                    found |= marked & resolve([row], float(cross_lo_floor))
        return found

    def _autonomy_incomplete_targets_for_node(
        self, workflow_id: str, node_id: str
    ) -> set[str]:
        lease = self.db.one(
            "SELECT * FROM autonomy_leases WHERE workflow_id = ? "
            "AND status IN ('active', 'paused') ORDER BY created_at DESC LIMIT 1",
            (workflow_id,),
        )
        if lease is None:
            return set()
        max_attempts = int(lease["max_attempts_per_node_qubit"])
        counts: dict[str, int] = {}
        for run in self.db.all(
            "SELECT parameters_json FROM runs WHERE autonomy_lease_id = ? "
            "AND node_id = ?",
            (lease["id"], node_id),
        ):
            for target in set(
                json_loads(run.get("parameters_json"), {}).get("qubits", [])
            ):
                counts[str(target)] = counts.get(str(target), 0) + 1
        quota_reached = {
            target for target, count in counts.items() if count >= max_attempts
        }
        scientifically_unmeasurable = (
            self._scientifically_unmeasurable_targets_by_node(
                workflow_id, lease_id=str(lease["id"])
            ).get(node_id, set())
        )
        return (
            quota_reached | scientifically_unmeasurable
        ) - self._node_target_resolution(workflow_id, node_id)

    def _scientifically_unmeasurable_targets_by_node(
        self, workflow_id: str, *, lease_id: str | None = None
    ) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        cutoffs = self._return_cutoffs(workflow_id)
        for row in self.db.all(
            "SELECT id, payload_json FROM events WHERE workflow_id = ? "
            "AND event_type = 'autonomy_targets_scientifically_unmeasurable' "
            "ORDER BY id",
            (workflow_id,),
        ):
            payload = json_loads(row.get("payload_json"), {})
            if lease_id is not None and payload.get("lease_id") != lease_id:
                continue
            node_id = str(payload.get("node_id", ""))
            targets = payload.get("targets", [])
            if not node_id or not isinstance(targets, list):
                continue
            names = {str(item) for item in targets}
            if node_id in {"03a", "04"}:
                # Marks a qubit received before returning to 03a are void.
                names = {
                    name
                    for name in names
                    if name not in cutoffs or int(row["id"]) > cutoffs[name]["event_id"]
                }
            result.setdefault(node_id, set()).update(names)
        return result

    def _03a_candidate_targets(self, workflow_id: str) -> set[str]:
        rows = self.db.all(
            "SELECT r.status, r.analysis_status, r.analysis_json, "
            "r.parameters_json, r.rowid AS run_rowid, d.decision FROM runs r "
            "LEFT JOIN decisions d ON d.run_id = r.id "
            "WHERE r.workflow_id = ? AND r.node_id = '03a' ORDER BY r.rowid",
            (workflow_id,),
        )
        rules = self.policy.raw["analysis"]["03a"]

        def resolve(selected: list[dict[str, Any]]) -> set[str]:
            return resolve_03a_candidate_targets(selected, float(rules["min_robust_snr"]))

        return self._drop_returned_targets(
            workflow_id, rows, resolve(rows), resolve
        ) | set(self._03a_cross_lo_candidates(workflow_id))

    def _03a_cross_lo_candidates(self, workflow_id: str) -> dict[str, float]:
        """Sub-floor coarse lines confirmed at the same RF under two LOs.

        Operator choice 2026-10-12 (q3 at 2.226 GHz, SNR 5.3-5.9): a coarse or
        refinement feature below the 03a SNR floor is still a credible
        candidate when two runs of the qubit put it at the same RF (within
        `cross_lo_rf_tolerance_mhz`) from IFs at least
        `cross_lo_min_if_separation_mhz` apart -- i.e. under different LOs, so
        it is not a fixed-IF artefact -- each with robust SNR at least
        `cross_lo_min_robust_snr`. Lines 04 rejected stay masked. Returns the
        latest confirmed RF per qubit.
        """

        rules = self.policy.raw["analysis"]["03a"]
        min_snr = float(rules.get("cross_lo_min_robust_snr") or 0.0)
        if min_snr <= 0.0:
            return {}
        tolerance = 1e6 * float(rules.get("cross_lo_rf_tolerance_mhz", 5.0))
        separation = 1e6 * float(rules.get("cross_lo_min_if_separation_mhz", 50.0))
        rejected = self._rejected_03a_lines(workflow_id)
        workflow = self._workflow(workflow_id)
        found: dict[str, float] = {}
        for qubit in sorted(set(json_loads(workflow["targets_json"], []))):
            sightings: list[tuple[float, float]] = []
            for row in self._qubit_runs(workflow_id, "03a", qubit):
                if row.get("analysis_status") not in {"pass", "needs_review", "failed"}:
                    continue
                analysis = json_loads(row.get("analysis_json"), {})
                if analysis.get("03a_stage") == "fine":
                    continue
                metrics = (
                    analysis.get("dataset_metrics", {}).get("qubits", {}).get(qubit, {})
                )
                if not isinstance(metrics, dict):
                    continue
                snr = metrics.get("robust_snr")
                intermediate = metrics.get("feature_coordinate")
                if not (_finite_number(snr) and _finite_number(intermediate)):
                    continue
                if float(snr) < min_snr:
                    continue
                lo_hz = self._03a_run_drive_lo(analysis.get("snapshot_path"), qubit)
                if lo_hz is None:
                    continue
                rf = lo_hz + float(intermediate)
                if any(low <= rf <= high for low, high in rejected.get(qubit, [])):
                    continue
                for earlier_rf, earlier_if in sightings:
                    if (
                        abs(rf - earlier_rf) <= tolerance
                        and abs(float(intermediate) - earlier_if) >= separation
                    ):
                        found[qubit] = rf
                sightings.append((rf, float(intermediate)))
        return found

    @staticmethod
    def _03a_run_drive_lo(snapshot_path: Any, qubit: str) -> float | None:
        """The XY LO a 03a run used, read from its snapshot state."""

        if not isinstance(snapshot_path, str) or not snapshot_path:
            return None
        try:
            snapshot = load_state(Path(snapshot_path) / "quam_state.json")
        except Exception:
            return None
        reference = _xy_output_of(snapshot, snapshot, qubit)
        if reference is None:
            return None
        node: Any = snapshot
        for part in reference.lstrip("#/").split("/"):
            node = node.get(part) if isinstance(node, dict) else None
        value = node.get("upconverter_frequency") if isinstance(node, dict) else None
        return float(value) if _finite_number(value) else None

    def _03a_shift_ready_targets(self, workflow_id: str) -> set[str]:
        rows = self.db.all(
            "SELECT r.status, r.analysis_status, r.analysis_json, "
            "r.parameters_json, r.rowid AS run_rowid, d.decision FROM runs r "
            "LEFT JOIN decisions d ON d.run_id = r.id "
            "WHERE r.workflow_id = ? AND r.node_id = '03a' ORDER BY r.rowid",
            (workflow_id,),
        )
        cutoffs = self._return_cutoffs(workflow_id)
        if cutoffs:
            # A qubit that returned from 04 shifts on its new evidence only.
            kept = []
            for row in rows:
                names = json_loads(row.get("parameters_json"), {}).get("qubits", [])
                stale = any(
                    str(name) in cutoffs
                    and int(row.get("run_rowid") or 0) <= cutoffs[str(name)]["run_rowid"]
                    for name in (names if isinstance(names, list) else [])
                )
                if not stale:
                    kept.append(row)
            rows = kept
        rules = self.policy.raw["analysis"]["03a"]
        return resolve_03a_shift_ready_targets(
            rows,
            min_snr=float(rules["min_robust_snr"]),
            min_edge_fraction=float(rules["min_edge_fraction"]),
            max_noise_confirmation_averages=int(
                rules["max_noise_confirmation_num_averages"]
            ),
        )

    def _node_has_completed_run(self, workflow_id: str, node_id: str) -> bool:
        cutoff = self._readout_reopen_cutoff(workflow_id, node_id) or ""
        row = self.db.one(
            "SELECT id FROM runs WHERE workflow_id = ? AND node_id = ? "
            "AND status = 'completed' AND COALESCE(started_at, '') >= ? LIMIT 1",
            (workflow_id, node_id, cutoff),
        )
        return row is not None

    def _validate_shared_parameter_first_batch(
        self,
        workflow: dict[str, Any],
        node_id: str,
        requested: set[str],
    ) -> None:
        if node_id not in SHARED_FIRST_BATCH_NODES:
            return
        cap = self._node_multiplex_cap(node_id)
        groups = self._reset_groups(workflow, node_id)
        if groups:
            # One shared first batch per reset group (2026-10-04).
            for group, members in groups.items():
                if requested & members and not self._group_has_completed_run(
                    workflow, node_id, group, members
                ) and requested != members:
                    raise AutonomyScopeError(
                        f"The first {group}-reset {node_id} run must multiplex "
                        f"every {group}-reset target {sorted(members)}"
                    )
            return
        if self._node_has_completed_run(workflow["id"], node_id):
            return
        active = self._active_targets_for_node(workflow, node_id)
        if cap is not None and len(active) > cap:
            # A capped node has no full-width batch to demand; the cap itself
            # is the shared-parameter unit.
            return
        if requested != active:
            if self._node_full_width_submission_failed(
                workflow["id"], node_id, active
            ):
                return
            raise AutonomyScopeError(
                f"The first {node_id} run must multiplex every active target "
                f"{sorted(active)} with the same parameters"
            )

    def _shared_xy_drive_cap(self, node_id: str) -> int | None:
        """The cap the shared-XY-drive rule imposes on this node, if any."""

        try:
            return self.policy.shared_xy_drive_cap(node_id, self._drive_mode())
        except Exception:  # pragma: no cover - wiring problems surface elsewhere
            return None

    def _node_multiplex_cap(self, node_id: str) -> int | None:
        """Largest multiplex group this node allows, if it caps one.

        Operator instruction 2026-09-21: 04 never submits more than five
        targets at once on this hardware. Nodes without the key are uncapped
        and keep multiplexing every active target.

        Operator instruction 2026-09-22: a chip that wires two or more qubits
        to the same XY output caps every node from `03a` onward at one qubit
        per run, which overrides a larger per-node cap.
        """

        caps: list[int] = []
        try:
            definition = self.policy.node_definition(node_id)
        except Exception:  # pragma: no cover - unknown node handled elsewhere
            definition = {}
        value = definition.get("max_multiplex_targets")
        if not isinstance(value, bool) and isinstance(value, int) and value > 0:
            caps.append(value)
        # Operator request 2026-10-11: a chip listed for this node measures one
        # qubit per run (02x diagnostic on CRSQ_26009_die(5,-1)).
        devices = definition.get("single_qubit_devices") or []
        if Path(self.settings.quam_state_root).name in {str(item) for item in devices}:
            caps.append(1)
        shared = self._shared_xy_drive_cap(node_id)
        if shared is not None:
            caps.append(shared)
        return min(caps) if caps else None

    def _node_is_advisory(self, node_id: str) -> bool:
        """True for a survey node whose result never blocks the workflow.

        Operator request 2026-10-04 for 02: a broadband survey that finds fewer
        candidates than expected, or no design match, still hands over to 02x,
        which fine-tunes from the stored readout frequencies anyway.
        """

        try:
            return bool(self.policy.node_definition(node_id).get("advisory", False))
        except Exception:  # pragma: no cover - unknown node handled elsewhere
            return False

    def _node_is_single_pass(self, node_id: str) -> bool:
        """True for a node that runs once on defaults and then moves on.

        Operator instruction 2026-09-21 for 07d and 07b: one full-width run
        with the node defaults, no retry, no parameter adjustment and no
        per-target chasing. Passing evidence still commits, because that is
        the node's output; a target that does not pass simply carries no
        update and does not hold the workflow up.
        """

        try:
            definition = self.policy.node_definition(node_id)
        except Exception:  # pragma: no cover - unknown node handled elsewhere
            return False
        return bool(definition.get("single_pass"))

    def _validate_single_pass_node(
        self, workflow: dict[str, Any], node_id: str, merged: dict[str, Any]
    ) -> None:
        """Refuse a second run at a node that runs once on defaults.

        The one exception is the active-reset repeat the operator asked for at
        07b: a qubit whose thermal run cleared the trigger fidelity earns
        exactly one more 07b run with active reset, and that repeat is its
        accepted result.
        """

        if not self._node_is_single_pass(node_id):
            return
        if not self._node_has_completed_run(workflow["id"], node_id):
            return
        if self._is_active_reset_repeat(workflow, node_id, merged):
            return
        # Single pass means one pass over each target, not one run over the
        # whole chip. A node that has to be split into multiplex subgroups
        # still needs a run per subgroup, so only refuse targets that have
        # already been measured here.
        measured = self._single_pass_measured_targets(workflow, node_id)
        requested = {str(name) for name in merged.get("qubits", [])}
        repeated = sorted(requested & measured)
        if not repeated:
            return
        raise AutonomyScopeError(
            f"{node_id} runs once per target on the node defaults and then "
            f"advances; {repeated} already have a completed run here, so do "
            "not schedule another for them"
        )

    def _single_pass_measured_targets(
        self, workflow: dict[str, Any], node_id: str
    ) -> set[str]:
        """Targets already measured by a completed run at this node.

        Operator instruction 2026-10-02: a run counts only if it used the
        current node defaults for `single_pass_default_keys`. Widening 07d's
        amplitude sweep therefore makes its earlier narrow-sweep runs eligible
        for one more pass, without loosening single pass otherwise.
        """

        definition = self.policy.node_definition(node_id)
        defaults = definition.get("defaults", {})
        keys = list(definition.get("single_pass_default_keys", []))
        cutoff = self._readout_reopen_cutoff(workflow["id"], node_id)
        measured: set[str] = set()
        for run in self.db.all(
            "SELECT parameters_json, started_at FROM runs WHERE workflow_id = ? "
            "AND node_id = ? AND status = 'completed'",
            (workflow["id"], node_id),
        ):
            if cutoff is not None and str(run.get("started_at") or "") < cutoff:
                continue
            parameters = json_loads(run.get("parameters_json"), {})
            if any(
                key in parameters and parameters[key] != defaults.get(key)
                for key in keys
            ):
                continue
            names = parameters.get("qubits", [])
            if isinstance(names, list):
                measured.update(str(name) for name in names)
        return measured

    def _dressed_power_limits(self, workflow_id: str) -> dict[str, float]:
        """Latest passing 02c dressed-limit power per qubit (2026-10-03 cap)."""

        limits: dict[str, float] = {}
        for row in self.db.all(
            "SELECT analysis_json FROM runs WHERE workflow_id = ? AND node_id = '02c' "
            "AND status = 'completed' AND analysis_status IN ('pass', 'needs_review') "
            "ORDER BY rowid",
            (workflow_id,),
        ):
            analysis = json_loads(row.get("analysis_json"), {})
            qubits = (analysis.get("dataset_metrics") or {}).get("qubits") or {}
            for name, entry in qubits.items():
                transition = entry.get("02c_transition") if isinstance(entry, dict) else None
                if not isinstance(transition, dict) or transition.get("validation_failures"):
                    continue
                value = transition.get("dressed_power_limit_dbm")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    limits[str(name)] = float(value)
        return limits

    def _readout_reopen_cutoff(self, workflow_id: str, node_id: str) -> str | None:
        """Time of the latest 07d reopen; earlier 07d/07b runs no longer count.

        For 02 it is the latest applied readout-LO recentre (2026-10-07): the
        broadband search before the LO move no longer counts as its single pass.
        """

        if node_id == "02":
            row = self.db.one(
                "SELECT created_at FROM proposals WHERE workflow_id = ? "
                "AND kind = 'state_commit' AND uses > 0 "
                "AND payload_json LIKE '%\"readout_lo_recenter\"%' "
                "ORDER BY created_at DESC LIMIT 1",
                (workflow_id,),
            )
            return str(row["created_at"]) if row else None
        if node_id not in READOUT_REOPEN_NODES:
            return None
        row = self.db.one(
            "SELECT created_at FROM events WHERE workflow_id = ? "
            "AND event_type = '07d_reopened' ORDER BY id DESC LIMIT 1",
            (workflow_id,),
        )
        return str(row["created_at"]) if row else None

    def _is_active_reset_repeat(
        self, workflow: dict[str, Any], node_id: str, merged: dict[str, Any]
    ) -> bool:
        """True for the single permitted active-reset repeat at 07b."""

        if node_id != str(
            self.policy.raw["reset_policy"]["active_qualification_node"]
        ):
            return False
        if _reset_type(merged) != "active":
            return False
        qualifying = self._active_reset_repeat_targets(workflow)
        if qualifying is None:
            return False
        # One repeat per qualifying target, not one repeat per node: a chip
        # with a shared XY drive output takes them one qubit at a time.
        already = self._active_reset_repeated_targets(workflow)
        requested = {str(name) for name in merged.get("qubits", [])}
        if requested & already:
            return False
        return bool(requested) and requested <= qualifying

    def _active_reset_repeated_targets(self, workflow: dict[str, Any]) -> set[str]:
        """Targets that have already taken their 07b active-reset repeat."""

        node_id = str(self.policy.raw["reset_policy"]["active_qualification_node"])
        repeated: set[str] = set()
        # After a 07d reopen only the new pass counts (2026-10-04).
        cutoff = self._readout_reopen_cutoff(workflow["id"], node_id)
        for run in self.db.all(
            "SELECT parameters_json, started_at FROM runs WHERE workflow_id = ? "
            "AND node_id = ?",
            (workflow["id"], node_id),
        ):
            if cutoff is not None and str(run.get("started_at") or "") < cutoff:
                continue
            parameters = json_loads(run.get("parameters_json"), {})
            if _reset_type(parameters) != "active":
                continue
            names = parameters.get("qubits", [])
            if isinstance(names, list):
                repeated.update(str(name) for name in names)
        return repeated

    def _active_reset_repeat_targets(
        self, workflow: dict[str, Any]
    ) -> set[str] | None:
        """Targets whose thermal 07b fidelity earns the active-reset repeat.

        Returns None when no completed thermal 07b evidence exists yet, and an
        empty set when it exists but nothing cleared the trigger.
        """

        node_id = str(self.policy.raw["reset_policy"]["active_qualification_node"])
        rules = self.policy.raw["analysis"].get(node_id, {})
        trigger = rules.get("active_reset_trigger_fidelity")
        if trigger is None:
            trigger = rules.get("active_reset_min_readout_fidelity")
        if trigger is None:
            return None
        trigger = float(trigger)
        needs_morphology = bool(rules.get("active_reset_requires_morphology", True))
        seen = False
        qualifying: set[str] = set()
        cutoff = self._readout_reopen_cutoff(workflow["id"], node_id)
        for run in self.db.all(
            "SELECT parameters_json, analysis_json, started_at FROM runs "
            "WHERE workflow_id = ? AND node_id = ? AND status = 'completed' "
            "ORDER BY rowid",
            (workflow["id"], node_id),
        ):
            if cutoff is not None and str(run.get("started_at") or "") < cutoff:
                continue
            parameters = json_loads(run.get("parameters_json"), {})
            if _reset_type(parameters) == "active":
                continue
            analysis = json_loads(run.get("analysis_json"), {})
            results = analysis.get("fit_quality", {})
            results = results.get("results") if isinstance(results, dict) else None
            if not isinstance(results, dict):
                continue
            seen = True
            for name, fit in results.items():
                if not isinstance(fit, dict):
                    continue
                fidelity = fit.get("readout_fidelity")
                if not isinstance(fidelity, (int, float)) or isinstance(
                    fidelity, bool
                ):
                    continue
                if float(fidelity) > trigger and (
                    not needs_morphology or fit.get("morphology_pass") is True
                ):
                    qualifying.add(str(name))
        return qualifying if seen else None

    def _validate_multiplex_cap(self, node_id: str, requested: set[str]) -> None:
        cap = self._node_multiplex_cap(node_id)
        if cap is None or len(requested) <= cap:
            return
        raise AutonomyScopeError(
            f"{node_id} multiplexes at most {cap} targets per run; "
            f"{len(requested)} were requested. Split them into groups of "
            f"{cap} or fewer."
        )

    def _node_full_width_submission_failed(
        self,
        workflow_id: str,
        node_id: str,
        active: set[str],
    ) -> bool:
        """True once a full-width attempt at this node failed to reach hardware.

        Operator instruction 2026-09-19: a program covering every active target
        can be too large to finish compiling and transferring before the QOP
        queue-submission deadline.  Such a failure is recorded as
        ``program_submission_timeout`` once a live probe has confirmed the
        instrument is healthy, and older runs recorded it as
        ``instrument_unreachable``; both are accepted here.  The shared-first-
        batch rule would otherwise block the only remaining recovery, measuring
        the targets in smaller multiplex subgroups, so it is waived once the
        full width has demonstrably been tried and could not be submitted.
        """

        for run in self.db.all(
            "SELECT parameters_json FROM runs WHERE workflow_id = ? "
            "AND node_id = ? AND status = 'failed' "
            "AND termination_cause IN "
            "('program_submission_timeout', 'instrument_unreachable')",
            (workflow_id, node_id),
        ):
            attempted = {
                str(target)
                for target in json_loads(run.get("parameters_json"), {}).get(
                    "qubits", []
                )
            }
            if attempted == active:
                return True
        return False

    def _node_is_finished_and_fully_decided(
        self, workflow: dict[str, Any], node_id: str
    ) -> bool:
        """True when the current node is done but cannot record `advance`.

        The playbook's scientific boundary requires `manual_review` on the
        target's evidence run. When that boundary lands on the node's last run
        there is nothing left to record `advance` on, and no further run can be
        scheduled either, because every active target is resolved or
        incomplete. Without this the workflow deadlocks at a node it has
        actually finished. Requesting the next node in sequence is then the
        advance, and it is allowed only when the node really is finished and
        every one of its runs already carries a decision.
        """

        current = str(workflow["current_node"] or "")
        if not current or current not in SUBGROUP_NODES:
            return False
        sequence = self._workflow_sequence(workflow)
        if current not in sequence:
            return False
        index = sequence.index(current)
        # 2026-10-12: inside a per-qubit pass the advance target is the pass's
        # next node (e.g. 07b -> deferred 05), not the static successor.
        if index >= len(sequence) - 1 or node_id != self._expected_next_node(
            workflow, current
        ):
            return False
        if not self._node_has_completed_run(workflow["id"], current):
            return False
        if self._node_is_single_pass(current):
            # Finished once every active target has had its one pass.
            unmeasured = self._active_targets_for_node(
                workflow, current
            ) - self._single_pass_measured_targets(workflow, current)
            if unmeasured:
                return False
        else:
            active = self._active_targets_for_node(workflow, current)
            unresolved = (
                active
                - self._node_target_resolution(workflow["id"], current)
                - self._autonomy_incomplete_targets_for_node(workflow["id"], current)
            )
            if unresolved:
                return False
        undecided = self.db.one(
            "SELECT r.id FROM runs r LEFT JOIN decisions d ON d.run_id = r.id "
            "WHERE r.workflow_id = ? AND r.node_id = ? AND d.decision IS NULL "
            "LIMIT 1",
            (workflow["id"], current),
        )
        return undecided is None

    def _validate_unresolved_retry_targets(
        self,
        workflow: dict[str, Any],
        node_id: str,
        requested: set[str],
    ) -> None:
        """After the first batch, refuse already-resolved targets on retry."""
        if node_id not in SUBGROUP_NODES or node_id == "03a":
            # 03a enforces stage-specific allowed subgroups separately.
            return
        if not self._node_has_completed_run(workflow["id"], node_id):
            return
        active = self._active_targets_for_node(workflow, node_id)
        unresolved = (
            active
            - self._node_target_resolution(workflow["id"], node_id)
            - self._autonomy_incomplete_targets_for_node(workflow["id"], node_id)
        )
        if not unresolved:
            raise AutonomyScopeError(
                f"All active {node_id} targets are already resolved or incomplete; "
                "do not schedule another run for this node"
            )
        if not requested or not requested.issubset(unresolved):
            raise AutonomyScopeError(
                f"{node_id} retry targets must be a non-empty subset of unresolved "
                f"targets {sorted(unresolved)}; already-resolved targets cannot be "
                "remeasured in the same node"
            )

    def _shared_parameter_retry_warnings(
        self,
        workflow: dict[str, Any],
        node_id: str,
        requested: set[str],
    ) -> list[str]:
        if node_id not in SUBGROUP_NODES:
            return []
        if not self._node_has_completed_run(workflow["id"], node_id):
            return []
        active = self._active_targets_for_node(workflow, node_id)
        unresolved = (
            active
            - self._node_target_resolution(workflow["id"], node_id)
            - self._autonomy_incomplete_targets_for_node(workflow["id"], node_id)
        )
        if len(requested) != 1 or len(unresolved) <= 1:
            return []
        return [
            f"{node_id} still has unresolved targets {sorted(unresolved)}; "
            "retry them together with the same parameters instead of isolating "
            f"{sorted(requested)} unless they require different sweep settings."
        ]

    def _validate_03a_run_request(
        self, workflow: dict[str, Any], parameters: dict[str, Any]
    ) -> None:
        workflow_id = workflow["id"]
        active = self._active_targets_for_node(workflow, "03a")
        requested = set(parameters["qubits"])
        rows = self.db.all(
            "SELECT status, analysis_status, analysis_json FROM runs "
            "WHERE workflow_id = ? AND node_id = '03a' AND status = 'completed'",
            (workflow_id,),
        )
        rules = self.policy.raw["analysis"]["03a"]
        if not rows:
            if requested != active:
                raise AutonomyScopeError(
                    "The first 03a coarse run must multiplex every active target"
                )
            if not math.isclose(
                float(parameters["frequency_span_in_mhz"]),
                float(rules["initial_coarse_span_mhz"]),
                abs_tol=1e-9,
            ):
                raise ServiceError("The first 03a coarse run must use an 800 MHz span")
            # Operator instruction 2026-10-14: the first coarse run uses
            # exactly 1.0 when every target is on a shared XY line, otherwise
            # exactly the private-line factor 0.01 (its cap is 0.05).
            caps = self.policy.spectroscopy_amplitude_factor_caps()
            initial = float(
                rules["initial_coarse_operation_amplitude_factor"]
                if all(caps.shared(name) for name in requested)
                else rules["initial_coarse_operation_amplitude_factor_private_xy"]
            )
            if not math.isclose(
                float(parameters["operation_amplitude_factor"]), initial, abs_tol=1e-9
            ):
                raise ServiceError(
                    "The first 03a coarse run must use operation_amplitude_factor "
                    f"{initial:g}"
                )
            state = load_state(self.settings.active_state)
            nonzero = [
                name
                for name in sorted(active)
                if abs(float(state["qubits"][name]["xy"]["intermediate_frequency"]))
                > 1.0
            ]
            if nonzero:
                raise ServiceError(
                    "Apply the initial 03a zero-IF state proposal before the "
                    f"800 MHz run; nonzero IF targets: {nonzero}"
                )
            return

        if any(
            row.get("analysis_status") not in {"pass", "needs_review", "failed"}
            for row in rows
        ):
            raise ServiceError("Analyze the completed 03a run before requesting another")
        candidates = self._03a_candidate_targets(workflow_id)
        is_fine = (
            float(parameters["frequency_span_in_mhz"])
            <= float(rules["final_max_span_mhz"])
            and float(parameters["operation_amplitude_factor"])
            <= float(rules["final_max_operation_amplitude_factor"])
        )
        is_refinement = (
            float(parameters["frequency_span_in_mhz"])
            <= float(rules["refinement_max_span_mhz"])
            and float(parameters["operation_amplitude_factor"])
            <= float(rules["refinement_max_operation_amplitude_factor"])
        )
        allowed = candidates if (is_fine or is_refinement) else active - candidates
        # Operator instruction 2026-10-10: a qubit that already passed a fine
        # scan (and was committed) is not measured again at 03a.
        allowed -= self._node_target_resolution(workflow_id, "03a")
        if not requested or not requested.issubset(allowed):
            stage = "fine" if is_fine else "refinement" if is_refinement else "coarse"
            raise AutonomyScopeError(
                f"03a {stage} run targets are outside the allowed subgroup: "
                f"{sorted(allowed)}"
            )

    def _active_targets_for_node(
        self, workflow: dict[str, Any], node_id: str
    ) -> set[str]:
        targets = set(json_loads(workflow["targets_json"], []))
        sequence = list(self.settings.workflow_sequence)
        if node_id in _after_readout_line_nodes(sequence):
            _, absent_targets = self._02c_target_resolution(workflow["id"])
            targets -= absent_targets
        node_index = sequence.index(node_id)
        if "02" in sequence[:node_index]:
            # 2026-10-07: a qubit whose resonator 02 could not find (marked
            # scientifically unmeasurable there) has no readout; later nodes
            # drop it instead of sweeping around an unset IF.
            targets -= self._scientifically_unmeasurable_targets_by_node(
                workflow["id"]
            ).get("02", set())
        for upstream in sequence[:node_index]:
            if upstream not in SUBGROUP_NODES or upstream in FLUX_NODES:
                # A flux node never removes a qubit from the rest of the
                # sequence (operator rule 2026-10-08).
                continue
            if self._carries_unresolved(workflow, upstream):
                # Operator rule 2026-10-12: inside a per-qubit pass only
                # 03a/04 decide whether a qubit continues.
                continue
            advanced = self.db.one(
                "SELECT d.id FROM decisions d JOIN runs r ON r.id = d.run_id "
                "WHERE d.workflow_id = ? AND r.node_id = ? "
                "AND d.decision = 'advance' ORDER BY d.created_at DESC LIMIT 1",
                (workflow["id"], upstream),
            )
            if advanced is not None:
                carried = self._node_target_resolution(workflow["id"], upstream)
                if self._node_is_single_pass(upstream):
                    # Operator choice 2026-10-03: every target a single-pass
                    # node measured continues downstream, passing or not,
                    # unless it was explicitly marked unmeasurable there.
                    excluded = self._scientifically_unmeasurable_targets_by_node(
                        workflow["id"]
                    ).get(upstream, set())
                    carried |= (
                        self._single_pass_measured_targets(workflow, upstream)
                        - excluded
                    )
                targets &= carried
        flux_eligible = self._flux_eligible_targets(workflow["id"], node_id)
        if flux_eligible is not None:
            targets &= flux_eligible
        if self._statistics_source(node_id) is not None:
            # Operator rule 2026-10-12: statistics only for a passed source
            # and a qualified readout.
            targets &= self._statistics_qualified(workflow["id"], node_id)
        serial = self._serial_pass_state(workflow)
        if serial is not None and node_index >= sequence.index(serial["node"]):
            # One qubit holds the whole tail of the sequence until its pass is
            # over; the others are not active here yet, or are already done.
            current = serial["current"]
            targets &= {current} if current is not None else set()
        return targets

    def _initial_03a_setup_taken(self, workflow: dict[str, Any]) -> bool:
        """True once 03a has run for the qubit the setup would place.

        Zeroing the IF and moving the drive LO belongs to the start of a 03a
        coarse search. A per-qubit pass starts a new one for every qubit, so
        the gate is per pass, not per workflow.
        """

        serial = self._serial_pass_state(workflow)
        if serial is None:
            return self._node_has_completed_run(workflow["id"], "03a")
        current = serial["current"]
        if current is None:
            return True
        # After a 04 -> 03a return the qubit searches again from scratch.
        return bool(self._qubit_runs(workflow["id"], "03a", str(current)))

    def _drive_lo_recenter_kwargs(self) -> dict[str, Any]:
        """Hardware rules every LO recenter entry point has to honour."""

        shared = self._serialize_from_node() is not None
        return {
            "allow_shared_output": shared,
            "band_window": self.policy.mw_band_lo_window,
            "max_if_abs_hz": float(self.policy.limits["qubit_if_abs_hz"]),
            # Operator rule 2026-10-12: a qubit the moved LO cannot reach is
            # parked at the IF limit rather than blocking the pass.
            "park_out_of_range": shared,
        }

    # -- XY LO plan and gate time, operator instruction 2026-10-12 -----------

    def _measured_drive_qubits(self, workflow: dict[str, Any]) -> set[str]:
        """Qubits whose drive frequency is known: 03a passed, or an LO recorded."""

        measured = set(self._node_target_resolution(workflow["id"], "03a"))
        try:
            state = load_state(self.settings.active_state)
        except Exception:  # pragma: no cover - state errors surface elsewhere
            return measured
        returned = set(self._return_cutoffs(workflow["id"]))
        for name, qubit in (state.get("qubits") or {}).items():
            extras = qubit.get("extras") if isinstance(qubit, dict) else None
            if (
                isinstance(extras, dict)
                and _finite_number(extras.get("xy_lo_freq"))
                and name not in returned
            ):
                measured.add(str(name))
        # A qubit back at 03a counts again once a new fine scan resolves it.
        return measured

    def _xy_lo_plan(
        self, workflow: dict[str, Any], node_id: str, state: dict[str, Any] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """The LO/IF patch the current pass qubit needs before `node_id`."""

        serial = self._serial_pass_state(workflow)
        if serial is None or serial["current"] is None:
            return [], {}
        if node_id == serial["node"] or not self._in_serial_tail(workflow, node_id):
            return [], {}
        qubit = str(serial["current"])
        if qubit not in self._node_target_resolution(workflow["id"], serial["node"]):
            return [], {}
        state = state if state is not None else load_state(self.settings.active_state)
        wiring = load_state(self.settings.wiring_path)
        config = self.policy.shared_xy_drive
        max_if_hz = float(self.policy.limits["qubit_if_abs_hz"])
        patch, details = xy_lo_plan_patch(
            state,
            wiring,
            qubit,
            self._measured_drive_qubits(workflow),
            grid_hz=float(self.policy.raw["instrument_limits"]["drive_lo_grid_hz"]),
            max_if_hz=max_if_hz,
            common_max_if_hz=float(config.get("common_lo_max_if_abs_hz", 300e6)),
            min_if_hz=float(config.get("lo_min_if_abs_hz", 20e6)),
            band_window=self.policy.mw_band_lo_window,
        )
        # 2026-10-13: a workflow on a subset of the chip (q1/q2 statistics)
        # leaves the other qubits on the shared output alone. They are not
        # played in this pass, and an IF already within the limit stays legal
        # under any LO; only an out-of-range one would still be corrected.
        targets = {str(name) for name in json_loads(workflow.get("targets_json"), [])}
        kept = []
        for item in patch:
            match = re.fullmatch(r"/qubits/([^/]+)/(xy/intermediate_frequency|extras/xy_lo_freq)", str(item.get("path", "")))
            if match is None or match.group(1) in targets:
                kept.append(item)
                continue
            if match.group(2) == "extras/xy_lo_freq":
                continue
            current = (
                state.get("qubits", {}).get(match.group(1), {}).get("xy", {}).get(
                    "intermediate_frequency"
                )
            )
            if _finite_number(current) and abs(float(current)) <= max_if_hz:
                continue
            kept.append(item)
        return kept, details

    def request_xy_lo_plan(
        self,
        workflow_id: str,
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Place the shared drive LO for the qubit whose pass is running."""

        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active":
            raise ServiceError("The XY LO plan requires an active workflow")
        node_id = str(workflow["current_node"])
        try:
            state = load_state(self.settings.active_state)
            patch, details = self._xy_lo_plan(workflow, node_id, state)
            if patch:
                self.policy.validate_state_patch(patch, state)
        except (PolicyError, ValueError) as exc:
            if autonomy_lease_id:
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "deterministic_setup_policy_violation",
                    f"XY LO plan: {exc}",
                )
            raise ServiceError(str(exc)) from exc
        if not patch:
            return {"proposal_created": False, "xy_lo_plan": details}
        payload = {
            "patch": patch,
            "reason": (
                f"Place the shared XY LO for {details['focus']} before {node_id} "
                f"({details['mode']} LO {details['lo_hz'] / 1e9:g} GHz) and record "
                "it in extras.xy_lo_freq; unreachable qubits are parked."
            ),
            "run_id": None,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "xy_lo_plan": details,
            "authorization_evidence": (
                "deterministic_setup" if autonomy_lease_id else "human_proposal"
            ),
        }
        if autonomy_lease_id:
            lease = self._active_autonomy(autonomy_lease_id, workflow_id)
            self._validate_autonomy_scope(
                lease, node_id=node_id, targets=[str(details["focus"])]
            )
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        proposal["xy_lo_plan"] = details
        return proposal

    def _xy_gate_length_cap(self, qubit: str) -> float | None:
        limits = self.policy.raw.get("xy_gate_length_ns") or {}
        try:
            wiring = load_state(self.settings.wiring_path)
            state = load_state(self.settings.active_state)
        except Exception:  # pragma: no cover - wiring problems surface elsewhere
            return None
        reference = _xy_output_of(state, wiring, qubit)
        shared = any(
            reference in names or reference == output
            for output, names in self.policy.shared_xy_drive_groups().items()
        ) if reference else False
        value = limits.get("shared_max" if shared else "private_max")
        return float(value) if _finite_number(value) else None

    def _x180_length_ns(self, qubit: str) -> float | None:
        try:
            state = load_state(self.settings.active_state)
            item = state["qubits"][qubit]
            backing = operation_backing_name(item, "x180")
            value = item["xy"]["operations"][backing].get("length")
        except Exception:
            return None
        return float(value) if _finite_number(value) else None

    def request_04_gate_time(
        self,
        workflow_id: str,
        length_ns: int,
        client_id: str,
        autonomy_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Lengthen the pass qubit's XY gate when 04's amplitude is maxed out.

        Operator instruction 2026-10-12: if the Rabi signal is changing but has
        not turned over at the largest amplitude 04 may sweep, the gate gets
        longer (at most 400 ns on a shared XY line, 100 ns on a private one).
        """

        if not client_id.strip():
            raise ServiceError("client_id is required for the audit log")
        workflow = self._workflow(workflow_id)
        if workflow["status"] != "active" or workflow["current_node"] != "04":
            raise ServiceError("The 04 gate-time setup requires an active 04 workflow")
        active = sorted(self._active_targets_for_node(workflow, "04"))
        if len(active) != 1:
            raise ServiceError(
                "The 04 gate-time setup changes one qubit at a time; "
                f"active 04 targets are {active}"
            )
        qubit = active[0]
        limits = self.policy.raw.get("xy_gate_length_ns") or {}
        cap = self._xy_gate_length_cap(qubit)
        step = int(limits.get("step", 4))
        minimum = int(limits.get("min", 16))
        current = self._x180_length_ns(qubit)
        problems: list[str] = []
        if isinstance(length_ns, bool) or not isinstance(length_ns, int):
            problems.append("length_ns must be an integer")
        elif cap is None or current is None:
            problems.append(f"{qubit} has no x180 length or gate-length limit")
        elif length_ns % step or length_ns < minimum:
            problems.append(f"length_ns must be a multiple of {step} ns and at least {minimum} ns")
        elif length_ns > cap:
            problems.append(f"{length_ns} ns is above the {cap:g} ns XY gate limit")
        elif length_ns <= current:
            problems.append(f"{length_ns} ns is not longer than the current {current:g} ns")
        state = load_state(self.settings.active_state)
        item = state["qubits"][qubit]
        rows = self._qubit_runs(workflow_id, "04", qubit)
        if not problems:
            if not rows:
                problems.append(f"{qubit} has no 04 run yet; sweep the amplitude first")
            else:
                latest = rows[-1]
                parameters = json_loads(latest.get("parameters_json"), {})
                analysis = json_loads(latest.get("analysis_json"), {})
                metrics = (
                    analysis.get("dataset_metrics", {}).get("qubits", {}).get(qubit, {})
                )
                recorded = (
                    metrics.get("operation_amplitude_before_run")
                    if isinstance(metrics, dict)
                    else None
                )
                baseline = (
                    abs(float(recorded))
                    if _finite_number(recorded)
                    else abs(operation_amplitude(item, "x180"))
                )
                swept = baseline * float(parameters.get("max_amp_factor", 1.9))
                maximum = float(
                    self.policy.limits[channel_kind(item)]["max_wf_amplitude"]
                )
                if swept < 0.9 * maximum:
                    problems.append(
                        f"{qubit}'s last 04 swept only to {swept:g} of {maximum:g}; "
                        "raise max_amp_factor before lengthening the gate"
                    )
        if problems:
            if autonomy_lease_id:
                self._record_refused_autonomy_action(
                    autonomy_lease_id,
                    workflow_id,
                    "deterministic_setup_policy_violation",
                    f"04 gate time: {'; '.join(problems)}",
                )
            raise ServiceError("; ".join(problems))
        operations = item["xy"]["operations"]
        patch = []
        for name, operation in operations.items():
            if not isinstance(operation, dict):
                continue
            length = operation.get("length")
            if (
                _finite_number(length)
                and current is not None
                and math.isclose(float(length), current)
                and not str(name).lower().startswith(("ef_", "saturation"))
            ) or name == operation_backing_name(item, "x180"):
                patch.append(
                    {
                        "op": "replace",
                        "path": f"/qubits/{qubit}/xy/operations/{name}/length",
                        "value": int(length_ns),
                    }
                )
        last = self.db.one(
            "SELECT MAX(rowid) AS rowid FROM runs WHERE workflow_id = ?", (workflow_id,)
        )
        details = {
            "qubit": qubit,
            "old_length_ns": current,
            "new_length_ns": int(length_ns),
            "max_length_ns": cap,
            "after_run_rowid": int((last or {}).get("rowid") or 0),
        }
        self.policy.validate_state_patch(patch, state)
        payload = {
            "patch": patch,
            "reason": (
                f"{qubit}'s 04 amplitude is at its limit without a Rabi turnover; "
                f"lengthen its XY gate {current:g} -> {length_ns} ns."
            ),
            "run_id": None,
            "base_hash": sha256_file(self.settings.active_state),
            "active_state": str(self.settings.active_state),
            "xy_gate_length": details,
            "authorization_evidence": (
                "deterministic_setup" if autonomy_lease_id else "human_proposal"
            ),
        }
        if autonomy_lease_id:
            lease = self._active_autonomy(autonomy_lease_id, workflow_id)
            self._validate_autonomy_scope(lease, node_id="04", targets=[qubit])
        proposal = self._create_proposal(
            workflow_id,
            "state_commit",
            payload,
            client_id,
            int(self.policy.raw["approval"]["state_commit_ttl_minutes"]),
            1,
            autonomy_lease_id=autonomy_lease_id,
        )
        if autonomy_lease_id:
            proposal = self._delegate_proposal_to_autonomy(
                proposal["id"], autonomy_lease_id, client_id
            )
        self.db.event("xy_gate_length_set", client_id, details, workflow_id)
        proposal["xy_gate_length"] = details
        return proposal

    def _serialize_from_node(
        self, workflow: dict[str, Any] | None = None
    ) -> str | None:
        """The node a per-qubit pass starts at, when this workflow needs one.

        Operator instruction 2026-09-22. A chip that drives two or more qubits
        from one OPX output also shares their LO, so the workflow cannot sweep
        a node across all of them. It takes one qubit through the whole tail of
        the sequence instead. `None` means every qubit has its own XY output
        and the ordinary node-by-node sweep applies.

        Operator request 2026-10-04: the measurement mode chosen on the lease
        approval page overrides the wiring default -- `per_qubit` serializes
        even when every qubit has its own output, `multiplex` never does.
        """

        try:
            config = self.policy.shared_xy_drive
            first = config.get("serialize_from_node")
            if not config.get("per_qubit_pass") or not isinstance(first, str):
                return None
            if first not in self.settings.workflow_sequence:
                return None
            # Operator instruction 2026-10-08: a shared XY output always runs
            # one qubit at a time, whatever mode the workflow stored.
            if self.policy.shared_xy_drive_groups():
                return first
            return first if self._drive_mode(workflow) == "per_qubit" else None
        except Exception:  # pragma: no cover - wiring problems surface elsewhere
            return None

    # Operator request 2026-10-04: the autonomy-lease approval page lets the
    # operator tick the optional nodes (`workflow_options.optional_nodes`) and
    # choose per-qubit or multiplexed measurement from the serialization node
    # on. The choices are stored on the workflow, so every later lease of that
    # workflow follows them.

    @staticmethod
    def _workflow_options(workflow: dict[str, Any] | None) -> dict[str, Any]:
        """Lease-page choices of a raw workflow row or a decoded workflow."""

        workflow = workflow or {}
        if isinstance(workflow.get("options"), dict):
            return workflow["options"]
        options = json_loads(workflow.get("options_json"), None)
        return options if isinstance(options, dict) else {}

    def _open_workflow_options(self) -> dict[str, Any]:
        """Options of the active or paused workflow (there is at most one)."""

        row = self.db.one(
            "SELECT options_json FROM workflows WHERE status IN ('active', 'paused') "
            "ORDER BY updated_at DESC LIMIT 1"
        )
        return self._workflow_options(row)

    def _resolved_options(self, workflow: dict[str, Any] | None) -> dict[str, Any]:
        if workflow is not None:
            return self._workflow_options(workflow)
        return self._open_workflow_options()

    def _drive_mode(self, workflow: dict[str, Any] | None = None) -> str | None:
        mode = self._resolved_options(workflow).get("drive_mode")
        return mode if mode in DRIVE_MODES else None

    def _workflow_sequence(
        self,
        workflow: dict[str, Any] | None = None,
        *,
        include_undecided_flux: bool = False,
    ) -> list[str]:
        """The configured sequence without the optional nodes left unticked.

        Operator request 2026-10-08: the flux nodes run only after a lease-page
        yes. Before that choice they are left out, except while a new lease is
        being offered (`include_undecided_flux`), so the page can still add them.
        """

        options = self._resolved_options(workflow)
        excluded = set(options.get("excluded_nodes", []))
        flux = options.get("flux_calibration")
        if flux is not True and not (include_undecided_flux and flux is None):
            excluded |= set(self._flux_nodes())
        return [node for node in self.settings.workflow_sequence if node not in excluded]

    def _flux_nodes(self) -> list[str]:
        """Every flux-calibration node of the configured sequence."""

        nodes, optional = self.policy.flux_calibration_nodes()
        configured = set(nodes) | set(optional)
        return [node for node in self.settings.workflow_sequence if node in configured]

    def _flux_tunable_targets(self, workflow_id: str) -> set[str]:
        rows = self.db.all(
            "SELECT status, analysis_status, analysis_json FROM runs "
            "WHERE workflow_id = ? AND node_id = '02b' ORDER BY rowid",
            (workflow_id,),
        )
        return resolve_02b_tunable_targets(rows)

    def _flux_eligible_targets(self, workflow_id: str, node_id: str) -> set[str] | None:
        """Which qubits a flux node may measure, or None for other nodes.

        Operator rule 2026-10-08: 03b takes the qubits 02b found tunable (flat
        and avoided-crossing qubits skip it); 06a takes the ones 03b resolved.
        """

        if node_id == "03b":
            return self._flux_tunable_targets(workflow_id)
        if node_id == "06a":
            return self._flux_tunable_targets(workflow_id) & self._node_target_resolution(
                workflow_id, "03b"
            )
        return None

    def _optional_nodes(self) -> list[str]:
        configured = {
            str(node)
            for node in self.policy.raw.get("workflow_options", {}).get(
                "optional_nodes", []
            )
        }
        return [node for node in self.settings.workflow_sequence if node in configured]

    def _broadband_required_targets(self, workflow: dict[str, Any]) -> list[str]:
        """Targets whose readout IF is zero or duplicated, forcing 02.

        Operator rule 2026-10-07: such a state has no resonator assignment
        yet, so 02 (with the design frequencies typed on the lease page) must
        run. Once 02 commits distinct IFs this list is empty again.
        """
        if "02" not in self.settings.workflow_sequence:
            return []
        targets = json_loads(workflow.get("targets_json"), None)
        if not isinstance(targets, list):
            targets = workflow.get("targets") or []
        try:
            return self.policy.readout_frequencies_unset(sorted(targets))
        except Exception:  # pragma: no cover - wiring problems surface elsewhere
            return []

    def _lease_option_offer(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """What the lease approval page asks the operator to choose."""

        sequence = list(self.settings.workflow_sequence)
        current = str(workflow["current_node"])
        remaining = sequence[sequence.index(current):] if current in sequence else sequence
        options = self._workflow_options(workflow)
        excluded = set(options.get("excluded_nodes", []))
        stored_parameters = dict(options.get("node_parameters") or {})
        forced = self._broadband_required_targets(workflow)
        history = self._chip_history(workflow)
        if (
            history is not None
            and "02" not in stored_parameters
            and isinstance(history.get("previous_02_parameters"), dict)
        ):
            # Operator choice 2026-10-11: starting from the beginning prefills
            # the chip's last res_num / res_design_freq.
            stored_parameters["02"] = history["previous_02_parameters"]
        optional = [
            {
                "node": node,
                "default": node not in excluded or (node == "02" and bool(forced)),
                **(
                    {"required": True, "required_because": forced}
                    if node == "02" and forced
                    else {}
                ),
                # Operator request 2026-10-07: prefill the operator-supplied
                # node parameters (02: res_num, res_design_freq).
                **(
                    {"parameters": stored_parameters[node]}
                    if isinstance(stored_parameters.get(node), dict)
                    else {}
                ),
            }
            for node in self._optional_nodes()
            if node in remaining
        ]
        config = self.policy.shared_xy_drive
        first = config.get("serialize_from_node")
        offer: dict[str, Any] = {"optional_nodes": optional, "drive_mode": None}
        if config.get("per_qubit_pass") and isinstance(first, str) and first in sequence:
            try:
                groups = self.policy.shared_xy_drive_groups()
            except Exception:  # pragma: no cover - wiring problems surface elsewhere
                groups = {}
            locked = self._drive_mode(workflow)
            offer["drive_mode"] = {
                "choices": list(DRIVE_MODES),
                "recommended": "per_qubit" if groups else "multiplex",
                # Operator instruction 2026-10-08: a shared XY output cannot
                # be multiplexed at all.
                "blocked": ["multiplex"] if groups else [],
                "shared_xy_drive_groups": groups,
                "serialize_from_node": first,
                # A workflow keeps the mode it started with; a later lease only
                # confirms it.
                "locked": "per_qubit" if groups and locked is not None else locked,
            }
        flux_nodes, flux_optional = self.policy.flux_calibration_nodes()
        if any(node in remaining for node in flux_nodes):
            try:
                shared_z = self.policy.shared_z_line_groups()
            except Exception:  # pragma: no cover - wiring problems surface elsewhere
                shared_z = {}
            stored = options.get("flux_calibration")
            offer["flux_calibration"] = {
                "nodes": [node for node in flux_nodes if node in remaining],
                "optional": [node for node in flux_optional if node in remaining],
                "shared_z_groups": shared_z,
                # Operator request 2026-10-08: any shared Z output disables yes.
                "blocked": bool(shared_z),
                "locked": stored if isinstance(stored, bool) else None,
            }
            for item in optional:
                if item["node"] in flux_optional and (shared_z or stored is False):
                    item["default"] = False
        if history is not None:
            offer["start_node"] = history
        return offer

    def _chip_history(self, workflow: dict[str, Any]) -> dict[str, Any] | None:
        """The latest earlier workflow on this chip and where it can resume.

        Operator request 2026-10-10: a new workflow may start from the
        beginning or at any node the chip's latest workflow reached, keeping
        the evidence before that node. Offered only to a workflow that has no
        run of its own yet.
        """

        if workflow.get("source_workflow_id"):
            return None
        if self.db.one(
            "SELECT 1 FROM runs WHERE workflow_id = ? LIMIT 1", (workflow["id"],)
        ):
            return None
        device = workflow.get("device_key") or self._device_key()
        source = self.db.one(
            "SELECT * FROM workflows WHERE device_key = ? AND id != ? "
            "AND status NOT IN ('active', 'paused') "
            "AND EXISTS (SELECT 1 FROM runs WHERE runs.workflow_id = workflows.id) "
            "ORDER BY created_at DESC LIMIT 1",
            (device, workflow["id"]),
        )
        if source is None:
            return None
        sequence = self._workflow_sequence(source)
        current = str(source.get("current_node") or "")
        if source["status"] == "completed" or current not in sequence:
            reached = len(sequence) - 1
        else:
            reached = sequence.index(current)
        options = self._workflow_options(source)
        nodes = []
        for node in sequence[: reached + 1]:
            ran = self.db.one(
                "SELECT 1 FROM runs WHERE workflow_id = ? AND node_id = ? LIMIT 1",
                (source["id"], node),
            )
            advanced = self.db.one(
                "SELECT 1 FROM decisions d JOIN runs r ON r.id = d.run_id "
                "WHERE r.workflow_id = ? AND r.node_id = ? AND d.decision = 'advance' "
                "LIMIT 1",
                (source["id"], node),
            )
            nodes.append(
                {
                    "node": node,
                    "ran": ran is not None,
                    "finished": advanced is not None,
                    "passed": sorted(
                        self._node_target_resolution(source["id"], node),
                        key=_natural_qubit_key,
                    )
                    if ran is not None
                    else [],
                }
            )
        # Starting at the configured first node is "from the beginning"; a
        # source that itself skipped 02 still offers 02x (2026-10-11).
        first = self.settings.workflow_sequence[0]
        start_nodes = [node for node in sequence[: reached + 1] if node != first]
        serial = self.policy.shared_xy_drive.get("serialize_from_node")
        if options.get("drive_mode") == "per_qubit" and serial in sequence:
            # A per-qubit pass cannot be entered half-way: start at or before it.
            start_nodes = [
                node
                for node in start_nodes
                if sequence.index(node) <= sequence.index(serial)
            ]
        # Operator choice 2026-10-11: a later start skips 02, which is only
        # safe when every target whose readout IF is unset was already found
        # absent at 02 there (no resonator; dropped from later nodes).
        absent_at_02 = self._scientifically_unmeasurable_targets_by_node(
            source["id"]
        ).get("02", set())
        unset = set(self._broadband_required_targets(workflow))
        start_blocked_by = sorted(unset - absent_at_02, key=_natural_qubit_key)
        if start_blocked_by:
            start_nodes = []
        previous_02 = (options.get("node_parameters") or {}).get("02")
        final_hash = self._workflow_final_state_hash(source["id"])
        qubit_start_nodes = (
            self._qubit_start_offer(source, sequence, str(serial), workflow)
            if options.get("drive_mode") == "per_qubit"
            and isinstance(serial, str)
            and serial in sequence
            and serial in start_nodes
            else {}
        )
        return {
            "qubit_start_nodes": qubit_start_nodes,
            "serialize_from_node": serial if qubit_start_nodes else None,
            "start_blocked_by": start_blocked_by,
            "absent_at_02": sorted(absent_at_02, key=_natural_qubit_key),
            "previous_02_parameters": previous_02 if isinstance(previous_02, dict) else None,
            "source_workflow_id": source["id"],
            "created_at": source["created_at"],
            "status": source["status"],
            "device_key": device,
            "reached_node": sequence[reached],
            "nodes": nodes,
            "start_nodes": start_nodes,
            "flux_calibration": options.get("flux_calibration") is True,
            "drive_mode": options.get("drive_mode"),
            "state_matches": (
                final_hash is None
                or final_hash == sha256_file(self.settings.active_state)
            ),
        }

    def _qubit_start_offer(
        self,
        source: dict[str, Any],
        sequence: list[str],
        serial: str,
        workflow: dict[str, Any],
    ) -> dict[str, list[str]]:
        """Per qubit, the pass nodes it may start at (operator 2026-10-12).

        A qubit may restart its pass at the serialization node or at any later
        node of its own pass in the source workflow, up to the node it reached
        there; a qubit that never entered its pass starts at the
        serialization node.
        """

        tail = sequence[sequence.index(serial):]
        _, absent = self._02c_target_resolution(source["id"])
        absent |= self._scientifically_unmeasurable_targets_by_node(source["id"]).get(
            "02", set()
        )
        offer: dict[str, list[str]] = {}
        targets = [
            str(name)
            for name in json_loads(workflow.get("targets_json"), [])
            if str(name) not in absent
        ]
        for name in sorted(targets, key=_natural_qubit_key):
            reached = 0
            for index, node in enumerate(tail):
                if self.db.one(
                    "SELECT 1 FROM runs WHERE workflow_id = ? AND node_id = ? "
                    "AND status = 'completed' AND parameters_json LIKE ? LIMIT 1",
                    (source["id"], node, f'%"{name}"%'),
                ):
                    reached = index
            nodes = [
                node
                for node in tail[: reached + 1]
                if node not in FLUX_NODES
            ] or [serial]
            # Operator choice 2026-10-13: a statistics node beyond the reached
            # node is offered when its single measurement passed there (q2
            # never reached 05st, but its 05 and 06b passed).
            for node in tail[reached + 1:]:
                source_node = self._statistics_source(node)
                if source_node is not None and name in self._node_target_resolution(
                    source["id"], source_node
                ):
                    nodes.append(node)
            offer[name] = nodes
        return offer

    def _workflow_final_state_hash(self, workflow_id: str) -> str | None:
        row = self.db.one(
            "SELECT payload_json FROM events WHERE workflow_id = ? "
            "AND event_type = 'state_committed' ORDER BY id DESC LIMIT 1",
            (workflow_id,),
        )
        value = json_loads(row["payload_json"], {}).get("after_hash") if row else None
        if isinstance(value, str) and value:
            return value
        run = self.db.one(
            "SELECT active_state_hash_after FROM runs WHERE workflow_id = ? "
            "AND active_state_hash_after IS NOT NULL ORDER BY rowid DESC LIMIT 1",
            (workflow_id,),
        )
        return str(run["active_state_hash_after"]) if run else None

    def _start_node_choice(
        self,
        offer: dict[str, Any],
        lease_options: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Validate the lease-page start node; return (history, node)."""

        history = offer.get("start_node")
        node = lease_options.get("start_node")
        if node in (None, "", "beginning"):
            return None, None
        if isinstance(history, dict) and history.get("start_blocked_by"):
            raise ServiceError(
                f"{history['start_blocked_by']} have no readout IF and were not found "
                "absent at 02 last time, so this measurement must start from the "
                "beginning with 02."
            )
        if not isinstance(history, dict) or node not in history.get("start_nodes", []):
            raise ServiceError(
                f"{node} is not a node this chip's last workflow reached; "
                "choose another start node or start from the beginning."
            )
        return history, str(node)

    def _qubit_start_choice(
        self,
        history: dict[str, Any] | None,
        start_node: str | None,
        lease_options: dict[str, Any],
    ) -> dict[str, str]:
        """Validate the per-qubit start points; keep only the later ones."""

        chosen = lease_options.get("qubit_start_nodes") or {}
        if not isinstance(chosen, dict):
            raise ServiceError("qubit_start_nodes must be an object")
        if not chosen:
            return {}
        serial = (history or {}).get("serialize_from_node")
        offered = (history or {}).get("qubit_start_nodes") or {}
        if history is None or start_node is None or start_node != serial:
            raise ServiceError(
                "Per-qubit start points apply only when the measurement starts at "
                f"{serial or 'the per-qubit pass'}."
            )
        result: dict[str, str] = {}
        for qubit, node in chosen.items():
            if str(node) not in (offered.get(str(qubit)) or []):
                raise ServiceError(
                    f"{qubit} cannot start at {node}; choose one of "
                    f"{offered.get(str(qubit)) or [serial]}."
                )
            if str(node) != serial:
                result[str(qubit)] = str(node)
        return result

    def _inherit_qubit_tail(
        self,
        connection: sqlite3.Connection,
        workflow_id: str,
        source_workflow_id: str,
        qubit_starts: dict[str, str],
    ) -> int:
        """Copy each qubit's own pass runs before its chosen start node."""

        full = list(self.settings.workflow_sequence)
        copied = 0
        for qubit, start in qubit_starts.items():
            serial = self.policy.shared_xy_drive.get("serialize_from_node")
            earlier = full[full.index(str(serial)): full.index(start)]
            if not earlier:
                continue
            placeholders = ",".join("?" for _ in earlier)
            rows = connection.execute(
                f"SELECT * FROM runs WHERE workflow_id = ? AND node_id IN ({placeholders}) "
                "AND status = 'completed' ORDER BY rowid",
                (source_workflow_id, *earlier),
            ).fetchall()
            for row in rows:
                names = json_loads(row["parameters_json"], {}).get("qubits", [])
                if [str(name) for name in names] != [qubit]:
                    continue
                run = dict(row)
                new_id = uuid.uuid4().hex
                run["inherited_from_run_id"] = run.get("inherited_from_run_id") or run["id"]
                run["id"] = new_id
                run["workflow_id"] = workflow_id
                run["autonomy_lease_id"] = None
                run["pid"] = None
                run["process_token"] = None
                columns = list(run)
                connection.execute(
                    f"INSERT INTO runs({', '.join(columns)}) "
                    f"VALUES ({', '.join('?' for _ in columns)})",
                    tuple(run[column] for column in columns),
                )
                decision = connection.execute(
                    "SELECT * FROM decisions WHERE run_id = ?", (row["id"],)
                ).fetchone()
                if decision is not None:
                    copy = dict(decision)
                    copy["id"] = uuid.uuid4().hex
                    copy["workflow_id"] = workflow_id
                    copy["run_id"] = new_id
                    keys = list(copy)
                    connection.execute(
                        f"INSERT INTO decisions({', '.join(keys)}) "
                        f"VALUES ({', '.join('?' for _ in keys)})",
                        tuple(copy[key] for key in keys),
                    )
                copied += 1
            for event in connection.execute(
                "SELECT payload_json FROM events WHERE workflow_id = ? "
                "AND event_type = 'autonomy_targets_scientifically_unmeasurable' "
                "ORDER BY id",
                (source_workflow_id,),
            ).fetchall():
                payload = json_loads(event["payload_json"], {})
                if str(payload.get("node_id", "")) not in earlier:
                    continue
                if qubit not in {str(name) for name in payload.get("targets", [])}:
                    continue
                self.db.event(
                    "autonomy_targets_scientifically_unmeasurable",
                    "inherited",
                    {
                        **payload,
                        "targets": [qubit],
                        "inherited_from_workflow_id": source_workflow_id,
                    },
                    workflow_id,
                    connection=connection,
                )
        return copied

    def _start_locks(
        self, history: dict[str, Any], start_node: str
    ) -> tuple[bool | None, str | None]:
        """Flux and drive-mode choices a mid-sequence start must keep.

        Operator rule 2026-10-10: starting after 02b keeps the earlier
        workflow's flux choice (no 03b without its 02b), and starting after
        the serialization node keeps its measurement mode.
        """

        full = list(self.settings.workflow_sequence)
        flux_nodes = [node for node in self._flux_nodes() if node in full]
        flux_lock = None
        if flux_nodes and full.index(start_node) > full.index(flux_nodes[0]):
            flux_lock = bool(history.get("flux_calibration"))
        serial = self.policy.shared_xy_drive.get("serialize_from_node")
        drive_lock = None
        if (
            isinstance(serial, str)
            and serial in full
            and full.index(start_node) > full.index(serial)
            and history.get("drive_mode") in DRIVE_MODES
        ):
            drive_lock = str(history["drive_mode"])
        return flux_lock, drive_lock

    def _inherit_evidence(
        self,
        connection: sqlite3.Connection,
        workflow_id: str,
        source_workflow_id: str,
        start_node: str,
    ) -> int:
        """Copy the source workflow's runs and decisions before `start_node`.

        The copies carry `inherited_from_run_id`, no lease and the original
        analysis, so every per-workflow guard sees the earlier evidence while
        lease attempts and the session results stay with the new runs.
        """

        full = list(self.settings.workflow_sequence)
        earlier = full[: full.index(start_node)]
        if not earlier:
            return 0
        placeholders = ",".join("?" for _ in earlier)
        rows = connection.execute(
            f"SELECT * FROM runs WHERE workflow_id = ? AND node_id IN ({placeholders}) "
            "ORDER BY rowid",
            (source_workflow_id, *earlier),
        ).fetchall()
        copied = 0
        for row in rows:
            run = dict(row)
            if run.get("status") != "completed":
                continue
            new_id = uuid.uuid4().hex
            run["inherited_from_run_id"] = run.get("inherited_from_run_id") or run["id"]
            run["id"] = new_id
            run["workflow_id"] = workflow_id
            run["autonomy_lease_id"] = None
            run["pid"] = None
            run["process_token"] = None
            columns = list(run)
            connection.execute(
                f"INSERT INTO runs({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                tuple(run[column] for column in columns),
            )
            decision = connection.execute(
                "SELECT * FROM decisions WHERE run_id = ?", (row["id"],)
            ).fetchone()
            if decision is not None:
                copy = dict(decision)
                copy["id"] = uuid.uuid4().hex
                copy["workflow_id"] = workflow_id
                copy["run_id"] = new_id
                keys = list(copy)
                connection.execute(
                    f"INSERT INTO decisions({', '.join(keys)}) "
                    f"VALUES ({', '.join('?' for _ in keys)})",
                    tuple(copy[key] for key in keys),
                )
            copied += 1
        self._inherit_unmeasurable_marks(
            connection, workflow_id, source_workflow_id, earlier
        )
        return copied

    def _inherit_unmeasurable_marks(
        self,
        connection: sqlite3.Connection,
        workflow_id: str,
        source_workflow_id: str,
        earlier: list[str],
    ) -> int:
        """Carry the source's scientifically-unmeasurable marks before the start.

        2026-10-11: a workflow started at 02x kept 02's runs but not 02's mark
        on q7 (no resonator), so q7 with IF 0 came back as a 02x target.
        """
        rows = connection.execute(
            "SELECT payload_json FROM events WHERE workflow_id = ? "
            "AND event_type = 'autonomy_targets_scientifically_unmeasurable' "
            "ORDER BY id",
            (source_workflow_id,),
        ).fetchall()
        carried = 0
        for row in rows:
            payload = json_loads(row["payload_json"], {})
            if str(payload.get("node_id", "")) not in earlier:
                continue
            self.db.event(
                "autonomy_targets_scientifically_unmeasurable",
                "inherited",
                {**payload, "inherited_from_workflow_id": source_workflow_id},
                workflow_id,
                connection=connection,
            )
            carried += 1
        return carried

    def _backfill_inherited_marks(self) -> None:
        """Give open mid-sequence workflows the marks their start skipped."""
        for workflow in self.db.all(
            "SELECT id, source_workflow_id, start_node FROM workflows "
            "WHERE source_workflow_id IS NOT NULL AND start_node IS NOT NULL "
            "AND status IN ('active', 'paused')"
        ):
            if self.db.one(
                "SELECT 1 FROM events WHERE workflow_id = ? AND event_type = "
                "'autonomy_targets_scientifically_unmeasurable' AND actor = 'inherited' "
                "LIMIT 1",
                (workflow["id"],),
            ):
                continue
            full = list(self.settings.workflow_sequence)
            if workflow["start_node"] not in full:
                continue
            earlier = full[: full.index(workflow["start_node"])]
            with self.db.transaction(immediate=True) as connection:
                self._inherit_unmeasurable_marks(
                    connection,
                    workflow["id"],
                    workflow["source_workflow_id"],
                    earlier,
                )

    def _apply_lease_options(
        self,
        connection: sqlite3.Connection,
        proposal: dict[str, Any],
        lease_options: dict[str, Any] | None,
        approval_method: str,
        actor: str,
    ) -> None:
        """Store the operator's lease-page choices before the lease activates."""

        payload = json_loads(proposal.get("payload_json"), {})
        offer = payload.get("workflow_options_offer")
        if not isinstance(offer, dict):
            return
        offered = [
            str(item["node"])
            for item in offer.get("optional_nodes", [])
            if isinstance(item, dict) and "node" in item
        ]
        drive_offer = offer.get("drive_mode")
        flux_offer = offer.get("flux_calibration")
        if lease_options is None:
            # The Dashboard always submits the choices (approve_from_browser
            # refuses a lease without them). A terminal approval cannot show
            # them, so it takes the defaults listed in the proposal payload.
            lease_options = {
                "optional_nodes": [
                    str(item["node"])
                    for item in offer.get("optional_nodes", [])
                    if isinstance(item, dict) and item.get("default", True)
                ],
                "drive_mode": (
                    (drive_offer.get("locked") or drive_offer.get("recommended"))
                    if isinstance(drive_offer, dict)
                    else None
                ),
                # A terminal approval cannot ask, so flux calibration stays off
                # unless the workflow already chose it.
                "flux_calibration": (
                    bool(flux_offer.get("locked"))
                    if isinstance(flux_offer, dict)
                    else None
                ),
            }
            if lease_options["flux_calibration"] is False:
                lease_options["optional_nodes"] = [
                    node
                    for node in lease_options["optional_nodes"]
                    if node not in set(flux_offer.get("optional", []))
                ]
        selected = [str(node) for node in lease_options.get("optional_nodes", [])]
        unknown = sorted(set(selected) - set(offered))
        if unknown:
            raise ServiceError(f"These experiments are not optional here: {unknown}")
        history, start_node = self._start_node_choice(offer, lease_options)
        qubit_starts = self._qubit_start_choice(history, start_node, lease_options)
        flux_history_lock, drive_history_lock = (
            self._start_locks(history, start_node)
            if history is not None and start_node is not None
            else (None, None)
        )
        full_sequence = list(self.settings.workflow_sequence)
        skipped_before_start: set[str] = set()
        if start_node is not None:
            # Operator choice 2026-10-11: nodes before the start do not run, so
            # their ticks and lease-page parameters (02's) are ignored.
            skipped_before_start = {
                node
                for node in offered
                if node in full_sequence
                and full_sequence.index(node) < full_sequence.index(start_node)
            }
            selected = [node for node in selected if node not in skipped_before_start]
            lease_options = {
                **lease_options,
                "node_parameters": {
                    node: values
                    for node, values in (lease_options.get("node_parameters") or {}).items()
                    if node not in skipped_before_start
                },
            }
        for item in offer.get("optional_nodes", []):
            if (
                start_node is not None
                and str(item.get("node")) in full_sequence
                and full_sequence.index(str(item.get("node")))
                < full_sequence.index(start_node)
            ):
                # Nodes before the start node do not run in this workflow.
                continue
            if isinstance(item, dict) and item.get("required") and item.get("node") not in selected:
                raise ServiceError(
                    f"{item.get('node')} is required: readout IFs of "
                    f"{item.get('required_because')} are zero or duplicated. Fill in "
                    "the resonator count and design frequencies."
                )
        drive_mode = lease_options.get("drive_mode")
        if drive_history_lock is not None:
            if drive_mode not in (None, drive_history_lock):
                raise ServiceError(
                    f"Starting at {start_node} keeps the earlier workflow's "
                    f"{drive_history_lock!r} measurement mode; start earlier to change it."
                )
            drive_mode = drive_history_lock
        if isinstance(drive_offer, dict):
            locked = drive_offer.get("locked")
            if locked is not None:
                if drive_mode not in (None, locked):
                    raise ServiceError(
                        f"This workflow already measures in {locked!r} mode; "
                        "start a new workflow to change it."
                    )
                drive_mode = locked
            elif drive_mode not in DRIVE_MODES:
                raise ServiceError(
                    "Choose whether the qubits are measured one at a time or "
                    "multiplexed before approving automatic measurement."
                )
            if drive_mode in (drive_offer.get("blocked") or []):
                raise ServiceError(
                    "This chip shares an XY output, so multiplexed measurement "
                    "is disabled; measure the qubits one at a time."
                )
        else:
            drive_mode = None
        flux_choice = lease_options.get("flux_calibration")
        if flux_history_lock is not None:
            if flux_choice not in (None, flux_history_lock):
                raise ServiceError(
                    f"Starting at {start_node} keeps the earlier workflow's flux "
                    f"choice ({'yes' if flux_history_lock else 'no'}); start at or "
                    "before 02b to change it."
                )
            flux_choice = flux_history_lock
        if isinstance(flux_offer, dict):
            flux_locked = flux_offer.get("locked")
            if flux_locked is not None:
                if flux_choice not in (None, flux_locked):
                    raise ServiceError(
                        "This workflow already chose whether to calibrate flux; "
                        "start a new workflow to change it."
                    )
                flux_choice = flux_locked
            elif flux_choice not in (True, False):
                raise ServiceError(
                    "Choose whether to calibrate flux before approving automatic "
                    "measurement."
                )
            if flux_choice and flux_offer.get("blocked"):
                groups = flux_offer.get("shared_z_groups") or {}
                raise ServiceError(
                    "Flux calibration is disabled because the wiring shares Z "
                    "lines: "
                    + "; ".join(
                        f"{output} -> {', '.join(names)}"
                        for output, names in sorted(groups.items())
                    )
                )
            needs_flux = sorted(set(flux_offer.get("optional", [])) & set(selected))
            if needs_flux and not flux_choice:
                raise ServiceError(
                    f"{needs_flux} need flux calibration; choose yes or untick them."
                )
        else:
            flux_choice = None

        workflow_row = connection.execute(
            "SELECT * FROM workflows WHERE id = ?", (proposal["workflow_id"],)
        ).fetchone()
        if workflow_row is None:
            raise ServiceError(f"Unknown workflow: {proposal['workflow_id']}")
        workflow = dict(workflow_row)
        options = self._workflow_options(workflow)
        excluded = (set(options.get("excluded_nodes", [])) - set(offered)) | (
            set(offered) - set(selected)
        )
        if flux_choice is True:
            excluded -= set(flux_offer.get("nodes", []))
        elif flux_choice is False or options.get("flux_calibration") is not True:
            excluded |= set(self._flux_nodes())
        sequence = [
            node for node in self.settings.workflow_sequence if node not in excluded
        ]
        current = start_node or str(workflow["current_node"])
        if current in excluded:
            started = connection.execute(
                "SELECT 1 FROM runs WHERE workflow_id = ? AND node_id = ? LIMIT 1",
                (workflow["id"], current),
            ).fetchone()
            if started is not None:
                raise ServiceError(
                    f"{current} has already started in this workflow and cannot be skipped."
                )
            full = list(self.settings.workflow_sequence)
            following = [
                node for node in full[full.index(current) + 1:] if node not in excluded
            ]
            if not following:
                raise ServiceError("At least one experiment must remain selected.")
            current = following[0]
        # Operator request 2026-10-07: parameters the operator typed on the
        # lease page for a ticked optional node (02: res_num, res_design_freq).
        # A terminal approval sends none and keeps what the workflow stored.
        node_parameters = {
            node: values
            for node, values in (options.get("node_parameters") or {}).items()
            if node not in excluded
        }
        supplied = lease_options.get("node_parameters") or {}
        if not isinstance(supplied, dict):
            raise ServiceError("node_parameters must be an object")
        for node, values in supplied.items():
            node = str(node)
            if node not in selected or node not in OPERATOR_NODE_PARAMETERS:
                raise ServiceError(f"{node} takes no lease-page parameters here.")
            if not isinstance(values, dict) or set(values) - OPERATOR_NODE_PARAMETERS[node]:
                raise ServiceError(
                    f"{node} lease-page parameters must be among "
                    f"{sorted(OPERATOR_NODE_PARAMETERS[node])}"
                )
            targets = sorted(json_loads(workflow.get("targets_json"), []))
            try:
                self.policy.validate_run(node, {"qubits": targets[:1], **values})
            except PolicyError as exc:
                raise ServiceError(f"{node}: {exc}") from exc
            node_parameters[node] = dict(values)
        options = {
            **options,
            "excluded_nodes": [
                node for node in self.settings.workflow_sequence if node in excluded
            ],
            "node_parameters": node_parameters,
        }
        if drive_mode is not None:
            options["drive_mode"] = drive_mode
        if flux_choice is not None:
            options["flux_calibration"] = bool(flux_choice)
        if qubit_starts:
            options["qubit_start_nodes"] = qubit_starts
            first_qubit = next(iter(history.get("qubit_start_nodes") or {}), None)
            if first_qubit is not None:
                current = qubit_starts.get(first_qubit, current)
        now = utc_now()
        connection.execute(
            "UPDATE workflows SET options_json = ?, current_node = ?, updated_at = ? "
            "WHERE id = ?",
            (json_dumps(options), current, now, workflow["id"]),
        )
        inherited = 0
        if history is not None and start_node is not None:
            inherited = self._inherit_evidence(
                connection,
                workflow["id"],
                str(history["source_workflow_id"]),
                start_node,
            )
            if qubit_starts:
                inherited += self._inherit_qubit_tail(
                    connection,
                    workflow["id"],
                    str(history["source_workflow_id"]),
                    qubit_starts,
                )
            connection.execute(
                "UPDATE workflows SET source_workflow_id = ?, start_node = ? WHERE id = ?",
                (str(history["source_workflow_id"]), start_node, workflow["id"]),
            )
        lease_id = proposal.get("autonomy_lease_id")
        lease_row = connection.execute(
            "SELECT allowed_nodes_json FROM autonomy_leases WHERE id = ?",
            (str(lease_id),),
        ).fetchone()
        if lease_row is not None:
            allowed = [
                node
                for node in json_loads(lease_row["allowed_nodes_json"], [])
                if node in sequence
                and (
                    start_node is None
                    or full_sequence.index(node) >= full_sequence.index(start_node)
                )
            ]
            if not allowed:
                raise ServiceError("At least one experiment must remain selected.")
            connection.execute(
                "UPDATE autonomy_leases SET allowed_nodes_json = ?, updated_at = ? "
                "WHERE id = ?",
                (json_dumps(allowed), now, str(lease_id)),
            )
        self.db.event(
            "workflow_options_selected",
            actor,
            {
                "lease_id": lease_id,
                "excluded_nodes": options["excluded_nodes"],
                "drive_mode": options.get("drive_mode"),
                "flux_calibration": options.get("flux_calibration"),
                "node_parameters": options["node_parameters"],
                "current_node": current,
                "start_node": start_node,
                "qubit_start_nodes": qubit_starts,
                "source_workflow_id": (
                    history.get("source_workflow_id") if history is not None else None
                ),
                "inherited_runs": inherited,
                "approval_method": approval_method,
            },
            workflow["id"],
            connection=connection,
        )

    def _finished_serial_passes(self, workflow_id: str, first: str) -> set[str]:
        """Qubits whose per-qubit pass is already over.

        A pass ends on the `advance` that leaves the tail: either back to the
        serialization node for the next qubit, or nowhere because the workflow
        is finished. Every other `advance` inside the tail just moves that same
        qubit to its next node.
        """

        sequence = list(self.settings.workflow_sequence)
        tail = set(sequence[sequence.index(first):])
        finished: set[str] = set()
        # Operator rule 2026-10-12: a 04 -> 03a return keeps the qubit's pass
        # open, and a pass can hand over to a qubit that starts after 03a, so
        # the server records each pass end explicitly.
        returned_runs = {
            str(event.get("run_id")) for event in self._03a_return_events(workflow_id)
        }
        for row in self.db.all(
            "SELECT payload_json FROM events WHERE workflow_id = ? "
            "AND event_type = 'serial_pass_finished'",
            (workflow_id,),
        ):
            qubit = json_loads(row["payload_json"], {}).get("qubit")
            if isinstance(qubit, str) and qubit:
                finished.add(qubit)
        for row in self.db.all(
            "SELECT r.id AS run_id, r.node_id AS node_id, "
            "r.parameters_json AS parameters_json, "
            "d.next_node AS next_node FROM decisions d "
            "JOIN runs r ON r.id = d.run_id "
            "WHERE d.workflow_id = ? AND d.decision = 'advance' "
            # 2026-10-13: a decision inherited from an earlier workflow (q2's
            # 10a -> 03a there) does not end the qubit's pass in this one.
            "AND r.inherited_from_run_id IS NULL",
            (workflow_id,),
        ):
            if row["node_id"] not in tail or row["run_id"] in returned_runs:
                continue
            if row["next_node"] not in (None, first):
                continue
            names = json_loads(row["parameters_json"], {}).get("qubits", [])
            if isinstance(names, list):
                finished.update(str(name) for name in names)
        return finished

    # -- Per-qubit pass rules, operator instruction 2026-10-12 ---------------

    def _carries_unresolved(self, workflow: dict[str, Any], node_id: str) -> bool:
        """True when a qubit continues past `node_id` whatever its result."""

        return (
            self._serial_pass_state(workflow) is not None
            and self._in_serial_tail(workflow, node_id)
            and node_id not in self._pass_prerequisite_nodes()
        )

    def _pass_prerequisite_nodes(self) -> set[str]:
        """Nodes whose failure leaves a qubit nothing to continue with."""

        value = self.policy.shared_xy_drive.get("pass_prerequisite_nodes", ["03a", "04"])
        return {str(node) for node in value}

    def _in_serial_tail(self, workflow: dict[str, Any], node_id: str) -> bool:
        first = self._serialize_from_node(workflow)
        sequence = list(self.settings.workflow_sequence)
        return (
            first is not None
            and first in sequence
            and node_id in sequence
            and sequence.index(node_id) >= sequence.index(first)
        )

    def _03a_return_events(self, workflow_id: str) -> list[dict[str, Any]]:
        return [
            {
                "event_id": int(row["id"]),
                "created_at": row["created_at"],
                **json_loads(row["payload_json"], {}),
            }
            for row in self.db.all(
                "SELECT id, created_at, payload_json FROM events WHERE workflow_id = ? "
                "AND event_type = 'serial_pass_03a_return' ORDER BY id",
                (workflow_id,),
            )
        ]

    def _return_cutoffs(self, workflow_id: str) -> dict[str, dict[str, int]]:
        """Per qubit: the last run rowid and event id before its latest return."""

        cutoffs: dict[str, dict[str, int]] = {}
        for event in self._03a_return_events(workflow_id):
            qubit = str(event.get("qubit") or "")
            if not qubit:
                continue
            cutoffs[qubit] = {
                "run_rowid": int(event.get("after_run_rowid") or 0),
                "event_id": int(event["event_id"]),
            }
        return cutoffs

    def _drop_returned_targets(
        self,
        workflow_id: str,
        rows: list[dict[str, Any]],
        resolved: set[str],
        resolver: Callable[[list[dict[str, Any]]], set[str]],
    ) -> set[str]:
        """Forget 03a/04 evidence a qubit gathered before it returned to 03a."""

        for qubit, cutoff in self._return_cutoffs(workflow_id).items():
            if qubit not in resolved:
                continue
            kept = [
                row
                for row in rows
                if int(row.get("run_rowid") or 0) > cutoff["run_rowid"]
            ]
            if qubit not in resolver(kept):
                resolved.discard(qubit)
        return resolved

    def _rejected_03a_lines(self, workflow_id: str) -> dict[str, list[list[float]]]:
        """Per qubit, the RF windows 04 rejected (no Rabi at that line)."""

        lines: dict[str, list[list[float]]] = {}
        for event in self._03a_return_events(workflow_id):
            qubit = str(event.get("qubit") or "")
            center = event.get("rejected_rf_hz")
            half = event.get("half_width_hz")
            if qubit and _finite_number(center) and _finite_number(half):
                lines.setdefault(qubit, []).append(
                    [float(center) - float(half), float(center) + float(half)]
                )
        return lines

    def _qubit_runs(
        self, workflow_id: str, node_id: str, qubit: str
    ) -> list[dict[str, Any]]:
        rows = self.db.all(
            "SELECT r.*, r.rowid AS run_rowid, d.decision AS decision FROM runs r "
            "LEFT JOIN decisions d ON d.run_id = r.id "
            "WHERE r.workflow_id = ? AND r.node_id = ? AND r.status = 'completed' "
            "ORDER BY r.rowid",
            (workflow_id, node_id),
        )
        cutoff = self._return_cutoffs(workflow_id).get(qubit)
        selected = []
        for row in rows:
            names = json_loads(row.get("parameters_json"), {}).get("qubits", [])
            if not isinstance(names, list) or qubit not in {str(n) for n in names}:
                continue
            if (
                cutoff is not None
                and node_id in {"03a", "04"}
                and int(row.get("run_rowid") or 0) <= cutoff["run_rowid"]
            ):
                continue
            selected.append(row)
        return selected

    def _04_return_check(
        self, workflow: dict[str, Any], qubit: str
    ) -> tuple[bool, str]:
        """May this qubit go back from 04 to 03a for another line?

        Operator rule 2026-10-12: only a qubit that never ran 03b (its line
        cannot be confirmed by flux), that shows no Rabi oscillation at all
        -- robust SNR below `no_rabi_max_robust_snr` after the top of the
        averaging ladder, or a failed run at the longest allowed gate -- and
        that has returned fewer than `max_returns_to_03a` times.
        """

        workflow_id = workflow["id"]
        rules = self.policy.raw["analysis"]["04"]
        if self._qubit_runs(workflow_id, "03b", qubit):
            return False, f"{qubit} ran 03b, so its 03a line is confirmed by flux"
        returns = sum(
            1
            for event in self._03a_return_events(workflow_id)
            if event.get("qubit") == qubit
        )
        maximum = int(rules.get("max_returns_to_03a", 3))
        if returns >= maximum:
            return False, f"{qubit} already returned to 03a {returns} times"
        if qubit in self._node_target_resolution(workflow_id, "04"):
            return False, f"{qubit} has a pi pulse"
        ladder_top = int(rules["max_noise_confirmation_num_averages"])
        no_rabi = float(rules.get("no_rabi_max_robust_snr", 4.0))
        gate_cap = self._xy_gate_length_cap(qubit)
        gate_length = self._x180_length_ns(qubit)
        at_gate_cap_after = None
        if gate_cap is not None and gate_length is not None and gate_length >= gate_cap:
            # Runs after the gate reached its cap; before any gate change, all.
            at_gate_cap_after = max(
                [
                    int(json_loads(row["payload_json"], {}).get("after_run_rowid") or 0)
                    for row in self.db.all(
                        "SELECT payload_json FROM events WHERE workflow_id = ? "
                        "AND event_type = 'xy_gate_length_set'",
                        (workflow_id,),
                    )
                    if json_loads(row["payload_json"], {}).get("qubit") == qubit
                ]
                or [0]
            )
        # Operator instruction 2026-10-12 (q3: 32 ns gate swept to 4% of full
        # scale): "no Rabi" counts only from a run that drove the qubit as hard
        # as allowed -- the amplitude swept to `return_min_amplitude_fraction`
        # of the waveform limit, at the longest allowed gate.
        fraction = float(rules.get("return_min_amplitude_fraction", 0.9))
        try:
            item = load_state(self.settings.active_state)["qubits"][qubit]
            amplitude_limit = float(
                self.policy.limits[channel_kind(item)]["max_wf_amplitude"]
            )
        except Exception:  # pragma: no cover - state problems surface elsewhere
            item, amplitude_limit = None, None
        for row in self._qubit_runs(workflow_id, "04", qubit):
            if row.get("analysis_status") not in {"needs_review", "failed"}:
                continue
            parameters = json_loads(row.get("parameters_json"), {})
            analysis = json_loads(row.get("analysis_json"), {})
            metrics = (
                analysis.get("dataset_metrics", {}).get("qubits", {}).get(qubit, {})
            )
            snr = metrics.get("robust_snr") if isinstance(metrics, dict) else None
            averages = parameters.get("num_averages")
            if at_gate_cap_after is None or int(row["run_rowid"]) <= at_gate_cap_after:
                continue
            recorded = (
                metrics.get("operation_amplitude_before_run")
                if isinstance(metrics, dict)
                else None
            )
            if _finite_number(recorded):
                baseline = abs(float(recorded))
            elif item is not None:
                baseline = abs(operation_amplitude(item, "x180"))
            else:
                continue
            swept = baseline * float(parameters.get("max_amp_factor", 1.9))
            if amplitude_limit is None or swept < fraction * amplitude_limit:
                continue
            if (
                isinstance(averages, int)
                and averages >= ladder_top
                and _finite_number(snr)
                and float(snr) < no_rabi
            ):
                return True, (
                    f"no Rabi oscillation at {averages} averages (robust SNR "
                    f"{float(snr):.3g}), amplitude swept to {swept:.3g} at the "
                    f"longest gate ({gate_cap:g} ns)"
                )
            return True, (
                f"no pi pulse at the longest gate ({gate_cap:g} ns) with the "
                f"amplitude swept to {swept:.3g}"
            )
        return False, (
            f"{qubit} has no 04 run showing the absence of a Rabi oscillation at "
            f"full drive: the amplitude must sweep to >= {fraction:g} of the "
            f"waveform limit at the longest allowed gate"
            + (f" ({gate_cap:g} ns)" if gate_cap is not None else "")
            + (
                f"; the x180 gate is {gate_length:g} ns -- lengthen it with "
                "jy_request_04_gate_time"
                if gate_cap is not None
                and gate_length is not None
                and gate_length < gate_cap
                else ""
            )
        )

    def _deferred_node_due(self, workflow: dict[str, Any], qubit: str) -> str | None:
        """A deferred node (05) the qubit must revisit now, if any.

        Operator rule 2026-10-12: a qubit whose 05 misses the floor first
        goes through 07d/07b and then returns to 05 with the better readout.
        """

        deferrable = self.policy.shared_xy_drive.get("deferrable_nodes") or {}
        workflow_id = workflow["id"]
        for node, after in deferrable.items():
            node, after = str(node), str(after)
            if node not in self._workflow_sequence(workflow):
                continue
            if qubit in self._node_target_resolution(workflow_id, node):
                continue
            after_runs = self._qubit_runs(workflow_id, after, qubit)
            if not after_runs:
                continue
            last_after = int(after_runs[-1]["run_rowid"])
            if any(
                int(row["run_rowid"]) > last_after
                for row in self._qubit_runs(workflow_id, node, qubit)
            ):
                continue
            return node
        return None

    def _deferral_pending(self, workflow: dict[str, Any], node_id: str, qubit: str) -> bool:
        """True while `node_id` may be left unresolved for a later revisit."""

        deferrable = self.policy.shared_xy_drive.get("deferrable_nodes") or {}
        after = deferrable.get(node_id)
        if after is None or str(after) not in self._workflow_sequence(workflow):
            return False
        return not self._qubit_runs(workflow["id"], str(after), qubit)

    def _qubit_start_node(self, workflow: dict[str, Any], qubit: str) -> str:
        """Where a qubit's pass starts (lease-page choice, 2026-10-12)."""

        first = self._serialize_from_node(workflow) or "03a"
        starts = self._workflow_options(workflow).get("qubit_start_nodes") or {}
        node = starts.get(qubit) if isinstance(starts, dict) else None
        return str(node) if isinstance(node, str) and node else first

    def _pass_next_node(
        self, workflow: dict[str, Any], current: str, qubit: str
    ) -> tuple[str | None, str]:
        """Where `qubit` goes after `current` inside its pass; None ends it."""

        workflow_id = workflow["id"]
        sequence = self._workflow_sequence(workflow)
        resolved = qubit in self._node_target_resolution(workflow_id, current)
        if current in self._pass_prerequisite_nodes() and not resolved:
            if current == "04":
                allowed, _why = self._04_return_check(workflow, qubit)
                if allowed:
                    return "03a", "return"
            return None, "end"
        if current not in self._pass_prerequisite_nodes() or resolved:
            revisit = self._deferred_node_due(workflow, qubit)
            if revisit is not None and sequence.index(current) > sequence.index(revisit):
                return revisit, "revisit"
        index = sequence.index(current)
        for node in sequence[index + 1:]:
            if node in FLUX_NODES and not self._flux_node_has_targets(
                workflow, current, node
            ):
                continue
            if self._node_is_single_pass(node) and qubit in (
                self._single_pass_measured_targets(workflow, node)
            ):
                # 05's revisit after 07b does not repeat 07d/07b.
                continue
            if self._statistics_source(node) is not None and qubit not in (
                self._statistics_qualified(workflow_id, node)
            ):
                # Its single measurement or its readout did not pass.
                continue
            return node, "continue"
        return None, "end"

    def _statistics_qualified(self, workflow_id: str, node_id: str) -> set[str]:
        """Qubits a statistics node may measure (operator rules 2026-10-12/13).

        Its own single measurement must have passed. A qubit whose readout did
        not pass `shared_xy_drive.statistics_readout_node` still qualifies
        (operator choice 2026-10-13), but `_validate_node_prerequisites` then
        requires the raw I/Q signal (use_state_discrimination=false).
        """

        source = self._statistics_source(node_id)
        if source is None:
            return set()
        return set(self._node_target_resolution(workflow_id, source))

    def _statistics_source(self, node_id: str) -> str | None:
        sources = self.policy.shared_xy_drive.get("statistics_sources") or {}
        value = sources.get(node_id) if isinstance(sources, dict) else None
        return str(value) if value else None

    def _record_pass_transition(
        self,
        workflow: dict[str, Any],
        run: dict[str, Any],
        qubit: str,
        transition: str,
        actor: str,
    ) -> None:
        workflow_id = workflow["id"]
        if transition == "end":
            self.db.event(
                "serial_pass_finished",
                actor,
                {"qubit": qubit, "run_id": run["id"], "node_id": run["node_id"]},
                workflow_id,
            )
            return
        # A return to 03a: everything this qubit measured at 03a/04 so far is
        # set aside, and the line 04 found no Rabi at is masked for it.
        rules = self.policy.raw["analysis"]["03a"]
        half_width = 1e6 * float(rules.get("rejected_line_min_half_width_mhz", 5.0))
        rejected = None
        for row in reversed(self._qubit_runs(workflow_id, "03a", qubit)):
            analysis = json_loads(row.get("analysis_json"), {})
            if analysis.get("03a_stage") != "fine":
                continue
            fit = analysis.get("fit_quality", {}).get("results", {}).get(qubit, {})
            metrics = analysis.get("dataset_metrics", {}).get("qubits", {}).get(qubit, {})
            if isinstance(fit, dict) and _finite_number(fit.get("drive_freq")):
                rejected = float(fit["drive_freq"])
                width = metrics.get("feature_fwhm_hz") if isinstance(metrics, dict) else None
                if _finite_number(width):
                    half_width = max(half_width, float(width))
                break
        if rejected is None:
            try:
                rejected = float(
                    load_state(self.settings.active_state)["qubits"][qubit]["extras"]["idle_freq"]
                )
            except Exception:
                rejected = None
        last = self.db.one(
            "SELECT MAX(rowid) AS rowid FROM runs WHERE workflow_id = ?", (workflow_id,)
        )
        _allowed, why = self._04_return_check(workflow, qubit)
        self.db.event(
            "serial_pass_03a_return",
            actor,
            {
                "qubit": qubit,
                "run_id": run["id"],
                "after_run_rowid": int((last or {}).get("rowid") or 0),
                "rejected_rf_hz": rejected,
                "half_width_hz": half_width,
                "reason": why,
            },
            workflow_id,
        )

    def _serial_pass_state(self, workflow: dict[str, Any]) -> dict[str, Any] | None:
        """Which qubit the per-qubit pass is on, and which ones are left."""

        first = self._serialize_from_node(workflow)
        if first is None:
            return None
        _, absent = self._02c_target_resolution(workflow["id"])
        # 2026-10-12: a qubit 02 found no resonator for has no pass either.
        absent = set(absent) | self._scientifically_unmeasurable_targets_by_node(
            workflow["id"]
        ).get("02", set())
        order = [
            str(name)
            for name in json_loads(workflow["targets_json"], [])
            if str(name) not in absent
        ]
        finished = self._finished_serial_passes(workflow["id"], first)
        remaining = [name for name in order if name not in finished]
        return {
            "node": first,
            "order": order,
            "finished": sorted(finished & set(order)),
            "remaining": remaining,
            "current": remaining[0] if remaining else None,
            "index": len(order) - len(remaining) + 1 if remaining else len(order),
            "total": len(order),
        }

    def _expected_next_node(
        self, workflow: dict[str, Any], current: str
    ) -> str | None:
        """The node an `advance` from `current` must name.

        Inside a per-qubit pass this is the next node in the sequence while the
        qubit still has usable evidence, and otherwise the serialization node
        again for the next qubit -- or nothing at all, once the last qubit's
        pass is over.
        """

        sequence = self._workflow_sequence(workflow)
        index = sequence.index(current)
        # Operator rule 2026-10-08: a flux node with no qubit to measure (all
        # flat, crossing, or not resolved upstream) is passed over.
        expected = next(
            (
                node
                for node in sequence[index + 1:]
                if (
                    node not in FLUX_NODES
                    or self._flux_node_has_targets(workflow, current, node)
                )
                and (
                    # 2026-10-12: a statistics node no qubit qualifies for.
                    self._statistics_source(node) is None
                    or bool(self._statistics_qualified(workflow["id"], node))
                )
            ),
            None,
        )
        serial = self._serial_pass_state(workflow)
        if (
            serial is None
            or serial["node"] not in sequence
            or index < sequence.index(serial["node"])
        ):
            return expected
        if serial["current"] is not None:
            # Operator rule 2026-10-12: only 03a/04 without a result end a
            # qubit's pass early (04 may first return to 03a); 05 is revisited
            # after 07b; every other node hands the qubit on.
            following, _kind = self._pass_next_node(workflow, current, serial["current"])
            if following is not None:
                return following
        remaining = [
            name for name in serial["remaining"] if name != serial["current"]
        ]
        return self._qubit_start_node(workflow, remaining[0]) if remaining else None

    def _flux_node_has_targets(
        self, workflow: dict[str, Any], current: str, node: str
    ) -> bool:
        """Would `node` have a qubit to measure once `current` advances?

        `_active_targets_for_node` only carries a node's resolution after its
        `advance` is recorded, so the advance being decided is applied here.
        """

        targets = self._active_targets_for_node(workflow, node)
        if (
            current in SUBGROUP_NODES
            and current not in FLUX_NODES
            and not self._carries_unresolved(workflow, current)
        ):
            carried = self._node_target_resolution(workflow["id"], current)
            if self._node_is_single_pass(current):
                excluded = self._scientifically_unmeasurable_targets_by_node(
                    workflow["id"]
                ).get(current, set())
                carried |= self._single_pass_measured_targets(workflow, current) - excluded
            targets &= carried
        return bool(targets)

    def _complete_autonomy_scope_if_needed(
        self,
        workflow_id: str,
        next_node: str | None,
        actor: str,
    ) -> None:
        for lease in self.db.all(
            "SELECT id, allowed_nodes_json FROM autonomy_leases "
            "WHERE workflow_id = ? AND status = 'active'",
            (workflow_id,),
        ):
            allowed_nodes = json_loads(lease["allowed_nodes_json"], [])
            if next_node in allowed_nodes:
                continue
            reason_text = (
                "Authorized node range completed; a new human lease is required "
                "before the next node."
            )
            now = utc_now()
            self.db.execute(
                "UPDATE autonomy_leases SET status = 'completed', "
                "stopped_reason = ?, updated_at = ? WHERE id = ?",
                (reason_text, now, lease["id"]),
            )
            self.db.execute(
                "UPDATE measurement_sessions SET status = 'completed', "
                "updated_at = ? WHERE autonomy_lease_id = ?",
                (now, lease["id"]),
            )
            self.db.event(
                "autonomy_scope_completed",
                actor,
                {"lease_id": lease["id"], "next_node": next_node},
                workflow_id,
            )

    def _rewind_invalidated_subgroup_evidence(
        self,
        workflow: dict[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        """Reopen 04/05 when a stricter policy invalidates prior advancement."""
        sequence = list(self.settings.workflow_sequence)
        current = workflow.get("current_node")
        if current not in sequence:
            return workflow
        current_index = sequence.index(current)
        for node_id in ("04", "05"):
            if node_id not in sequence or sequence.index(node_id) >= current_index:
                continue
            advanced = self.db.one(
                "SELECT d.id FROM decisions d JOIN runs r ON r.id = d.run_id "
                "WHERE d.workflow_id = ? AND r.node_id = ? "
                "AND d.decision = 'advance' ORDER BY d.created_at DESC LIMIT 1",
                (workflow["id"], node_id),
            )
            if advanced is None:
                continue
            eligible = self._active_targets_for_node(workflow, node_id)
            resolved = self._node_target_resolution(workflow["id"], node_id)
            missing = eligible - resolved
            if not missing:
                continue
            self.db.execute(
                "UPDATE workflows SET current_node = ?, updated_at = ? WHERE id = ?",
                (node_id, utc_now(), workflow["id"]),
            )
            self.db.event(
                "workflow_evidence_revalidation",
                actor,
                {
                    "reopened_node": node_id,
                    "missing_targets": sorted(missing),
                    "reason": (
                        "Current 04/05 robust-SNR policy invalidated prior "
                        "completion evidence before a new autonomy lease."
                    ),
                },
                workflow["id"],
            )
            return self._workflow(workflow["id"])
        return workflow

    def _uses_state_discrimination(
        self, node_id: str, parameters: dict[str, Any]
    ) -> bool:
        try:
            definition = self.policy.node_definition(node_id)
        except Exception:
            return False
        if "use_state_discrimination" not in definition.get("allowed_parameters", {}):
            return False
        default = definition.get("defaults", {}).get("use_state_discrimination", False)
        return parameters.get("use_state_discrimination", default) is not False

    def _state_discrimination_qualified_targets(self, workflow_id: str) -> set[str]:
        """Qubits whose 07b allows state discrimination (operator 2026-10-14).

        A completed 07b run of this workflow must have reached
        `analysis."07b".min_readout_fidelity` with two Gaussian blobs. 07b
        always passes now, so passing alone says nothing about the readout.
        """

        readout = self.policy.shared_xy_drive.get("statistics_readout_node") or "07b"
        floor = float(self.policy.raw["analysis"]["07b"]["min_readout_fidelity"])
        cutoff = self._readout_reopen_cutoff(workflow_id, readout)
        qualified: set[str] = set()
        for run in self.db.all(
            "SELECT analysis_json, started_at FROM runs WHERE workflow_id = ? "
            "AND node_id = ? AND status = 'completed' ORDER BY rowid",
            (workflow_id, readout),
        ):
            if cutoff is not None and str(run.get("started_at") or "") < cutoff:
                continue
            results = (
                json_loads(run.get("analysis_json"), {})
                .get("fit_quality", {})
                .get("results", {})
            )
            if not isinstance(results, dict):
                continue
            for name, fit in results.items():
                fidelity = fit.get("readout_fidelity") if isinstance(fit, dict) else None
                if (
                    isinstance(fit, dict)
                    and fit.get("morphology_pass") is True
                    and _finite_number(fidelity)
                    and floor <= float(fidelity) <= 1.0
                ):
                    qualified.add(str(name))
        return qualified

    def _validate_state_discrimination(
        self,
        workflow: dict[str, Any],
        node_id: str,
        parameters: dict[str, Any],
    ) -> None:
        """Refuse a discriminated run for a qubit whose readout cannot support it.

        Operator choices 2026-10-12/14: with a poor discriminator every short
        run is flat (q1 05st 2026-10-12 crashed, "min() arg is an empty
        sequence"). Such qubits run on the raw I/Q signal, which 07b's
        committed rotation angle puts on I.
        """

        if not self.policy.shared_xy_drive.get(
            "statistics_readout_node"
        ) or not self._uses_state_discrimination(node_id, parameters):
            return
        floor = float(self.policy.raw["analysis"]["07b"]["min_readout_fidelity"])
        requested = {str(name) for name in parameters.get("qubits", [])}
        unqualified = sorted(
            requested - self._state_discrimination_qualified_targets(workflow["id"]),
            key=_natural_qubit_key,
        )
        if unqualified:
            raise AutonomyScopeError(
                f"{unqualified} have no 07b readout with fidelity >= {floor:g} and "
                f"two Gaussian blobs; run {node_id} with "
                "use_state_discrimination=false (raw I/Q signal)"
            )

    def _validate_node_prerequisites(
        self,
        workflow: dict[str, Any],
        node_id: str,
        parameters: dict[str, Any],
    ) -> None:
        """Validate all baselines before one non-refitted statistics run.

        Refusals are AutonomyScopeError (2026-10-12): a missing baseline is a
        correctable planning error and must not halt the lease.
        """
        self._validate_state_discrimination(workflow, node_id, parameters)
        if node_id == "07d":
            # 2026-10-13: the node rejects a duration step that is not a
            # multiple of 4 ns (2000 / 8 = 250 crashed run 8e3fd899) -- refuse
            # it here, before a worker takes the hardware lock.
            defaults = self.policy.node_definition("07d").get("defaults", {})
            longest = parameters.get("max_duration_in_ns", defaults.get("max_duration_in_ns"))
            count = parameters.get("num_durations", defaults.get("num_durations"))
            if _finite_positive(longest) and _finite_positive(count):
                step = float(longest) / float(count)
                if step != int(step) or int(step) % 4:
                    raise AutonomyScopeError(
                        f"07d max_duration_in_ns / num_durations = {longest} / {count} "
                        f"= {step:g} ns must be a multiple of 4 ns"
                    )
        prerequisites = {
            "05st": ("05", "t1_seconds"),
            "06st_t2star": ("06", "coherence_seconds"),
            "06st_t2e": ("06b", "coherence_seconds"),
        }
        if node_id not in prerequisites:
            return
        baseline_node, lifetime_key = prerequisites[node_id]
        requested = {str(name) for name in parameters.get("qubits", [])}
        reset_type = _reset_type(parameters)
        # Operator rule 2026-10-12: each statistics node needs only its own
        # single measurement (05st <- 05, 06st_t2star <- 06, 06st_t2e <- 06b).
        # Active reset keeps its rule: every baseline repeated with it.
        required_baselines = (
            [baseline_node]
            if self._statistics_source(node_id) is not None
            and _reset_type(parameters) != "active"
            else self._active_reset_verification_nodes()
        )
        accepted: dict[str, dict[str, dict[str, Any]]] = {}
        missing_by_node: dict[str, list[str]] = {}
        for required_node in required_baselines:
            accepted[required_node] = {}
            for target in sorted(requested):
                fit = self._latest_accepted_target_fit(
                    workflow["id"], required_node, target, reset_type
                )
                if fit is None:
                    missing_by_node.setdefault(required_node, []).append(target)
                else:
                    accepted[required_node][target] = fit
        if missing_by_node:
            details = "; ".join(
                f"{name}: {targets}" for name, targets in missing_by_node.items()
            )
            qualifier = "repeated " if reset_type == "active" else ""
            raise AutonomyScopeError(
                f"{node_id} requires accepted {qualifier}"
                f"{'/'.join(required_baselines)} evidence "
                f"using reset_type={reset_type} for every target; missing {details}"
            )
        maximum = float(parameters["max_wait_time_in_ns"])
        for target in sorted(requested):
            evidence = accepted[baseline_node][target]
            lifetime = evidence.get(lifetime_key) if evidence else None
            coverage = evidence.get("coverage_lifetimes") if evidence else None
            if not _finite_positive(lifetime):
                raise AutonomyScopeError(
                    f"{node_id} cannot resolve a finite {baseline_node} lifetime for {target}"
                )
            if not _finite_positive(coverage) or float(coverage) < 3.5:
                raise AutonomyScopeError(
                    f"{node_id} requires {baseline_node} to demonstrate decay to "
                    f"equilibrium (at least 3.5 lifetimes) for {target}"
                )
            lifetime_ns = float(lifetime) * 1e9
            ratio = maximum / lifetime_ns
            if not 3.5 <= ratio <= 5.5:
                raise AutonomyScopeError(
                    f"{node_id} max_wait_time_in_ns for {target} must be about four "
                    f"times the fitted lifetime; requested ratio is {ratio:.3g}"
                )

    def _active_reset_verification_nodes(self) -> tuple[str, ...]:
        configured = self.policy.raw.get("reset_policy", {}).get(
            "active_statistics_verification_nodes", ["05", "06", "06b"]
        )
        return tuple(str(node) for node in configured)

    def _is_active_reset_verification_node(
        self, workflow: dict[str, Any], node_id: str
    ) -> bool:
        if node_id not in self._active_reset_verification_nodes():
            return False
        sequence = list(self.settings.workflow_sequence)
        current = workflow.get("current_node")
        qualification_node = str(
            self.policy.raw.get("reset_policy", {}).get(
                "active_qualification_node", "07b"
            )
        )
        if current not in sequence or qualification_node not in sequence:
            return False
        return sequence.index(current) > sequence.index(qualification_node)

    def _validate_active_reset_prerequisite(
        self,
        workflow: dict[str, Any],
        node_id: str,
        parameters: dict[str, Any],
    ) -> None:
        if not _is_active_reset(parameters):
            return
        qualification_node = str(
            self.policy.raw.get("reset_policy", {}).get(
                "active_qualification_node", "07b"
            )
        )
        # 07b is the qualifying experiment, so it must be allowed to try active
        # reset before such evidence exists.
        if node_id == qualification_node:
            return
        requested = {str(name) for name in parameters.get("qubits", [])}
        qualified = self._active_reset_qualified_targets(workflow["id"])
        missing = requested - qualified
        if missing:
            floor = float(self.policy.raw["reset_policy"]["active_min_readout_fidelity"])
            morphology = bool(
                self.policy.raw.get("analysis", {})
                .get("07b", {})
                .get("active_reset_requires_morphology", True)
            )
            raise ServiceError(
                "Active reset requires an accepted active-reset 07b result with "
                f"{'compact clouds and ' if morphology else ''}readout fidelity "
                f">= {floor:g}; missing qualification for {sorted(missing)}"
            )

    def _reset_parameter_name(self, node_id: str) -> str | None:
        """The node's reset parameter, if it has one."""
        try:
            allowed = self.policy.node_definition(node_id).get("allowed_parameters", {})
        except Exception:
            return None
        for name in ("reset_type", "reset_type_thermal_or_active"):
            if name in allowed:
                return name
        return None

    def _reset_groups(
        self, workflow: dict[str, Any], node_id: str
    ) -> dict[str, set[str]] | None:
        """Operator choice 2026-10-04: after 07b, run active and thermal apart.

        Qubits qualified for active reset by an accepted active-reset 07b run
        use active reset at every later node that has a reset parameter; the
        others stay thermal. Returns None where no grouping applies.
        """
        policy = self.policy.raw.get("reset_policy", {})
        if not policy.get("group_by_reset_after_qualification", False):
            return None
        if self._reset_parameter_name(node_id) is None:
            return None
        sequence = list(self.settings.workflow_sequence)
        qualification = str(policy.get("active_qualification_node", "07b"))
        if node_id not in sequence or qualification not in sequence:
            return None
        if sequence.index(node_id) <= sequence.index(qualification):
            return None
        active = self._active_targets_for_node(workflow, node_id)
        qualified = self._active_reset_qualified_targets(workflow["id"]) & active
        groups = {"active": qualified, "thermal": active - qualified}
        return {name: members for name, members in groups.items() if members}

    def _validate_reset_group(
        self, workflow: dict[str, Any], node_id: str, parameters: dict[str, Any]
    ) -> None:
        groups = self._reset_groups(workflow, node_id)
        if not groups:
            return
        requested = {str(name) for name in parameters.get("qubits", [])}
        wanted = "active" if _is_active_reset(parameters) else "thermal"
        if wanted == "active":
            # Unqualified qubits are refused by the active-reset prerequisite
            # check with its own, more specific message.
            return
        members = groups.get(wanted, set())
        outside = sorted(requested - members)
        if outside:
            raise AutonomyScopeError(
                f"{node_id} runs active-reset and thermal qubits separately; "
                f"{outside} are not in the {wanted} group "
                f"({ {name: sorted(group) for name, group in groups.items()} })"
            )

    def _group_has_completed_run(
        self, workflow: dict[str, Any], node_id: str, group: str, members: set[str]
    ) -> bool:
        cutoff = self._readout_reopen_cutoff(workflow["id"], node_id) or ""
        for run in self.db.all(
            "SELECT parameters_json FROM runs WHERE workflow_id = ? AND node_id = ? "
            "AND status = 'completed' AND COALESCE(started_at, '') >= ?",
            (workflow["id"], node_id, cutoff),
        ):
            parameters = json_loads(run.get("parameters_json"), {})
            run_group = "active" if _is_active_reset(parameters) else "thermal"
            names = {str(name) for name in parameters.get("qubits", [])}
            if run_group == group and names & members:
                return True
        return False

    def _active_reset_qualified_targets(self, workflow_id: str) -> set[str]:
        qualification_node = str(
            self.policy.raw.get("reset_policy", {}).get(
                "active_qualification_node", "07b"
            )
        )
        floor = float(self.policy.raw["reset_policy"]["active_min_readout_fidelity"])
        qualified: set[str] = set()
        rows = self.db.all(
            "SELECT r.status, r.analysis_status, r.analysis_json, "
            "r.parameters_json, r.started_at, d.decision FROM runs r "
            "LEFT JOIN decisions d ON d.run_id = r.id "
            "WHERE r.workflow_id = ? AND r.node_id = ? ORDER BY r.rowid",
            (workflow_id, qualification_node),
        )
        cutoff = self._readout_reopen_cutoff(workflow_id, qualification_node)
        if cutoff is not None:
            rows = [row for row in rows if str(row.get("started_at") or "") >= cutoff]
        for row in rows:
            if row.get("status") != "completed" or row.get("decision") not in {
                "advance",
                "repeat",
            }:
                continue
            parameters = json_loads(row.get("parameters_json"), {})
            if not _is_active_reset(parameters):
                continue
            analysis = json_loads(row.get("analysis_json"), {})
            results = analysis.get("fit_quality", {}).get("results", {})
            if not isinstance(results, dict):
                continue
            for target, fit in results.items():
                fidelity = fit.get("readout_fidelity") if isinstance(fit, dict) else None
                if (
                    isinstance(fit, dict)
                    and fit.get("fit_successful") is True
                    and fit.get("active_reset_qualified") is True
                    and fit.get("morphology_pass") is True
                    and _finite_positive(fidelity)
                    and floor <= float(fidelity) <= 1.0
                ):
                    qualified.add(str(target))
        return qualified

    def _latest_accepted_target_fit(
        self,
        workflow_id: str,
        node_id: str,
        target: str,
        reset_type: str,
    ) -> dict[str, Any] | None:
        rows = self.db.all(
            "SELECT r.id AS run_id, r.status, r.analysis_status, r.analysis_json, "
            "r.parameters_json, d.decision FROM runs r "
            "LEFT JOIN decisions d ON d.run_id = r.id "
            "WHERE r.workflow_id = ? AND r.node_id = ? ORDER BY r.rowid DESC",
            (workflow_id, node_id),
        )
        for row in rows:
            # 2026-10-14 (q3 05 #560): an operator-approved commit of this
            # node's own output from the run makes it accepted evidence.
            operator_accepted = target in self._operator_committed_targets(
                workflow_id, [row], node_id
            )
            # A multiplexed run may resolve this target while another target in
            # the same run causes the workflow-level decision to be repeat.
            if not operator_accepted and row.get("decision") not in {"advance", "repeat"}:
                continue
            run_parameters = json_loads(row.get("parameters_json"), {})
            if _reset_type(run_parameters) != reset_type:
                continue
            if target not in run_parameters.get("qubits", []):
                continue
            if not operator_accepted and target not in self._resolve_targets_from_rows(
                node_id, [row]
            ):
                continue
            analysis = json_loads(row.get("analysis_json"), {})
            fit = analysis.get("fit_quality", {}).get("results", {}).get(target)
            if isinstance(fit, dict):
                return fit
        return None

    def _latest_target_fit(
        self, workflow_id: str, node_id: str, target: str
    ) -> dict[str, Any] | None:
        rows = self.db.all(
            "SELECT parameters_json, analysis_json, analysis_status FROM runs "
            "WHERE workflow_id = ? AND node_id = ? AND status = 'completed' "
            "ORDER BY started_at DESC",
            (workflow_id, node_id),
        )
        for row in rows:
            parameters = json_loads(row.get("parameters_json"), {})
            if target not in parameters.get("qubits", []):
                continue
            if row.get("analysis_status") not in {"pass", "needs_review"}:
                continue
            analysis = json_loads(row.get("analysis_json"), {})
            fit = analysis.get("fit_quality", {}).get("results", {}).get(target)
            if isinstance(fit, dict):
                return fit
        return None

    def _target_has_lifetime_coverage(
        self,
        workflow_id: str,
        node_id: str,
        target: str,
        minimum: float,
    ) -> bool:
        fit = self._latest_target_fit(workflow_id, node_id, target)
        coverage = fit.get("coverage_lifetimes") if fit else None
        return _finite_positive(coverage) and float(coverage) >= float(minimum)

    def _workflow(self, workflow_id: str) -> dict[str, Any]:
        workflow = self.db.one("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
        if workflow is None:
            raise ServiceError(f"Unknown workflow: {workflow_id}")
        return workflow

    def _reject_retained_hardware_lock(self) -> None:
        """Permit reconnecting to a live worker, but never start through quarantine."""
        if not self.settings.lock_path.exists():
            return
        active = self.db.one(
            "SELECT id, pid FROM runs WHERE status IN ('starting','running','stopping') "
            "ORDER BY started_at DESC LIMIT 1"
        )
        if active is not None:
            pid = active.get("pid")
            if isinstance(pid, int) and self.runner.process_is_alive(pid):
                return
        raise ServiceError(
            "A retained hardware lock places JY in recovery-only quarantine. "
            f"Use the local operator console ({self.operator_console_url}) to inspect "
            "and recover it before entering a new measurement mode. To close all "
            "remaining JY programs first, issue '恢復' or 'Recover'."
        )

    def _decode_workflow(self, row: dict[str, Any] | None) -> dict[str, Any]:
        if row is None:
            raise ServiceError("Workflow row is missing")
        result = dict(row)
        result["targets"] = json_loads(result.pop("targets_json"), [])
        result["initial_parameters"] = json_loads(
            result.pop("initial_parameters_json"), {}
        )
        result["options"] = self._workflow_options(row)
        result.pop("options_json", None)
        return result

    def _decode_proposal(self, row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = json_loads(result.pop("payload_json"), {})
        delegated = bool(result.get("autonomy_lease_id")) and result["kind"] != "autonomy_lease"
        session = self.db.one(
            "SELECT id FROM measurement_sessions WHERE workflow_id = ? "
            "AND status IN ('active','paused','halted','stopping',"
            "'recovery_required') ORDER BY started_at DESC LIMIT 1",
            (result["workflow_id"],),
        )
        dashboard_session_id = str(session["id"]) if session is not None else None
        browser_url = (
            self.session_dashboard_url(dashboard_session_id)
            if session is not None
            else self.browser_approval_url(result["id"])
        )
        result["dashboard_session_id"] = dashboard_session_id
        result["browser_url"] = browser_url
        result["human_approval"] = {
            "required": result["status"] == "pending",
            "browser_url": browser_url,
            "confirmation": f"APPROVE {result['id']}",
            "command": f"jy-agent approve {result['id']}",
            "instruction": (
                "This action is delegated by the referenced human-approved autonomy "
                "lease; no additional proposal approval is required."
                if delegated
                else
                "A human must open browser_url, review the result plot(s), Decision, "
                "Reason, next action, and complete payload, then type the displayed "
                "confirmation. The terminal command is fallback only; no MCP "
                "approval tool exists."
            ),
        }
        if result["status"] == "pending" and not delegated:
            result["operator_handoff"] = approval_operator_handoff()
        return result

    def _decode_run(self, row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["parameters"] = json_loads(result.pop("parameters_json"), {})
        result["analysis"] = json_loads(result.pop("analysis_json"), None)
        result["elapsed_seconds"] = _elapsed_seconds(result)
        return result


def _elapsed_seconds(row: dict[str, Any]) -> float | None:
    started_at = row.get("started_at")
    if not started_at:
        return None
    try:
        started = datetime.fromisoformat(str(started_at))
        finished_at = row.get("finished_at")
        finished = (
            datetime.fromisoformat(str(finished_at))
            if finished_at
            else datetime.now(timezone.utc)
        )
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if finished.tzinfo is None:
            finished = finished.replace(tzinfo=timezone.utc)
        return max(0.0, (finished - started).total_seconds())
    except (TypeError, ValueError):
        return None


def _reset_type(parameters: Any) -> str:
    if not isinstance(parameters, dict):
        return "thermal"
    value = parameters.get(
        "reset_type",
        parameters.get("reset_type_thermal_or_active", "thermal"),
    )
    return str(value).strip().casefold()


def _is_active_reset(parameters: Any) -> bool:
    return _reset_type(parameters).startswith("active")


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
