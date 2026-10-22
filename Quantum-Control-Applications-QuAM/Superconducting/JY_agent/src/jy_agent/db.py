from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .util import json_dumps, utc_now


SCHEMA = """
CREATE TABLE IF NOT EXISTS workflows (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    targets_json TEXT NOT NULL,
    initial_parameters_json TEXT NOT NULL,
    current_node TEXT,
    client_id TEXT NOT NULL,
    options_json TEXT
);
CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    approved_at TEXT,
    approved_by TEXT,
    max_uses INTEGER NOT NULL,
    uses INTEGER NOT NULL DEFAULT 0,
    source_client TEXT NOT NULL,
    autonomy_lease_id TEXT,
    FOREIGN KEY(workflow_id) REFERENCES workflows(id)
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    status TEXT NOT NULL,
    pid INTEGER,
    started_at TEXT,
    finished_at TEXT,
    snapshot_id INTEGER,
    snapshot_path TEXT,
    active_state_hash_before TEXT,
    active_state_hash_after TEXT,
    error TEXT,
    log_path TEXT,
    analysis_status TEXT NOT NULL DEFAULT 'not_started',
    analysis_json TEXT,
    autonomy_lease_id TEXT,
    process_token TEXT,
    stop_intent TEXT,
    stop_requested_at TEXT,
    exit_code INTEGER,
    exit_receipt_path TEXT,
    termination_cause TEXT,
    FOREIGN KEY(workflow_id) REFERENCES workflows(id),
    FOREIGN KEY(proposal_id) REFERENCES proposals(id)
);
CREATE TABLE IF NOT EXISTS autonomy_leases (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    targets_json TEXT NOT NULL,
    allowed_nodes_json TEXT NOT NULL,
    duration_hours REAL NOT NULL,
    max_total_runs INTEGER,
    max_attempts_per_node_qubit INTEGER NOT NULL,
    auto_state_commit_statuses_json TEXT NOT NULL,
    halt_conditions_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    activated_at TEXT,
    expires_at TEXT,
    updated_at TEXT NOT NULL,
    source_client TEXT NOT NULL,
    approved_by TEXT,
    stopped_reason TEXT,
    FOREIGN KEY(workflow_id) REFERENCES workflows(id),
    FOREIGN KEY(proposal_id) REFERENCES proposals(id)
);
CREATE TABLE IF NOT EXISTS measurement_sessions (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source_client TEXT NOT NULL,
    autonomy_lease_id TEXT,
    FOREIGN KEY(workflow_id) REFERENCES workflows(id),
    FOREIGN KEY(autonomy_lease_id) REFERENCES autonomy_leases(id)
);
CREATE TABLE IF NOT EXISTS emergency_stops (
    run_id TEXT PRIMARY KEY,
    autonomy_lease_id TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    force_after TEXT NOT NULL,
    actor TEXT NOT NULL,
    status TEXT NOT NULL,
    completed_at TEXT,
    result_json TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id),
    FOREIGN KEY(autonomy_lease_id) REFERENCES autonomy_leases(id)
);
CREATE TABLE IF NOT EXISTS shutdown_requests (
    session_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    active_run_id TEXT,
    stop_error TEXT,
    quarantine INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES measurement_sessions(id),
    FOREIGN KEY(workflow_id) REFERENCES workflows(id)
);
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    next_node TEXT,
    next_parameters_json TEXT NOT NULL,
    state_patch_json TEXT NOT NULL,
    client_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(workflow_id) REFERENCES workflows(id),
    FOREIGN KEY(run_id) REFERENCES runs(id)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    workflow_id TEXT,
    actor TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS idempotency_records (
    operation TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    response_json TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    PRIMARY KEY(operation, operation_id)
);
CREATE TABLE IF NOT EXISTS dashboard_pairings (
    id TEXT PRIMARY KEY,
    instance_nonce TEXT NOT NULL,
    code_sha256 TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT,
    created_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dashboard_devices (
    id TEXT PRIMARY KEY,
    instance_nonce TEXT NOT NULL,
    token_sha256 TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    last_seen_at TEXT,
    revoked_at TEXT,
    created_by TEXT NOT NULL
);
-- Operator request 2026-10-15: token numbers of every Claude Code API call,
-- kept after Claude Code deletes old transcripts. No conversation text.
CREATE TABLE IF NOT EXISTS usage_calls (
    message_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    speed TEXT NOT NULL,
    turn TEXT NOT NULL,
    tools_json TEXT NOT NULL,
    input INTEGER NOT NULL,
    cache_write_5m INTEGER NOT NULL,
    cache_write_1h INTEGER NOT NULL,
    cache_read INTEGER NOT NULL,
    output INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS ux_one_open_workflow
ON workflows ((1)) WHERE status IN ('active', 'paused');
CREATE UNIQUE INDEX IF NOT EXISTS ux_one_open_autonomy_lease
ON autonomy_leases (workflow_id)
WHERE status IN ('pending', 'active', 'paused');
CREATE UNIQUE INDEX IF NOT EXISTS ux_measurement_session_workflow
ON measurement_sessions (workflow_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_decision_run
ON decisions (run_id);
CREATE INDEX IF NOT EXISTS ix_dashboard_pairings_instance_status
ON dashboard_pairings (instance_nonce, status, expires_at);
CREATE INDEX IF NOT EXISTS ix_dashboard_devices_instance_status
ON dashboard_devices (instance_nonce, status, expires_at);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._migrate(connection)
            connection.executescript(INDEXES)
            connection.execute(
                "INSERT INTO schema_meta(key, value) VALUES ('schema_version', '6') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """Apply additive migrations for runtime databases created by older JY builds."""
        for table, column, declaration in (
            # Operator choices made on the lease approval page (2026-10-04).
            ("workflows", "options_json", "TEXT"),
            ("proposals", "autonomy_lease_id", "TEXT"),
            ("runs", "autonomy_lease_id", "TEXT"),
            ("runs", "process_token", "TEXT"),
            ("runs", "stop_intent", "TEXT"),
            ("runs", "stop_requested_at", "TEXT"),
            ("runs", "exit_code", "INTEGER"),
            ("runs", "exit_receipt_path", "TEXT"),
            ("runs", "termination_cause", "TEXT"),
            # Operator request 2026-10-10: per-chip history and starting a
            # workflow at a node an earlier workflow on the same chip reached.
            # Evidence before that node is carried over as inherited runs.
            ("workflows", "device_key", "TEXT"),
            ("workflows", "source_workflow_id", "TEXT"),
            ("workflows", "start_node", "TEXT"),
            ("runs", "inherited_from_run_id", "TEXT"),
            # Operator decisions 2026-10-20: the approval page's full-auto
            # judgement choice, and why a workflow ended (an operator end
            # brings back the full lease page; a recovery end continues).
            ("autonomy_leases", "auto_judgement", "INTEGER NOT NULL DEFAULT 0"),
            ("workflows", "end_kind", "TEXT"),
        ):
            columns = {
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            if column not in columns:
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                )
                if (table, column) == ("workflows", "end_kind"):
                    # Workflows that ended before end_kind existed were ended
                    # by the operator as far as JY can tell, so a later
                    # Recover must not turn them into a continuation.
                    connection.execute(
                        "UPDATE workflows SET end_kind = 'operator' "
                        "WHERE status NOT IN ('active', 'paused')"
                    )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(query, params).fetchone()
        return dict(row) if row is not None else None

    def all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> int:
        with self.connect() as connection:
            cursor = connection.execute(query, params)
            return cursor.rowcount

    def event(
        self,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
        workflow_id: str | None = None,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        query = (
            "INSERT INTO events(created_at, event_type, workflow_id, actor, payload_json) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        params = (utc_now(), event_type, workflow_id, actor, json_dumps(payload))
        if connection is not None:
            connection.execute(query, params)
            return
        self.execute(
            query,
            params,
        )
