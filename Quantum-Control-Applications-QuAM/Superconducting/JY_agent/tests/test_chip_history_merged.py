"""Chip history merged across workflows (operator decision 2026-10-21).

The Approval-page table is node x qubit across every earlier workflow of the
chip (each cell the qubit's most recent pass), each qubit resumes from the
workflow holding its most recent result and inherits from there, a workflow
with runs of its own sees its passes in the table, and a state.json change
limits only the qubit whose committed values it touched.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

from jy_agent.approval_web import _chip_progress_section, _lease_options_fields
from jy_agent.service import AgentService
from jy_agent.util import atomic_write_json, json_dumps

from test_autonomy import AUTO_PHRASE
from test_core import make_settings
from test_shared_xy_drive import chip_state, chip_wiring

QUBITS = ["q1", "q2", "q3"]


def add_run(
    service: AgentService,
    workflow_id: str,
    node: str,
    qubits: list[str],
    when: str,
    *,
    inherited_from: str | None = None,
) -> str:
    """A completed, passing run of `node` dated `when`."""

    proposal_id, run_id = uuid.uuid4().hex, uuid.uuid4().hex
    service.db.execute(
        "INSERT INTO proposals(id, workflow_id, kind, payload_json, status, created_at, "
        "expires_at, max_uses, uses, source_client) VALUES (?, ?, 'run', '{}', "
        "'approved', ?, ?, 1, 1, 'test')",
        (proposal_id, workflow_id, when, when),
    )
    analysis = {"analysis_status": "pass", "passing_targets": qubits}
    service.db.execute(
        "INSERT INTO runs(id, workflow_id, proposal_id, node_id, parameters_json, status, "
        "started_at, analysis_status, analysis_json, inherited_from_run_id) "
        "VALUES (?, ?, ?, ?, ?, 'completed', ?, 'pass', ?, ?)",
        (run_id, workflow_id, proposal_id, node, json_dumps({"qubits": qubits}), when,
         json_dumps(analysis), inherited_from),
    )
    service.db.execute(
        "INSERT INTO decisions(id, workflow_id, run_id, decision, reason, next_node, "
        "next_parameters_json, state_patch_json, client_id, created_at) VALUES "
        "(?, ?, ?, 'advance', 'test', NULL, '{}', '[]', 'test', ?)",
        (uuid.uuid4().hex, workflow_id, run_id, when),
    )
    return run_id


def commit(service: AgentService, workflow_id: str, run_id: str, path: str, value) -> None:
    """A state commit of `run_id`'s result writing `value` at `path`."""

    proposal_id = uuid.uuid4().hex
    service.db.execute(
        "INSERT INTO proposals(id, workflow_id, kind, payload_json, status, created_at, "
        "expires_at, max_uses, uses, source_client) VALUES (?, ?, 'state_commit', ?, "
        "'consumed', '2026-10-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00', 1, 1, 'test')",
        (proposal_id, workflow_id, json_dumps({"run_id": run_id})),
    )
    service.db.event(
        "state_committed",
        "test",
        {"proposal_id": proposal_id, "patch": [{"op": "replace", "path": path, "value": value}]},
        workflow_id,
    )


class MergedChipHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        settings = make_settings(Path(self.folder.name), chip_state(QUBITS))
        atomic_write_json(settings.wiring_path, chip_wiring(False, QUBITS))
        self.service = AgentService(settings)
        # A run's passing_targets stand for its evidence here.
        self.service._resolve_targets_from_rows = lambda node, rows: set().union(
            set(), *(set(json.loads(row["analysis_json"] or "{}").get("passing_targets") or [])
                     for row in rows)
        )
        service = self.service
        # W1 (oldest): every qubit through 02a; q1's pass to 05, q2's to 04.
        self.w1 = self.workflow("2026-10-01T00:00:00+00:00", "03a")
        for node in ("02x", "02c", "02a"):
            add_run(service, self.w1, node, QUBITS, "2026-10-01T01:00:00+00:00")
        add_run(service, self.w1, "03a", ["q1"], "2026-10-01T02:00:00+00:00")
        self.q1_04 = add_run(service, self.w1, "04", ["q1"], "2026-10-01T03:00:00+00:00")
        add_run(service, self.w1, "05", ["q1"], "2026-10-01T04:00:00+00:00")
        add_run(service, self.w1, "03a", ["q2"], "2026-10-01T05:00:00+00:00")
        add_run(service, self.w1, "04", ["q2"], "2026-10-01T06:00:00+00:00")
        # W2: q2 and q3 again; q2 to 03a, q3 to 04.
        self.w2 = self.workflow("2026-10-02T00:00:00+00:00", "03a")
        self.w2_runs = {}
        for node in ("02x", "02c", "02a"):
            self.w2_runs[node] = add_run(
                service, self.w2, node, ["q2", "q3"], "2026-10-02T01:00:00+00:00"
            )
        self.q2_03a = add_run(service, self.w2, "03a", ["q2"], "2026-10-02T02:00:00+00:00")
        add_run(service, self.w2, "03a", ["q3"], "2026-10-02T03:00:00+00:00")
        add_run(service, self.w2, "04", ["q3"], "2026-10-02T04:00:00+00:00")
        # W3 (latest): q3 only, its 02x-02a inherited from W2, then 03a again.
        self.w3 = self.workflow("2026-10-03T00:00:00+00:00", "03a")
        for node in ("02x", "02c", "02a"):
            add_run(service, self.w3, node, ["q2", "q3"], "2026-10-02T01:00:00+00:00",
                    inherited_from=self.w2_runs[node])
        add_run(service, self.w3, "03a", ["q3"], "2026-10-03T01:00:00+00:00")

    def tearDown(self) -> None:
        self.folder.cleanup()

    def workflow(self, created: str, current: str) -> str:
        entered = self.service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE, QUBITS)
        workflow_id = entered["workflow"]["id"]
        self.service.db.execute(
            "UPDATE workflows SET status = 'stopped', current_node = ?, created_at = ?, "
            "options_json = ? WHERE id = ?",
            (current, created, json_dumps({"drive_mode": "per_qubit", "excluded_nodes": []}),
             workflow_id),
        )
        self.service.db.execute(
            "UPDATE autonomy_leases SET status = 'revoked' WHERE workflow_id = ?", (workflow_id,)
        )
        self.service.db.execute(
            "UPDATE measurement_sessions SET status = 'stopped' WHERE workflow_id = ?",
            (workflow_id,),
        )
        return workflow_id

    def fresh(self) -> dict:
        entered = self.service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE, QUBITS)
        return self.service.proposal(entered["approval"]["id"])

    def test_the_table_merges_every_workflow(self) -> None:
        offer = self.fresh()["payload"]["workflow_options_offer"]
        history = offer["start_node"]
        cells = {item["node"]: item["cells"] for item in history["merged"]["nodes"]}
        self.assertEqual(history["merged"]["qubits"], QUBITS)
        self.assertEqual(cells["05"]["q1"]["workflow_id"], self.w1)
        self.assertEqual(cells["04"]["q2"]["workflow_id"], self.w1)
        self.assertEqual(cells["03a"]["q2"]["workflow_id"], self.w2)
        self.assertEqual(cells["04"]["q3"]["workflow_id"], self.w2)
        self.assertEqual(cells["03a"]["q3"]["workflow_id"], self.w3)
        self.assertEqual(cells["03a"]["q3"]["date"], "2026-10-03T01:00:00+00:00")
        # The inherited copy in W3 ties with its original and ranks below it.
        self.assertEqual(cells["02c"]["q3"]["workflow_id"], self.w2)
        self.assertEqual(cells["02c"]["q1"]["workflow_id"], self.w1)
        # The single-workflow fields still describe the latest workflow.
        self.assertEqual(history["source_workflow_id"], self.w3)
        self.assertEqual(
            [item["workflow_id"] for item in history["workflows"]], [self.w3, self.w2, self.w1]
        )
        # W1's own table lists q1's 04 and 05 although W1 stood at 03a.
        w1_nodes = {item["node"]: item for item in history["workflows"][2]["nodes"]}
        self.assertEqual(w1_nodes["05"]["passed"], ["q1"])

    def test_each_qubit_resumes_from_its_own_most_recent_workflow(self) -> None:
        offer = self.fresh()["payload"]["workflow_options_offer"]
        history = offer["start_node"]
        self.assertEqual(
            history["qubit_sources"], {"q1": self.w1, "q2": self.w2, "q3": self.w3}
        )
        qubits = offer["qubit_plan"]["qubits"]
        # q1 reached 05 in its own pass in W1, the oldest workflow.
        self.assertEqual(qubits["q1"], ["skip", "beginning", "02c", "02a", "03a", "04", "05"])
        # q2's most recent result is W2's 03a, not W1's older 04.
        self.assertEqual(qubits["q2"], ["skip", "beginning", "02c", "02a", "03a"])
        self.assertEqual(qubits["q3"], ["skip", "beginning", "02c", "02a", "03a"])
        self.assertEqual(history["qubit_start_nodes"]["q1"], ["03a", "04", "05"])

    def test_inheritance_takes_each_qubit_from_its_source(self) -> None:
        # Marks follow the qubit's source too: W1's mark on q2 stays behind
        # (q2 comes from W2); W2's mark on q3 comes along with W2's runs.
        for workflow_id, qubit in ((self.w1, "q2"), (self.w2, "q3")):
            self.service.db.event(
                "autonomy_targets_scientifically_unmeasurable",
                "test",
                {"node_id": "02a", "targets": [qubit], "lease_id": "old"},
                workflow_id,
            )
        proposal = self.fresh()
        workflow_id = proposal["workflow_id"]
        self.service._approve_pending_proposal(
            proposal["id"], "human", "unit_test",
            lease_options={
                "optional_nodes": [], "drive_mode": "per_qubit", "flux_calibration": None,
                "node_parameters": {}, "qubit_plan": {"q1": "05", "q2": "03a", "q3": "skip"},
            },
        )
        workflow = self.service.db.one("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
        options = json.loads(workflow["options_json"])
        self.assertEqual(options["qubit_sources"], {"q1": self.w1, "q2": self.w2})
        # The most recent source is recorded for older readers.
        self.assertEqual(workflow["source_workflow_id"], self.w2)
        self.assertEqual(workflow["start_node"], "03a")
        rows = self.service.db.all(
            "SELECT r.node_id, r.parameters_json, o.workflow_id AS origin FROM runs r "
            "JOIN runs o ON o.id = r.inherited_from_run_id WHERE r.workflow_id = ? "
            "ORDER BY r.rowid",
            (workflow_id,),
        )
        copied = {
            (row["node_id"], tuple(json.loads(row["parameters_json"])["qubits"])): row["origin"]
            for row in rows
        }
        for node in ("02x", "02c", "02a"):
            self.assertEqual(copied[(node, ("q2", "q3"))], self.w2)
            self.assertEqual(copied[(node, ("q1",))], self.w1)
        self.assertEqual(copied[("03a", ("q1",))], self.w1)
        self.assertEqual(copied[("04", ("q1",))], self.w1)
        self.assertNotIn(("03a", ("q2",)), copied)
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            self.service._node_target_resolution(workflow_id, "02c"), {"q1", "q2", "q3"}
        )
        self.assertEqual(
            self.service._scientifically_unmeasurable_targets_by_node(workflow_id),
            {"02a": {"q3"}},
        )

    def test_a_workflow_with_runs_shows_its_own_passes(self) -> None:
        proposal = self.fresh()
        workflow_id = proposal["workflow_id"]
        add_run(self.service, workflow_id, "02x", ["q1"], "2026-10-04T01:00:00+00:00")
        workflow = self.service._workflow(workflow_id)
        history = self.service._chip_history(workflow)
        self.assertTrue(history["display_only"])
        self.assertEqual(history["current_workflow_id"], workflow_id)
        self.assertEqual(history["start_nodes"], [])
        self.assertEqual(history["qubit_start_nodes"], {})
        cells = {item["node"]: item["cells"] for item in history["merged"]["nodes"]}
        self.assertEqual(cells["02x"]["q1"]["workflow_id"], workflow_id)
        self.assertTrue(cells["02x"]["q1"]["current"])
        self.assertEqual(cells["05"]["q1"]["workflow_id"], self.w1)
        offer = self.service._lease_option_offer(self.service._workflow(workflow_id))
        self.assertNotIn("start_node", offer)
        self.assertNotIn("qubit_plan", offer)
        self.assertTrue(offer["chip_history"]["display_only"])
        html = _lease_options_fields(
            {"kind": "autonomy_lease", "payload": {"workflow_options_offer": offer}}, "en"
        )
        self.assertIn("chip-merged-table", html)
        self.assertNotIn('name="start_node"', html)
        home = _chip_progress_section(self.service, {"id": workflow_id}, "en")
        self.assertIn("chip-merged-table", home)
        self.assertIn(workflow_id[:8] + " *", home)

    def test_a_state_change_limits_only_that_qubit(self) -> None:
        # W1 committed q1's 04 result; state.json holds something else now.
        commit(self.service, self.w1, self.q1_04, "/qubits/q1/xy/intermediate_frequency", 123.0)
        # W2 committed q2's 03a result, which state.json still holds.
        state = json.loads(Path(self.service.settings.active_state).read_text(encoding="utf-8"))
        held = state["qubits"]["q2"]["xy"]["intermediate_frequency"]
        commit(self.service, self.w2, self.q2_03a, "/qubits/q2/xy/intermediate_frequency", held)
        offer = self.fresh()["payload"]["workflow_options_offer"]
        history = offer["start_node"]
        self.assertEqual(history["qubit_state_mismatch_node"], {"q1": "04"})
        self.assertEqual(
            history["qubit_state_matches"], {"q1": False, "q2": True, "q3": True}
        )
        self.assertFalse(history["state_matches"])
        qubits = offer["qubit_plan"]["qubits"]
        self.assertEqual(qubits["q1"], ["skip", "beginning", "02c", "02a", "03a", "04"])
        self.assertEqual(qubits["q2"], ["skip", "beginning", "02c", "02a", "03a"])
        self.assertEqual(history["qubit_start_nodes"]["q1"], ["03a", "04"])


class CompletedWorkflowNextActionTests(unittest.TestCase):
    def test_a_completed_workflow_reports_complete(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), chip_state(QUBITS))
            atomic_write_json(settings.wiring_path, chip_wiring(False, QUBITS))
            service = AgentService(settings)
            entered = service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE, QUBITS)
            workflow_id = entered["workflow"]["id"]
            service.db.execute(
                "UPDATE workflows SET status = 'completed', current_node = NULL WHERE id = ?",
                (workflow_id,),
            )
            result = service.next_action(workflow_id)
            self.assertEqual(result["action"], "complete")
            self.assertIsNone(result["current_node"])
            self.assertEqual(result["workflow_status"], "completed")


if __name__ == "__main__":
    unittest.main()
