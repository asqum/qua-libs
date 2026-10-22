"""The lease page picks the qubits and where each one starts (operator 2026-10-15).

Each qubit is not measured, starts from the beginning, or resumes at a node
it reached in the chip's latest workflow. The workflow starts at the earliest
choice; a qubit resuming later carries only its own earlier results, so it
waits until the workflow reaches its node.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

from jy_agent.approval_web import _lease_options_from_fields
from jy_agent.service import AgentService, ServiceError, _restrict_to_qubit
from jy_agent.util import json_dumps, utc_now

from jy_agent.util import atomic_write_json

from test_autonomy import AUTO_PHRASE
from test_core import make_settings
from test_shared_xy_drive import chip_state, chip_wiring


def add_run(service: AgentService, workflow_id: str, node: str, qubits: list[str],
            passing: list[str], *, advance: bool) -> str:
    now = utc_now()
    proposal_id, run_id = uuid.uuid4().hex, uuid.uuid4().hex
    service.db.execute(
        "INSERT INTO proposals(id, workflow_id, kind, payload_json, status, created_at, "
        "expires_at, max_uses, uses, source_client) VALUES (?, ?, 'run', '{}', "
        "'approved', ?, ?, 1, 1, 'test')",
        (proposal_id, workflow_id, now, now),
    )
    analysis = {
        "analysis_status": "pass" if passing == qubits else "needs_review",
        "passing_targets": passing,
        "fit_quality": {"results": {name: {"fit_successful": name in passing} for name in qubits}},
    }
    service.db.execute(
        "INSERT INTO runs(id, workflow_id, proposal_id, node_id, parameters_json, status, "
        "started_at, analysis_status, analysis_json) VALUES (?, ?, ?, ?, ?, 'completed', ?, ?, ?)",
        (run_id, workflow_id, proposal_id, node, json_dumps({"qubits": qubits}), now,
         analysis["analysis_status"], json_dumps(analysis)),
    )
    service.db.execute(
        "INSERT INTO decisions(id, workflow_id, run_id, decision, reason, next_node, "
        "next_parameters_json, state_patch_json, client_id, created_at) VALUES "
        "(?, ?, ?, ?, 'test', NULL, '{}', '[]', 'test', ?)",
        (uuid.uuid4().hex, workflow_id, run_id, "advance" if advance else "repeat", now),
    )
    return run_id


class QubitPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        settings = make_settings(Path(self.folder.name), chip_state(["q1", "q2"]))
        atomic_write_json(settings.wiring_path, chip_wiring(False, ["q1", "q2"]))
        self.service = AgentService(settings)
        # The real per-node evidence rules need SNR metrics; here a run's
        # passing_targets stand for its evidence.
        self.service._resolve_targets_from_rows = lambda node, rows: set().union(
            set(), *(set(json.loads(row["analysis_json"] or "{}").get("passing_targets") or [])
                     for row in rows)
        )
        entered = self.service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE)
        source = entered["workflow"]["id"]
        both = ["q1", "q2"]
        add_run(self.service, source, "02x", both, both, advance=True)
        add_run(self.service, source, "02c", both, ["q1"], advance=True)
        add_run(self.service, source, "02a", ["q1"], ["q1"], advance=True)
        self.service.db.execute(
            "UPDATE workflows SET status = 'stopped', current_node = '02a', options_json = ? "
            "WHERE id = ?",
            (json_dumps({"flux_calibration": False, "drive_mode": "multiplex",
                         "excluded_nodes": []}), source),
        )
        self.service.db.execute(
            "UPDATE autonomy_leases SET status = 'revoked' WHERE workflow_id = ?", (source,)
        )
        self.service.db.execute(
            "UPDATE measurement_sessions SET status = 'stopped' WHERE workflow_id = ?", (source,)
        )
        self.source = source
        entered = self.service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE)
        self.proposal = self.service.proposal(entered["approval"]["id"])
        self.workflow_id = entered["workflow"]["id"]

    def tearDown(self) -> None:
        self.folder.cleanup()

    def offer(self) -> dict:
        return self.proposal["payload"]["workflow_options_offer"]["qubit_plan"]

    def choice(self, plan: dict) -> dict:
        return {"optional_nodes": [], "drive_mode": "multiplex", "flux_calibration": None,
                "node_parameters": {}, "qubit_plan": plan}

    def test_each_qubit_gets_its_own_choices(self) -> None:
        qubits = self.offer()["qubits"]
        # q1 cleared 02x-02a: it may resume at any of them after 02x (the
        # beginning). q2 failed 02c, the first node it did not clear, so 02c
        # is its last offered node.
        self.assertEqual(qubits["q1"], ["skip", "beginning", "02c", "02a"])
        self.assertEqual(qubits["q2"], ["skip", "beginning", "02c"])

    def test_all_skipped_is_refused(self) -> None:
        with self.assertRaisesRegex(ServiceError, "at least one qubit"):
            self.service.preview_lease_options(
                self.proposal["id"], self.choice({"q1": "skip", "q2": "skip"})
            )

    def test_skip_narrows_the_targets(self) -> None:
        self.service._approve_pending_proposal(
            self.proposal["id"], "human", "unit_test",
            lease_options=self.choice({"q1": "beginning", "q2": "skip"}),
        )
        workflow = self.service.db.one("SELECT * FROM workflows WHERE id = ?", (self.workflow_id,))
        self.assertEqual(json.loads(workflow["targets_json"]), ["q1"])
        lease = self.service.db.one(
            "SELECT targets_json FROM autonomy_leases WHERE proposal_id = ?", (self.proposal["id"],)
        )
        self.assertEqual(json.loads(lease["targets_json"]), ["q1"])
        self.assertEqual(workflow["current_node"], "02x")

    def test_a_later_qubit_carries_only_its_own_results(self) -> None:
        self.service._approve_pending_proposal(
            self.proposal["id"], "human", "unit_test",
            lease_options=self.choice({"q1": "02a", "q2": "beginning"}),
        )
        workflow = self.service.db.one("SELECT * FROM workflows WHERE id = ?", (self.workflow_id,))
        self.assertEqual(workflow["current_node"], "02x")
        rows = self.service.db.all(
            "SELECT node_id, parameters_json, analysis_json FROM runs WHERE workflow_id = ? "
            "AND inherited_from_run_id IS NOT NULL", (self.workflow_id,),
        )
        self.assertEqual(sorted(row["node_id"] for row in rows), ["02c", "02x"])
        for row in rows:
            self.assertEqual(json.loads(row["parameters_json"])["qubits"], ["q1"])
            analysis = json.loads(row["analysis_json"])
            self.assertEqual(analysis["passing_targets"], ["q1"])
            self.assertEqual(list(analysis["fit_quality"]["results"]), ["q1"])
        # q1 already cleared 02x and 02c here, so those measure q2 alone.
        self.assertEqual(self.service._node_target_resolution(self.workflow_id, "02x"), {"q1"})
        self.assertEqual(self.service._node_target_resolution(self.workflow_id, "02c"), {"q1"})
        preview_targets = json.loads(workflow["targets_json"])
        self.assertEqual(preview_targets, ["q1", "q2"])

    def test_form_set_all_row(self) -> None:
        options = _lease_options_from_fields(
            {"qubit_plan__all": ["skip"], "qubit_plan__q1": ["beginning"],
             "qubit_plan__q2": ["beginning"], "drive_mode": ["multiplex"]},
            self.proposal,
        )
        self.assertEqual(options["qubit_plan"], {"q1": "skip", "q2": "skip"})
        options = _lease_options_from_fields(
            {"qubit_plan__all": [""], "qubit_plan__q1": ["skip"], "drive_mode": ["multiplex"]},
            self.proposal,
        )
        self.assertEqual(options["qubit_plan"], {"q1": "skip", "q2": "beginning"})

    def test_restrict_to_qubit(self) -> None:
        restricted = _restrict_to_qubit(
            {"passing_targets": ["q1", "q2"], "fit_quality": {"results": {"q1": 1, "q2": 2}},
             "plots": ["a.png"]},
            "q2",
        )
        self.assertEqual(
            restricted,
            {"passing_targets": ["q2"], "fit_quality": {"results": {"q2": 2}}, "plots": ["a.png"]},
        )


if __name__ == "__main__":
    unittest.main()
