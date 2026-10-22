"""Per-chip history and starting a workflow at a node reached earlier.

Operator request 2026-10-10: a new workflow may start from the beginning or
at any node the chip's latest workflow reached. The evidence before that node
is carried over; starting after 02b keeps the earlier flux choice; the
approval page previews the nodes before the approval code appears.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from jy_agent.service import AgentService, ServiceError
from jy_agent.util import json_dumps, utc_now

from test_autonomy import AUTO_PHRASE
from test_core import make_settings, sample_state


def flux_service(folder: str) -> AgentService:
    """A service whose sequence includes the flux nodes."""
    settings = replace(
        make_settings(Path(folder), sample_state(0.2, 0.1)),
        workflow_sequence=("02x", "02c", "02a", "02b", "03a", "03b", "04", "05"),
    )
    return AgentService(settings)


def add_run(service: AgentService, workflow_id: str, node: str, *, advance: bool) -> str:
    """A completed, analyzed run of `node` with a decision."""

    now = utc_now()
    proposal_id = uuid.uuid4().hex
    run_id = uuid.uuid4().hex
    service.db.execute(
        "INSERT INTO proposals(id, workflow_id, kind, payload_json, status, created_at, "
        "expires_at, max_uses, uses, source_client) VALUES (?, ?, 'run', '{}', "
        "'approved', ?, ?, 1, 1, 'test')",
        (proposal_id, workflow_id, now, now),
    )
    analysis = {"analysis_status": "pass", "passing_targets": ["q1"]}
    service.db.execute(
        "INSERT INTO runs(id, workflow_id, proposal_id, node_id, parameters_json, status, "
        "started_at, analysis_status, analysis_json) VALUES (?, ?, ?, ?, ?, 'completed', ?, "
        "'pass', ?)",
        (run_id, workflow_id, proposal_id, node, json_dumps({"qubits": ["q1"]}), now,
         json_dumps(analysis)),
    )
    service.db.execute(
        "INSERT INTO decisions(id, workflow_id, run_id, decision, reason, next_node, "
        "next_parameters_json, state_patch_json, client_id, created_at) VALUES "
        "(?, ?, ?, ?, 'test', NULL, '{}', '[]', 'test', ?)",
        (uuid.uuid4().hex, workflow_id, run_id, "advance" if advance else "repeat", now),
    )
    return run_id


class ChipHistoryStartNodeTests(unittest.TestCase):
    def earlier_workflow(self, service: AgentService, *, flux: bool) -> str:
        entered = service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE, ["q1"])
        workflow_id = entered["workflow"]["id"]
        for node in ("02x", "02c", "02a", "03a"):
            add_run(service, workflow_id, node, advance=True)
        add_run(service, workflow_id, "04", advance=False)
        service.db.execute(
            "UPDATE workflows SET status = 'stopped', current_node = '04', options_json = ? "
            "WHERE id = ?",
            (json_dumps({"flux_calibration": flux, "drive_mode": "multiplex",
                         "excluded_nodes": []}), workflow_id),
        )
        service.db.event(
            "autonomy_targets_scientifically_unmeasurable",
            "test",
            {"node_id": "02c", "targets": ["q1"], "lease_id": "old"},
            workflow_id,
        )
        service.db.execute(
            "UPDATE autonomy_leases SET status = 'revoked' WHERE workflow_id = ?",
            (workflow_id,),
        )
        service.db.execute(
            "UPDATE measurement_sessions SET status = 'stopped' WHERE workflow_id = ?",
            (workflow_id,),
        )
        return workflow_id

    def new_entry(self, service: AgentService) -> dict:
        entered = service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE, ["q1"])
        proposal = service.proposal(entered["approval"]["id"])
        return {"entered": entered, "proposal": proposal}

    def test_history_offers_the_nodes_the_last_workflow_reached(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(make_settings(Path(folder), sample_state(0.2, 0.1)))
            source = self.earlier_workflow(service, flux=False)
            fresh = self.new_entry(service)
            history = fresh["proposal"]["payload"]["workflow_options_offer"]["start_node"]
            self.assertEqual(history["source_workflow_id"], source)
            self.assertEqual(history["reached_node"], "04")
            self.assertEqual(history["start_nodes"], ["02c", "02a", "03a", "04"])
            nodes = {item["node"]: item for item in history["nodes"]}
            self.assertTrue(nodes["03a"]["finished"])
            self.assertFalse(nodes["04"]["finished"])
            self.assertFalse(history["flux_calibration"])

    def test_start_at_04_inherits_earlier_evidence_and_keeps_flux(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = flux_service(folder)
            source = self.earlier_workflow(service, flux=False)
            fresh = self.new_entry(service)
            proposal_id = fresh["proposal"]["id"]
            workflow_id = fresh["entered"]["workflow"]["id"]
            choice = {
                "optional_nodes": [],
                "drive_mode": "multiplex",
                "flux_calibration": None,
                "node_parameters": {},
                "start_node": "04",
            }
            preview = service.preview_lease_options(proposal_id, choice)
            statuses = {item["node"]: item["status"] for item in preview["nodes"]}
            self.assertEqual(statuses["03a"], "inherited")
            self.assertEqual(statuses["04"], "run")
            self.assertEqual(statuses["03b"], "skipped_flux")
            # The preview changes nothing.
            self.assertEqual(service.proposal(proposal_id)["status"], "pending")
            self.assertEqual(
                service.db.one("SELECT COUNT(*) AS n FROM runs WHERE workflow_id = ?",
                               (workflow_id,))["n"],
                0,
            )
            with self.assertRaisesRegex(ServiceError, "flux"):
                service._approve_pending_proposal(
                    proposal_id, "human", "unit_test",
                    lease_options={**choice, "flux_calibration": True},
                )
            service._approve_pending_proposal(
                proposal_id, "human", "unit_test", lease_options=choice
            )
            workflow = service.db.one("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
            self.assertEqual(workflow["current_node"], "04")
            self.assertEqual(workflow["start_node"], "04")
            self.assertEqual(workflow["source_workflow_id"], source)
            self.assertIs(json.loads(workflow["options_json"])["flux_calibration"], False)
            inherited = service.db.all(
                "SELECT node_id FROM runs WHERE workflow_id = ? "
                "AND inherited_from_run_id IS NOT NULL ORDER BY rowid",
                (workflow_id,),
            )
            self.assertEqual([row["node_id"] for row in inherited], ["02x", "02c", "02a", "03a"])
            carried = service.db.all(
                "SELECT d.decision FROM decisions d JOIN runs r ON r.id = d.run_id "
                "WHERE r.workflow_id = ? AND r.inherited_from_run_id IS NOT NULL",
                (workflow_id,),
            )
            self.assertEqual([row["decision"] for row in carried], ["advance"] * 4)
            # The source's unmeasurable mark before the start node comes along.
            self.assertEqual(
                service._scientifically_unmeasurable_targets_by_node(workflow_id),
                {"02c": {"q1"}},
            )
            lease = service.db.one(
                "SELECT allowed_nodes_json FROM autonomy_leases WHERE proposal_id = ?",
                (proposal_id,),
            )
            self.assertEqual(json.loads(lease["allowed_nodes_json"])[0], "04")
            session = fresh["entered"]["dashboard"]["id"]
            self.assertEqual(service.dashboard_review(session)["history"], [])

    def test_unknown_start_node_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(make_settings(Path(folder), sample_state(0.2, 0.1)))
            self.earlier_workflow(service, flux=False)
            fresh = self.new_entry(service)
            with self.assertRaisesRegex(ServiceError, "reached"):
                service._approve_pending_proposal(
                    fresh["proposal"]["id"], "human", "unit_test",
                    lease_options={"optional_nodes": [], "drive_mode": "multiplex",
                                   "start_node": "06"},
                )


class SkippedBroadbandTests(unittest.TestCase):
    """Operator choice 2026-10-11: a later start needs no 02 when the unset-IF
    qubits were already found absent at 02; otherwise only the beginning."""

    def service(self, folder: str) -> AgentService:
        state = sample_state(0.2, 0.1)
        state["qubits"]["q1"]["resonator"]["intermediate_frequency"] = 0
        settings = replace(
            make_settings(Path(folder), state),
            workflow_sequence=("02", "02x", "02c", "02a", "03a", "04"),
        )
        return AgentService(settings)

    def source(self, service: AgentService, *, absent_at_02: bool) -> str:
        entered = service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE, ["q1"])
        workflow_id = entered["workflow"]["id"]
        for node in ("02", "02x", "02c", "02a"):
            add_run(service, workflow_id, node, advance=True)
        if absent_at_02:
            service.db.event(
                "autonomy_targets_scientifically_unmeasurable",
                "test",
                {"node_id": "02", "targets": ["q1"], "lease_id": "old"},
                workflow_id,
            )
        service.db.execute(
            "UPDATE workflows SET status = 'stopped', current_node = '03a', options_json = ? "
            "WHERE id = ?",
            (json_dumps({"excluded_nodes": [], "drive_mode": "multiplex",
                         "node_parameters": {"02": {"res_num": 1, "res_design_freq": [5.6]}}}),
             workflow_id),
        )
        service.db.execute(
            "UPDATE autonomy_leases SET status = 'revoked' WHERE workflow_id = ?",
            (workflow_id,),
        )
        service.db.execute(
            "UPDATE measurement_sessions SET status = 'stopped' WHERE workflow_id = ?",
            (workflow_id,),
        )
        return workflow_id

    def offer(self, service: AgentService) -> dict:
        entered = service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE, ["q1"])
        proposal = service.proposal(entered["approval"]["id"])
        return proposal

    def test_absent_unset_qubit_starts_only_from_the_beginning(self) -> None:
        """2026-10-15: the per-qubit plan replaced the single later start.

        The chip history still records that q1 was found absent at 02 (so the
        old single start was not blocked), but a qubit found absent at 02 has
        no record to resume from: the plan offers it only "not measured" or
        "from the beginning", and from the beginning its unset IF needs 02.
        """

        with tempfile.TemporaryDirectory() as folder:
            service = self.service(folder)
            self.source(service, absent_at_02=True)
            proposal = self.offer(service)
            offer = proposal["payload"]["workflow_options_offer"]
            self.assertEqual(offer["start_node"]["start_blocked_by"], [])
            self.assertIn("02x", offer["start_node"]["start_nodes"])
            self.assertEqual(offer["qubit_plan"]["qubits"], {"q1": ["skip", "beginning"]})
            broadband = next(item for item in offer["optional_nodes"] if item["node"] == "02")
            self.assertTrue(broadband.get("required"))
            # The last 02 parameters prefill a fresh start.
            self.assertEqual(broadband["parameters"], {"res_num": 1, "res_design_freq": [5.6]})
            from jy_agent.approval_web import _lease_options_from_fields

            parsed = _lease_options_from_fields(
                {
                    "optional_node": ["02"],
                    "drive_mode": ["multiplex"],
                    # A stray single start point is ignored next to the plan.
                    "start_node": ["02x"],
                    "qubit_plan__q1": ["beginning"],
                    "node02_res_num": ["1"],
                    "node02_res_design_freq": ["5.6"],
                },
                proposal,
            )
            self.assertEqual(parsed["optional_nodes"], ["02"])
            self.assertEqual(
                parsed["node_parameters"], {"02": {"res_num": 1, "res_design_freq": [5.6]}}
            )
            with self.assertRaisesRegex(ServiceError, "cannot start at 02x"):
                service.preview_lease_options(
                    proposal["id"], {**parsed, "qubit_plan": {"q1": "02x"}}
                )
            with self.assertRaisesRegex(ServiceError, "02 is required"):
                service.preview_lease_options(
                    proposal["id"], {**parsed, "optional_nodes": [], "node_parameters": {}}
                )
            service._approve_pending_proposal(
                proposal["id"], "human", "unit_test", lease_options=parsed
            )
            workflow = service.db.one(
                "SELECT current_node FROM workflows WHERE id = ?", (proposal["workflow_id"],)
            )
            self.assertEqual(workflow["current_node"], "02")

    def test_a_skipped_unset_qubit_does_not_force_02(self) -> None:
        """Only the qubits the plan measures can make 02 required."""

        from copy import deepcopy

        from test_core import sample_multiplex_state_and_wiring
        from jy_agent.util import atomic_write_json

        state, wiring, _targets = sample_multiplex_state_and_wiring()
        keep = ["q3", "q4"]
        state["qubits"] = {name: deepcopy(state["qubits"][name]) for name in keep}
        state["qubits"]["q3"]["resonator"]["intermediate_frequency"] = 0
        state["active_qubit_names"] = keep
        wiring["wiring"]["qubits"] = {name: wiring["wiring"]["qubits"][name] for name in keep}
        with tempfile.TemporaryDirectory() as folder:
            settings = replace(
                make_settings(Path(folder), state),
                workflow_sequence=("02", "02x", "02c", "02a", "03a", "04"),
            )
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            entered = service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE)
            proposal = service.proposal(entered["approval"]["id"])
            broadband = next(
                item
                for item in proposal["payload"]["workflow_options_offer"]["optional_nodes"]
                if item["node"] == "02"
            )
            self.assertEqual(broadband["required_because"], ["q3"])
            base = {"optional_nodes": [], "drive_mode": "multiplex", "node_parameters": {}}
            with self.assertRaisesRegex(ServiceError, r"02 is required: readout IFs of \['q3'\]"):
                service.preview_lease_options(
                    proposal["id"], {**base, "qubit_plan": {"q3": "beginning", "q4": "beginning"}}
                )
            service._approve_pending_proposal(
                proposal["id"],
                "human",
                "unit_test",
                lease_options={**base, "qubit_plan": {"q3": "skip", "q4": "beginning"}},
            )
            workflow = service.db.one(
                "SELECT current_node, targets_json FROM workflows WHERE id = ?",
                (proposal["workflow_id"],),
            )
            self.assertEqual(json.loads(workflow["targets_json"]), ["q4"])
            self.assertEqual(workflow["current_node"], "02x")

    def test_unset_qubit_not_found_absent_forces_the_beginning(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = self.service(folder)
            self.source(service, absent_at_02=False)
            proposal = self.offer(service)
            history = proposal["payload"]["workflow_options_offer"]["start_node"]
            self.assertEqual(history["start_blocked_by"], ["q1"])
            self.assertEqual(history["start_nodes"], [])
            with self.assertRaisesRegex(ServiceError, "from the beginning"):
                service._approve_pending_proposal(
                    proposal["id"], "human", "unit_test",
                    lease_options={"optional_nodes": [], "drive_mode": "multiplex",
                                   "start_node": "02x"},
                )


if __name__ == "__main__":
    unittest.main()
