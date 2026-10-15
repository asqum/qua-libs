"""Readout length around 03a (operator instruction 2026-10-15).

Before 03a every target reads out for 10000 ns; a qubit's first 03a drops it
to 4000 ns; after that only the 10000 ns cap applies, so 07d's choice stays.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jy_agent.service import AgentService, AutonomyScopeError
from jy_agent.util import json_dumps

from test_core import make_settings, sample_state, start_test_workflow


def build(folder: str, length: int):
    state = sample_state(0.2, 0.1)
    state["qubits"]["q1"]["resonator"]["operations"]["readout"]["length"] = length
    service = AgentService(make_settings(Path(folder), state))
    return service, start_test_workflow(service)


def at(service: AgentService, workflow: dict, node: str) -> dict:
    service.db.execute(
        "UPDATE workflows SET current_node = ? WHERE id = ?", (node, workflow["id"])
    )
    return service._workflow(workflow["id"])


def plan(service: AgentService, workflow: dict) -> dict:
    return service._readout_length_normalize_patch(workflow)[1]


class ReadoutLengthStageTests(unittest.TestCase):
    def test_before_03a_every_target_reads_out_for_10000_ns(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = build(folder, 1300)
            workflow = at(service, workflow, "02c")
            self.assertEqual(
                plan(service, workflow)["q1"], {"length_ns": [1300, 10000], "reason": "pre_03a"}
            )
            required = service._required_setup_for_node("02c", ["q1"], workflow["id"])
            self.assertIn(
                "jy_request_readout_length_normalize", [item["tool"] for item in required]
            )
            with self.assertRaisesRegex(AutonomyScopeError, "10000/4000 ns"):
                service.request_run(workflow["id"], "02c", {"qubits": ["q1"]}, "Run.", "unittest")

    def test_first_03a_drops_10000_to_4000(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = build(folder, 10000)
            workflow = at(service, workflow, "03a")
            self.assertEqual(
                plan(service, workflow)["q1"], {"length_ns": [10000, 4000], "reason": "first_03a"}
            )

    def test_after_03a_only_the_cap_applies(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = build(folder, 10000)
            workflow = at(service, workflow, "03a")
            service.db.execute(
                "INSERT INTO proposals(id, workflow_id, kind, payload_json, status, "
                "created_at, expires_at, max_uses, source_client) "
                "VALUES ('p', ?, 'run', '{}', 'consumed', '', '', 1, 'unittest')",
                (workflow["id"],),
            )
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, parameters_json, "
                "status) VALUES ('r', ?, 'p', '03a', ?, 'completed')",
                (workflow["id"], json_dumps({"qubits": ["q1"]})),
            )
            self.assertEqual(plan(service, workflow), {})
            workflow = at(service, workflow, "05")
            self.assertEqual(plan(service, workflow), {})

    def test_the_cap_is_10000_ns(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = build(folder, 20000)
            workflow = at(service, workflow, "05")
            self.assertEqual(
                plan(service, workflow)["q1"], {"length_ns": [20000, 10000], "reason": "cap"}
            )


if __name__ == "__main__":
    unittest.main()
