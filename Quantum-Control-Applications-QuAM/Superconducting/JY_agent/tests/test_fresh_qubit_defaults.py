"""Fresh-qubit defaults (operator instruction 2026-10-14).

A qubit with no JY run on the chip config.toml names is measured from scratch:
before its first run its extras are emptied and starting values written. The
x180 gate is 400 ns on a shared XY line and 40 ns on a private one.
"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from jy_agent.service import AgentService, AutonomyScopeError
from jy_agent.state import load_state
from jy_agent.util import atomic_write_json, json_dumps

from test_core import make_settings, start_test_workflow
from test_shared_xy_drive import TARGETS, chip_state, chip_wiring


def build(folder: str, shared: bool, state: dict | None = None):
    state = state or chip_state()
    for qubit in state["qubits"].values():
        qubit["extras"] = {"idle_freq": 4.2e9, "T1": 3e-5, "readout_fidelity": 0.9}
        qubit["xy"]["operations"]["x180_DragCosine"].update(length=800, alpha=-0.8)
        qubit["xy"]["operations"]["x90_DragCosine"].update(
            length="#../x180_DragCosine/length", detuning=-3e5
        )
    settings = dataclasses.replace(
        make_settings(Path(folder), state), fresh_qubit_defaults=True
    )
    atomic_write_json(settings.wiring_path, chip_wiring(shared))
    service = AgentService(settings)
    workflow = start_test_workflow(service, list(TARGETS))
    return service, workflow


def apply(service: AgentService, proposal: dict) -> None:
    service._approve_pending_proposal(proposal["id"], "unit-test-human", "local_browser")
    service.apply_state_commit(proposal["id"], "unittest")


class FreshQubitDefaultsTests(unittest.TestCase):
    def test_new_chip_needs_the_defaults_before_any_run(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = build(folder, shared=False)
            self.assertEqual(service._fresh_qubits(service._workflow(workflow["id"])), TARGETS)
            plan = service.next_action(workflow["id"])
            self.assertEqual(
                plan["required_setup"][0]["tool"], "jy_request_fresh_qubit_defaults"
            )
            with self.assertRaisesRegex(AutonomyScopeError, "fresh_qubit_defaults"):
                service.request_run(
                    workflow["id"], workflow["current_node"], {"qubits": TARGETS},
                    "First run.", "unittest",
                )

    def test_private_line_defaults_are_written_and_clear_the_need(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = build(folder, shared=False)
            proposal = service.request_fresh_qubit_defaults(workflow["id"], "unittest")
            apply(service, proposal)
            qubit = load_state(service.settings.active_state)["qubits"]["q1"]
            operations = qubit["xy"]["operations"]
            self.assertEqual(qubit["extras"], {})
            self.assertEqual(qubit["T1"], 1e-4)
            self.assertEqual(qubit["T2ramsey"], 1e-4)
            self.assertEqual(qubit["T2echo"], 1e-4)
            self.assertEqual(qubit["anharmonicity"], 2e8)
            self.assertEqual(operations["x180_DragCosine"]["length"], 40)
            self.assertEqual(operations["x180_DragCosine"]["amplitude"], 0.05)
            self.assertEqual(operations["x180_DragCosine"]["alpha"], 0.0)
            self.assertEqual(operations["x90_DragCosine"]["amplitude"], 0.025)
            self.assertEqual(operations["x90_DragCosine"]["detuning"], 0.0)
            # A reference keeps following its target.
            self.assertEqual(operations["x90_DragCosine"]["length"], "#../x180_DragCosine/length")
            self.assertEqual(operations["saturation"]["amplitude"], 0.5)
            self.assertEqual(operations["saturation"]["length"], 20000)
            readout = qubit["resonator"]["operations"]["readout"]
            self.assertEqual(readout["length"], 2000)
            self.assertEqual(readout["threshold"], 0.0)
            self.assertEqual(readout["integration_weights_angle"], 0.0)
            self.assertEqual(qubit["resonator"]["depletion_time"], 2000)
            self.assertEqual(qubit["resonator"]["confusion_matrix"], [[1.0, 0.0], [0.0, 1.0]])
            self.assertEqual(service._fresh_qubits(service._workflow(workflow["id"])), [])
            with self.assertRaisesRegex(Exception, "measurement record"):
                service.request_fresh_qubit_defaults(workflow["id"], "unittest")

    def test_shared_line_gets_the_long_gate(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = build(folder, shared=True)
            proposal = service.request_fresh_qubit_defaults(workflow["id"], "unittest")
            self.assertEqual(proposal["fresh_qubit_defaults"]["q1"]["x180_length_ns"], 400)
            self.assertEqual(proposal["fresh_qubit_defaults"]["q1"]["xy_line"], "shared")

    def test_a_qubit_with_a_run_on_this_chip_keeps_its_state(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = build(folder, shared=False)
            service.db.execute(
                "INSERT INTO proposals(id, workflow_id, kind, payload_json, status, "
                "created_at, expires_at, max_uses, source_client) "
                "VALUES ('p', ?, 'run', '{}', 'consumed', '', '', 1, 'unittest')",
                (workflow["id"],),
            )
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
                "parameters_json, status) VALUES (?, ?, ?, '02x', ?, 'failed')",
                ("old-run", workflow["id"], "p", json_dumps({"qubits": ["q1", "q2"]})),
            )
            self.assertEqual(service._fresh_qubits(service._workflow(workflow["id"])), ["q3"])
            proposal = service.request_fresh_qubit_defaults(workflow["id"], "unittest")
            touched = {item["path"].split("/")[2] for item in proposal["payload"]["patch"]}
            self.assertEqual(touched, {"q3"})

    def test_extras_may_only_be_emptied(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = build(folder, shared=False)
            with self.assertRaisesRegex(Exception, "may only be emptied"):
                service.request_state_commit(
                    workflow["id"],
                    [{"op": "add", "path": "/qubits/q1/extras", "value": {"T1": 1.0}}],
                    "Overwrite extras.",
                    "unittest",
                )


if __name__ == "__main__":
    unittest.main()
