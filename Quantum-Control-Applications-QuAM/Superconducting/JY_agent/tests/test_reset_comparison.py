"""Thermal repeat of an active-reset statistics run, operator request 2026-10-19.

After q1's statistics ran with active reset, 05st may run once more with
thermal reset to compare. The run does not move the workflow and commits
nothing.
"""

from __future__ import annotations

import tempfile
import unittest

from jy_agent.service import ServiceError
from jy_agent.util import json_dumps

from test_per_qubit_pass_rules import build, insert_run


def set_parameters(service, run_id: str, parameters: dict) -> None:
    service.db.execute(
        "UPDATE runs SET parameters_json = ? WHERE id = ?", (json_dumps(parameters), run_id)
    )


class ResetComparisonTests(unittest.TestCase):
    def test_comparison_needs_a_decided_active_run_of_the_pass_qubit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "06")
            qubit = service._serial_pass_state(workflow)["current"]
            thermal = {"qubits": [qubit], "reset_type": "thermal"}
            self.assertFalse(service._is_thermal_reset_comparison(workflow, "05st", thermal))

            run_id = insert_run(
                service, workflow["id"], "05st", qubit, {"analysis_status": "pass"},
                decision="advance", next_node="06",
            )
            set_parameters(service, run_id, {"qubits": [qubit], "reset_type": "active"})
            self.assertTrue(service._is_thermal_reset_comparison(workflow, "05st", thermal))
            self.assertFalse(
                service._is_thermal_reset_comparison(
                    workflow, "05st", {"qubits": [qubit], "reset_type": "active"}
                )
            )
            other = next(name for name in ("q1", "q2", "q3") if name != qubit)
            self.assertFalse(
                service._is_thermal_reset_comparison(
                    workflow, "05st", {"qubits": [other], "reset_type": "thermal"}
                )
            )
            self.assertFalse(service._is_thermal_reset_comparison(workflow, "04", thermal))

            # One comparison per node and qubit.
            compared = insert_run(
                service, workflow["id"], "05st", qubit, {"analysis_status": "pass"}
            )
            service.db.execute(
                "UPDATE proposals SET payload_json = ? WHERE id = "
                "(SELECT proposal_id FROM runs WHERE id = ?)",
                (json_dumps({"reset_comparison": True, "prerequisite_verification": True}), compared),
            )
            self.assertFalse(service._is_thermal_reset_comparison(workflow, "05st", thermal))

            # Its decision keeps the node and refuses a state patch.
            with self.assertRaises(ServiceError):
                service.record_decision(
                    workflow["id"], compared, "advance", "compare", "06", None,
                    [{"op": "replace", "path": "/qubits/q1/extras/T1", "value": 1e-4}], "test",
                )
            result = service.record_decision(
                workflow["id"], compared, "advance", "compare", None, None, None, "test"
            )
            self.assertEqual(result["next_action"]["next_node"], "06")
            self.assertEqual(service._workflow(workflow["id"])["current_node"], "06")


if __name__ == "__main__":
    unittest.main()
