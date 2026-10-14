from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jy_agent.service import AgentService, ServiceError
from jy_agent.util import atomic_write_json
from tests.test_core import (
    ENTRY_PHRASE,
    make_settings,
    sample_multiplex_state_and_wiring,
)

FULL_SCALE = "/ports/mw_outputs/con1/7/1/full_scale_power_dbm"


def _transition(frequency: float) -> dict:
    return {
        "02c_transition": {
            "validation_failures": [],
            "dressed_frequency_hz": frequency,
            "dressed_power_limit_dbm": -25.0,
        }
    }


class Test02cLineRescale(unittest.TestCase):
    """Operator choice 2026-10-14 (CRSQ_26009_die(5,-1) q3, #507)."""

    def _service(self, folder: str) -> tuple[AgentService, dict, str]:
        state, wiring, _ = sample_multiplex_state_and_wiring()
        state["ports"] = {
            "mw_outputs": {"con1": {"7": {"1": {"full_scale_power_dbm": -11}}}}
        }
        settings = make_settings(Path(folder), state)
        atomic_write_json(settings.wiring_path, wiring)
        service = AgentService(settings)
        workflow = service.start_workflow(
            ["q3", "q4"], {"multiplexed": True}, "unittest", ENTRY_PHRASE
        )
        service.db.execute(
            "UPDATE workflows SET current_node = '02c' WHERE id = ?", (workflow["id"],)
        )
        run_proposal = service.request_run(
            workflow["id"], "02c", {"qubits": ["q3", "q4"]}, "Offline 02c.", "unittest"
        )
        analysis = {
            "dataset_metrics": {
                "qubits": {"q3": _transition(6.0e9), "q4": _transition(6.1e9)}
            },
            "passing_targets": ["q4"],
        }
        service.db.execute(
            "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
            "parameters_json, status, analysis_status, analysis_json) "
            "VALUES ('run-02c', ?, ?, '02c', ?, 'completed', 'needs_review', ?)",
            (
                workflow["id"],
                run_proposal["id"],
                json.dumps({"qubits": ["q3", "q4"]}),
                json.dumps(analysis),
            ),
        )
        return service, state, workflow["id"]

    def test_needs_review_transition_without_pass_is_not_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _, wid = self._service(folder)
            self.assertEqual(service._node_target_resolution(wid, "02c"), {"q4"})

    def test_power_preserving_line_rescale_commits_and_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, state, wid = self._service(folder)
            q5_amp = state["qubits"]["q5"]["resonator"]["operations"]["readout"]["amplitude"]
            patch = [
                {"op": "replace", "path": FULL_SCALE, "value": -8},
                {
                    "op": "replace",
                    "path": "/qubits/q3/resonator/operations/readout/amplitude",
                    "value": 0.098,
                },
                {
                    "op": "replace",
                    "path": "/qubits/q5/resonator/operations/readout/amplitude",
                    "value": q5_amp * 10 ** (-3 / 20),
                },
            ]
            proposal = service.request_state_commit(
                wid, patch, "Raise the line for q3.", "unittest", run_id="run-02c"
            )
            service.db.execute(
                "UPDATE proposals SET status = 'consumed' WHERE id = ?",
                (proposal["id"],),
            )
            self.assertEqual(
                service._node_target_resolution(wid, "02c"), {"q3", "q4"}
            )

    def test_operator_commit_resolves_other_nodes(self) -> None:
        # 2026-10-14 (q3 at 04): an operator-approved commit from a run
        # resolves the qubit; a lease-delegated one does not.
        with tempfile.TemporaryDirectory() as folder:
            service, _, wid = self._service(folder)
            rows = [{"run_id": "run-02c"}]
            for lease, expected in ((None, {"q3"}), ("lease-x", set())):
                service.db.execute("DELETE FROM proposals WHERE kind = 'state_commit'")
                service.db.execute(
                    "INSERT INTO proposals(id, workflow_id, kind, payload_json, status, "
                    "created_at, expires_at, max_uses, source_client, autonomy_lease_id) "
                    "VALUES ('c1', ?, 'state_commit', ?, 'consumed', 'x', 'x', 1, 'u', ?)",
                    (
                        wid,
                        json.dumps({
                            "run_id": "run-02c",
                            "patch": [{
                                "op": "replace",
                                "path": "/qubits/q3/xy/operations/x180/amplitude",
                                "value": 0.03,
                            }],
                        }),
                        lease,
                    ),
                )
                self.assertEqual(
                    service._operator_committed_targets(wid, rows, "04"), expected
                )
                # A commit that does not write the node's outputs (an
                # x180 amplitude is not a T1) does not resolve 05.
                self.assertEqual(
                    service._operator_committed_targets(wid, rows, "05"), set()
                )

    def test_non_target_power_change_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, state, wid = self._service(folder)
            q5_amp = state["qubits"]["q5"]["resonator"]["operations"]["readout"]["amplitude"]
            patch = [
                {"op": "replace", "path": FULL_SCALE, "value": -8},
                {
                    "op": "replace",
                    "path": "/qubits/q5/resonator/operations/readout/amplitude",
                    "value": q5_amp,
                },
            ]
            with self.assertRaisesRegex(ServiceError, "workflow target"):
                service.request_state_commit(
                    wid, patch, "Changes q5's power.", "unittest", run_id="run-02c"
                )
            # A line member left out of the patch also blocks the port.
            with self.assertRaisesRegex(ServiceError, "workflow target"):
                service.request_state_commit(
                    wid, patch[:1], "Leaves q5 out.", "unittest", run_id="run-02c"
                )


if __name__ == "__main__":
    unittest.main()
