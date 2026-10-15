"""06 calibrates the frequency before it measures T2* (operator 2026-10-15).

First 06 of a qubit: 1016 ns, corrects the frequency, never resolves T2*.
Then 1016 ns again until both fringes sit on the detuning (confirmed).
Then the window lengthens: about 4 x T2* from a reliable fit, else the ladder.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jy_agent.service import AgentService, AutonomyScopeError
from jy_agent.util import atomic_write_json

from test_core import make_settings, sample_state, start_test_workflow


def build(folder: str):
    state = sample_state(0.2, 0.1)
    state["qubits"]["q1"]["extras"] = {"idle_freq": 4.0e9}
    service = AgentService(make_settings(Path(folder), state))
    object.__setattr__(
        service.settings, "workflow_sequence", (*service.settings.workflow_sequence, "06")
    )
    workflow = start_test_workflow(service)
    service.db.execute(
        "UPDATE workflows SET current_node = '06' WHERE id = ?", (workflow["id"],)
    )
    service.db.execute(
        "INSERT INTO proposals(id, workflow_id, kind, payload_json, status, "
        "created_at, expires_at, max_uses, source_client) "
        "VALUES ('p', ?, 'run', '{}', 'consumed', '', '', 1, 'unittest')",
        (workflow["id"],),
    )
    return service, workflow


def add_run(service, workflow, run_id, window, *, matches, decay=None, relative=None,
            correction=None):
    fit = {
        "fringe_matches_detuning": matches,
        "coherence_seconds": decay,
        "relative_uncertainty": relative,
    }
    analysis = {
        "fit_quality": {"results": {"q1": fit}},
        "frequency_correction_patch": (
            [{"op": "replace", "path": "/qubits/q1/extras/idle_freq", "value": correction}]
            if correction is not None
            else []
        ),
    }
    service.db.execute(
        "INSERT INTO runs(id, workflow_id, proposal_id, node_id, parameters_json, status, "
        "analysis_status, analysis_json) VALUES (?, ?, 'p', '06', ?, 'completed', "
        "'needs_review', ?)",
        (run_id, workflow["id"], json.dumps({"qubits": ["q1"], "max_wait_time_in_ns": window}),
         json.dumps(analysis)),
    )


def set_idle(service, value):
    path = service.settings.active_state
    state = json.loads(path.read_text(encoding="utf-8"))
    state["qubits"]["q1"]["extras"]["idle_freq"] = value
    atomic_write_json(path, state)


class RamseyFrequencyStageTests(unittest.TestCase):
    def test_first_06_must_use_the_calibration_window(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = build(folder)
            self.assertEqual(service._06_stage(workflow["id"], "q1")["stage"], "calibrate")
            with self.assertRaisesRegex(AutonomyScopeError, "frequency-calibrate stage"):
                service.request_run(
                    workflow["id"], "06", {"qubits": ["q1"], "max_wait_time_in_ns": 40016, "use_state_discrimination": False},
                    "Long first window.", "unittest",
                )
            service.request_run(
                workflow["id"], "06", {"qubits": ["q1"], "max_wait_time_in_ns": 1016, "use_state_discrimination": False},
                "Calibrate.", "unittest",
            )
            plan = service.next_action(workflow["id"])
            self.assertEqual(plan["ramsey_stages"]["q1"]["max_wait_time_in_ns"], 1016)

    def test_first_run_corrects_the_frequency_and_never_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = build(folder)
            add_run(service, workflow, "r1", 1016, matches=True)
            result = service._06_calibration_run_result(
                {"id": "r1", "workflow_id": workflow["id"],
                 "parameters_json": json.dumps({"qubits": ["q1"]})},
                {
                    "analysis_status": "pass",
                    "passing_targets": ["q1"],
                    "candidate_state_patch": [
                        {"op": "replace", "path": "/qubits/q1/xy/intermediate_frequency", "value": 1.0},
                        {"op": "replace", "path": "/qubits/q1/extras/idle_freq", "value": 4.001e9},
                        {"op": "replace", "path": "/qubits/q1/T2ramsey", "value": 2e-7},
                    ],
                    "frequency_correction_patch": [],
                },
            )
            self.assertEqual(result["passing_targets"], [])
            self.assertEqual(result["analysis_status"], "needs_review")
            self.assertEqual(result["candidate_state_patch"], [])
            self.assertEqual(
                [item["path"] for item in result["frequency_correction_patch"]],
                ["/qubits/q1/xy/intermediate_frequency", "/qubits/q1/extras/idle_freq"],
            )

    def test_uncommitted_correction_blocks_the_next_06(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = build(folder)
            add_run(service, workflow, "r1", 1016, matches=False, correction=4.002e9)
            with self.assertRaisesRegex(AutonomyScopeError, "commit the frequency correction"):
                service.request_run(
                    workflow["id"], "06", {"qubits": ["q1"], "max_wait_time_in_ns": 1016, "use_state_discrimination": False},
                    "Confirm.", "unittest",
                )
            set_idle(service, 4.002e9)
            self.assertEqual(service._06_stage(workflow["id"], "q1")["stage"], "confirm")
            service.request_run(
                workflow["id"], "06", {"qubits": ["q1"], "max_wait_time_in_ns": 1016, "use_state_discrimination": False},
                "Confirm.", "unittest",
            )

    def test_confirmed_qubit_lengthens_from_a_reliable_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = build(folder)
            add_run(service, workflow, "r1", 1016, matches=True)
            add_run(service, workflow, "r2", 1016, matches=True, decay=3.0e-6, relative=0.3)
            stage = service._06_stage(workflow["id"], "q1")
            self.assertEqual(stage["stage"], "measure")
            # 4 x 3 us = 12000 ns -> 16 + 24 x 500
            self.assertEqual(stage["max_wait_time_in_ns"], 12016)

    def test_without_an_estimate_the_window_climbs_the_ladder(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = build(folder)
            add_run(service, workflow, "r1", 1016, matches=True)
            add_run(service, workflow, "r2", 1016, matches=True, decay=5e-5, relative=2.0)
            self.assertEqual(service._06_stage(workflow["id"], "q1")["max_wait_time_in_ns"], 4016)
            add_run(service, workflow, "r3", 4016, matches=True)
            self.assertEqual(service._06_stage(workflow["id"], "q1")["max_wait_time_in_ns"], 16016)

    def test_window_cap_is_100016(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow = build(folder)
            with self.assertRaisesRegex(Exception, "exceeds 100016"):
                service.policy.validate_run(
                    "06", {"qubits": ["q1"], "max_wait_time_in_ns": 100516}
                )


if __name__ == "__main__":
    unittest.main()
