"""Automatic-mode continuity.

Operator instruction 2026-09-20. In automatic mode the lease must survive
everything that can be corrected, so that a bring-up reaches T1/T2/fidelity
without collecting a fresh human approval on the way. A refusal that happens
before the worker reaches hardware is a correctable planning error; a
registered node-parameter crash is recoverable; an unrecognized worker crash
pauses for the operator and is resumable. Only losing track of the hardware or
the state file still ends the lease.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jy_agent.policy import PolicyEngine, PolicyError
from jy_agent.service import AgentService, ServiceError
from jy_agent.util import json_dumps

from test_autonomy import AUTO_PHRASE, activate_lease
from test_core import make_settings, sample_state, start_test_workflow


VALIDATION_TRACEBACK = """Traceback (most recent call last):
  File "worker.py", line 148, in run_request
    runnable = inspected.copy(**parameters)
pydantic_core._pydantic_core.ValidationError: 2 validation errors for Parameters
min_power_dbm
  Input should be a valid integer, got a number with a fractional part \
[type=int_from_float, input_value=-35.5, input_type=float]
max_power_dbm
  Input should be a valid integer, got a number with a fractional part \
[type=int_from_float, input_value=-25.5, input_type=float]
"""

ZERO_DIVISION_TRACEBACK = """Traceback (most recent call last):
  File "03a_Qubit_Spectroscopy.py", line 120, in <module>
    np.sqrt(detunings[q.name] / q.freq_vs_flux_01_quad_term)
ZeroDivisionError: float division by zero
"""

UNKNOWN_TRACEBACK = """Traceback (most recent call last):
  File "07b_IQ_Blobs.py", line 88, in <module>
    raise RuntimeError("something nobody has classified yet")
RuntimeError: something nobody has classified yet
"""

FETCH_ERROR_TRACEBACK = """Traceback (most recent call last):
  File "03a_Qubit_Spectroscopy.py", line 213, in <module>
    n = results.fetch_all()[0]
  File "qm/api/v2/job_result_api.py", line 76, in _group_results
    raise DataFetchingError(f"{response_val.details}")
qm.exceptions.DataFetchingError: UNKNOWN: Unexpected error in RPC handling
"""


def insert_failed_run(
    service: AgentService,
    workflow_id: str,
    proposal_id: str,
    lease_id: str,
    *,
    run_id: str,
    node_id: str,
    parameters: dict,
    traceback_text: str,
) -> dict:
    service.db.execute(
        "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
        "parameters_json, status, analysis_status, analysis_json, error, "
        "termination_cause, autonomy_lease_id) VALUES "
        "(?, ?, ?, ?, ?, 'failed', 'failed', ?, ?, 'worker_exception', ?)",
        (
            run_id,
            workflow_id,
            proposal_id,
            node_id,
            json_dumps(parameters),
            json_dumps({"analysis_status": "failed", "plots": []}),
            traceback_text,
            lease_id,
        ),
    )
    return service.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))


class WorkerCrashClassificationTests(unittest.TestCase):
    def build(self, folder: str) -> tuple[AgentService, dict, dict, dict]:
        service = AgentService(make_settings(Path(folder), sample_state(0.2, 0.1)))
        workflow = start_test_workflow(service)
        proposal, lease = activate_lease(service, workflow["id"])
        service._start_dashboard_session(
            workflow["id"], "autonomous", "unittest-agent",
            autonomy_lease_id=lease["id"],
        )
        return service, workflow, proposal, lease

    def test_registered_node_parameter_crash_keeps_the_lease_active(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal, lease = self.build(folder)
            insert_failed_run(
                service, workflow["id"], proposal["id"], lease["id"],
                run_id="02c-fractional-power",
                node_id="02c",
                parameters={
                    "qubits": ["q1"],
                    "min_power_dbm": -35.5,
                    "max_power_dbm": -25.5,
                },
                traceback_text=VALIDATION_TRACEBACK,
            )

            service.autonomy_watchdog_sweep()

            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "active"
            )
            event = service.db.one(
                "SELECT payload_json FROM events "
                "WHERE event_type = 'autonomy_recoverable_worker_exception' "
                "ORDER BY id DESC LIMIT 1"
            )
            self.assertIsNotNone(event)
            self.assertIn("02c_fractional_power_limits", event["payload_json"])

    def test_registered_03a_zero_division_keeps_the_lease_active(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal, lease = self.build(folder)
            insert_failed_run(
                service, workflow["id"], proposal["id"], lease["id"],
                run_id="03a-arbitrary-frequency",
                node_id="03a",
                parameters={
                    "qubits": ["q1"],
                    "arbitrary_qubit_frequency_in_ghz": 4.5,
                },
                traceback_text=ZERO_DIVISION_TRACEBACK,
            )

            service.autonomy_watchdog_sweep()

            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "active"
            )

    def test_same_traceback_on_another_node_is_not_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal, lease = self.build(folder)
            insert_failed_run(
                service, workflow["id"], proposal["id"], lease["id"],
                run_id="05-zero-division",
                node_id="05",
                parameters={"qubits": ["q1"]},
                traceback_text=ZERO_DIVISION_TRACEBACK,
            )

            service.autonomy_watchdog_sweep()

            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "paused"
            )

    def test_unknown_crash_pauses_and_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal, lease = self.build(folder)
            insert_failed_run(
                service, workflow["id"], proposal["id"], lease["id"],
                run_id="07b-unknown",
                node_id="07b",
                parameters={"qubits": ["q1"]},
                traceback_text=UNKNOWN_TRACEBACK,
            )

            service.autonomy_watchdog_sweep()

            status = service.autonomy_status(lease_id=lease["id"])
            self.assertEqual(status["status"], "paused")
            self.assertIn("Unrecognized worker failure", status["stopped_reason"])

            service.resume_measurement_mode(
                workflow["id"], "unit-test-human", "恢復量測"
            )

            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "active"
            )

    def test_resume_survives_the_next_watchdog_sweep(self) -> None:
        """The failed run stays in the table, so the sweep must not re-pause.

        Without a guard the watchdog undoes the resume on its next pass and
        `恢復量測` can never take effect, which strands the lease for good.
        """

        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal, lease = self.build(folder)
            insert_failed_run(
                service, workflow["id"], proposal["id"], lease["id"],
                run_id="07b-unknown-again",
                node_id="07b",
                parameters={"qubits": ["q1"]},
                traceback_text=UNKNOWN_TRACEBACK,
            )
            service.autonomy_watchdog_sweep()
            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "paused"
            )

            service.resume_measurement_mode(
                workflow["id"], "unit-test-human", "恢復量測"
            )
            service.autonomy_watchdog_sweep()

            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "active"
            )

    def test_fetch_stage_data_fetching_error_keeps_the_lease_active(self) -> None:
        """Operator decision 2026-09-20: halve the group instead of pausing."""

        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal, lease = self.build(folder)
            insert_failed_run(
                service, workflow["id"], proposal["id"], lease["id"],
                run_id="03a-fetch-error",
                node_id="03a",
                parameters={"qubits": ["q1", "q2", "q3"]},
                traceback_text=FETCH_ERROR_TRACEBACK,
            )

            service.autonomy_watchdog_sweep()

            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "active"
            )
            run = service.db.one(
                "SELECT * FROM runs WHERE id = ?", ("03a-fetch-error",)
            )
            recoverable = service.recoverable_worker_exception(run)
            self.assertIsNotNone(recoverable)
            self.assertEqual(
                recoverable["id"], "fetch_stage_data_fetching_error"
            )
            self.assertIn("Halve the multiplex group", recoverable["remedy"])


class PreHardwareRefusalTests(unittest.TestCase):
    def build(self, folder: str) -> tuple[AgentService, dict, dict]:
        service = AgentService(make_settings(Path(folder), sample_state(0.2, 0.1)))
        workflow = start_test_workflow(service)
        _proposal, lease = activate_lease(service, workflow["id"])
        return service, workflow, lease

    def test_parameter_policy_violation_refuses_without_halting(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = self.build(folder)

            with self.assertRaises(PolicyError):
                service.start_authorized_run(
                    workflow["id"], lease["id"], "02x",
                    {"qubits": ["q1"], "num_averages": 10_000_000},
                    "Deliberately over the averaging limit.", "unittest-agent",
                )

            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "active"
            )
            refusal = service.db.one(
                "SELECT payload_json FROM events "
                "WHERE event_type = 'autonomy_action_refused' ORDER BY id DESC LIMIT 1"
            )
            self.assertIsNotNone(refusal)
            self.assertIn("parameter_policy_violation", refusal["payload_json"])

    def test_analyzing_an_unfinished_run_refuses_without_halting(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = self.build(folder)
            proposal = service._create_proposal(
                workflow["id"], "run", {}, "unittest", 60, 1
            )
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
                "parameters_json, status, autonomy_lease_id) VALUES "
                "('still-running', ?, ?, '02x', ?, 'running', ?)",
                (
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": ["q1"]}),
                    lease["id"],
                ),
            )

            with self.assertRaises(ServiceError):
                service.analyze_authorized_run(
                    "still-running", lease["id"], "unittest-agent"
                )

            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "active"
            )

    def test_recoverable_scheduling_block_is_recognized_by_type(self) -> None:
        from jy_agent.service import AutonomyQuotaError, AutonomyScopeError

        recognize = AgentService._is_recoverable_scheduling_block
        self.assertTrue(recognize(AutonomyScopeError("reworded entirely")))
        self.assertTrue(recognize(AutonomyQuotaError("reworded entirely")))
        self.assertTrue(recognize(PolicyError("reworded entirely")))
        self.assertFalse(recognize(RuntimeError("an unrelated failure")))


class NodeParameterPreventionTests(unittest.TestCase):
    def policy(self, folder: str) -> PolicyEngine:
        state = sample_state(0.2, 0.1)
        state["qubits"]["q1"]["freq_vs_flux_01_quad_term"] = 0
        return PolicyEngine(make_settings(Path(folder), state))

    def test_02c_rejects_fractional_power_limits(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            policy = self.policy(folder)
            with self.assertRaises(PolicyError):
                policy.validate_run(
                    "02c",
                    {"qubits": ["q1"], "min_power_dbm": -35.5, "max_power_dbm": -25},
                )

    def test_02c_accepts_whole_number_power_limits(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            policy = self.policy(folder)
            merged, _warnings = policy.validate_run(
                "02c",
                {"qubits": ["q1"], "min_power_dbm": -46, "max_power_dbm": -6},
            )
            self.assertEqual(merged["min_power_dbm"], -46)

    def test_03a_rejects_arbitrary_frequency_before_flux_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            policy = self.policy(folder)
            with self.assertRaisesRegex(PolicyError, "freq_vs_flux_01_quad_term"):
                policy.validate_run(
                    "03a",
                    {
                        "qubits": ["q1"],
                        "arbitrary_qubit_frequency_in_ghz": 4.5,
                        "frequency_span_in_mhz": 200.0,
                    },
                )

    def test_03a_allows_arbitrary_frequency_once_flux_term_is_known(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state = sample_state(0.2, 0.1)
            state["qubits"]["q1"]["freq_vs_flux_01_quad_term"] = -1.5e9
            policy = PolicyEngine(make_settings(Path(folder), state))
            merged, _warnings = policy.validate_run(
                "03a",
                {
                    "qubits": ["q1"],
                    "arbitrary_qubit_frequency_in_ghz": 4.5,
                    "frequency_span_in_mhz": 200.0,
                },
            )
            self.assertEqual(merged["arbitrary_qubit_frequency_in_ghz"], 4.5)


class LeaseDurationTests(unittest.TestCase):
    def test_entry_can_request_a_longer_budget(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            result = service.enter_autonomy_mode(
                "unittest-agent", AUTO_PHRASE, duration_hours=16
            )
            self.assertEqual(result["authorization"]["duration_hours"], 16.0)

    def test_entry_hours_must_be_an_approval_page_choice(self) -> None:
        """2026-10-20: the page offers 8/16/24 h; 20 h used to show as 8 h."""

        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            with self.assertRaisesRegex(ServiceError, "Approval page choices"):
                service.enter_autonomy_mode(
                    "unittest-agent", AUTO_PHRASE, duration_hours=20
                )
            self.assertIsNone(service.db.one("SELECT id FROM workflows LIMIT 1"))

    def test_entry_without_a_budget_uses_the_configured_default(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            result = service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE)
            self.assertEqual(
                result["authorization"]["duration_hours"],
                float(service.policy.raw["autonomy"]["duration_hours"]),
            )

    def test_entry_above_the_policy_ceiling_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            with self.assertRaisesRegex(ServiceError, "policy maximum"):
                service.enter_autonomy_mode(
                    "unittest-agent", AUTO_PHRASE, duration_hours=500
                )
            self.assertIsNone(
                service.db.one("SELECT id FROM autonomy_leases LIMIT 1")
            )

    def test_entry_with_a_nonpositive_budget_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            with self.assertRaises(ServiceError):
                service.enter_autonomy_mode(
                    "unittest-agent", AUTO_PHRASE, duration_hours=0
                )


if __name__ == "__main__":
    unittest.main()
