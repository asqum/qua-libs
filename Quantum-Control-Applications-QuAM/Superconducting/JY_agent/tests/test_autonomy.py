from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from jy_agent.approval_server import create_approval_app
from jy_agent.service import (
    AgentService,
    AutonomyEvidenceError,
    AutonomyQuotaError,
    AutonomyScopeError,
    SHUTDOWN_HINT,
    ServiceError,
    host_pause_operator_handoff,
    recover_operator_handoff,
    recovery_console_operator_handoff,
)
from jy_agent.state import StateError
from jy_agent.util import atomic_write_json, json_dumps, sha256_file

from test_core import (
    make_settings,
    sample_multiplex_state_and_wiring,
    sample_state,
    start_test_workflow,
)


AUTO_PHRASE = "進入 JY 自動量測模式"


def activate_lease(service: AgentService, workflow_id: str) -> tuple[dict, dict]:
    proposal = service.request_autonomy_lease(
        workflow_id,
        "unittest-agent",
        AUTO_PHRASE,
        "Run the bounded commissioning workflow while the operator is away.",
    )
    service._approve_pending_proposal(
        proposal["id"], "unit-test-human", "unit_test"
    )
    return proposal, service.autonomy_status(lease_id=proposal["autonomy_lease_id"])


class AutonomyTests(unittest.TestCase):
    def test_02a_retry_keeps_unresolved_subgroup_without_halt(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            workflow = service.start_workflow(
                targets,
                {"multiplexed": True},
                "unittest-agent",
                "進入 JY 量測模式",
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '02a' WHERE id = ?",
                (workflow["id"],),
            )
            _, lease = activate_lease(service, workflow["id"])
            first_proposal = service.request_run(
                workflow["id"],
                "02a",
                {
                    "qubits": targets,
                    "frequency_span_in_mhz": 20,
                    "multiplexed": True,
                },
                "Initial shared-parameter 02a batch.",
                "unittest-agent",
                autonomy_lease_id=lease["id"],
            )
            analysis = {
                "fit_quality": {
                    "results": {
                        targets[0]: {"RO_frequency": 6.0e9},
                        targets[1]: {"RO_frequency": 6.1e9},
                    }
                },
                "dataset_metrics": {
                    "qubits": {
                        targets[0]: {"edge_fraction": 0.4, "robust_snr": 20.0},
                        targets[1]: {"edge_fraction": 0.01, "robust_snr": 1.0},
                        **{
                            name: {"edge_fraction": 0.4, "robust_snr": 20.0}
                            for name in targets[2:]
                        },
                    }
                },
            }
            # Mark remaining targets resolved via metrics as well.
            for name in targets[2:]:
                analysis["fit_quality"]["results"][name] = {"RO_frequency": 6.2e9}
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
                "parameters_json, status, analysis_status, analysis_json) "
                "VALUES (?, ?, ?, '02a', ?, 'completed', 'needs_review', ?)",
                (
                    "02a-first-batch",
                    workflow["id"],
                    first_proposal["id"],
                    json_dumps({"qubits": targets, "multiplexed": True}),
                    json_dumps(analysis),
                ),
            )

            with self.assertRaisesRegex(AutonomyScopeError, "unresolved"):
                service.request_run(
                    workflow["id"],
                    "02a",
                    {
                        "qubits": targets,
                        "frequency_span_in_mhz": 40,
                        "multiplexed": True,
                    },
                    "Must not remeasure the already-resolved target.",
                    "unittest-agent",
                    autonomy_lease_id=lease["id"],
                )
            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "active"
            )

            proposal = service.request_run(
                workflow["id"],
                "02a",
                {
                    "qubits": [targets[1]],
                    "frequency_span_in_mhz": 40,
                    "multiplexed": True,
                },
                "Retry only the unresolved edge-limited target.",
                "unittest-agent",
                autonomy_lease_id=lease["id"],
            )
            self.assertEqual(
                proposal["payload"]["parameters"]["qubits"], [targets[1]]
            )
            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "active"
            )

    def test_02a_first_run_still_requires_every_active_target(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            workflow = service.start_workflow(
                targets,
                {"multiplexed": True},
                "unittest-agent",
                "進入 JY 量測模式",
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '02a' WHERE id = ?",
                (workflow["id"],),
            )
            _, lease = activate_lease(service, workflow["id"])

            with self.assertRaisesRegex(AutonomyScopeError, "first 02a run"):
                service.request_run(
                    workflow["id"],
                    "02a",
                    {
                        "qubits": [targets[0]],
                        "frequency_span_in_mhz": 20,
                        "multiplexed": True,
                    },
                    "Isolated first batch is not allowed.",
                    "unittest-agent",
                    autonomy_lease_id=lease["id"],
                )
            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "active"
            )

    def test_02a_repeat_decision_keeps_unresolved_retry_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            workflow = service.start_workflow(
                targets,
                {"multiplexed": True},
                "unittest-agent",
                "進入 JY 量測模式",
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '02a' WHERE id = ?",
                (workflow["id"],),
            )
            proposal = service.request_run(
                workflow["id"],
                "02a",
                {"qubits": targets, "multiplexed": True},
                "Collect the initial shared-parameter evidence.",
                "unittest-agent",
            )
            run_id = "02a-edge-run"
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
                "parameters_json, status, analysis_status, analysis_json) "
                "VALUES (?, ?, ?, '02a', ?, 'completed', 'needs_review', '{}')",
                (
                    run_id,
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": targets, "multiplexed": True}),
                ),
            )

            decision = service.record_decision(
                workflow["id"],
                run_id,
                "repeat",
                "One feature is edge-limited; retry only that unresolved target.",
                "02a",
                {"qubits": [targets[0]], "frequency_span_in_mhz": 40},
                [],
                "unittest-agent",
            )

            self.assertEqual(
                decision["next_action"]["new_parameters"]["qubits"], [targets[0]]
            )
            self.assertFalse(decision.get("parameter_adjustments"))

    def test_03a_coarse_retry_with_candidates_soft_rejects_without_halt(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            workflow = service.start_workflow(
                targets,
                {"multiplexed": True},
                "unittest-agent",
                "進入 JY 量測模式",
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '03a' WHERE id = ?",
                (workflow["id"],),
            )
            _, lease = activate_lease(service, workflow["id"])
            for name in targets:
                state["qubits"][name]["xy"]["intermediate_frequency"] = 0.0
            atomic_write_json(settings.active_state, state)
            first_proposal = service.request_run(
                workflow["id"],
                "03a",
                {
                    "qubits": targets,
                    "frequency_span_in_mhz": 800,
                    "frequency_step_in_mhz": 1,
                    "multiplexed": True,
                    "num_averages": 100,
                    "operation": "saturation",
                    "operation_amplitude_factor": 0.01,
                    "operation_len_in_ns": 50000,
                },
                "Initial 03a coarse batch.",
                "unittest-agent",
                autonomy_lease_id=lease["id"],
            )
            analysis = {
                "03a_stage": "coarse_candidate",
                "fit_quality": {
                    "results": {
                        name: {"fit_successful": True, "drive_freq": 5.0e9}
                        for name in targets
                    }
                },
                "dataset_metrics": {
                    "qubits": {
                        targets[0]: {
                            "edge_fraction": 0.4,
                            "robust_snr": 20.0,
                            "feature_fwhm_hz": 7_500_000.0,
                        },
                        **{
                            name: {
                                "edge_fraction": 0.4,
                                "robust_snr": 2.0,
                                "feature_fwhm_hz": 7_500_000.0,
                            }
                            for name in targets[1:]
                        },
                    }
                },
            }
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
                "parameters_json, status, analysis_status, analysis_json) "
                "VALUES (?, ?, ?, '03a', ?, 'completed', 'needs_review', ?)",
                (
                    "03a-coarse",
                    workflow["id"],
                    first_proposal["id"],
                    json_dumps({"qubits": targets, "multiplexed": True}),
                    json_dumps(analysis),
                ),
            )

            with self.assertRaisesRegex(AutonomyScopeError, "allowed subgroup"):
                service.start_authorized_run(
                    workflow["id"],
                    lease["id"],
                    "03a",
                    {
                        "qubits": targets,
                        "frequency_span_in_mhz": 800,
                        "frequency_step_in_mhz": 1,
                        "multiplexed": True,
                        "num_averages": 500,
                        "operation": "saturation",
                        "operation_amplitude_factor": 0.01,
                        "operation_len_in_ns": 50000,
                    },
                    "Must not remeasure SNR-cleared candidates in coarse retry.",
                    "unittest-agent",
                )
            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "active"
            )

    def test_measurement_entry_creates_conversational_workflow_without_lease(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            entered = service.enter_measurement_mode(
                "unittest-agent",
                "進入 JY 量測模式",
                ["q1"],
            )
            self.assertEqual(entered["workflow"]["status"], "active")
            self.assertEqual(entered["mode"], "conversational")
            self.assertEqual(entered["approval_model"], "per_run_and_state_commit")
            self.assertEqual(
                entered["next_action"], "discuss_and_propose_one_experiment"
            )
            self.assertEqual(entered["browser_url"], "http://127.0.0.1:8766/")
            self.assertEqual(entered["operator_handoff"]["reply"], "已核准")
            self.assertIn("需要使用者做什麼:", entered["operator_handoff"]["chat"])
            self.assertIn("完成後回傳: 已核准", entered["operator_handoff"]["chat"])
            self.assertEqual(entered["operator_handoff"]["shutdown_hint"], SHUTDOWN_HINT)
            self.assertIn(SHUTDOWN_HINT, entered["operator_handoff"]["chat"])
            self.assertEqual(
                entered["operator_handoff"]["chat"].splitlines(),
                [
                    "- 需要使用者做什麼: 開啟對話中的 Dashboard 網址並登入，到 Approval 頁核准。",
                    "- 完成後回傳: 已核准",
                    f"- {SHUTDOWN_HINT}",
                ],
            )

    def test_autonomy_entry_creates_workflow_and_bounded_lease(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            entered = service.enter_autonomy_mode(
                "unittest-agent",
                AUTO_PHRASE,
                ["q1"],
                reason="One autonomous entry requests bounded operation.",
            )
            self.assertEqual(entered["workflow"]["status"], "active")
            self.assertEqual(entered["authorization"]["status"], "pending")
            self.assertEqual(entered["approval"]["kind"], "autonomy_lease")
            self.assertEqual(entered["next_action"], "human_approval")
            self.assertEqual(entered["browser_url"], "http://127.0.0.1:8766/")
            self.assertEqual(entered["dashboard"]["mode"], "autonomous")
            self.assertEqual(entered["operator_handoff"]["reply"], "已核准")
            self.assertIn("Approval 頁核准", entered["operator_handoff"]["do"])

    def test_pending_autonomy_wait_timeout_returns_approval_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            entered = service.enter_autonomy_mode(
                "unittest-agent", AUTO_PHRASE, ["q1"]
            )
            waited = service.wait_for_autonomy_status(
                entered["authorization"]["id"], timeout_seconds=0
            )
            self.assertTrue(waited["timed_out"])
            self.assertEqual(waited["authorization"]["status"], "pending")
            self.assertEqual(waited["operator_handoff"]["reply"], "已核准")
            self.assertIn(
                "完成後回傳: 已核准", waited["operator_handoff"]["chat"]
            )

    def test_operator_handoff_recover_phrases_are_fixed(self) -> None:
        self.assertEqual(recover_operator_handoff()["reply"], "恢復")
        self.assertEqual(recovery_console_operator_handoff()["reply"], "恢復")
        self.assertEqual(host_pause_operator_handoff()["reply"], "已核准")
        self.assertIn("需要使用者做什麼:", recover_operator_handoff()["chat"])
        self.assertEqual(recover_operator_handoff()["shutdown_hint"], SHUTDOWN_HINT)
        self.assertIn(SHUTDOWN_HINT, recover_operator_handoff()["chat"])
        chat_lines = recover_operator_handoff()["chat"].splitlines()
        self.assertEqual(len(chat_lines), 3)
        self.assertTrue(all(line.startswith("- ") for line in chat_lines))
        self.assertEqual(chat_lines[0], "- 需要使用者做什麼: 讓系統安全收尾殘留的 workflow、process 與 JY 服務。")
        self.assertEqual(chat_lines[1], "- 完成後回傳: 恢復")
        self.assertEqual(chat_lines[2], f"- {SHUTDOWN_HINT}")

    def test_expired_conversational_proposal_does_not_block_autonomy_entry(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            workflow = start_test_workflow(service)
            stale = service.request_run(
                workflow["id"],
                "02x",
                {"qubits": ["q1"]},
                "Stale conversational proposal.",
                "unittest-agent",
            )
            service.db.execute(
                "UPDATE proposals SET expires_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00+00:00", stale["id"]),
            )

            entered = service.enter_autonomy_mode(
                "unittest-agent", AUTO_PHRASE, ["q1"]
            )

            self.assertEqual(service.proposal(stale["id"])["status"], "expired")
            self.assertEqual(entered["approval"]["kind"], "autonomy_lease")
            self.assertEqual(entered["next_action"], "human_approval")

    def test_phrase_only_entry_resumes_existing_workflow_and_lease(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            first = service.enter_autonomy_mode(
                "unittest-agent", AUTO_PHRASE, ["q1"]
            )
            service._approve_pending_proposal(
                first["approval"]["id"], "unit-test-human", "unit_test"
            )
            service.pause_autonomy(
                first["authorization"]["id"], "unittest-agent", "Pause test"
            )
            resumed = service.enter_autonomy_mode(
                "unittest-agent", AUTO_PHRASE
            )
            self.assertEqual(resumed["authorization"]["status"], "active")
            self.assertEqual(
                resumed["next_action"], "continue_authorized_workflow"
            )

    def test_phrase_only_entry_uses_active_qubit_names_and_defaults_multiplexed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            entered = service.enter_measurement_mode(
                "unittest-agent", "進入 JY 量測模式"
            )
            workflow = entered["workflow"]
            self.assertEqual(workflow["targets"], ["q1"])
            self.assertTrue(workflow["initial_parameters"]["multiplexed"])

    def test_phrase_only_entry_requires_active_qubit_names_for_brand_new_workflow(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state = sample_state(0.2, 0.1)
            del state["active_qubit_names"]
            service = AgentService(make_settings(Path(folder), state))
            with self.assertRaisesRegex(ServiceError, "active_qubit_names"):
                service.enter_measurement_mode(
                    "unittest-agent", "進入 JY 量測模式"
                )

    def test_explicit_targets_override_active_qubit_names(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            entered = service.enter_measurement_mode(
                "unittest-agent",
                "進入 JY 量測模式",
                [targets[0]],
                multiplexed=False,
            )
            self.assertEqual(entered["workflow"]["targets"], [targets[0]])
            self.assertFalse(entered["workflow"]["initial_parameters"]["multiplexed"])

    def test_phrase_only_entry_uses_multiplex_active_qubit_names(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            entered = service.enter_measurement_mode(
                "unittest-agent", "進入 JY 量測模式"
            )
            self.assertEqual(entered["workflow"]["targets"], targets)
            self.assertTrue(entered["workflow"]["initial_parameters"]["multiplexed"])

    def test_phrase_only_entry_resumes_non_multiplexed_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            workflow = start_test_workflow(service)
            resumed = service.enter_measurement_mode(
                "unittest-agent", "進入 JY 量測模式"
            )
            self.assertEqual(resumed["workflow"]["id"], workflow["id"])
            self.assertFalse(
                resumed["workflow"]["initial_parameters"].get("multiplexed", False)
            )

    def test_autonomy_phrase_is_not_accepted_by_conversational_entry(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            with self.assertRaisesRegex(ServiceError, "exact JY"):
                service.enter_measurement_mode(
                    "unittest-agent", "進入 JY 自動量測模式", ["q1"]
                )

    def test_conversational_mode_refuses_live_autonomy_lease(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            entered = service.enter_autonomy_mode(
                "unittest-agent", AUTO_PHRASE, ["q1"]
            )
            service._approve_pending_proposal(
                entered["approval"]["id"], "unit-test-human", "unit_test"
            )
            with self.assertRaisesRegex(ServiceError, "live autonomy lease"):
                service.enter_measurement_mode(
                    "unittest-agent", "進入 JY 量測模式", ["q1"]
                )

    def test_autonomy_phrase_can_create_workflow_before_lease_request(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            workflow = service.start_workflow(
                ["q1"], {}, "unittest-agent", AUTO_PHRASE
            )
            self.assertEqual(workflow["status"], "active")
            proposal = service.request_autonomy_lease(
                workflow["id"],
                "unittest-agent",
                AUTO_PHRASE,
                "Bounded autonomous commissioning.",
            )
            self.assertEqual(proposal["kind"], "autonomy_lease")

    def test_lease_has_requested_limits_and_eight_hours_from_approval(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            workflow = start_test_workflow(service)
            proposal, lease = activate_lease(service, workflow["id"])
            self.assertEqual(proposal["status"], "pending")
            self.assertEqual(lease["status"], "active")
            self.assertIsNone(lease["max_total_runs"])
            self.assertEqual(lease["max_attempts_per_node_qubit"], 20)
            self.assertEqual(lease["auto_state_commit_analysis_statuses"], ["pass"])
            activated = datetime.fromisoformat(lease["activated_at"])
            expires = datetime.fromisoformat(lease["expires_at"])
            self.assertAlmostEqual(
                (expires - activated).total_seconds(), 8 * 3600, delta=1
            )
            self.assertEqual(lease["control_url"], "http://127.0.0.1:8766/")

    def test_expired_pending_proposal_allows_reissuing_same_lease_scope(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            workflow = start_test_workflow(service)
            first = service.request_autonomy_lease(
                workflow["id"],
                "unittest-agent",
                AUTO_PHRASE,
                "First authorization request.",
            )
            service.db.execute(
                "UPDATE proposals SET expires_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00+00:00", first["id"]),
            )

            replacement = service.request_autonomy_lease(
                workflow["id"],
                "unittest-agent",
                AUTO_PHRASE,
                "Replacement for an expired approval page.",
            )

            self.assertNotEqual(replacement["id"], first["id"])
            self.assertEqual(service.proposal(first["id"])["status"], "expired")
            self.assertEqual(
                service.autonomy_status(lease_id=first["autonomy_lease_id"])[
                    "status"
                ],
                "expired",
            )
            self.assertEqual(replacement["status"], "pending")

    def test_attempt_twenty_one_marks_target_incomplete_without_halting(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            workflow = start_test_workflow(service)
            proposal, lease = activate_lease(service, workflow["id"])
            for index in range(20):
                service.db.execute(
                    """
                    INSERT INTO runs(
                        id, workflow_id, proposal_id, node_id, parameters_json,
                        status, analysis_status, autonomy_lease_id
                    ) VALUES (?, ?, ?, '02x', ?, 'completed', 'needs_review', ?)
                    """,
                    (
                        f"attempt-{index}",
                        workflow["id"],
                        proposal["id"],
                        json_dumps({"qubits": ["q1"]}),
                        lease["id"],
                    ),
                )
            with self.assertRaisesRegex(AutonomyQuotaError, "attempt limit"):
                service.start_authorized_run(
                    workflow["id"],
                    lease["id"],
                    "02x",
                    {"qubits": ["q1"]},
                    "Attempt twenty-one.",
                    "unittest-agent",
                )
            current = service.autonomy_status(lease_id=lease["id"])
            self.assertEqual(current["status"], "active")
            self.assertEqual(
                current["incomplete_targets_by_node"], {"02x": ["q1"]}
            )

    def test_attempt_counter_is_independent_per_qubit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state = sample_state(0.2, 0.1)
            state["qubits"]["q2"] = deepcopy(state["qubits"]["q1"])
            service = AgentService(make_settings(Path(folder), state))
            workflow = start_test_workflow(service, ["q1", "q2"])
            service.db.execute(
                "UPDATE workflows SET current_node = '04' WHERE id = ?",
                (workflow["id"],),
            )
            proposal, lease = activate_lease(service, workflow["id"])
            for index in range(20):
                service.db.execute(
                    """
                    INSERT INTO runs(
                        id, workflow_id, proposal_id, node_id, parameters_json,
                        status, analysis_status, autonomy_lease_id
                    ) VALUES (?, ?, ?, '04', ?, 'completed', 'needs_review', ?)
                    """,
                    (
                        f"q1-attempt-{index}",
                        workflow["id"],
                        proposal["id"],
                        json_dumps({"qubits": ["q1"]}),
                        lease["id"],
                    ),
                )
            with self.assertRaises(AutonomyQuotaError):
                service.request_run(
                    workflow["id"],
                    "04",
                    {"qubits": ["q1"]},
                    "q1 is exhausted.",
                    "unittest-agent",
                    autonomy_lease_id=lease["id"],
                )
            q2 = service.request_run(
                workflow["id"],
                "04",
                {"qubits": ["q2"]},
                "q2 starts with an independent counter.",
                "unittest-agent",
                autonomy_lease_id=lease["id"],
            )
            self.assertEqual(q2["status"], "approved")
            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"],
                "active",
            )

    def test_exhausted_unresolved_qubit_is_excluded_from_downstream_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state = sample_state(0.2, 0.1)
            state["qubits"]["q2"] = deepcopy(state["qubits"]["q1"])
            service = AgentService(make_settings(Path(folder), state))
            workflow = start_test_workflow(service, ["q1", "q2"])
            service.db.execute(
                "UPDATE workflows SET current_node = '03a' WHERE id = ?",
                (workflow["id"],),
            )
            proposal, lease = activate_lease(service, workflow["id"])
            q1_analysis = {
                "03a_stage": "fine",
                "fit_quality": {
                    "results": {
                        "q1": {"fit_successful": True, "drive_freq": 5.0e9}
                    }
                },
                "dataset_metrics": {
                    "qubits": {
                        "q1": {
                            "edge_fraction": 0.4,
                            "robust_snr": 12.0,
                            "feature_fwhm_hz": 1_500_000.0,
                        }
                    }
                },
            }
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, analysis_json, autonomy_lease_id
                ) VALUES ('q1-resolved', ?, ?, '03a', ?, 'completed', 'pass', ?, ?)
                """,
                (
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": ["q1"]}),
                    json_dumps(q1_analysis),
                    lease["id"],
                ),
            )
            for index in range(20):
                service.db.execute(
                    """
                    INSERT INTO runs(
                        id, workflow_id, proposal_id, node_id, parameters_json,
                        status, analysis_status, analysis_json, autonomy_lease_id
                    ) VALUES (?, ?, ?, '03a', ?, 'completed', 'needs_review', ?, ?)
                    """,
                    (
                        f"q2-unresolved-{index}",
                        workflow["id"],
                        proposal["id"],
                        json_dumps({"qubits": ["q2"]}),
                        json_dumps({"03a_stage": "fine"}),
                        lease["id"],
                    ),
                )
            result = service.record_decision(
                workflow["id"],
                "q2-unresolved-19",
                "advance",
                "q2 reached its quota without usable evidence; continue q1.",
                "04",
                {},
                [],
                "unittest-agent",
            )
            self.assertEqual(result["next_action"]["next_node"], "04")
            current = service._workflow(workflow["id"])
            self.assertEqual(
                service._active_targets_for_node(current, "04"), {"q1"}
            )
            downstream = service.request_run(
                workflow["id"],
                "04",
                {"qubits": ["q1"]},
                "Continue only the resolved target.",
                "unittest-agent",
                autonomy_lease_id=lease["id"],
            )
            self.assertEqual(
                downstream["payload"]["parameters"]["qubits"], ["q1"]
            )
            with self.assertRaisesRegex(ServiceError, "active workflow targets"):
                service.request_run(
                    workflow["id"],
                    "04",
                    {"qubits": ["q2"]},
                    "Do not carry incomplete q2 downstream.",
                    "unittest-agent",
                    autonomy_lease_id=lease["id"],
                )
            status = service.autonomy_status(lease_id=lease["id"])
            self.assertEqual(status["incomplete_targets_by_node"], {"03a": ["q2"]})
            self.assertEqual(status["resolved_targets_by_node"]["03a"], ["q1"])

    def test_scientific_boundary_excludes_target_before_attempt_quota(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state = sample_state(0.2, 0.1)
            state["qubits"]["q2"] = deepcopy(state["qubits"]["q1"])
            settings = replace(
                make_settings(Path(folder), state),
                workflow_sequence=("07b", "06"),
            )
            service = AgentService(settings)
            workflow = start_test_workflow(service, ["q1", "q2"])
            proposal, lease = activate_lease(service, workflow["id"])
            passing = {
                "fit_quality": {
                    "results": {
                        "q1": {
                            "fit_successful": True,
                            "readout_fidelity": 0.82,
                        }
                    }
                },
                "dataset_metrics": {"qubits": {"q1": {}}},
            }
            failing = {
                "fit_quality": {
                    "results": {
                        "q2": {
                            "fit_successful": False,
                            "readout_fidelity": 0.61,
                        }
                    }
                },
                "dataset_metrics": {"qubits": {"q2": {}}},
            }
            for run_id, target, analysis_status, analysis in (
                ("q1-passing-07b", "q1", "pass", passing),
                ("q2-floor-07b", "q2", "needs_review", failing),
            ):
                service.db.execute(
                    """
                    INSERT INTO runs(
                        id, workflow_id, proposal_id, node_id, parameters_json,
                        status, analysis_status, analysis_json, autonomy_lease_id
                    ) VALUES (?, ?, ?, '07b', ?, 'completed', ?, ?, ?)
                    """,
                    (
                        run_id,
                        workflow["id"],
                        proposal["id"],
                        json_dumps({"qubits": [target]}),
                        analysis_status,
                        json_dumps(analysis),
                        lease["id"],
                    ),
                )
            service.record_decision(
                workflow["id"],
                "q2-floor-07b",
                "manual_review",
                "q2 failed at the configured safe parameter floor.",
                "07b",
                {},
                [],
                "unittest-agent",
            )
            marked = service.mark_scientifically_unmeasurable(
                workflow["id"],
                lease["id"],
                "q2-floor-07b",
                ["q2"],
                "No policy-authorized recovery remains at the safe floor.",
                "unittest-agent",
            )
            self.assertEqual(marked["targets"], ["q2"])
            self.assertEqual(
                marked["incomplete_targets_by_node"], {"07b": ["q2"]}
            )
            advanced = service.record_decision(
                workflow["id"],
                "q1-passing-07b",
                "advance",
                "Continue only the resolved q1 target.",
                "06",
                {},
                [],
                "unittest-agent",
            )
            self.assertEqual(advanced["next_action"]["next_node"], "06")
            current = service._workflow(workflow["id"])
            self.assertEqual(service._active_targets_for_node(current, "06"), {"q1"})
            status = service.autonomy_status(lease_id=lease["id"])
            self.assertEqual(
                status["scientifically_unmeasurable_targets_by_node"],
                {"07b": ["q2"]},
            )

    def test_needs_review_does_not_halt_but_missing_snapshot_does(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            service = AgentService(make_settings(root, sample_state(0.2, 0.1)))
            workflow = start_test_workflow(service)
            proposal, lease = activate_lease(service, workflow["id"])
            snapshot = service.settings.data_root / "snapshot-ok"
            snapshot.mkdir()
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, snapshot_id, snapshot_path, analysis_status,
                    autonomy_lease_id
                ) VALUES ('review-run', ?, ?, '02x', ?, 'completed', 1, ?,
                          'needs_review', ?)
                """,
                (
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": ["q1"]}),
                    str(snapshot),
                    lease["id"],
                ),
            )
            service.autonomy_watchdog_sweep()
            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"],
                "active",
            )
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, snapshot_id, snapshot_path, analysis_status,
                    autonomy_lease_id
                ) VALUES ('missing-run', ?, ?, '02x', ?, 'completed', NULL, NULL,
                          'needs_review', ?)
                """,
                (
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": ["q1"]}),
                    lease["id"],
                ),
            )
            service.autonomy_watchdog_sweep()
            halted = service.autonomy_status(lease_id=lease["id"])
            self.assertEqual(halted["status"], "halted")
            self.assertIn("Snapshot is missing", halted["stopped_reason"])

    def test_instrument_connectivity_failure_pauses_instead_of_halting(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            workflow = start_test_workflow(service)
            proposal, lease = activate_lease(service, workflow["id"])
            service._start_dashboard_session(
                workflow["id"],
                "autonomous",
                "unittest-agent",
                autonomy_lease_id=lease["id"],
            )
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
                "parameters_json, status, analysis_status, analysis_json, "
                "termination_cause, autonomy_lease_id) VALUES "
                "('instrument-offline-run', ?, ?, '02x', ?, 'failed', 'failed', ?, "
                "'instrument_unreachable', ?)",
                (
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": ["q1"]}),
                    json_dumps(
                        {
                            "analysis_status": "failed",
                            "failure_category": "instrument_unreachable",
                            "plots": [],
                        }
                    ),
                    lease["id"],
                ),
            )

            service.autonomy_watchdog_sweep()

            status = service.autonomy_status(lease_id=lease["id"])
            self.assertEqual(status["status"], "paused")
            self.assertIn("Instrument connectivity error", status["stopped_reason"])
            self.assertEqual(service._workflow(workflow["id"])["status"], "active")
            self.assertEqual(
                service.db.one(
                    "SELECT status FROM measurement_sessions WHERE workflow_id = ?",
                    (workflow["id"],),
                )["status"],
                "paused",
            )

    def test_resume_after_instrument_outage_is_not_re_paused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            workflow = start_test_workflow(service)
            proposal, lease = activate_lease(service, workflow["id"])
            service._start_dashboard_session(
                workflow["id"],
                "autonomous",
                "unittest-agent",
                autonomy_lease_id=lease["id"],
            )
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
                "parameters_json, status, analysis_status, analysis_json, "
                "termination_cause, autonomy_lease_id) VALUES "
                "('instrument-offline-run', ?, ?, '02x', ?, 'failed', 'failed', ?, "
                "'instrument_unreachable', ?)",
                (
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": ["q1"]}),
                    json_dumps(
                        {
                            "analysis_status": "failed",
                            "failure_category": "instrument_unreachable",
                            "plots": [],
                        }
                    ),
                    lease["id"],
                ),
            )
            service.autonomy_watchdog_sweep()
            session_id = service.dashboard_session_id_for_workflow(workflow["id"])

            resumed = service.resume_measurement_mode(
                workflow["id"], "on-site-operator", "恢復量測"
            )
            self.assertEqual(resumed["authorization"]["status"], "active")

            service.autonomy_watchdog_sweep()
            service.dashboard_review(session_id)

            status = service.autonomy_status(lease_id=lease["id"])
            self.assertEqual(status["status"], "active")
            self.assertEqual(
                service.dashboard_status(session_id)["status"], "active"
            )

    def test_start_run_while_instrument_paused_does_not_halt(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            workflow = start_test_workflow(service)
            proposal, lease = activate_lease(service, workflow["id"])
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
                "parameters_json, status, analysis_status, analysis_json, "
                "termination_cause, autonomy_lease_id) VALUES "
                "('instrument-offline-run', ?, ?, '02x', ?, 'failed', 'failed', ?, "
                "'instrument_unreachable', ?)",
                (
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": ["q1"]}),
                    json_dumps(
                        {
                            "analysis_status": "failed",
                            "failure_category": "instrument_unreachable",
                            "plots": [],
                        }
                    ),
                    lease["id"],
                ),
            )
            service.autonomy_watchdog_sweep()
            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "paused"
            )

            with self.assertRaisesRegex(ServiceError, "new actions are blocked"):
                service.start_authorized_run(
                    workflow["id"],
                    lease["id"],
                    "02x",
                    {"qubits": ["q1"]},
                    "Do not halt a paused lease.",
                    "unittest-agent",
                )
            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"],
                "paused",
            )

    def test_pass_backed_decision_patch_is_delegated_and_committed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            service = AgentService(make_settings(root, sample_state(0.2, 0.1)))
            workflow = start_test_workflow(service)
            proposal, lease = activate_lease(service, workflow["id"])
            snapshot = service.settings.data_root / "snapshot-pass"
            snapshot.mkdir()
            run_id = "accepted-run"
            active_hash = sha256_file(service.settings.active_state)
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, snapshot_id, snapshot_path, active_state_hash_before,
                    active_state_hash_after, analysis_status, analysis_json,
                    autonomy_lease_id
                ) VALUES (?, ?, ?, '02x', ?, 'completed', 7, ?, ?, ?, 'pass', ?, ?)
                """,
                (
                    run_id,
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": ["q1"]}),
                    str(snapshot),
                    active_hash,
                    active_hash,
                    json_dumps({"analysis_status": "pass", "plots": []}),
                    lease["id"],
                ),
            )
            patch = [
                {
                    "op": "replace",
                    "path": "/qubits/q1/xy/intermediate_frequency",
                    "value": 110_000_000,
                }
            ]
            service.record_decision(
                workflow["id"],
                run_id,
                "repeat",
                "Accepted frequency; repeat with refined acquisition settings.",
                "02x",
                {},
                patch,
                "unittest-agent",
            )
            state_proposal = service.request_state_commit(
                workflow["id"],
                patch,
                "Commit the accepted frequency.",
                "unittest-agent",
                run_id,
                autonomy_lease_id=lease["id"],
            )
            self.assertEqual(state_proposal["status"], "approved")
            self.assertFalse(state_proposal["human_approval"]["required"])
            result = service.apply_state_commit(
                state_proposal["id"], "unittest-agent"
            )
            self.assertEqual(result["status"], "committed")
            state = json.loads(service.settings.active_state.read_text(encoding="utf-8"))
            self.assertEqual(
                state["qubits"]["q1"]["xy"]["intermediate_frequency"],
                110_000_000,
            )

    def test_server_generated_bootstrap_can_commit_under_lease(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(make_settings(Path(folder), sample_state()))
            workflow = start_test_workflow(service)
            _, lease = activate_lease(service, workflow["id"])
            setup = service.request_bootstrap(
                workflow["id"],
                ["q1"],
                "unittest-agent",
                autonomy_lease_id=lease["id"],
            )
            self.assertEqual(setup["status"], "approved")
            result = service.apply_authorized_setup_state(
                setup["id"], lease["id"], "unittest-agent"
            )
            self.assertEqual(result["status"], "committed")

    def test_server_generated_07b_prerequisite_can_commit_under_lease(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            settings = replace(
                settings,
                workflow_sequence=(*settings.workflow_sequence, "07b"),
            )
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            service.db.execute(
                "UPDATE workflows SET current_node = '07b' WHERE id = ?",
                (workflow["id"],),
            )
            _, lease = activate_lease(service, workflow["id"])
            setup = service.request_07b_prerequisites(
                workflow["id"],
                ["q1"],
                "unittest-agent",
                autonomy_lease_id=lease["id"],
            )
            self.assertEqual(setup["status"], "approved")
            self.assertEqual(
                setup["payload"]["authorization_evidence"],
                "deterministic_setup",
            )
            result = service.apply_authorized_setup_state(
                setup["id"], lease["id"], "unittest-agent"
            )
            self.assertEqual(result["status"], "committed")
            state = json.loads(
                service.settings.active_state.read_text(encoding="utf-8")
            )
            self.assertEqual(
                state["qubits"]["q1"]["extras"]["readout_fidelity"], 0.0
            )

    def test_07b_tail_evidence_can_apply_fixed_power_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            settings = replace(
                settings,
                workflow_sequence=(*settings.workflow_sequence, "07b"),
            )
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            service.db.execute(
                "UPDATE workflows SET current_node = '07b' WHERE id = ?",
                (workflow["id"],),
            )
            proposal, lease = activate_lease(service, workflow["id"])
            analysis = {
                "analysis_status": "needs_review",
                "dataset_metrics": {
                    "qubits": {
                        "q1": {
                            "morphology_pass": False,
                            "morphology_failures": ["g-cloud has a long tail"],
                        }
                    }
                },
                "plots": ["figure_IQ_blobs.png"],
                "candidate_state_patch": [],
            }
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
                "parameters_json, status, analysis_status, analysis_json) "
                "VALUES ('tail-run', ?, ?, '07b', ?, 'completed', "
                "'needs_review', ?)",
                (
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": ["q1"]}),
                    json_dumps(analysis),
                ),
            )
            setup = service.request_07b_tail_power_reduction(
                workflow["id"],
                ["q1"],
                "tail-run",
                "unittest-agent",
                autonomy_lease_id=lease["id"],
            )
            self.assertEqual(setup["status"], "approved")
            reduced = setup["payload"]["patch"][0]["value"]
            self.assertAlmostEqual(reduced, 0.1 / 2**0.5)
            service.apply_authorized_setup_state(
                setup["id"], lease["id"], "unittest-agent"
            )
            state = json.loads(settings.active_state.read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                state["qubits"]["q1"]["resonator"]["operations"]["readout"][
                    "amplitude"
                ],
                0.1 / 2**0.5,
            )

    def test_reopen_07b_cancels_pending_next_node_lease(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            settings = replace(
                settings,
                workflow_sequence=(*settings.workflow_sequence, "07b", "06"),
            )
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            service.db.execute(
                "UPDATE workflows SET current_node = '06' WHERE id = ?",
                (workflow["id"],),
            )
            pending = service.request_autonomy_lease(
                workflow["id"],
                "unittest-agent",
                AUTO_PHRASE,
                "Pending next-node lease.",
            )
            analysis = {
                "analysis_status": "needs_review",
                "dataset_metrics": {
                    "qubits": {"q1": {"morphology_pass": False}}
                },
                "plots": ["figure_IQ_blobs.png"],
            }
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
                "parameters_json, status, analysis_status, analysis_json) "
                "VALUES ('retracted-07b', ?, ?, '07b', ?, 'completed', "
                "'needs_review', ?)",
                (
                    workflow["id"],
                    pending["id"],
                    json_dumps({"qubits": ["q1"]}),
                    json_dumps(analysis),
                ),
            )
            result = service.reopen_07b_after_morphology_rule_change(
                workflow["id"],
                "retracted-07b",
                "Operator rejected long-tail morphology.",
                "unittest-agent",
            )
            self.assertEqual(result["workflow"]["current_node"], "07b")
            self.assertEqual(
                service.autonomy_status(lease_id=pending["autonomy_lease_id"])[
                    "status"
                ],
                "revoked",
            )
            self.assertEqual(service.proposal(pending["id"])["status"], "cancelled")

    def test_state_hash_conflict_halts_lease(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            service = AgentService(make_settings(root, sample_state(0.2, 0.1)))
            workflow = start_test_workflow(service)
            proposal, lease = activate_lease(service, workflow["id"])
            snapshot = service.settings.data_root / "snapshot-pass"
            snapshot.mkdir()
            run_id = "conflict-run"
            active_hash = sha256_file(service.settings.active_state)
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, snapshot_id, snapshot_path, active_state_hash_before,
                    active_state_hash_after, analysis_status, analysis_json,
                    autonomy_lease_id
                ) VALUES (?, ?, ?, '02x', ?, 'completed', 8, ?, ?, ?, 'pass', ?, ?)
                """,
                (
                    run_id,
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": ["q1"]}),
                    str(snapshot),
                    active_hash,
                    active_hash,
                    json_dumps({"analysis_status": "pass", "plots": []}),
                    lease["id"],
                ),
            )
            patch = [
                {
                    "op": "replace",
                    "path": "/qubits/q1/xy/intermediate_frequency",
                    "value": 110_000_000,
                }
            ]
            service.record_decision(
                workflow["id"], run_id, "repeat", "Accepted.", "02x", {}, patch,
                "unittest-agent",
            )
            state_proposal = service.request_state_commit(
                workflow["id"], patch, "Commit.", "unittest-agent", run_id,
                autonomy_lease_id=lease["id"],
            )
            changed = sample_state(0.2, 0.1)
            changed["qubits"]["q1"]["xy"]["intermediate_frequency"] = 105_000_000
            atomic_write_json(service.settings.active_state, changed)
            with self.assertRaises(StateError):
                service.apply_state_commit(state_proposal["id"], "unittest-agent")
            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "halted"
            )

    def test_needs_review_commit_is_refused_without_pausing_lease(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            service = AgentService(make_settings(root, sample_state(0.2, 0.1)))
            workflow = start_test_workflow(service)
            proposal, lease = activate_lease(service, workflow["id"])
            snapshot = service.settings.data_root / "snapshot-review"
            snapshot.mkdir()
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, snapshot_id, snapshot_path, analysis_status,
                    analysis_json, autonomy_lease_id
                ) VALUES ('needs-review-run', ?, ?, '02x', ?, 'completed', 9, ?,
                          'needs_review', ?, ?)
                """,
                (
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": ["q1"]}),
                    str(snapshot),
                    json_dumps({"analysis_status": "needs_review"}),
                    lease["id"],
                ),
            )
            patch = [
                {
                    "op": "replace",
                    "path": "/qubits/q1/xy/intermediate_frequency",
                    "value": 110_000_000,
                }
            ]
            with self.assertRaises(AutonomyEvidenceError):
                service.commit_authorized_state(
                    workflow["id"],
                    lease["id"],
                    patch,
                    "Do not commit low-confidence evidence.",
                    "unittest-agent",
                    "needs-review-run",
                )
            self.assertEqual(
                service.autonomy_status(lease_id=lease["id"])["status"], "active"
            )

    def test_three_manual_controls_and_review_only_route(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            _, lease = activate_lease(service, workflow["id"])
            paused = service.pause_autonomy(
                lease["id"], "unit-test-human", "Operator pause."
            )
            self.assertEqual(paused["status"], "paused")
            resumed = service.resume_autonomy(
                lease["id"], "unit-test-human", "恢復量測"
            )
            self.assertEqual(resumed["status"], "active")
            stopped = service.stop_autonomy(
                lease["id"], "unit-test-human", "Operator stop."
            )
            self.assertEqual(stopped["status"], "revoked")
            paths = {route.path for route in create_approval_app(settings).routes}
            self.assertIn("/autonomy/{lease_id}", paths)

    def test_emergency_control_stops_active_worker_and_schedules_force(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            workflow = start_test_workflow(service)
            proposal, lease = activate_lease(service, workflow["id"])
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, pid, analysis_status, autonomy_lease_id
                ) VALUES ('active-emergency-run', ?, ?, '02x', ?, 'running', 999999,
                          'not_started', ?)
                """,
                (
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": ["q1"]}),
                    lease["id"],
                ),
            )
            with patch.object(
                service.runner,
                "stop",
                return_value={"run_id": "active-emergency-run", "status": "stopping"},
            ) as graceful, patch.object(service, "_schedule_force_stop") as schedule:
                result = service.stop_autonomy(
                    lease["id"],
                    "unit-test-human",
                    "Emergency stop test.",
                    emergency=True,
                )
            self.assertEqual(result["status"], "emergency_stopped")
            graceful.assert_called_once_with(
                "active-emergency-run", "unit-test-human"
            )
            schedule.assert_called_once_with(
                "active-emergency-run", "unit-test-human"
            )


if __name__ == "__main__":
    unittest.main()
