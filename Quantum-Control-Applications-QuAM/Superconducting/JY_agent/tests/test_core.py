from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from collections import UserList
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlencode

import numpy as np
import xarray as xr
from starlette.requests import Request

from jy_agent.approval_web import handle_browser_approval
from jy_agent.analysis import (
    SnapshotAnalyzer,
    _feature_fwhm,
    _select_higher_transition_pair,
    _spectroscopy_peak_candidates,
)
from jy_agent.analysis_02c import (
    analyze_02c_transitions,
    node_selected_readout_powers_dbm,
    selected_readout_power_dbm,
)
from jy_agent.config import Settings
from jy_agent.policy import PolicyEngine, PolicyError
from jy_agent.reports import lightweight_report
from jy_agent.service import AgentService, AutonomyScopeError
from jy_agent.state import (
    bootstrap_patch,
    commit_state,
    load_state,
    recorded_updates_to_patch,
)
from jy_agent.util import atomic_write_json, json_compatible, json_dumps, sha256_file
from jy_agent.worker import _assert_active_state_source, _restore_state_if_changed


AGENT_ROOT = Path(__file__).resolve().parents[1]
SUPERCONDUCTING_ROOT = AGENT_ROOT.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
ENTRY_PHRASE = "進入 JY 量測模式"
EXIT_PHRASE = "退出 JY 量測模式"


def sample_state(x180: float = 0.0, x90: float = 0.0) -> dict:
    return {
        "qubits": {
            "q1": {
                "xy": {
                    "operations": {
                        "x180_DragCosine": {
                            "amplitude": x180,
                            "__class__": "quam.components.pulses.DragCosinePulse",
                        },
                        "x90_DragCosine": {
                            "amplitude": x90,
                            "__class__": "quam.components.pulses.DragCosinePulse",
                        },
                        "x180": "#./x180_DragCosine",
                        "x90": "#./x90_DragCosine",
                        "saturation": {
                            "amplitude": 0.1,
                            "__class__": "quam.components.pulses.SquarePulse",
                        },
                    },
                    "intermediate_frequency": 100_000_000,
                    "__class__": "quam.components.channels.MWChannel",
                },
                "resonator": {
                    "intermediate_frequency": 20_000_000,
                    "operations": {
                        "readout": {
                            "amplitude": 0.1,
                            "full_scale_power_dbm": 5,
                            "integration_weights_angle": 0.0,
                        }
                    },
                },
                "extras": {},
            }
        },
        "active_qubit_names": ["q1"],
    }


def sample_multiplex_state_and_wiring() -> tuple[dict, dict, list[str]]:
    targets = [f"q{index}" for index in range(3, 9)]
    frequencies = {
        "q3": -75_000_000,
        "q4": 77_000_000,
        "q5": -24_000_000,
        "q6": -25_000_000,
        "q7": 76_000_000,
        "q8": -77_000_000,
    }
    template = sample_state(0.2, 0.1)["qubits"]["q1"]
    state = {"qubits": {}, "active_qubit_names": list(targets)}
    wiring = {"wiring": {"qubits": {}}}
    for name in targets:
        qubit = deepcopy(template)
        qubit["resonator"]["intermediate_frequency"] = frequencies[name]
        state["qubits"][name] = qubit
        line = 7 if int(name[1:]) <= 5 else 8
        wiring["wiring"]["qubits"][name] = {
            "rr": {
                "opx_input": f"#/ports/mw_inputs/con1/{line}/1",
                "opx_output": f"#/ports/mw_outputs/con1/{line}/1",
            }
        }
    return state, wiring, targets


def make_settings(temp_root: Path, state: dict | None = None) -> Settings:
    active_state = temp_root / "state.json"
    atomic_write_json(active_state, state or sample_state())
    wiring_path = temp_root / "wiring.json"
    atomic_write_json(wiring_path, {})
    qualibrate_config_path = temp_root / "config.toml"
    qualibrate_config_path.write_text("# synthetic test config\n", encoding="utf-8")
    data_root = temp_root / "Data"
    data_root.mkdir()
    return Settings(
        agent_root=AGENT_ROOT,
        superconducting_root=SUPERCONDUCTING_ROOT,
        calibration_graph=SUPERCONDUCTING_ROOT / "calibration_graph",
        data_root=data_root,
        quam_state_root=temp_root,
        active_state=active_state,
        wiring_path=wiring_path,
        qualibrate_config_path=qualibrate_config_path,
        qualibrate_project="unittest",
        runtime=temp_root / "runtime",
        policy_path=AGENT_ROOT / "rules" / "policies.yaml",
        playbook_path=AGENT_ROOT / "rules" / "PLAYBOOK.md",
        host="127.0.0.1",
        port=8765,
        mcp_path="/mcp",
        qualibrate_python=Path(sys.executable),
        workflow_sequence=("02x", "02c", "02a", "03a", "04", "05"),
        require_explicit_qubits=True,
        measurement_mode_entry_phrase=ENTRY_PHRASE,
        measurement_mode_exit_phrase=EXIT_PHRASE,
        fresh_qubit_defaults=False,
    )


def start_test_workflow(
    service: AgentService, targets: list[str] | None = None
) -> dict:
    return service.start_workflow(
        targets or ["q1"],
        {},
        "unittest",
        ENTRY_PHRASE,
    )


class CoreTests(unittest.TestCase):
    def test_english_entry_and_shutdown_aliases_are_operational(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = replace(
                make_settings(Path(folder), sample_state(0.2, 0.1)),
                measurement_mode_entry_phrases=(
                    ENTRY_PHRASE,
                    "Enter JY measurement mode",
                ),
                measurement_mode_shutdown_phrases=(
                    EXIT_PHRASE,
                    "End measurement",
                ),
            )
            service = AgentService(settings)
            entered = service.enter_measurement_mode(
                "english-client", "Enter JY measurement mode", ["q1"]
            )
            stopped = service.leave_measurement_mode(
                entered["workflow"]["id"], "english-client", "End measurement"
            )
            self.assertEqual(stopped["status"], "stopped")

    def test_conversational_instrument_failure_pauses_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            entered = service.enter_measurement_mode(
                "unittest-agent", ENTRY_PHRASE, ["q1"]
            )
            workflow_id = entered["workflow"]["id"]
            proposal = service.request_run(
                workflow_id,
                "02x",
                {"qubits": ["q1"]},
                "Synthetic instrument outage.",
                "unittest-agent",
            )
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
                "parameters_json, status, analysis_status, analysis_json, "
                "termination_cause) VALUES "
                "('conversation-instrument-offline', ?, ?, '02x', '{}', "
                "'failed', 'failed', ?, 'instrument_unreachable')",
                (
                    workflow_id,
                    proposal["id"],
                    json_dumps(
                        {
                            "analysis_status": "failed",
                            "failure_category": "instrument_unreachable",
                        }
                    ),
                ),
            )

            polled = service.poll_run("conversation-instrument-offline")

            self.assertEqual(service._workflow(workflow_id)["status"], "paused")
            self.assertEqual(polled["operator_handoff"]["reply"], "恢復量測")
            self.assertIn("完成後回傳: 恢復量測", polled["operator_handoff"]["chat"])
            self.assertIn(
                "如需結束量測，請於網頁首頁結束量測後再對話輸入「結束量測」。",
                polled["operator_handoff"]["chat"],
            )
            self.assertEqual(
                service.dashboard_status(entered["dashboard"]["id"])["status"],
                "paused",
            )

            resumed = service.resume_measurement_mode(
                workflow_id,
                "on-site-operator",
                "恢復量測",
            )
            self.assertEqual(resumed["status"], "active")
            self.assertEqual(resumed["current_node"], "02x")
            self.assertEqual(
                resumed["resume_policy"],
                "restart_current_node_as_new_run",
            )
            self.assertEqual(
                service.dashboard_status(entered["dashboard"]["id"])["status"],
                "active",
            )

    def test_resume_measurement_refuses_retained_hardware_lock(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            service = AgentService(settings)
            entered = service.enter_measurement_mode(
                "unittest-agent", ENTRY_PHRASE, ["q1"]
            )
            workflow_id = entered["workflow"]["id"]
            service.db.execute(
                "UPDATE workflows SET status = 'paused' WHERE id = ?",
                (workflow_id,),
            )
            atomic_write_json(settings.lock_path, {"run_id": "unverified-worker-exit"})

            with self.assertRaisesRegex(Exception, "recovery-only quarantine"):
                service.resume_measurement_mode(
                    workflow_id,
                    "on-site-operator",
                    "恢復量測",
                )

    def test_recover_close_converges_open_workflow_without_deleting_lock(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            service = AgentService(settings)
            entered = service.enter_measurement_mode(
                "unittest-agent", ENTRY_PHRASE, ["q1"]
            )
            atomic_write_json(settings.lock_path, {"run_id": "retained-lock"})

            result = service.prepare_recovery_close("unit-test-operator")

            self.assertEqual(result["status"], "recovery_required")
            self.assertTrue(result["services_may_stop"])
            self.assertTrue(settings.lock_path.exists())
            self.assertEqual(
                service._workflow(entered["workflow"]["id"])["status"],
                "recovery_required",
            )

    def test_concurrent_clients_cannot_create_two_open_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            services = (AgentService(settings), AgentService(settings))
            barrier = threading.Barrier(2)

            def start(service: AgentService) -> str:
                barrier.wait(timeout=5)
                try:
                    service.start_workflow(
                        ["q1"], {}, "concurrent-client", ENTRY_PHRASE
                    )
                    return "created"
                except Exception as exc:
                    return f"rejected:{exc}"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(start, services))

            self.assertEqual(outcomes.count("created"), 1)
            self.assertEqual(
                sum(value.startswith("rejected:") for value in outcomes), 1
            )
            workflows = services[0].db.all(
                "SELECT id FROM workflows WHERE status IN ('active', 'paused')"
            )
            events = services[0].db.all(
                "SELECT id FROM events WHERE event_type = 'workflow_started'"
            )
            self.assertEqual(len(workflows), 1)
            self.assertEqual(len(events), 1)

    def test_conversational_run_can_select_registered_node_outside_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            workflow = start_test_workflow(service)
            with self.assertRaisesRegex(Exception, "Current workflow node"):
                service.request_run(
                    workflow["id"],
                    "02a",
                    {"qubits": ["q1"]},
                    "Sequence-bound request must stay on 02x.",
                    "unittest",
                )
            proposal = service.request_conversational_run(
                workflow["id"],
                "02a",
                {"qubits": ["q1"]},
                "Discuss and run one registered dressed-resonator experiment.",
                "unittest",
            )
            self.assertEqual(proposal["payload"]["node_id"], "02a")
            self.assertTrue(proposal["payload"]["conversational"])
            self.assertEqual(proposal["status"], "pending")
            service.db.execute(
                "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
                "parameters_json, status, analysis_status, analysis_json) "
                "VALUES (?, ?, ?, '02a', ?, 'completed', 'pass', ?)",
                (
                    "conversational-run",
                    workflow["id"],
                    proposal["id"],
                    json.dumps({"qubits": ["q1"]}),
                    json.dumps({"fit_quality": {"failures": []}}),
                ),
            )
            decision = service.record_decision(
                workflow["id"],
                "conversational-run",
                "advance",
                "The one-off result is usable; discuss a different experiment next.",
                "04",
                {},
                [],
                "unittest",
            )
            self.assertEqual(decision["mode"], "conversational")
            self.assertEqual(decision["next_action"]["next_node"], "04")
            self.assertEqual(service.status(workflow["id"])["workflow"]["current_node"], "02x")

    def test_json_compatible_unwraps_quam_style_nested_lists(self) -> None:
        class QuamListLike(UserList):
            pass

        converted = json_compatible(
            {
                "#/qubits/q1/resonator/confusion_matrix": {
                    "old": QuamListLike(
                        [QuamListLike([0.9, 0.1]), QuamListLike([0.2, 0.8])]
                    ),
                    "new": QuamListLike(
                        [QuamListLike([0.88, 0.12]), QuamListLike([0.18, 0.82])]
                    ),
                }
            }
        )
        self.assertEqual(converted["#/qubits/q1/resonator/confusion_matrix"]["new"][1][1], 0.82)
        self.assertEqual(json.loads(json.dumps(converted)), converted)

    def test_bootstrap_uses_half_x180_limit_and_half_x90(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder))
            policy = PolicyEngine(settings)
            patch, details = bootstrap_patch(
                load_state(settings.active_state), ["q1"], policy.raw
            )
            values = {item["path"]: item["value"] for item in patch}
            self.assertEqual(
                values["/qubits/q1/xy/operations/x180_DragCosine/amplitude"],
                0.5,
            )
            self.assertEqual(
                values["/qubits/q1/xy/operations/x90_DragCosine/amplitude"],
                0.25,
            )
            self.assertEqual(details["q1"]["max_x180_wf_amplitude"], 1.0)

    def test_bootstrap_preserves_nonzero_amplitudes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            policy = PolicyEngine(settings)
            patch, _ = bootstrap_patch(
                load_state(settings.active_state), ["q1"], policy.raw
            )
            self.assertEqual(patch, [])

    def test_03a_rejects_if_sweep_outside_400_mhz(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state = sample_state(0.2, 0.1)
            state["qubits"]["q1"]["xy"]["intermediate_frequency"] = 300_000_000
            settings = make_settings(Path(folder), state)
            policy = PolicyEngine(settings)
            with self.assertRaisesRegex(PolicyError, "±400 MHz"):
                policy.validate_run(
                    "03a",
                    {
                        "qubits": ["q1"],
                        "frequency_span_in_mhz": 300,
                        "frequency_step_in_mhz": 1,
                    },
                )

    def test_02x_rejects_span_above_60_mhz(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            policy = PolicyEngine(settings)
            with self.assertRaisesRegex(PolicyError, "60 MHz"):
                policy.validate_run(
                    "02x",
                    {
                        "qubits": ["q1"],
                        "frequency_span_in_mhz": 80,
                        "frequency_step_in_mhz": 0.1,
                    },
                )

    def test_02x_rejects_sweep_that_includes_upconverter(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state = sample_state(0.2, 0.1)
            state["qubits"]["q1"]["resonator"]["intermediate_frequency"] = 10_000_000
            settings = make_settings(Path(folder), state)
            policy = PolicyEngine(settings)
            with self.assertRaisesRegex(PolicyError, "upconverter"):
                policy.validate_run(
                    "02x",
                    {
                        "qubits": ["q1"],
                        "frequency_span_in_mhz": 20,
                        "frequency_step_in_mhz": 0.05,
                    },
                )

    def test_02x_allows_20_mhz_span_away_from_upconverter(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            policy = PolicyEngine(settings)
            merged, _ = policy.validate_run(
                "02x",
                {
                    "qubits": ["q1"],
                    "frequency_span_in_mhz": 20,
                    "frequency_step_in_mhz": 0.05,
                },
            )
            self.assertEqual(merged["frequency_span_in_mhz"], 20)

    def test_02c_rejects_more_than_30_power_points(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            policy = PolicyEngine(settings)
            with self.assertRaisesRegex(PolicyError, "num_power_points"):
                policy.validate_run(
                    "02c",
                    {
                        "qubits": ["q1"],
                        "num_power_points": 40,
                        "num_averages": 100,
                    },
                )

    def test_02c_rejects_more_than_100_averages(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            policy = PolicyEngine(settings)
            with self.assertRaisesRegex(PolicyError, "num_averages"):
                policy.validate_run(
                    "02c",
                    {
                        "qubits": ["q1"],
                        "num_power_points": 30,
                        "num_averages": 200,
                    },
                )

    def test_02c_allows_30_power_points_and_100_averages(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            policy = PolicyEngine(settings)
            merged, _ = policy.validate_run(
                "02c",
                {
                    "qubits": ["q1"],
                    "num_power_points": 30,
                    "num_averages": 100,
                },
            )
            self.assertEqual(merged["num_power_points"], 30)
            self.assertEqual(merged["num_averages"], 100)

    def test_04_rejects_waveform_limit_violation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.5, 0.25))
            policy = PolicyEngine(settings)
            with self.assertRaisesRegex(PolicyError, "reduce max_amp_factor"):
                policy.validate_run(
                    "04",
                    {
                        "qubits": ["q1"],
                        "max_amp_factor": 3.0,
                    },
                )

    def test_05_accepts_defaults_and_validates_wait_grid(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            policy = PolicyEngine(settings)
            parameters, warnings = policy.validate_run("05", {"qubits": ["q1"]})
            self.assertEqual(parameters["min_wait_time_in_ns"], 16)
            self.assertEqual(parameters["max_wait_time_in_ns"], 200000)
            self.assertIn("joint flux mode", warnings[0])
            with self.assertRaisesRegex(PolicyError, "multiples of 4 ns"):
                policy.validate_run(
                    "05",
                    {"qubits": ["q1"], "wait_time_step_in_ns": 2001},
                )

    def test_05_requires_calibrated_x180(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state())
            policy = PolicyEngine(settings)
            with self.assertRaisesRegex(PolicyError, "calibrate it before T1"):
                policy.validate_run("05", {"qubits": ["q1"]})

    def test_t1_state_patch_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            policy = PolicyEngine(settings)
            valid = [
                {"op": "add", "path": "/qubits/q1/T1", "value": 35e-6}
            ]
            policy.validate_state_patch(valid)
            with self.assertRaisesRegex(PolicyError, "finite and positive"):
                policy.validate_state_patch(
                    [{"op": "add", "path": "/qubits/q1/T1", "value": 0.0}]
                )

    def test_bare_resonator_frequency_patch_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            policy = PolicyEngine(settings)
            valid = [
                {
                    "op": "add",
                    "path": "/qubits/q1/extras/bare_resonator_freq",
                    "value": 6_048_286_875.0,
                }
            ]
            policy.validate_state_patch(valid)
            with self.assertRaisesRegex(PolicyError, "finite and positive"):
                policy.validate_state_patch(
                    [
                        {
                            "op": "add",
                            "path": "/qubits/q1/extras/bare_resonator_freq",
                            "value": 0.0,
                        }
                    ]
                )

    def test_recorded_qualibrate_updates_convert_to_targeted_patch(self) -> None:
        state = sample_state(0.2, 0.1)
        state["qubits"]["q1"]["extras"]["dressed_resonator_freq"] = 6.048e9
        updates = {
            "#/qubits/q1/resonator/intermediate_frequency": {
                "key": "#/qubits/q1/resonator/intermediate_frequency",
                "attr": "intermediate_frequency",
                "old": 20_000_000,
                "new": 22_400_000,
            },
            "#/qubits/q1/extras/dressed_resonator_freq": {
                "key": "#/qubits/q1/extras/dressed_resonator_freq",
                "attr": "dressed_resonator_freq",
                "old": 6.048e9,
                "new": 6.0504e9,
            },
        }
        patch = recorded_updates_to_patch(updates, state)
        self.assertEqual(
            patch,
            [
                {
                    "op": "replace",
                    "path": "/qubits/q1/extras/dressed_resonator_freq",
                    "value": 6.0504e9,
                },
                {
                    "op": "replace",
                    "path": "/qubits/q1/resonator/intermediate_frequency",
                    "value": 22_400_000,
                },
            ],
        )

    def test_q3_q8_multiplex_workflow_forces_every_run_to_multiplex(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            workflow = service.start_workflow(
                targets,
                {"multiplexed": True},
                "unittest",
                ENTRY_PHRASE,
            )
            proposal = service.request_run(
                workflow["id"],
                "02x",
                {"qubits": targets},
                "Validate q3-q8 multiplex proposal.",
                "unittest",
            )
            self.assertTrue(proposal["payload"]["parameters"]["multiplexed"])
            self.assertTrue(
                any(
                    "6 qubits across 2 readout lines" in warning
                    for warning in proposal["payload"]["warnings"]
                )
            )
            with self.assertRaisesRegex(
                Exception,
                "requires multiplexed=true",
            ):
                service.request_run(
                    workflow["id"],
                    "02x",
                    {"qubits": targets, "multiplexed": False},
                    "Attempt to disable workflow multiplexing.",
                    "unittest",
                )

    def test_multiplex_rejects_resonator_collision_on_shared_line(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            state["qubits"]["q4"]["resonator"]["intermediate_frequency"] = (
                state["qubits"]["q3"]["resonator"]["intermediate_frequency"]
                + 500_000
            )
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            policy = PolicyEngine(settings)
            with self.assertRaisesRegex(PolicyError, "only 0.5 MHz apart"):
                policy.validate_run(
                    "02x",
                    {"qubits": targets, "multiplexed": True},
                )

    def test_state_commit_checks_hash_and_creates_backup(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = make_settings(root, sample_state(0.2, 0.1))
            before_hash = sha256_file(settings.active_state)
            result = commit_state(
                settings.active_state,
                [
                    {
                        "op": "replace",
                        "path": "/qubits/q1/xy/intermediate_frequency",
                        "value": 110_000_000,
                    }
                ],
                root / "backups",
                before_hash,
            )
            self.assertNotEqual(result["before_hash"], result["after_hash"])
            self.assertTrue(Path(result["backup_path"]).is_file())
            self.assertEqual(
                load_state(settings.active_state)["qubits"]["q1"]["xy"][
                    "intermediate_frequency"
                ],
                110_000_000,
            )

    def test_state_commit_is_blocked_by_hardware_lock_at_request_and_apply(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            patch_payload = [
                {
                    "op": "replace",
                    "path": "/qubits/q1/xy/intermediate_frequency",
                    "value": 110_000_000,
                }
            ]
            settings.lock_path.write_text('{"run_id":"guard"}', encoding="utf-8")
            with self.assertRaisesRegex(Exception, "hardware lock"):
                service.request_state_commit(
                    workflow["id"], patch_payload, "Blocked request.", "unittest"
                )
            settings.lock_path.unlink()
            proposal = service.request_state_commit(
                workflow["id"], patch_payload, "Approved before lock.", "unittest"
            )
            service._approve_pending_proposal(
                proposal["id"], "unit-test-human", "local_browser"
            )
            before_hash = sha256_file(settings.active_state)
            settings.lock_path.write_text('{"run_id":"guard"}', encoding="utf-8")
            with self.assertRaisesRegex(Exception, "hardware lock"):
                service.apply_state_commit(proposal["id"], "unittest")
            self.assertEqual(sha256_file(settings.active_state), before_hash)

    def test_worker_guard_restores_changed_active_state(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            active = root / "state.json"
            recovery = root / "recovery.json"
            atomic_write_json(active, sample_state(0.2, 0.1))
            shutil.copy2(active, recovery)
            before_hash = sha256_file(active)
            changed = sample_state(0.2, 0.1)
            changed["qubits"]["q1"]["xy"]["intermediate_frequency"] = 123
            atomic_write_json(active, changed)
            observed, restored, error = _restore_state_if_changed(
                active, recovery, before_hash
            )
            self.assertNotEqual(observed, before_hash)
            self.assertTrue(restored)
            self.assertIsNone(error)
            self.assertEqual(sha256_file(active), before_hash)

    def test_worker_requires_same_state_source_as_quam_load(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected_root = root / "expected"
            expected_root.mkdir()
            expected_state = expected_root / "state.json"
            other_state = root / "other" / "state.json"
            other_state.parent.mkdir()
            atomic_write_json(expected_state, sample_state(0.2, 0.1))
            atomic_write_json(other_state, sample_state(0.2, 0.1))

            with patch(
                "quam_libs.components.quam_root.QuAM.get_quam_state_path",
                return_value=expected_root,
            ):
                _assert_active_state_source(expected_state)
                with self.assertRaisesRegex(
                    RuntimeError, "ACTIVE_STATE_SOURCE_MISMATCH"
                ):
                    _assert_active_state_source(other_state)

    def test_idempotent_mutation_replays_only_matching_completed_request(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            calls: list[int] = []

            def invoke() -> dict:
                calls.append(1)
                return {"status": "done", "count": len(calls)}

            first = service.idempotent_call(
                "unit-test", "operation-1234", {"value": 1}, invoke
            )
            replay = service.idempotent_call(
                "unit-test", "operation-1234", {"value": 1}, invoke
            )
            self.assertEqual(first, replay)
            self.assertEqual(len(calls), 1)
            with self.assertRaisesRegex(Exception, "different request"):
                service.idempotent_call(
                    "unit-test", "operation-1234", {"value": 2}, invoke
                )

    def test_run_proposal_is_pending_and_not_self_approved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            proposal = service.request_run(
                workflow["id"],
                "02x",
                {"qubits": ["q1"]},
                "Locate the bare resonator.",
                "unittest",
            )
            self.assertEqual(proposal["status"], "pending")
            self.assertTrue(proposal["human_approval"]["required"])
            with self.assertRaisesRegex(Exception, "not been approved"):
                service.execute_run(proposal["id"], "unittest")

    def test_browser_approval_requires_local_human_form_submission(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            proposal = service.request_run(
                workflow["id"],
                "02x",
                {"qubits": ["q1"]},
                "Locate the bare resonator.",
                "unittest",
            )
            proposal_id = proposal["id"]
            self.assertEqual(
                proposal["human_approval"]["browser_url"],
                f"http://127.0.0.1:8766/approve/{proposal_id}",
            )
            self.assertEqual(
                proposal["human_approval"]["confirmation"],
                f"APPROVE {proposal_id}",
            )

            def request(method: str, body: bytes = b"", origin: str | None = None):
                headers = []
                if method == "POST":
                    headers.append(
                        (b"content-type", b"application/x-www-form-urlencoded")
                    )
                    if origin is None:
                        origin = "http://127.0.0.1:8766"
                if origin is not None:
                    headers.append((b"origin", origin.encode("ascii")))

                async def receive():
                    return {
                        "type": "http.request",
                        "body": body,
                        "more_body": False,
                    }

                return Request(
                    {
                        "type": "http",
                        "http_version": "1.1",
                        "method": method,
                        "scheme": "http",
                        "path": f"/approve/{proposal_id}",
                        "raw_path": f"/approve/{proposal_id}".encode("ascii"),
                        "root_path": "",
                        "query_string": b"",
                        "headers": headers,
                        "client": ("127.0.0.1", 50123),
                        "server": ("127.0.0.1", 8765),
                        "path_params": {"proposal_id": proposal_id},
                    },
                    receive,
                )

            get_response = asyncio.run(
                handle_browser_approval(request("GET"), service)
            )
            self.assertEqual(get_response.status_code, 200)
            self.assertIn(
                "完整提案資料", get_response.body.decode("utf-8")
            )
            self.assertIn(
                "frame-ancestors 'none'",
                get_response.headers["content-security-policy"],
            )

            token = service.approval_csrf_token(proposal_id)
            form = urlencode(
                {
                    "csrf_token": token,
                    "confirmation": f"APPROVE {proposal_id}",
                }
            ).encode("utf-8")
            rejected = asyncio.run(
                handle_browser_approval(
                    request("POST", form, "http://malicious.invalid"),
                    service,
                )
            )
            self.assertEqual(rejected.status_code, 403)
            self.assertEqual(service.proposal(proposal_id)["status"], "pending")

            approved = asyncio.run(
                handle_browser_approval(
                    request("POST", form),
                    service,
                )
            )
            self.assertEqual(approved.status_code, 200)
            stored = service.proposal(proposal_id)
            self.assertEqual(stored["status"], "approved")
            self.assertIn("via local browser", stored["approved_by"])
            event = service.db.one(
                "SELECT payload_json FROM events WHERE event_type = ? "
                "ORDER BY id DESC LIMIT 1",
                ("proposal_approved",),
            )
            self.assertEqual(
                json.loads(event["payload_json"])["approval_method"],
                "local_browser",
            )

    def test_run_targets_must_match_workflow_targets(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state = sample_state(0.2, 0.1)
            state["qubits"]["q2"] = deepcopy(state["qubits"]["q1"])
            settings = make_settings(Path(folder), state)
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            # Subgroup nodes now accept any subset of the active targets, so the
            # rule this pins is "a qubit outside the workflow is refused", by
            # exception type rather than by wording.
            with self.assertRaises(AutonomyScopeError) as refused:
                service.request_run(
                    workflow["id"],
                    "02x",
                    {"qubits": ["q2"]},
                    "Attempt an out-of-scope target.",
                    "unittest",
                )
            self.assertIn("q1", str(refused.exception))

    def test_poll_failed_run_backfills_failed_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            proposal = service.request_run(
                workflow["id"],
                "02x",
                {"qubits": ["q1"]},
                "Offline execution failure test.",
                "unittest",
            )
            run_id = "offline-failed-run"
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, error
                ) VALUES (?, ?, ?, '02x', ?, 'failed', 'not_started', ?)
                """,
                (
                    run_id,
                    workflow["id"],
                    proposal["id"],
                    json.dumps({"qubits": ["q1"]}),
                    "Compilation failed.",
                ),
            )
            result = service.poll_run(run_id)
            self.assertEqual(result["analysis_status"], "failed")
            self.assertEqual(result["analysis"]["analysis_status"], "failed")
            self.assertEqual(result["analysis"]["candidate_state_patch"], [])
            self.assertIn("Compilation failed", result["analysis"]["error"])

    def test_a_run_decision_cannot_be_reused_to_skip_a_node(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            proposal = service.request_run(
                workflow["id"],
                "02x",
                {"qubits": ["q1"]},
                "Locate the bare resonator.",
                "unittest",
            )
            run_id = "offline-run"
            # Advancing a node now requires per-qubit evidence, so the run must
            # carry an analysis document that resolves q1, not just 'pass'.
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, analysis_json
                ) VALUES (?, ?, ?, '02x', ?, 'completed', 'pass', ?)
                """,
                (
                    run_id,
                    workflow["id"],
                    proposal["id"],
                    json.dumps({"qubits": ["q1"]}),
                    json.dumps(
                        {
                            "analysis_status": "pass",
                            "node_id": "02x",
                            "passing_targets": ["q1"],
                            "fit_quality": {
                                "results": {"q1": {"fit_successful": True}}
                            },
                            "dataset_metrics": {
                                "qubits": {
                                    "q1": {"robust_snr": 20.0, "edge_fraction": 0.4}
                                }
                            },
                        }
                    ),
                ),
            )
            service.record_decision(
                workflow["id"],
                run_id,
                "advance",
                "Evidence passes.",
                "02c",
                {},
                [],
                "unittest",
            )
            with self.assertRaisesRegex(Exception, "does not match current"):
                service.record_decision(
                    workflow["id"],
                    run_id,
                    "advance",
                    "Reuse old evidence.",
                    "02c",
                    {},
                    [],
                    "unittest",
                )

    def test_04_advance_moves_workflow_to_05(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            service.db.execute(
                "UPDATE workflows SET current_node = '04' WHERE id = ?",
                (workflow["id"],),
            )
            proposal = service.request_run(
                workflow["id"],
                "04",
                {"qubits": ["q1"]},
                "Offline transition test.",
                "unittest",
            )
            run_id = "offline-04-run"
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, analysis_json
                ) VALUES (?, ?, ?, '04', ?, 'completed', 'pass', ?)
                """,
                (
                    run_id,
                    workflow["id"],
                    proposal["id"],
                    json.dumps({"qubits": ["q1"]}),
                    json.dumps(
                        {
                            "fit_quality": {
                                "results": {"q1": {"Pi_amplitude": 0.2}}
                            },
                            "dataset_metrics": {
                                "qubits": {
                                    "q1": {
                                        "edge_fraction": 0.4,
                                        "robust_snr": 15.0,
                                    }
                                }
                            },
                        }
                    ),
                ),
            )
            result = service.record_decision(
                workflow["id"],
                run_id,
                "advance",
                "Power Rabi evidence passes.",
                "05",
                {},
                [],
                "unittest",
            )
            self.assertEqual(result["next_action"]["next_node"], "05")
            current = service.status(workflow["id"])["workflow"]
            self.assertEqual(current["current_node"], "05")
            self.assertEqual(current["status"], "active")

    def test_state_patch_must_stay_within_workflow_targets(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state = sample_state(0.2, 0.1)
            state["qubits"]["q2"] = deepcopy(state["qubits"]["q1"])
            settings = make_settings(Path(folder), state)
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            with self.assertRaisesRegex(Exception, "workflow target"):
                service.request_state_commit(
                    workflow["id"],
                    [
                        {
                            "op": "replace",
                            "path": "/qubits/q2/xy/intermediate_frequency",
                            "value": 110_000_000,
                        }
                    ],
                    "Attempt an out-of-scope state change.",
                    "unittest",
                )

    def test_runner_releases_lock_if_request_preparation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            proposal = service.request_run(
                workflow["id"],
                "02x",
                {"qubits": ["q1"]},
                "Locate the bare resonator.",
                "unittest",
            )
            service.approve_interactively(proposal["id"])
            with patch(
                "jy_agent.runner.atomic_write_json",
                side_effect=OSError("synthetic request write failure"),
            ):
                with self.assertRaisesRegex(OSError, "synthetic request"):
                    service.execute_run(proposal["id"], "unittest")
            self.assertFalse(settings.lock_path.exists())

    def test_measurement_mode_exit_phrase_stops_workflow_and_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            service = AgentService(settings)
            with self.assertRaisesRegex(Exception, "exact measurement-mode"):
                service.start_workflow(["q1"], {}, "unittest", "start")
            workflow = start_test_workflow(service)
            stopped = service.leave_measurement_mode(
                workflow["id"], "unittest", EXIT_PHRASE
            )
            self.assertEqual(stopped["status"], "stopped")
            with self.assertRaisesRegex(Exception, "not active"):
                service.request_run(
                    workflow["id"],
                    "02x",
                    {"qubits": ["q1"]},
                    "Should remain gated after shutdown.",
                    "unittest",
                )
            replacement = start_test_workflow(service)
            self.assertEqual(replacement["status"], "active")

    def test_stop_wording_variants_all_stop_the_workflow(self) -> None:
        for phrase in (
            "結束量測",
            "退出JY量測模式",
            "退出 JY 自動量測模式",
            "停止量測",
            "停止 JY 自動量測",
        ):
            with self.subTest(phrase=phrase), tempfile.TemporaryDirectory() as folder:
                service = AgentService(
                    make_settings(Path(folder), sample_state(0.2, 0.1))
                )
                entered = service.enter_measurement_mode(
                    "unittest-agent", ENTRY_PHRASE, ["q1"]
                )
                proposal = service.request_run(
                    entered["workflow"]["id"],
                    "02x",
                    {"qubits": ["q1"]},
                    "Pending proposal should be cancelled during shutdown.",
                    "unittest-agent",
                )
                stopped = service.leave_measurement_mode(
                    entered["workflow"]["id"], "unittest-agent", phrase
                )
                self.assertEqual(stopped["status"], "stopped")
                self.assertEqual(service.proposal(proposal["id"])["status"], "cancelled")
                self.assertEqual(
                    service.dashboard_status(entered["dashboard"]["id"])["status"],
                    "stopped",
                )

    def test_settings_loads_paths_only_from_home_qualibrate_config(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            agent_root = root / "JY_agent"
            (agent_root / "config").mkdir(parents=True)
            shutil.copy2(
                AGENT_ROOT / "config" / "agent.yaml",
                agent_root / "config" / "agent.yaml",
            )
            state_root = root / "quam"
            state_root.mkdir()
            atomic_write_json(state_root / "state.json", sample_state(0.2, 0.1))
            atomic_write_json(state_root / "wiring.json", {})
            data_root = root / "Data"
            data_root.mkdir()
            calibration_graph = root / "calibration_graph"
            calibration_graph.mkdir()
            home = root / "home"
            config_path = home / ".qualibrate" / "config.toml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                "\n".join(
                    [
                        "[quam]",
                        f'state_path = "{state_root.as_posix()}"',
                        "[qualibrate]",
                        'project = "test-project"',
                        "[qualibrate.storage]",
                        'type = "local_storage"',
                        f'location = "{data_root.as_posix()}"',
                        "[qualibrate.calibration_library]",
                        f'folder = "{calibration_graph.as_posix()}"',
                    ]
                ),
                encoding="utf-8",
            )
            with patch(
                "jy_agent.config.Path.home", return_value=home
            ), patch.dict(
                os.environ,
                {"QUALIBRATE_CONFIG_FILE": str(root / "wrong-config.toml")},
            ):
                settings = Settings.load(agent_root)
            self.assertEqual(settings.active_state, state_root / "state.json")
            self.assertEqual(settings.wiring_path, state_root / "wiring.json")
            self.assertEqual(settings.data_root, data_root)
            self.assertEqual(settings.calibration_graph, calibration_graph)
            self.assertEqual(settings.qualibrate_project, "test-project")
            self.assertEqual(settings.qualibrate_config_path, config_path.resolve())


class AnalysisTests(unittest.TestCase):
    def test_07b_normalizes_percent_fidelity_for_analysis_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
            quality = analyzer._fit_quality(
                "07b",
                {
                    "results": {
                        "q1": {
                            "fidelity": 88.6,
                            "angle": 0.1,
                            "threshold": 0.02,
                            "confusion_matrix": [[0.9, 0.1], [0.128, 0.872]],
                        }
                    }
                },
                dataset_metrics={
                    "qubits": {"q1": {"morphology_pass": True, "clouds": {}}}
                },
                targets=["q1"],
            )
            self.assertEqual(quality["failures"], [])
            fit = quality["results"]["q1"]
            self.assertAlmostEqual(fit["readout_fidelity"], 0.886)
            self.assertEqual(fit["protected_node_fidelity"], 88.6)
            self.assertEqual(fit["reset_type"], "thermal")
            self.assertFalse(fit["active_reset_qualified"])
            patch, warnings = analyzer._normalize_07b_fidelity_patch(
                [
                    {
                        "op": "replace",
                        "path": "/qubits/q1/extras/readout_fidelity",
                        "value": 88.6,
                    }
                ],
                quality,
            )
            self.assertAlmostEqual(patch[0]["value"], 0.886)
            self.assertTrue(warnings)

    def test_07b_active_reset_qualifies_at_eighty_percent(self) -> None:
        """Operator instruction 2026-09-21: the floor is 0.80, not 0.85.

        Operator decision 2026-10-14: two Gaussian blobs are required too
        (`active_reset_requires_morphology: true`), the same rule as state
        discrimination.
        """

        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))

            def qualify(fidelity: float, morphology_pass: bool) -> dict:
                return analyzer._fit_quality(
                    "07b",
                    {
                        "initial_parameters": {
                            "reset_type_thermal_or_active": "active"
                        },
                        "results": {"q1": {"fidelity": fidelity}},
                    },
                    dataset_metrics={
                        "qubits": {
                            "q1": {
                                "morphology_pass": morphology_pass,
                                "clouds": {},
                            }
                        }
                    },
                    targets=["q1"],
                )["results"]["q1"]

            self.assertFalse(qualify(79.9, True)["active_reset_qualified"])
            self.assertTrue(qualify(80.0, True)["active_reset_qualified"])
            # Without two Gaussian blobs there is no active reset.
            self.assertFalse(qualify(84.0, False)["active_reset_qualified"])

            thermal = analyzer._fit_quality(
                "07b",
                {
                    "initial_parameters": {
                        "reset_type_thermal_or_active": "thermal"
                    },
                    "results": {"q1": {"fidelity": 95.0}},
                },
                dataset_metrics={
                    "qubits": {"q1": {"morphology_pass": True, "clouds": {}}}
                },
                targets=["q1"],
            )["results"]["q1"]
            # A thermal run never qualifies, however good the number.
            self.assertFalse(thermal["active_reset_qualified"])

    def test_07b_long_tail_fails_even_when_fidelity_can_pass(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
            rng = np.random.default_rng(7)
            samples = 10_000
            round_g = rng.normal(0.0, 0.12, size=(samples, 2))
            round_e = rng.normal(0.0, 0.12, size=(samples, 2)) + [1.0, 0.0]
            dataset = xr.Dataset(
                {
                    "I_g": (("qubit", "N"), round_g[:, 0][None, :]),
                    "Q_g": (("qubit", "N"), round_g[:, 1][None, :]),
                    "I_e": (("qubit", "N"), round_e[:, 0][None, :]),
                    "Q_e": (("qubit", "N"), round_e[:, 1][None, :]),
                },
                coords={"qubit": ["q1"], "N": np.arange(samples)},
            )
            self.assertTrue(
                analyzer._iq_blob_morphology(dataset)["qubits"]["q1"][
                    "morphology_pass"
                ]
            )

            tail_count = 1500
            tail = np.column_stack(
                (
                    np.linspace(0.3, 3.0, tail_count),
                    rng.normal(0.0, 0.03, tail_count),
                )
            )
            tailed_g = round_g.copy()
            tailed_g[:tail_count] = tail
            dataset["I_g"] = (("qubit", "N"), tailed_g[:, 0][None, :])
            dataset["Q_g"] = (("qubit", "N"), tailed_g[:, 1][None, :])
            result = analyzer._iq_blob_morphology(dataset)["qubits"]["q1"]
            self.assertFalse(result["morphology_pass"])
            self.assertTrue(result["morphology_failures"])

    def test_07b_round_but_strongly_overlapping_clouds_fail_fidelity_floor(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
            # With the floor configured to block (since 2026-10-13 it may not).
            analyzer.policy.raw["analysis"]["07b"]["min_readout_fidelity_blocks_pass"] = True
            quality = analyzer._fit_quality(
                "07b",
                {"results": {"q1": {"fidelity": 60.3}}},
                dataset_metrics={
                    "qubits": {"q1": {"morphology_pass": True, "clouds": {}}}
                },
                targets=["q1"],
            )

            self.assertFalse(quality["results"]["q1"]["fit_successful"])
            self.assertTrue(any("below 0.8" in item for item in quality["failures"]))

    def test_07b_always_passes_but_gates_discrimination(self) -> None:
        """Operator decision 2026-10-14: the angle always commits."""

        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))

            def judge(fidelity: float, morphology_pass: bool) -> dict:
                return analyzer._fit_quality(
                    "07b",
                    {"results": {"q1": {"fidelity": fidelity, "angle": 0.3}}},
                    dataset_metrics={
                        "qubits": {
                            "q1": {"morphology_pass": morphology_pass, "clouds": {}}
                        }
                    },
                    targets=["q1"],
                )

            good = judge(85.0, True)
            self.assertEqual(good["failures"], [])
            self.assertTrue(good["results"]["q1"]["fit_successful"])
            self.assertTrue(good["results"]["q1"]["state_discrimination_qualified"])
            for fidelity, shape, reason in (
                (55.0, True, "below 0.8"),
                (90.0, False, "Gaussian blobs"),
            ):
                quality = judge(fidelity, shape)
                result = quality["results"]["q1"]
                self.assertEqual(quality["failures"], [])
                self.assertTrue(result["fit_successful"])
                self.assertEqual(result["angle"], 0.3)
                self.assertFalse(result["state_discrimination_qualified"])
                self.assertTrue(
                    any(reason in item for item in result["state_discrimination_reasons"])
                )

    def test_07b_sparse_tail_fails_p99_rule_without_axis_elongation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            policy = PolicyEngine(settings)
            # The radius rule is kept behind morphology_mode; the default is
            # now "two_blob" (operator instruction 2026-10-02).
            policy.raw["analysis"]["07b"]["morphology_mode"] = "radius"
            analyzer = SnapshotAnalyzer(settings, policy)
            rng = np.random.default_rng(17)
            samples = 10_000
            ground = rng.normal(0.0, 0.12, size=(samples, 2))
            excited = rng.normal(0.0, 0.12, size=(samples, 2)) + [1.0, 0.0]
            tail_count = 150
            ground[:tail_count] = np.column_stack(
                (
                    np.linspace(0.5, 0.8, tail_count),
                    rng.normal(0.0, 0.03, tail_count),
                )
            )
            dataset = xr.Dataset(
                {
                    "I_g": (("qubit", "N"), ground[:, 0][None, :]),
                    "Q_g": (("qubit", "N"), ground[:, 1][None, :]),
                    "I_e": (("qubit", "N"), excited[:, 0][None, :]),
                    "Q_e": (("qubit", "N"), excited[:, 1][None, :]),
                },
                coords={"qubit": ["q1"], "N": np.arange(samples)},
            )

            result = analyzer._iq_blob_morphology(dataset)["qubits"]["q1"]
            self.assertFalse(result["morphology_pass"])
            self.assertTrue(
                any("p99/median radius" in item for item in result["morphology_failures"])
            )

    def test_04_edge_uses_fitted_pi_amplitude_not_raw_local_extremum(self) -> None:
        fit_quality = {
            "results": {"q1": {"Pi_amplitude": 0.52}},
            "failures": [],
        }
        dataset_metrics = {
            "qubits": {
                    "q1": {
                        "sweep_min_coordinate": 0.0,
                        "sweep_max_coordinate": 1.9,
                        "feature_coordinate": 0.01,
                        "edge_fraction": 0.01 / 1.9,
                }
            }
        }
        warnings = SnapshotAnalyzer._normalize_04_fitted_edge_metrics(
            fit_quality, dataset_metrics, {"q1": 0.5}
        )
        metrics = dataset_metrics["qubits"]["q1"]
        self.assertAlmostEqual(metrics["feature_coordinate"], 1.04)
        self.assertAlmostEqual(metrics["edge_fraction"], (1.9 - 1.04) / 1.9)
        self.assertEqual(metrics["raw_extremum_coordinate"], 0.01)
        self.assertEqual(
            metrics["edge_evidence_source"], "fitted_Pi_amplitude_factor"
        )
        self.assertTrue(warnings)

    def test_03a_prefers_broader_higher_peak_over_lower_two_photon_peak(self) -> None:
        frequency = np.linspace(-200_000_000.0, 200_000_000.0, 801)
        lower_narrow = 1.4 * np.exp(
            -4 * np.log(2) * ((frequency + 90_000_000.0) / 5_000_000.0) ** 2
        )
        upper_broad = np.exp(
            -4 * np.log(2) * ((frequency - 0.0) / 20_000_000.0) ** 2
        )
        values = lower_narrow + upper_broad
        candidates = _spectroscopy_peak_candidates(
            frequency,
            values,
            min_distance_hz=10_000_000.0,
        )
        pair = _select_higher_transition_pair(
            candidates,
            min_separation_hz=50_000_000.0,
            max_separation_hz=150_000_000.0,
            lower_max_width_ratio=0.60,
            min_peak_snr=5.0,
        )
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertAlmostEqual(
            pair["lower_two_photon_candidate"]["coordinate_hz"],
            -90_000_000.0,
            delta=1_000_000.0,
        )
        self.assertAlmostEqual(
            pair["upper_fundamental_candidate"]["coordinate_hz"],
            0.0,
            delta=1_000_000.0,
        )
        self.assertAlmostEqual(pair["separation_hz"], 90_000_000.0, delta=2e6)

    def test_03a_feature_fwhm_measurement(self) -> None:
        frequency = np.linspace(-50_000_000.0, 50_000_000.0, 2001)
        narrow = np.exp(-4 * np.log(2) * (frequency / 8_000_000.0) ** 2)
        broad = np.exp(-4 * np.log(2) * (frequency / 22_000_000.0) ** 2)
        narrow_width = _feature_fwhm(frequency, narrow, int(np.argmax(narrow)))
        broad_width = _feature_fwhm(frequency, broad, int(np.argmax(broad)))
        self.assertIsNotNone(narrow_width)
        self.assertIsNotNone(broad_width)
        self.assertAlmostEqual(narrow_width, 8_000_000.0, delta=100_000.0)
        self.assertAlmostEqual(broad_width, 22_000_000.0, delta=100_000.0)

    def _copy_fixture(self, name: str, target: Path) -> None:
        target.mkdir(parents=True)
        for source in (FIXTURES / name).iterdir():
            shutil.copy2(source, target / source.name)
        (target / "figure.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"synthetic-offline-fixture"
        )

    def test_02x_good_synthetic_snapshot_passes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = make_settings(root / "settings", sample_state(0.2, 0.1))
            snapshot = root / "snapshot"
            self._copy_fixture("02x_good", snapshot)
            frequency = np.linspace(-10_000_000, 10_000_000, 401)
            signal = 1.0 - 0.3 * np.exp(-((frequency / 1_500_000) ** 2))
            signal += 0.0001 * np.sin(np.arange(frequency.size))
            dataset = xr.Dataset(
                {"IQ_abs": (("qubit", "freq"), signal[None, :])},
                coords={"qubit": ["q1"], "freq": frequency},
            )
            dataset.to_netcdf(snapshot / "ds.h5", engine="scipy")
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
            result = analyzer.analyze_snapshot(snapshot, "02x")
            self.assertEqual(result["analysis_status"], "pass")
            self.assertGreater(
                result["dataset_metrics"]["qubits"]["q1"]["robust_snr"], 5
            )

    def test_02c_empty_node_fit_is_not_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = make_settings(root / "settings", sample_state(0.2, 0.1))
            snapshot = root / "snapshot"
            self._copy_fixture("02c_ambiguous", snapshot)
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
            result = analyzer.analyze_snapshot(snapshot, "02c")
            self.assertEqual(result["analysis_status"], "needs_review")
            self.assertFalse(
                any("fit" in item.lower() for item in result["failure_reasons"])
            )

    def test_02c_plateau_rule_finds_maximum_dressed_power(self) -> None:
        power = np.linspace(-50.0, -10.0, 41)
        frequency = np.linspace(-10_000_000.0, 10_000_000.0, 201)
        dressed = 2_000_000.0
        bare = -3_000_000.0
        tracked = np.empty_like(power)
        for index, value in enumerate(power):
            if value <= -34.0:
                tracked[index] = dressed + 20_000.0 * np.sin(index)
            elif value >= -26.0:
                tracked[index] = bare + 20_000.0 * np.sin(index)
            else:
                fraction = (value + 34.0) / 8.0
                tracked[index] = dressed + fraction * (bare - dressed)
        dataset = xr.Dataset(
            {"rr_min_response": (("qubit", "power_dbm"), tracked[None, :])},
            coords={
                "qubit": ["q1"],
                "power_dbm": power,
                "freq": frequency,
            },
        )
        result = analyze_02c_transitions(dataset)
        transition = result["qubits"]["q1"]
        self.assertEqual(transition["validation_failures"], [])
        self.assertAlmostEqual(
            transition["dressed_frequency_offset_hz"], dressed, delta=50_000
        )
        self.assertAlmostEqual(
            transition["bare_frequency_offset_hz"], bare, delta=50_000
        )
        self.assertGreaterEqual(transition["depletion_point_count"], 2)
        self.assertGreaterEqual(transition["transition_monotonic_fraction"], 0.7)
        self.assertGreaterEqual(transition["dressed_power_limit_dbm"], -35.0)
        self.assertLessEqual(transition["dressed_power_limit_dbm"], -31.0)
        derivative = np.zeros_like(power)
        crossing_index = int(
            np.argmin(np.abs(power - transition["dressed_power_limit_dbm"]))
        )
        derivative[crossing_index:] = -60_000.0
        dataset["rr_min_response_diff_avg"] = (
            ("qubit", "power_dbm"),
            derivative[None, :],
        )
        selected = node_selected_readout_powers_dbm(dataset, -50_000)["q1"]
        self.assertAlmostEqual(
            selected, transition["dressed_power_limit_dbm"], places=9
        )
        self.assertIsNone(
            selected_readout_power_dbm(
                {"full_scale_power_dbm": -2, "amplitude": 0.05}
            )
        )

    def test_02c_accepts_moderately_wide_plateaus(self) -> None:
        power = np.linspace(-50.0, -10.0, 41)
        frequency = np.linspace(-10_000_000.0, 10_000_000.0, 201)
        dressed = 2_000_000.0
        bare = -3_000_000.0
        tracked = np.empty_like(power)
        for index, value in enumerate(power):
            if value <= -34.0:
                tracked[index] = dressed + 1_200_000.0 * np.sin(index)
            elif value >= -26.0:
                tracked[index] = bare + 1_200_000.0 * np.sin(index)
            else:
                fraction = (value + 34.0) / 8.0
                tracked[index] = dressed + fraction * (bare - dressed)
        dataset = xr.Dataset(
            {"rr_min_response": (("qubit", "power_dbm"), tracked[None, :])},
            coords={
                "qubit": ["q1"],
                "power_dbm": power,
                "freq": frequency,
            },
        )
        transition = analyze_02c_transitions(dataset)["qubits"]["q1"]
        self.assertEqual(transition["validation_failures"], [])
        self.assertGreater(transition["dressed_plateau_width_hz"], 1_750_000.0)
        self.assertGreaterEqual(transition["depletion_point_count"], 2)

    def test_02c_accepts_bimodal_transition_without_depletion_points(self) -> None:
        """Separable dressed/bare passes even when the middle is unjudgeable."""
        power = np.linspace(-50.0, -10.0, 41)
        frequency = np.linspace(-10_000_000.0, 10_000_000.0, 201)
        dressed = 2_000_000.0
        bare = -3_000_000.0
        tracked = np.empty_like(power)
        for index, value in enumerate(power):
            if value <= -34.0:
                tracked[index] = dressed
            elif value >= -26.0:
                tracked[index] = bare
            else:
                # Jumps between the two frequencies instead of drifting: no
                # intermediate points and a non-monotonic transition.
                tracked[index] = bare if index % 2 else dressed
        dataset = xr.Dataset(
            {"rr_min_response": (("qubit", "power_dbm"), tracked[None, :])},
            coords={
                "qubit": ["q1"],
                "power_dbm": power,
                "freq": frequency,
            },
        )
        transition = analyze_02c_transitions(dataset)["qubits"]["q1"]
        self.assertEqual(transition["validation_failures"], [])
        self.assertEqual(transition["depletion_point_count"], 0)
        self.assertTrue(
            any("depletion" in note for note in transition["advisory_notes"])
        )
        self.assertGreaterEqual(transition["dressed_classified_point_count"], 2)
        self.assertGreaterEqual(transition["bare_classified_point_count"], 2)
        # The proposed power stops at the first jump instead of being pushed to
        # the last dressed-looking point of the bimodal region.
        self.assertLessEqual(transition["dressed_power_limit_dbm"], -33.0)
        self.assertGreaterEqual(transition["dressed_power_limit_dbm"], -36.0)

    def test_02c_rejects_when_frequencies_stay_inseparable(self) -> None:
        power = np.linspace(-50.0, -10.0, 41)
        frequency = np.linspace(-10_000_000.0, 10_000_000.0, 201)
        # Dressed and bare differ by less than the required separation.
        tracked = np.where(power <= -30.0, 1_000_000.0, 1_000_150.0)
        dataset = xr.Dataset(
            {"rr_min_response": (("qubit", "power_dbm"), tracked[None, :])},
            coords={
                "qubit": ["q1"],
                "power_dbm": power,
                "freq": frequency,
            },
        )
        transition = analyze_02c_transitions(dataset)["qubits"]["q1"]
        self.assertTrue(
            any(
                "not distinct" in reason
                for reason in transition["validation_failures"]
            )
        )

    def test_02c_plateau_rule_rejects_missing_transition(self) -> None:
        power = np.linspace(-50.0, -10.0, 41)
        frequency = np.linspace(-10_000_000.0, 10_000_000.0, 201)
        tracked = np.full_like(power, 1_000_000.0)
        dataset = xr.Dataset(
            {"rr_min_response": (("qubit", "power_dbm"), tracked[None, :])},
            coords={
                "qubit": ["q1"],
                "power_dbm": power,
                "freq": frequency,
            },
        )
        transition = analyze_02c_transitions(dataset)["qubits"]["q1"]
        self.assertTrue(transition["validation_failures"])
        self.assertTrue(
            any(
                "not distinct" in reason
                for reason in transition["validation_failures"]
            )
        )

    def test_05_t1_fit_passes_and_proposes_target_t1(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = make_settings(root / "settings", sample_state(0.2, 0.1))
            snapshot = root / "snapshot"
            snapshot.mkdir()
            atomic_write_json(
                snapshot / "node.json",
                {"id": 701, "data": {"outcomes": {"q1": "successful"}}},
            )
            atomic_write_json(
                snapshot / "data.json",
                {"ds": "./ds.h5", "initial_parameters": {"qubits": ["q1"]}},
            )
            (snapshot / "figure.png").write_bytes(b"synthetic-t1-plot")
            idle_time = np.linspace(0.016, 200.0, 121)
            signal = 0.04 + 0.35 * np.exp(-idle_time / 35.0)
            signal += 0.00005 * np.sin(np.arange(idle_time.size))
            dataset = xr.Dataset(
                {"I": (("qubit", "idle_time"), signal[None, :])},
                coords={"qubit": ["q1"], "idle_time": idle_time},
            )
            dataset.idle_time.attrs["units"] = "us"
            dataset.to_netcdf(snapshot / "ds.h5", engine="scipy")
            proposed_state = sample_state(0.2, 0.1)
            proposed_state["qubits"]["q1"]["T1"] = 35e-6
            atomic_write_json(snapshot / "quam_state.json", proposed_state)
            run = {
                "node_id": "05",
                "parameters_json": json.dumps({"qubits": ["q1"]}),
                "active_state_hash_before": sha256_file(settings.active_state),
            }
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
            result = analyzer.analyze_snapshot(snapshot, "05", run)
            self.assertEqual(result["analysis_status"], "pass")
            fit = result["fit_quality"]["results"]["q1"]
            self.assertAlmostEqual(fit["t1_seconds"], 35e-6, delta=1e-6)
            self.assertEqual(result["candidate_state_patch"][0]["op"], "add")
            self.assertEqual(
                result["candidate_state_patch"][0]["path"], "/qubits/q1/T1"
            )
            self.assertAlmostEqual(
                result["candidate_state_patch"][0]["value"], 35e-6, delta=1e-6
            )

    def test_05_selects_best_decay_quadrature_and_normalizes_patch(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = make_settings(root / "settings", sample_state(0.2, 0.1))
            snapshot = root / "snapshot"
            snapshot.mkdir()
            atomic_write_json(
                snapshot / "node.json",
                {"id": 703, "data": {"outcomes": {"q1": "successful"}}},
            )
            atomic_write_json(
                snapshot / "data.json",
                {"ds": "./ds.h5", "initial_parameters": {"qubits": ["q1"]}},
            )
            (snapshot / "figure.png").write_bytes(b"synthetic-t1-plot")
            idle_time = np.linspace(0.016, 200.0, 121)
            poor_i = 0.02 + 0.002 * np.sin(np.arange(idle_time.size) * 2.1)
            clean_q = -0.04 - 0.35 * np.exp(-idle_time / 45.0)
            clean_q += 0.00005 * np.sin(np.arange(idle_time.size))
            dataset = xr.Dataset(
                {
                    "I": (("qubit", "idle_time"), poor_i[None, :]),
                    "Q": (("qubit", "idle_time"), clean_q[None, :]),
                },
                coords={"qubit": ["q1"], "idle_time": idle_time},
            )
            dataset.idle_time.attrs["units"] = "us"
            dataset.to_netcdf(snapshot / "ds.h5", engine="scipy")
            proposed_state = sample_state(0.2, 0.1)
            proposed_state["qubits"]["q1"]["T1"] = 999e-6
            atomic_write_json(snapshot / "quam_state.json", proposed_state)
            run = {
                "node_id": "05",
                "parameters_json": json.dumps({"qubits": ["q1"]}),
                "active_state_hash_before": sha256_file(settings.active_state),
            }

            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
            result = analyzer.analyze_snapshot(snapshot, "05", run)

            self.assertEqual(result["analysis_status"], "pass")
            fit = result["fit_quality"]["results"]["q1"]
            self.assertEqual(fit["signal"], "Q")
            self.assertAlmostEqual(fit["t1_seconds"], 45e-6, delta=1e-6)
            self.assertAlmostEqual(
                result["candidate_state_patch"][0]["value"], 45e-6, delta=1e-6
            )
            self.assertTrue(
                any("Q quadrature" in item for item in result["warnings"])
            )
            selected_plots = [
                Path(item)
                for item in result["plots"]
                if "-05-q1-Q.png" in item
            ]
            self.assertEqual(len(selected_plots), 1)
            self.assertTrue(selected_plots[0].is_file())

    def test_05_t1_fit_rejects_insufficient_time_span(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = make_settings(root / "settings", sample_state(0.2, 0.1))
            snapshot = root / "snapshot"
            snapshot.mkdir()
            atomic_write_json(
                snapshot / "node.json",
                {"id": 702, "data": {"outcomes": {"q1": "successful"}}},
            )
            atomic_write_json(
                snapshot / "data.json",
                {"ds": "./ds.h5", "initial_parameters": {"qubits": ["q1"]}},
            )
            (snapshot / "figure.png").write_bytes(b"synthetic-t1-plot")
            idle_time = np.linspace(0.016, 100.0, 101)
            signal = 0.04 + 0.35 * np.exp(-idle_time / 1000.0)
            dataset = xr.Dataset(
                {"I": (("qubit", "idle_time"), signal[None, :])},
                coords={"qubit": ["q1"], "idle_time": idle_time},
            )
            dataset.idle_time.attrs["units"] = "us"
            dataset.to_netcdf(snapshot / "ds.h5", engine="scipy")
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
            result = analyzer.analyze_snapshot(snapshot, "05")
            self.assertEqual(result["analysis_status"], "needs_review")
            self.assertTrue(
                any(
                    "less than 3.5 fitted T1 lifetimes" in item
                    for item in result["failure_reasons"]
                )
            )

    def test_lightweight_report_has_required_fields(self) -> None:
        report = lightweight_report(
            {"plots": ["C:/data/figure.png"]},
            "repeat",
            "Peak is at the sweep edge.",
            "03a",
            {"frequency_span_in_mhz": 200},
        )
        self.assertIn("![result plot", report)
        self.assertIn("Decision: repeat", report)
        self.assertIn("Reason:", report)
        self.assertIn("Next action:", report)

    def test_03a_candidate_patch_excludes_x180_amplitude(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
            accepted, rejected = analyzer._target_node_changes(
                [
                    {
                        "op": "replace",
                        "path": "/qubits/q1/xy/intermediate_frequency",
                        "value": 120_000_000,
                    },
                    {
                        "op": "replace",
                        "path": (
                            "/qubits/q1/xy/operations/"
                            "x180_DragCosine/amplitude"
                        ),
                        "value": 0.3,
                    },
                ],
                "03a",
                ["q1"],
            )
            self.assertEqual(len(accepted), 1)
            self.assertTrue(accepted[0]["path"].endswith("intermediate_frequency"))
            self.assertEqual(len(rejected), 1)

    def test_03a_arbitrary_center_normalizes_fit_frequency(self) -> None:
        fit_quality = {
            "results": {
                "q1": {"fit_successful": True, "drive_freq": 3.2474e9}
            },
            "failures": [],
        }
        metrics = {"qubits": {"q1": {"feature_coordinate": 47.4e6}}}
        warnings = SnapshotAnalyzer._normalize_03a_arbitrary_center_fit(
            fit_quality,
            metrics,
            {"parameters": {"arbitrary_qubit_frequency_in_ghz": 2.985}},
        )
        self.assertEqual(fit_quality["results"]["q1"]["drive_freq"], 3.0324e9)
        self.assertTrue(warnings)

    def test_03a_stage_supports_intermediate_refinement(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
            stage = analyzer._03a_stage(
                {
                    "parameters": {
                        "frequency_span_in_mhz": 100.0,
                        "operation_amplitude_factor": 0.075,
                    }
                }
            )
            self.assertEqual(stage, "refinement_candidate")

    def test_02x_candidate_patch_includes_bare_resonator_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
            accepted, rejected = analyzer._target_node_changes(
                [
                    {
                        "op": "add",
                        "path": "/qubits/q1/extras/bare_resonator_freq",
                        "value": 6_048_286_875.0,
                    }
                ],
                "02x",
                ["q1"],
            )
            self.assertEqual(len(accepted), 1)
            self.assertEqual(rejected, [])

    def test_02a_candidate_paths_include_recorded_frequency_updates(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
            accepted, rejected = analyzer._target_node_changes(
                [
                    {
                        "op": "replace",
                        "path": "/qubits/q1/resonator/intermediate_frequency",
                        "value": 22_400_000,
                    },
                    {
                        "op": "replace",
                        "path": "/qubits/q1/extras/dressed_resonator_freq",
                        "value": 6.0504e9,
                    },
                ],
                "02a",
                ["q1"],
            )
            self.assertEqual(len(accepted), 2)
            self.assertEqual(rejected, [])

    def test_02a_chained_dressed_frequency_uses_recorded_if_delta(self) -> None:
        state = sample_state(0.2, 0.1)
        state["qubits"]["q1"]["extras"]["dressed_resonator_freq"] = 6.048e9
        patch = [
            {
                "op": "replace",
                "path": "/qubits/q1/extras/dressed_resonator_freq",
                "value": 6.048e9,
            },
            {
                "op": "replace",
                "path": "/qubits/q1/resonator/intermediate_frequency",
                "value": 22_400_000,
            },
        ]
        normalized = SnapshotAnalyzer._normalize_chained_frequency_updates(
            patch, "02a", state
        )
        values = {item["path"]: item["value"] for item in normalized}
        self.assertEqual(
            values["/qubits/q1/extras/dressed_resonator_freq"],
            6.0504e9,
        )

    def test_chained_frequency_uses_recorded_rf_when_extras_are_stale(self) -> None:
        state = sample_state(0.2, 0.1)
        state["qubits"]["q1"]["extras"]["dressed_resonator_freq"] = 5.9e9
        patch = [
            {
                "op": "replace",
                "path": "/qubits/q1/extras/dressed_resonator_freq",
                "value": 6.048e9,
            },
            {
                "op": "replace",
                "path": "/qubits/q1/resonator/intermediate_frequency",
                "value": 22_400_000,
            },
        ]
        normalized = SnapshotAnalyzer._normalize_chained_frequency_updates(
            patch, "02c", state
        )
        values = {item["path"]: item["value"] for item in normalized}
        self.assertEqual(
            values["/qubits/q1/extras/dressed_resonator_freq"],
            6.0504e9,
        )


if __name__ == "__main__":
    unittest.main()
