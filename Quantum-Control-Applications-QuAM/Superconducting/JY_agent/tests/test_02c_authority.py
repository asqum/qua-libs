from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import xarray as xr

from jy_agent.analysis_02c import analyze_02c_transitions
from jy_agent.service import AgentService
from jy_agent.state_patch_02c import derive_02c_state_patch
from jy_agent.util import atomic_write_json
from jy_agent.workflow_02c import resolve_02c_targets
from tests.test_core import (
    ENTRY_PHRASE,
    make_settings,
    sample_multiplex_state_and_wiring,
    sample_state,
)


class Test02cAuthority(unittest.TestCase):
    def test_jy_patch_uses_dressed_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            current = json.loads(settings.active_state.read_text(encoding="utf-8"))
            metrics = {
                "qubits": {
                    "q1": {
                        "02c_transition": {
                            "base_frequency_hz": 6_000_000_000.0,
                            "dressed_frequency_hz": 6_001_000_000.0,
                            "dressed_power_limit_dbm": -30.0,
                            "validation_failures": [],
                            "skip_subsequent_experiments": False,
                        }
                    }
                }
            }
            patch, errors = derive_02c_state_patch(
                settings, current, metrics, ["q1"]
            )
            self.assertEqual(errors, [])
            values = {item["path"]: item["value"] for item in patch}
            self.assertEqual(
                values["/qubits/q1/resonator/intermediate_frequency"],
                21_000_000.0,
            )
            self.assertAlmostEqual(
                values["/qubits/q1/resonator/operations/readout/amplitude"],
                # 3 dB below the dressed boundary (operator rule 2026-10-13;
                # 10 dB from 2026-10-02).
                10.0 ** ((-30.0 - 3.0 - 5.0) / 20.0),
            )
            self.assertEqual(
                values["/qubits/q1/extras/dressed_resonator_freq"],
                6_001_000_000.0,
            )

    def test_strict_bare_only_trace_can_skip_qubit(self) -> None:
        dataset = self._bare_only_dataset()
        result = analyze_02c_transitions(
            dataset,
            run_parameters={
                "frequency_span_in_mhz": 10.0,
                "frequency_step_in_mhz": 0.1,
                "num_averages": 200,
            },
            qubit_context={
                "q1": {
                    "expected_bare_frequency_hz": 6_000_000_000.0,
                    "readout_length_ns": 1000,
                }
            },
        )
        transition = result["qubits"]["q1"]
        self.assertEqual(transition["classification"], "bare_only_absent")
        self.assertTrue(transition["skip_subsequent_experiments"])
        self.assertEqual(transition["validation_failures"], [])

    def test_bare_only_trace_without_acquisition_depth_is_not_skipped(self) -> None:
        dataset = self._bare_only_dataset()
        result = analyze_02c_transitions(
            dataset,
            run_parameters={
                "frequency_span_in_mhz": 10.0,
                "frequency_step_in_mhz": 0.1,
                "num_averages": 10,
            },
            qubit_context={
                "q1": {
                    "expected_bare_frequency_hz": 6_000_000_000.0,
                    "readout_length_ns": 100,
                }
            },
        )
        transition = result["qubits"]["q1"]
        self.assertEqual(transition["classification"], "ambiguous")
        self.assertFalse(transition["skip_subsequent_experiments"])
        self.assertFalse(
            transition["bare_only_checks"]["acquisition_is_sufficient"]
        )
        self.assertTrue(transition["validation_failures"])

    def test_02c_workflow_accepts_multiplex_subgroup(self) -> None:
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
            service.db.execute(
                "UPDATE workflows SET current_node = '02c' WHERE id = ?",
                (workflow["id"],),
            )
            with self.assertRaisesRegex(
                Exception, "first 02c run must multiplex every active target"
            ):
                service.request_run(
                    workflow["id"],
                    "02c",
                    {"qubits": ["q3", "q4", "q5"]},
                    "First 02c cannot start as a subgroup.",
                    "unittest",
                )
            first = service.request_run(
                workflow["id"],
                "02c",
                {"qubits": targets},
                "First 02c multiplexes every active target.",
                "unittest",
            )
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, analysis_json
                ) VALUES (
                    'offline-02c-first', ?, ?, '02c', ?, 'completed',
                    'needs_review', '{}'
                )
                """,
                (
                    workflow["id"],
                    first["id"],
                    json.dumps({"qubits": targets}),
                ),
            )
            subgroup = ["q3", "q4", "q5"]
            proposal = service.request_run(
                workflow["id"],
                "02c",
                {"qubits": subgroup},
                "Refine one 02c subgroup.",
                "unittest",
            )
            self.assertEqual(proposal["payload"]["parameters"]["qubits"], subgroup)
            self.assertTrue(proposal["payload"]["parameters"]["multiplexed"])

    def test_resolved_targets_track_confirmed_absence(self) -> None:
        analysis = {
            "dataset_metrics": {
                "qubits": {
                    "q1": {
                        "02c_transition": {
                            "validation_failures": [],
                            "dressed_frequency_hz": 6.0e9,
                            "dressed_power_limit_dbm": -32.0,
                        }
                    },
                    "q2": {
                        "02c_transition": {
                            "validation_failures": [],
                            "skip_subsequent_experiments": True,
                        }
                    },
                }
            }
        }
        resolved, absent = resolve_02c_targets(
            [
                {
                    "analysis_status": "pass",
                    "analysis_json": json.dumps(analysis),
                }
            ]
        )
        self.assertEqual(resolved, {"q1", "q2"})
        self.assertEqual(absent, {"q2"})

    @staticmethod
    def _bare_only_dataset() -> xr.Dataset:
        power = np.linspace(-50.0, -10.0, 41)
        frequency = np.linspace(-5_000_000.0, 5_000_000.0, 101)
        tracked = np.zeros((1, power.size))
        resonance = 1.0 - 0.2 * np.exp(-((frequency / 500_000.0) ** 2))
        resonance += 0.001 * np.sin(np.arange(frequency.size))
        iq = np.repeat(resonance[:, None], power.size, axis=1)[None, :, :]
        return xr.Dataset(
            {
                "rr_min_response": (("qubit", "power_dbm"), tracked),
                "IQ_abs": (("qubit", "freq", "power_dbm"), iq),
            },
            coords={
                "qubit": ["q1"],
                "power_dbm": power,
                "freq": frequency,
                "freq_full": (
                    ("qubit", "freq"),
                    (6_000_000_000.0 + frequency)[None, :],
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
