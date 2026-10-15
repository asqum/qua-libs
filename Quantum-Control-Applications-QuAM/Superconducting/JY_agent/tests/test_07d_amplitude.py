from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from jy_agent.analysis import _normalize_07d_amplitude_patch
from jy_agent.service import AgentService
from jy_agent.util import atomic_write_json
from tests.test_core import (
    ENTRY_PHRASE,
    make_settings,
    sample_multiplex_state_and_wiring,
    sample_state,
)
from tests.test_workflow_subgroups import _with_readout_nodes

AMP = "/qubits/q1/resonator/operations/readout/amplitude"


class Normalize07dAmplitudeTests(unittest.TestCase):
    """2026-09-30/10-02: the node's set_output_power restores the pre-run
    amplitude, so JY reads the optimal factor from the fidelity map."""

    def snapshot(self, folder: str, best_amp_index: int) -> Path:
        amps = np.linspace(0.2, 3.16, 22)
        fidelity = np.full((1, 3, amps.size, 2), 60.0)
        fidelity[0, 1, best_amp_index, 1] = 90.0
        xr.Dataset(
            {"fidelity": (("qubit", "freq", "amp", "duration"), fidelity)},
            coords={
                "qubit": ["q1"],
                "freq": [-1e5, 0.0, 1e5],
                "amp": amps,
                "duration": [500.0, 1000.0],
            },
        ).to_netcdf(Path(folder) / "ds.h5")
        return Path(folder)

    def test_amplitude_is_pre_run_times_optimal_factor(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            current = sample_state(0.2, 0.1)
            before = current["qubits"]["q1"]["resonator"]["operations"]["readout"]["amplitude"]
            snapshot = self.snapshot(folder, 10)
            patch = [{"op": "replace", "path": AMP, "value": before}]
            out, warnings = _normalize_07d_amplitude_patch(patch, snapshot, current)
            factor = float(np.linspace(0.2, 3.16, 22)[10])
            self.assertAlmostEqual(out[0]["value"], before * factor)
            self.assertFalse(any("top of the amplitude sweep" in w for w in warnings))

    def test_top_edge_optimum_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            current = sample_state(0.2, 0.1)
            patch = [{"op": "replace", "path": AMP, "value": 0.1}]
            _, warnings = _normalize_07d_amplitude_patch(
                patch, self.snapshot(folder, 21), current
            )
            self.assertTrue(any("top of the amplitude sweep" in w for w in warnings))

    def test_other_paths_are_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            current = sample_state(0.2, 0.1)
            patch = [
                {
                    "op": "replace",
                    "path": "/qubits/q1/resonator/operations/readout/length",
                    "value": 1040,
                }
            ]
            out, _ = _normalize_07d_amplitude_patch(
                patch, self.snapshot(folder, 3), current
            )
            self.assertEqual(out, patch)


class SinglePassDefaultKeysTests(unittest.TestCase):
    """A 07d run on an old sweep (3.16 before 2026-10-13) no longer counts."""

    def test_run_on_old_default_is_eligible_again(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            _with_readout_nodes(service)
            workflow = service.start_workflow(
                targets, {"multiplexed": True}, "unittest", ENTRY_PHRASE
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '07d' WHERE id = ?",
                (workflow["id"],),
            )
            first = sorted(targets)[:4]
            proposal = service.request_run(
                workflow["id"], "07d", {"qubits": first}, "Group 1.", "unittest"
            )
            for run_id, factor in (("old-sweep", 3.16), ("older-sweep", 1.41)):
                service.db.execute(
                    """
                    INSERT INTO runs(
                        id, workflow_id, proposal_id, node_id, parameters_json,
                        status, analysis_status
                    ) VALUES (?, ?, ?, '07d', ?, 'completed', 'pass')
                    """,
                    (
                        run_id,
                        workflow["id"],
                        proposal["id"],
                        json.dumps({"qubits": first, "max_amplitude_factor": factor}),
                    ),
                )
            self.assertEqual(
                service._single_pass_measured_targets(workflow, "07d"), set()
            )
            service.request_run(
                workflow["id"], "07d", {"qubits": first}, "Redo on 1.778.", "unittest"
            )
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status
                ) VALUES ('new-sweep', ?, ?, '07d', ?, 'completed', 'pass')
                """,
                (
                    workflow["id"],
                    proposal["id"],
                    json.dumps({"qubits": first, "max_amplitude_factor": 1.778}),
                ),
            )
            with self.assertRaisesRegex(Exception, "runs once per target"):
                service.request_run(
                    workflow["id"], "07d", {"qubits": first}, "Third.", "unittest"
                )


if __name__ == "__main__":
    unittest.main()
