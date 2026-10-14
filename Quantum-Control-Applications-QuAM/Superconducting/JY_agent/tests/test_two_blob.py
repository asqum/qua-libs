from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from jy_agent.analysis import SnapshotAnalyzer, _select_07d_readout_patch
from jy_agent.analysis_07d import cloud_outlier_fractions, two_blob_metrics
from jy_agent.policy import PolicyEngine
from jy_agent.service import AgentService
from jy_agent.util import atomic_write_json
from tests.test_core import (
    ENTRY_PHRASE,
    make_settings,
    sample_multiplex_state_and_wiring,
    sample_state,
)
from tests.test_workflow_subgroups import _with_readout_nodes

RNG = np.random.default_rng(7)


def blobs(n: int, separation: float, thermal: float = 0.0, smear: float = 0.0):
    """g and e shots: unit-sigma Gaussians, optional thermal swap and e smear."""
    g = RNG.normal(size=n) + 1j * RNG.normal(size=n)
    e = separation + RNG.normal(size=n) + 1j * RNG.normal(size=n)
    swap = RNG.random(n) < thermal
    g[swap] += separation
    streak = RNG.random(n) < smear
    e[streak] = RNG.uniform(-3 * separation, -separation, streak.sum()) + 1j * RNG.normal(
        size=streak.sum()
    )
    return g, e


class CloudOutlierTests(unittest.TestCase):
    def test_thermal_population_is_on_a_blob(self) -> None:
        g, e = blobs(10000, 8.0, thermal=0.10)
        result = cloud_outlier_fractions(g, e, 3.0)
        self.assertLess(result["g_on_neither_blob"], 0.03)
        self.assertGreater(result["g_on_e_blob"], 0.08)

    def test_smear_is_off_both_blobs(self) -> None:
        g, e = blobs(10000, 8.0, smear=0.2)
        result = cloud_outlier_fractions(g, e, 3.0)
        self.assertGreater(result["e_on_neither_blob"], 0.15)


class TwoBlobSelectionTests(unittest.TestCase):
    """07d picks the best fidelity among two-blob points."""

    def dataset(self, clean_fidelity: float = 97.0) -> xr.Dataset:
        runs, freqs, amps, durs = 200, 3, 3, 1
        shape = (1, runs, freqs, amps, durs)
        arrays = {k: np.zeros(shape) for k in ("I_g", "Q_g", "I_e", "Q_e")}
        fidelity = np.full((1, freqs, amps, durs), 80.0)
        for f in range(freqs):
            for a in range(amps):
                # The loudest amplitude column is smeared but scores best.
                g, e = blobs(runs, 6.0, smear=0.3 if a == 2 else 0.0)
                arrays["I_g"][0, :, f, a, 0] = g.real
                arrays["Q_g"][0, :, f, a, 0] = g.imag
                arrays["I_e"][0, :, f, a, 0] = e.real
                arrays["Q_e"][0, :, f, a, 0] = e.imag
        fidelity[0, 1, 2, 0] = 98.0
        fidelity[0, 1, 0, 0] = clean_fidelity
        dims = ("qubit", "run", "freq", "amp", "duration")
        return xr.Dataset(
            {
                **{k: (dims, v) for k, v in arrays.items()},
                "fidelity": (("qubit", "freq", "amp", "duration"), fidelity),
            },
            coords={
                "qubit": ["q1"],
                "run": np.arange(runs),
                "freq": [-1e5, 0.0, 1e5],
                "amp": [0.5, 1.0, 2.0],
                "duration": [1000.0],
            },
        )

    def test_two_blob_point_within_tolerance_is_preferred(self) -> None:
        result = two_blob_metrics(self.dataset(97.0), {"neighborhood_points": 0})["q1"]
        self.assertEqual(result["unconstrained"]["amplitude_factor"], 2.0)
        self.assertTrue(result["two_blob"])
        self.assertEqual(result["selected"]["amplitude_factor"], 0.5)

    def test_costly_two_blob_point_keeps_the_fidelity_optimum(self) -> None:
        # 98 -> 95 costs 3 points, above the 2-point tolerance (2026-10-03).
        result = two_blob_metrics(self.dataset(95.0), {"neighborhood_points": 0})["q1"]
        self.assertFalse(result["two_blob"])
        self.assertEqual(result["selected"]["amplitude_factor"], 2.0)
        self.assertAlmostEqual(result["two_blob_fidelity_cost_percent"], 3.0)

    def test_points_above_the_dressed_limit_are_never_chosen(self) -> None:
        result = two_blob_metrics(
            self.dataset(95.0), {"neighborhood_points": 0}, {"q1": 1.5}
        )["q1"]
        self.assertEqual(result["unconstrained"]["amplitude_factor"], 0.5)
        self.assertEqual(result["selected"]["amplitude_factor"], 0.5)

    def test_patch_is_rebuilt_from_the_selected_point(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            self.dataset().to_netcdf(Path(folder) / "ds.h5")
            current = sample_state(0.2, 0.1)
            resonator = current["qubits"]["q1"]["resonator"]
            old_if = resonator["intermediate_frequency"]
            old_amp = resonator["operations"]["readout"]["amplitude"]
            patch, warnings, failures = _select_07d_readout_patch(
                [], Path(folder), current, ["q1"], {"neighborhood_points": 0}
            )
            values = {item["path"]: item["value"] for item in patch}
            base = "/qubits/q1/resonator"
            self.assertEqual(failures, [])
            self.assertEqual(values[f"{base}/intermediate_frequency"], old_if)
            self.assertAlmostEqual(
                values[f"{base}/operations/readout/amplitude"], old_amp * 0.5
            )
            self.assertEqual(values[f"{base}/operations/readout/length"], 1000)
            self.assertTrue(any("two-blob point" in w for w in warnings))
            self.assertTrue(any("fidelity optimum 98.0%" in w for w in warnings))


class IqBlobTwoBlobModeTests(unittest.TestCase):
    def test_07b_passes_thermal_tails_and_fails_smear(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
            g1, e1 = blobs(10000, 8.0, thermal=0.08)
            g2, e2 = blobs(10000, 8.0, smear=0.3)
            dataset = xr.Dataset(
                {
                    "I_g": (("qubit", "N"), np.vstack([g1.real, g2.real])),
                    "Q_g": (("qubit", "N"), np.vstack([g1.imag, g2.imag])),
                    "I_e": (("qubit", "N"), np.vstack([e1.real, e2.real])),
                    "Q_e": (("qubit", "N"), np.vstack([e1.imag, e2.imag])),
                },
                coords={"qubit": ["q1", "q2"]},
            )
            metrics = analyzer._iq_blob_morphology(dataset)["qubits"]
            self.assertTrue(metrics["q1"]["morphology_pass"])
            self.assertFalse(metrics["q2"]["morphology_pass"])


class Reopen07dTests(unittest.TestCase):
    def test_reopen_resets_single_pass_and_needs_no_active_lease(self) -> None:
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
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, started_at
                ) VALUES ('old', ?, ?, '07d', ?, 'completed', 'pass',
                          '2000-01-01T00:00:00+00:00')
                """,
                (workflow["id"], proposal["id"], json.dumps({"qubits": first})),
            )
            self.assertEqual(
                service._single_pass_measured_targets(workflow, "07d"), set(first)
            )
            with self.assertRaisesRegex(Exception, "only from 07b or 06"):
                service.reopen_07d_after_readout_rule_change(
                    workflow["id"], "rule change", "unittest"
                )
            service.db.execute(
                "UPDATE workflows SET current_node = '07b' WHERE id = ?",
                (workflow["id"],),
            )
            result = service.reopen_07d_after_readout_rule_change(
                workflow["id"], "two-blob rule", "unittest"
            )
            self.assertEqual(result["workflow"]["current_node"], "07d")
            self.assertEqual(
                service._single_pass_measured_targets(workflow, "07d"), set()
            )
            self.assertNotIn(
                first[0], service._node_target_resolution(workflow["id"], "07d")
            )


class ReopenActiveRepeatTests(unittest.TestCase):
    """2026-10-04: after a 07d reopen the old active-reset 07b run still
    counted as the repeat, so the new pass got none."""

    def test_old_active_run_no_longer_counts_after_reopen(self) -> None:
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
                "UPDATE workflows SET current_node = '07b' WHERE id = ?",
                (workflow["id"],),
            )
            proposal = service.request_run(
                workflow["id"], "07b", {"qubits": targets}, "Thermal.", "unittest"
            )
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, started_at
                ) VALUES ('old-active', ?, ?, '07b', ?, 'completed', 'pass',
                          '2000-01-01T00:00:00+00:00')
                """,
                (
                    workflow["id"],
                    proposal["id"],
                    json.dumps(
                        {"qubits": targets, "reset_type_thermal_or_active": "active"}
                    ),
                ),
            )
            self.assertEqual(
                service._active_reset_repeated_targets(workflow), set(targets)
            )
            service.reopen_07d_after_readout_rule_change(
                workflow["id"], "restart", "unittest"
            )
            self.assertEqual(service._active_reset_repeated_targets(workflow), set())


class ReopenDownstreamTests(unittest.TestCase):
    """2026-10-04: restarting from 07d must also re-run 06 (with reset groups)."""

    def test_old_06_runs_no_longer_resolve_targets_after_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            _with_readout_nodes(service)
            object.__setattr__(
                service.settings,
                "workflow_sequence",
                tuple(service.settings.workflow_sequence) + ("06",),
            )
            workflow = service.start_workflow(
                targets, {"multiplexed": True}, "unittest", ENTRY_PHRASE
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '06' WHERE id = ?",
                (workflow["id"],),
            )
            proposal = service.request_run(
                workflow["id"], "06",
                {"qubits": targets, "use_state_discrimination": False},
                "Ramsey.", "unittest"
            )
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, started_at
                ) VALUES ('old-06', ?, ?, '06', ?, 'completed', 'pass',
                          '2000-01-01T00:00:00+00:00')
                """,
                (workflow["id"], proposal["id"], json.dumps({"qubits": targets})),
            )
            self.assertTrue(service._node_has_completed_run(workflow["id"], "06"))
            service.reopen_07d_after_readout_rule_change(
                workflow["id"], "restart", "unittest"
            )
            self.assertFalse(service._node_has_completed_run(workflow["id"], "06"))
            self.assertEqual(
                service._node_target_resolution(workflow["id"], "06"), set()
            )


if __name__ == "__main__":
    unittest.main()
