"""06 passes only when both Ramsey fringes match the set detuning.

Operator instruction 2026-10-10: with detuning d, both fitted fringe
frequencies (detuning signs -1 and +1) must lie within 20% of d (1 MHz ->
+-200 kHz). A good fit that misses it stays unresolved and may commit only its
drive-frequency correction, so the next 06 confirms the corrected frequency.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import xarray as xr

from jy_agent.analysis import SnapshotAnalyzer
from jy_agent.policy import PolicyEngine
from jy_agent.state import patch_qubit_targets
from jy_agent.util import atomic_write_json, sha256_file

from test_core import make_settings, sample_state


def two_qubit_state() -> dict:
    state = sample_state(0.2, 0.1)
    state["qubits"]["q2"] = deepcopy(state["qubits"]["q1"])
    state["active_qubit_names"] = ["q1", "q2"]
    return state


def write_06_snapshot(snapshot: Path, fringes_mhz: dict[str, tuple[float, float]]) -> None:
    names = sorted(fringes_mhz)
    snapshot.mkdir()
    atomic_write_json(
        snapshot / "node.json",
        {"id": 806, "data": {"outcomes": {name: "successful" for name in names}}},
    )
    fit_results = {}
    for name, (minus, plus) in fringes_mhz.items():
        fit = xr.Dataset(
            {
                "fit": (
                    ("qubit", "sign", "fit_vals"),
                    np.array([[[minus * 1e-3], [plus * 1e-3]]]),
                )
            },
            coords={"qubit": [name], "sign": [-1, 1], "fit_vals": ["f"]},
        )
        fit.to_netcdf(snapshot / f"fit_results.{name}.raw_fit_results.h5", engine="scipy")
        fit_results[name] = {
            "freq_offset": (plus - minus) / 2 * 1e6,
            "decay": 1e-6,
            "decay_error": 0.05e-6,
            "qubit_name": name,
            "raw_fit_results": f"./fit_results.{name}.raw_fit_results.h5",
        }
    atomic_write_json(
        snapshot / "data.json",
        {
            "ds": "./ds.h5",
            "fit_results": fit_results,
            "initial_parameters": {
                "qubits": names,
                "max_wait_time_in_ns": 4000,
                "num_time_points": 100,
                "frequency_detuning_in_mhz": 1.0,
            },
        },
    )
    (snapshot / "figure.png").write_bytes(b"synthetic-ramsey-plot")
    time = np.linspace(16, 4000, 100)
    dataset = xr.Dataset(
        {"I": (("qubit", "time"), np.vstack([np.cos(time * 1e-3 * 6.28)] * len(names)))},
        coords={"qubit": names, "time": time},
    )
    dataset.to_netcdf(snapshot / "ds.h5", engine="scipy")
    proposed = two_qubit_state()
    for name, (minus, plus) in fringes_mhz.items():
        qubit = proposed["qubits"][name]
        qubit["xy"]["intermediate_frequency"] -= (plus - minus) / 2 * 1e6
        qubit["T2ramsey"] = 1e-6
        qubit["extras"]["idle_freq"] = 4.0e9 + qubit["xy"]["intermediate_frequency"]
    atomic_write_json(snapshot / "quam_state.json", proposed)


class RamseyFringeConfirmationTests(unittest.TestCase):
    def analyze(self, root: Path, fringes_mhz: dict[str, tuple[float, float]]) -> dict:
        settings = make_settings(root / "settings", two_qubit_state())
        snapshot = root / "snapshot"
        write_06_snapshot(snapshot, fringes_mhz)
        run = {
            "node_id": "06",
            "parameters_json": json.dumps({"qubits": sorted(fringes_mhz)}),
            "active_state_hash_before": sha256_file(settings.active_state),
        }
        analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
        return analyzer.analyze_snapshot(snapshot, "06", run)

    def test_first_run_passes_when_both_fringes_match(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            result = self.analyze(Path(folder), {"q1": (0.9, 1.1), "q2": (1.15, 0.85)})
            self.assertEqual(result["analysis_status"], "pass", result["failure_reasons"])
            self.assertEqual(result["passing_targets"], ["q1", "q2"])
            self.assertEqual(result["frequency_correction_patch"], [])
            fit = result["fit_quality"]["results"]["q1"]
            np.testing.assert_allclose(fit["fringe_frequencies_hz"], [0.9e6, 1.1e6])
            self.assertTrue(fit["fringe_matches_detuning"])

    def test_missed_detuning_commits_only_the_frequency_correction(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            result = self.analyze(Path(folder), {"q1": (0.9, 1.1), "q2": (0.5, 1.5)})
            self.assertEqual(result["analysis_status"], "needs_review")
            self.assertEqual(result["passing_targets"], ["q1"])
            self.assertIn("within 20%", " ".join(result["failure_reasons_by_target"]["q2"]))
            self.assertEqual(patch_qubit_targets(result["candidate_state_patch"]), {"q1"})
            correction = {item["path"] for item in result["frequency_correction_patch"]}
            self.assertEqual(
                correction,
                {"/qubits/q2/xy/intermediate_frequency", "/qubits/q2/extras/idle_freq"},
            )
            self.assertFalse(result["fit_quality"]["results"]["q2"]["fit_successful"])


if __name__ == "__main__":
    unittest.main()
