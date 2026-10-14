from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jy_agent.analysis import SnapshotAnalyzer
from jy_agent.policy import PolicyEngine
from jy_agent.workflow_subgroups import resolve_node_targets

from test_core import make_settings, sample_state


class StatisticsAnalysisTests(unittest.TestCase):
    def _analyzer(self, folder: str) -> SnapshotAnalyzer:
        settings = make_settings(Path(folder), sample_state(0.2, 0.1))
        return SnapshotAnalyzer(settings, PolicyEngine(settings))

    def test_statistics_metadata_does_not_require_or_refit_raw_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            analyzer = self._analyzer(folder)
            metrics = analyzer._dataset_metrics(
                Path(folder) / "snapshot-without-ds",
                "05st",
                {
                    "initial_parameters": {"histo_num": 100},
                    "t1_stats": {"q1": {"mu_us": 40.0, "sigma_us": 0.54}},
                },
            )

            self.assertNotIn("error", metrics)
            self.assertEqual(metrics["validation_mode"], "baseline_prerequisites_only")
            self.assertFalse(metrics["raw_iteration_refit_required"])
            self.assertEqual(
                metrics["protected_node_summaries"]["q1"]["mu_us"], 40.0
            )

    def test_one_hundred_repetition_run_completes_without_secondary_fit_gate(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            analyzer = self._analyzer(folder)
            result = analyzer._fit_quality(
                "06st_t2star",
                {
                    "initial_parameters": {"histo_num": 100},
                    # Informational output is not re-fitted and cannot invalidate
                    # a completed statistics run.
                    "t2*_stats": {"q1": {"mu_us": None, "sigma_us": None}},
                },
                {
                    "validation_mode": "baseline_prerequisites_only",
                    "raw_iteration_refit_required": False,
                    "completed_iterations": 100,
                },
                ["q1"],
            )

            self.assertEqual(result["failures"], [])
            self.assertEqual(result["results"]["q1"]["histo_num"], 100)
            self.assertTrue(result["results"]["q1"]["fit_successful"])
            self.assertFalse(
                result["results"]["q1"]["raw_iteration_refit_required"]
            )

    def test_a_run_that_lost_the_instrument_is_not_a_result(self) -> None:
        # 2026-10-13 (q1 06st_t2e #493): 3 of 100 repetitions completed.
        with tempfile.TemporaryDirectory() as folder:
            analyzer = self._analyzer(folder)
            result = analyzer._fit_quality(
                "06st_t2e",
                {"initial_parameters": {"histo_num": 100}},
                {"completed_iterations": 3},
                ["q1"],
            )
            self.assertFalse(result["results"]["q1"]["fit_successful"])
            self.assertIn("completed only 3 of 100", " ".join(result["failures"]))

    def test_completed_iterations_are_read_from_the_dataset(self) -> None:
        import numpy as np
        import xarray as xr

        from jy_agent.analysis import _completed_iterations

        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder)
            xr.Dataset(
                {"I": (("iteration", "idle_time"), np.zeros((7, 4)))}
            ).to_netcdf(snapshot / "ds.h5")
            self.assertEqual(_completed_iterations(snapshot), 7)
            self.assertIsNone(_completed_iterations(snapshot / "missing"))

    def test_a_mean_dominated_by_failed_fits_is_annotated_not_failed(self) -> None:
        # Operator choice 2026-10-13 (q1 06st_t2e #494: 26481 +- 119514 us).
        with tempfile.TemporaryDirectory() as folder:
            analyzer = self._analyzer(folder)
            result = analyzer._fit_quality(
                "06st_t2e",
                {"initial_parameters": {"histo_num": 100}},
                {
                    "completed_iterations": 100,
                    "protected_node_summaries": {"q1": {"mu_us": 26481.0, "sigma_us": 119514.0}},
                    "robust_summaries": {"q1": {"fits": 99, "iterations": 100,
                                                "median_us": 40.6, "q25_us": 30.4, "q75_us": 65.8}},
                },
                ["q1"],
            )
            self.assertEqual(result["failures"], [])
            self.assertTrue(result["results"]["q1"]["fit_successful"])
            self.assertIs(result["results"]["q1"]["summary_robust"], False)
            self.assertEqual(result["results"]["q1"]["robust_summary"]["median_us"], 40.6)

    def test_robust_refit_takes_the_larger_signal_in_either_direction(self) -> None:
        import numpy as np
        import xarray as xr

        from jy_agent.analysis import _robust_statistics

        with tempfile.TemporaryDirectory() as folder:
            times = np.linspace(0.016, 160.0, 80)
            rising = 1.0 - np.exp(-times / 40.0)  # q2-like trace
            data = np.stack([rising + 0.001 * np.sin(k + times) for k in range(5)])
            xr.Dataset(
                {
                    "I": (("iteration", "qubit", "idle_time"), data[:, None, :]),
                    "Q": (("iteration", "qubit", "idle_time"), 0.01 * data[:, None, :]),
                },
                coords={"idle_time": times, "qubit": ["q2"]},
            ).to_netcdf(Path(folder) / "ds.h5")
            summary = _robust_statistics(Path(folder))["q2"]
            self.assertEqual(summary["fits"], 5)
            self.assertAlmostEqual(summary["median_us"], 40.0, delta=1.0)

    def test_t2star_refit_follows_the_ramsey_fringes(self) -> None:
        # 2026-10-15: a plain decay misreads fringes; 06st_t2star fits a damped cosine.
        import numpy as np
        import xarray as xr

        from jy_agent.analysis import _robust_statistics

        with tempfile.TemporaryDirectory() as folder:
            times = np.linspace(0.016, 60.0, 120)
            rng = np.random.default_rng(3)
            data = np.stack(
                [
                    0.4 * np.exp(-times / 20.0) * np.cos(2 * np.pi * 0.1 * times + 0.3)
                    + 0.5
                    + 0.01 * rng.normal(size=times.size)
                    for _ in range(5)
                ]
            )
            xr.Dataset(
                {"state": (("iteration", "qubit", "idle_time"), data[:, None, :])},
                coords={"idle_time": times, "qubit": ["q1"]},
            ).to_netcdf(Path(folder) / "ds.h5")
            summary = _robust_statistics(Path(folder), "06st_t2star")["q1"]
            self.assertEqual(summary["model"], "damped_cosine")
            self.assertEqual(summary["kept"], 5)
            self.assertAlmostEqual(summary["median_us"], 20.0, delta=1.5)

    def test_statistics_still_requires_exactly_one_hundred_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            analyzer = self._analyzer(folder)
            result = analyzer._fit_quality(
                "06st_t2e",
                {"initial_parameters": {"histo_num": 99}},
                {},
                ["q1"],
            )

            self.assertFalse(result["results"]["q1"]["fit_successful"])
            self.assertIn("exactly 100 repetitions", "\n".join(result["failures"]))

    def test_single_completed_statistics_run_resolves_its_target(self) -> None:
        analysis = {
            "fit_quality": {
                "results": {
                    "q1": {
                        "fit_successful": True,
                        "histo_num": 100,
                        "raw_iteration_refit_required": False,
                    }
                }
            },
            "dataset_metrics": {"qubits": {}},
        }
        resolved = resolve_node_targets(
            "05st",
            [
                {
                    "status": "completed",
                    "analysis_status": "pass",
                    "analysis_json": json.dumps(analysis),
                    "parameters_json": json.dumps({"qubits": ["q1"]}),
                    "decision": "advance",
                }
            ],
        )

        self.assertEqual(resolved, {"q1"})


if __name__ == "__main__":
    unittest.main()


class FilteredStatisticsTests(unittest.TestCase):
    """Operator decisions 2026-10-13: filtered statistics and 07b below target."""

    def test_the_record_takes_the_filtered_mean_and_the_working_lifetime_stays(self) -> None:
        from jy_agent.analysis import _filtered_statistics_patch

        patch = [
            {"op": "add", "path": "/qubits/q1/T2echo", "value": 0.0265},
            {"op": "replace", "path": "/qubits/q1/extras/T2", "value": 0.0265},
            {"op": "replace", "path": "/qubits/q1/extras/T2_dev", "value": 0.1195},
        ]
        kept, warnings = _filtered_statistics_patch(
            "06st_t2e",
            patch,
            {"q1": {"kept": 88, "iterations": 100, "filtered_mean_us": 48.3, "filtered_std_us": 27.2}},
        )
        values = {item["path"]: item["value"] for item in kept}
        self.assertNotIn("/qubits/q1/T2echo", values)
        self.assertAlmostEqual(values["/qubits/q1/extras/T2"], 48.3e-6)
        self.assertAlmostEqual(values["/qubits/q1/extras/T2_dev"], 27.2e-6)
        self.assertIn("88/100", warnings[0])

    def test_07b_below_the_fidelity_target_still_passes_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            analyzer = SnapshotAnalyzer(settings, PolicyEngine(settings))
            analyzer.policy.raw["analysis"]["07b"]["min_readout_fidelity_blocks_pass"] = False
            result = analyzer._fit_quality(
                "07b",
                {"results": {"q1": {"fidelity": 62.8, "angle": -3.1, "threshold": -1e-4}},
                 "initial_parameters": {"reset_type_thermal_or_active": "thermal"}},
                {"qubits": {"q1": {"morphology_pass": True}}},
                ["q1"],
            )
            self.assertEqual(result["failures"], [])
            self.assertTrue(result["results"]["q1"]["fit_successful"])
            self.assertTrue(result["results"]["q1"]["below_fidelity_target"])
            analyzer.policy.raw["analysis"]["07b"]["min_readout_fidelity_blocks_pass"] = True
            blocked = analyzer._fit_quality(
                "07b",
                {"results": {"q1": {"fidelity": 62.8}},
                 "initial_parameters": {"reset_type_thermal_or_active": "thermal"}},
                {"qubits": {"q1": {"morphology_pass": True}}},
                ["q1"],
            )
            self.assertFalse(blocked["results"]["q1"]["fit_successful"])
