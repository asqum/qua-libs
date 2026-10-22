"""T1/T2*/T2e caps, operator instruction 2026-10-20.

At most 1000 averages and 101 time points; the 06 calibration window is
816 ns (101 points at 8 ns), on the 16 + 400n ns grid.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jy_agent.policy import PolicyEngine, PolicyError
from tests.test_core import make_settings, sample_state


class NodeCapTests(unittest.TestCase):
    def policy(self, folder: str) -> PolicyEngine:
        return PolicyEngine(make_settings(Path(folder), sample_state(0.2, 0.1)))

    def test_averages_above_1000_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            policy = self.policy(folder)
            for node in ("05", "06", "06b"):
                with self.assertRaises(PolicyError):
                    policy.validate_run(node, {"qubits": ["q1"], "num_averages": 2000})
                policy.validate_run(node, {"qubits": ["q1"], "num_averages": 1000})

    def test_more_than_101_points_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            policy = self.policy(folder)
            with self.assertRaises(PolicyError):
                policy.validate_run("06", {"qubits": ["q1"], "num_time_points": 400})
            with self.assertRaises(PolicyError):
                policy.validate_run(
                    "06b",
                    {"qubits": ["q1"], "max_wait_time_in_ns": 200000, "wait_time_step_in_ns": 1000},
                )
            policy.validate_run(
                "05", {"qubits": ["q1"], "max_wait_time_in_ns": 200016, "wait_time_step_in_ns": 2000}
            )

    def test_defaults_respect_the_caps(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            policy = self.policy(folder)
            for node in ("05", "06", "06b"):
                merged, _ = policy.validate_run(node, {"qubits": ["q1"]})
                self.assertLessEqual(merged["num_averages"], 1000)

    def test_the_816_ns_calibration_window_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            merged, _ = self.policy(folder).validate_run(
                "06", {"qubits": ["q1"], "max_wait_time_in_ns": 816, "num_time_points": 101}
            )
            self.assertAlmostEqual(merged["frequency_detuning_in_mhz"], 5.0, places=5)


if __name__ == "__main__":
    unittest.main()
