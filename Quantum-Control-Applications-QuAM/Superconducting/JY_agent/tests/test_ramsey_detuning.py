from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jy_agent.policy import PolicyEngine, PolicyError
from tests.test_core import make_settings, sample_state


class RamseyDetuningWindowTests(unittest.TestCase):
    """Detuning [MHz] x (max - min wait) [us] = 4 (2026-10-02, exact 2026-10-15)."""

    def policy(self, folder: str) -> PolicyEngine:
        return PolicyEngine(make_settings(Path(folder), sample_state(0.2, 0.1)))

    def test_defaults_satisfy_the_rule(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            merged, _ = self.policy(folder).validate_run("06", {"qubits": ["q1"]})
            window_us = (merged["max_wait_time_in_ns"] - merged["min_wait_time_in_ns"]) / 1000
            self.assertAlmostEqual(
                merged["frequency_detuning_in_mhz"] * window_us, 4.0, places=6
            )

    def test_omitted_detuning_follows_the_window(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            policy = self.policy(folder)
            for wait_ns, expected_mhz in ((1016, 4.0), (10016, 0.4), (20016, 0.2), (60016, 4 / 60)):
                merged, _ = policy.validate_run(
                    "06", {"qubits": ["q1"], "max_wait_time_in_ns": wait_ns}
                )
                self.assertAlmostEqual(
                    merged["frequency_detuning_in_mhz"], expected_mhz, places=5
                )

    def test_matching_explicit_detuning_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            merged, _ = self.policy(folder).validate_run(
                "06",
                {
                    "qubits": ["q1"],
                    "max_wait_time_in_ns": 20016,
                    "frequency_detuning_in_mhz": 0.2,
                },
            )
            self.assertEqual(merged["frequency_detuning_in_mhz"], 0.2)

    def test_mismatched_explicit_detuning_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            policy = self.policy(folder)
            # The 2026-09-30 run: 2 MHz over 60 us is 120 MHz*us, far off.
            with self.assertRaisesRegex(PolicyError, r"0\.06667 MHz"):
                policy.validate_run(
                    "06",
                    {
                        "qubits": ["q1"],
                        "max_wait_time_in_ns": 60016,
                        "frequency_detuning_in_mhz": 2,
                    },
                )
            # 2026-10-15: the product is exact; 25% off is no longer accepted.
            with self.assertRaisesRegex(PolicyError, r"0\.2 MHz"):
                policy.validate_run(
                    "06",
                    {
                        "qubits": ["q1"],
                        "max_wait_time_in_ns": 20016,
                        "frequency_detuning_in_mhz": 0.22,
                    },
                )

    def test_wait_times_sit_on_the_grid(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            policy = self.policy(folder)
            with self.assertRaisesRegex(PolicyError, "min_wait_time_in_ns must be 16"):
                policy.validate_run(
                    "06", {"qubits": ["q1"], "min_wait_time_in_ns": 20, "max_wait_time_in_ns": 20020}
                )
            for wait_ns in (20000, 20100, 16):
                with self.assertRaisesRegex(PolicyError, "multiple of 500"):
                    policy.validate_run("06", {"qubits": ["q1"], "max_wait_time_in_ns": wait_ns})

    def test_other_nodes_are_not_constrained(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            definition = self.policy(folder).node_definition("06b")
            self.assertNotIn("detuning_wait_product_mhz_us", definition)


if __name__ == "__main__":
    unittest.main()
