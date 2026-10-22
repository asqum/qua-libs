"""Fitting crashes release the lock and keep the lease, operator 2026-10-19.

q2 06 run 842fcbcf crashed in quam_libs/experiments/ramsey/analysis/fitting.py
after the data was acquired. Such a crash is classified `analysis_failure`:
the worker releases the hardware lock and the lease stays active so the agent
repeats the run.
"""

from __future__ import annotations

import unittest

from jy_agent.failures import classify_failure


def raise_from(filename: str, source: str) -> BaseException:
    namespace: dict = {}
    exec(compile(source, filename, "exec"), namespace)
    try:
        namespace["run"]()
    except BaseException as exc:  # noqa: BLE001 - the test inspects it
        return exc
    raise AssertionError("expected an exception")


FITTING = "D:/QM/Superconducting/quam_libs/experiments/ramsey/analysis/fitting.py"
QM_CLIENT = "C:/env/lib/site-packages/qm/api/job_api.py"
NODE = "D:/QM/Superconducting/calibration_graph/06_Ramsey.py"


class AnalysisFailureTests(unittest.TestCase):
    def test_a_fitting_key_error_is_an_analysis_failure(self) -> None:
        exc = raise_from(FITTING, "def run():\n    raise KeyError(\"not all values found in index 'sign'\")\n")
        result = classify_failure(exc)
        self.assertEqual(result.category, "analysis_failure")
        self.assertTrue(result.safe_to_release_lock)

    def test_a_crash_through_qm_is_not(self) -> None:
        exc = raise_from(QM_CLIENT, "def run():\n    raise KeyError('x')\n")
        self.assertNotEqual(classify_failure(exc).category, "analysis_failure")

    def test_a_crash_in_the_node_body_is_not(self) -> None:
        exc = raise_from(NODE, "def run():\n    raise ValueError('bad parameter')\n")
        self.assertIsNone(classify_failure(exc).category)


if __name__ == "__main__":
    unittest.main()
