from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jy_agent.service import AgentService, AutonomyScopeError
from jy_agent.util import atomic_write_json
from tests.test_core import make_settings

PORT = "/ports/mw_outputs/con1/7/2"


def _state() -> dict:
    return {
        "ports": {"mw_outputs": {"con1": {"7": {"2": {"upconverter_frequency": 2.2265e9}}}}},
        "qubits": {
            name: {"xy": {"intermediate_frequency": value}}
            for name, value in {"q1": 208.5e6, "q3": 0.0, "q6": -226.5e6}.items()
        },
    }


def _wiring() -> dict:
    return {
        "wiring": {
            "qubits": {
                name: {"xy": {"opx_output": "#" + PORT}} for name in ("q1", "q3", "q6")
            }
        }
    }


class SharedLoCompanionTests(unittest.TestCase):
    """Operator choice 2026-10-14: the lease moves a shared LO on its own."""

    def _service(self, folder: str) -> AgentService:
        settings = make_settings(Path(folder), _state())
        atomic_write_json(settings.wiring_path, _wiring())
        return AgentService(settings)

    def test_rf_preserving_and_parked_companions_stay_in_scope(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = self._service(folder)
            patch = [
                # q1 keeps its RF 2.435 GHz; q6 is parked at -400 MHz.
                {"op": "replace", "path": "/qubits/q1/xy/intermediate_frequency", "value": -65e6},
                {"op": "replace", "path": "/qubits/q6/xy/intermediate_frequency", "value": -400e6},
                {"op": "replace", "path": f"{PORT}/upconverter_frequency", "value": 2.5e9},
                {"op": "replace", "path": "/qubits/q3/xy/intermediate_frequency", "value": 9.5e6},
            ]
            lease = {"targets_json": '["q3"]'}
            service._validate_autonomy_patch_targets(lease, patch)

    def test_rf_changing_companion_still_needs_approval(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = self._service(folder)
            patch = [
                {"op": "replace", "path": "/qubits/q1/xy/intermediate_frequency", "value": -50e6},
                {"op": "replace", "path": f"{PORT}/upconverter_frequency", "value": 2.5e9},
                {"op": "replace", "path": "/qubits/q3/xy/intermediate_frequency", "value": 9.5e6},
            ]
            with self.assertRaises(AutonomyScopeError):
                service._validate_autonomy_patch_targets({"targets_json": '["q3"]'}, patch)
            # Without an LO move, a non-target IF is never in scope.
            with self.assertRaises(AutonomyScopeError):
                service._validate_autonomy_patch_targets(
                    {"targets_json": '["q3"]'}, [patch[0], patch[2]]
                )


if __name__ == "__main__":
    unittest.main()
