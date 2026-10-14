"""02x probe headroom and unique grid cells (operator debug request 2026-10-11)."""

from __future__ import annotations

import math
import unittest

from jy_agent.setup_02x import (
    _probe_amplitude,
    probe_02x_headroom_patch,
    unique_grid_location_patch,
    unique_thread_patch,
)


def line_state(full_scale: float, names: list[str], grid: list[str]) -> tuple[dict, dict]:
    state = {
        "ports": {"mw_outputs": {"con1": {"7": {"1": {"full_scale_power_dbm": full_scale}}}}},
        "qubits": {
            name: {
                "grid_location": cell,
                "resonator": {
                    "opx_output": "#/ports/mw_outputs/con1/7/1",
                    "operations": {"readout": {"amplitude": 0.05}},
                },
            }
            for name, cell in zip(names, grid)
        },
    }
    return state, {}


class Probe02xHeadroomTests(unittest.TestCase):
    def test_low_full_scale_clips_and_is_raised_keeping_power(self) -> None:
        names = [f"q{i}" for i in range(1, 7)]
        state, wiring = line_state(-11, names, ["0,0"] * 6)
        self.assertGreater(6 * _probe_amplitude(-11), 1.0)
        patch, details = probe_02x_headroom_patch(state, wiring, names)
        line = details["/ports/mw_outputs/con1/7/1"]
        self.assertEqual(line["new_full_scale_dbm"], 7)
        self.assertLessEqual(line["summed_probe_amplitude"][1], 1.0)
        amplitude = next(
            item["value"] for item in patch if item["path"].endswith("q1/resonator/operations/readout/amplitude")
        )
        old_power = -11 + 20 * math.log10(0.05)
        new_power = 7 + 20 * math.log10(amplitude)
        self.assertAlmostEqual(old_power, new_power, places=9)

    def test_high_full_scale_needs_nothing(self) -> None:
        names = ["q1", "q2", "q3"]
        state, wiring = line_state(10, names, ["0,0", "1,0", "2,0"])
        self.assertEqual(probe_02x_headroom_patch(state, wiring, names), ([], {}))

    def test_shared_grid_cells_are_spread(self) -> None:
        state, _ = line_state(10, ["q1", "q2", "q3", "q10"], ["0,0", "1,0", "0,0", "1,0"])
        patch, details = unique_grid_location_patch(state)
        cells = {name: qubit["grid_location"] for name, qubit in state["qubits"].items()}
        for item in patch:
            cells[item["path"].split("/")[2]] = item["value"]
        self.assertEqual(len(set(cells.values())), 4)
        self.assertEqual(cells["q10"], "3,0")
        state, _ = line_state(10, ["q1", "q2"], ["0,0", "1,0"])
        self.assertEqual(unique_grid_location_patch(state), ([], {}))



class UniqueThreadTests(unittest.TestCase):
    def test_shared_resonator_thread_gets_one_thread_per_qubit(self) -> None:
        state = {
            "qubits": {
                name: {"resonator": {"thread": "a"}, "xy": {"thread": xy}}
                for name, xy in (("q1", "b"), ("q2", "c"), ("q10", "b"))
            }
        }
        patch, details = unique_thread_patch(state)
        values = {item["path"]: item["value"] for item in patch}
        self.assertEqual(values["/qubits/q1/xy/thread"], "a")
        self.assertEqual(values["/qubits/q2/resonator/thread"], "b")
        self.assertEqual(values["/qubits/q10/resonator/thread"], "c")
        self.assertEqual(values["/qubits/q10/xy/thread"], "c")
        self.assertNotIn("/qubits/q1/resonator/thread", values)
        self.assertIn("q2", details)

    def test_distinct_threads_need_nothing(self) -> None:
        state = {"qubits": {"q1": {"resonator": {"thread": "f"}, "xy": {"thread": "f"}},
                            "q2": {"resonator": {"thread": "l"}, "xy": {"thread": "l"}}}}
        self.assertEqual(unique_thread_patch(state), ([], {}))


class SingleQubit02xDeviceTests(unittest.TestCase):
    def test_listed_chip_runs_02x_one_qubit_at_a_time(self) -> None:
        import tempfile
        from dataclasses import replace
        from pathlib import Path

        from jy_agent.service import AgentService

        from test_core import make_settings, sample_state

        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            service = AgentService(settings)
            self.assertIsNone(service._node_multiplex_cap("02x"))
            original = service.policy.node_definition

            def listed_definition(node_id):
                definition = dict(original(node_id))
                if node_id in {"02x", "02c"}:
                    definition["single_qubit_devices"] = ["CRSQ_26009_die(5,-1)"]
                return definition

            service.policy.node_definition = listed_definition
            listed = Path(folder) / "CRSQ_26009_die(5,-1)"
            listed.mkdir()
            service.settings = replace(settings, quam_state_root=listed)
            self.assertEqual(service._node_multiplex_cap("02x"), 1)
            self.assertEqual(service._node_multiplex_cap("02c"), 1)
            self.assertIsNone(service._node_multiplex_cap("02a"))


if __name__ == "__main__":
    unittest.main()
