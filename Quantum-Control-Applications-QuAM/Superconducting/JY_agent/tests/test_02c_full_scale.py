from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from jy_agent.policy import PolicyEngine
from jy_agent.service import AgentService
from jy_agent.state_patch_02c import derive_02c_state_patch, reconcile_02c_full_scale
from jy_agent.util import atomic_write_json
from tests.test_core import ENTRY_PHRASE, make_settings, sample_state

PORT = "/ports/mw_outputs/con1/7/1/full_scale_power_dbm"


def line_state(names: list[str], full_scale: int = -8) -> tuple[dict, dict]:
    """Qubits sharing one readout port, full scale stored on the port."""
    template = sample_state(0.2, 0.1)["qubits"]["q1"]
    del template["resonator"]["operations"]["readout"]["full_scale_power_dbm"]
    state: dict = {
        "qubits": {},
        "active_qubit_names": list(names),
        "ports": {
            "mw_outputs": {
                "con1": {"7": {"1": {"full_scale_power_dbm": full_scale}}}
            }
        },
    }
    wiring: dict = {"wiring": {"qubits": {}}}
    for index, name in enumerate(names):
        qubit = deepcopy(template)
        qubit["resonator"]["intermediate_frequency"] = 50_000_000 * (index + 1)
        qubit["resonator"]["opx_output"] = f"#/wiring/qubits/{name}/rr/opx_output"
        qubit["resonator"]["operations"]["readout"]["amplitude"] = 0.03
        state["qubits"][name] = qubit
        wiring["wiring"]["qubits"][name] = {
            "rr": {
                "opx_input": "#/ports/mw_inputs/con1/7/1",
                "opx_output": "#/ports/mw_outputs/con1/7/1",
            }
        }
    return state, wiring


def metrics(powers: dict[str, float]) -> dict:
    return {
        "qubits": {
            name: {
                "02c_transition": {
                    "base_frequency_hz": 6.0e9,
                    "dressed_frequency_hz": 6.0e9,
                    "dressed_power_limit_dbm": power,
                    "validation_failures": [],
                    "skip_subsequent_experiments": False,
                }
            }
            for name, power in powers.items()
        }
    }


class FullScale02cTests(unittest.TestCase):
    """Operator instruction 2026-10-02: 10 dB backoff, smallest legal full scale."""

    def derive(self, folder, names, powers, targets=None):
        state, wiring = line_state(names)
        settings = make_settings(Path(folder), state)
        atomic_write_json(settings.wiring_path, wiring)
        raw = PolicyEngine(settings).raw
        # These cases test the full-scale arithmetic on the 10 dB backoff they
        # were written for; the live backoff is 3 dB since 2026-10-13.
        raw["analysis"]["02c"]["readout_power_backoff_db"] = 10.0
        patch, errors = derive_02c_state_patch(
            settings, state, metrics(powers), targets or list(powers), raw
        )
        values = {item["path"]: item["value"] for item in patch}
        return settings, state, raw, values, patch, errors

    def test_low_powers_take_the_minimum_full_scale(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            names = ["q1", "q2", "q3", "q4", "q5"]
            *_, values, _, errors = self.derive(folder, names, {n: -38.0 for n in names})
            self.assertEqual(errors, [])
            self.assertEqual(values[PORT], -11)
            self.assertAlmostEqual(
                values["/qubits/q1/resonator/operations/readout/amplitude"],
                10 ** ((-48.0 + 11.0) / 20.0),
            )

    def test_full_scale_is_the_smallest_integer_within_the_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            names = ["q1", "q2", "q3", "q4", "q5"]
            # -11 dBm per tone after backoff: 5 * 10**((-11 - fs)/20) <= 0.5
            # first holds at fs = 9.
            *_, values, _, errors = self.derive(folder, names, {n: -1.0 for n in names})
            self.assertEqual(errors, [])
            self.assertEqual(values[PORT], 9)
            amplitudes = [
                values[f"/qubits/{n}/resonator/operations/readout/amplitude"]
                for n in names
            ]
            self.assertLessEqual(sum(amplitudes), 0.5)
            self.assertGreater(5 * 10 ** ((-11.0 - 8) / 20.0), 0.5)

    def test_pulse_limit_drives_a_single_loud_tone(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            # q1 at -20 dBm: 10**((-20 - fs)/20) <= 0.125 first holds at fs = -1.
            *_, values, _, errors = self.derive(
                folder, ["q1", "q2"], {"q1": -10.0, "q2": -50.0}
            )
            self.assertEqual(errors, [])
            self.assertEqual(values[PORT], -1)

    def test_unreachable_power_fails_that_qubit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            *_, values, _, errors = self.derive(folder, ["q1"], {"q1": 30.0})
            self.assertNotIn(PORT, values)
            self.assertNotIn("/qubits/q1/resonator/operations/readout/amplitude", values)
            self.assertTrue(errors and errors[0].startswith("q1 "))

    def test_subgroup_keeps_the_shared_full_scale(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            *_, values, _, errors = self.derive(
                folder, ["q1", "q2", "q3"], {"q2": -38.0}, targets=["q2"]
            )
            self.assertEqual(errors, [])
            self.assertNotIn(PORT, values)
            self.assertAlmostEqual(
                values["/qubits/q2/resonator/operations/readout/amplitude"],
                10 ** ((-48.0 + 8.0) / 20.0),
            )

    def test_partial_pass_keeps_power_at_the_old_full_scale(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings, state, raw, values, patch, _ = self.derive(
                folder, ["q1", "q2"], {"q1": -38.0, "q2": -38.0}
            )
            self.assertEqual(values[PORT], -11)
            kept = [i for i in patch if i["path"].startswith("/qubits/q1/")]
            dropped = [i for i in patch if i not in kept]
            result, warnings = reconcile_02c_full_scale(
                settings, state, kept, dropped, {"q1"}, raw
            )
            out = {i["path"]: i["value"] for i in result}
            self.assertNotIn(PORT, out)
            self.assertAlmostEqual(
                out["/qubits/q1/resonator/operations/readout/amplitude"],
                10 ** ((-48.0 + 8.0) / 20.0),
            )
            self.assertTrue(warnings)

    def test_every_qubit_passing_restores_the_full_scale(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings, state, raw, _, patch, _ = self.derive(
                folder, ["q1", "q2"], {"q1": -38.0, "q2": -38.0}
            )
            kept = [i for i in patch if i["path"].startswith("/qubits/")]
            dropped = [i for i in patch if i not in kept]
            result, _ = reconcile_02c_full_scale(
                settings, state, kept, dropped, {"q1", "q2"}, raw
            )
            self.assertEqual({i["path"]: i["value"] for i in result}[PORT], -11)

    def test_port_full_scale_patch_passes_policy(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings, state, _, _, patch, _ = self.derive(
                folder, ["q1", "q2"], {"q1": -38.0, "q2": -38.0}
            )
            PolicyEngine(settings).validate_state_patch(patch, state)


class FullScale02cCommitTests(unittest.TestCase):
    """The 2026-10-02 bring-up: the 02c commit was refused because the port
    path names no qubit."""

    def service(self, folder: str, names: list[str], targets: list[str]):
        state, wiring = line_state(names)
        state["active_qubit_names"] = list(targets)
        settings = make_settings(Path(folder), state)
        atomic_write_json(settings.wiring_path, wiring)
        service = AgentService(settings)
        workflow = service.start_workflow(
            targets, {"multiplexed": True}, "unittest", ENTRY_PHRASE
        )
        return service, workflow

    def add_run(self, service, workflow, node_id: str) -> str:
        run_id = f"offline-{node_id}"
        proposal = service.request_run(
            workflow["id"],
            "02x",
            {"qubits": list(workflow["targets"])},
            "Proposal row for an offline run.",
            "unittest",
        )
        service.db.execute(
            """
            INSERT INTO runs(
                id, workflow_id, proposal_id, node_id, parameters_json,
                status, analysis_status, analysis_json
            ) VALUES (?, ?, ?, ?, '{}', 'completed', 'pass', '{}')
            """,
            (run_id, workflow["id"], proposal["id"], node_id),
        )
        return run_id

    def patch(self) -> list[dict]:
        return [
            {"op": "replace", "path": PORT, "value": -11},
            {
                "op": "replace",
                "path": "/qubits/q1/resonator/operations/readout/amplitude",
                "value": 0.02,
            },
        ]

    def test_02c_run_may_commit_its_readout_line_full_scale(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = self.service(folder, ["q1", "q2"], ["q1", "q2"])
            run_id = self.add_run(service, workflow, "02c")
            proposal = service.request_state_commit(
                workflow["id"], self.patch(), "02c full scale.", "unittest", run_id
            )
            self.assertEqual(proposal["payload"]["patch"][0]["path"], PORT)

    def test_other_nodes_may_not_commit_a_port_full_scale(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = self.service(folder, ["q1", "q2"], ["q1", "q2"])
            run_id = self.add_run(service, workflow, "07d")
            with self.assertRaisesRegex(Exception, "must belong to a workflow target"):
                service.request_state_commit(
                    workflow["id"], self.patch(), "07d full scale.", "unittest", run_id
                )

    def test_port_shared_with_a_non_target_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow = self.service(folder, ["q1", "q2"], ["q1"])
            run_id = self.add_run(service, workflow, "02c")
            with self.assertRaisesRegex(Exception, "must belong to a workflow target"):
                service.request_state_commit(
                    workflow["id"], self.patch(), "02c full scale.", "unittest", run_id
                )


if __name__ == "__main__":
    unittest.main()
