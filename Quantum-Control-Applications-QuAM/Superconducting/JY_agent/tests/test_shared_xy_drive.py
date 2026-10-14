"""A chip that shares one XY drive output runs one qubit at a time.

Operator instruction 2026-09-22. `IQM_4SQ` routes the XY drive of all four
qubits to `#/ports/mw_outputs/con1/8/3`, so they also share one LO. 02x, 02c
and 02a still multiplex every target because they only use the readout line.
From 03a the workflow takes ONE qubit through the whole remaining sequence --
to the end, or until that qubit is unmeasurable -- and only then returns to 03a
for the next qubit. That is what makes the shared LO safe: it moves between
passes, never inside one, and the qubits it drags along have their IF
compensated so their RF is preserved.

The trigger is the wiring file, so the rule switches itself off for a chip that
gives every qubit its own output.
"""

from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from jy_agent.policy import PolicyError
from jy_agent.service import AgentService
from jy_agent.state import StateError, drive_lo_recenter_patch, shared_xy_drive_groups
from jy_agent.util import atomic_write_json, json_dumps

from test_autonomy import activate_lease
from test_core import make_settings, sample_state, start_test_workflow
from test_next_action_planner import analysis_document

TARGETS = ["q1", "q2", "q3"]
SHARED_XY = "#/ports/mw_outputs/con1/8/3"


def chip_state(targets: list[str] = TARGETS) -> dict:
    state = sample_state(0.2, 0.1)
    template = state["qubits"]["q1"]
    for index, name in enumerate(targets):
        qubit = deepcopy(template)
        qubit["xy"]["intermediate_frequency"] = 20_000_000.0 * index
        # Distinct readout IFs: equal ones would force 02 (rule 2026-10-07).
        qubit["resonator"]["intermediate_frequency"] = (
            float(qubit["resonator"]["intermediate_frequency"]) + 5_000_000.0 * index
        )
        state["qubits"][name] = qubit
    state["active_qubit_names"] = list(targets)
    state["ports"] = {
        "mw_outputs": {
            "con1": {
                "8": {"3": {"band": 1, "upconverter_frequency": 3_000_000_000}}
            }
        }
    }
    return state


def chip_wiring(shared: bool, targets: list[str] = TARGETS) -> dict:
    qubits: dict[str, dict] = {}
    for index, name in enumerate(targets, start=2):
        qubits[name] = {
            "rr": {
                "opx_input": "#/ports/mw_inputs/con1/8/2",
                "opx_output": "#/ports/mw_outputs/con1/8/1",
            },
            "xy": {
                "opx_output": (
                    SHARED_XY
                    if shared
                    else f"#/ports/mw_outputs/con1/8/{index}"
                )
            },
        }
    return {"wiring": {"qubits": qubits}}


def build(folder: str, shared: bool, node_id: str) -> tuple[AgentService, dict, dict]:
    settings = make_settings(Path(folder), chip_state())
    atomic_write_json(settings.wiring_path, chip_wiring(shared))
    service = AgentService(settings)
    workflow = start_test_workflow(service, list(TARGETS))
    service.db.execute(
        "UPDATE workflows SET current_node = ? WHERE id = ?",
        (node_id, workflow["id"]),
    )
    _proposal, lease = activate_lease(service, workflow["id"])
    return service, service._workflow(workflow["id"]), lease


def t1_analysis_document(qubit: str) -> dict:
    """A 05 result that clears every registered T1 acceptance rule."""

    document = analysis_document("05", {qubit: 999.0})
    document["fit_quality"]["results"][qubit] = {
        "fit_successful": True,
        "t1_seconds": 3.0e-5,
        "relative_uncertainty": 0.05,
        "r_squared": 0.99,
        "coverage_lifetimes": 5.0,
        "samples_per_lifetime": 8.0,
        "fit_at_search_boundary": False,
    }
    return document


def coarse_search_document(qubit: str) -> dict:
    """A 03a result that clears every registered fine-scan acceptance rule."""

    document = analysis_document("03a", {qubit: 999.0})
    document["03a_stage"] = "fine"
    document["fit_quality"]["results"][qubit] = {
        "fit_successful": True,
        "drive_freq": 3_020_000_000.0,
    }
    document["dataset_metrics"]["qubits"][qubit]["feature_fwhm_hz"] = 2_000_000.0
    return document


def record_node(
    service: AgentService,
    workflow: dict,
    lease: dict,
    node_id: str,
    qubit: str,
    run_id: str,
    passing: bool = True,
) -> str:
    """One completed, analyzed run of `node_id` for a single qubit."""

    parameters: dict = {"qubits": [qubit]}
    if node_id == "03a":
        # 2026-10-14: the first coarse run uses 1.0 on a shared XY line.
        parameters["operation_amplitude_factor"] = (
            1.0 if service.policy.shared_xy_drive_groups() else 0.01
        )
    proposal = service.request_run(
        workflow["id"],
        node_id,
        parameters,
        f"{node_id} for {qubit}.",
        "unittest-agent",
        autonomy_lease_id=lease["id"],
    )
    if not passing:
        analysis = analysis_document(node_id, {qubit: 1.0})
    elif node_id == "05":
        analysis = t1_analysis_document(qubit)
    elif node_id == "03a":
        analysis = coarse_search_document(qubit)
    else:
        analysis = analysis_document(node_id, {qubit: 999.0})
    service.db.execute(
        "INSERT INTO runs(id, workflow_id, proposal_id, node_id, "
        "parameters_json, status, analysis_status, analysis_json, "
        "autonomy_lease_id) "
        "VALUES (?, ?, ?, ?, ?, 'completed', 'needs_review', ?, ?)",
        (
            run_id,
            workflow["id"],
            proposal["id"],
            node_id,
            json_dumps({"qubits": [qubit]}),
            json_dumps(analysis),
            lease["id"],
        ),
    )
    return run_id


class SharedXyDriveTests(unittest.TestCase):
    def test_wiring_reader_finds_only_the_shared_outputs(self) -> None:
        self.assertEqual(
            shared_xy_drive_groups(chip_wiring(shared=True)),
            {SHARED_XY: ["q1", "q2", "q3"]},
        )
        self.assertEqual(shared_xy_drive_groups(chip_wiring(shared=False)), {})
        self.assertEqual(shared_xy_drive_groups({}), {})

    def test_the_tail_belongs_to_one_qubit_at_a_time(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, shared=True, node_id="04")

            plan = service.next_action(workflow["id"])

            self.assertEqual(plan["action"], "run")
            self.assertEqual(plan["run"]["qubits"], ["q1"])
            self.assertEqual(plan["targets"]["active"], ["q1"])
            self.assertEqual(plan["serial_pass"]["qubit"], "q1")
            self.assertEqual(plan["serial_pass"]["position"], 1)
            self.assertEqual(plan["serial_pass"]["total"], 3)
            self.assertEqual(plan["serial_pass"]["remaining"], TARGETS)

    def test_private_outputs_keep_the_node_by_node_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, shared=False, node_id="04")

            plan = service.next_action(workflow["id"])

            self.assertEqual(plan["run"]["qubits"], TARGETS)
            self.assertNotIn("serial_pass", plan)

    def test_nodes_before_03a_still_multiplex_on_a_shared_output(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, shared=True, node_id="02x")

            plan = service.next_action(workflow["id"])

            self.assertEqual(plan["run"]["qubits"], TARGETS)
            merged, _warnings = service.policy.validate_run(
                "02x", {"qubits": list(TARGETS)}
            )
            self.assertEqual(merged["qubits"], TARGETS)

    def test_a_multi_qubit_tail_request_is_refused_before_the_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, _lease = build(folder, shared=True, node_id="04")

            with self.assertRaises(PolicyError) as caught:
                service.policy.validate_run("04", {"qubits": ["q1", "q2"]})
            self.assertIn("shares XY drive outputs", str(caught.exception))

            merged, _warnings = service.policy.validate_run("04", {"qubits": ["q1"]})
            self.assertEqual(merged["qubits"], ["q1"])

    def test_a_resolved_qubit_walks_on_to_the_next_node(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, shared=True, node_id="04")
            record_node(service, workflow, lease, "04", "q1", "04-q1")

            decision = service.record_decision(
                workflow["id"],
                "04-q1",
                "advance",
                "q1 has a power Rabi; stay on q1 and take it to 05.",
                "05",
                None,
                [],
                "unittest-agent",
            )

            self.assertEqual(decision["next_action"]["next_node"], "05")
            plan = service.next_action(workflow["id"])
            self.assertEqual(plan["current_node"], "05")
            self.assertEqual(plan["serial_pass"]["qubit"], "q1")

    def test_a_finished_pass_returns_to_03a_for_the_next_qubit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, shared=True, node_id="05")
            record_node(service, workflow, lease, "05", "q1", "05-q1")

            # 05 is the last node of this test sequence, so q1's pass is over
            # and the next advance restarts the tail for q2 instead of ending
            # the workflow.
            self.assertEqual(
                service._expected_next_node(
                    service._workflow(workflow["id"]), "05"
                ),
                "03a",
            )

            service.record_decision(
                workflow["id"],
                "05-q1",
                "advance",
                "q1 is done; start q2 at 03a.",
                "03a",
                None,
                [],
                "unittest-agent",
            )

            plan = service.next_action(workflow["id"])
            self.assertEqual(plan["current_node"], "03a")
            self.assertEqual(plan["serial_pass"]["qubit"], "q2")
            self.assertEqual(plan["serial_pass"]["finished"], ["q1"])
            self.assertEqual(plan["targets"]["active"], ["q2"])

    def test_an_unmeasurable_qubit_ends_its_pass_instead_of_the_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, shared=True, node_id="04")
            record_node(service, workflow, lease, "04", "q1", "04-q1", passing=False)
            service.mark_scientifically_unmeasurable(
                workflow["id"],
                lease["id"],
                "04-q1",
                ["q1"],
                "q1 shows no Rabi at any amplitude.",
                "unittest-agent",
            )

            decision = service.record_decision(
                workflow["id"],
                "04-q1",
                "advance",
                "q1 cannot be measured here; move on to q2.",
                "03a",
                None,
                [],
                "unittest-agent",
            )

            self.assertEqual(decision["next_action"]["next_node"], "03a")
            plan = service.next_action(workflow["id"])
            self.assertEqual(plan["serial_pass"]["qubit"], "q2")

    def test_the_last_pass_completes_the_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, shared=True, node_id="05")
            for index, name in enumerate(TARGETS):
                workflow = service._workflow(workflow["id"])
                run_id = f"05-{name}"
                record_node(service, workflow, lease, "05", name, run_id)
                following = TARGETS[index + 1 :]
                service.record_decision(
                    workflow["id"],
                    run_id,
                    "advance",
                    f"{name} is done.",
                    "03a" if following else None,
                    None,
                    [],
                    "unittest-agent",
                )
                if following:
                    service.db.execute(
                        "UPDATE workflows SET current_node = '05' WHERE id = ?",
                        (workflow["id"],),
                    )

            self.assertEqual(
                service._workflow(workflow["id"])["status"], "completed"
            )

    def test_a_shared_lo_moves_for_one_qubit_and_compensates_the_rest(self) -> None:
        state = chip_state()
        wiring = chip_wiring(shared=True)

        patch, details = drive_lo_recenter_patch(
            state,
            wiring,
            ["q2"],
            100_000_000,
            force_zero_if=True,
            allow_shared_output=True,
            max_if_abs_hz=400_000_000,
        )

        by_path = {item["path"]: item["value"] for item in patch}
        # q2 sits at LO 3.0 GHz + 20 MHz, so the LO moves there and q2's IF is
        # zeroed; q1 and q3 keep their RF by absorbing the 20 MHz move.
        self.assertEqual(by_path["/ports/mw_outputs/con1/8/3/upconverter_frequency"], 3_020_000_000.0)
        self.assertEqual(by_path["/qubits/q2/xy/intermediate_frequency"], 0.0)
        self.assertEqual(by_path["/qubits/q1/xy/intermediate_frequency"], -20_000_000.0)
        self.assertEqual(by_path["/qubits/q3/xy/intermediate_frequency"], 20_000_000.0)
        self.assertEqual(details["q2"]["shared_with"], ["q1", "q3"])

    def test_a_shared_lo_will_not_move_for_two_qubits_at_once(self) -> None:
        with self.assertRaises(StateError) as caught:
            drive_lo_recenter_patch(
                chip_state(),
                chip_wiring(shared=True),
                ["q1", "q2"],
                100_000_000,
                force_zero_if=True,
                allow_shared_output=True,
            )
        self.assertIn("one qubit at a time", str(caught.exception))

    def test_a_private_output_still_refuses_a_shared_move(self) -> None:
        with self.assertRaises(StateError) as caught:
            drive_lo_recenter_patch(
                chip_state(),
                chip_wiring(shared=True),
                ["q1"],
                100_000_000,
                force_zero_if=True,
            )
        self.assertIn("requires a private output", str(caught.exception))


    def test_each_pass_gets_its_own_03a_zero_if_setup(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, shared=True, node_id="03a")

            setup = service._required_setup_for_node("03a", ["q1"], workflow["id"])
            self.assertIn(
                "jy_request_initial_03a_zero_if",
                [item["tool"] for item in setup],
            )

            record_node(
                service, workflow, lease, "03a", "q1", "03a-q1", passing=False
            )
            workflow = service._workflow(workflow["id"])
            # q1 has had its coarse search, so its own pass must not repeat the
            # setup...
            self.assertTrue(service._initial_03a_setup_taken(workflow))
            with self.assertRaises(Exception):
                service.request_initial_03a_zero_if(workflow["id"], "unittest-agent")

            service.mark_scientifically_unmeasurable(
                workflow["id"],
                lease["id"],
                "03a-q1",
                ["q1"],
                "q1 shows no qubit line anywhere in the band.",
                "unittest-agent",
            )
            service.record_decision(
                workflow["id"],
                "03a-q1",
                "advance",
                "q1 cannot be found; hand the tail to q2.",
                "03a",
                None,
                [],
                "unittest-agent",
            )
            workflow = service._workflow(workflow["id"])

            # ... but q2's pass starts from scratch and needs it again.
            self.assertEqual(service._serial_pass_state(workflow)["current"], "q2")
            self.assertFalse(service._initial_03a_setup_taken(workflow))
            setup = service._required_setup_for_node("03a", ["q2"], workflow["id"])
            self.assertIn(
                "jy_request_initial_03a_zero_if",
                [item["tool"] for item in setup],
            )


class DriveLoBandWindowTests(unittest.TestCase):
    """An LO near a band edge is a hardware error waiting to happen."""

    def window(self, folder: str) -> tuple:
        settings = make_settings(Path(folder), chip_state())
        atomic_write_json(settings.wiring_path, chip_wiring(shared=True))
        return AgentService(settings)

    def test_the_window_is_the_band_minus_its_margin(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = self.window(folder)

            self.assertEqual(
                service.policy.mw_band_lo_window(1), (150_000_000.0, 5_400_000_000.0)
            )
            self.assertEqual(
                service.policy.mw_band_lo_window(2),
                (4_600_000_000.0, 7_400_000_000.0),
            )
            self.assertIsNone(service.policy.mw_band_lo_window(None))
            self.assertIsNone(service.policy.mw_band_lo_window(9))

    def test_a_commit_outside_the_band_window_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = self.window(folder)
            patch = [
                {
                    "op": "replace",
                    "path": "/ports/mw_outputs/con1/8/3/upconverter_frequency",
                    "value": 5_450_000_000,
                }
            ]

            with self.assertRaises(PolicyError) as caught:
                service.policy.validate_state_patch(patch, chip_state())
            self.assertIn("outside the usable part of band 1", str(caught.exception))

    def test_an_rf_preserving_move_out_of_band_is_refused(self) -> None:
        with self.assertRaises(StateError) as caught:
            drive_lo_recenter_patch(
                chip_state(),
                chip_wiring(shared=True),
                ["q1"],
                100_000_000,
                target_rf_hz={"q1": 5_900_000_000.0},
                allow_shared_output=True,
                band_window=lambda band: (150_000_000.0, 5_400_000_000.0),
            )
        self.assertIn("outside the usable part of its band", str(caught.exception))

    def test_a_coarse_window_shift_is_clamped_to_the_band(self) -> None:
        patch, details = drive_lo_recenter_patch(
            chip_state(),
            chip_wiring(shared=True),
            ["q1"],
            100_000_000,
            target_lo_hz={"q1": 5_900_000_000.0},
            allow_shared_output=True,
            band_window=lambda band: (150_000_000.0, 5_400_000_000.0),
        )

        by_path = {item["path"]: item["value"] for item in patch}
        self.assertEqual(
            by_path["/ports/mw_outputs/con1/8/3/upconverter_frequency"],
            5_400_000_000.0,
        )
        self.assertTrue(details["q1"]["clamped_to_band"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
