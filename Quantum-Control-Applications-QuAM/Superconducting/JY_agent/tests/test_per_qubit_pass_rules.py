"""Per-qubit pass rules, operator instruction 2026-10-12.

* Only 03a (no frequency) and 04 (no pi pulse) end a qubit's pass early; any
  other node hands the qubit on. 05 runs after 07b (the 05 -> 07d/07b -> 05
  deferral is retired).
* A qubit that never ran 03b and shows no Rabi at all returns from 04 to 03a,
  at most three times; the rejected line is masked for it.
* A fine 03a line commits only when an earlier 03a run showed it too.
* Each qubit records its drive LO in extras.xy_lo_freq; one LO is shared when
  every measured qubit fits, otherwise the qubit uses its own and the rest are
  parked at the IF limit.
"""

from __future__ import annotations

import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from jy_agent.analysis import SnapshotAnalyzer
from jy_agent.service import AgentService, AutonomyScopeError, ServiceError
from jy_agent.state import drive_lo_recenter_patch
from jy_agent.util import atomic_write_json, json_dumps, json_loads
from jy_agent.xy_lo import xy_lo_plan_patch

from test_autonomy import activate_lease
from test_core import make_settings, start_test_workflow
from test_next_action_planner import analysis_document
from test_shared_xy_drive import (
    SHARED_XY,
    TARGETS,
    chip_state,
    chip_wiring,
    t1_analysis_document,
)

SEQUENCE = ("02x", "02c", "02a", "03a", "04", "07d", "07b", "05", "06")


def build(folder: str, node_id: str, state: dict | None = None):
    settings = replace(
        make_settings(Path(folder), state or chip_state()),
        workflow_sequence=SEQUENCE,
    )
    atomic_write_json(settings.wiring_path, chip_wiring(shared=True))
    service = AgentService(settings)
    workflow = start_test_workflow(service, list(TARGETS))
    service.db.execute(
        "UPDATE workflows SET current_node = ? WHERE id = ?", (node_id, workflow["id"])
    )
    _proposal, lease = activate_lease(service, workflow["id"])
    return service, service._workflow(workflow["id"]), lease


def insert_run(
    service: AgentService,
    workflow_id: str,
    node_id: str,
    qubit: str,
    analysis: dict,
    *,
    num_averages: int | None = None,
    lease_id: str | None = None,
    decision: str | None = None,
    next_node: str | None = None,
) -> str:
    proposal_id = uuid.uuid4().hex
    run_id = uuid.uuid4().hex
    service.db.execute(
        "INSERT INTO proposals(id, workflow_id, kind, payload_json, status, created_at, "
        "expires_at, max_uses, uses, source_client) VALUES (?, ?, 'run', '{}', "
        "'approved', '2026-10-12T00:00:00+00:00', '2026-10-12T01:00:00+00:00', 1, 1, 'test')",
        (proposal_id, workflow_id),
    )
    parameters = {"qubits": [qubit]}
    if num_averages is not None:
        parameters["num_averages"] = num_averages
    service.db.execute(
        "INSERT INTO runs(id, workflow_id, proposal_id, node_id, parameters_json, status, "
        "analysis_status, analysis_json, autonomy_lease_id) "
        "VALUES (?, ?, ?, ?, ?, 'completed', ?, ?, ?)",
        (
            run_id,
            workflow_id,
            proposal_id,
            node_id,
            json_dumps(parameters),
            analysis.get("analysis_status", "needs_review"),
            json_dumps(analysis),
            lease_id,
        ),
    )
    if decision is not None:
        service.db.execute(
            "INSERT INTO decisions(id, workflow_id, run_id, decision, reason, next_node, "
            "next_parameters_json, state_patch_json, client_id, created_at) VALUES "
            "(?, ?, ?, ?, 'test', ?, '{}', '[]', 'test', '2026-10-12T00:00:00+00:00')",
            (uuid.uuid4().hex, workflow_id, run_id, decision, next_node),
        )
    return run_id


def fine_03a(qubit: str, frequency: float, snr: float = 999.0) -> dict:
    document = analysis_document("03a", {qubit: snr})
    document["03a_stage"] = "fine"
    document["fit_quality"]["results"][qubit] = {
        "fit_successful": True,
        "drive_freq": frequency,
    }
    document["dataset_metrics"]["qubits"][qubit]["feature_fwhm_hz"] = 2_000_000.0
    return document


def advance(service: AgentService, workflow_id: str, run_id: str, next_node):
    return service.record_decision(
        workflow_id, run_id, "advance", "test advance", next_node, None, [], "unittest"
    )


class PassContinuationTests(unittest.TestCase):
    def test_05_runs_after_07b_and_then_06(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "03a")
            wid = workflow["id"]
            insert_run(service, wid, "03a", "q1", fine_03a("q1", 3.02e9), decision="advance", next_node="04")
            insert_run(service, wid, "04", "q1", analysis_document("04", {"q1": 999.0}), decision="advance", next_node="07d")
            service.db.execute("UPDATE workflows SET current_node = '07d' WHERE id = ?", (wid,))
            run_07d = insert_run(service, wid, "07d", "q1", analysis_document("07d", {"q1": 999.0}), lease_id=lease["id"])
            advance(service, wid, run_07d, "07b")
            run_07b = insert_run(service, wid, "07b", "q1", analysis_document("07b", {"q1": 999.0}), lease_id=lease["id"])
            workflow = service._workflow(wid)
            self.assertEqual(service._expected_next_node(workflow, "07b"), "05")
            advance(service, wid, run_07b, "05")
            plan = service.next_action(wid)
            self.assertEqual(plan["current_node"], "05")
            self.assertEqual(plan["serial_pass"]["qubit"], "q1")
            run_05 = insert_run(service, wid, "05", "q1", t1_analysis_document("q1"), lease_id=lease["id"])
            workflow = service._workflow(wid)
            self.assertEqual(service._expected_next_node(workflow, "05"), "06")
            advance(service, wid, run_05, "06")
            self.assertEqual(service._workflow(wid)["current_node"], "06")

    def test_a_decided_07b_advances_to_05_by_requesting_it(self) -> None:
        # 2026-10-12: a `repeat` on the single 07b run leaves nothing to record
        # `advance` on; requesting the pass's next node (05) is the advance.
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "03a")
            wid = workflow["id"]
            insert_run(service, wid, "03a", "q1", fine_03a("q1", 3.02e9), decision="advance", next_node="04")
            insert_run(service, wid, "04", "q1", analysis_document("04", {"q1": 999.0}), decision="advance", next_node="07d")
            insert_run(service, wid, "07d", "q1", analysis_document("07d", {"q1": 999.0}), decision="advance", next_node="07b")
            service.db.execute("UPDATE workflows SET current_node = '07b' WHERE id = ?", (wid,))
            insert_run(service, wid, "07b", "q1", analysis_document("07b", {"q1": 2.0}), lease_id=lease["id"], decision="repeat", next_node="07b")
            workflow = service._workflow(wid)
            self.assertTrue(service._node_is_finished_and_fully_decided(workflow, "05"))
            self.assertFalse(service._node_is_finished_and_fully_decided(workflow, "06"))

    def test_a_failed_05_still_continues_to_06(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "03a")
            wid = workflow["id"]
            insert_run(service, wid, "03a", "q1", fine_03a("q1", 3.02e9), decision="advance", next_node="04")
            insert_run(service, wid, "04", "q1", analysis_document("04", {"q1": 999.0}), decision="advance", next_node="07d")
            insert_run(service, wid, "07d", "q1", analysis_document("07d", {"q1": 999.0}), decision="advance", next_node="07b")
            insert_run(service, wid, "07b", "q1", analysis_document("07b", {"q1": 999.0}), decision="advance", next_node="05")
            service.db.execute("UPDATE workflows SET current_node = '05' WHERE id = ?", (wid,))
            run_05 = insert_run(service, wid, "05", "q1", analysis_document("05", {"q1": 9.0}), lease_id=lease["id"])
            # No deferral any more: a T1 below the floor needs a mark first.
            with self.assertRaisesRegex(ServiceError, "missing"):
                advance(service, wid, run_05, "06")
            service.mark_scientifically_unmeasurable(
                wid, lease["id"], run_05, ["q1"], "T1 SNR stays low.", "unittest"
            )
            self.assertEqual(advance(service, wid, run_05, "06")["next_action"]["next_node"], "06")

    def test_03a_without_a_line_still_ends_the_pass(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "03a")
            wid = workflow["id"]
            run = insert_run(service, wid, "03a", "q1", analysis_document("03a", {"q1": 2.0}), lease_id=lease["id"])
            service.mark_scientifically_unmeasurable(wid, lease["id"], run, ["q1"], "No line.", "unittest")
            advance(service, wid, run, "03a")
            plan = service.next_action(wid)
            self.assertEqual(plan["serial_pass"]["qubit"], "q2")
            self.assertEqual(plan["serial_pass"]["finished"], ["q1"])


def full_drive(document: dict, qubit: str = "q1", amplitude: float = 0.52) -> dict:
    """A 04 run whose sweep reached the waveform limit (0.52 x 1.9 ~ 0.99)."""
    document["dataset_metrics"]["qubits"][qubit]["operation_amplitude_before_run"] = amplitude
    return document


class ReturnTo03aTests(unittest.TestCase):
    def at_04(self, folder: str, gate_ns: int = 400):
        # 2026-10-12: "no Rabi" counts only at the longest gate (400 ns shared since 2026-10-14).
        state = chip_state()
        for qubit in state["qubits"].values():
            qubit["xy"]["operations"]["x180_DragCosine"]["length"] = gate_ns
        service, workflow, lease = build(folder, "03a", state)
        wid = workflow["id"]
        insert_run(service, wid, "03a", "q1", fine_03a("q1", 2.51e9), decision="advance", next_node="04")
        service.db.execute("UPDATE workflows SET current_node = '04' WHERE id = ?", (wid,))
        return service, wid, lease

    def test_no_rabi_at_4000_averages_returns_to_03a_and_masks_the_line(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, wid, lease = self.at_04(folder)
            run = insert_run(service, wid, "04", "q1", full_drive(analysis_document("04", {"q1": 2.5})), num_averages=4000, lease_id=lease["id"])
            workflow = service._workflow(wid)
            self.assertTrue(service._04_return_check(workflow, "q1")[0])
            self.assertEqual(service._expected_next_node(workflow, "04"), "03a")
            # No incomplete mark is needed to leave 04 this way.
            advance(service, wid, run, "03a")

            plan = service.next_action(wid)
            self.assertEqual(plan["serial_pass"]["qubit"], "q1")
            self.assertEqual(plan["serial_pass"]["finished"], [])
            self.assertNotIn("q1", service._node_target_resolution(wid, "03a"))
            self.assertIn(
                "jy_request_initial_03a_zero_if",
                [item["tool"] for item in plan["required_setup"]],
            )
            lines = service._rejected_03a_lines(wid)["q1"]
            self.assertEqual(len(lines), 1)
            low, high = lines[0]
            self.assertLessEqual(low, 2.51e9 - 5e6)
            self.assertGreaterEqual(high, 2.51e9 + 5e6)

    def test_a_noisy_but_visible_rabi_does_not_return(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, wid, lease = self.at_04(folder)
            insert_run(service, wid, "04", "q1", analysis_document("04", {"q1": 5.5}), num_averages=4000, lease_id=lease["id"])
            allowed, why = service._04_return_check(service._workflow(wid), "q1")
            self.assertFalse(allowed)
            self.assertIn("no 04 run", why)
            # Without a return the pass of q1 ends at 04.
            self.assertEqual(service._expected_next_node(service._workflow(wid), "04"), "03a")
            self.assertEqual(service._pass_next_node(service._workflow(wid), "04", "q1")[1], "end")

    def test_a_qubit_that_ran_03b_does_not_return(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, wid, lease = self.at_04(folder)
            insert_run(service, wid, "03b", "q1", analysis_document("03b", {"q1": 999.0}))
            insert_run(service, wid, "04", "q1", analysis_document("04", {"q1": 2.0}), num_averages=4000, lease_id=lease["id"])
            allowed, why = service._04_return_check(service._workflow(wid), "q1")
            self.assertFalse(allowed)
            self.assertIn("03b", why)

    def test_at_most_three_returns(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, wid, lease = self.at_04(folder)
            for _ in range(3):
                service.db.event(
                    "serial_pass_03a_return", "test",
                    {"qubit": "q1", "run_id": "x", "after_run_rowid": 0,
                     "rejected_rf_hz": 2.4e9, "half_width_hz": 5e6},
                    wid,
                )
            insert_run(service, wid, "04", "q1", analysis_document("04", {"q1": 2.0}), num_averages=4000, lease_id=lease["id"])
            allowed, why = service._04_return_check(service._workflow(wid), "q1")
            self.assertFalse(allowed)
            self.assertIn("3 times", why)

    def test_earlier_04_marks_are_void_after_a_return(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, wid, lease = self.at_04(folder)
            run = insert_run(service, wid, "04", "q1", full_drive(analysis_document("04", {"q1": 2.0})), num_averages=4000, lease_id=lease["id"])
            service.mark_scientifically_unmeasurable(wid, lease["id"], run, ["q1"], "flat", "unittest")
            self.assertIn("q1", service._scientifically_unmeasurable_targets_by_node(wid).get("04", set()))
            advance(service, wid, run, "03a")
            self.assertNotIn("q1", service._scientifically_unmeasurable_targets_by_node(wid).get("04", set()))


    def test_an_underdriven_or_short_gate_04_does_not_return(self) -> None:
        # q3 2026-10-12: a 32 ns gate swept to 4% of full scale is no evidence.
        with tempfile.TemporaryDirectory() as folder:
            service, wid, lease = self.at_04(folder)
            insert_run(service, wid, "04", "q1", full_drive(analysis_document("04", {"q1": 2.0}), amplitude=0.06), num_averages=4000, lease_id=lease["id"])
            allowed, why = service._04_return_check(service._workflow(wid), "q1")
            self.assertFalse(allowed)
            self.assertIn("full drive", why)
        with tempfile.TemporaryDirectory() as folder:
            service, wid, lease = self.at_04(folder, gate_ns=32)
            insert_run(service, wid, "04", "q1", full_drive(analysis_document("04", {"q1": 2.0})), num_averages=4000, lease_id=lease["id"])
            allowed, why = service._04_return_check(service._workflow(wid), "q1")
            self.assertFalse(allowed)
            self.assertIn("jy_request_04_gate_time", why)


class Reproducible03aLineTests(unittest.TestCase):
    def test_a_fine_line_needs_an_earlier_sighting(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "03a")
            wid = workflow["id"]
            fine = fine_03a("q1", 2.5025e9)
            fine["analysis_status"] = "pass"
            fine["passing_targets"] = ["q1"]
            fine["candidate_state_patch"] = [
                {"op": "replace", "path": "/qubits/q1/extras/idle_freq", "value": 2.5025e9}
            ]
            run_id = insert_run(service, wid, "03a", "q1", fine, lease_id=lease["id"])
            run = service.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
            gated = service._require_reproduced_03a_line(run, json_loads(json_dumps(fine), {}))
            self.assertEqual(gated["passing_targets"], [])
            self.assertEqual(gated["candidate_state_patch"], [])
            self.assertEqual(gated["analysis_status"], "needs_review")

            coarse = analysis_document("03a", {"q1": 6.2})
            coarse["03a_stage"] = "coarse_candidate"
            coarse["fit_quality"]["results"]["q1"] = {"fit_successful": True, "drive_freq": 2.504e9}
            insert_run(service, wid, "03a", "q1", coarse)
            confirmed = service._require_reproduced_03a_line(run, json_loads(json_dumps(fine), {}))
            self.assertEqual(confirmed["passing_targets"], ["q1"])
            self.assertEqual(confirmed["analysis_status"], "pass")

    def test_a_far_or_weak_earlier_feature_does_not_confirm(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "03a")
            wid = workflow["id"]
            for frequency, snr in ((2.540e9, 6.0), (2.5025e9, 3.0)):
                document = analysis_document("03a", {"q1": snr})
                document["fit_quality"]["results"]["q1"] = {"fit_successful": True, "drive_freq": frequency}
                insert_run(service, wid, "03a", "q1", document)
            fine = fine_03a("q1", 2.5025e9)
            fine["passing_targets"] = ["q1"]
            run_id = insert_run(service, wid, "03a", "q1", fine)
            run = service.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
            gated = service._require_reproduced_03a_line(run, fine)
            self.assertEqual(gated["passing_targets"], [])


class RejectedLineAnalysisTests(unittest.TestCase):
    def analyzer(self, folder: str) -> SnapshotAnalyzer:
        service, _workflow, _lease = build(folder, "03a")
        return service.analyzer

    def test_a_coarse_fit_on_a_rejected_line_moves_to_the_next_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            analyzer = self.analyzer(folder)
            fit = {"results": {"q3": {"fit_successful": True, "drive_freq": 2.510e9}}}
            metrics = {"qubits": {"q3": {
                "feature_coordinate": 10e6,
                "robust_snr": 8.8,
                "peak_candidates": [
                    {"coordinate_hz": 11e6, "robust_snr": 3.9, "fwhm_hz": 6e6, "edge_fraction": 0.5, "prominence": 1e-6},
                    {"coordinate_hz": -276e6, "robust_snr": 4.5, "fwhm_hz": 6e6, "edge_fraction": 0.15, "prominence": 1e-6},
                ],
            }}}
            run = {"rejected_03a_lines": {"q3": [[2.505e9, 2.515e9]]}}
            warnings, failures = analyzer._avoid_rejected_03a_lines(fit, metrics, run, "coarse_candidate")
            self.assertEqual(failures, [])
            self.assertAlmostEqual(fit["results"]["q3"]["drive_freq"], 2.224e9)
            self.assertEqual(metrics["qubits"]["q3"]["feature_selection"], "outside_rejected_line")
            self.assertTrue(warnings)

    def test_a_fine_scan_on_a_rejected_line_fails(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            analyzer = self.analyzer(folder)
            fit = {"results": {"q3": {"fit_successful": True, "drive_freq": 2.510e9}}}
            metrics = {"qubits": {"q3": {"feature_coordinate": 0.0, "peak_candidates": []}}}
            run = {"rejected_03a_lines": {"q3": [[2.505e9, 2.515e9]]}}
            _warnings, failures = analyzer._avoid_rejected_03a_lines(fit, metrics, run, "fine")
            self.assertTrue(failures)
            self.assertFalse(fit["results"]["q3"]["fit_successful"])


class XyLoPlanTests(unittest.TestCase):
    def state(self, idle: dict[str, float], lo: float = 2.5e9) -> dict:
        state = chip_state()
        state["ports"]["mw_outputs"]["con1"]["8"]["3"]["upconverter_frequency"] = lo
        for name, frequency in idle.items():
            state["qubits"][name]["extras"] = {"idle_freq": frequency}
            state["qubits"][name]["xy"]["intermediate_frequency"] = frequency - lo
        return state

    def plan(self, state: dict, focus: str, measured: list[str]):
        return xy_lo_plan_patch(
            state, chip_wiring(shared=True), focus, measured,
            grid_hz=100e6, max_if_hz=400e6, common_max_if_hz=300e6, min_if_hz=20e6,
        )

    def test_one_lo_is_shared_when_every_measured_qubit_fits(self) -> None:
        state = self.state({"q1": 2.435e9, "q2": 2.5025e9, "q3": 2.9e9})
        patch, details = self.plan(state, "q2", ["q1", "q2"])
        self.assertEqual(details["mode"], "common")
        lo = details["lo_hz"]
        self.assertTrue(all(abs(f - lo) <= 300e6 for f in (2.435e9, 2.5025e9)))
        self.assertTrue(all(abs(f - lo) >= 20e6 for f in (2.435e9, 2.5025e9)))
        recorded = {item["path"]: item["value"] for item in patch if item["path"].endswith("xy_lo_freq")}
        self.assertEqual(set(recorded), {"/qubits/q1/extras/xy_lo_freq", "/qubits/q2/extras/xy_lo_freq"})

    def test_an_unreachable_qubit_gets_its_own_lo_and_the_rest_are_parked(self) -> None:
        state = self.state({"q1": 2.435e9, "q2": 2.5025e9, "q3": 3.4e9})
        patch, details = self.plan(state, "q3", ["q1", "q2", "q3"])
        self.assertEqual(details["mode"], "own")
        # Its own grid LO, away from the leakage: 3.3 or 3.5 GHz, not 3.4 GHz.
        self.assertIn(details["lo_hz"], (3.3e9, 3.5e9))
        by_path = {item["path"]: item["value"] for item in patch}
        self.assertIn("/qubits/q3/extras/xy_lo_freq", by_path)
        self.assertNotIn("/qubits/q1/extras/xy_lo_freq", by_path)
        self.assertEqual(sorted(details["parked"]), ["q1", "q2"])
        self.assertEqual(by_path["/qubits/q1/xy/intermediate_frequency"], -400e6)
        self.assertLessEqual(abs(details["intermediate_frequency_hz"]["q3"]), 300e6)

    def test_a_placed_lo_needs_no_patch(self) -> None:
        state = self.state({"q1": 2.435e9, "q2": 2.5025e9})
        patch, details = self.plan(state, "q2", ["q1", "q2"])
        for item in patch:
            path = item["path"]
            if path.endswith("upconverter_frequency"):
                state["ports"]["mw_outputs"]["con1"]["8"]["3"]["upconverter_frequency"] = item["value"]
            elif path.endswith("intermediate_frequency"):
                state["qubits"][path.split("/")[2]]["xy"]["intermediate_frequency"] = item["value"]
            else:
                state["qubits"][path.split("/")[2]]["extras"]["xy_lo_freq"] = item["value"]
        again, _details = self.plan(state, "q2", ["q1", "q2"])
        self.assertEqual(again, [])

    def test_the_03a_zero_if_move_parks_instead_of_refusing(self) -> None:
        state = self.state({"q1": 2.435e9, "q2": 2.5025e9, "q3": 2.9e9})
        patch, details = drive_lo_recenter_patch(
            state, chip_wiring(shared=True), ["q3"], 100e6,
            force_zero_if=True, allow_shared_output=True,
            max_if_abs_hz=400e6, park_out_of_range=True,
        )
        by_path = {item["path"]: item["value"] for item in patch}
        self.assertEqual(by_path[f"{SHARED_XY[1:]}/upconverter_frequency"], 2.9e9)
        self.assertEqual(by_path["/qubits/q1/xy/intermediate_frequency"], -400e6)
        self.assertIn("q1", details["q3"]["parked_rf_hz"])

    def test_the_pass_requires_the_lo_plan_after_03a(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state = self.state({"q1": 3.02e9, "q2": 3.1e9, "q3": 3.2e9}, lo=3.0e9)
            service, workflow, lease = build(folder, "03a", state)
            wid = workflow["id"]
            insert_run(service, wid, "03a", "q1", fine_03a("q1", 3.02e9), decision="advance", next_node="04")
            service.db.execute("UPDATE workflows SET current_node = '04' WHERE id = ?", (wid,))
            tools = [item["tool"] for item in service._required_setup_for_node("04", ["q1"], wid)]
            self.assertIn("jy_request_xy_lo_plan", tools)
            with self.assertRaisesRegex(Exception, "jy_request_xy_lo_plan"):
                service.request_run(wid, "04", {"qubits": ["q1"]}, "04", "unittest", autonomy_lease_id=lease["id"])
            proposal = service.request_xy_lo_plan(wid, "unittest", lease["id"])
            service.apply_authorized_setup_state(proposal["id"], lease["id"], "unittest")
            tools = [item["tool"] for item in service._required_setup_for_node("04", ["q1"], wid)]
            self.assertNotIn("jy_request_xy_lo_plan", tools)


class GateTimeTests(unittest.TestCase):
    def test_gate_time_needs_a_maxed_amplitude_and_respects_the_cap(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state = chip_state()
            for qubit in state["qubits"].values():
                qubit["xy"]["operations"]["x180_DragCosine"]["length"] = 40
            service, workflow, lease = build(folder, "04", state)
            wid = workflow["id"]
            with self.assertRaisesRegex(ServiceError, "no 04 run"):
                service.request_04_gate_time(wid, 200, "unittest", lease["id"])
            low = analysis_document("04", {"q1": 3.0})
            low["dataset_metrics"]["qubits"]["q1"]["operation_amplitude_before_run"] = 0.2
            insert_run(service, wid, "04", "q1", low, lease_id=lease["id"])
            with self.assertRaisesRegex(ServiceError, "raise max_amp_factor"):
                service.request_04_gate_time(wid, 200, "unittest", lease["id"])
            maxed = analysis_document("04", {"q1": 3.0})
            maxed["dataset_metrics"]["qubits"]["q1"]["operation_amplitude_before_run"] = 0.52
            insert_run(service, wid, "04", "q1", maxed, lease_id=lease["id"])
            with self.assertRaisesRegex(ServiceError, "above the 400 ns"):
                service.request_04_gate_time(wid, 404, "unittest", lease["id"])
            proposal = service.request_04_gate_time(wid, 200, "unittest", lease["id"])
            paths = [item["path"] for item in proposal["payload"]["patch"]]
            self.assertIn("/qubits/q1/xy/operations/x180_DragCosine/length", paths)


class PerQubitStartTests(unittest.TestCase):
    # 02c is left out: its resolution needs a full dressed-plateau analysis.
    SEQUENCE = ("02x", "02a", "03a", "04", "07d", "07b", "05", "06")

    def service(self, folder: str) -> AgentService:
        settings = replace(
            make_settings(Path(folder), chip_state()), workflow_sequence=self.SEQUENCE
        )
        atomic_write_json(settings.wiring_path, chip_wiring(shared=True))
        return AgentService(settings)

    def earlier(self, service: AgentService) -> str:
        from test_autonomy import AUTO_PHRASE

        entered = service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE, list(TARGETS))
        wid = entered["workflow"]["id"]
        for node in ("02x", "02a"):
            document = analysis_document(node, {name: 999.0 for name in TARGETS})
            run = insert_run(service, wid, node, "q1", document, decision="advance", next_node="x")
            service.db.execute(
                "UPDATE runs SET parameters_json = ? WHERE id = ?",
                (json_dumps({"qubits": list(TARGETS)}), run),
            )
        insert_run(service, wid, "03a", "q1", fine_03a("q1", 3.02e9), decision="advance", next_node="04")
        insert_run(service, wid, "04", "q1", analysis_document("04", {"q1": 999.0}), decision="advance", next_node="05")
        insert_run(service, wid, "05", "q1", analysis_document("05", {"q1": 9.0}), decision="advance", next_node="03a")
        insert_run(service, wid, "03a", "q2", analysis_document("03a", {"q2": 3.0}), decision="repeat", next_node="03a")
        service.db.execute(
            "UPDATE workflows SET status = 'stopped', current_node = '03a', options_json = ? "
            "WHERE id = ?",
            (json_dumps({"flux_calibration": False, "drive_mode": "per_qubit",
                         "excluded_nodes": []}), wid),
        )
        service.db.execute("UPDATE autonomy_leases SET status = 'revoked' WHERE workflow_id = ?", (wid,))
        service.db.execute("UPDATE measurement_sessions SET status = 'stopped' WHERE workflow_id = ?", (wid,))
        return wid

    def test_each_qubit_may_restart_its_pass_where_it_reached(self) -> None:
        from test_autonomy import AUTO_PHRASE

        with tempfile.TemporaryDirectory() as folder:
            service = self.service(folder)
            self.earlier(service)
            entered = service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE, list(TARGETS))
            proposal = service.proposal(entered["approval"]["id"])
            history = proposal["payload"]["workflow_options_offer"]["start_node"]
            self.assertEqual(history["serialize_from_node"], "03a")
            self.assertEqual(
                history["qubit_start_nodes"],
                {"q1": ["03a", "04", "07d", "07b", "05"], "q2": ["03a"], "q3": ["03a"]},
            )
            choice = {
                "optional_nodes": [],
                "drive_mode": "per_qubit",
                "flux_calibration": False,
                "node_parameters": {},
                "start_node": "03a",
                "qubit_start_nodes": {"q1": "05", "q2": "03a", "q3": "03a"},
            }
            with self.assertRaisesRegex(ServiceError, "cannot start at 04"):
                service.preview_lease_options(
                    proposal["id"], {**choice, "qubit_start_nodes": {"q2": "04"}}
                )
            preview = service.preview_lease_options(proposal["id"], choice)
            self.assertEqual(
                preview["qubit_starts"],
                [{"qubit": "q1", "start_node": "05"}, {"qubit": "q2", "start_node": "03a"},
                 {"qubit": "q3", "start_node": "03a"}],
            )
            service._approve_pending_proposal(
                proposal["id"], "human", "unit_test", lease_options=choice
            )
            wid = entered["workflow"]["id"]
            plan = service.next_action(wid)
            self.assertEqual(plan["current_node"], "05")
            self.assertEqual(plan["serial_pass"]["qubit"], "q1")
            self.assertEqual(plan["targets"]["active"], ["q1"])
            inherited = service.db.all(
                "SELECT node_id FROM runs WHERE workflow_id = ? AND parameters_json = ?",
                (wid, json_dumps({"qubits": ["q1"]})),
            )
            self.assertEqual(sorted(row["node_id"] for row in inherited), ["03a", "04"])
            # q2 had no inherited tail run and starts its pass at 03a.
            self.assertEqual(service._qubit_start_node(service._workflow(wid), "q2"), "03a")

    def test_the_approval_page_offers_and_parses_per_qubit_starts(self) -> None:
        from test_autonomy import AUTO_PHRASE
        from jy_agent.approval_web import _lease_options_fields, _lease_options_from_fields

        with tempfile.TemporaryDirectory() as folder:
            service = self.service(folder)
            self.earlier(service)
            entered = service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE, list(TARGETS))
            proposal = service.proposal(entered["approval"]["id"])
            html = _lease_options_fields(proposal, "zh-Hant")
            self.assertIn('name="qubit_start__q1"', html)
            self.assertIn('<option value="05">05</option>', html)
            parsed = _lease_options_from_fields(
                {
                    "drive_mode": ["per_qubit"],
                    "flux_calibration": ["no"],
                    "start_node": ["03a"],
                    "qubit_start__q1": ["05"],
                    "qubit_start__q2": ["03a"],
                },
                proposal,
            )
            self.assertEqual(parsed["qubit_start_nodes"], {"q1": "05", "q2": "03a"})
            later = _lease_options_from_fields(
                {"drive_mode": ["per_qubit"], "flux_calibration": ["no"],
                 "start_node": ["02a"], "qubit_start__q1": ["05"]},
                proposal,
            )
            self.assertEqual(later["qubit_start_nodes"], {})

    def test_per_qubit_starts_need_the_pass_start(self) -> None:
        from test_autonomy import AUTO_PHRASE

        with tempfile.TemporaryDirectory() as folder:
            service = self.service(folder)
            self.earlier(service)
            entered = service.enter_autonomy_mode("unittest-agent", AUTO_PHRASE, list(TARGETS))
            proposal = service.proposal(entered["approval"]["id"])
            with self.assertRaisesRegex(ServiceError, "apply only when"):
                service.preview_lease_options(
                    proposal["id"],
                    {
                        "optional_nodes": [],
                        "drive_mode": "per_qubit",
                        "flux_calibration": False,
                        "node_parameters": {},
                        "start_node": "02a",
                        "qubit_start_nodes": {"q1": "05"},
                    },
                )


class CrossLoCandidateTests(unittest.TestCase):
    """Operator choice 2026-10-12: a sub-floor line seen at one RF under two LOs."""

    def coarse(self, folder: str, tag: str, snr: float, lo_hz: float, if_hz: float) -> dict:
        snapshot = Path(folder) / f"snap_{tag}"
        snapshot.mkdir()
        state = chip_state()
        state["ports"]["mw_outputs"]["con1"]["8"]["3"]["upconverter_frequency"] = lo_hz
        atomic_write_json(snapshot / "quam_state.json", {**state, **chip_wiring(shared=True)})
        document = analysis_document("03a", {"q1": snr})
        document["03a_stage"] = "coarse_candidate"
        document["snapshot_path"] = str(snapshot)
        document["dataset_metrics"]["qubits"]["q1"]["feature_coordinate"] = if_hz
        return document

    def test_a_line_at_one_rf_under_two_los_is_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "03a")
            wid = workflow["id"]
            insert_run(service, wid, "03a", "q1", self.coarse(folder, "a", 5.26, 2.0e9, 226e6), decision="repeat", next_node="03a")
            insert_run(service, wid, "03a", "q1", self.coarse(folder, "b", 5.88, 2.2e9, 26e6), decision="repeat", next_node="03a")
            self.assertEqual(service._03a_cross_lo_candidates(wid), {"q1": 2.226e9})
            self.assertIn("q1", service._03a_candidate_targets(wid))

    def test_the_same_if_twice_or_a_weak_line_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "03a")
            wid = workflow["id"]
            # Same LO and IF: a fixed-IF artefact could do this.
            insert_run(service, wid, "03a", "q1", self.coarse(folder, "a", 5.5, 2.2e9, 26e6), decision="repeat", next_node="03a")
            insert_run(service, wid, "03a", "q1", self.coarse(folder, "b", 5.5, 2.2e9, 26e6), decision="repeat", next_node="03a")
            # Different LO but below the cross-LO SNR.
            insert_run(service, wid, "03a", "q1", self.coarse(folder, "c", 4.8, 2.0e9, 226e6), decision="repeat", next_node="03a")
            self.assertEqual(service._03a_cross_lo_candidates(wid), {})
            self.assertNotIn("q1", service._03a_candidate_targets(wid))


    def test_a_marked_cross_lo_fine_line_resolves_at_the_cross_lo_floor(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "03a")
            wid = workflow["id"]
            unmarked = fine_03a("q1", 2.2265e9, snr=5.5)
            insert_run(service, wid, "03a", "q1", unmarked)
            self.assertNotIn("q1", service._node_target_resolution(wid, "03a"))
            marked = fine_03a("q1", 2.2265e9, snr=5.5)
            marked["fit_quality"]["results"]["q1"]["cross_lo_confirmed"] = True
            insert_run(service, wid, "03a", "q1", marked)
            self.assertIn("q1", service._node_target_resolution(wid, "03a"))
            weak = fine_03a("q2", 2.6e9, snr=4.5)
            weak["fit_quality"]["results"]["q2"]["cross_lo_confirmed"] = True
            insert_run(service, wid, "03a", "q2", weak)
            self.assertNotIn("q2", service._node_target_resolution(wid, "03a"))

class StatisticsGateTests(unittest.TestCase):
    SEQUENCE = ("02x", "02c", "02a", "03a", "04", "07d", "07b", "05", "06", "06b",
                "10a", "05st", "06st_t2star", "06st_t2e")

    def test_statistics_run_only_after_their_single_measurement_passed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = replace(make_settings(Path(folder), chip_state()), workflow_sequence=self.SEQUENCE)
            atomic_write_json(settings.wiring_path, chip_wiring(shared=True))
            service = AgentService(settings)
            workflow = start_test_workflow(service, list(TARGETS))
            wid = workflow["id"]
            service.db.execute("UPDATE workflows SET current_node = '10a' WHERE id = ?", (wid,))
            activate_lease(service, wid)
            failed_06 = analysis_document("06", {"q1": 5.0})
            failed_06["fit_quality"]["results"]["q1"]["fit_successful"] = False
            insert_run(service, wid, "03a", "q1", fine_03a("q1", 3.02e9), decision="advance", next_node="04")
            insert_run(service, wid, "04", "q1", analysis_document("04", {"q1": 999.0}), decision="advance", next_node="05")
            insert_run(service, wid, "05", "q1", t1_analysis_document("q1"), decision="advance", next_node="07d")
            readout = analysis_document("07b", {"q1": 999.0})
            readout["fit_quality"]["results"]["q1"] = {"fit_successful": True, "readout_fidelity": 0.85}
            insert_run(service, wid, "07b", "q1", readout, decision="advance", next_node="06")
            insert_run(service, wid, "06", "q1", failed_06, decision="advance", next_node="06b")
            echo = analysis_document("06b", {"q1": 999.0})
            echo["fit_quality"]["results"]["q1"] = {
                "fit_successful": True, "coherence_seconds": 4.66e-5,
                "relative_uncertainty": 0.1, "coverage_lifetimes": 3.57,
                "samples_per_lifetime": 27.7, "r_squared": 0.882,
                "fit_at_search_boundary": False,
            }
            insert_run(service, wid, "06b", "q1", echo, decision="advance", next_node="10a")
            workflow = service._workflow(wid)
            self.assertIn("q1", service._node_target_resolution(wid, "05"))
            self.assertNotIn("q1", service._node_target_resolution(wid, "06"))
            self.assertEqual(service._expected_next_node(workflow, "10a"), "05st")
            self.assertEqual(service._expected_next_node(workflow, "05st"), "06st_t2e")
            self.assertEqual(service._active_targets_for_node(workflow, "06st_t2star"), set())
            self.assertEqual(service._active_targets_for_node(workflow, "06st_t2e"), {"q1"})

    def test_a_failed_readout_runs_statistics_on_raw_iq_only(self) -> None:
        # 2026-10-13: q1/q2 (07b 55%/60%) get statistics, discrimination off.
        with tempfile.TemporaryDirectory() as folder:
            settings = replace(make_settings(Path(folder), chip_state()), workflow_sequence=self.SEQUENCE)
            atomic_write_json(settings.wiring_path, chip_wiring(shared=True))
            service = AgentService(settings)
            workflow = start_test_workflow(service, list(TARGETS))
            wid = workflow["id"]
            service.db.execute("UPDATE workflows SET current_node = '10a' WHERE id = ?", (wid,))
            activate_lease(service, wid)
            insert_run(service, wid, "03a", "q1", fine_03a("q1", 3.02e9), decision="advance", next_node="04")
            insert_run(service, wid, "04", "q1", analysis_document("04", {"q1": 999.0}), decision="advance", next_node="05")
            insert_run(service, wid, "05", "q1", t1_analysis_document("q1"), decision="advance", next_node="07d")
            poor = analysis_document("07b", {"q1": 1.0})
            poor["fit_quality"]["results"]["q1"] = {"fit_successful": False, "readout_fidelity": 0.55}
            insert_run(service, wid, "07b", "q1", poor, decision="advance", next_node="06")
            workflow = service._workflow(wid)
            self.assertEqual(service._statistics_qualified(wid, "05st"), {"q1"})
            self.assertEqual(service._pass_next_node(workflow, "10a", "q1"), ("05st", "continue"))
            with self.assertRaisesRegex(AutonomyScopeError, "use_state_discrimination=false"):
                service._validate_node_prerequisites(
                    workflow, "05st", {"qubits": ["q1"], "use_state_discrimination": True}
                )
            try:
                service._validate_node_prerequisites(
                    workflow, "05st", {"qubits": ["q1"], "use_state_discrimination": False,
                                       "max_wait_time_in_ns": 200000}
                )
            except AutonomyScopeError as exc:
                self.assertNotIn("use_state_discrimination", str(exc))

    def test_discrimination_needs_eighty_percent_and_two_blobs_at_every_node(self) -> None:
        # Operator decision 2026-10-14: fidelity >= 0.80 AND two Gaussian blobs.
        with tempfile.TemporaryDirectory() as folder:
            settings = replace(make_settings(Path(folder), chip_state()), workflow_sequence=self.SEQUENCE)
            atomic_write_json(settings.wiring_path, chip_wiring(shared=True))
            service = AgentService(settings)
            workflow = start_test_workflow(service, list(TARGETS))
            wid = workflow["id"]
            readout = analysis_document("07b", {"q1": 1.0, "q2": 1.0})
            readout["fit_quality"]["results"] = {
                "q1": {"fit_successful": True, "readout_fidelity": 0.85, "morphology_pass": True},
                "q2": {"fit_successful": True, "readout_fidelity": 0.85, "morphology_pass": False},
            }
            insert_run(service, wid, "07b", "q1", readout, decision="advance", next_node="06")
            self.assertEqual(service._state_discrimination_qualified_targets(wid), {"q1"})
            workflow = service._workflow(wid)
            service._validate_state_discrimination(
                workflow, "06", {"qubits": ["q1"], "use_state_discrimination": True}
            )
            with self.assertRaisesRegex(AutonomyScopeError, "two Gaussian blobs"):
                service._validate_state_discrimination(
                    workflow, "06", {"qubits": ["q1", "q2"]}
                )
            service._validate_state_discrimination(
                workflow, "06", {"qubits": ["q1", "q2"], "use_state_discrimination": False}
            )

    def test_the_documented_resume_phrase_resumes_a_paused_lease(self) -> None:
        from jy_agent.service import AUTONOMY_RESUME_PHRASES
        self.assertIn("繼續 JY 自動量測", AUTONOMY_RESUME_PHRASES)

    def test_06b_accepts_r_squared_down_to_the_policy_floor(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, _lease = build(folder, "03a")
            self.assertEqual(service.policy.raw["analysis"]["06b"]["min_r_squared"], 0.8)


if __name__ == "__main__":
    unittest.main()


class ReadoutDurationStepTests(unittest.TestCase):
    """2026-10-13: 07d's duration step must be a multiple of 4 ns (run 8e3fd899)."""

    def test_a_step_that_is_not_a_multiple_of_four_is_refused_before_launch(self) -> None:
        from jy_agent.service import AutonomyScopeError

        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "07d")
            with self.assertRaisesRegex(AutonomyScopeError, "multiple of 4"):
                service._validate_node_prerequisites(
                    workflow, "07d", {"qubits": ["q1"], "max_duration_in_ns": 2000, "num_durations": 8}
                )
            service._validate_node_prerequisites(
                workflow, "07d", {"qubits": ["q1"], "max_duration_in_ns": 4000, "num_durations": 8}
            )
