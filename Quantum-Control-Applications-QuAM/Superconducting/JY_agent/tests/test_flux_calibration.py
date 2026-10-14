"""Flux calibration (02b, 03b, 06a) chosen on the lease page.

Operator request 2026-10-08. The lease approval page asks whether to calibrate
flux. Yes runs 02b after 02a and 03b after 03a, and allows the optional 06a
after 06. Any Z output shared by two qubits disables yes. 02b sorts each qubit
into tunable / flat / avoided crossing / failed: only tunable qubits commit and
go on to 03b/06a, flat and crossing qubits skip them, and no flux node ever
removes a qubit from the rest of the sequence.
"""

from __future__ import annotations

import tempfile
import unittest
import uuid
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from jy_agent.analysis_flux import (
    AVOIDED_CROSSING,
    FAILED,
    FLAT,
    TUNABLE,
    analyze_03b_qubit,
    analyze_06a_qubit,
    classify_02b_trace,
    flux_dataset_metrics,
    flux_fit_quality,
    render_02b_plots,
)
from jy_agent.approval_web import _lease_options_fields, _lease_options_from_fields
from jy_agent.guide_web import NODES, guide_asset_path, guide_section
from jy_agent.policy import PolicyError
from jy_agent.service import AgentService, ServiceError
from jy_agent.state import shared_z_line_groups, z_offset
from jy_agent.util import atomic_write_json, json_dumps, utc_now

from test_autonomy import AUTO_PHRASE
from test_core import AGENT_ROOT, make_settings, start_test_workflow
from test_shared_xy_drive import TARGETS, chip_state, chip_wiring, coarse_search_document

SEQUENCE = (
    "02x", "02c", "02a", "02b", "03a", "03b", "04", "05", "07d", "07b", "06",
    "06a", "06b", "10a",
)
RULES_02B = {
    "min_dip_snr": 4.0,
    "flat_max_swing_steps": 3.0,
    "flat_max_span_steps": 4.0,
    "min_fit_r_squared": 0.8,
    "crossing_outlier_sigma": 5.0,
    "crossing_min_steps": 4.0,
    "crossing_min_fraction_of_swing": 0.25,
    "crossing_min_points": 2,
    "min_edge_fraction": 0.05,
}


def flux_state() -> dict:
    state = chip_state()
    for name in TARGETS:
        qubit = state["qubits"][name]
        qubit["z"] = {
            "independent_offset": 0.1,
            "joint_offset": "#./independent_offset",
            "min_offset": "#./independent_offset",
            "operations": {"const": {"amplitude": 0.5}},
        }
        qubit["resonator"]["depletion_time"] = 2000
        qubit["resonator"]["operations"]["readout"]["length"] = 2000
        qubit["xy"]["operations"]["saturation"]["length"] = 20000
    return state


def flux_wiring(shared_z: bool, shared_xy: bool = False) -> dict:
    wiring = chip_wiring(shared_xy)
    for index, name in enumerate(TARGETS, start=1):
        port = 4 if shared_z and name in {"q1", "q2"} else index
        wiring["wiring"]["qubits"][name]["z"] = {
            "opx_output": f"#/ports/analog_outputs/con1/1/{port}"
        }
    return wiring


def build(folder: str, shared_z: bool = False) -> tuple[AgentService, dict, dict]:
    settings = replace(make_settings(Path(folder), flux_state()), workflow_sequence=SEQUENCE)
    atomic_write_json(settings.wiring_path, flux_wiring(shared_z))
    service = AgentService(settings)
    workflow = start_test_workflow(service, list(TARGETS))
    proposal = service.request_autonomy_lease(
        workflow["id"], "unittest-agent", AUTO_PHRASE, "Bring up with flux calibration."
    )
    return service, workflow, proposal


def approve(service: AgentService, proposal: dict, options: dict | None) -> dict:
    return service.approve_from_browser(
        proposal["id"],
        f"APPROVE {proposal['id']}",
        service.approval_csrf_token(proposal["id"]),
        "127.0.0.1",
        lease_options=options,
    )


def insert_run(
    service: AgentService,
    workflow: dict,
    proposal: dict,
    node_id: str,
    qubits: list[str],
    analysis: dict,
    *,
    advance: bool = False,
) -> str:
    run_id = uuid.uuid4().hex
    service.db.execute(
        "INSERT INTO runs(id, workflow_id, proposal_id, node_id, parameters_json, "
        "status, analysis_status, analysis_json) "
        "VALUES (?, ?, ?, ?, ?, 'completed', ?, ?)",
        (
            run_id,
            workflow["id"],
            proposal["id"],
            node_id,
            json_dumps({"qubits": qubits}),
            analysis.get("analysis_status", "pass"),
            json_dumps(analysis),
        ),
    )
    if advance:
        service.db.execute(
            "INSERT INTO decisions(id, workflow_id, run_id, decision, reason, "
            "next_node, next_parameters_json, state_patch_json, client_id, created_at) "
            "VALUES (?, ?, ?, 'advance', 'test', NULL, '{}', '[]', 'unittest', ?)",
            (uuid.uuid4().hex, workflow["id"], run_id, utc_now()),
        )
    return run_id


def analysis_02b(classes: dict[str, str]) -> dict:
    results = {
        name: {
            "classification": value,
            "resolved": value != FAILED,
            "fit_successful": value == TUNABLE,
            "flux_tunable": value == TUNABLE,
        }
        for name, value in classes.items()
    }
    return {
        "analysis_status": "pass",
        "node_id": "02b",
        "passing_targets": sorted(name for name, value in classes.items() if value != FAILED),
        "fit_quality": {"results": results},
        "dataset_metrics": {"qubits": {name: {} for name in classes}},
    }


def resonator_map(
    flux: np.ndarray,
    frequency: np.ndarray,
    centre: np.ndarray,
    seed: int = 1,
    depth: float = 0.8,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    width = 0.25e6
    rows = [
        1.0 - depth / (1.0 + ((frequency - f0) / width) ** 2)
        for f0 in centre
    ]
    return np.asarray(rows) + rng.normal(0, 0.01, (flux.size, frequency.size))


class FluxWiringTests(unittest.TestCase):
    def test_only_outputs_shared_by_two_qubits_count(self) -> None:
        self.assertEqual(shared_z_line_groups(flux_wiring(False)), {})
        self.assertEqual(
            shared_z_line_groups(flux_wiring(True)),
            {"#/ports/analog_outputs/con1/1/4": ["q1", "q2"]},
        )

    def test_z_offset_follows_quam_references(self) -> None:
        qubit = flux_state()["qubits"]["q1"]
        self.assertEqual(z_offset(qubit, "joint_offset"), 0.1)
        qubit["z"]["joint_offset"] = -0.2
        self.assertEqual(z_offset(qubit, "joint_offset"), -0.2)


class FluxLeaseOptionTests(unittest.TestCase):
    def test_flux_nodes_stay_out_until_the_page_says_yes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal = build(folder)
            self.assertNotIn("02b", service._workflow_sequence(workflow))
            offer = proposal["payload"]["workflow_options_offer"]["flux_calibration"]
            self.assertEqual(offer["nodes"], ["02b", "03b"])
            self.assertEqual(offer["optional"], ["06a"])
            self.assertFalse(offer["blocked"])
            # The lease keeps the flux nodes until the choice is made.
            self.assertIn("02b", proposal["payload"]["allowed_nodes"])

    def test_yes_runs_02b_03b_and_the_ticked_06a(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal = build(folder)
            approve(
                service,
                proposal,
                {"optional_nodes": ["06a"], "drive_mode": "multiplex", "flux_calibration": True},
            )
            current = service._workflow(workflow["id"])
            sequence = service._workflow_sequence(current)
            for node in ("02b", "03b", "06a"):
                self.assertIn(node, sequence)
            self.assertLess(sequence.index("02a"), sequence.index("02b"))
            self.assertLess(sequence.index("03a"), sequence.index("03b"))
            self.assertLess(sequence.index("06"), sequence.index("06a"))
            lease = service.autonomy_status(lease_id=proposal["autonomy_lease_id"])
            self.assertIn("03b", lease["allowed_nodes"])
            self.assertIs(service._workflow_options(current)["flux_calibration"], True)

    def test_yes_without_06a_skips_it(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal = build(folder)
            approve(
                service,
                proposal,
                {"optional_nodes": [], "drive_mode": "multiplex", "flux_calibration": True},
            )
            sequence = service._workflow_sequence(service._workflow(workflow["id"]))
            self.assertIn("03b", sequence)
            self.assertNotIn("06a", sequence)

    def test_no_leaves_every_flux_node_out(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal = build(folder)
            approve(
                service,
                proposal,
                {"optional_nodes": [], "drive_mode": "multiplex", "flux_calibration": False},
            )
            current = service._workflow(workflow["id"])
            self.assertFalse(
                {"02b", "03b", "06a"} & set(service._workflow_sequence(current))
            )
            lease = service.autonomy_status(lease_id=proposal["autonomy_lease_id"])
            self.assertNotIn("02b", lease["allowed_nodes"])
            self.assertEqual(service._expected_next_node(current, "02a"), "03a")

    def test_the_choice_is_required_and_06a_needs_yes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, proposal = build(folder)
            with self.assertRaisesRegex(ServiceError, "calibrate flux"):
                approve(service, proposal, {"optional_nodes": [], "drive_mode": "multiplex"})
            with self.assertRaisesRegex(ServiceError, "need flux calibration"):
                approve(
                    service,
                    proposal,
                    {"optional_nodes": ["06a"], "drive_mode": "multiplex", "flux_calibration": False},
                )
            self.assertEqual(service.proposal(proposal["id"])["status"], "pending")

    def test_a_shared_z_output_disables_yes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, proposal = build(folder, shared_z=True)
            offer = proposal["payload"]["workflow_options_offer"]
            self.assertTrue(offer["flux_calibration"]["blocked"])
            html = _lease_options_fields(proposal, "zh-Hant")
            self.assertIn('name="flux_calibration" value="yes" required disabled>', html)
            self.assertIn('name="flux_calibration" value="no" required checked>', html)
            self.assertIn('value="06a" disabled>', html)
            self.assertIn("1/4", html)
            with self.assertRaisesRegex(ServiceError, "shares Z"):
                approve(
                    service,
                    proposal,
                    {"optional_nodes": [], "drive_mode": "multiplex", "flux_calibration": True},
                )
            approve(
                service,
                proposal,
                {"optional_nodes": [], "drive_mode": "multiplex", "flux_calibration": False},
            )

    def test_terminal_approval_defaults_to_no(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal = build(folder)
            service._approve_pending_proposal(proposal["id"], "unit-test-human", "unit_test")
            current = service._workflow(workflow["id"])
            self.assertIs(service._workflow_options(current)["flux_calibration"], False)
            self.assertNotIn("06a", service._workflow_sequence(current))

    def test_form_renders_explanation_and_parses_the_choice(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _service, _workflow, proposal = build(folder)
            html = _lease_options_fields(proposal, "zh-Hant")
            self.assertIn("02b, 03b", html)
            self.assertIn('name="flux_calibration" value="yes" required>', html)
            self.assertIn('name="optional_node" value="06a" checked>', html)
            parsed = _lease_options_from_fields(
                {"drive_mode": ["multiplex"], "flux_calibration": ["yes"], "optional_node": ["06a"]},
                proposal,
            )
            self.assertIs(parsed["flux_calibration"], True)
            with self.assertRaisesRegex(ServiceError, "yes or no"):
                _lease_options_from_fields({"flux_calibration": ["maybe"]}, proposal)


class FluxPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.service, _workflow, _proposal = build(self.folder.name)
        self.policy = self.service.policy

    def tearDown(self) -> None:
        self.folder.cleanup()

    def test_defaults_pass(self) -> None:
        for node in ("02b", "03b", "06a"):
            self.policy.validate_run(node, {"qubits": ["q1"]})

    def test_operator_sweep_limits(self) -> None:
        with self.assertRaisesRegex(PolicyError, "0.9 V"):
            self.policy.validate_run("02b", {"qubits": ["q1"], "min_flux_offset_in_v": -1.0})
        with self.assertRaisesRegex(PolicyError, "flux points"):
            self.policy.validate_run("02b", {"qubits": ["q1"], "num_flux_points": 101})
        with self.assertRaisesRegex(PolicyError, "0.4 V"):
            self.policy.validate_run("03b", {"qubits": ["q1"], "max_flux_offset_in_v": 0.5})
        with self.assertRaisesRegex(PolicyError, "0.1 V"):
            self.policy.validate_run("06a", {"qubits": ["q1"], "flux_span": 0.3})
        with self.assertRaisesRegex(PolicyError, "about 100"):
            self.policy.validate_run("06a", {"qubits": ["q1"], "flux_step": 0.0005})

    def test_amplitude_scale_and_runtime(self) -> None:
        state = flux_state()
        state["qubits"]["q1"]["z"]["operations"]["const"]["amplitude"] = 0.1
        atomic_write_json(self.service.settings.active_state, state)
        with self.assertRaisesRegex(PolicyError, "amplitude_scale"):
            self.policy.validate_run(
                "03b",
                {"qubits": ["q1"], "min_flux_offset_in_v": -0.4, "max_flux_offset_in_v": 0.4},
            )
        with self.assertRaisesRegex(PolicyError, "within 10 min"):
            self.policy.validate_run("02b", {"qubits": ["q1"], "num_averages": 50000})
        estimate = self.policy.estimate_run_seconds(
            "02b",
            {**self.policy.nodes["02b"]["defaults"], "qubits": ["q1", "q2"]},
            flux_state(),
        )
        self.assertGreater(estimate, 0)
        self.assertLess(estimate, 600)

    def test_state_patch_offsets_are_bounded(self) -> None:
        good = [{"op": "replace", "path": "/qubits/q1/z/joint_offset", "value": 0.2}]
        self.policy.validate_state_patch(good)
        with self.assertRaisesRegex(PolicyError, "0.9 V"):
            self.policy.validate_state_patch(
                [{"op": "replace", "path": "/qubits/q1/z/joint_offset", "value": 1.5}]
            )
        with self.assertRaisesRegex(PolicyError, "not allowlisted"):
            self.policy.validate_state_patch(
                [{"op": "replace", "path": "/qubits/q1/z/operations/aSWAP/amplitude", "value": 0.1}]
            )


class SharedZPolicyTests(unittest.TestCase):
    def test_flux_nodes_are_refused_on_a_shared_z_chip(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, _proposal = build(folder, shared_z=True)
            with self.assertRaisesRegex(PolicyError, "shares Z lines"):
                service.policy.validate_run("02b", {"qubits": ["q3"]})


class FluxAnalysisTests(unittest.TestCase):
    flux = np.linspace(-0.9, 0.9, 75)
    frequency = np.arange(-3.75e6, 3.75e6, 0.1e6)

    def tunable_centre(self) -> np.ndarray:
        return 1.0e6 * np.cos(2 * np.pi * self.flux / 0.8)

    def test_tunable_resonator(self) -> None:
        magnitude = resonator_map(self.flux, self.frequency, self.tunable_centre())
        result = classify_02b_trace(
            self.flux, self.frequency, magnitude, RULES_02B, {"offset": 0.0, "dv_phi0": 0.8}
        )
        self.assertEqual(result["classification"], TUNABLE, result.get("reason"))
        self.assertAlmostEqual(result["period_v"], 0.8, delta=0.05)

    def test_idle_offset_off_the_extremum_fails(self) -> None:
        magnitude = resonator_map(self.flux, self.frequency, self.tunable_centre())
        result = classify_02b_trace(
            self.flux, self.frequency, magnitude, RULES_02B, {"offset": 0.2, "dv_phi0": 0.8}
        )
        self.assertEqual(result["classification"], FAILED)
        self.assertIn("extremum", result["reason"])

    def test_flat_resonator(self) -> None:
        magnitude = resonator_map(self.flux, self.frequency, np.full(self.flux.size, 0.3e6))
        result = classify_02b_trace(
            self.flux, self.frequency, magnitude, RULES_02B, {"offset": 0.4, "dv_phi0": 0.1}
        )
        self.assertEqual(result["classification"], FLAT, result.get("reason"))

    def test_avoided_crossing(self) -> None:
        centre = self.tunable_centre()
        centre[40:43] += 2.2e6
        centre[43:45] -= 2.2e6
        magnitude = resonator_map(self.flux, self.frequency, centre)
        result = classify_02b_trace(
            self.flux, self.frequency, magnitude, RULES_02B, {"offset": 0.0, "dv_phi0": 0.8}
        )
        self.assertEqual(result["classification"], AVOIDED_CROSSING, result.get("reason"))
        self.assertTrue(result["crossings"])

    def test_outliers_at_sweep_start_are_not_a_crossing(self) -> None:
        # Run #334 (2026-10-08): q8-q10 left the cosine only at the first
        # 2-3 flux points, a settling transient rather than a crossing.
        centre = self.tunable_centre()
        centre[0:3] -= 2.2e6
        magnitude = resonator_map(self.flux, self.frequency, centre)
        result = classify_02b_trace(
            self.flux, self.frequency, magnitude, RULES_02B, {"offset": 0.0, "dv_phi0": 0.8}
        )
        self.assertEqual(result["classification"], TUNABLE, result.get("reason"))
        self.assertFalse(result["crossings"])
        self.assertTrue(result["edge_outliers"])

    def test_no_dip_fails(self) -> None:
        magnitude = resonator_map(
            self.flux, self.frequency, self.tunable_centre(), depth=0.0
        )
        result = classify_02b_trace(
            self.flux, self.frequency, magnitude, RULES_02B, {"offset": 0.0, "dv_phi0": 0.8}
        )
        self.assertEqual(result["classification"], FAILED)

    def test_broad_dip_on_sloped_background_is_visible(self) -> None:
        # Real 02b rows (2026-10-08): a ~1 MHz-wide dip on a sloped baseline
        # fills the 7.5 MHz window, so the row's own spread is not noise.
        rng = np.random.default_rng(3)
        slope = np.linspace(-0.4, 0.4, self.frequency.size)
        rows = [
            1.0 + slope - 0.8 / (1.0 + ((self.frequency - f0) / 1.0e6) ** 2)
            for f0 in self.tunable_centre()
        ]
        magnitude = np.asarray(rows) + rng.normal(0, 0.01, (self.flux.size, self.frequency.size))
        result = classify_02b_trace(
            self.flux, self.frequency, magnitude, RULES_02B, {"offset": 0.0, "dv_phi0": 0.8}
        )
        self.assertGreater(result["dip_snr"], RULES_02B["min_dip_snr"])
        self.assertNotIn("not visible", result.get("reason") or "")

    def test_dataset_metrics_fit_quality_and_crossing_plot(self) -> None:
        import xarray as xr

        centres = {
            "q1": self.tunable_centre(),
            "q2": np.full(self.flux.size, 0.3e6),
            "q3": self.tunable_centre(),
        }
        centres["q3"][40:43] += 2.2e6
        centres["q3"][43:45] -= 2.2e6
        data = np.stack(
            [resonator_map(self.flux, self.frequency, centres[name], seed=i) for i, name in enumerate(TARGETS)]
        )
        dataset = xr.Dataset(
            {"IQ_abs": (("qubit", "flux", "freq"), data)},
            coords={"qubit": TARGETS, "flux": self.flux, "freq": self.frequency},
        )
        fits = {name: {"offset": 0.0, "dv_phi0": 0.8} for name in TARGETS}
        metrics = flux_dataset_metrics("02b", dataset, {"fit_results": fits}, RULES_02B)
        classes = {name: metrics["qubits"][name]["classification"] for name in TARGETS}
        self.assertEqual(classes, {"q1": TUNABLE, "q2": FLAT, "q3": AVOIDED_CROSSING})
        results, failures = flux_fit_quality("02b", metrics, list(TARGETS))
        self.assertEqual(failures, [])
        self.assertTrue(all(item["resolved"] for item in results.values()))
        self.assertEqual([n for n, r in results.items() if r["flux_tunable"]], ["q1"])
        with tempfile.TemporaryDirectory() as folder:
            plots = render_02b_plots(Path(folder), "#1_02b", metrics)
            self.assertEqual(len(plots), 1)
            self.assertIn("-q3-avoided-crossing", plots[0])
            self.assertTrue(Path(plots[0]).is_file())

    def test_03b_parabola(self) -> None:
        flux = np.linspace(-0.05, 0.05, 51)
        frequency = np.arange(-50e6, 50e6, 1e6)
        rng = np.random.default_rng(3)
        line = -2e10 * (flux - 0.01) ** 2 + 5e6
        signal = np.asarray(
            [np.exp(-(((frequency - f0) / 2e6) ** 2)) for f0 in line]
        ) + rng.normal(0, 0.02, (flux.size, frequency.size))
        good = {"flux_shift": 0.01, "quad_term": -2e10, "drive_freq": 5e6}
        rules = {"min_peak_snr": 4.0, "min_visible_fraction": 0.5, "min_fit_r_squared": 0.7, "min_edge_fraction": 0.05}
        result = analyze_03b_qubit(flux, frequency, signal, rules, good)
        self.assertTrue(result["fit_successful"], result.get("reason"))
        bad = analyze_03b_qubit(flux, frequency, signal, rules, {**good, "quad_term": 2e10})
        self.assertFalse(bad["fit_successful"])
        edge = analyze_03b_qubit(flux, frequency, signal, rules, {**good, "flux_shift": 0.049})
        self.assertFalse(edge["fit_successful"])
        elsewhere = analyze_03b_qubit(flux, frequency, signal, rules, {**good, "flux_shift": -0.03})
        self.assertFalse(elsewhere["fit_successful"])
        self.assertIn("disagrees", elsewhere["reason"])

    def test_03b_wide_window_tracks_the_line_not_the_noise(self) -> None:
        # Run #383: a 300 MHz window with single-sample noise spikes and a
        # weaker second line made the unsmoothed argmax miss the parabola.
        flux = np.linspace(-0.2, 0.2, 81)
        frequency = np.arange(-150e6, 150e6, 1e6)
        rng = np.random.default_rng(5)
        line = -8e9 * (flux - 0.05) ** 2 + 20e6
        signal = np.asarray(
            [
                np.exp(-(((frequency - f0) / 4e6) ** 2))
                + 0.4 * np.exp(-(((frequency - f0 + 18e6) / 3e6) ** 2))
                for f0 in line
            ]
        )
        signal = signal + rng.normal(0, 0.12, signal.shape)
        spikes = rng.integers(0, frequency.size, flux.size)
        signal[np.arange(flux.size), spikes] += 0.9
        visible = np.abs(line) < 150e6
        signal[~visible] = rng.normal(0, 0.12, (int((~visible).sum()), frequency.size))
        node = {"flux_shift": 0.05, "quad_term": -8e9, "drive_freq": 20e6}
        rules = {"min_peak_snr": 4.0, "min_visible_fraction": 0.5, "min_fit_r_squared": 0.7, "min_edge_fraction": 0.05}
        result = analyze_03b_qubit(flux, frequency, signal, rules, node)
        self.assertTrue(result["fit_successful"], result.get("reason"))
        self.assertAlmostEqual(result["vertex_v"], 0.05, delta=0.01)

    def test_06a_offset_inside_the_window(self) -> None:
        flux = np.arange(-0.04, 0.041, 0.001)
        times = np.arange(16, 516, 8) / 1000.0
        fringes = 4.0 + 2000 * (flux - 0.005) ** 2
        state = np.asarray(
            [0.5 + 0.4 * np.cos(2 * np.pi * f * times) * np.exp(-times / 5) for f in fringes]
        )
        rules = {"max_offset_fraction_of_span": 0.4, "min_fit_r_squared": 0.5, "min_valid_flux_fraction": 0.5}
        good = analyze_06a_qubit(
            flux, times, state, rules, {"flux_offset": 0.005, "freq_offset": 1e5}, 0.08
        )
        self.assertTrue(good["fit_successful"], good.get("reason"))
        bad = analyze_06a_qubit(
            flux, times, state, rules, {"flux_offset": 0.039, "freq_offset": 1e5}, 0.08
        )
        self.assertFalse(bad["fit_successful"])


class FluxSnapshotAnalysisTests(unittest.TestCase):
    def test_02b_commits_only_the_tunable_qubit_and_flags_the_crossing(self) -> None:
        import json

        import xarray as xr

        from jy_agent.analysis import SnapshotAnalyzer
        from jy_agent.util import sha256_file

        flux = FluxAnalysisTests.flux
        frequency = FluxAnalysisTests.frequency
        tunable = 1.0e6 * np.cos(2 * np.pi * flux / 0.8)
        crossing = tunable.copy()
        crossing[40:43] += 2.2e6
        crossing[43:45] -= 2.2e6
        centres = {"q1": tunable, "q2": np.full(flux.size, 0.3e6), "q3": crossing}
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, _proposal = build(folder)
            snapshot = Path(folder) / "#1_02b_Resonator_Spectroscopy_vs_Flux_000000"
            snapshot.mkdir()
            data = np.stack(
                [resonator_map(flux, frequency, centres[name], seed=i) for i, name in enumerate(TARGETS)]
            )
            xr.Dataset(
                {"IQ_abs": (("qubit", "flux", "freq"), data)},
                coords={"qubit": TARGETS, "flux": flux, "freq": frequency},
            ).to_netcdf(snapshot / "ds.h5", engine="scipy")
            (snapshot / "node.json").write_text(
                json.dumps({"id": 1, "data": {"outcomes": {n: "successful" for n in TARGETS}}}),
                encoding="utf-8",
            )
            fits = {name: {"offset": 0.0, "dv_phi0": 0.8} for name in TARGETS}
            (snapshot / "data.json").write_text(
                json.dumps({"fit_results": fits, "initial_parameters": {"qubits": TARGETS}}),
                encoding="utf-8",
            )
            (snapshot / "figure.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            current = json.loads(service.settings.active_state.read_text(encoding="utf-8"))
            atomic_write_json(snapshot / "quam_state.json", current)
            run_id = uuid.uuid4().hex
            updates = {}
            for name in TARGETS:
                old_if = current["qubits"][name]["resonator"]["intermediate_frequency"]
                updates.update(
                    {
                        f"#/qubits/{name}/z/joint_offset": {"key": f"#/qubits/{name}/z/joint_offset", "new": 0.05},
                        f"#/qubits/{name}/phi0_voltage": {"key": f"#/qubits/{name}/phi0_voltage", "new": 0.8},
                        f"#/qubits/{name}/resonator/intermediate_frequency": {
                            "key": f"#/qubits/{name}/resonator/intermediate_frequency",
                            "new": old_if + 100_000.0,
                        },
                        f"#/qubits/{name}/extras/dressed_resonator_freq": {
                            "key": f"#/qubits/{name}/extras/dressed_resonator_freq",
                            "new": 6.0e9,
                        },
                        f"#/qubits/{name}/extras/FluxPeriodV": {
                            "key": f"#/qubits/{name}/extras/FluxPeriodV", "new": 0.8
                        },
                    }
                )
            requests = service.settings.runtime / "requests"
            requests.mkdir(parents=True, exist_ok=True)
            atomic_write_json(requests / f"{run_id}.state_updates.json", updates)
            analysis = SnapshotAnalyzer(service.settings, service.policy).analyze_snapshot(
                snapshot,
                "02b",
                {
                    "id": run_id,
                    "node_id": "02b",
                    "parameters_json": json.dumps({"qubits": TARGETS}),
                    "active_state_hash_before": sha256_file(service.settings.active_state),
                },
            )
            self.assertEqual(analysis["analysis_status"], "pass", analysis["failure_reasons"])
            patch = {item["path"]: item["value"] for item in analysis["candidate_state_patch"]}
            self.assertEqual({path.split("/")[2] for path in patch}, {"q1"})
            self.assertEqual(patch["/qubits/q1/z/joint_offset"], 0.05)
            # The recorded RF read the pre-update IF; JY adds the IF move.
            self.assertEqual(patch["/qubits/q1/extras/dressed_resonator_freq"], 6.0e9 + 100_000.0)
            self.assertNotIn("/qubits/q1/extras/FluxPeriodV", patch)
            self.assertEqual([n["qubit"] for n in analysis["operator_notices"]], ["q3"])
            self.assertTrue(any("q3-avoided-crossing" in plot for plot in analysis["plots"]))
            classes = {
                name: result["classification"]
                for name, result in analysis["fit_quality"]["results"].items()
            }
            self.assertEqual(classes, {"q1": TUNABLE, "q2": FLAT, "q3": AVOIDED_CROSSING})


class FluxWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.service, workflow, self.proposal = build(self.folder.name)
        approve(
            self.service,
            self.proposal,
            {"optional_nodes": ["06a"], "drive_mode": "multiplex", "flux_calibration": True},
        )
        self.workflow = self.service._workflow(workflow["id"])

    def tearDown(self) -> None:
        self.folder.cleanup()

    def test_only_tunable_qubits_reach_03b_and_nobody_is_dropped_downstream(self) -> None:
        insert_run(
            self.service,
            self.workflow,
            self.proposal,
            "02b",
            list(TARGETS),
            analysis_02b({"q1": TUNABLE, "q2": FLAT, "q3": AVOIDED_CROSSING}),
            advance=True,
        )
        self.assertEqual(self.service._node_target_resolution(self.workflow["id"], "02b"), set(TARGETS))
        self.assertEqual(self.service._active_targets_for_node(self.workflow, "03b"), {"q1"})
        self.assertEqual(self.service._active_targets_for_node(self.workflow, "04"), set(TARGETS))
        # 06a waits for an accepted 03b.
        self.assertEqual(self.service._active_targets_for_node(self.workflow, "06a"), set())

    def test_failed_02b_qubit_still_runs_the_rest(self) -> None:
        insert_run(
            self.service,
            self.workflow,
            self.proposal,
            "02b",
            list(TARGETS),
            analysis_02b({"q1": TUNABLE, "q2": FAILED, "q3": TUNABLE}),
            advance=True,
        )
        self.assertEqual(self.service._active_targets_for_node(self.workflow, "03a"), set(TARGETS))
        self.assertEqual(self.service._active_targets_for_node(self.workflow, "03b"), {"q1", "q3"})

    def test_03b_is_passed_over_when_no_qubit_tunes(self) -> None:
        insert_run(
            self.service,
            self.workflow,
            self.proposal,
            "02b",
            list(TARGETS),
            analysis_02b({"q1": FLAT, "q2": FLAT, "q3": AVOIDED_CROSSING}),
            advance=True,
        )
        insert_run(
            self.service, self.workflow, self.proposal, "03a", ["q1"], coarse_search_document("q1")
        )
        self.assertEqual(self.service._expected_next_node(self.workflow, "03a"), "04")
        self.assertEqual(self.service._expected_next_node(self.workflow, "06"), "06b")

    def test_03b_follows_03a_for_a_tunable_resolved_qubit(self) -> None:
        insert_run(
            self.service,
            self.workflow,
            self.proposal,
            "02b",
            list(TARGETS),
            analysis_02b({"q1": TUNABLE, "q2": FLAT, "q3": FLAT}),
            advance=True,
        )
        insert_run(
            self.service, self.workflow, self.proposal, "03a", ["q1"], coarse_search_document("q1")
        )
        self.assertEqual(self.service._expected_next_node(self.workflow, "03a"), "03b")


class GuideTests(unittest.TestCase):
    def test_every_example_image_exists_and_paths_are_confined(self) -> None:
        html = guide_section("zh-Hant")
        for node, *_rest, source in NODES:
            self.assertIn(f'id="node-{node}"', html)
            if source:
                self.assertIsNotNone(guide_asset_path(AGENT_ROOT, f"{node}.png"), node)
        self.assertIn("02b", html)
        self.assertIsNone(guide_asset_path(AGENT_ROOT, "../config/agent.yaml"))
        self.assertIsNone(guide_asset_path(AGENT_ROOT, "missing.png"))
        self.assertIn("How to operate", guide_section("en"))

    def test_guide_routes(self) -> None:
        from starlette.testclient import TestClient

        from jy_agent.approval_server import create_approval_app
        from test_core import sample_state

        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            service = AgentService(settings)
            entered = service.enter_measurement_mode("unittest-agent", "進入 JY 量測模式", ["q1"])
            session_id = entered["dashboard"]["id"]
            with TestClient(create_approval_app(settings), client=("127.0.0.1", 50123)) as client:
                page = client.get(f"/session/{session_id}/guide")
                self.assertEqual(page.status_code, 200)
                self.assertIn("/guide/assets/04.png", page.text)
                self.assertIn('data-session-events=""', page.text)
                home = client.get(f"/session/{session_id}/home")
                self.assertIn(f"/session/{session_id}/guide", home.text)
                image = client.get("/guide/assets/04.png")
                self.assertEqual(image.status_code, 200)
                self.assertEqual(image.headers["content-type"], "image/png")
                self.assertEqual(client.get("/guide/assets/nope.png").status_code, 404)
                self.assertEqual(client.get("/guide").status_code, 200)


if __name__ == "__main__":
    unittest.main()
