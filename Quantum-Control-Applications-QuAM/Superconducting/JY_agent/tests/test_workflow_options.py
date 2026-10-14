"""Optional nodes and the measurement mode are chosen on the lease page.

Operator request 2026-10-04. The autonomy-lease approval page lists the
optional nodes (02 and the statistics nodes) as checkboxes and asks whether the
nodes from 03a on measure one qubit at a time or multiplexed. The wiring decides
the recommendation -- a shared XY output recommends one at a time -- and
automatic measurement starts only after the operator has chosen.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from jy_agent.approval_web import _lease_options_fields, _lease_options_from_fields
from jy_agent.service import AgentService, ServiceError
from jy_agent.util import atomic_write_json

from test_autonomy import AUTO_PHRASE
from test_core import make_settings, start_test_workflow
from test_shared_xy_drive import TARGETS, chip_state, chip_wiring

SEQUENCE = (
    "02", "02x", "02c", "02a", "03a", "04", "05", "07d", "07b", "06", "06b",
    "10a", "05st", "06st_t2star", "06st_t2e",
)


def build(folder: str, shared: bool) -> tuple[AgentService, dict, dict]:
    settings = replace(make_settings(Path(folder), chip_state()), workflow_sequence=SEQUENCE)
    atomic_write_json(settings.wiring_path, chip_wiring(shared))
    service = AgentService(settings)
    workflow = start_test_workflow(service, list(TARGETS))
    proposal = service.request_autonomy_lease(
        workflow["id"],
        "unittest-agent",
        AUTO_PHRASE,
        "Run the bounded commissioning workflow while the operator is away.",
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


class WorkflowOptionTests(unittest.TestCase):
    def test_offer_lists_optional_nodes_and_recommends_per_qubit_on_shared_drive(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _service, _workflow, proposal = build(folder, shared=True)
            offer = proposal["payload"]["workflow_options_offer"]
            self.assertEqual(
                [item["node"] for item in offer["optional_nodes"]],
                ["02", "05st", "06st_t2star", "06st_t2e"],
            )
            self.assertTrue(all(item["default"] for item in offer["optional_nodes"]))
            self.assertEqual(offer["drive_mode"]["recommended"], "per_qubit")
            self.assertEqual(
                offer["drive_mode"]["shared_xy_drive_groups"],
                {"#/ports/mw_outputs/con1/8/3": sorted(TARGETS)},
            )
            self.assertIsNone(offer["drive_mode"]["locked"])

    def test_separate_outputs_recommend_multiplex(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _service, _workflow, proposal = build(folder, shared=False)
            offer = proposal["payload"]["workflow_options_offer"]
            self.assertEqual(offer["drive_mode"]["recommended"], "multiplex")

    def test_browser_approval_requires_the_choices(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, proposal = build(folder, shared=True)
            with self.assertRaisesRegex(ServiceError, "measurement mode"):
                approve(service, proposal, None)
            with self.assertRaisesRegex(ServiceError, "one at a time or"):
                approve(service, proposal, {"optional_nodes": ["02"], "drive_mode": None})
            self.assertEqual(service.proposal(proposal["id"])["status"], "pending")

    def test_unticked_02_moves_the_workflow_to_02x_and_out_of_the_lease(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal = build(folder, shared=True)
            approve(
                service,
                proposal,
                {"optional_nodes": ["05st", "06st_t2star", "06st_t2e"], "drive_mode": "per_qubit"},
            )
            current = service._workflow(workflow["id"])
            self.assertEqual(current["current_node"], "02x")
            options = service._workflow_options(current)
            self.assertEqual(options["excluded_nodes"], ["02"])
            self.assertEqual(options["drive_mode"], "per_qubit")
            lease = service.autonomy_status(lease_id=proposal["autonomy_lease_id"])
            self.assertEqual(lease["status"], "active")
            self.assertNotIn("02", lease["allowed_nodes"])
            self.assertEqual(lease["allowed_nodes"][0], "02x")

    def test_unticked_statistics_node_is_skipped_on_advance(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal = build(folder, shared=False)
            approve(
                service,
                proposal,
                {"optional_nodes": ["02", "06st_t2star", "06st_t2e"], "drive_mode": "multiplex"},
            )
            current = service._workflow(workflow["id"])
            self.assertNotIn("05st", service._workflow_sequence(current))
            # This test is about ticks; the 2026-10-12 pass gate (statistics
            # only after a passed 06/06b) is covered in test_per_qubit_pass_rules.
            service.policy.shared_xy_drive["statistics_sources"] = {}
            self.assertEqual(service._expected_next_node(current, "10a"), "06st_t2star")
            lease = service.autonomy_status(lease_id=proposal["autonomy_lease_id"])
            self.assertNotIn("05st", lease["allowed_nodes"])

    def test_drive_mode_overrides_the_wiring_default(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            # Operator instruction 2026-10-08: a shared XY output can no longer
            # be multiplexed.
            service, workflow, proposal = build(folder, shared=True)
            offer = proposal["payload"]["workflow_options_offer"]
            self.assertEqual(offer["drive_mode"]["blocked"], ["multiplex"])
            with self.assertRaisesRegex(ServiceError, "multiplexed measurement is disabled"):
                approve(service, proposal, {"optional_nodes": [], "drive_mode": "multiplex"})
            self.assertEqual(service.proposal(proposal["id"])["status"], "pending")
            # A workflow stored as multiplex before the rule is still capped.
            service.db.execute(
                "UPDATE workflows SET options_json = ? WHERE id = ?",
                ('{"drive_mode": "multiplex"}', workflow["id"]),
            )
            current = service._workflow(workflow["id"])
            self.assertEqual(service._serialize_from_node(current), "03a")
            self.assertEqual(service.policy.shared_xy_drive_cap("04", "multiplex"), 1)
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal = build(folder, shared=False)
            approve(service, proposal, {"optional_nodes": [], "drive_mode": "per_qubit"})
            current = service._workflow(workflow["id"])
            self.assertEqual(service._serialize_from_node(current), "03a")
            self.assertEqual(service._shared_xy_drive_cap("04"), 1)
            self.assertIsNone(service.policy.shared_xy_drive_cap("04"))

    def test_rejects_a_node_that_is_not_optional(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, proposal = build(folder, shared=True)
            with self.assertRaisesRegex(ServiceError, "not optional"):
                approve(service, proposal, {"optional_nodes": ["04"], "drive_mode": "per_qubit"})

    def test_approval_form_renders_and_parses_the_choices(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, proposal = build(folder, shared=True)
            html = _lease_options_fields(proposal, "zh-Hant")
            for node in ("02", "05st", "06st_t2star", "06st_t2e"):
                self.assertIn(f'name="optional_node" value="{node}" checked', html)
            # 2026-10-08: a shared XY output pre-selects per_qubit and
            # disables multiplex.
            self.assertIn('name="drive_mode" value="per_qubit" required checked>', html)
            self.assertIn('name="drive_mode" value="multiplex" required disabled>', html)
            self.assertIn("con1/8/3", html)
            self.assertIn("（建議）", html)
            parsed = _lease_options_from_fields(
                {"optional_node": ["05st"], "drive_mode": ["per_qubit"]}, proposal
            )
            self.assertEqual(
                parsed,
                {
                    "optional_nodes": ["05st"],
                    "drive_mode": "per_qubit",
                    "flux_calibration": None,
                    "node_parameters": {},
                    "start_node": None,
                    "qubit_start_nodes": {},
                },
            )
            self.assertIsNone(_lease_options_from_fields({}, {"kind": "run", "payload": {}}))


    def test_ticked_02_shows_and_requires_res_num_and_design_frequencies(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, proposal = build(folder, shared=True)
            html = _lease_options_fields(proposal, "en")
            self.assertIn('name="node02_res_num"', html)
            self.assertIn('name="node02_res_design_freq"', html)
            ticked = {"optional_node": ["02"], "drive_mode": ["per_qubit"]}
            with self.assertRaisesRegex(ServiceError, "res_num and res_design_freq"):
                _lease_options_from_fields(ticked, proposal)
            with self.assertRaisesRegex(ServiceError, "2 entries but res_num is 3"):
                _lease_options_from_fields(
                    {**ticked, "node02_res_num": ["3"], "node02_res_design_freq": ["5.9, 6.0"]},
                    proposal,
                )
            parsed = _lease_options_from_fields(
                {**ticked, "node02_res_num": ["2"], "node02_res_design_freq": ["5.9，6.05"]},
                proposal,
            )
            self.assertEqual(
                parsed["node_parameters"],
                {"02": {"res_num": 2, "res_design_freq": [5.9, 6.05]}},
            )

    def test_02_parameters_are_stored_prefilled_and_used_by_every_02_run(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal = build(folder, shared=True)
            values = {"res_num": 2, "res_design_freq": [5.9, 6.05]}
            approve(
                service,
                proposal,
                {
                    "optional_nodes": ["02"],
                    "drive_mode": "per_qubit",
                    "node_parameters": {"02": values},
                },
            )
            current = service._workflow(workflow["id"])
            self.assertEqual(service._workflow_options(current)["node_parameters"], {"02": values})
            offer = service._lease_option_offer(current)
            self.assertEqual(offer["optional_nodes"][0]["parameters"], values)
            self.assertIn('value="5.9, 6.05"', _lease_options_fields(
                {"kind": "autonomy_lease", "payload": {"workflow_options_offer": offer}}, "en"
            ))
            self.assertEqual(service.next_action(workflow["id"])["operator_parameters"], values)

    def test_02_parameters_must_match_the_policy(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, proposal = build(folder, shared=True)
            with self.assertRaisesRegex(ServiceError, "entries but res_num"):
                approve(
                    service,
                    proposal,
                    {
                        "optional_nodes": ["02"],
                        "drive_mode": "per_qubit",
                        "node_parameters": {"02": {"res_num": 3, "res_design_freq": [5.9]}},
                    },
                )
            with self.assertRaisesRegex(ServiceError, "no lease-page parameters"):
                approve(
                    service,
                    proposal,
                    {
                        "optional_nodes": ["05st"],
                        "drive_mode": "per_qubit",
                        "node_parameters": {"02": {"res_num": 1, "res_design_freq": [5.9]}},
                    },
                )
            self.assertEqual(service.proposal(proposal["id"])["status"], "pending")

class ForcedBroadbandTests(unittest.TestCase):
    """2026-10-07: zero or duplicate readout IFs force 02 instead of refusing."""

    def build_unset(self, folder: str) -> tuple[AgentService, dict, dict]:
        state = chip_state()
        for name in TARGETS:
            state["qubits"][name]["resonator"]["intermediate_frequency"] = 0
        settings = replace(make_settings(Path(folder), state), workflow_sequence=SEQUENCE)
        atomic_write_json(settings.wiring_path, chip_wiring(False))
        service = AgentService(settings)
        workflow = start_test_workflow(service, list(TARGETS))
        proposal = service.request_autonomy_lease(
            workflow["id"], "unittest-agent", AUTO_PHRASE, "Bring up a new chip."
        )
        return service, workflow, proposal

    def test_entry_is_allowed_and_02_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, proposal = self.build_unset(folder)
            self.assertEqual(
                service.policy.readout_frequencies_unset(list(TARGETS)), sorted(TARGETS)
            )
            item = proposal["payload"]["workflow_options_offer"]["optional_nodes"][0]
            self.assertEqual(item["node"], "02")
            self.assertTrue(item["required"])
            html = _lease_options_fields(proposal, "en")
            self.assertIn('<input type="hidden" name="optional_node" value="02">', html)
            self.assertIn("Required: readout IFs", html)
            with self.assertRaisesRegex(ServiceError, "02 is required"):
                approve(service, proposal, {"optional_nodes": [], "drive_mode": "multiplex"})
            self.assertEqual(service._workflow(workflow["id"])["current_node"], "02")

    def test_over_budget_line_is_scaled_by_a_required_setup(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state = chip_state()
            for name in TARGETS:
                readout = state["qubits"][name]["resonator"]["operations"]["readout"]
                readout["amplitude"] = 0.3
            settings = replace(make_settings(Path(folder), state), workflow_sequence=SEQUENCE)
            atomic_write_json(settings.wiring_path, chip_wiring(False))
            service = AgentService(settings)
            # Entry is allowed although the line sums to 0.9 > 0.5.
            workflow = start_test_workflow(service, list(TARGETS))
            patch, details = service.policy.readout_amplitude_budget_patch(list(TARGETS))
            self.assertEqual(len(patch), len(TARGETS))
            line = next(iter(details.values()))
            self.assertAlmostEqual(sum(line["qubits"].values()), 0.45)
            setup = service._required_setup_for_node("02", list(TARGETS), workflow["id"])
            self.assertIn(
                "jy_request_readout_amplitude_budget", [item["tool"] for item in setup]
            )
            proposal = service.request_readout_amplitude_budget(workflow["id"], "unittest")
            self.assertEqual(proposal["payload"]["patch"], patch)

    def test_readout_lo_recentre_covers_design_and_reopens_02(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _proposal = self.build_unset(folder)
            state = service.policy.raw  # noqa: F841 - keep the engine loaded
            from jy_agent.state import load_state

            active = load_state(service.settings.active_state)
            active["ports"]["mw_outputs"]["con1"]["8"]["1"] = {
                "band": 2,
                "upconverter_frequency": 5_000_000_000,
            }
            active["ports"]["mw_inputs"] = {
                "con1": {"8": {"2": {"band": 2, "downconverter_frequency": 5_000_000_000}}}
            }
            atomic_write_json(service.settings.active_state, active)
            designs = [5.6, 5.7, 5.8][: len(TARGETS)]
            service.db.execute(
                "UPDATE workflows SET options_json = ? WHERE id = ?",
                (
                    json.dumps(
                        {"node_parameters": {"02": {"res_num": len(designs), "res_design_freq": designs}}}
                    ),
                    workflow["id"],
                ),
            )
            current = service._workflow(workflow["id"])
            plan = service._readout_lo_recenter_plan(current)
            line = plan["details"]["#/ports/mw_outputs/con1/8/1"]
            new_lo = line["new_lo_hz"]
            self.assertEqual(new_lo % 100_000_000, 0)
            for frequency in line["design_frequencies_hz"].values():
                # Inside +-max IF and outside the LO guard, with margin.
                self.assertLess(abs(frequency - new_lo), 400e6 - 30e6)
                self.assertGreater(abs(frequency - new_lo), 10e6 + 30e6)
            values = {item["path"]: item["value"] for item in plan["patch"]}
            self.assertEqual(
                values["/ports/mw_inputs/con1/8/2/downconverter_frequency"], new_lo
            )
            self.assertIn(
                "jy_request_readout_lo_recenter",
                [item["tool"] for item in service._required_setup_for_node("02", list(TARGETS), workflow["id"])],
            )
            service.policy.validate_state_patch(plan["patch"], active)
            bad = [dict(item) for item in plan["patch"]]
            for item in bad:
                if item["path"].endswith("downconverter_frequency"):
                    item["value"] = 5_600_000_000
            from jy_agent.policy import PolicyError

            with self.assertRaisesRegex(PolicyError, "must equal the readout output LO written"):
                service.policy.validate_state_patch(bad, active)

    def test_only_02_skips_the_spacing_check(self) -> None:
        from jy_agent.policy import PolicyError

        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, _proposal = self.build_unset(folder)
            targets = sorted(TARGETS)
            service.policy.validate_multiplex_targets(
                targets, allow_unset_readout=True
            )
            with self.assertRaisesRegex(PolicyError, "MHz apart"):
                service.policy.validate_multiplex_targets(targets)


if __name__ == "__main__":
    unittest.main()
