from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jy_agent.service import AgentService
from jy_agent.util import atomic_write_json
from jy_agent.workflow_subgroups import (
    resolve_03a_candidate_targets,
    resolve_03a_shift_ready_targets,
    resolve_node_targets,
)
from tests.test_core import (
    ENTRY_PHRASE,
    make_settings,
    sample_multiplex_state_and_wiring,
    start_test_workflow,
)


def analysis_for(node_id: str, names: list[str]) -> dict:
    if node_id == "03a":
        results = {
            name: {"fit_successful": True, "drive_freq": 5.0e9}
            for name in names
        }
    elif node_id in {"02x", "02a"}:
        results = {name: {"RO_frequency": 6.0e9} for name in names}
    elif node_id == "05":
        results = {
            name: {
                "t1_seconds": 35e-6,
                "relative_uncertainty": 0.1,
                "r_squared": 0.95,
                "coverage_lifetimes": 4.0,
                "samples_per_lifetime": 10.0,
                "fit_at_search_boundary": False,
            }
            for name in names
        }
    else:
        results = {name: {"Pi_amplitude": 0.21} for name in names}
    return {
        "03a_stage": "fine" if node_id == "03a" else None,
        "fit_quality": {"results": results},
        "dataset_metrics": {
            "qubits": {
                name: {
                    "edge_fraction": 0.4, "robust_snr": 30.0,
                    "feature_fwhm_hz": 7_500_000.0,
                }
                for name in names
            }
        },
    }


class TestWorkflowSubgroups(unittest.TestCase):
    def test_mixed_02x_run_resolves_only_usable_qubits(self) -> None:
        analysis = analysis_for("02x", ["q1", "q2"])
        analysis["dataset_metrics"]["qubits"]["q2"]["robust_snr"] = 1.0
        analysis["dataset_metrics"]["qubits"]["q2"]["edge_fraction"] = 0.01
        resolved = resolve_node_targets(
            "02x",
            [
                {
                    "status": "completed",
                    "analysis_status": "needs_review",
                    "analysis_json": json.dumps(analysis),
                }
            ],
        )
        self.assertEqual(resolved, {"q1"})

    def test_02x_metrics_resolve_when_fit_results_are_null(self) -> None:
        analysis = analysis_for("02x", ["q1", "q2"])
        analysis["fit_quality"]["results"] = None
        resolved = resolve_node_targets(
            "02x",
            [
                {
                    "status": "completed",
                    "analysis_status": "pass",
                    "analysis_json": json.dumps(analysis),
                }
            ],
        )
        self.assertEqual(resolved, {"q1", "q2"})

    def test_mixed_03a_run_resolves_only_usable_qubits(self) -> None:
        analysis = analysis_for("03a", ["q1", "q2"])
        analysis["fit_quality"]["results"]["q2"]["fit_successful"] = False
        resolved = resolve_node_targets(
            "03a",
            [
                {
                    "status": "completed",
                    "analysis_status": "needs_review",
                    "analysis_json": json.dumps(analysis),
                }
            ],
        )
        self.assertEqual(resolved, {"q1"})

    def test_03a_coarse_peak_is_candidate_but_not_final(self) -> None:
        analysis = analysis_for("03a", ["q1"])
        analysis["03a_stage"] = "coarse_candidate"
        rows = [
            {
                "status": "completed",
                "analysis_status": "pass",
                "analysis_json": json.dumps(analysis),
            }
        ]
        self.assertEqual(resolve_03a_candidate_targets(rows), {"q1"})
        self.assertEqual(resolve_node_targets("03a", rows), set())

    def test_03a_broad_fine_peak_is_not_final(self) -> None:
        analysis = analysis_for("03a", ["q1"])
        analysis["dataset_metrics"]["qubits"]["q1"][
            "feature_fwhm_hz"
        ] = 22_000_000.0
        rows = [{
            "status": "completed",
            "analysis_status": "needs_review",
            "analysis_json": json.dumps(analysis),
        }]
        self.assertEqual(resolve_03a_candidate_targets(rows), {"q1"})
        self.assertEqual(resolve_node_targets("03a", rows), set())

    def test_03a_moderately_wide_fine_peak_can_be_final(self) -> None:
        analysis = analysis_for("03a", ["q1"])
        analysis["dataset_metrics"]["qubits"]["q1"][
            "feature_fwhm_hz"
        ] = 14_000_000.0
        rows = [{
            "status": "completed",
            "analysis_status": "pass",
            "analysis_json": json.dumps(analysis),
        }]
        self.assertEqual(resolve_node_targets("03a", rows), {"q1"})

    def test_03a_fine_peak_below_half_mhz_is_not_final(self) -> None:
        analysis = analysis_for("03a", ["q1"])
        analysis["dataset_metrics"]["qubits"]["q1"][
            "feature_fwhm_hz"
        ] = 400_000.0
        rows = [{
            "status": "completed",
            "analysis_status": "needs_review",
            "analysis_json": json.dumps(analysis),
        }]
        self.assertEqual(resolve_03a_candidate_targets(rows), {"q1"})
        self.assertEqual(resolve_node_targets("03a", rows), set())

    def test_03a_fine_peak_above_half_mhz_can_be_final(self) -> None:
        analysis = analysis_for("03a", ["q1"])
        analysis["dataset_metrics"]["qubits"]["q1"][
            "feature_fwhm_hz"
        ] = 1_250_000.0
        rows = [{
            "status": "completed",
            "analysis_status": "pass",
            "analysis_json": json.dumps(analysis),
        }]
        self.assertEqual(resolve_node_targets("03a", rows), {"q1"})

    def test_03a_candidate_requires_snr_of_six(self) -> None:
        analysis = analysis_for("03a", ["q1", "q5"])
        analysis["03a_stage"] = "coarse_candidate"
        analysis["dataset_metrics"]["qubits"]["q1"]["robust_snr"] = 5.4
        analysis["dataset_metrics"]["qubits"]["q5"]["robust_snr"] = 6.2
        rows = [
            {
                "status": "completed",
                "analysis_status": "needs_review",
                "analysis_json": json.dumps(analysis),
            }
        ]
        self.assertEqual(resolve_03a_candidate_targets(rows), {"q5"})

    def test_03a_noisy_interior_trace_must_reach_2000_before_shift(self) -> None:
        analysis = analysis_for("03a", ["q1", "q2"])
        analysis["03a_stage"] = "coarse_candidate"
        analysis["dataset_metrics"]["qubits"]["q1"]["robust_snr"] = 5.4
        analysis["dataset_metrics"]["qubits"]["q2"].update(
            {"edge_fraction": 0.01, "robust_snr": 5.4}
        )
        analysis["fit_quality"]["results"]["q2"]["fit_successful"] = False
        row = {
            "status": "completed",
            "analysis_status": "needs_review",
            "analysis_json": json.dumps(analysis),
            "parameters_json": json.dumps(
                {"qubits": ["q1", "q2"], "num_averages": 100}
            ),
        }
        self.assertEqual(resolve_03a_shift_ready_targets([row]), {"q2"})
        row["parameters_json"] = json.dumps(
            {"qubits": ["q1", "q2"], "num_averages": 2000}
        )
        self.assertEqual(
            resolve_03a_shift_ready_targets([row]),
            {"q1", "q2"},
        )

    def test_manual_review_rejects_false_candidate_and_allows_shift(self) -> None:
        candidate = analysis_for("03a", ["q1"])
        candidate["03a_stage"] = "coarse_candidate"
        noise = analysis_for("03a", ["q1"])
        noise["03a_stage"] = "refinement_candidate"
        noise["dataset_metrics"]["qubits"]["q1"]["robust_snr"] = 6.68
        rows = [
            {
                "status": "completed",
                "analysis_status": "needs_review",
                "analysis_json": json.dumps(candidate),
                "parameters_json": json.dumps(
                    {"qubits": ["q1"], "num_averages": 500}
                ),
                "decision": "repeat",
            },
            {
                "status": "completed",
                "analysis_status": "needs_review",
                "analysis_json": json.dumps(noise),
                "parameters_json": json.dumps(
                    {"qubits": ["q1"], "num_averages": 2000}
                ),
                "decision": "manual_review",
            },
        ]
        self.assertEqual(resolve_03a_candidate_targets(rows), set())
        self.assertEqual(resolve_03a_shift_ready_targets(rows), {"q1"})

        recovered = analysis_for("03a", ["q1"])
        recovered["03a_stage"] = "coarse_candidate"
        rows.append(
            {
                "status": "completed",
                "analysis_status": "pass",
                "analysis_json": json.dumps(recovered),
                "parameters_json": json.dumps(
                    {"qubits": ["q1"], "num_averages": 500}
                ),
                "decision": "repeat",
            }
        )
        self.assertEqual(resolve_03a_candidate_targets(rows), {"q1"})

    def test_03a_subgroups_accumulate_before_advance(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, all_targets = sample_multiplex_state_and_wiring()
            targets = all_targets[:3]
            for name in targets:
                state["qubits"][name]["xy"]["intermediate_frequency"] = 0.0
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            workflow = service.start_workflow(
                targets,
                {"multiplexed": True},
                "unittest",
                ENTRY_PHRASE,
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '03a' WHERE id = ?",
                (workflow["id"],),
            )

            initial = service.request_run(
                workflow["id"],
                "03a",
                {
                    "qubits": targets,
                    "frequency_span_in_mhz": 800.0,
                    "operation_amplitude_factor": 0.01,
                },
                "Initial all-target 03a coarse search.",
                "unittest",
            )
            self._insert_run(
                service,
                workflow["id"],
                initial["id"],
                "offline-03a-initial",
                "03a",
                targets,
                stage_03a="coarse_candidate",
            )

            refinement = service.request_run(
                workflow["id"],
                "03a",
                {
                    "qubits": [targets[0]],
                    "frequency_span_in_mhz": 100.0,
                    "operation_amplitude_factor": 0.04,
                },
                "Refine a coarse candidate at intermediate power.",
                "unittest",
            )
            self.assertEqual(
                refinement["payload"]["parameters"]["qubits"], [targets[0]]
            )

            first = service.request_run(
                workflow["id"],
                "03a",
                {
                    "qubits": [targets[0]],
                    "frequency_span_in_mhz": 50.0,
                    "operation_amplitude_factor": 0.02,
                },
                "Measure the first spectroscopy subgroup.",
                "unittest",
            )
            first_run = "offline-03a-first"
            self._insert_run(
                service,
                workflow["id"],
                first["id"],
                first_run,
                "03a",
                [targets[0]],
            )
            with self.assertRaisesRegex(Exception, "Cannot advance from 03a"):
                service.record_decision(
                    workflow["id"],
                    first_run,
                    "advance",
                    "Only one subgroup is resolved.",
                    "04",
                    {},
                    [],
                    "unittest",
                )

            second = service.request_run(
                workflow["id"],
                "03a",
                {
                    "qubits": targets[1:],
                    "frequency_span_in_mhz": 50.0,
                    "operation_amplitude_factor": 0.02,
                },
                "Measure the remaining spectroscopy subgroup.",
                "unittest",
            )
            second_run = "offline-03a-second"
            self._insert_run(
                service,
                workflow["id"],
                second["id"],
                second_run,
                "03a",
                targets[1:],
            )
            result = service.record_decision(
                workflow["id"],
                second_run,
                "advance",
                "All active targets now have spectroscopy evidence.",
                "04",
                {},
                [],
                "unittest",
            )
            self.assertEqual(result["next_action"]["next_node"], "04")

    def test_05_subgroup_allowed_after_full_width_submission_failure(self) -> None:
        """The waiver applies at uncapped shared-first-batch nodes.

        04 is capped at five targets, so its first batch is never the full
        active set and the waiver has nothing to waive there; 05 still
        multiplexes everything and is the right place to test it.
        """

        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            workflow = service.start_workflow(
                targets,
                {"multiplexed": True},
                "unittest",
                ENTRY_PHRASE,
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '05' WHERE id = ?",
                (workflow["id"],),
            )
            with self.assertRaisesRegex(
                Exception, "first 05 run must multiplex every active target"
            ):
                service.request_run(
                    workflow["id"],
                    "05",
                    {"qubits": targets[:2]},
                    "Subgroup is refused while full width is untried.",
                    "unittest",
                )
            # A full-width attempt that never reached hardware: too large to
            # submit before the QOP queue deadline, so no snapshot exists.
            oversized = service.request_run(
                workflow["id"],
                "05",
                {"qubits": targets},
                "Full-width 05 that cannot be submitted.",
                "unittest",
            )
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, termination_cause
                ) VALUES (?, ?, ?, '05', ?, 'failed', 'failed',
                          'instrument_unreachable')
                """,
                (
                    "offline-05-oversized",
                    workflow["id"],
                    oversized["id"],
                    json.dumps({"qubits": targets}),
                ),
            )
            proposal = service.request_run(
                workflow["id"],
                "05",
                {"qubits": targets[:2]},
                "Split 05 into a smaller multiplex subgroup.",
                "unittest",
            )
            self.assertEqual(
                sorted(proposal["payload"]["parameters"]["qubits"]),
                sorted(targets[:2]),
            )

    def test_05_subgroup_allowed_after_probe_confirmed_submission_timeout(
        self,
    ) -> None:
        """The new category must waive the rule exactly like the old one.

        A full-width attempt whose probe confirmed the instrument is healthy
        records `program_submission_timeout`; the halves still have to be
        schedulable.
        """

        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            workflow = service.start_workflow(
                targets,
                {"multiplexed": True},
                "unittest",
                ENTRY_PHRASE,
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '05' WHERE id = ?",
                (workflow["id"],),
            )
            oversized = service.request_run(
                workflow["id"],
                "05",
                {"qubits": targets},
                "Full-width 05 that cannot be submitted.",
                "unittest",
            )
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, termination_cause
                ) VALUES (?, ?, ?, '05', ?, 'failed', 'failed',
                          'program_submission_timeout')
                """,
                (
                    "offline-05-submission-timeout",
                    workflow["id"],
                    oversized["id"],
                    json.dumps({"qubits": targets}),
                ),
            )
            half = targets[: max(1, len(targets) // 2)]
            proposal = service.request_run(
                workflow["id"],
                "05",
                {"qubits": half},
                "Halve the multiplex group after a submission timeout.",
                "unittest",
            )
            self.assertEqual(
                sorted(proposal["payload"]["parameters"]["qubits"]),
                sorted(half),
            )

    def test_probe_confirmed_submission_timeout_does_not_pause_the_lease(
        self,
    ) -> None:
        """The whole point of the category: the agent may carry on alone."""

        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            run = {
                "id": "run-submission-timeout",
                "workflow_id": "workflow-1",
                "status": "failed",
                "termination_cause": "program_submission_timeout",
                "analysis_json": json.dumps(
                    {
                        "failure_category": "program_submission_timeout",
                        "measurement_paused": False,
                        "attempted_target_count": len(targets),
                        "instrument_probe": {
                            "reachable": True,
                            "cause": "none",
                        },
                    }
                ),
            }
            self.assertFalse(
                service._is_instrument_connectivity_failure(run)
            )
            analysis = service._program_submission_timeout(run)
            self.assertIsNotNone(analysis)
            self.assertFalse(analysis["measurement_paused"])

            escalated = dict(run)
            escalated["analysis_json"] = json.dumps(
                {
                    "failure_category": "program_submission_timeout",
                    "measurement_paused": True,
                    "attempted_target_count": 1,
                }
            )
            self.assertTrue(
                service._program_submission_timeout(escalated)[
                    "measurement_paused"
                ]
            )

    def test_finished_node_advances_when_every_run_is_decided(self) -> None:
        """A finished node must not trap the workflow.

        The playbook's scientific boundary requires `manual_review` on the
        failed target's evidence run. When that lands on the node's last run
        there is nothing left to record `advance` on, and no further run can be
        scheduled because no target is unresolved. Requesting the next node in
        sequence then has to be accepted as the advance.
        """

        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            workflow = start_test_workflow(service, targets)
            service.db.execute(
                "UPDATE workflows SET current_node = '04' WHERE id = ?",
                (workflow["id"],),
            )
            proposal = service.request_run(
                workflow["id"], "04", {"qubits": targets},
                "Full width 05.", "unittest",
            )
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, analysis_json
                ) VALUES (?, ?, ?, '04', ?, 'completed', 'pass', ?)
                """,
                (
                    "offline-05-final",
                    workflow["id"],
                    proposal["id"],
                    json.dumps({"qubits": targets}),
                    json.dumps(analysis_for("04", targets)),
                ),
            )
            # Decided, but not with `advance` - as a boundary review would be.
            service.record_decision(
                workflow["id"], "offline-05-final", "manual_review",
                "Boundary recorded on the node's last run.",
                None, None, None, "unittest",
            )

            resolved = set(targets)
            with patch.object(
                AgentService, "_node_target_resolution", return_value=resolved
            ):
                # Nothing is unresolved and every run is decided, so the
                # node is finished with no run left to record `advance` on.
                self.assertTrue(
                    service._node_is_finished_and_fully_decided(
                        service._workflow(workflow["id"]), "05"
                    )
                )
                # ...so requesting the next node must be what advances it.
                service.request_run(
                    workflow["id"], "05", {"qubits": targets},
                    "Advance to 05 after the boundary.", "unittest",
                )
            self.assertEqual(
                service._workflow(workflow["id"])["current_node"], "05"
            )

    def test_04_caps_the_multiplex_group_at_five(self) -> None:
        """Operator instruction 2026-09-21: 04 only, and only five at a time."""

        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            self.assertEqual(service._node_multiplex_cap("04"), 5)
            # The cap is per node; everything else stays uncapped.
            for other in ("02x", "02c", "02a", "03a", "05"):
                self.assertIsNone(service._node_multiplex_cap(other))

            workflow = service.start_workflow(
                targets,
                {"multiplexed": True},
                "unittest",
                ENTRY_PHRASE,
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '04' WHERE id = ?",
                (workflow["id"],),
            )
            self.assertGreater(len(targets), 5)

            # Above the cap is refused, even though it is the full active set
            # and would satisfy the shared-first-batch rule.
            with self.assertRaisesRegex(
                Exception, "multiplexes at most 5 targets"
            ):
                service.request_run(
                    workflow["id"],
                    "04",
                    {"qubits": targets},
                    "Full width exceeds the 04 cap.",
                    "unittest",
                )

            # A capped first batch is accepted with no prior failed run, which
            # is what makes the cap replace the shared-first-batch rule.
            proposal = service.request_run(
                workflow["id"],
                "04",
                {"qubits": targets[:5]},
                "First capped 04 group.",
                "unittest",
            )
            self.assertEqual(
                sorted(proposal["payload"]["parameters"]["qubits"]),
                sorted(targets[:5]),
            )

    def test_next_action_proposes_a_capped_04_group(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            workflow = service.start_workflow(
                targets,
                {"multiplexed": True},
                "unittest",
                ENTRY_PHRASE,
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '04' WHERE id = ?",
                (workflow["id"],),
            )
            plan = service.next_action(workflow["id"])
            self.assertEqual(plan["action"], "run")
            self.assertLessEqual(len(plan["run"]["qubits"]), 5)
            self.assertIn("at most 5 targets", plan["reason"])

    def test_04_accepts_a_multiplexed_subgroup(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, all_targets = sample_multiplex_state_and_wiring()
            targets = all_targets[:3]
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            workflow = service.start_workflow(
                targets,
                {"multiplexed": True},
                "unittest",
                ENTRY_PHRASE,
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '04' WHERE id = ?",
                (workflow["id"],),
            )
            with self.assertRaisesRegex(
                Exception, "first 04 run must multiplex every active target"
            ):
                service.request_run(
                    workflow["id"],
                    "04",
                    {"qubits": targets[1:]},
                    "First 04 cannot start as a subgroup.",
                    "unittest",
                )
            first = service.request_run(
                workflow["id"],
                "04",
                {"qubits": targets},
                "First 04 multiplexes every active target.",
                "unittest",
            )
            analysis = analysis_for("04", targets)
            analysis["fit_quality"]["results"][targets[1]]["Pi_amplitude"] = 0.0
            analysis["fit_quality"]["results"][targets[2]]["Pi_amplitude"] = 0.0
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, analysis_json
                ) VALUES (?, ?, ?, '04', ?, 'completed', 'needs_review', ?)
                """,
                (
                    "offline-04-first",
                    workflow["id"],
                    first["id"],
                    json.dumps({"qubits": targets}),
                    json.dumps(analysis),
                ),
            )
            proposal = service.request_run(
                workflow["id"],
                "04",
                {"qubits": targets[1:]},
                "Measure only unresolved Power-Rabi targets.",
                "unittest",
            )
            self.assertEqual(
                proposal["payload"]["parameters"]["qubits"], targets[1:]
            )
            self.assertTrue(proposal["payload"]["parameters"]["multiplexed"])

    def test_05_accepts_a_multiplexed_subgroup(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, all_targets = sample_multiplex_state_and_wiring()
            targets = all_targets[:3]
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            workflow = service.start_workflow(
                targets,
                {"multiplexed": True},
                "unittest",
                ENTRY_PHRASE,
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '05' WHERE id = ?",
                (workflow["id"],),
            )
            with self.assertRaisesRegex(
                Exception, "first 05 run must multiplex every active target"
            ):
                service.request_run(
                    workflow["id"],
                    "05",
                    {"qubits": [targets[0]]},
                    "First 05 cannot start as an isolated qubit.",
                    "unittest",
                )
            first = service.request_run(
                workflow["id"],
                "05",
                {"qubits": targets},
                "First 05 multiplexes every active target.",
                "unittest",
            )
            analysis = analysis_for("05", targets)
            analysis["fit_quality"]["results"][targets[1]]["r_squared"] = 0.5
            analysis["fit_quality"]["results"][targets[2]]["r_squared"] = 0.5
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, analysis_json
                ) VALUES (?, ?, ?, '05', ?, 'completed', 'needs_review', ?)
                """,
                (
                    "offline-05-first",
                    workflow["id"],
                    first["id"],
                    json.dumps({"qubits": targets}),
                    json.dumps(analysis),
                ),
            )
            proposal = service.request_run(
                workflow["id"],
                "05",
                {"qubits": [targets[1]]},
                "Isolate one T1 trace so another qubit cannot abort its fit.",
                "unittest",
            )
            self.assertEqual(
                proposal["payload"]["parameters"]["qubits"], [targets[1]]
            )
            self.assertTrue(proposal["payload"]["parameters"]["multiplexed"])
            self.assertTrue(
                any(
                    "retry them together with the same parameters" in warning
                    for warning in proposal["payload"]["warnings"]
                )
            )

    def test_05_resolution_is_per_qubit(self) -> None:
        analysis = analysis_for("05", ["q1", "q2"])
        analysis["fit_quality"]["results"]["q2"]["r_squared"] = 0.5
        rows = [
            {
                "status": "completed",
                "analysis_status": "needs_review",
                "parameters_json": json.dumps({"qubits": ["q1", "q2"]}),
                "analysis_json": json.dumps(analysis),
            }
        ]
        self.assertEqual(resolve_node_targets("05", rows), {"q1"})

    def test_04_and_05_require_reference_quality_snr(self) -> None:
        rabi = analysis_for("04", ["q1", "q3"])
        rabi["dataset_metrics"]["qubits"]["q1"]["robust_snr"] = 13.9
        rabi["dataset_metrics"]["qubits"]["q3"]["robust_snr"] = 14.9
        t1 = analysis_for("05", ["q1", "q3"])
        t1["dataset_metrics"]["qubits"]["q1"]["robust_snr"] = 24.9
        t1["dataset_metrics"]["qubits"]["q3"]["robust_snr"] = 25.1
        self.assertEqual(
            resolve_node_targets(
                "04",
                [{"analysis_status": "needs_review", "analysis_json": json.dumps(rabi)}],
                min_04_snr=14.0,
            ),
            {"q3"},
        )
        self.assertEqual(
            resolve_node_targets(
                "05",
                [{"analysis_status": "needs_review", "analysis_json": json.dumps(t1)}],
                min_05_snr=25.0,
            ),
            {"q3"},
        )

    def test_new_lease_reopens_advanced_04_when_snr_policy_regresses(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, _wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            service = AgentService(settings)
            target = targets[0]
            workflow = start_test_workflow(service, [target])
            service.db.execute(
                "UPDATE workflows SET current_node = '05' WHERE id = ?",
                (workflow["id"],),
            )
            proposal = service._create_proposal(
                workflow["id"], "run", {}, "unittest", 60, 1
            )
            analysis = analysis_for("04", [target])
            # Must stay below the current analysis."04".min_robust_snr floor so
            # the revalidation path is exercised; the floor was lowered from
            # 14.0 to 7.0 on 2026-09-20.
            analysis["dataset_metrics"]["qubits"][target]["robust_snr"] = 6.0
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, analysis_json
                ) VALUES ('old-04', ?, ?, '04', ?, 'completed', 'pass', ?)
                """,
                (
                    workflow["id"],
                    proposal["id"],
                    json.dumps({"qubits": [target]}),
                    json.dumps(analysis),
                ),
            )
            service.db.execute(
                """
                INSERT INTO decisions(
                    id, workflow_id, run_id, decision, reason, next_node,
                    next_parameters_json, state_patch_json, client_id, created_at
                ) VALUES ('old-04-advance', ?, 'old-04', 'advance',
                          'Accepted under an older rule.', '05', '{}', '[]',
                          'unittest', '2026-08-19T00:00:00+00:00')
                """,
                (workflow["id"],),
            )
            lease = service.request_autonomy_lease(
                workflow["id"],
                "unittest",
                service.autonomy_mode_entry_phrase,
                "Revalidate old evidence before continuing.",
            )
            self.assertEqual(lease["payload"]["allowed_nodes"][0], "04")
            self.assertEqual(service.status(workflow["id"])["workflow"]["current_node"], "04")
            event = service.db.one(
                "SELECT payload_json FROM events "
                "WHERE event_type = 'workflow_evidence_revalidation' "
                "ORDER BY id DESC LIMIT 1"
            )
            self.assertIn(target, json.loads(event["payload_json"])["missing_targets"])
    @staticmethod
    def _insert_run(
        service: AgentService,
        workflow_id: str,
        proposal_id: str,
        run_id: str,
        node_id: str,
        targets: list[str],
        stage_03a: str = "fine",
    ) -> None:
        analysis = analysis_for(node_id, targets)
        if node_id == "03a":
            analysis["03a_stage"] = stage_03a
        service.db.execute(
            """
            INSERT INTO runs(
                id, workflow_id, proposal_id, node_id, parameters_json,
                status, analysis_status, analysis_json
            ) VALUES (?, ?, ?, ?, ?, 'completed', 'pass', ?)
            """,
            (
                run_id,
                workflow_id,
                proposal_id,
                node_id,
                json.dumps(
                    {
                        "qubits": targets,
                        "frequency_span_in_mhz": (
                            800.0 if stage_03a == "coarse_candidate" else 50.0
                        ),
                        "operation_amplitude_factor": (
                            0.10 if stage_03a == "coarse_candidate" else 0.02
                        ),
                    }
                ),
                json.dumps(analysis),
            ),
        )


if __name__ == "__main__":
    unittest.main()


def _with_readout_nodes(service: AgentService) -> None:
    """Extend the test fixture's short sequence to reach 07d and 07b."""

    sequence = list(service.settings.workflow_sequence)
    for node_id in ("07d", "07b"):
        if node_id not in sequence:
            sequence.append(node_id)
    object.__setattr__(
        service.settings, "workflow_sequence", tuple(sequence)
    )


class SinglePassNodeTests(unittest.TestCase):
    """07d and 07b run once on defaults and then advance.

    Operator instruction 2026-09-21: no retry, no parameter adjustment, no
    per-target chasing at these two nodes.
    """

    def test_only_07d_and_07b_are_single_pass(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, _wiring, _targets = sample_multiplex_state_and_wiring()
            service = AgentService(make_settings(Path(folder), state))
            for node_id in ("07d", "07b"):
                self.assertTrue(
                    service._node_is_single_pass(node_id), node_id
                )
            for node_id in ("02x", "02c", "02a", "03a", "04", "05", "06"):
                self.assertFalse(
                    service._node_is_single_pass(node_id), node_id
                )

    def test_07d_sits_immediately_before_07b(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, _wiring, _targets = sample_multiplex_state_and_wiring()
            service = AgentService(make_settings(Path(folder), state))
            # The shipped sequence, not the truncated test fixture one.
            import yaml

            raw = yaml.safe_load(
                (Path("config") / "agent.yaml").read_text(encoding="utf-8")
            )
            sequence = list(raw["workflow"]["sequence"])
            self.assertEqual(
                sequence.index("07b"), sequence.index("07d") + 1
            )

    def test_07d_defaults_are_registered_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            _with_readout_nodes(service)
            definition = service.policy.node_definition("07d")
            self.assertEqual(
                definition["script"],
                "07d_Readout_Frequency_Duration_Power_Optimization.py",
            )
            defaults = definition["defaults"]
            # 40 -> 100 on 2026-10-02 for the two-blob point selection.
            self.assertEqual(defaults["num_runs"], 100)
            self.assertEqual(defaults["plotting_dimension"], "2D")
            self.assertTrue(defaults["multiplexed"])
            # Every default must satisfy its own declared validator.
            service.policy.validate_run(
                "07d", {"qubits": list(targets), **defaults}
            )

    def test_a_second_single_pass_run_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            _with_readout_nodes(service)
            workflow = start_test_workflow(service, targets)
            service.db.execute(
                "UPDATE workflows SET current_node = '07b' WHERE id = ?",
                (workflow["id"],),
            )
            proposal = service.request_run(
                workflow["id"], "07b", {"qubits": targets},
                "The one 07b run.", "unittest",
            )
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status
                ) VALUES (?, ?, ?, '07b', ?, 'completed', 'needs_review')
                """,
                (
                    "offline-07b-once",
                    workflow["id"],
                    proposal["id"],
                    json.dumps({"qubits": targets}),
                ),
            )
            with self.assertRaisesRegex(Exception, "runs once per target"):
                service.request_run(
                    workflow["id"], "07b", {"qubits": targets},
                    "A second 07b run.", "unittest",
                )
            plan = service.next_action(workflow["id"])
            self.assertEqual(plan["action"], "record_decision")

    def test_single_pass_failures_advance_and_stay_active(self) -> None:
        """Operator choice 2026-10-03: q1/q2 failed 07b's blob test but carry
        on to 06, which needs no state discrimination."""

        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            _with_readout_nodes(service)
            workflow = start_test_workflow(service, targets)
            service.db.execute(
                "UPDATE workflows SET current_node = '07b' WHERE id = ?",
                (workflow["id"],),
            )
            proposal = service.request_run(
                workflow["id"], "07b", {"qubits": targets},
                "The one 07b run.", "unittest",
            )
            passing = sorted(targets)[:4]
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, analysis_json
                ) VALUES (?, ?, ?, '07b', ?, 'completed', 'needs_review', ?)
                """,
                (
                    "offline-07b-partial",
                    workflow["id"],
                    proposal["id"],
                    json.dumps({"qubits": targets}),
                    json.dumps(
                        {
                            "passing_targets": passing,
                            "fit_quality": {
                                "results": {
                                    name: {
                                        "fit_successful": name in passing,
                                        "readout_fidelity": 0.9 if name in passing else 0.7,
                                        "morphology_pass": True,
                                    }
                                    for name in targets
                                }
                            },
                        }
                    ),
                ),
            )
            object.__setattr__(
                service.settings,
                "workflow_sequence",
                (*service.settings.workflow_sequence, "06"),
            )
            service.record_decision(
                workflow["id"],
                "offline-07b-partial",
                "advance",
                "Single pass done; failures carry no update.",
                "06",
                None,
                None,
                "unittest",
            )
            refreshed = service._workflow(workflow["id"])
            self.assertEqual(
                service._active_targets_for_node(refreshed, "06"), set(targets)
            )

    def test_07d_runs_in_groups_of_at_most_four(self) -> None:
        """Operator instruction 2026-09-29: the node's batching drops qubits
        above four, so 07d covers its targets in capped groups."""

        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            _with_readout_nodes(service)
            self.assertEqual(service._node_multiplex_cap("07d"), 4)
            self.assertGreater(len(targets), 4)
            workflow = service.start_workflow(
                targets, {"multiplexed": True}, "unittest", ENTRY_PHRASE
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '07d' WHERE id = ?",
                (workflow["id"],),
            )
            plan = service.next_action(workflow["id"])
            self.assertEqual(plan["action"], "run")
            self.assertEqual(plan["run"]["qubits"], sorted(targets)[:4])

            with self.assertRaisesRegex(Exception, "at most 4 targets"):
                service.request_run(
                    workflow["id"], "07d", {"qubits": targets},
                    "Full width exceeds the 07d cap.", "unittest",
                )
            first = sorted(targets)[:4]
            proposal = service.request_run(
                workflow["id"], "07d", {"qubits": first},
                "First 07d group.", "unittest",
            )
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status
                ) VALUES (?, ?, ?, '07d', ?, 'completed', 'needs_review')
                """,
                (
                    "offline-07d-group-1",
                    workflow["id"],
                    proposal["id"],
                    json.dumps({"qubits": first}),
                ),
            )
            # The measured group is refused again; the rest is still open.
            with self.assertRaisesRegex(Exception, "runs once per target"):
                service.request_run(
                    workflow["id"], "07d", {"qubits": first},
                    "Remeasure the first group.", "unittest",
                )
            rest = sorted(set(targets) - set(first))
            service.request_run(
                workflow["id"], "07d", {"qubits": rest},
                "Second 07d group.", "unittest",
            )
            # 07d stays out of SUBGROUP_NODES, so 07b still receives every
            # target regardless of 07d's per-target evidence.
            self.assertEqual(
                service._active_targets_for_node(
                    service._workflow(workflow["id"]), "07b"
                ),
                set(targets),
            )


class ActiveResetRepeatTests(unittest.TestCase):
    """07b earns one active-reset repeat above the trigger fidelity.

    Operator instruction 2026-09-21: a thermal 07b result above 0.80 readout
    fidelity is repeated once with active reset; that repeat is the accepted
    result and those qubits use active reset downstream.
    """

    def build(self, folder: str):
        state, wiring, targets = sample_multiplex_state_and_wiring()
        settings = make_settings(Path(folder), state)
        atomic_write_json(settings.wiring_path, wiring)
        service = AgentService(settings)
        _with_readout_nodes(service)
        workflow = start_test_workflow(service, targets)
        service.db.execute(
            "UPDATE workflows SET current_node = '07b' WHERE id = ?",
            (workflow["id"],),
        )
        return service, workflow, targets

    def _record_thermal_run(self, service, workflow, targets, fidelities):
        proposal = service.request_run(
            workflow["id"], "07b", {"qubits": targets},
            "The one thermal 07b run.", "unittest",
        )
        service.db.execute(
            """
            INSERT INTO runs(
                id, workflow_id, proposal_id, node_id, parameters_json,
                status, analysis_status, analysis_json
            ) VALUES (?, ?, ?, '07b', ?, 'completed', 'needs_review', ?)
            """,
            (
                "offline-07b-thermal",
                workflow["id"],
                proposal["id"],
                json.dumps(
                    {"qubits": targets, "reset_type_thermal_or_active": "thermal"}
                ),
                json.dumps(
                    {
                        "fit_quality": {
                            "results": {
                                name: {"readout_fidelity": value, "morphology_pass": True}
                                for name, value in fidelities.items()
                            }
                        }
                    }
                ),
            ),
        )
        service.record_decision(
            workflow["id"], "offline-07b-thermal", "repeat",
            "Single-pass thermal run recorded.", None, None, None, "unittest",
        )

    def test_trigger_selects_only_qubits_above_the_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, targets = self.build(folder)
            fidelities = {name: 0.75 for name in targets}
            fidelities[targets[0]] = 0.84
            fidelities[targets[1]] = 0.805
            fidelities[targets[2]] = 0.80  # exactly at the trigger, not above
            self._record_thermal_run(service, workflow, targets, fidelities)

            earned = service._active_reset_repeat_targets(
                service._workflow(workflow["id"])
            )
            self.assertEqual(earned, {targets[0], targets[1]})

            plan = service.next_action(workflow["id"])
            self.assertEqual(plan["action"], "run")
            self.assertEqual(
                plan["run"]["parameters"]["reset_type_thermal_or_active"],
                "active",
            )
            self.assertEqual(
                sorted(plan["run"]["qubits"]), sorted([targets[0], targets[1]])
            )

    def test_reset_groups_split_later_nodes(self) -> None:
        """2026-10-04: qualified qubits run 06 with active reset, the rest
        thermal, each group with its own shared first batch."""

        with tempfile.TemporaryDirectory() as folder:
            service, workflow, targets = self.build(folder)
            # The live workflow is multiplexed; the shared-first-batch rule
            # applies only then.
            service.db.execute(
                "UPDATE workflows SET initial_parameters_json = ? WHERE id = ?",
                (json.dumps({"multiplexed": True}), workflow["id"]),
            )
            object.__setattr__(
                service.settings,
                "workflow_sequence",
                (*service.settings.workflow_sequence, "06"),
            )
            qualified = sorted(targets)[:2]
            proposal = service.request_run(
                workflow["id"], "07b", {"qubits": targets},
                "Thermal 07b.", "unittest",
            )
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, analysis_json
                ) VALUES (?, ?, ?, '07b', ?, 'completed', 'pass', ?)
                """,
                (
                    "offline-07b-active-q",
                    workflow["id"],
                    proposal["id"],
                    json.dumps(
                        {"qubits": qualified, "reset_type_thermal_or_active": "active"}
                    ),
                    json.dumps(
                        {
                            "fit_quality": {
                                "results": {
                                    name: {
                                        "fit_successful": True,
                                        "active_reset_qualified": True,
                                        "morphology_pass": True,
                                        "readout_fidelity": 0.95,
                                    }
                                    for name in qualified
                                }
                            }
                        }
                    ),
                ),
            )
            service.record_decision(
                workflow["id"], "offline-07b-active-q", "repeat",
                "Active repeat accepted.", None, None, None, "unittest",
            )
            service.db.execute(
                "UPDATE workflows SET current_node = '06' WHERE id = ?",
                (workflow["id"],),
            )
            current = service._workflow(workflow["id"])
            groups = service._reset_groups(current, "06")
            self.assertEqual(groups["active"], set(qualified))
            self.assertEqual(groups["thermal"], set(targets) - set(qualified))

            plan = service.next_action(workflow["id"])
            self.assertEqual(plan["run"]["qubits"], qualified)
            self.assertEqual(plan["run"]["parameters"]["reset_type"], "active")

            # Mixing groups, or a partial first batch, is refused.
            with self.assertRaises(Exception):
                service.request_run(
                    workflow["id"], "06",
                    {"qubits": targets, "reset_type": "thermal"},
                    "Everyone thermal.", "unittest",
                )
            with self.assertRaises(Exception):
                service.request_run(
                    workflow["id"], "06",
                    {"qubits": qualified[:1], "reset_type": "active"},
                    "Partial active batch.", "unittest",
                )
            service.request_run(
                workflow["id"], "06",
                {"qubits": qualified, "reset_type": "active"},
                "Active group first batch.", "unittest",
            )

    def test_active_repeat_is_allowed_for_targets_that_passed(self) -> None:
        """2026-10-03: the planner proposed the repeat for eight qubits that
        had passed the thermal run, and the retry guard refused them."""

        with tempfile.TemporaryDirectory() as folder:
            service, workflow, targets = self.build(folder)
            proposal = service.request_run(
                workflow["id"], "07b", {"qubits": targets},
                "The one thermal 07b run.", "unittest",
            )
            passing = sorted(targets)[:2]
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status, analysis_json
                ) VALUES (?, ?, ?, '07b', ?, 'completed', 'needs_review', ?)
                """,
                (
                    "offline-07b-thermal-pass",
                    workflow["id"],
                    proposal["id"],
                    json.dumps(
                        {"qubits": targets, "reset_type_thermal_or_active": "thermal"}
                    ),
                    json.dumps(
                        {
                            "fit_quality": {
                                "results": {
                                    name: {
                                        "fit_successful": name in passing,
                                        "readout_fidelity": 0.9 if name in passing else 0.7,
                                        "morphology_pass": True,
                                    }
                                    for name in targets
                                }
                            }
                        }
                    ),
                ),
            )
            service.record_decision(
                workflow["id"], "offline-07b-thermal-pass", "repeat",
                "Thermal run recorded.", None, None, None, "unittest",
            )
            plan = service.next_action(workflow["id"])
            self.assertEqual(sorted(plan["run"]["qubits"]), passing)
            service.request_run(
                workflow["id"], "07b",
                {"qubits": passing, "reset_type_thermal_or_active": "active"},
                "Active-reset repeat for the passing pair.", "unittest",
            )

    def test_active_repeat_is_allowed_once_then_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, targets = self.build(folder)
            self._record_thermal_run(
                service, workflow, targets, {name: 0.9 for name in targets}
            )
            # A thermal repeat is still refused.
            with self.assertRaisesRegex(Exception, "runs once per target"):
                service.request_run(
                    workflow["id"], "07b",
                    {"qubits": targets, "reset_type_thermal_or_active": "thermal"},
                    "A second thermal run.", "unittest",
                )
            # The active repeat is allowed exactly once.
            proposal = service.request_run(
                workflow["id"], "07b",
                {"qubits": targets, "reset_type_thermal_or_active": "active"},
                "Active-reset repeat.", "unittest",
            )
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status
                ) VALUES (?, ?, ?, '07b', ?, 'completed', 'needs_review')
                """,
                (
                    "offline-07b-active",
                    workflow["id"],
                    proposal["id"],
                    json.dumps(
                        {"qubits": targets,
                         "reset_type_thermal_or_active": "active"}
                    ),
                ),
            )
            service.record_decision(
                workflow["id"], "offline-07b-active", "repeat",
                "Active repeat recorded.", None, None, None, "unittest",
            )
            with self.assertRaisesRegex(Exception, "runs once per target"):
                service.request_run(
                    workflow["id"], "07b",
                    {"qubits": targets, "reset_type_thermal_or_active": "active"},
                    "A second active run.", "unittest",
                )
            self.assertEqual(service.next_action(workflow["id"])["action"], "advance")

    def test_no_repeat_when_nothing_clears_the_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, targets = self.build(folder)
            self._record_thermal_run(
                service, workflow, targets, {name: 0.72 for name in targets}
            )
            self.assertEqual(
                service._active_reset_repeat_targets(
                    service._workflow(workflow["id"])
                ),
                set(),
            )
            self.assertEqual(service.next_action(workflow["id"])["action"], "advance")


class SinglePassSubgroupTests(unittest.TestCase):
    """Single pass means one pass per target, not one run per chip.

    A single-pass node that has to be split into multiplex subgroups still
    needs a run per subgroup; only an already-measured target is refused.
    """

    def test_second_subgroup_is_allowed_but_a_repeat_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            state, wiring, targets = sample_multiplex_state_and_wiring()
            settings = make_settings(Path(folder), state)
            atomic_write_json(settings.wiring_path, wiring)
            service = AgentService(settings)
            _with_readout_nodes(service)
            workflow = start_test_workflow(service, targets)
            service.db.execute(
                "UPDATE workflows SET current_node = '07b' WHERE id = ?",
                (workflow["id"],),
            )
            first, second = targets[:2], targets[2:]
            proposal = service.request_run(
                workflow["id"], "07b", {"qubits": first},
                "First 07b subgroup.", "unittest",
            )
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, analysis_status
                ) VALUES (?, ?, ?, '07b', ?, 'completed', 'needs_review')
                """,
                (
                    "offline-07b-first",
                    workflow["id"],
                    proposal["id"],
                    json.dumps({"qubits": first}),
                ),
            )
            service.record_decision(
                workflow["id"], "offline-07b-first", "repeat",
                "First subgroup done.", None, None, None, "unittest",
            )

            # The node is not finished while targets remain unmeasured.
            self.assertFalse(
                service._node_is_finished_and_fully_decided(
                    service._workflow(workflow["id"]), "06"
                )
            )
            # Re-running a measured target is refused...
            with self.assertRaisesRegex(Exception, "already have a completed run"):
                service.request_run(
                    workflow["id"], "07b", {"qubits": first},
                    "Repeat of the first subgroup.", "unittest",
                )
            # ...but the remaining subgroup is allowed.
            service.request_run(
                workflow["id"], "07b", {"qubits": second},
                "Second 07b subgroup.", "unittest",
            )
