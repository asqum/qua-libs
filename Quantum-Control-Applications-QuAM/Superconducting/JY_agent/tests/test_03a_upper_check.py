"""03a upward check for multi-photon lines, operator instruction 2026-10-19.

A fine 03a line commits only after one 03a run at coarse drive sweeps the
check span above it. The highest credible line 50-300 MHz higher means the
fine line was a multi-photon line: the upper line becomes the f01 candidate and
gets its own check. Once that check finds nothing, the line commits, with the
anharmonicity from the ladder f0k/k = f01 - (k-1) alpha/2 when one order fits.
"""

from __future__ import annotations

import tempfile
import unittest
import uuid

from jy_agent.service import AutonomyScopeError
from jy_agent.util import json_dumps, json_loads

from test_next_action_planner import analysis_document
from test_per_qubit_pass_rules import build, fine_03a

REFERENCE = 2.896e9


def insert_03a(service, workflow_id: str, analysis: dict, parameters: dict) -> str:
    proposal_id = uuid.uuid4().hex
    run_id = uuid.uuid4().hex
    service.db.execute(
        "INSERT INTO proposals(id, workflow_id, kind, payload_json, status, created_at, "
        "expires_at, max_uses, uses, source_client) VALUES (?, ?, 'run', '{}', "
        "'approved', '2026-10-19T00:00:00+00:00', '2026-10-19T01:00:00+00:00', 1, 1, 'test')",
        (proposal_id, workflow_id),
    )
    service.db.execute(
        "INSERT INTO runs(id, workflow_id, proposal_id, node_id, parameters_json, status, "
        "analysis_status, analysis_json) VALUES (?, ?, ?, '03a', ?, 'completed', ?, ?)",
        (
            run_id,
            workflow_id,
            proposal_id,
            json_dumps(parameters),
            analysis.get("analysis_status", "needs_review"),
            json_dumps(analysis),
        ),
    )
    return run_id


def fine_document(frequency: float) -> dict:
    document = fine_03a("q1", frequency)
    document["analysis_status"] = "pass"
    document["passing_targets"] = ["q1"]
    document["candidate_state_patch"] = [
        {"op": "replace", "path": "/qubits/q1/extras/idle_freq", "value": frequency}
    ]
    return document


FINE_PARAMETERS = {
    "qubits": ["q1"],
    "frequency_span_in_mhz": 200,
    "operation_amplitude_factor": 0.05,
}
CHECK_PARAMETERS = {
    "qubits": ["q1"],
    "frequency_span_in_mhz": 300,
    "frequency_step_in_mhz": 1,
    "operation_amplitude_factor": 1.0,
    "num_averages": 2000,
}


def check_document(
    upper_offset_hz: float | None,
    reference: float = REFERENCE,
    extra: tuple[tuple[float, float], ...] = (),
) -> dict:
    """The node fitted the reference near the lower edge of the check sweep.

    `extra` adds (offset above the reference, SNR) peaks.
    """
    center = reference + 150e6
    document = analysis_document("03a", {"q1": 9.0})
    document["03a_stage"] = "coarse_candidate"
    document["fit_quality"]["results"]["q1"] = {
        "fit_successful": True,
        "drive_freq": reference,
    }
    candidates = [
        {
            "coordinate_hz": reference - center,
            "fwhm_hz": 30e6,
            "prominence": 1e-5,
            "robust_snr": 9.0,
            "edge_fraction": 0.0,
        }
    ]
    if upper_offset_hz is not None:
        candidates.append(
            {
                "coordinate_hz": reference + upper_offset_hz - center,
                "fwhm_hz": 12e6,
                "prominence": 8e-6,
                "robust_snr": 7.5,
                "edge_fraction": 0.33,
            }
        )
    for offset, snr in extra:
        candidates.append(
            {
                "coordinate_hz": reference + offset - center,
                "fwhm_hz": 8e6,
                "prominence": 4e-6,
                "robust_snr": snr,
                "edge_fraction": 0.3,
            }
        )
    metrics = document["dataset_metrics"]["qubits"]["q1"]
    metrics.update(
        {
            "feature_coordinate": reference - center,
            "edge_fraction": 0.0,
            "peak_candidates": candidates,
        }
    )
    document["failure_reasons"] = ["q1 feature lies within 5% of a sweep edge."]
    document["failure_reasons_by_target"] = {
        "q1": ["q1 feature lies within 5% of a sweep edge."]
    }
    document["passing_targets"] = []
    return document


class UpperCheckTests(unittest.TestCase):
    def hold_fine(self, service, wid: str, frequency: float = REFERENCE) -> None:
        fine = fine_document(frequency)
        run_id = insert_03a(service, wid, fine, FINE_PARAMETERS)
        run = service.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
        held = service._require_03a_upper_check(run, json_loads(json_dumps(fine), {}))
        self.assertEqual(held["passing_targets"], [])
        self.assertEqual(held["candidate_state_patch"], [])
        self.assertEqual(held["analysis_status"], "needs_review")
        self.assertTrue(held["fit_quality"]["results"]["q1"]["upper_check_pending"])
        service.db.execute(
            "UPDATE runs SET analysis_json = ? WHERE id = ?", (json_dumps(held), run_id)
        )

    def judge_check(
        self,
        service,
        wid: str,
        upper_offset_hz: float | None,
        reference: float = REFERENCE,
        extra: tuple[tuple[float, float], ...] = (),
    ) -> dict:
        document = check_document(upper_offset_hz, reference, extra)
        run_id = insert_03a(service, wid, document, CHECK_PARAMETERS)
        run = service.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
        judged = service._03a_upper_check_result(run, json_loads(json_dumps(document), {}))
        service.db.execute(
            "UPDATE runs SET analysis_status = ?, analysis_json = ? WHERE id = ?",
            (judged["analysis_status"], json_dumps(judged), run_id),
        )
        return judged

    def test_a_fine_line_waits_for_its_check_and_only_it_may_run(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "03a")
            wid = workflow["id"]
            self.hold_fine(service, wid)
            self.assertEqual(service._03a_upper_check_pending(wid), {"q1": REFERENCE})
            service._validate_03a_run_request(workflow, dict(CHECK_PARAMETERS))
            with self.assertRaises(AutonomyScopeError):
                service._validate_03a_run_request(
                    workflow, {**CHECK_PARAMETERS, "qubits": ["q2"]}
                )
            parameters = service._03a_upper_check_run_parameters("q1")
            self.assertEqual(parameters["frequency_span_in_mhz"], 300)
            self.assertEqual(parameters["operation_amplitude_factor"], 1.0)

    def test_an_upper_line_is_checked_again_then_commits_with_anharmonicity(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "03a")
            wid = workflow["id"]
            self.hold_fine(service, wid)
            judged = self.judge_check(service, wid, 104e6)
            self.assertEqual(judged["03a_stage"], "upper_check")
            fit = judged["fit_quality"]["results"]["q1"]
            self.assertAlmostEqual(fit["drive_freq"], REFERENCE + 104e6)
            self.assertEqual(fit["upper_check"]["outcome"], "reference_is_f02_half")
            self.assertAlmostEqual(fit["upper_check"]["anharmonicity_hz"], 208e6)
            self.assertEqual(judged["passing_targets"], ["q1"])
            self.assertEqual(service._03a_upper_check_pending(wid), {})

            # The f02/2 line itself never commits.
            stale = fine_document(REFERENCE)
            run_id = insert_03a(service, wid, stale, FINE_PARAMETERS)
            run = service.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
            refused = service._require_03a_upper_check(run, json_loads(json_dumps(stale), {}))
            self.assertEqual(refused["passing_targets"], [])
            self.assertIn("multi-photon", " ".join(refused["failure_reasons"]))

            # The fine scan of the upper line waits for its own check ...
            upper_line = REFERENCE + 104.5e6
            self.hold_fine(service, wid, upper_line)
            self.assertEqual(service._03a_upper_check_pending(wid), {"q1": upper_line})
            # ... which finds nothing, so its next fine scan commits it with the
            # anharmonicity from the first check, from the fine positions.
            self.judge_check(service, wid, None, reference=upper_line)
            upper = fine_document(upper_line)
            run_id = insert_03a(service, wid, upper, FINE_PARAMETERS)
            run = service.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
            accepted = service._require_03a_upper_check(run, json_loads(json_dumps(upper), {}))
            self.assertEqual(accepted["passing_targets"], ["q1"])
            values = {item["path"]: item["value"] for item in accepted["candidate_state_patch"]}
            self.assertAlmostEqual(values["/qubits/q1/anharmonicity"], 209e6)
            self.assertAlmostEqual(values["/qubits/q1/extras/anharmonicity"], 209e6)
            service.policy.validate_state_patch(
                [item for item in accepted["candidate_state_patch"] if "anharmonicity" in item["path"]],
                json_loads(service.settings.active_state.read_text(encoding="utf-8"), {}),
            )

    def test_the_highest_line_wins_and_the_ladder_names_f03_third(self) -> None:
        # q2 #648: reference 2.7675, weak f02/2 at +122 MHz, strong f01 at +241.
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "03a")
            wid = workflow["id"]
            self.hold_fine(service, wid)
            judged = self.judge_check(
                service, wid, 241e6, extra=((122e6, 3.7), (130e6, 2.0))
            )
            check = judged["fit_quality"]["results"]["q1"]["upper_check"]
            self.assertAlmostEqual(check["upper_line_hz"], REFERENCE + 241e6)
            self.assertEqual(check["outcome"], "reference_is_f03_third")
            self.assertEqual(check["photon_order"], 3)
            self.assertAlmostEqual(check["anharmonicity_hz"], 241e6)

    def test_the_highest_credible_line_wins_over_a_stronger_lower_one(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "03a")
            wid = workflow["id"]
            self.hold_fine(service, wid)
            judged = self.judge_check(service, wid, 200e6, extra=((100e6, 20.0),))
            check = judged["fit_quality"]["results"]["q1"]["upper_check"]
            self.assertAlmostEqual(check["upper_line_hz"], REFERENCE + 200e6)
            # 200 MHz up with f02/2 at +100: order 3, alpha 200 MHz.
            self.assertEqual(check["photon_order"], 3)

    def test_an_unsupported_ladder_records_no_anharmonicity(self) -> None:
        # 241 MHz up with no intermediate line: order 3 (241 MHz) and order 4
        # (161 MHz) both lack their intermediate lines, so no order is named.
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "03a")
            wid = workflow["id"]
            self.hold_fine(service, wid)
            judged = self.judge_check(service, wid, 241e6)
            check = judged["fit_quality"]["results"]["q1"]["upper_check"]
            self.assertEqual(check["outcome"], "reference_is_multiphoton")
            self.assertIsNone(check["anharmonicity_hz"])
            self.assertEqual(judged["passing_targets"], ["q1"])

    def test_no_upper_line_confirms_f01_without_anharmonicity(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "03a")
            wid = workflow["id"]
            self.hold_fine(service, wid)
            judged = self.judge_check(service, wid, None)
            fit = judged["fit_quality"]["results"]["q1"]
            self.assertEqual(fit["upper_check"]["outcome"], "reference_is_f01")
            self.assertFalse(fit["fit_successful"])
            self.assertEqual(service._03a_upper_check_pending(wid), {})

            again = fine_document(REFERENCE + 0.5e6)
            run_id = insert_03a(service, wid, again, FINE_PARAMETERS)
            run = service.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
            accepted = service._require_03a_upper_check(run, json_loads(json_dumps(again), {}))
            self.assertEqual(accepted["passing_targets"], ["q1"])
            self.assertFalse(
                any("anharmonicity" in item["path"] for item in accepted["candidate_state_patch"])
            )

    def test_centring_follows_the_latest_closed_check(self) -> None:
        # q2 #651: an older refinement candidate must not pull the drive back.
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "03a")
            wid = workflow["id"]
            older = analysis_document("03a", {"q1": 9.0})
            older["03a_stage"] = "refinement_candidate"
            older["analysis_status"] = "pass"
            older["passing_targets"] = ["q1"]
            older["fit_quality"]["results"]["q1"] = {
                "fit_successful": True,
                "drive_freq": REFERENCE - 240e6,
            }
            insert_03a(service, wid, older, FINE_PARAMETERS)
            self.hold_fine(service, wid)
            self.judge_check(service, wid, None)
            proposal = service.request_03a_candidate_center(wid, ["q1"], "test")
            centre = proposal["payload"]["03a_candidate_center"]["q1"]
            self.assertAlmostEqual(centre["target_rf_hz"], REFERENCE)
            self.assertEqual(centre["candidate_evidence"]["03a_stage"], "upper_check")

    def test_a_line_closer_than_the_minimum_separation_does_not_count(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "03a")
            wid = workflow["id"]
            self.hold_fine(service, wid)
            judged = self.judge_check(service, wid, 30e6)
            self.assertEqual(
                judged["fit_quality"]["results"]["q1"]["upper_check"]["outcome"],
                "reference_is_f01",
            )


if __name__ == "__main__":
    unittest.main()
