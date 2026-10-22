"""Dashboard pause -> continue or re-plan, operator request 2026-10-19.

A manual pause from Home offers, on the Approval page, to continue the same
lease or to re-plan: each qubit is not measured, starts from the beginning,
or restarts at a node it reached. Re-planning ends the workflow and starts a
new one whose lease is active at once.
"""

from __future__ import annotations

import tempfile
import unittest

from jy_agent.approval_web import _replan_card
from jy_agent.service import ServiceError

from test_per_qubit_pass_rules import build


def pause(service, lease) -> dict:
    return service.control_autonomy_from_browser(
        lease["id"], "pause", service.autonomy_csrf_token(lease["id"]), "operator"
    )


def replan_proposal(service, workflow_id: str) -> dict:
    row = service.db.one(
        "SELECT id FROM proposals WHERE workflow_id = ? AND kind = 'autonomy_replan' "
        "AND status = 'pending'",
        (workflow_id,),
    )
    return service.proposal(row["id"]) if row else None


class DashboardReplanTests(unittest.TestCase):
    def test_a_manual_pause_offers_continue_or_replan(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "02c")
            pause(service, lease)
            proposal = replan_proposal(service, workflow["id"])
            self.assertIsNotNone(proposal)
            plan = proposal["payload"]["qubit_plan"]["qubits"]
            self.assertTrue(all(choices[:2] == ["skip", "beginning"] for choices in plan.values()))
            html = _replan_card(service, "session", proposal, "zh-Hant")
            self.assertIn('value="replan_continue"', html)
            self.assertIn('value="replan_apply"', html)
            self.assertIn("qubit_plan__q1", html)

            # Continue resumes the same lease and drops the offer.
            service.replan_continue_from_browser(
                proposal["id"], service.approval_csrf_token(proposal["id"]), "operator"
            )
            self.assertEqual(service.autonomy_status(lease_id=lease["id"])["status"], "active")
            self.assertIsNone(replan_proposal(service, workflow["id"]))

    def test_replan_starts_a_new_workflow_with_an_active_lease(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "02c")
            pause(service, lease)
            proposal = replan_proposal(service, workflow["id"])
            plan = {name: "beginning" for name in proposal["payload"]["qubit_plan"]["qubits"]}
            with self.assertRaises(ServiceError):
                service.replan_apply_from_browser(
                    proposal["id"], "APPROVE wrong", service.approval_csrf_token(proposal["id"]),
                    plan, "operator",
                )
            result = service.replan_apply_from_browser(
                proposal["id"],
                f"APPROVE {proposal['id']}",
                service.approval_csrf_token(proposal["id"]),
                plan,
                "operator",
            )
            self.assertEqual(service._workflow(workflow["id"])["status"], "stopped")
            new = service._workflow(result["new_workflow_id"])
            self.assertEqual(new["status"], "active")
            leases = service.db.all(
                "SELECT status FROM autonomy_leases WHERE workflow_id = ?", (new["id"],)
            )
            self.assertEqual([row["status"] for row in leases], ["active"])

    def test_a_resume_from_home_drops_the_offer(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "02c")
            pause(service, lease)
            service.control_autonomy_from_browser(
                lease["id"], "resume", service.autonomy_csrf_token(lease["id"]), "operator"
            )
            self.assertIsNone(replan_proposal(service, workflow["id"]))


if __name__ == "__main__":
    unittest.main()
