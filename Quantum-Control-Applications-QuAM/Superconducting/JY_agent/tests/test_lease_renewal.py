"""Lease renewal with `繼續JY自動量測模式`, operator decisions 2026-10-20.

An expired (or otherwise ended) lease of a still-active workflow is renewed:
the new lease copies the old scope, the Approval page shows only a renewal
card with the approval code, and approving it activates the lease on the same
workflow without moving the current node. A paused lease resumes without a
proposal; an active lease is only reported.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from urllib.parse import urlencode

from jy_agent.approval_web import _lease_renewal_card, handle_session_dashboard
from jy_agent.service import ServiceError

from test_live_dashboard_telemetry import _request
from test_per_qubit_pass_rules import build


PHRASE = "繼續JY自動量測模式"


def expire(service, lease) -> None:
    service.db.execute(
        "UPDATE autonomy_leases SET expires_at = '2000-01-01T00:00:00+00:00' WHERE id = ?",
        (lease["id"],),
    )
    assert service.autonomy_status(lease_id=lease["id"])["status"] == "expired"


def leases(service, workflow_id: str) -> list[dict]:
    return service.db.all(
        "SELECT * FROM autonomy_leases WHERE workflow_id = ? ORDER BY created_at, rowid",
        (workflow_id,),
    )


class LeaseRenewalTests(unittest.TestCase):
    def test_an_expired_lease_is_renewed_on_the_same_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "04")
            old_row = service._autonomy_row(lease["id"])
            expire(service, lease)

            result = service.continue_autonomy_mode("unittest-agent", PHRASE)
            self.assertEqual(result["next_action"], "human_approval")
            self.assertTrue(result["browser_url"])
            self.assertIn("operator_handoff", result)
            self.assertEqual(len(result["operator_handoff"]["lines"]), 3)
            self.assertEqual(result["renewal_of"], lease["id"])
            proposal = result["approval"]
            payload = proposal["payload"]
            self.assertEqual(payload["renewal_of"], lease["id"])
            self.assertNotIn("workflow_options_offer", payload)
            self.assertEqual(payload["renewal"]["current_node"], "04")
            new_lease = result["authorization"]
            self.assertEqual(new_lease["status"], "pending")
            self.assertEqual(new_lease["workflow_id"], workflow["id"])
            new_row = service._autonomy_row(new_lease["id"])
            for column in (
                "targets_json",
                "allowed_nodes_json",
                "max_attempts_per_node_qubit",
                "auto_state_commit_statuses_json",
                "halt_conditions_json",
                "duration_hours",
            ):
                self.assertEqual(new_row[column], old_row[column], column)

            html = _lease_renewal_card(
                service, "session", proposal, service._decode_workflow(service._workflow(workflow["id"])), "zh-Hant"
            )
            self.assertIn('value="lease_renew"', html)
            self.assertIn("續期", html)
            self.assertIn("/ 04", html)
            self.assertIn(f"APPROVE {proposal['id']}", html)
            self.assertNotIn("qubit_plan", html)
            self.assertNotIn('value="preview"', html)
            english = _lease_renewal_card(service, "session", proposal, workflow, "en")
            self.assertIn("Renew the authorization", english)
            self.assertIn('name="duration_hours"', english)
            self.assertIn('name="auto_judgement"', english)

            csrf = service.approval_csrf_token(proposal["id"])
            with self.assertRaises(ServiceError):
                service.renew_lease_from_browser(
                    proposal["id"], "APPROVE wrong", csrf, "127.0.0.1", actor="operator"
                )
            self.assertEqual(service._autonomy_row(new_lease["id"])["status"], "pending")
            service.renew_lease_from_browser(
                proposal["id"], f"APPROVE {proposal['id']}", csrf, "127.0.0.1", actor="operator"
            )
            self.assertEqual(
                service.autonomy_status(lease_id=new_lease["id"])["status"], "active"
            )
            after = service._workflow(workflow["id"])
            self.assertEqual(after["status"], "active")
            self.assertEqual(after["current_node"], "04")
            self.assertEqual(after.get("options_json"), workflow.get("options_json"))
            self.assertEqual(
                service.db.one("SELECT COUNT(*) AS n FROM workflows")["n"], 1
            )

    def test_the_dashboard_shows_and_approves_the_renewal_card(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "04")
            expire(service, lease)
            result = service.continue_autonomy_mode("unittest-agent", PHRASE)
            session_id = result["dashboard"]["id"]
            proposal_id = result["approval"]["id"]
            path = f"/session/{session_id}/approval"

            def call(**kwargs):
                return asyncio.run(
                    handle_session_dashboard(
                        _request(path, {"session_id": session_id}, **kwargs),
                        service,
                        view="approval",
                    )
                )

            page = call().body.decode("utf-8")
            self.assertIn('value="lease_renew"', page)
            self.assertIn(f"APPROVE {proposal_id}", page)
            self.assertNotIn("qubit_plan", page)
            self.assertNotIn('name="drive_mode"', page)

            def body(confirmation: str) -> bytes:
                return urlencode(
                    {
                        "operation": "lease_renew",
                        "proposal_id": proposal_id,
                        "csrf_token": service.approval_csrf_token(proposal_id),
                        "confirmation": confirmation,
                    }
                ).encode("utf-8")

            refused = call(method="POST", body=body("APPROVE nope"))
            self.assertEqual(refused.status_code, 400)
            self.assertEqual(service.proposal(proposal_id)["status"], "pending")
            approved = call(method="POST", body=body(f"APPROVE {proposal_id}"))
            self.assertEqual(approved.status_code, 200)
            self.assertEqual(service.proposal(proposal_id)["status"], "approved")
            self.assertEqual(
                service.autonomy_status(workflow_id=workflow["id"])["status"], "active"
            )
            self.assertEqual(service._workflow(workflow["id"])["current_node"], "04")

    def test_a_renewal_is_not_a_full_lease_form_approval(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "04")
            expire(service, lease)
            proposal = service.continue_autonomy_mode("unittest-agent", PHRASE)["approval"]
            # A plain lease proposal is refused by the renewal handler.
            service.db.execute(
                "UPDATE proposals SET payload_json = '{}' WHERE id = ?", (proposal["id"],)
            )
            with self.assertRaises(ServiceError):
                service.renew_lease_from_browser(
                    proposal["id"],
                    f"APPROVE {proposal['id']}",
                    service.approval_csrf_token(proposal["id"]),
                    "127.0.0.1",
                    actor="operator",
                )

    def test_continue_again_returns_the_pending_renewal(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "04")
            expire(service, lease)
            first = service.continue_autonomy_mode("unittest-agent", PHRASE)
            second = service.continue_autonomy_mode(
                "unittest-agent", "Continue JY automatic measurement mode"
            )
            self.assertEqual(first["approval"]["id"], second["approval"]["id"])
            self.assertEqual(len(leases(service, workflow["id"])), 2)

    def test_a_paused_lease_resumes_without_a_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "04")
            service.pause_autonomy(lease["id"], "operator", "coffee")
            result = service.continue_autonomy_mode("unittest-agent", PHRASE)
            self.assertEqual(result["next_action"], "continue_authorized_workflow")
            self.assertEqual(result["authorization"]["id"], lease["id"])
            self.assertEqual(result["authorization"]["status"], "active")
            self.assertNotIn("approval", result)
            self.assertTrue(result["browser_url"])
            self.assertIn("operator_handoff", result)
            self.assertEqual(len(leases(service, workflow["id"])), 1)

    def test_the_old_resume_phrase_still_resumes_a_paused_lease(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "04")
            service.pause_autonomy(lease["id"], "operator", "coffee")
            result = service.continue_autonomy_mode("unittest-agent", "繼續 JY 自動量測")
            self.assertEqual(result["authorization"]["status"], "active")
            expire(service, lease)
            with self.assertRaises(ServiceError):
                service.continue_autonomy_mode("unittest-agent", "繼續 JY 自動量測")

    def test_an_active_lease_only_reports_status(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "04")
            result = service.continue_autonomy_mode("unittest-agent", PHRASE)
            self.assertEqual(result["next_action"], "continue_authorized_workflow")
            self.assertEqual(result["authorization"]["id"], lease["id"])
            self.assertTrue(result["browser_url"])
            self.assertIn("operator_handoff", result)
            self.assertEqual(len(leases(service, workflow["id"])), 1)

    def test_no_open_workflow_points_to_the_entry_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, lease = build(folder, "04")
            service.stop_workflow(workflow["id"], "operator", "done")
            with self.assertRaises(ServiceError) as caught:
                service.continue_autonomy_mode("unittest-agent", PHRASE)
            self.assertIn(service.settings.autonomy_mode_entry_phrase, str(caught.exception))

    def test_a_wrong_phrase_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, _lease = build(folder, "04")
            with self.assertRaises(ServiceError):
                service.continue_autonomy_mode("unittest-agent", "continue")

    def test_waiting_on_an_expired_lease_hands_off_the_continue_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, _workflow, lease = build(folder, "04")
            expire(service, lease)
            waited = service.wait_for_autonomy_status(lease["id"], timeout_seconds=0)
            self.assertEqual(waited["operator_handoff"]["reply"], PHRASE)


if __name__ == "__main__":
    unittest.main()
