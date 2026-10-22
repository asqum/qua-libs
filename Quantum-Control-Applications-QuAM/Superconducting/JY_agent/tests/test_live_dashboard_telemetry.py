from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlencode

import numpy as np
import xarray as xr

from starlette.requests import Request

from jy_agent.approval_web import (
    handle_autonomy_control,
    handle_autonomy_control_asset,
    handle_browser_approval,
    handle_session_dashboard,
)
from jy_agent.policy import PolicyEngine, PolicyError
from jy_agent.service import AgentService, ServiceError
from jy_agent.util import json_dumps

from test_autonomy import AUTO_PHRASE
from test_core import make_settings, sample_state, start_test_workflow
from test_core import SUPERCONDUCTING_ROOT


FULL_SEQUENCE = (
    "02x",
    "02c",
    "02a",
    "03a",
    "04",
    "05",
    "07b",
    "06",
    "06b",
    "10a",
    "05st",
    "06st_t2star",
    "06st_t2e",
)


def _request(
    path: str,
    path_params: dict[str, object],
    *,
    method: str = "GET",
    body: bytes = b"",
    query: str = "",
) -> Request:
    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    headers = [(b"host", b"127.0.0.1:8766")]
    if method == "POST":
        headers.append(
            (b"content-type", b"application/x-www-form-urlencoded")
        )
        headers.append((b"origin", b"http://127.0.0.1:8766"))
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "root_path": "",
            "query_string": query.encode("ascii"),
            "headers": headers,
            "client": ("127.0.0.1", 50123),
            "server": ("127.0.0.1", 8766),
            "path_params": path_params,
        },
        receive,
    )


class LiveDashboardTelemetryTests(unittest.TestCase):
    def test_autonomy_wait_wakes_when_pending_lease_is_approved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            entered = service.enter_autonomy_mode(
                "unittest-agent", AUTO_PHRASE, ["q1"]
            )
            proposal_id = entered["approval"]["id"]

            def approve() -> None:
                time.sleep(0.05)
                service.approve_from_browser(
                    proposal_id,
                    f"APPROVE {proposal_id}",
                    service.approval_csrf_token(proposal_id),
                    "127.0.0.1",
                    lease_options={"optional_nodes": [], "drive_mode": "multiplex"},
                )

            thread = threading.Thread(target=approve)
            thread.start()
            waited = service.wait_for_autonomy_status(
                entered["authorization"]["id"], timeout_seconds=2
            )
            thread.join(timeout=2)

            self.assertFalse(waited["timed_out"])
            self.assertEqual(waited["authorization"]["status"], "active")

    def test_split_dashboard_pages_and_browser_resume_have_distinct_roles(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            entered = service.enter_autonomy_mode(
                "unittest-agent", AUTO_PHRASE, ["q1"]
            )
            session_id = entered["dashboard"]["id"]
            proposal_id = entered["approval"]["id"]

            home = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{session_id}/home", {"session_id": session_id}
                    ),
                    service,
                    view="home",
                )
            ).body.decode("utf-8")
            self.assertIn("JY 量測首頁", home)
            self.assertIn('name="operation" value="shutdown"', home)
            self.assertIn(f"SHUTDOWN {session_id}", home)
            self.assertNotIn('name="operation" value="approve"', home)
            self.assertNotIn("data-experiment-selector", home)

            approval = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{session_id}/approval",
                        {"session_id": session_id},
                    ),
                    service,
                    view="approval",
                )
            ).body.decode("utf-8")
            # Operator request 2026-10-10: the code appears only after a preview.
            self.assertIn('name="operation" value="preview"', approval)
            self.assertNotIn(f"APPROVE {proposal_id}", approval)
            self.assertNotIn('name="operation" value="shutdown"', approval)
            self.assertNotIn("data-experiment-selector", approval)

            service._approve_pending_proposal(
                proposal_id, "unit-test-human", "unit_test"
            )
            lease_id = entered["authorization"]["id"]
            service.pause_autonomy(lease_id, "unit-test-human", "Test pause")
            home_controls = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{session_id}/home", {"session_id": session_id}
                    ),
                    service,
                    view="home",
                )
            ).body.decode("utf-8")
            self.assertIn('name="action" value="resume"', home_controls)
            self.assertIn("結束自動授權（保留網站）", home_controls)
            self.assertIn("緊急停止 worker（保留網站）", home_controls)
            self.assertIn('name="operation" value="shutdown"', home_controls)

            results = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{session_id}/results",
                        {"session_id": session_id},
                    ),
                    service,
                    view="results",
                )
            ).body.decode("utf-8")
            self.assertIn("結果摘要", results)
            self.assertNotIn('name="action" value="resume"', results)
            self.assertNotIn("緊急停止 worker（保留網站）", results)
            self.assertNotIn('name="operation" value="shutdown"', results)

            resume_body = urlencode(
                {
                    "operation": "control",
                    "action": "resume",
                    "csrf_token": service.autonomy_csrf_token(lease_id),
                }
            ).encode("utf-8")
            resumed = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{session_id}/home",
                        {"session_id": session_id},
                        method="POST",
                        body=resume_body,
                    ),
                    service,
                    view="home",
                )
            )
            self.assertEqual(resumed.status_code, 200)
            self.assertEqual(
                service.autonomy_status(lease_id=lease_id)["status"], "active"
            )

    def test_autonomous_entry_dashboard_shows_initial_lease_approval_form(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            entered = service.enter_autonomy_mode(
                "unittest-agent", AUTO_PHRASE, ["q1"]
            )
            session_id = entered["dashboard"]["id"]
            proposal_id = entered["approval"]["id"]

            review = service.dashboard_review(session_id)
            self.assertEqual(review["pending_proposal"]["id"], proposal_id)
            self.assertEqual(
                review["pending_proposal"]["kind"], "autonomy_lease"
            )

            response = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{session_id}/approval", {"session_id": session_id}
                    ),
                    service,
                    view="approval",
                )
            )
            body = response.body.decode("utf-8")
            self.assertIn("待核准動作", body)
            # Operator request 2026-10-10: choose, preview, then agree + code.
            self.assertNotIn(f"APPROVE {proposal_id}", body)
            self.assertNotIn('name="confirmation"', body)
            self.assertIn('name="drive_mode" value="per_qubit" required', body)

            preview_body = urlencode(
                {
                    "operation": "preview",
                    "proposal_id": proposal_id,
                    "csrf_token": service.approval_csrf_token(proposal_id),
                    "drive_mode": "multiplex",
                }
            ).encode("utf-8")
            previewed = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{session_id}/approval",
                        {"session_id": session_id},
                        method="POST",
                        body=preview_body,
                    ),
                    service,
                    view="approval",
                )
            ).body.decode("utf-8")
            self.assertIn("此次量測會執行的實驗", previewed)
            self.assertIn(f"APPROVE {proposal_id}", previewed)
            self.assertIn('name="agree" value="yes" required', previewed)
            self.assertIn('value="multiplex" required checked', previewed)
            self.assertEqual(service.proposal(proposal_id)["status"], "pending")
            digest = re.search(r'name="preview_digest" value="([0-9a-f]+)"', previewed).group(1)

            without_agree = urlencode(
                {
                    "operation": "approve",
                    "proposal_id": proposal_id,
                    "csrf_token": service.approval_csrf_token(proposal_id),
                    "confirmation": f"APPROVE {proposal_id}",
                    "drive_mode": "multiplex",
                    "preview_digest": digest,
                }
            ).encode("utf-8")
            refused = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{session_id}/approval",
                        {"session_id": session_id},
                        method="POST",
                        body=without_agree,
                    ),
                    service,
                    view="approval",
                )
            )
            self.assertEqual(refused.status_code, 400)
            self.assertEqual(service.proposal(proposal_id)["status"], "pending")
            changed = urlencode(
                {
                    "operation": "approve",
                    "proposal_id": proposal_id,
                    "csrf_token": service.approval_csrf_token(proposal_id),
                    "confirmation": f"APPROVE {proposal_id}",
                    "drive_mode": "per_qubit",
                    "preview_digest": digest,
                    "agree": "yes",
                }
            ).encode("utf-8")
            refused = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{session_id}/approval",
                        {"session_id": session_id},
                        method="POST",
                        body=changed,
                    ),
                    service,
                    view="approval",
                )
            )
            self.assertEqual(refused.status_code, 400)
            self.assertIn("重新預覽", refused.body.decode("utf-8"))

            approval_body = urlencode(
                {
                    "operation": "approve",
                    "proposal_id": proposal_id,
                    "csrf_token": service.approval_csrf_token(proposal_id),
                    "confirmation": f"APPROVE {proposal_id}",
                    # Operator request 2026-10-04: a lease needs the
                    # measurement-mode choice from the approval page.
                    "drive_mode": "multiplex",
                    "preview_digest": digest,
                    "agree": "yes",
                }
            ).encode("utf-8")
            approved = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{session_id}/approval",
                        {"session_id": session_id},
                        method="POST",
                        body=approval_body,
                    ),
                    service,
                    view="approval",
                )
            )
            self.assertEqual(approved.status_code, 200)
            self.assertEqual(service.proposal(proposal_id)["status"], "approved")
            self.assertEqual(
                service.autonomy_status(
                    lease_id=entered["authorization"]["id"]
                )["status"],
                "active",
            )
            self.assertIn(
                "AI agent 會在本次有限授權下自動繼續",
                approved.body.decode("utf-8"),
            )
            waited = service.wait_for_autonomy_status(
                entered["authorization"]["id"], timeout_seconds=0
            )
            self.assertTrue(waited["timed_out"])
            self.assertEqual(waited["authorization"]["status"], "active")

    def test_conversational_entry_returns_shared_dashboard_with_pending_approval(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            entered = service.enter_measurement_mode(
                "unittest-agent", "進入 JY 量測模式", ["q1"]
            )
            session_id = entered["dashboard"]["id"]
            self.assertEqual(entered["browser_url"], "http://127.0.0.1:8766/")
            proposal = service.request_run(
                entered["workflow"]["id"],
                "02x",
                {"qubits": ["q1"]},
                "Run the first conversational experiment.",
                "unittest-agent",
            )
            review = service.dashboard_review(session_id)
            self.assertEqual(review["pending_proposal"]["id"], proposal["id"])

            response = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{session_id}/approval", {"session_id": session_id}
                    ),
                    service,
                    view="approval",
                )
            )
            body = response.body.decode("utf-8")
            self.assertIn("待核准動作", body)
            self.assertIn(f"APPROVE {proposal['id']}", body)
            self.assertIn("/assets/session-dashboard.js", body)
            self.assertIn("/assets/language.js", body)
            self.assertIn("data-language-selector", body)
            self.assertIn("<summary>時間預測</summary>", body)
            self.assertIn("<summary>參數</summary>", body)

            english = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{session_id}/approval",
                        {"session_id": session_id},
                        query="lang=en",
                    ),
                    service,
                    view="approval",
                )
            ).body.decode("utf-8")
            self.assertIn("Action awaiting approval", english)
            self.assertNotIn("待核准動作", english)
            self.assertIn('<option value="en" selected>', english)

            approval_body = urlencode(
                {
                    "operation": "approve",
                    "proposal_id": proposal["id"],
                    "csrf_token": service.approval_csrf_token(proposal["id"]),
                    "confirmation": f"APPROVE {proposal['id']}",
                    "drive_mode": "multiplex",
                }
            ).encode("utf-8")
            approved = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{session_id}/approval",
                        {"session_id": session_id},
                        method="POST",
                        body=approval_body,
                    ),
                    service,
                    view="approval",
                )
            )
            self.assertEqual(approved.status_code, 200)
            self.assertEqual(service.proposal(proposal["id"])["status"], "approved")
            self.assertIn(
                "請回對話視窗輸入「已核准」後便會開始執行實驗",
                approved.body.decode("utf-8"),
            )

    def test_shutdown_then_reentry_starts_a_new_dashboard_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                make_settings(Path(folder), sample_state(0.2, 0.1))
            )
            first = service.enter_measurement_mode(
                "unittest-agent", "進入 JY 量測模式", ["q1"]
            )
            service.leave_measurement_mode(
                first["workflow"]["id"],
                "unittest-agent",
                "退出 JY 量測模式",
            )
            second = service.enter_measurement_mode(
                "unittest-agent", "進入 JY 量測模式", ["q1"]
            )
            self.assertNotEqual(first["dashboard"]["id"], second["dashboard"]["id"])
            self.assertEqual(
                service.dashboard_status(first["dashboard"]["id"])["status"],
                "stopped",
            )
            self.assertEqual(second["dashboard"]["status"], "active")

    def test_replacement_autonomy_lease_reuses_same_session_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service = AgentService(
                replace(
                    make_settings(Path(folder), sample_state(0.2, 0.1)),
                    workflow_sequence=FULL_SEQUENCE,
                )
            )
            workflow = start_test_workflow(service)
            first = service.request_autonomy_lease(
                workflow["id"],
                "unittest-agent",
                AUTO_PHRASE,
                "Exercise a direct MCP lease request.",
            )
            first_session_id = first["dashboard"]["id"]
            first_lease_id = first["autonomy"]["id"]
            self.assertEqual(first["browser_url"], "http://127.0.0.1:8766/")
            self.assertEqual(
                first["human_approval"]["browser_url"], first["browser_url"]
            )
            self.assertEqual(first["autonomy"]["control_url"], first["browser_url"])
            approval_body = urlencode(
                {
                    "csrf_token": service.approval_csrf_token(first["id"]),
                    "confirmation": f"APPROVE {first['id']}",
                    "drive_mode": "multiplex",
                }
            ).encode("utf-8")
            legacy_approval = asyncio.run(
                handle_browser_approval(
                    _request(
                        f"/approve/{first['id']}",
                        {"proposal_id": first["id"]},
                        method="POST",
                        body=approval_body,
                    ),
                    service,
                )
            )
            # Operator request 2026-10-10: a lease is approved only on the
            # Dashboard approval page, after its preview.
            self.assertEqual(legacy_approval.status_code, 400)
            self.assertEqual(service.proposal(first["id"])["status"], "pending")
            service.stop_autonomy(
                first_lease_id,
                "unittest-agent",
                "Replace the lease without replacing the service dashboard.",
            )
            second = service.request_autonomy_lease(
                workflow["id"],
                "unittest-agent",
                AUTO_PHRASE,
                "Replace a stopped direct MCP lease.",
            )
            self.assertEqual(second["dashboard"]["id"], first_session_id)
            self.assertNotEqual(second["autonomy"]["id"], first_lease_id)
            self.assertEqual(second["dashboard"]["status"], "active")
            self.assertEqual(
                service.dashboard_status(first_session_id)["autonomy_lease_id"],
                second["autonomy"]["id"],
            )
            self.assertEqual(second["autonomy"]["control_url"], second["browser_url"])

    def test_shared_decay_fit_survives_zero_initial_decay_guess(self) -> None:
        sys.path.insert(0, str(SUPERCONDUCTING_ROOT))
        try:
            from quam_libs.lib import fit as shared_fit
        finally:
            sys.path.pop(0)
        time = np.linspace(0.0, 200.0, 101)
        values = 0.0007 * np.exp(-time / 45.0) - 0.0026
        data = xr.DataArray(
            values.reshape(1, -1),
            dims=("qubit", "idle_time"),
            coords={"qubit": ["q1"], "idle_time": time},
        )
        with patch.object(shared_fit.ca.guess, "exp_decay", return_value=0.0):
            result = shared_fit.fit_decay_exp(data, "idle_time")
        self.assertEqual(result.sizes["fit_vals"], 12)
        decay = float(result.sel(qubit="q1", fit_vals="decay"))
        self.assertTrue(np.isfinite(decay))
        self.assertLess(decay, 0.0)

    def test_approved_lease_url_becomes_live_dashboard_with_result_and_timing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), sample_state(0.2, 0.1))
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            proposal = service.request_autonomy_lease(
                workflow["id"],
                "unittest-agent",
                AUTO_PHRASE,
                "Exercise the live control page.",
            )
            service._approve_pending_proposal(
                proposal["id"], "unit-test-human", "unit_test"
            )
            lease_id = proposal["autonomy_lease_id"]
            dashboard = proposal["dashboard"]

            snapshot = settings.data_root / "snapshot-live"
            snapshot.mkdir()
            plot = snapshot / "result.png"
            plot.write_bytes(
                bytes.fromhex(
                    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                    "0000000d49444154789c6360000000020001e221bc330000000049454e44ae426082"
                )
            )
            run_id = "live-run"
            parameters = {"qubits": ["q1"], "num_averages": 2000}
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, started_at, finished_at, snapshot_id, snapshot_path,
                    analysis_status, analysis_json, autonomy_lease_id
                ) VALUES (?, ?, ?, '05', ?, 'completed', ?, ?, 801, ?, 'pass', ?, ?)
                """,
                (
                    run_id,
                    workflow["id"],
                    proposal["id"],
                    json_dumps(parameters),
                    "2026-08-19T10:00:00+00:00",
                    "2026-08-19T10:00:10+00:00",
                    str(snapshot),
                    json_dumps(
                        {
                            "analysis_status": "pass",
                            "plots": [str(plot)],
                            "failure_reasons": [],
                            "warnings": [],
                            "fit_quality": {"results": {}},
                            "dataset_metrics": {
                                "qubits": {"q1": {"robust_snr": 12.0}}
                            },
                        }
                    ),
                    lease_id,
                ),
            )

            second_plot = snapshot / "result-2.png"
            second_plot.write_bytes(plot.read_bytes())
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, started_at, finished_at, snapshot_id, snapshot_path,
                    analysis_status, analysis_json, autonomy_lease_id
                ) VALUES ('live-run-2', ?, ?, '05', ?, 'completed', ?, ?, 802,
                          ?, 'pass', ?, ?)
                """,
                (
                    workflow["id"],
                    proposal["id"],
                    json_dumps({"qubits": ["q1"], "num_averages": 4000}),
                    "2026-08-19T10:01:00+00:00",
                    "2026-08-19T10:01:10+00:00",
                    str(snapshot),
                    json_dumps(
                        {
                            "analysis_status": "pass",
                            "plots": [str(second_plot)],
                            "failure_reasons": [],
                            "warnings": [],
                            "fit_quality": {"results": {}},
                            "dataset_metrics": {
                                "qubits": {"q1": {"robust_snr": 12.0}}
                            },
                        }
                    ),
                    lease_id,
                ),
            )
            service.db.execute(
                """
                INSERT INTO decisions(
                    id, workflow_id, run_id, decision, reason, next_node,
                    next_parameters_json, state_patch_json, client_id, created_at
                ) VALUES ('live-decision-2', ?, 'live-run-2', 'repeat', ?, '05',
                          '{}', '[]', 'unittest-agent',
                          '2026-08-19T10:01:11+00:00')
                """,
                (workflow["id"], "Second result remains in the same authorization."),
            )
            service.db.execute(
                """
                INSERT INTO decisions(
                    id, workflow_id, run_id, decision, reason, next_node,
                    next_parameters_json, state_patch_json, client_id, created_at
                ) VALUES ('live-decision', ?, ?, 'repeat', ?, '05', ?, '[]',
                          'unittest-agent', '2026-08-19T10:00:11+00:00')
                """,
                (
                    workflow["id"],
                    run_id,
                    "Increase the wait-time coverage.",
                    json_dumps({"max_wait_time_in_ns": 300000}),
                ),
            )

            response = asyncio.run(
                handle_browser_approval(
                    _request(
                        f"/approve/{proposal['id']}",
                        {"proposal_id": proposal["id"]},
                    ),
                    service,
                )
            )
            self.assertEqual(response.status_code, 303)
            self.assertEqual(
                response.headers["location"],
                f"/session/{dashboard['id']}/approval?lang=zh-Hant",
            )

            legacy_control = asyncio.run(
                handle_autonomy_control(
                    _request(
                        f"/autonomy/{lease_id}",
                        {"lease_id": lease_id},
                    ),
                    service,
                )
            )
            self.assertEqual(legacy_control.status_code, 303)
            self.assertEqual(
                legacy_control.headers["location"],
                f"/session/{dashboard['id']}/home?lang=zh-Hant",
            )

            approval_response = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{dashboard['id']}/approval",
                        {"session_id": dashboard["id"]},
                    ),
                    service,
                    view="approval",
                )
            )
            self.assertIn("一次核准", approval_response.body.decode("utf-8"))

            dashboard_response = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{dashboard['id']}/results",
                        {"session_id": dashboard["id"]},
                    ),
                    service,
                    view="results",
                )
            )
            dashboard_body = dashboard_response.body.decode("utf-8")
            self.assertNotIn("有限自動量測授權", dashboard_body)
            self.assertIn("data-experiment-selector", dashboard_body)
            self.assertIn('data-experiment-panel="1"', dashboard_body)
            self.assertIn('data-experiment-panel="2"', dashboard_body)
            self.assertIn("實驗 1", dashboard_body)
            self.assertIn("實驗 2", dashboard_body)
            for summary in ("時間預測", "參數", "分析", "下一步"):
                self.assertIn(f"<summary>{summary}</summary>", dashboard_body)
            self.assertIn("Increase the wait-time coverage.", dashboard_body)
            self.assertIn("Second result remains in the same authorization.", dashboard_body)
            self.assertIn("num_averages", dashboard_body)

            english_dashboard = asyncio.run(
                handle_session_dashboard(
                    _request(
                        f"/session/{dashboard['id']}/results",
                        {"session_id": dashboard["id"]},
                        query="lang=en",
                    ),
                    service,
                    view="results",
                )
            ).body.decode("utf-8")
            self.assertIn("Experiment 1 / 2", english_dashboard)
            for summary in ("Prediction", "Parameters", "Analysis", "Next action"):
                self.assertIn(f"<summary>{summary}</summary>", english_dashboard)
            self.assertNotIn("<summary>參數</summary>", english_dashboard)

            asset = asyncio.run(
                handle_autonomy_control_asset(
                    _request(
                        f"/autonomy/{lease_id}/assets/0",
                        {"lease_id": lease_id, "asset_index": 0},
                    ),
                    service,
                )
            )
            self.assertEqual(Path(asset.path).resolve(), plot.resolve())
            second_asset = asyncio.run(
                handle_autonomy_control_asset(
                    _request(
                        f"/autonomy/{lease_id}/assets/1",
                        {"lease_id": lease_id, "asset_index": 1},
                    ),
                    service,
                )
            )
            self.assertEqual(Path(second_asset.path).resolve(), second_plot.resolve())

            telemetry = service.run_telemetry("05", ["q1"])
            self.assertEqual(telemetry["prediction"]["median_seconds"], 10.0)
            experience = service.decision_experience("05", ["q1"])
            self.assertEqual(experience[0]["Decision"], "repeat")
            self.assertEqual(experience[0]["elapsed_seconds"], 10.0)
            retry = service.snr_retry_recommendation("live-run-2")
            # Operator instruction 2026-10-20: 05 is capped at 1000 averages and
            # a run too noisy there has failed, so no higher-averaging retry.
            self.assertFalse(retry["retry_recommended"])

            revision = service.autonomy_ui_revision(lease_id)
            service.db.event(
                "run_analyzed",
                "unittest-analyzer",
                {"run_id": "live-run-2", "analysis_status": "needs_review"},
                workflow["id"],
            )
            event_update = service.autonomy_ui_events_after(lease_id, revision)
            self.assertEqual(
                [item["event_type"] for item in event_update["events"]],
                ["run_analyzed"],
            )

    def test_sequence_and_reset_defaults_include_06b_after_06(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = replace(
                make_settings(Path(folder), sample_state(0.2, 0.1)),
                workflow_sequence=FULL_SEQUENCE,
            )
            policy = PolicyEngine(settings)

            self.assertEqual(
                FULL_SEQUENCE[FULL_SEQUENCE.index("06") + 1],
                "06b",
            )
            for node_id in ("04", "05", "07b", "06", "06b", "10a", "05st", "06st_t2star", "06st_t2e"):
                defaults = policy.node_definition(node_id)["defaults"]
                reset = defaults.get(
                    "reset_type",
                    defaults.get("reset_type_thermal_or_active"),
                )
                self.assertEqual(reset, "thermal", node_id)

    def test_statistics_policy_and_four_lifetime_prerequisite(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = replace(
                make_settings(Path(folder), sample_state(0.2, 0.1)),
                workflow_sequence=FULL_SEQUENCE,
            )
            policy = PolicyEngine(settings)
            self.assertEqual(
                policy.node_script("05st").parent.relative_to(
                    settings.superconducting_root
                ),
                Path("side_project/StatisticsMustDo"),
            )
            with self.assertRaisesRegex(PolicyError, "exactly histo_num=100"):
                policy.validate_run(
                    "05st", {"qubits": ["q1"], "histo_num": 99}
                )

            service = AgentService(settings)
            # Statistics prerequisites only; the 2026-10-12 07b readout gate is
            # covered in test_per_qubit_pass_rules.
            service.policy.shared_xy_drive["statistics_readout_node"] = None
            workflow = start_test_workflow(service)
            service.db.execute(
                "UPDATE workflows SET current_node = '05st' WHERE id = ?",
                (workflow["id"],),
            )
            seed_proposal = service._create_proposal(
                workflow["id"], "run", {}, "unittest-agent", 60, 1
            )

            fits = {
                "05": {
                    "t1_seconds": 50e-6,
                    "relative_uncertainty": 0.1,
                    "r_squared": 0.99,
                    "coverage_lifetimes": 4.0,
                    "samples_per_lifetime": 10.0,
                    "fit_at_search_boundary": False,
                },
                "06": {
                    "fit_successful": True,
                    "coherence_seconds": 40e-6,
                    "relative_uncertainty": 0.1,
                    "coverage_lifetimes": 4.0,
                    "samples_per_lifetime": 10.0,
                },
                "06b": {
                    "fit_successful": True,
                    "coherence_seconds": 45e-6,
                    "relative_uncertainty": 0.1,
                    "r_squared": 0.99,
                    "coverage_lifetimes": 4.0,
                    "samples_per_lifetime": 10.0,
                    "fit_at_search_boundary": False,
                },
            }

            def seed_baseline(node_id: str, reset_type: str = "thermal") -> None:
                run_id = f"accepted-{node_id}-{reset_type}"
                analysis = {
                    "fit_quality": {"results": {"q1": fits[node_id]}},
                    "dataset_metrics": {
                        "qubits": {"q1": {"robust_snr": 30.0}}
                    },
                }
                service.db.execute(
                    """
                    INSERT INTO runs(
                        id, workflow_id, proposal_id, node_id, parameters_json,
                        status, started_at, analysis_status, analysis_json
                    ) VALUES (?, ?, ?, ?, ?, 'completed',
                              '2026-08-19T10:00:00+00:00', 'pass', ?)
                    """,
                    (
                        run_id,
                        workflow["id"],
                        seed_proposal["id"],
                        node_id,
                        json_dumps(
                            {
                                "qubits": ["q1"],
                                "reset_type": reset_type,
                            }
                        ),
                        json_dumps(analysis),
                    ),
                )
                service.db.execute(
                    """
                    INSERT INTO decisions(
                        id, workflow_id, run_id, decision, reason, next_node,
                        next_parameters_json, state_patch_json, client_id, created_at
                    ) VALUES (?, ?, ?, 'advance', 'accepted baseline', NULL,
                              '{}', '[]', 'unittest-agent',
                              '2026-08-19T10:01:00+00:00')
                    """,
                    (f"decision-{node_id}-{reset_type}", workflow["id"], run_id),
                )

            for baseline in ("05", "06", "06b"):
                seed_baseline(baseline)

            accepted = service.request_run(
                workflow["id"],
                "05st",
                {"qubits": ["q1"], "max_wait_time_in_ns": 200000},
                "Run one 100-repetition T1 statistics experiment.",
                "unittest-agent",
            )
            self.assertEqual(accepted["payload"]["parameters"]["histo_num"], 100)
            self.assertEqual(
                accepted["payload"]["parameters"]["reset_type"], "thermal"
            )
            with self.assertRaisesRegex(ServiceError, "about four times"):
                service.request_run(
                    workflow["id"],
                    "05st",
                    {"qubits": ["q1"], "max_wait_time_in_ns": 100000},
                    "Reject an under-covered statistics range.",
                    "unittest-agent",
                )

    def test_active_reset_requires_qualified_07b_and_all_active_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = replace(
                make_settings(Path(folder), sample_state(0.2, 0.1)),
                workflow_sequence=FULL_SEQUENCE,
            )
            service = AgentService(settings)
            workflow = start_test_workflow(service)
            service.db.execute(
                "UPDATE workflows SET current_node = '06' WHERE id = ?",
                (workflow["id"],),
            )

            with self.assertRaisesRegex(ServiceError, "Active reset requires"):
                service.request_run(
                    workflow["id"],
                    "06",
                    {"qubits": ["q1"], "reset_type": "active"},
                    "Active reset is not yet qualified.",
                    "unittest-agent",
                )

            seed_proposal = service._create_proposal(
                workflow["id"], "run", {}, "unittest-agent", 60, 1
            )
            active_blob_analysis = {
                "fit_quality": {
                    "results": {
                        "q1": {
                            "fit_successful": True,
                            "readout_fidelity": 0.86,
                            "morphology_pass": True,
                            "active_reset_qualified": True,
                        }
                    }
                }
            }
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, started_at, analysis_status, analysis_json
                ) VALUES ('qualified-active-07b', ?, ?, '07b', ?, 'completed',
                          '2026-08-19T10:00:00+00:00', 'pass', ?)
                """,
                (
                    workflow["id"],
                    seed_proposal["id"],
                    json_dumps(
                        {
                            "qubits": ["q1"],
                            "reset_type_thermal_or_active": "active",
                        }
                    ),
                    json_dumps(active_blob_analysis),
                ),
            )
            service.db.execute(
                """
                INSERT INTO decisions(
                    id, workflow_id, run_id, decision, reason, next_node,
                    next_parameters_json, state_patch_json, client_id, created_at
                ) VALUES ('qualified-active-07b-decision', ?, 'qualified-active-07b',
                          'advance', 'active reset passed', '06', '{}', '[]',
                          'unittest-agent', '2026-08-19T10:01:00+00:00')
                """,
                (workflow["id"],),
            )

            verification = service.request_run(
                workflow["id"],
                "05",
                {"qubits": ["q1"], "reset_type": "active"},
                "Revalidate T1 with qualified active reset before statistics.",
                "unittest-agent",
            )
            self.assertTrue(
                verification["payload"]["prerequisite_verification"]
            )
            service.db.execute(
                """
                INSERT INTO runs(
                    id, workflow_id, proposal_id, node_id, parameters_json,
                    status, started_at, analysis_status, analysis_json
                ) VALUES ('active-05-verification', ?, ?, '05', ?, 'completed',
                          '2026-08-19T10:02:00+00:00', 'pass', ?)
                """,
                (
                    workflow["id"],
                    verification["id"],
                    json_dumps(verification["payload"]["parameters"]),
                    json_dumps(
                        {
                            "fit_quality": {
                                "results": {
                                    "q1": {
                                        "t1_seconds": 50e-6,
                                        "relative_uncertainty": 0.1,
                                        "r_squared": 0.99,
                                        "coverage_lifetimes": 4.0,
                                        "samples_per_lifetime": 10.0,
                                        "fit_at_search_boundary": False,
                                    }
                                }
                            },
                            "dataset_metrics": {
                                "qubits": {"q1": {"robust_snr": 30.0}}
                            },
                        }
                    ),
                ),
            )
            decision = service.record_decision(
                workflow["id"],
                "active-05-verification",
                "advance",
                "The active-reset T1 prerequisite is usable.",
                "06",
                {},
                [],
                "unittest-agent",
            )
            self.assertEqual(decision["mode"], "prerequisite_verification")
            self.assertEqual(
                service.status(workflow["id"])["workflow"]["current_node"], "06"
            )

            service.db.execute(
                "UPDATE workflows SET current_node = '05st' WHERE id = ?",
                (workflow["id"],),
            )
            with self.assertRaisesRegex(ServiceError, "repeated 05/06/06b"):
                service.request_run(
                    workflow["id"],
                    "05st",
                    {
                        "qubits": ["q1"],
                        "reset_type": "active",
                        "max_wait_time_in_ns": 200000,
                    },
                    "Do not start statistics before all active baselines pass.",
                    "unittest-agent",
                )

if __name__ == "__main__":
    unittest.main()
