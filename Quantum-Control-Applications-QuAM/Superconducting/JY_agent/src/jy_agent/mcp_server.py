from __future__ import annotations

import os
from typing import Any, Callable

from mcp.server import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .approval_web import (
    handle_autonomy_control,
    handle_autonomy_control_asset,
    handle_autonomy_events,
    handle_autonomy_events_script,
    handle_browser_approval,
    handle_browser_approval_asset,
)
from .config import Settings
from .dashboard_access import DashboardAccessManager
from .operator_web import handle_operator_console
from .reports import lightweight_report
from .service import AgentService, ServiceError
from .util import json_loads


settings = Settings.load()

INSTRUCTIONS = f"""
JY is a policy-gated superconducting-qubit measurement server with two explicit
modes. Any of {list(settings.measurement_mode_entry_phrases)!r} starts conversational mode: run
one policy-registered experiment at a time with `jy_request_conversational_run`
and a human approval for each run and state commit on the shared session
dashboard. Any of {list(settings.autonomy_mode_entry_phrases)!r} starts bounded automation:
call jy_enter_autonomy_mode, show its session Dashboard URL, wait for the one
lease approval, keep the automatic agent task alive with
jy_wait_for_autonomy_status while no action is ready, and then use only
jy_autonomy_* execution tools. The automation lease
enforces target/node scope, an expiry (eight hours by default; pass
duration_hours at entry only when the operator asks for a different budget),
twenty attempts per node/qubit, and pass-backed state commits.
Call jy_get_next_action for the deterministic plan -- current node, per-target
resolved/incomplete/unresolved sets, attempts left, required setup tools, and
either the parameters to change or a note that no registered rule covers the
situation. Low confidence and
manual_review do not halt the lease, but they do not bypass evidence requirements.
A `needs_review` run still commits its `passing_targets`; nothing else.
The lease survives everything correctable. A request refused before the worker
reaches hardware -- out-of-policy parameter, out-of-scope targets, a premature
analyze -- is recorded and refused, not halted; fix it and continue. A worker
crash matching a registered signature in autonomy.recoverable_worker_exceptions
keeps the lease and tells you the remedy; any other worker crash pauses the
lease for the operator and resumes with the resume phrase. Only snapshot_missing,
state_hash_conflict, and hardware_lock_anomaly end the lease.
Recognized instrument-connectivity failures pause the experiment and scheduling
instead of becoming a generic crash. Preserve the same workflow/session, wait for
the operator to repair connectivity, then use jy_resume_measurement_mode with one
of {list(settings.measurement_resume_phrases)!r}. The interrupted Python call is
not continued mid-stack; the current node is run again as a new run. If entry
fails because stale JY lifecycle state remains, instruct the
operator to issue `恢復` or `Recover`; the repository-root lifecycle wrapper handles
that recovery outside MCP and never auto-deletes a retained hardware lock.
Any configured exit, stop, or shutdown phrase (including
{settings.measurement_mode_exit_phrase!r} and
{settings.measurement_mode_shutdown_phrase!r}) means full service shutdown: stop
any active worker safely, permanently stop the workflow, then run
`stop_server.ps1` from the JY_agent directory so MCP, Approval HTTP, and the
public tunnel all stop. This risk-reducing shutdown requires no new approval.
Outside measurement mode, do not call experiment or state-changing tools. Always
read jy://playbook and use deterministic snapshot evidence.
For every state-changing tool that accepts operation_id, generate one stable UUID
before the first call and reuse that same operation_id only when retrying the exact
same request after a client timeout.
Never modify calibration_graph/ or Script/.
On either entry phrase, call the matching `jy_enter_*` tool with `client_id` and
`activation_phrase` only unless the operator explicitly overrides them. Omit
`targets` and `multiplexed`: a new workflow uses `state.json` `active_qubit_names`
and defaults `multiplexed=true`. Explicit `targets` or `multiplexed=false` remain
optional overrides.
For either mode, surface only the entry result's top-level `browser_url` to the
operator. Approval and control URL fields are protocol aliases for that same
session Dashboard and must not be presented as additional websites.
Keep the final experiment report to result plot(s), Decision, Reason, and Next action
(next node plus new parameters), with only two or three short sentences.
When ending a turn while still in measurement mode, copy `operator_handoff.chat`
verbatim as a three-item markdown list. Do not join the items into one
paragraph. The operator must see separate lines for (1) 需要使用者做什麼,
(2) 完成後回傳, and (3) 如需結束量測，請於網頁首頁結束量測後再對話輸入「結束量測」。
Successful entry shows only the top-level `browser_url`. After the operator
sends `已核准` / `Approved`, the first line of the next reply is exactly `已核准`.
A live turn may execute `結束量測` directly; a paused or disconnected turn needs
Dashboard Home shutdown first, then `結束量測` to verify.
""".strip()

service = AgentService(settings)
dashboard_access = DashboardAccessManager(service.db)
mcp = MCPServer(
    "JY superconducting-qubit bring-up",
    version="0.1.0",
    instructions=INSTRUCTIONS,
)


def _mutation(
    operation: str,
    operation_id: str | None,
    request: dict[str, Any],
    callback: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    return service.idempotent_call(operation, operation_id, request, callback)


@mcp.custom_route(
    "/healthz",
    methods=["GET"],
    name="healthz",
    include_in_schema=False,
)
async def healthz(request: Request) -> Response:
    """Identify the managed JY listener for idempotent local bootstrapping."""
    status = service.status()
    return JSONResponse(
        {
            "service": "jy-mcp",
            "pid": os.getpid(),
            "instance_nonce": os.getenv("JY_SERVICE_INSTANCE_NONCE", ""),
            "remote_approval_enabled": status["server"][
                "remote_approval_enabled"
            ],
            "approval_endpoint": status["server"]["approval_endpoint"],
            "active_run": status["active_run"] is not None,
            "recovery_only": status["server"]["recovery_only"],
        }
    )


@mcp.custom_route(
    "/operator",
    methods=["GET", "POST"],
    name="operator_console",
    include_in_schema=False,
)
async def operator_console(request: Request) -> Response:
    """Local-only shutdown and retained-lock recovery console."""
    return await handle_operator_console(request, service, dashboard_access)


@mcp.custom_route(
    "/approve/{proposal_id}",
    methods=["GET", "POST"],
    name="browser_approval",
    include_in_schema=False,
)
async def browser_approval(request: Request) -> Response:
    """Backward-compatible localhost approval page on the MCP listener."""
    return await handle_browser_approval(request, service)


@mcp.custom_route(
    "/approve/{proposal_id}/assets/{asset_index}",
    methods=["GET"],
    name="browser_approval_asset",
    include_in_schema=False,
)
async def browser_approval_asset(request: Request) -> Response:
    """Serve a proposal-bound result image on the legacy local route."""
    return await handle_browser_approval_asset(request, service)


@mcp.custom_route(
    "/autonomy/{lease_id}",
    methods=["GET", "POST"],
    name="autonomy_control",
    include_in_schema=False,
)
async def autonomy_control(request: Request) -> Response:
    """Backward-compatible local autonomous measurement control page."""
    return await handle_autonomy_control(request, service)


@mcp.custom_route(
    "/autonomy/{lease_id}/assets/{asset_index}",
    methods=["GET"],
    name="autonomy_control_asset",
    include_in_schema=False,
)
async def autonomy_control_asset(request: Request) -> Response:
    """Serve the latest lease-bound result image on the control page."""
    return await handle_autonomy_control_asset(request, service)


@mcp.custom_route(
    "/autonomy/{lease_id}/events",
    methods=["GET"],
    name="autonomy_events",
    include_in_schema=False,
)
async def autonomy_events(request: Request) -> Response:
    """Notify the compatibility control page when result history changes."""
    return await handle_autonomy_events(request, service)


@mcp.custom_route(
    "/assets/autonomy-events.js",
    methods=["GET"],
    name="autonomy_events_script",
    include_in_schema=False,
)
async def autonomy_events_script(request: Request) -> Response:
    """Serve the event-driven compatibility-page client."""
    return await handle_autonomy_events_script(request)


@mcp.resource("jy://playbook")
def playbook() -> str:
    """Canonical bring-up workflow and node-specific decision guidance."""
    return settings.playbook_path.read_text(encoding="utf-8")


@mcp.resource("jy://policies")
def policies() -> str:
    """Server-enforced parameter, state, approval, and hardware limits."""
    return settings.policy_path.read_text(encoding="utf-8")


@mcp.prompt()
def begin_jy_bringup(client_id: str) -> str:
    """Start a policy-gated JY bring-up session."""
    return (
        f"Wait until the user says one of "
        f"{list(settings.measurement_mode_entry_phrases)!r}. Then read jy://playbook and "
        "call jy_enter_measurement_mode with "
        f"client_id={client_id!r} only unless the operator explicitly overrides "
        "targets or multiplexed. A new workflow uses state.json "
        "active_qubit_names and defaults multiplexed=true. Preserve and show the "
        "returned session browser URL. Call jy_list_experiments, then discuss and "
        "propose one experiment at a time; each run/state approval appears on that "
        "page."
    )


@mcp.prompt()
def begin_jy_autonomy(client_id: str) -> str:
    """Start the bounded automatic bring-up mode."""
    return (
        f"Wait until the user says one of {list(settings.autonomy_mode_entry_phrases)!r}. "
        "Read jy://playbook and jy://policies, then call jy_enter_autonomy_mode "
        f"with client_id={client_id!r} only unless the operator explicitly overrides "
        "targets or multiplexed. A new workflow uses state.json "
        "active_qubit_names and defaults multiplexed=true. Show its session "
        "browser URL and wait for the one lease approval with "
        "jy_wait_for_autonomy_status. Use only jy_autonomy_* execution tools within "
        "scope and wait on new events between actions."
    )


@mcp.tool()
def jy_enter_measurement_mode(
    client_id: str,
    activation_phrase: str,
    targets: list[str] | None = None,
    multiplexed: bool | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Enter conversational mode for one human-approved experiment at a time.

    Omit targets to use state.json active_qubit_names. Omit multiplexed to
    default true for a new workflow; omitted multiplexed does not change an
    existing workflow.
    """
    request = {
        "client_id": client_id,
        "activation_phrase": activation_phrase,
        "targets": targets,
        "multiplexed": multiplexed,
    }
    return _mutation(
        "jy_enter_measurement_mode",
        operation_id,
        request,
        lambda: service.enter_measurement_mode(
            client_id, activation_phrase, targets, multiplexed=multiplexed
        ),
    )


@mcp.tool()
def jy_enter_autonomy_mode(
    client_id: str,
    activation_phrase: str,
    targets: list[str] | None = None,
    multiplexed: bool | None = None,
    duration_hours: float | None = None,
    reason: str = "Operator entered JY autonomous measurement mode.",
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Enter bounded automatic mode and return its one lease approval page.

    Omit targets to use state.json active_qubit_names. Omit multiplexed to
    default true for a new workflow; omitted multiplexed does not change an
    existing workflow. Pass duration_hours only when the operator asks for a
    different budget than the configured default; it applies to a new lease
    and the human still approves the number shown on the Dashboard.
    """
    request = {
        "client_id": client_id,
        "activation_phrase": activation_phrase,
        "targets": targets,
        "multiplexed": multiplexed,
        "duration_hours": duration_hours,
        "reason": reason,
    }
    return _mutation(
        "jy_enter_autonomy_mode",
        operation_id,
        request,
        lambda: service.enter_autonomy_mode(
            client_id,
            activation_phrase,
            targets,
            multiplexed=multiplexed,
            duration_hours=duration_hours,
            reason=reason,
        ),
    )


@mcp.tool()
def jy_get_status(workflow_id: str | None = None) -> dict[str, Any]:
    """Get server, hardware-lock, workflow, run, and analysis state."""
    return service.status(workflow_id)


@mcp.tool()
def jy_list_experiments() -> dict[str, Any]:
    """List conversationally runnable nodes and scripts needing policy onboarding."""
    return service.experiment_catalog()


@mcp.tool()
def jy_get_run_telemetry(
    node_id: str | None = None,
    targets: list[str] | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Get complete run parameters, elapsed times, and a robust duration forecast."""
    return service.run_telemetry(node_id=node_id, targets=targets, limit=limit)


@mcp.tool()
def jy_get_decision_experience(
    node_id: str | None = None,
    targets: list[str] | None = None,
    limit: int = 10,
    include_analysis: bool = False,
) -> list[dict[str, Any]]:
    """Get prior per-run Decision/Reason/Next records with evidence and timing.

    Each entry carries `evidence`, a per-qubit digest of the gated numbers.
    Set include_analysis only when the digest is not enough; the full analysis
    document per entry is large.
    """
    return service.decision_experience(
        node_id=node_id,
        targets=targets,
        limit=limit,
        include_analysis=include_analysis,
    )


@mcp.tool()
def jy_get_next_action(workflow_id: str) -> dict[str, Any]:
    """Say what this workflow needs next: wait, analyze, decide, setup, run, or advance.

    Deterministic aggregation of the same state the guards enforce: current
    node, per-target resolved/incomplete/unresolved sets, attempts left, the
    required deterministic setup tools, the shared-first-batch rule, and the
    configured averaging ladder. `run.parameters` carries only the values that
    differ from the node defaults. When `notes` says no registered rule covers
    the situation, read the cited playbook section instead of guessing.
    """
    return service.next_action(workflow_id)


@mcp.tool()
def jy_recommend_0405_snr_retry(run_id: str) -> dict[str, Any]:
    """Recommend the next bounded num_averages step for noisy 04/05 targets."""
    return service.snr_retry_recommendation(run_id)


@mcp.tool()
def jy_request_autonomy_lease(
    workflow_id: str,
    client_id: str,
    activation_phrase: str,
    reason: str,
    targets: list[str] | None = None,
    allowed_nodes: list[str] | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Create one bounded authorization and bind its canonical session Dashboard."""
    request = {
        "workflow_id": workflow_id,
        "client_id": client_id,
        "activation_phrase": activation_phrase,
        "reason": reason,
        "targets": targets,
        "allowed_nodes": allowed_nodes,
    }
    return _mutation(
        "jy_request_autonomy_lease",
        operation_id,
        request,
        lambda: service.request_autonomy_lease(
            workflow_id,
            client_id,
            activation_phrase,
            reason,
            targets,
            allowed_nodes,
        ),
    )


@mcp.tool()
def jy_get_autonomy_status(
    lease_id: str | None = None,
    workflow_id: str | None = None,
) -> dict[str, Any]:
    """Get authorization scope, expiry, counters, stop reason, and control URL."""
    return service.autonomy_status(lease_id=lease_id, workflow_id=workflow_id)


@mcp.tool()
def jy_wait_for_autonomy_status(
    lease_id: str,
    after_event_id: int = 0,
    timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    """Wait up to 55 seconds for lease approval, a new result, or a control event."""
    return service.wait_for_autonomy_status(
        lease_id,
        after_event_id=after_event_id,
        timeout_seconds=timeout_seconds,
    )


@mcp.tool()
def jy_mark_scientifically_unmeasurable(
    workflow_id: str,
    lease_id: str,
    run_id: str,
    targets: list[str],
    reason: str,
    client_id: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Mark reviewed non-passing targets incomplete at a scientific boundary."""
    request = {
        "workflow_id": workflow_id,
        "lease_id": lease_id,
        "run_id": run_id,
        "targets": targets,
        "reason": reason,
        "client_id": client_id,
    }
    return _mutation(
        "jy_mark_scientifically_unmeasurable",
        operation_id,
        request,
        lambda: service.mark_scientifically_unmeasurable(
            workflow_id, lease_id, run_id, targets, reason, client_id
        ),
    )


@mcp.tool()
def jy_pause_autonomy(
    lease_id: str,
    client_id: str,
    reason: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Pause scheduling after the active run; the current worker is not stopped."""
    request = {"lease_id": lease_id, "client_id": client_id, "reason": reason}
    return _mutation(
        "jy_pause_autonomy",
        operation_id,
        request,
        lambda: service.pause_autonomy(lease_id, client_id, reason),
    )


@mcp.tool()
def jy_resume_autonomy(
    lease_id: str,
    client_id: str,
    activation_phrase: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Resume an unexpired paused lease using the exact autonomy phrase."""
    request = {
        "lease_id": lease_id,
        "client_id": client_id,
        "activation_phrase": activation_phrase,
    }
    return _mutation(
        "jy_resume_autonomy",
        operation_id,
        request,
        lambda: service.resume_autonomy(lease_id, client_id, activation_phrase),
    )


@mcp.tool()
def jy_stop_autonomy(
    lease_id: str,
    client_id: str,
    reason: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Revoke the lease and request a graceful stop of its active worker."""
    request = {"lease_id": lease_id, "client_id": client_id, "reason": reason}
    return _mutation(
        "jy_stop_autonomy",
        operation_id,
        request,
        lambda: service.stop_autonomy(lease_id, client_id, reason),
    )


@mcp.tool()
def jy_emergency_stop_autonomy(
    lease_id: str,
    client_id: str,
    reason: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Revoke and force-stop the exact worker after the configured grace period."""
    request = {"lease_id": lease_id, "client_id": client_id, "reason": reason}
    return _mutation(
        "jy_emergency_stop_autonomy",
        operation_id,
        request,
        lambda: service.stop_autonomy(lease_id, client_id, reason, emergency=True),
    )


@mcp.tool()
def jy_start_workflow(
    targets: list[str],
    client_id: str,
    activation_phrase: str,
    initial_parameters: dict[str, Any] | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Create a workflow for explicit qubits. Prefer jy_enter_* for conversation.

    Phrase-only jy_enter_* already uses state.json active_qubit_names and
    defaults multiplexed=true. Set initial_parameters to {"multiplexed": false}
    only when a non-multiplexed workflow is required.
    """
    request = {
        "targets": targets,
        "initial_parameters": initial_parameters,
        "client_id": client_id,
        "activation_phrase": activation_phrase,
    }
    return _mutation(
        "jy_start_workflow",
        operation_id,
        request,
        lambda: service.start_workflow(
            targets, initial_parameters, client_id, activation_phrase
        ),
    )


@mcp.tool()
def jy_leave_measurement_mode(
    workflow_id: str,
    client_id: str,
    exit_phrase: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Permanently stop a workflow before the host shuts down all JY services."""
    request = {
        "workflow_id": workflow_id,
        "client_id": client_id,
        "exit_phrase": exit_phrase,
    }
    return _mutation(
        "jy_leave_measurement_mode",
        operation_id,
        request,
        lambda: service.leave_measurement_mode(
            workflow_id, client_id, exit_phrase
        ),
    )


@mcp.tool()
def jy_resume_measurement_mode(
    workflow_id: str,
    client_id: str,
    activation_phrase: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Resume the same paused workflow/session after instrument inspection."""
    request = {
        "workflow_id": workflow_id,
        "client_id": client_id,
        "activation_phrase": activation_phrase,
    }
    return _mutation(
        "jy_resume_measurement_mode",
        operation_id,
        request,
        lambda: service.resume_measurement_mode(
            workflow_id, client_id, activation_phrase
        ),
    )


@mcp.tool()
def jy_reopen_07b_after_morphology_rule_change(
    workflow_id: str,
    run_id: str,
    reason: str,
    client_id: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Backtrack 06 to 07b when reanalysis rejects long-tail IQ morphology."""
    request = {
        "workflow_id": workflow_id,
        "run_id": run_id,
        "reason": reason,
        "client_id": client_id,
    }
    return _mutation(
        "jy_reopen_07b_after_morphology_rule_change",
        operation_id,
        request,
        lambda: service.reopen_07b_after_morphology_rule_change(
            workflow_id, run_id, reason, client_id
        ),
    )


@mcp.tool()
def jy_reopen_07d_after_readout_rule_change(
    workflow_id: str,
    reason: str,
    client_id: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Return from 07b/06 to 07d after a readout-selection rule change.

    Requires no running node and no active or paused autonomy lease; earlier
    07d/07b runs then no longer count toward single pass.
    """
    request = {"workflow_id": workflow_id, "reason": reason, "client_id": client_id}
    return _mutation(
        "jy_reopen_07d_after_readout_rule_change",
        operation_id,
        request,
        lambda: service.reopen_07d_after_readout_rule_change(
            workflow_id, reason, client_id
        ),
    )


@mcp.tool()
def jy_stop_workflow(
    workflow_id: str,
    client_id: str,
    reason: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Permanently stop an active or paused workflow with no running node."""
    request = {
        "workflow_id": workflow_id,
        "client_id": client_id,
        "reason": reason,
    }
    return _mutation(
        "jy_stop_workflow",
        operation_id,
        request,
        lambda: service.stop_workflow(workflow_id, client_id, reason),
    )


@mcp.tool()
def jy_request_run(
    workflow_id: str,
    node_id: str,
    parameters: dict[str, Any],
    reason: str,
    client_id: str,
    auto_retry_count: int = 0,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Validate parameters and create a human-approved run proposal.

    auto_retry_count is bounded per approval. Separately approved manual retry
    proposals have no lifetime total limit.
    """
    request = {
        "workflow_id": workflow_id,
        "node_id": node_id,
        "parameters": parameters,
        "reason": reason,
        "client_id": client_id,
        "auto_retry_count": auto_retry_count,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_run",
        operation_id,
        request,
        lambda: service.request_run(
            workflow_id,
            node_id,
            parameters,
            reason,
            client_id,
            auto_retry_count,
            autonomy_lease_id,
        ),
    )


@mcp.tool()
def jy_request_conversational_run(
    workflow_id: str,
    node_id: str,
    parameters: dict[str, Any],
    reason: str,
    client_id: str,
    auto_retry_count: int = 0,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Propose one registered experiment with its own human approval.

    Unlike the automatic-sequence tool, node_id may be any policy-registered
    experiment. Targets must remain inside the active conversational workflow.
    """
    request = {
        "workflow_id": workflow_id,
        "node_id": node_id,
        "parameters": parameters,
        "reason": reason,
        "client_id": client_id,
        "auto_retry_count": auto_retry_count,
    }
    return _mutation(
        "jy_request_conversational_run",
        operation_id,
        request,
        lambda: service.request_conversational_run(
            workflow_id,
            node_id,
            parameters,
            reason,
            client_id,
            auto_retry_count,
        ),
    )


@mcp.tool()
def jy_autonomy_start_run(
    workflow_id: str,
    lease_id: str,
    node_id: str,
    parameters: dict[str, Any],
    reason: str,
    client_id: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Validate and launch one run without another approval, within the lease.

    Normally node_id is the current node. After qualified active-reset 07b,
    05/06/06b may also be requested as conditional prerequisite verification
    before active-reset statistics; such runs do not move the current node.
    """
    request = {
        "workflow_id": workflow_id,
        "lease_id": lease_id,
        "node_id": node_id,
        "parameters": parameters,
        "reason": reason,
        "client_id": client_id,
    }
    return _mutation(
        "jy_autonomy_start_run",
        operation_id,
        request,
        lambda: service.start_authorized_run(
            workflow_id, lease_id, node_id, parameters, reason, client_id
        ),
    )


@mcp.tool()
def jy_get_proposal(proposal_id: str) -> dict[str, Any]:
    """Inspect a run or state proposal and its approval/consumption status."""
    return service.proposal(proposal_id)


@mcp.tool()
def jy_execute_run(
    proposal_id: str, client_id: str, operation_id: str | None = None
) -> dict[str, Any]:
    """Start an already human-approved run in the isolated Qualibrate worker."""
    request = {"proposal_id": proposal_id, "client_id": client_id}
    return _mutation(
        "jy_execute_run",
        operation_id,
        request,
        lambda: service.execute_run(proposal_id, client_id),
    )


@mcp.tool()
def jy_poll_run(run_id: str) -> dict[str, Any]:
    """Poll an experiment worker and retrieve its exact snapshot information."""
    return service.poll_run(run_id)


@mcp.tool()
def jy_stop_run(
    run_id: str, client_id: str, operation_id: str | None = None
) -> dict[str, Any]:
    """Request a graceful stop of the exact active worker process."""
    request = {"run_id": run_id, "client_id": client_id}
    return _mutation(
        "jy_stop_run",
        operation_id,
        request,
        lambda: service.stop_run(run_id, client_id),
    )


@mcp.tool()
def jy_analyze_run(
    run_id: str, operation_id: str | None = None
) -> dict[str, Any]:
    """Analyze a completed snapshot without trusting the node's success flag."""
    return _mutation(
        "jy_analyze_run",
        operation_id,
        {"run_id": run_id},
        lambda: service.analyze_run(run_id),
    )


@mcp.tool()
def jy_autonomy_analyze_run(
    run_id: str,
    lease_id: str,
    client_id: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Analyze a lease-bound snapshot and enforce hard-stop handling."""
    request = {"run_id": run_id, "lease_id": lease_id, "client_id": client_id}
    return _mutation(
        "jy_autonomy_analyze_run",
        operation_id,
        request,
        lambda: service.analyze_authorized_run(run_id, lease_id, client_id),
    )


@mcp.tool()
def jy_record_decision(
    workflow_id: str,
    run_id: str,
    decision: str,
    reason: str,
    client_id: str,
    next_node: str | None = None,
    new_parameters: dict[str, Any] | None = None,
    state_patch: list[dict[str, Any]] | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Record advance/repeat/manual_review/stop and next-node parameters."""
    request = {
        "workflow_id": workflow_id,
        "run_id": run_id,
        "decision": decision,
        "reason": reason,
        "next_node": next_node,
        "new_parameters": new_parameters,
        "state_patch": state_patch,
        "client_id": client_id,
    }
    return _mutation(
        "jy_record_decision",
        operation_id,
        request,
        lambda: service.record_decision(
            workflow_id,
            run_id,
            decision,
            reason,
            next_node,
            new_parameters,
            state_patch,
            client_id,
        ),
    )


@mcp.tool()
def jy_request_state_commit(
    workflow_id: str,
    patch: list[dict[str, Any]],
    reason: str,
    client_id: str,
    run_id: str | None = None,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Create a state patch proposal; a valid lease may delegate its approval."""
    request = {
        "workflow_id": workflow_id,
        "patch": patch,
        "reason": reason,
        "client_id": client_id,
        "run_id": run_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_state_commit",
        operation_id,
        request,
        lambda: service.request_state_commit(
            workflow_id,
            patch,
            reason,
            client_id,
            run_id,
            autonomy_lease_id=autonomy_lease_id,
        ),
    )


@mcp.tool()
def jy_autonomy_commit_state(
    workflow_id: str,
    lease_id: str,
    patch: list[dict[str, Any]],
    reason: str,
    client_id: str,
    run_id: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Apply a decision-recorded patch from a passing authorized run."""
    request = {
        "workflow_id": workflow_id,
        "lease_id": lease_id,
        "patch": patch,
        "reason": reason,
        "client_id": client_id,
        "run_id": run_id,
    }
    return _mutation(
        "jy_autonomy_commit_state",
        operation_id,
        request,
        lambda: service.commit_authorized_state(
            workflow_id, lease_id, patch, reason, client_id, run_id
        ),
    )


@mcp.tool()
def jy_autonomy_apply_setup_state(
    proposal_id: str,
    lease_id: str,
    client_id: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Apply a delegated server-generated bootstrap or LO setup patch."""
    request = {
        "proposal_id": proposal_id,
        "lease_id": lease_id,
        "client_id": client_id,
    }
    return _mutation(
        "jy_autonomy_apply_setup_state",
        operation_id,
        request,
        lambda: service.apply_authorized_setup_state(
            proposal_id, lease_id, client_id
        ),
    )


@mcp.tool()
def jy_request_bootstrap(
    workflow_id: str,
    qubits: list[str],
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Propose x180/x90 bootstrap only for missing or zero amplitudes."""
    request = {
        "workflow_id": workflow_id,
        "qubits": qubits,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_bootstrap",
        operation_id,
        request,
        lambda: service.request_bootstrap(
            workflow_id, qubits, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_request_07b_prerequisites(
    workflow_id: str,
    qubits: list[str],
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Propose missing readout-fidelity keys required by 07b state recording."""
    request = {
        "workflow_id": workflow_id,
        "qubits": qubits,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_07b_prerequisites",
        operation_id,
        request,
        lambda: service.request_07b_prerequisites(
            workflow_id, qubits, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_request_07b_tail_power_reduction(
    workflow_id: str,
    qubits: list[str],
    evidence_run_id: str,
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Reduce readout amplitude one fixed step after failed cloud morphology."""
    request = {
        "workflow_id": workflow_id,
        "qubits": qubits,
        "evidence_run_id": evidence_run_id,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_07b_tail_power_reduction",
        operation_id,
        request,
        lambda: service.request_07b_tail_power_reduction(
            workflow_id,
            qubits,
            evidence_run_id,
            client_id,
            autonomy_lease_id,
        ),
    )


@mcp.tool()
def jy_request_drive_lo_recenter(
    workflow_id: str,
    qubits: list[str],
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Propose a 100 MHz-grid LO and residual IF before a wide 03a run."""
    request = {
        "workflow_id": workflow_id,
        "qubits": qubits,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_drive_lo_recenter",
        operation_id,
        request,
        lambda: service.request_drive_lo_recenter(
            workflow_id, qubits, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_request_fresh_qubit_defaults(
    workflow_id: str,
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Empty extras and write starting values for qubits this chip never measured."""
    request = {
        "workflow_id": workflow_id,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_fresh_qubit_defaults",
        operation_id,
        request,
        lambda: service.request_fresh_qubit_defaults(
            workflow_id, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_request_initial_03a_zero_if(
    workflow_id: str,
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Propose all-active-target IF=0 setup before the first 800 MHz 03a run."""
    request = {
        "workflow_id": workflow_id,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_initial_03a_zero_if",
        operation_id,
        request,
        lambda: service.request_initial_03a_zero_if(
            workflow_id, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_request_readout_full_scale_rescale(
    workflow_id: str,
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Lower readout full scales, ignoring line members marked unmeasurable.

    Each measurable member keeps its delivered power; the line's full scale
    becomes the smallest integer within the pulse and line-sum limits. With
    autonomy_lease_id it is delegated; apply it with
    jy_autonomy_apply_setup_state.
    """
    request = {
        "workflow_id": workflow_id,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_readout_full_scale_rescale",
        operation_id,
        request,
        lambda: service.request_readout_full_scale_rescale(
            workflow_id, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_request_readout_length_normalize(
    workflow_id: str,
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Cap readout length and depletion time at the configured values.

    For a new chip whose template readout pulses are long (e.g. 100 us), so
    every multiplexed point waits for the longest one. With autonomy_lease_id
    it is delegated; apply it with jy_autonomy_apply_setup_state.
    """
    request = {
        "workflow_id": workflow_id,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_readout_length_normalize",
        operation_id,
        request,
        lambda: service.request_readout_length_normalize(
            workflow_id, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_request_02x_probe_setup(
    workflow_id: str,
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Give each qubit its own QUA thread and grid cell; raise 02x probe headroom.

    Elements sharing one QUA thread run on one core, so multiplexed readout
    reported one element's response on every channel. 02x plays every line
    tone at a fixed -10 dBm; on a low full scale the tones sum above 1 and
    clip. Shared grid_location values make node plots overlap. Each qubit keeps its delivered readout power. With
    autonomy_lease_id it is delegated; apply it with
    jy_autonomy_apply_setup_state.
    """
    request = {
        "workflow_id": workflow_id,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_02x_probe_setup",
        operation_id,
        request,
        lambda: service.request_02x_probe_setup(
            workflow_id, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_request_xy_lo_plan(
    workflow_id: str,
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Place the shared XY LO for the qubit whose per-qubit pass is running.

    Required before each node after 03a. One 100 MHz-grid LO is shared by
    every measured qubit when it keeps them all within the common IF limit;
    otherwise the qubit uses its own LO and qubits it cannot reach are parked.
    The LO is recorded in extras.xy_lo_freq. With autonomy_lease_id it is
    delegated; apply it with jy_autonomy_apply_setup_state.
    """
    request = {
        "workflow_id": workflow_id,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_xy_lo_plan",
        operation_id,
        request,
        lambda: service.request_xy_lo_plan(workflow_id, client_id, autonomy_lease_id),
    )


@mcp.tool()
def jy_request_04_gate_time(
    workflow_id: str,
    length_ns: int,
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Lengthen the 04 qubit's XY gate when its amplitude sweep is maxed out.

    Use when the Rabi signal changes but has not turned over and the last 04
    swept to the amplitude limit. At most 1000 ns on a shared XY line, 100 ns
    on a private one, in 4 ns steps. With autonomy_lease_id it is delegated;
    apply it with jy_autonomy_apply_setup_state, then rerun 04.
    """
    request = {
        "workflow_id": workflow_id,
        "length_ns": length_ns,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_04_gate_time",
        operation_id,
        request,
        lambda: service.request_04_gate_time(
            workflow_id, length_ns, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_request_node_extras_prerequisites(
    workflow_id: str,
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Add every missing extras key the remaining nodes record (0.0 placeholder).

    Required setup when, e.g., a new chip lacks extras/bare_resonator_freq:
    Qualibrate's state recorder raises KeyError on a missing key. With
    autonomy_lease_id it is delegated; apply it with
    jy_autonomy_apply_setup_state.
    """
    request = {
        "workflow_id": workflow_id,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_node_extras_prerequisites",
        operation_id,
        request,
        lambda: service.request_node_extras_prerequisites(
            workflow_id, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_request_readout_lo_recenter(
    workflow_id: str,
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Move readout LOs so 02 covers every operator design frequency.

    Required setup at 02 when a design frequency lies outside the readout LO
    +- max IF window. Assigned qubits keep their RF; the earlier 02 pass no
    longer counts. With autonomy_lease_id it is delegated; apply it with
    jy_autonomy_apply_setup_state.
    """
    request = {
        "workflow_id": workflow_id,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_readout_lo_recenter",
        operation_id,
        request,
        lambda: service.request_readout_lo_recenter(
            workflow_id, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_request_readout_amplitude_budget(
    workflow_id: str,
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Scale an over-budget readout line's amplitudes down to the target sum.

    Required setup when a line sums above max_aggregate_readout_amplitude
    (e.g. a new chip's template). With autonomy_lease_id it is delegated;
    apply it with jy_autonomy_apply_setup_state.
    """
    request = {
        "workflow_id": workflow_id,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_readout_amplitude_budget",
        operation_id,
        request,
        lambda: service.request_readout_amplitude_budget(
            workflow_id, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_request_02c_window_recenter(
    workflow_id: str,
    qubits: list[str],
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Centre unresolved 02c readout IFs between their dressed and bare lines.

    Use before a narrow 02c retry whose window must be 2-5x the bare-dressed
    separation; the result lists each target's recommended span. With
    autonomy_lease_id the setup is delegated (apply it with
    jy_autonomy_apply_setup_state).
    """
    request = {
        "workflow_id": workflow_id,
        "qubits": qubits,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_02c_window_recenter",
        operation_id,
        request,
        lambda: service.request_02c_window_recenter(
            workflow_id, qubits, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_request_03a_window_shift(
    workflow_id: str,
    lo_centers_in_ghz: dict[str, float],
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Propose 100 MHz-grid, IF=0 LO windows for unresolved 03a targets."""
    request = {
        "workflow_id": workflow_id,
        "lo_centers_in_ghz": lo_centers_in_ghz,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_03a_window_shift",
        operation_id,
        request,
        lambda: service.request_03a_window_shift(
            workflow_id, lo_centers_in_ghz, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_request_03a_candidate_center(
    workflow_id: str,
    qubits: list[str],
    client_id: str,
    autonomy_lease_id: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Center credible unresolved 03a candidates on a safe LO/IF grid."""
    request = {
        "workflow_id": workflow_id,
        "qubits": qubits,
        "client_id": client_id,
        "autonomy_lease_id": autonomy_lease_id,
    }
    return _mutation(
        "jy_request_03a_candidate_center",
        operation_id,
        request,
        lambda: service.request_03a_candidate_center(
            workflow_id, qubits, client_id, autonomy_lease_id
        ),
    )


@mcp.tool()
def jy_apply_state_commit(
    proposal_id: str, client_id: str, operation_id: str | None = None
) -> dict[str, Any]:
    """Atomically apply an already human-approved patch and save a backup."""
    request = {"proposal_id": proposal_id, "client_id": client_id}
    return _mutation(
        "jy_apply_state_commit",
        operation_id,
        request,
        lambda: service.apply_state_commit(proposal_id, client_id),
    )


@mcp.tool()
def jy_render_report(
    run_id: str,
    decision: str,
    reason: str,
    next_node: str | None = None,
    new_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render the lightweight plot/Decision/Reason/Next-action report."""
    run = service.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if run is None:
        raise ServiceError(f"Unknown run: {run_id}")
    analysis = json_loads(run.get("analysis_json"), None)
    if analysis is None:
        raise ServiceError("Analyze the run before rendering its report")
    return {
        "run_id": run_id,
        "report_markdown": lightweight_report(
            analysis, decision, reason, next_node, new_parameters
        ),
        "plots": analysis.get("plots", []),
        "Decision": decision,
        "Reason": reason,
        "next_action": {
            "next_node": next_node,
            "new_parameters": new_parameters or {},
        },
    }


@mcp.tool()
def jy_list_historical_runs(
    node_id: str | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    """List replay candidates from Data without training or changing a model."""
    return service.list_historical_runs(node_id, limit)


def main() -> None:
    service.start_autonomy_watchdog()
    mcp.run(
        transport="streamable-http",
        host=settings.host,
        port=settings.port,
        streamable_http_path=settings.mcp_path,
    )


if __name__ == "__main__":
    main()
