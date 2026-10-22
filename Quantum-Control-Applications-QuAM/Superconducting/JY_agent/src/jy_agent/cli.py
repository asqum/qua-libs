from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .service import AgentService


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _approve(service: "AgentService", proposal_id: str) -> None:
    proposal = service.proposal(proposal_id)
    _print(proposal)
    require_tty = bool(
        service.policy.raw["approval"].get("require_interactive_tty", True)
    )
    if require_tty and not sys.stdin.isatty():
        raise SystemExit(
            "Approval refused: run this command in an interactive human terminal."
        )
    expected = f"APPROVE {proposal_id}"
    answer = input(f"\nType exactly '{expected}' to approve: ").strip()
    if answer != expected:
        raise SystemExit("Approval cancelled.")
    _print(service.approve_interactively(proposal_id))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="JY superconducting-qubit bring-up agent"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create/check the local runtime database")
    sub.add_parser(
        "history-site",
        help="Rebuild the static measurement history site (runtime/history_site)",
    )
    status_parser = sub.add_parser("status", help="Show local workflow status")
    status_parser.add_argument("--workflow-id")
    approve_parser = sub.add_parser(
        "approve", help="Interactively approve one run or state proposal"
    )
    approve_parser.add_argument("proposal_id")
    serve_parser = sub.add_parser(
        "serve", help="Start persistent localhost Streamable HTTP"
    )
    serve_parser.add_argument("--host")
    serve_parser.add_argument("--port", type=int)
    sub.add_parser(
        "stdio", help="Start the protocol-only client-owned STDIO MCP"
    )
    doctor_parser = sub.add_parser(
        "doctor", help="Diagnose project config and perform a real STDIO handshake"
    )
    doctor_parser.add_argument("--repository-root", required=True, type=Path)
    doctor_parser.add_argument("--timeout-seconds", type=float, default=90.0)
    recovery_parser = sub.add_parser(
        "recover-lock",
        help="Locally inspect and recover a retained hardware lock",
    )
    recovery_parser.add_argument("run_id")
    recovery_parser.add_argument("--inspect-only", action="store_true")
    close_parser = sub.add_parser(
        "recover-close",
        help="Converge stale JY workflow state before stopping all services",
    )
    close_parser.add_argument("--timeout-seconds", type=float, default=60.0)
    approval_parser = sub.add_parser(
        "approval-serve", help="Start the review-only Approval HTTP service"
    )
    approval_parser.add_argument("--host")
    approval_parser.add_argument("--port", type=int)
    replay_parser = sub.add_parser(
        "replay", help="Analyze an existing snapshot without hardware"
    )
    replay_parser.add_argument("snapshot", type=Path)
    replay_parser.add_argument("--node", required=True)
    autonomy_status_parser = sub.add_parser(
        "autonomy-status", help="Show one bounded autonomy lease"
    )
    autonomy_status_parser.add_argument("lease_id")
    for command, help_text in (
        ("autonomy-pause", "Pause scheduling after the active run"),
        ("autonomy-stop", "Revoke and gracefully stop the active run"),
        ("autonomy-emergency-stop", "Revoke and force-stop after the grace period"),
    ):
        control_parser = sub.add_parser(command, help=help_text)
        control_parser.add_argument("lease_id")
        control_parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.command not in {"serve", "stdio", "approval-serve"}:
        # A short-lived command exits right away; write the static history
        # site before it does (2026-10-22).
        from .service import AgentService as _Service

        _Service.HISTORY_SITE_ASYNC = False

    if args.command == "doctor":
        from .doctor import run_doctor

        _print(
            asyncio.run(
                run_doctor(args.repository_root, args.timeout_seconds)
            )
        )
        return

    from .config import Settings

    settings = Settings.load()
    if args.command == "recover-lock":
        from .db import Database
        from .recovery import HardwareLockRecovery

        recovery = HardwareLockRecovery(settings, Database(settings.database_path))
        inspection = recovery.inspect(args.run_id)
        _print(inspection)
        if args.inspect_only:
            if not inspection["ready"]:
                raise SystemExit(2)
            return
        if not inspection["ready"]:
            raise SystemExit("Recovery refused because one or more checks failed.")
        if not sys.stdin.isatty():
            raise SystemExit(
                "Recovery refused: run this command in an interactive local terminal."
            )
        expected = f"RECOVER HARDWARE LOCK {args.run_id}"
        print(
            "\nInspect the connected QOP/OPX and instruments. Confirm that no "
            "job or output is active."
        )
        answer = input(f"Type exactly '{expected}' to recover the lock: ").strip()
        _print(
            recovery.recover(
                args.run_id,
                actor=getpass.getuser(),
                operator_confirmation=answer,
            )
        )
        return

    from .service import AgentService

    service = AgentService(settings)
    if args.command == "init":
        _print(
            {
                "status": "ready",
                "database": str(settings.database_path),
                "qualibrate_config": str(settings.qualibrate_config_path),
                "qualibrate_project": settings.qualibrate_project,
                "active_state": str(settings.active_state),
                "wiring": str(settings.wiring_path),
                "data_root": str(settings.data_root),
                "measurement_mode_entry_phrase": (
                    settings.measurement_mode_entry_phrase
                ),
                "qualibrate_python": str(settings.qualibrate_python),
            }
        )
    elif args.command == "recover-close":
        deadline = time.monotonic() + max(0.0, min(args.timeout_seconds, 300.0))
        while True:
            result = service.prepare_recovery_close(getpass.getuser())
            if result["status"] != "waiting_for_worker":
                _print(result)
                if not result.get("services_may_stop"):
                    raise SystemExit(2)
                break
            if time.monotonic() >= deadline:
                result["status"] = "worker_stop_timeout"
                result["reason"] = (
                    "The worker did not exit before the recovery timeout. Use the "
                    "Dashboard emergency stop only after checking the instrument."
                )
                _print(result)
                raise SystemExit(2)
            time.sleep(0.5)
    elif args.command == "history-site":
        from .history_site import write_history_site

        index = write_history_site(service, everything=True)
        _print({"status": "written", "index": str(index)})
    elif args.command == "status":
        _print(service.status(args.workflow_id))
    elif args.command == "approve":
        _approve(service, args.proposal_id)
    elif args.command == "replay":
        _print(service.analyzer.analyze_snapshot(args.snapshot, args.node))
    elif args.command == "autonomy-status":
        _print(service.autonomy_status(lease_id=args.lease_id))
    elif args.command == "autonomy-pause":
        _print(
            service.pause_autonomy(
                args.lease_id, getpass.getuser(), args.reason
            )
        )
    elif args.command == "autonomy-stop":
        _print(
            service.stop_autonomy(
                args.lease_id, getpass.getuser(), args.reason
            )
        )
    elif args.command == "autonomy-emergency-stop":
        _print(
            service.stop_autonomy(
                args.lease_id,
                getpass.getuser(),
                args.reason,
                emergency=True,
            )
        )
    elif args.command == "serve":
        host = args.host or settings.host
        port = args.port or settings.port
        from .mcp_server import mcp, service as mcp_service

        mcp_service.start_autonomy_watchdog()
        mcp.run(
            transport="streamable-http",
            host=host,
            port=port,
            streamable_http_path=settings.mcp_path,
        )
    elif args.command == "stdio":
        from .mcp_server import mcp

        mcp.run(transport="stdio")
    elif args.command == "approval-serve":
        from .approval_server import run_approval_server

        run_approval_server(
            settings,
            host=args.host or settings.approval_host,
            port=args.port or settings.approval_port,
        )


if __name__ == "__main__":
    main()
