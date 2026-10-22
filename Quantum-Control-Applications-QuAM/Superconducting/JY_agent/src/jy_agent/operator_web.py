from __future__ import annotations

import getpass
import hmac
import ipaddress
import json
from html import escape
from urllib.parse import parse_qs

from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

from .dashboard_access import DashboardAccessManager
from .recovery import HardwareLockRecovery, HardwareLockRecoveryError
from .service import AgentService, ServiceError


OPERATOR_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
        "base-uri 'none'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}
MAX_OPERATOR_FORM_BYTES = 8192


async def handle_operator_console(
    request: Request,
    service: AgentService,
    access: DashboardAccessManager,
) -> Response:
    client_host = request.client.host if request.client else ""
    request_host = request.url.hostname or ""
    if not _is_loopback(client_host) or not _is_loopback(request_host):
        return HTMLResponse(
            "<!doctype html><html><head><meta charset=\"utf-8\"><title>Forbidden</title>"
            "</head><body><h1>Forbidden</h1><p>The JY operator console is "
            "available only through a loopback URL on the lab computer.</p>"
            "</body></html>",
            status_code=403,
            headers=OPERATOR_SECURITY_HEADERS,
        )

    success: str | None = None
    error: str | None = None
    if request.method == "POST":
        try:
            if not request.headers.get("content-type", "").lower().startswith(
                "application/x-www-form-urlencoded"
            ):
                raise ServiceError("Invalid operator form encoding")
            body = await request.body()
            if len(body) > MAX_OPERATOR_FORM_BYTES:
                raise ServiceError("Operator form was too large")
            fields = parse_qs(
                body.decode("utf-8"),
                keep_blank_values=True,
                strict_parsing=True,
                max_num_fields=8,
            )
            supplied_csrf = _single(fields, "csrf_token")
            if not hmac.compare_digest(supplied_csrf, service.operator_csrf_token()):
                raise ServiceError("The operator-console token was invalid")
            operation = _single(fields, "operation")
            actor = f"{getpass.getuser()} via local operator console"
            if operation == "recover_lock":
                if _single(fields, "hardware_attestation") != "confirmed":
                    raise HardwareLockRecoveryError(
                        "Confirm that the connected hardware has no active job/output."
                    )
                run_id = _single(fields, "run_id")
                recovery = HardwareLockRecovery(service.settings, service.db)
                result = recovery.recover(
                    run_id,
                    actor=actor,
                    operator_confirmation=_single(fields, "confirmation"),
                )
                service.reconcile_full_shutdown_request()
                success = (
                    f"Hardware lock for run {result['run_id']} was safely archived. "
                    "Any queued full shutdown will now continue automatically."
                )
            elif operation == "close_stale_qm":
                receipt = service.close_stale_qm_for_retained_lock(actor)
                if receipt.get("problems"):
                    error = (
                        "The stale-QM cleanup ran but this chip is not clear yet: "
                        + " ".join(str(item) for item in receipt["problems"])
                    )
                else:
                    closed = receipt.get("summary", {}).get("closed") or []
                    success = (
                        "No quantum machine of this chip remains open"
                        + (f" (closed {', '.join(closed)})" if closed else "")
                        + ". Other users' machines were left alone."
                    )
            else:
                raise ServiceError("Unknown operator-console operation")
        except (
            UnicodeDecodeError,
            ValueError,
            ServiceError,
            HardwareLockRecoveryError,
        ) as exc:
            error = str(exc)
    return _page(
        service,
        access,
        success=success,
        error=error,
        status_code=400 if error else 200,
    )


def _page(
    service: AgentService,
    access: DashboardAccessManager,
    *,
    success: str | None = None,
    error: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    csrf = service.operator_csrf_token()
    notice = ""
    if error:
        notice = f'<p class="notice error">{escape(error)}</p>'
    elif success:
        notice = f'<p class="notice success">{escape(success)}</p>'

    recovery_html = _recovery_section(service, csrf)
    status = service.status()
    workflow = status.get("workflow") or {}
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JY local operator console</title><style>
:root{{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif}}*{{box-sizing:border-box}}
body{{margin:0;background:#08101d;color:#edf4ff;line-height:1.5}}main{{max-width:1100px;margin:24px auto;padding:0 16px 48px}}
.card{{background:#121d31;border:1px solid #2b3b5d;border-radius:14px;padding:18px;margin:16px 0}}
.highlight{{border-color:#68a1ff}}.warning{{background:#5b3a12;color:#ffe7b0;padding:12px;border-radius:8px}}
.notice{{padding:12px;border-radius:8px}}.error{{background:#5a1e2a}}.success{{background:#14532d}}
input,button{{width:100%;padding:11px;border-radius:8px;font:inherit}}input{{background:#07101f;color:white;border:1px solid #6680b5}}
button{{margin-top:10px;border:0;background:#3568e8;color:white;font-weight:750;cursor:pointer}}button.danger{{background:#b4232d}}
code{{display:block;padding:12px;background:#070d18;border-radius:8px;overflow-wrap:anywhere;user-select:all}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #2b3b5d;text-align:left;vertical-align:top}}
.scroll{{overflow:auto}}label.check{{display:flex;gap:10px;align-items:flex-start}}label.check input{{width:auto;margin-top:5px}}
</style></head><body><main>
<h1>JY local operator console</h1>
<p class="warning">Local lab computer only. This control surface is served on the MCP loopback port and is not routed through the public Dashboard tunnel.</p>
{notice}
<section class="card"><h2>Current state</h2><p>Workflow: {escape(str(workflow.get('id') or 'none'))}</p>
<p>Workflow status: {escape(str(workflow.get('status') or 'none'))}; active worker: {escape(str(bool(status.get('active_run'))))}; hardware lock: {escape(str(bool(status.get('hardware_lock'))))}</p></section>
<section class="card"><h2>Public Dashboard access</h2>
<p>Phones and browsers open the same Cloudflare URL and sign in with the fixed
password stored at <code>{escape(str(service.settings.runtime / 'dashboard-password.txt'))}</code>.
There is no device-pairing step.</p></section>
{recovery_html}
</main></body></html>"""
    return HTMLResponse(html, status_code=status_code, headers=OPERATOR_SECURITY_HEADERS)


def _recovery_section(service: AgentService, csrf: str) -> str:
    if not service.settings.lock_path.exists():
        return '<section class="card"><h2>Hardware-lock recovery</h2><p>No retained hardware lock is present.</p></section>'
    try:
        lock = json.loads(service.settings.lock_path.read_text(encoding="utf-8-sig"))
        run_id = str(lock.get("run_id") or "")
        inspection = HardwareLockRecovery(service.settings, service.db).inspect(run_id)
    except Exception as exc:
        return (
            '<section class="card"><h2>Hardware-lock recovery</h2>'
            f'<p class="notice error">Inspection failed: {escape(str(exc))}</p></section>'
        )
    checks = inspection.get("checks", {})
    failure_list = "".join(
        f"<li>{escape(str(item))}</li>" for item in inspection.get("failures", [])
    )
    qm_html = (
        _stale_qm_html(service, run_id, csrf) if checks.get("forced_stop") else ""
    )
    if not inspection.get("ready"):
        return f"""<section class="card"><h2>Hardware-lock recovery</h2>
        <p class="notice error">Recovery is blocked. The lock remains untouched.</p>
        <ul>{failure_list}</ul>{qm_html}<code>{escape(json.dumps(checks, ensure_ascii=False, indent=2))}</code></section>"""
    mismatch = ""
    if checks.get("configured_state_differs_from_run"):
        mismatch = (
            '<p class="warning">The currently configured state path differs from the state file used by this run. '
            'The verified run-specific active/recovery/database hashes still match; this difference will remain in the receipt.</p>'
        )
    confirmation = f"RECOVER HARDWARE LOCK {run_id}"
    return f"""<section class="card"><h2>Hardware-lock recovery</h2>
    <p>All software checks pass: lock/run/request identity, no live worker, no active database run, matching run state and recovery hashes, and no restore error.</p>{mismatch}{qm_html}
    <code>{escape(json.dumps(checks, ensure_ascii=False, indent=2))}</code>
    <p class="warning">Before continuing, physically inspect the connected QOP/OPX and instruments. Do not recover while any job or output is active.</p>
    <form method="post" action="/operator"><input type="hidden" name="operation" value="recover_lock">
    <input type="hidden" name="csrf_token" value="{escape(csrf)}"><input type="hidden" name="run_id" value="{escape(run_id)}">
    <label class="check"><input type="checkbox" name="hardware_attestation" value="confirmed" required>
    <span>I inspected the connected hardware and confirm there is no active job/output.</span></label>
    <p>Type exactly:</p><code>{escape(confirmation)}</code>
    <input name="confirmation" autocomplete="off" spellcheck="false" required>
    <button type="submit" class="danger">Safely archive the retained lock</button></form></section>"""


_QM_VERDICT_LABELS = {
    "ours": "this chip's machine",
    "overlapping": "shares this chip's ports - blocks recovery, not closed",
    "unrelated": "other user, ignored",
    "unreadable": "ports unreadable - blocks recovery",
}


def _stale_qm_html(service: AgentService, run_id: str, csrf: str) -> str:
    """The forced-stop QM cleanup form and the last recorded cleanup result."""

    receipt = HardwareLockRecovery(service.settings, service.db).read_qm_receipt(run_id)
    rows = ""
    status = "<p>No stale-QM cleanup has been recorded for this run yet.</p>"
    if receipt is not None:
        report = receipt.get("report") if isinstance(receipt.get("report"), dict) else {}
        for machine in report.get("machines") or []:
            if not isinstance(machine, dict):
                continue
            verdict = str(machine.get("verdict") or "unreadable")
            label = _QM_VERDICT_LABELS.get(verdict, _QM_VERDICT_LABELS["unreadable"])
            if verdict == "ours":
                label += ": closed" if machine.get("closed") is True else ": STILL OPEN"
            rows += (
                f"<tr><td>{escape(str(machine.get('qm_id') or '?'))}</td>"
                f"<td>{escape(verdict)}</td><td>{escape(label)}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="3">No open quantum machine was listed.</td></tr>'
        problems = receipt.get("problems") or []
        verdict_text = (
            "Blocking: " + " ".join(str(item) for item in problems)
            if problems
            else "Clear: no machine of this chip, or sharing its ports, remains open."
        )
        status = (
            f"<p>Last check {escape(str(receipt.get('checked_at') or '?'))} "
            f"(valid {HardwareLockRecovery.QM_RECEIPT_MAX_AGE_SECONDS // 60} minutes). "
            f"{escape(verdict_text)}</p>"
            '<div class="scroll"><table><tr><th>Quantum machine</th><th>Class</th>'
            f"<th>Meaning</th></tr>{rows}</table></div>"
        )
    return f"""<h3>This chip's stale quantum machine</h3>
    <p>The run was force-stopped, so its quantum machine may still be open on the shared QOP.
    The cleanup closes only a machine whose every port belongs to this chip's state.json (halting its job first);
    machines of other users (no shared port) are listed as ignored and never touched.</p>
    {status}
    <form method="post" action="/operator"><input type="hidden" name="operation" value="close_stale_qm">
    <input type="hidden" name="csrf_token" value="{escape(csrf)}">
    <button type="submit">Close this chip's stale quantum machine</button></form>"""


def _single(fields: dict[str, list[str]], name: str) -> str:
    values = fields.get(name)
    if values is None or len(values) != 1:
        raise ValueError(f"Operator form must include one {name} field")
    return values[0]


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.casefold() == "localhost"
