from __future__ import annotations

import asyncio
import getpass
import hmac
import ipaddress
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any, Collection
from urllib.parse import parse_qs, urlencode

from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from .guide_web import guide_asset_path, guide_section
from .service import AgentService, ServiceError


MAX_FORM_BYTES = 4096
# Confirmation, token and ids plus the lease-page checkboxes, the mode and
# flux choices and the 02 inputs.
MAX_APPROVAL_FORM_FIELDS = 20
SUMMARY_THUMBNAIL_LIMIT = 6
TAIPEI_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Taipei")
CACHEABLE_ASSET_CACHE_CONTROL = "private, max-age=604800, immutable"

SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self'; form-action 'self'; "
        "base-uri 'none'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "same-origin",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}

CACHEABLE_ASSET_HEADERS = {
    **SECURITY_HEADERS,
    "Cache-Control": CACHEABLE_ASSET_CACHE_CONTROL,
}

AUTONOMY_EVENTS_SCRIPT = """(() => {
  const eventsUrl = document.body.dataset.autonomyEvents;
  if (!eventsUrl || typeof EventSource === "undefined") return;
  const stream = new EventSource(eventsUrl);
  let pendingRefresh = null;
  stream.addEventListener("refresh", () => {
    if (pendingRefresh !== null) window.clearTimeout(pendingRefresh);
    pendingRefresh = window.setTimeout(() => window.location.reload(), 350);
  });
  window.addEventListener("beforeunload", () => stream.close());
})();
"""

SESSION_DASHBOARD_SCRIPT = """(() => {
  // Folded sections keep what this browser opened across automatic refreshes.
  document.querySelectorAll('details[data-remember-open]').forEach((item) => {
    const key = 'jy-open:' + item.dataset.rememberOpen;
    let saved = null;
    try { saved = window.localStorage.getItem(key); } catch (_) {}
    if (saved === '1') item.open = true;
    if (saved === '0') item.open = false;
    item.addEventListener('toggle', () => {
      try { window.localStorage.setItem(key, item.open ? '1' : '0'); } catch (_) {}
    });
  });
  const selector = document.querySelector('[data-experiment-selector]');
  const panels = Array.from(document.querySelectorAll('[data-experiment-panel]'));
  const available = panels.map((panel) => panel.dataset.experimentPanel);
  const show = (value) => {
    panels.forEach((panel) => { panel.hidden = panel.dataset.experimentPanel !== value; });
    if (selector) selector.value = value;
    const url = new URL(window.location.href);
    url.searchParams.set('experiment', value);
    window.history.replaceState({}, '', url);
  };
  if (panels.length) {
    if (selector) selector.addEventListener('change', () => show(selector.value));
    document.querySelectorAll('[data-experiment-link]').forEach((link) => {
      link.addEventListener('click', (event) => {
        const index = available.indexOf(link.dataset.experimentLink);
        if (index < 0) return;
        event.preventDefault();
        show(available[index]);
        if (panels[index].scrollIntoView) {
          panels[index].scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });
    const requested = new URL(window.location.href).searchParams.get('experiment');
    show(available.includes(requested) ? requested : available[available.length - 1]);
  }
  const eventsUrl = document.body.dataset.sessionEvents;
  if (!eventsUrl || typeof EventSource === 'undefined') return;
  const stream = new EventSource(eventsUrl);
  let pendingRefresh = null;
  stream.addEventListener('refresh', () => {
    if (pendingRefresh !== null) window.clearTimeout(pendingRefresh);
    pendingRefresh = window.setTimeout(() => window.location.reload(), 350);
  });
  window.addEventListener('beforeunload', () => stream.close());
})();
"""

DASHBOARD_EVENT_TYPES = {
    "home": None,
    # Operator request 2026-10-08: static operating guide, never refreshed.
    "guide": frozenset(),
    # Operator request 2026-10-03: model / token / estimated-cost page.
    "usage": {"run_started", "run_analyzed", "decision_recorded"},
    "approval": {
        "proposal_created",
        "proposal_approved",
        "autonomy_lease_expired",
        "workflow_stopped",
        "full_shutdown_requested",
    },
    "results": {
        "run_started",
        "run_analyzed",
        "run_failed",
        "instrument_error_paused",
        "measurement_paused_for_instrument_error",
        "decision_recorded",
        "autonomy_pause",
        "autonomy_stop",
        "autonomy_emergency_stop",
        "autonomy_halted",
        "autonomy_lease_expired",
        "autonomy_scope_completed",
        "autonomy_resumed",
        "measurement_mode_resumed",
        "workflow_stopped",
        "full_shutdown_requested",
        "hardware_lock_recovered",
    },
}

LANGUAGE_SCRIPT = """(() => {
  const selector = document.querySelector('[data-language-selector]');
  if (!selector) return;
  const url = new URL(window.location.href);
  const requested = url.searchParams.get('lang');
  let saved = null;
  try { saved = window.localStorage.getItem('jy-dashboard-language'); } catch (_) {}
  if (!requested && (saved === 'en' || saved === 'zh-Hant')) {
    url.searchParams.set('lang', saved);
    window.location.replace(url);
    return;
  }
  selector.addEventListener('change', () => {
    try { window.localStorage.setItem('jy-dashboard-language', selector.value); } catch (_) {}
    const next = new URL(window.location.href);
    next.searchParams.set('lang', selector.value);
    window.location.assign(next);
  });
})();
"""


@dataclass(frozen=True)
class ApprovalPrincipal:
    actor: str
    method: str
    display_name: str


def _requested_language(request: Request) -> str:
    return "en" if request.query_params.get("lang", "").casefold() == "en" else "zh-Hant"


def _ui(language: str, traditional_chinese: str, english: str) -> str:
    return english if language == "en" else traditional_chinese


_ZH_ENUMS = {
    "pending": "待核准",
    "approved": "已核准",
    "rejected": "已拒絕",
    "expired": "已過期",
    "active": "進行中",
    "paused": "已暫停",
    "stopped": "已停止",
    "cancelled_by_shutdown": "因完整關機取消",
    "force_stopped": "已強制停止",
    "halted": "已中止",
    "stopping": "正在安全停止",
    "recovery_required": "需要現場安全復原",
    "recovery_only": "僅限本機復原",
    "waiting_for_run": "等待量測停止",
    "waiting_for_force_exit": "等待強制停止完成",
    "ready": "準備關閉服務",
    "quarantine_ready": "準備隔離並關閉服務",
    "dispatching": "正在關閉服務",
    "running": "執行中",
    "completed": "已完成",
    "failed": "失敗",
    "instrument_unreachable": "儀器無法連線",
    "program_submission_timeout": "程式提交逾時（儀器正常）",
    "qop_compile_failure": "QOP 編譯失敗",
    "pass": "通過",
    "needs_review": "需要審閱",
    "manual_review": "人工審閱",
    "conversational": "對話量測",
    "autonomous": "有限自動量測",
    "run": "實驗執行",
    "autonomy_lease": "有限自動授權",
    "state_commit": "狀態寫入",
    "advance": "進到下一節點",
    "repeat": "重複量測",
    "stop": "停止",
}


def _localized_enum(language: str, value: Any) -> str:
    rendered = str(value)
    return _ZH_ENUMS.get(rendered, rendered) if language != "en" else rendered


def _limit_label(language: str, value: Any) -> str:
    if value is None:
        return _ui(language, "無限制", "Unlimited")
    return str(value)


def _language_switch(language: str) -> str:
    zh_selected = " selected" if language != "en" else ""
    en_selected = " selected" if language == "en" else ""
    label = _ui(language, "顯示語言", "Display language")
    return f"""
    <div class="language-switch">
      <label for="language-selector">{escape(label)}</label>
      <select id="language-selector" data-language-selector>
        <option value="zh-Hant"{zh_selected}>繁體中文</option>
        <option value="en"{en_selected}>English</option>
      </select>
    </div>"""


def _json_details(label: str, value: Any, *, card: bool = False) -> str:
    class_name = ' class="card"' if card else ""
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return (
        f"<details{class_name}><summary>{escape(label)}</summary>"
        f"<pre>{escape(payload)}</pre></details>"
    )


def _principal_label(principal: ApprovalPrincipal, language: str) -> str:
    if principal.method == "public_token_browser":
        return _ui(language, "已授權操作人員", "authorized operator")
    if principal.method == "local_browser":
        return _ui(
            language,
            f"{getpass.getuser()}（本機瀏覽器）",
            f"{getpass.getuser()} (local browser)",
        )
    return principal.display_name


def _session_view_redirect(
    session_id: str | None,
    view: str,
    language: str,
) -> RedirectResponse | None:
    if not session_id:
        return None
    query = urlencode({"lang": language})
    return RedirectResponse(
        f"/session/{session_id}/{view}?{query}", status_code=303
    )


async def handle_browser_approval(
    request: Request,
    service: AgentService,
) -> Response:
    """Render or submit one human-only approval review form."""
    language = _requested_language(request)
    try:
        principal = _approval_principal(request, service)
    except ServiceError as exc:
        return _message_page(
            _ui(language, "禁止存取", "Forbidden"),
            str(exc),
            status_code=403,
            language=language,
        )

    proposal_id = str(request.path_params.get("proposal_id", ""))
    try:
        proposal = service.proposal(proposal_id)
    except ServiceError as exc:
        return _message_page(
            _ui(language, "找不到提案", "Proposal not found"),
            str(exc),
            status_code=404,
            language=language,
        )

    if request.method == "GET":
        redirect = _session_view_redirect(
            proposal.get("dashboard_session_id"), "approval", language
        )
        if redirect is not None:
            return redirect
        return _proposal_page(
            service, proposal, principal=principal, language=language
        )

    allowed_origins = {
        *service.approval_allowed_origins,
        _request_origin(request),
    }
    if not _has_trusted_submission_source(request, allowed_origins):
        return _proposal_page(
            service,
            proposal,
            principal=principal,
            error=(
                "Approval refused because the browser submission source was not "
                "trusted."
            ),
            status_code=403,
            language=language,
        )
    if not request.headers.get("content-type", "").lower().startswith(
        "application/x-www-form-urlencoded"
    ):
        return _proposal_page(
            service,
            proposal,
            principal=principal,
            error="Approval refused because the form encoding was invalid.",
            status_code=415,
            language=language,
        )

    body = await request.body()
    if len(body) > MAX_FORM_BYTES:
        return _proposal_page(
            service,
            proposal,
            principal=principal,
            error="Approval form was too large.",
            status_code=413,
            language=language,
        )
    try:
        fields = parse_qs(
            body.decode("utf-8"),
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=MAX_APPROVAL_FORM_FIELDS,
        )
        if _lease_option_offer(proposal) is not None:
            # Operator request 2026-10-10: automatic measurement is approved
            # only on the Dashboard approval page, after its node preview.
            raise ServiceError(
                _ui(
                    language,
                    "自動量測請在量測 Dashboard 的「核准」頁預覽並核准。",
                    "Approve automatic measurement on the Dashboard approval page, after its preview.",
                )
            )
        approved = service.approve_from_browser(
            proposal_id,
            _single_field(fields, "confirmation"),
            _single_field(fields, "csrf_token"),
            request.client.host if request.client is not None else "",
            actor=principal.actor,
            approval_method=principal.method,
            lease_options=_lease_options_from_fields(fields, proposal),
        )
    except (UnicodeDecodeError, ValueError, ServiceError) as exc:
        return _proposal_page(
            service,
            service.proposal(proposal_id),
            principal=principal,
            error=str(exc),
            status_code=400,
            language=language,
        )
    redirect = _session_view_redirect(
        approved.get("dashboard_session_id"), "approval", language
    )
    if redirect is not None:
        return redirect
    return _proposal_page(
        service,
        approved,
        principal=principal,
        success=_ui(
            language,
            "提案已核准；可以關閉此頁並回到對話。",
            "Proposal approved. You may close this page and return to the conversation.",
        ),
        language=language,
    )


async def handle_browser_approval_asset(
    request: Request,
    service: AgentService,
) -> Response:
    """Serve one proposal-bound raster result image without exposing file paths."""
    language = _requested_language(request)
    try:
        _approval_principal(request, service)
        proposal_id = str(request.path_params.get("proposal_id", ""))
        asset_index = int(str(request.path_params.get("asset_index", "")))
        path = service.proposal_review_asset(proposal_id, asset_index)
    except (ValueError, ServiceError):
        return _message_page(
            _ui(language, "找不到結果圖", "Result image not found"),
            _ui(
                language,
                "此提案無法取得這張結果圖。",
                "This result image is unavailable for the proposal.",
            ),
            status_code=404,
            language=language,
        )
    return FileResponse(
        path,
        filename=path.name,
        content_disposition_type="inline",
        headers=SECURITY_HEADERS,
    )


async def handle_autonomy_control(
    request: Request,
    service: AgentService,
) -> Response:
    """Render and submit authenticated pause/stop/emergency controls."""
    language = _requested_language(request)
    try:
        principal = _approval_principal(request, service)
        lease_id = str(request.path_params.get("lease_id", ""))
        status = service.autonomy_status(lease_id=lease_id)
    except ServiceError as exc:
        return _message_page(
            _ui(language, "無法使用自動量測", "Autonomy unavailable"),
            str(exc),
            status_code=404,
            language=language,
        )
    if request.method == "GET":
        redirect = _session_view_redirect(
            service.dashboard_session_id_for_workflow(str(status["workflow_id"])),
            "home",
            language,
        )
        if redirect is not None:
            return redirect
        return _autonomy_page(service, status, principal, language=language)
    allowed_origins = {
        *service.approval_allowed_origins,
        _request_origin(request),
    }
    if not _has_trusted_submission_source(request, allowed_origins):
        return _autonomy_page(
            service,
            status,
            principal,
            error="Control refused because the browser source was not trusted.",
            status_code=403,
            language=language,
        )
    if not request.headers.get("content-type", "").lower().startswith(
        "application/x-www-form-urlencoded"
    ):
        return _autonomy_page(
            service,
            status,
            principal,
            error="Invalid form encoding.",
            status_code=415,
            language=language,
        )
    body = await request.body()
    if len(body) > MAX_FORM_BYTES:
        return _autonomy_page(
            service,
            status,
            principal,
            error="Control form was too large.",
            status_code=413,
            language=language,
        )
    try:
        fields = parse_qs(
            body.decode("utf-8"),
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=3,
        )
        result = service.control_autonomy_from_browser(
            lease_id,
            _single_field(fields, "action"),
            _single_field(fields, "csrf_token"),
            principal.actor,
        )
    except (UnicodeDecodeError, ValueError, ServiceError) as exc:
        return _autonomy_page(
            service,
            service.autonomy_status(lease_id=lease_id),
            principal,
            error=str(exc),
            status_code=400,
            language=language,
        )
    redirect = _session_view_redirect(
        service.dashboard_session_id_for_workflow(str(result["workflow_id"])),
        "home",
        language,
    )
    if redirect is not None:
        return redirect
    return _autonomy_page(
        service,
        result,
        principal,
        success=_ui(
            language,
            f"控制已套用：{_localized_enum(language, result['status'])}。",
            f"Control applied: {result['status']}.",
        ),
        language=language,
    )


async def handle_autonomy_control_asset(
    request: Request,
    service: AgentService,
) -> Response:
    """Serve the latest lease-bound result image without exposing disk paths."""
    language = _requested_language(request)
    try:
        _approval_principal(request, service)
        lease_id = str(request.path_params.get("lease_id", ""))
        asset_index = int(str(request.path_params.get("asset_index", "")))
        path = service.autonomy_review_asset(lease_id, asset_index)
    except (ValueError, ServiceError):
        return _message_page(
            _ui(language, "找不到結果圖", "Result image not found"),
            _ui(
                language,
                "此授權無法取得這張即時結果圖。",
                "This live result image is not available for the authorization.",
            ),
            status_code=404,
            language=language,
        )
    return FileResponse(
        path,
        filename=path.name,
        content_disposition_type="inline",
        headers=SECURITY_HEADERS,
    )


async def handle_autonomy_events_script(request: Request) -> Response:
    """Serve the same-origin EventSource client without inline JavaScript."""
    return Response(
        AUTONOMY_EVENTS_SCRIPT,
        media_type="text/javascript",
        headers=SECURITY_HEADERS,
    )


async def handle_language_script(request: Request) -> Response:
    """Serve the same-origin language selector without inline JavaScript."""
    return Response(
        LANGUAGE_SCRIPT,
        media_type="text/javascript",
        headers=SECURITY_HEADERS,
    )


async def handle_autonomy_events(
    request: Request,
    service: AgentService,
) -> Response:
    """Notify the page only for a new analyzed result or a control change."""
    language = _requested_language(request)
    try:
        _approval_principal(request, service)
        lease_id = str(request.path_params.get("lease_id", ""))
        service.autonomy_status(lease_id=lease_id)
        after_id = max(0, int(request.query_params.get("after", "0")))
    except (ValueError, ServiceError) as exc:
        return _message_page(
            _ui(language, "無法使用自動量測", "Autonomy unavailable"),
            str(exc),
            status_code=404,
            language=language,
        )

    async def event_stream():
        cursor = after_id
        yield ": connected\n\n"
        while not await request.is_disconnected():
            update = service.autonomy_ui_events_after(lease_id, cursor)
            cursor = int(update["cursor"])
            events = update["events"]
            if events:
                data = json.dumps(
                    {
                        "cursor": cursor,
                        "events": [item["event_type"] for item in events],
                    },
                    separators=(",", ":"),
                )
                yield f"id: {cursor}\nevent: refresh\ndata: {data}\n\n"
            await asyncio.sleep(0.75)

    stream_headers = dict(SECURITY_HEADERS)
    stream_headers["Cache-Control"] = "no-cache, no-store"
    stream_headers["X-Accel-Buffering"] = "no"
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=stream_headers,
    )


async def handle_session_dashboard(
    request: Request,
    service: AgentService,
    *,
    view: str | None = None,
) -> Response:
    """Render one page of the shared measurement Dashboard."""
    language = _requested_language(request)
    resolved_view = view or str(request.path_params.get("view") or "")
    lease_options: dict[str, Any] | None = None
    if resolved_view not in DASHBOARD_EVENT_TYPES:
        resolved_view = "home"
    try:
        principal = _approval_principal(request, service)
        session_id = str(request.path_params.get("session_id") or "")
        if not session_id:
            session_id = service.current_dashboard_session_id() or ""
        if not session_id:
            return _dashboard_idle_page(principal, language=language)
        review = service.dashboard_review(session_id)
    except ServiceError as exc:
        return _message_page(
            _ui(language, "無法使用量測工作階段", "Measurement session unavailable"),
            str(exc),
            status_code=404,
            language=language,
        )
    if request.method == "GET" and view is None and request.url.path.startswith(
        "/session/"
    ):
        return _session_view_redirect(session_id, "home", language) or _message_page(
            _ui(language, "無法使用量測工作階段", "Measurement session unavailable"),
            "The session route could not be resolved.",
            status_code=404,
            language=language,
        )
    if request.method == "GET":
        return _session_dashboard_page(
            service,
            review,
            principal,
            view=resolved_view,
            language=language,
            query=dict(request.query_params),
        )

    allowed_origins = {*service.approval_allowed_origins, _request_origin(request)}
    if not _has_trusted_submission_source(request, allowed_origins):
        return _session_dashboard_page(
            service,
            review,
            principal,
            view=resolved_view,
            error="Request refused because the browser source was not trusted.",
            status_code=403,
            language=language,
        )
    if not request.headers.get("content-type", "").lower().startswith(
        "application/x-www-form-urlencoded"
    ):
        return _session_dashboard_page(
            service,
            review,
            principal,
            view=resolved_view,
            error="Invalid form encoding.",
            status_code=415,
            language=language,
        )
    body = await request.body()
    if len(body) > MAX_FORM_BYTES:
        return _session_dashboard_page(
            service,
            review,
            principal,
            view=resolved_view,
            error="Form was too large.",
            status_code=413,
            language=language,
        )
    try:
        fields = parse_qs(
            body.decode("utf-8"),
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=MAX_APPROVAL_FORM_FIELDS,
        )
        operation = _single_field(fields, "operation")
        if view is not None and operation not in {
            "home": {"shutdown", "control"},
            "approval": {"approve", "preview"},
            "results": set(),
            "usage": set(),
            "guide": set(),
        }[resolved_view]:
            raise ServiceError("This control does not belong to the current Dashboard page.")
        if operation == "preview":
            resolved_view = "approval"
            proposal_id = _single_field(fields, "proposal_id")
            pending = review.get("pending_proposal") or {}
            if proposal_id != str(pending.get("id", "")):
                raise ServiceError("This proposal is no longer the pending session action.")
            if not hmac.compare_digest(
                _single_field(fields, "csrf_token"),
                service.approval_csrf_token(proposal_id),
            ):
                raise ServiceError("The browser approval token was invalid.")
            lease_options = _lease_options_from_fields(fields, pending)
            if lease_options is None:
                raise ServiceError("This approval has no choices to preview.")
            return _session_dashboard_page(
                service,
                review,
                principal,
                view=resolved_view,
                language=language,
                lease_preview=_lease_preview(service, proposal_id, lease_options),
            )
        if operation == "approve":
            resolved_view = "approval"
            proposal_id = _single_field(fields, "proposal_id")
            pending = review.get("pending_proposal") or {}
            if proposal_id != str(pending.get("id", "")):
                raise ServiceError("This proposal is no longer the pending session action.")
            lease_options = _lease_options_from_fields(fields, pending)
            if lease_options is not None:
                # Operator request 2026-10-10: approval needs the preview of
                # exactly these choices and the ticked agree box.
                digest = fields.get("preview_digest", [""])
                if len(digest) != 1 or not hmac.compare_digest(
                    digest[0], service.lease_preview_digest(proposal_id, lease_options)
                ):
                    raise ServiceError(
                        _ui(
                            language,
                            "選項在預覽後已變更或尚未預覽；請按「預覽此次量測的實驗」後再核准。",
                            "The choices changed after the preview, or there was no preview; preview again before approving.",
                        )
                    )
                if fields.get("agree") != ["yes"]:
                    raise ServiceError(
                        _ui(
                            language,
                            "請先勾選「我已確認以上會執行的實驗、起點與選項」。",
                            "Tick the box confirming the experiments, start point and choices first.",
                        )
                    )
            service.approve_from_browser(
                proposal_id,
                _single_field(fields, "confirmation"),
                _single_field(fields, "csrf_token"),
                request.client.host if request.client is not None else "",
                actor=principal.actor,
                approval_method=principal.method,
                lease_options=lease_options,
            )
            if str(review.get("session", {}).get("mode")) == "autonomous":
                success = _ui(
                    language,
                    "已核准；AI agent 會在本次有限授權下自動繼續，不需回到對話重複同意。若對話已被平台暫停，請回對話視窗輸入「已核准」後便會開始執行實驗。",
                    "Approved. The AI agent will continue under this bounded authorization without another approval message. If the host paused the conversation, return to it and enter ‘已核准’ to wake the experiment loop.",
                )
            else:
                success = _ui(
                    language,
                    "請回對話視窗輸入「已核准」後便會開始執行實驗。",
                    "Return to the conversation and enter ‘已核准’ to start the experiment.",
                )
        elif operation == "control":
            resolved_view = "home"
            session = review["session"]
            lease_id = str(session.get("autonomy_lease_id") or "")
            if not lease_id:
                raise ServiceError("Manual autonomy controls are unavailable in conversational mode.")
            result = service.control_autonomy_from_browser(
                lease_id,
                _single_field(fields, "action"),
                _single_field(fields, "csrf_token"),
                principal.actor,
            )
            success = _ui(
                language,
                f"控制已套用：{_localized_enum(language, result['status'])}。",
                f"Control applied: {result['status']}.",
            )
        elif operation == "shutdown":
            resolved_view = "home"
            result = service.request_full_shutdown_from_browser(
                session_id,
                _single_field(fields, "confirmation"),
                _single_field(fields, "csrf_token"),
                principal.actor,
            )
            if result.get("status") in {"recovery_required", "quarantine_ready"}:
                success = _ui(
                    language,
                    "完整結束已登記；服務會在確認 worker 已離開後關閉，保留的 hardware lock 會被隔離，之後只能從實驗室電腦進行 recovery。",
                    "Full shutdown is queued. After the worker exit is verified, services will close and any retained hardware lock will be quarantined for local recovery on the lab PC.",
                )
            else:
                success = _ui(
                    language,
                    "完整結束已登記；系統會安全等待目前工作停止，接著關閉 workflow、Dashboard 與 tunnel。",
                    "Full shutdown is queued. JY will safely wait for current work, then stop the workflow, Dashboard, and tunnel.",
                )
        else:
            raise ServiceError("Unknown dashboard operation.")
    except (UnicodeDecodeError, ValueError, ServiceError) as exc:
        retained = None
        if lease_options is not None:
            # Keep the operator's choices (and their preview, if it still
            # applies) instead of resetting the form.
            pending_id = str((review.get("pending_proposal") or {}).get("id", ""))
            try:
                retained = _lease_preview(service, pending_id, lease_options)
            except (ServiceError, ValueError):
                retained = {"options": lease_options}
        return _session_dashboard_page(
            service,
            service.dashboard_review(session_id),
            principal,
            view=resolved_view,
            error=str(exc),
            status_code=400,
            language=language,
            lease_preview=retained,
        )
    return _session_dashboard_page(
        service,
        service.dashboard_review(session_id),
        principal,
        view=resolved_view,
        success=success,
        language=language,
    )


async def handle_session_dashboard_asset(
    request: Request,
    service: AgentService,
) -> Response:
    language = _requested_language(request)
    try:
        _approval_principal(request, service)
        session_id = str(request.path_params.get("session_id", ""))
        asset_index = int(str(request.path_params.get("asset_index", "")))
        path = service.dashboard_review_asset(session_id, asset_index)
        version = request.query_params.get("v")
        current = _asset_version(path)
        if request.query_params.get("thumb") == "1":
            path = service.dashboard_asset_thumbnail(path)
    except (ValueError, ServiceError, OSError):
        return _message_page(
            _ui(language, "找不到結果圖", "Result image not found"),
            _ui(
                language,
                "此工作階段無法取得這張結果圖。",
                "This session result image is unavailable.",
            ),
            status_code=404,
            language=language,
        )
    # 2026-10-04: a versioned image URL names one immutable file, so the
    # browser may keep it; refreshing the page then reloads only new plots.
    headers = (
        CACHEABLE_ASSET_HEADERS if version and version == current else SECURITY_HEADERS
    )
    return FileResponse(
        path,
        filename=path.name,
        content_disposition_type="inline",
        headers=headers,
    )


async def handle_chip_history_asset(
    request: Request,
    service: AgentService,
) -> Response:
    """A plot of a successful run on this session's chip (operator 2026-10-13)."""
    language = _requested_language(request)
    try:
        _approval_principal(request, service)
        session_id = str(request.path_params.get("session_id", ""))
        run_id = str(request.path_params.get("run_id", ""))
        position = int(str(request.path_params.get("position", "")))
        path = service.chip_history_asset(session_id, run_id, position)
        version = request.query_params.get("v")
        current = _asset_version(path)
        if request.query_params.get("thumb") == "1":
            path = service.dashboard_asset_thumbnail(path)
    except (ValueError, ServiceError, OSError):
        return _message_page(
            _ui(language, "找不到結果圖", "Result image not found"),
            _ui(language, "無法取得這張晶片歷史結果圖。", "This chip history image is unavailable."),
            status_code=404,
            language=language,
        )
    headers = (
        CACHEABLE_ASSET_HEADERS if version and version == current else SECURITY_HEADERS
    )
    return FileResponse(
        path,
        filename=path.name,
        content_disposition_type="inline",
        headers=headers,
    )


def _chip_history_html(service: AgentService, session_id: str, language: str) -> str:
    from .chip_history import render_chip_history

    try:
        history = service.chip_history_for_session(session_id)
    except (ServiceError, ValueError) as exc:
        return f'<section class="card"><p class="muted">{escape(str(exc))}</p></section>'

    def image_url(run_id: str, position: int, thumb: bool) -> str:
        url = f"/session/{session_id}/chip-assets/{run_id}/{position}"
        return url + ("?thumb=1" if thumb else "")

    return render_chip_history(history, image_url, language)


def _asset_version(path: Any) -> str:
    from .service import _asset_token

    return _asset_token(path)


def _asset_url(session_id: str, item: dict[str, Any], position: int, *, thumb: bool = False) -> str:
    index = item.get("asset_indices", [])[position]
    tokens = item.get("asset_tokens") or []
    query = f"?v={tokens[position]}" if position < len(tokens) else ""
    if thumb:
        query += ("&" if query else "?") + "thumb=1"
    return f"/session/{escape(session_id)}/assets/{index}{escape(query)}"


async def handle_session_dashboard_script(request: Request) -> Response:
    return Response(
        SESSION_DASHBOARD_SCRIPT,
        media_type="text/javascript",
        headers=SECURITY_HEADERS,
    )


async def handle_session_dashboard_events(
    request: Request,
    service: AgentService,
    *,
    view: str = "home",
) -> Response:
    language = _requested_language(request)
    try:
        _approval_principal(request, service)
        session_id = str(request.path_params.get("session_id", ""))
        service.dashboard_status(session_id)
        after_id = max(0, int(request.query_params.get("after", "0")))
    except (ValueError, ServiceError) as exc:
        return _message_page(
            _ui(language, "無法使用量測工作階段", "Measurement session unavailable"),
            str(exc),
            status_code=404,
            language=language,
        )

    async def event_stream():
        cursor = after_id
        yield ": connected\n\n"
        while not await request.is_disconnected():
            update = service.dashboard_ui_events_after(
                session_id,
                cursor,
                DASHBOARD_EVENT_TYPES.get(view),
            )
            cursor = int(update["cursor"])
            events = update["events"]
            if events:
                data = json.dumps(
                    {
                        "cursor": cursor,
                        "events": [item["event_type"] for item in events],
                    },
                    separators=(",", ":"),
                )
                yield f"id: {cursor}\nevent: refresh\ndata: {data}\n\n"
            await asyncio.sleep(0.75)

    stream_headers = dict(SECURITY_HEADERS)
    stream_headers["Cache-Control"] = "no-cache, no-store"
    stream_headers["X-Accel-Buffering"] = "no"
    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=stream_headers
    )


def _dashboard_idle_page(
    principal: ApprovalPrincipal,
    *,
    language: str = "zh-Hant",
) -> HTMLResponse:
    html = f"""<!doctype html>
<html lang="{escape(language)}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script defer src="/assets/language.js"></script>
<title>{_ui(language, 'JY 量測首頁', 'JY measurement home')}</title>
<style>
:root {{ color-scheme:dark; font-family:system-ui,-apple-system,sans-serif; }}
body {{ margin:0; background:#08101d; color:#edf4ff; line-height:1.55; }}
main {{ max-width:760px; margin:48px auto; padding:0 16px; }}
.card {{ background:#121d31; border:1px solid #2b3b5d; border-radius:14px; padding:18px; }}
.muted {{ color:#9fb2d7; }}
.language-switch {{ position:fixed; top:10px; right:12px; padding:8px 10px; background:#121d31; border:1px solid #2b3b5d; border-radius:10px; }}
.language-switch label {{ display:block; color:#9fb2d7; font-size:.75rem; }}
.language-switch select {{ padding:6px; color:white; background:#07101f; border:1px solid #6680b5; border-radius:7px; }}
</style></head><body>{_language_switch(language)}<main>
<h1>{_ui(language, 'JY 量測首頁', 'JY measurement home')}</h1>
<section class="card">
<h2>{_ui(language, '目前沒有開啟中的量測工作階段', 'No measurement session is open')}</h2>
<p>{_ui(language, '服務已可連線；在 AI 對話中進入量測模式後，此入口會顯示新的工作階段。', 'The service is reachable. Enter measurement mode in the AI conversation to make the new session appear here.')}</p>
<p>{_ui(language, '若進入模式因上次殘留的程式、workflow 或設定失敗，請在 repository 的 AI 對話輸入「恢復」；英文輸入 “Recover”。', 'If entry fails because a program, workflow, or lifecycle setting was left over, send “Recover” in the repository AI conversation.')}</p>
<p><a href="/guide?lang={escape(language)}" style="color:#cfe0ff">{_ui(language, '使用說明：操作方式與各節點的成功範例', 'Guide: how to operate JY and what each node looks like')}</a></p>
<p class="muted">{_ui(language, '審閱者', 'Reviewer')}：{escape(_principal_label(principal, language))}</p>
</section></main></body></html>"""
    return HTMLResponse(html, headers=SECURITY_HEADERS)


async def handle_guide_page(request: Request, service: AgentService) -> Response:
    """The guide on its own, reachable without a measurement session."""

    language = _requested_language(request)
    try:
        _approval_principal(request, service)
    except ServiceError as exc:
        return _message_page(
            _ui(language, "無法開啟說明", "Guide unavailable"),
            str(exc),
            status_code=403,
            language=language,
        )
    session_id = service.current_dashboard_session_id()
    back = (
        f'<p><a href="/session/{escape(session_id)}/home?lang={escape(language)}">'
        f'{_ui(language, "回到量測 Dashboard", "Back to the measurement Dashboard")}</a></p>'
        if session_id
        else ""
    )
    html = f"""<!doctype html>
<html lang="{escape(language)}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script defer src="/assets/language.js"></script>
<title>{_ui(language, 'JY 使用說明', 'JY guide')}</title>
<style>
:root {{ color-scheme:dark; font-family:system-ui,-apple-system,sans-serif; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#08101d; color:#edf4ff; line-height:1.55; }}
main {{ max-width:1040px; margin:24px auto; padding:0 16px 52px; }}
a {{ color:#cfe0ff; }}
.card {{ background:#121d31; border:1px solid #2b3b5d; border-radius:14px; padding:18px; margin:16px 0; }}
.muted {{ color:#9fb2d7; }} .summary-ok {{ color:#8ce0b0; }}
figure {{ margin:0; padding:10px; background:#070d18; border-radius:10px; }}
img {{ display:block; width:100%; height:auto; border-radius:7px; }}
.language-switch {{ position:fixed; top:10px; right:12px; z-index:10; padding:8px 10px; background:#121d31; border:1px solid #2b3b5d; border-radius:10px; }}
.language-switch label {{ display:block; color:#9fb2d7; font-size:.75rem; }}
.language-switch select {{ padding:6px; color:white; background:#07101f; border:1px solid #6680b5; border-radius:7px; }}
</style></head><body>{_language_switch(language)}<main>
<h1>{_ui(language, 'JY 使用說明', 'JY guide')}</h1>
{back}
{guide_section(language)}
</main></body></html>"""
    return HTMLResponse(html, headers=SECURITY_HEADERS)


async def handle_guide_asset(request: Request, service: AgentService) -> Response:
    language = _requested_language(request)
    try:
        _approval_principal(request, service)
    except ServiceError:
        path = None
    else:
        path = guide_asset_path(
            service.settings.agent_root, str(request.path_params.get("name", ""))
        )
    if path is None:
        return _message_page(
            _ui(language, "找不到說明圖", "Guide image not found"),
            _ui(language, "沒有這張說明圖。", "There is no such guide image."),
            status_code=404,
            language=language,
        )
    return FileResponse(
        path,
        filename=path.name,
        content_disposition_type="inline",
        headers=CACHEABLE_ASSET_HEADERS,
    )


def _session_dashboard_page(
    service: AgentService,
    review: dict[str, Any],
    principal: ApprovalPrincipal,
    *,
    view: str = "home",
    error: str | None = None,
    success: str | None = None,
    status_code: int = 200,
    language: str = "zh-Hant",
    lease_preview: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
) -> HTMLResponse:
    session = review["session"]
    session_id = str(session["id"])
    workflow = session["workflow"]
    authorization = session.get("authorization")
    mode = str(session["mode"])
    pending = review.get("pending_proposal")
    notice = ""
    if error:
        notice = f'<p class="notice error">{escape(error)}</p>'
    elif success:
        notice = f'<p class="notice success">{escape(success)}</p>'
    operational_notice = _instrument_pause_notice(review, language)

    approval_html = ""
    if pending is not None and _lease_option_offer(pending) is not None:
        approval_html = _lease_approval_card(
            service, session_id, pending, language, lease_preview
        )
    elif pending is not None:
        proposal_id = str(pending["id"])
        csrf = service.approval_csrf_token(proposal_id)
        confirmation = f"APPROVE {proposal_id}"
        approval_html = f"""
        <section class="card approval">
          <h2>{_ui(language, '待核准動作', 'Action awaiting approval')}</h2>
          {_proposed_action_section(pending, language)}
          <p>{_ui(language, '確認內容後輸入以下完整文字：', 'Review the action, then enter the complete text below:')}</p>
          <code class="confirmation">{escape(confirmation)}</code>
          <form method="post" action="/session/{escape(session_id)}/approval?lang={escape(language)}">
            <input type="hidden" name="operation" value="approve">
            <input type="hidden" name="proposal_id" value="{escape(proposal_id)}">
            <input type="hidden" name="csrf_token" value="{escape(csrf)}">
            {_lease_options_fields(pending, language)}
            <label for="confirmation">{_ui(language, '核准文字', 'Confirmation')}</label>
            <input id="confirmation" name="confirmation" type="text"
                   autocomplete="off" autocapitalize="off" spellcheck="false" required>
            <button type="submit">{_ui(language, '核准', 'Approve')}</button>
          </form>
        </section>"""
    elif mode == "conversational":
        approval_html = (
            f'<section class="card"><h2>{_ui(language, "核准", "Approval")}</h2>'
            f'<p>{_ui(language, "目前沒有待核准動作。下一個實驗提案建立後，此區會自動更新。", "No action is awaiting approval. This section updates when the next experiment is proposed.")}</p></section>'
        )
    elif authorization is not None:
        approval_html = f"""
        <section class="card approval"><h2>{_ui(language, '有限自動量測授權', 'Bounded automatic measurement authorization')}</h2>
          <p>{_ui(language, '本工作階段採一次核准：授權有效且量測未超出目標、節點、期限與安全參數時，後續實驗不再逐次要求人工同意。', 'This session uses one approval. Later experiments do not require per-run approval while they remain within the authorized targets, nodes, time window, and safety limits.')}</p>
          <dl>
            <dt>{_ui(language, '授權', 'Authorization')}</dt><dd>{escape(str(authorization.get('id')))}</dd>
            <dt>{_ui(language, '狀態', 'Status')}</dt><dd>{escape(_localized_enum(language, authorization.get('status')))}</dd>
            <dt>{_ui(language, '目標', 'Targets')}</dt><dd>{escape(', '.join(authorization.get('targets', [])))}</dd>
            <dt>{_ui(language, '允許的節點', 'Allowed nodes')}</dt><dd>{escape(', '.join(authorization.get('allowed_nodes', [])))}</dd>
            {_workflow_options_rows(workflow, language)}
            <dt>{_ui(language, '每個節點／qubit 上限', 'Per node/qubit limit')}</dt><dd>{escape(str(authorization.get('max_attempts_per_node_qubit')))}</dd>
            <dt>{_ui(language, '到期時間', 'Expires')}</dt><dd>{escape(_format_taipei_time(authorization.get('expires_at')))}</dd>
          </dl>
          <p class="muted">{_ui(language, '只有超出原授權範圍時，系統才會顯示新的人工核准要求。', 'A new human approval appears only when an action exceeds the existing authorization.')}</p>
        </section>"""

    autonomy_controls = ""
    if authorization is not None and authorization["status"] in {"active", "paused"}:
        lease_id = str(authorization["id"])
        csrf = service.autonomy_csrf_token(lease_id)
        pause_or_resume = (
            f'<button name="action" value="pause" class="pause">{_ui(language, "目前實驗完成後暫停排程", "Pause scheduling after the current run")}</button>'
            if authorization["status"] == "active"
            else f'<button name="action" value="resume" class="resume">{_ui(language, "繼續排程", "Resume scheduling")}</button>'
        )
        autonomy_controls = f"""
        <section class="card"><h2>{_ui(language, '自動授權控制', 'Automation authorization controls')}</h2>
          <p>{_ui(language, '暫停會讓目前實驗完成後不再排新實驗；繼續可恢復同一份未過期授權。', 'Pause lets the current run finish and prevents new scheduling; Resume continues the same unexpired authorization.')}</p>
          <form method="post" action="/session/{escape(session_id)}/home?lang={escape(language)}">
            <input type="hidden" name="operation" value="control">
            <input type="hidden" name="csrf_token" value="{escape(csrf)}">
            {pause_or_resume}
            <button name="action" value="stop" class="stop">{_ui(language, '結束自動授權（保留網站）', 'End automation authorization (keep site)')}</button>
            <p class="muted">{_ui(language, '結束自動授權會撤銷本次 lease，要求目前 worker 安全停止，但保留 workflow、Dashboard 與 server；之後需重新授權才能自動續跑。', 'Ending automation revokes this lease and asks the current worker to stop gracefully, while keeping the workflow, Dashboard, and servers. Automatic work requires a new authorization afterward.')}</p>
            <button name="action" value="emergency_stop" class="emergency">{_ui(language, '緊急停止 worker（保留網站）', 'Emergency-stop worker (keep site)')}</button>
            <p class="warning">{_ui(language, '緊急停止會先要求安全停止，逾時後強制終止 worker；Dashboard 與 server 保持開啟，hardware lock 可能保留並要求現場復原。', 'Emergency stop first requests a graceful stop, then force-terminates the worker after the grace period. The Dashboard and servers stay open; the hardware lock may remain and require local recovery.')}</p>
          </form>
        </section>"""

    shutdown = session.get("shutdown") or {}
    shutdown_status = str(shutdown.get("status") or "")
    shutdown_control = ""
    if (
        session.get("status") != "stopped"
        and shutdown_status != "dispatching"
    ) or shutdown_status == "failed":
        shutdown_csrf = service.dashboard_shutdown_csrf_token(session_id)
        shutdown_confirmation = service.dashboard_shutdown_confirmation(session_id)
        shutdown_notice = ""
        if shutdown_status:
            shutdown_notice = (
                f"<p>{_ui(language, '關閉狀態', 'Shutdown status')}："
                f"{escape(_localized_enum(language, shutdown_status))}</p>"
            )
        if shutdown_status == "failed":
            shutdown_notice += f"""
            <p class="notice error">{_ui(language, '上一次關閉服務的啟動程序失敗；服務仍在運作。請檢查下列錯誤後按一次按鈕重試。', 'The previous service-stop launch failed, so the services are still running. Review the error below and press the button once to retry.')}</p>
            <code class="confirmation">{escape(str(shutdown.get('error') or 'No error detail was recorded.'))}</code>"""
        recovery_help = ""
        if session.get("status") == "recovery_required" or shutdown_status in {
            "recovery_required",
            "quarantine_ready",
        }:
            recovery_help = f"""
            <p class="warning">{_ui(language, 'Hardware lock 會被保留為 recovery quarantine，但不再阻擋網站與 MCP 關機。下次服務只會以本機 recovery-only 模式啟動；請在實驗室電腦使用下列操作頁完成檢查。', 'The hardware lock is retained as a recovery quarantine but no longer blocks Dashboard/MCP shutdown. The next start is local recovery-only; inspect it from the lab PC at the operator page below.')}</p>
            <code class="confirmation">{escape(str(session.get('operator_console_url')))}</code>"""
        shutdown_control = f"""
        <section class="card shutdown"><h2>{_ui(language, '完整結束量測', 'End measurement completely')}</h2>
          {shutdown_notice}
          <p>{_ui(language, '此動作會安全停止或等待 active run，永久結束 workflow，並由本機協調器關閉 MCP service、Dashboard 與 tunnel。即使自動授權已 halted，這個按鈕仍可使用。', 'This safely stops or waits for the active run, permanently ends the workflow, and asks the local coordinator to stop MCP, Dashboard, and the tunnel. It remains available when automation is halted.')}</p>
          {recovery_help}
          <p>{_ui(language, '為避免誤觸，請輸入以下完整文字：', 'To prevent accidental shutdown, enter the complete text below:')}</p>
          <code class="confirmation">{escape(shutdown_confirmation)}</code>
          <form method="post" action="/session/{escape(session_id)}/home?lang={escape(language)}">
            <input type="hidden" name="operation" value="shutdown">
            <input type="hidden" name="csrf_token" value="{escape(shutdown_csrf)}">
            <label for="shutdown-confirmation">{_ui(language, '關機確認文字', 'Shutdown confirmation')}</label>
            <input id="shutdown-confirmation" name="confirmation" type="text"
                   autocomplete="off" autocapitalize="off" spellcheck="false" required>
            <button type="submit" class="shutdown-button">{_ui(language, '結束量測並關閉所有 JY 服務', 'End measurement and stop all JY services')}</button>
          </form>
        </section>"""
    elif shutdown_status:
        shutdown_control = f"""
        <section class="card"><h2>{_ui(language, '完整結束量測', 'Full shutdown')}</h2>
          <p>{_ui(language, '關閉狀態', 'Shutdown status')}：{escape(_localized_enum(language, shutdown_status))}</p>
        </section>"""

    summary_html = _session_summary_section(
        session_id, review.get("history", []), language
    )
    history_html = _session_history_section(
        session_id, review.get("history", []), language
    )
    revision = int(review.get("event_revision", 0))
    events_url = (
        ""
        if view == "guide"
        else f"/session/{session_id}/{view}/events?after={revision}"
    )
    authorization_status = (
        str(authorization.get("status"))
        if authorization is not None
        else _ui(language, "逐次核准", "per-run approval")
    )
    pending_count = 1 if pending is not None else 0
    result_count = len(review.get("history", []))
    home_path = f"/session/{session_id}/home?lang={language}"
    approval_path = f"/session/{session_id}/approval?lang={language}"
    results_path = f"/session/{session_id}/results?lang={language}"
    usage_path = f"/session/{session_id}/usage?lang={language}"
    guide_path = f"/session/{session_id}/guide?lang={language}"
    nav = f"""
    <nav class="dashboard-nav" aria-label="{_ui(language, 'Dashboard 分頁', 'Dashboard pages')}">
      <a class="{'active' if view == 'home' else ''}" href="{escape(home_path)}">{_ui(language, '首頁', 'Home')}</a>
      <a class="{'active' if view == 'approval' else ''}" href="{escape(approval_path)}">{_ui(language, '核准', 'Approval')} <span class="badge">{pending_count}</span></a>
      <a class="{'active' if view == 'results' else ''}" href="{escape(results_path)}">{_ui(language, '實驗結果', 'Experiment results')} <span class="badge">{result_count}</span></a>
      <a class="{'active' if view == 'usage' else ''}" href="{escape(usage_path)}">{_ui(language, 'Token 用量', 'Token usage')}</a>
      <a class="{'active' if view == 'guide' else ''}" href="{escape(guide_path)}">{_ui(language, '使用說明', 'Guide')}</a>
    </nav>"""
    status_html = f"""
    <section class="card"><dl>
    <dt>{_ui(language, '模式', 'Mode')}</dt><dd>{escape(_localized_enum(language, mode))}</dd>
    <dt>{_ui(language, '授權狀態', 'Authorization status')}</dt><dd>{escape(_localized_enum(language, authorization_status))}</dd>
    <dt>{_ui(language, '工作階段狀態', 'Session status')}</dt><dd>{escape(_localized_enum(language, session.get('status')))}</dd>
    <dt>{_ui(language, 'Workflow 狀態', 'Workflow status')}</dt><dd>{escape(_localized_enum(language, workflow.get('status')))}</dd>
    <dt>{_ui(language, '工作流程', 'Workflow')}</dt><dd>{escape(str(workflow['id']))}</dd>
    <dt>{_ui(language, '目前節點', 'Current node')}</dt><dd>{escape(str(workflow.get('current_node')))}</dd>
    <dt>{_ui(language, '目標', 'Targets')}</dt><dd>{escape(', '.join(workflow.get('targets', [])))}</dd>
    <dt>{_ui(language, '工作階段開始時間', 'Session started')}</dt><dd>{escape(_format_taipei_time(session['started_at']))}</dd>
    </dl></section>"""
    device_html = f"""
    <section class="card"><h2>{_ui(language, '連接其他裝置', 'Connect another device')}</h2>
    <p>{_ui(language, '在手機或另一台瀏覽器開啟同一個 Dashboard 網址，輸入實驗室固定密碼（存於實驗室電腦的 runtime/dashboard-password.txt）。每個瀏覽器各自登入，有效 12 小時。不要把密碼放入網址或聊天訊息。', 'Open this same Dashboard URL on the phone or other browser and enter the shared lab password (kept on the lab PC in runtime/dashboard-password.txt). Each browser signs in separately for 12 hours. Never put the password in a URL or a chat message.')}</p>
    </section>"""
    continuity_html = f"""
    <section class="card"><h2>{_ui(language, 'AI／網路中斷時怎麼辦', 'If the AI app or network disconnects')}</h2>
    <ol>
      <li>{_ui(language, 'Dashboard 的暫停、停止與完整關機不依賴 AI 對話；手機或電腦仍能開啟頁面時，直接使用頁面控制即可。', 'Dashboard pause, stop, and full-shutdown controls do not depend on the AI conversation. If the page remains reachable, use those controls directly.')}</li>
      <li>{_ui(language, '網路恢復後，可在原對話輸入「結束量測」；英文可輸入 “End measurement”，系統會再次確認並完成安全關機。', 'After connectivity returns, send “End measurement” in the original conversation so JY verifies and completes safe shutdown.')}</li>
      <li>{_ui(language, '若下一次進入量測模式因殘留程式或工作階段失敗，輸入「恢復」；英文可輸入 “Recover”。此命令會安全收尾並關閉可驗證的 JY 程式，不會強制刪除 hardware lock。', 'If the next entry fails because a program or session was left open, send “Recover”. It safely closes verified JY state and programs without force-deleting a hardware lock.')}</li>
    </ol>
    </section>"""
    if view == "guide":
        page_title = _ui(language, "JY 使用說明", "JY guide")
        page_intro = _ui(language, "JY agent 的操作方式、量測順序，以及每個節點成功時應該看到的圖。", "How to operate the JY agent, the measurement order, and what each node looks like when it succeeds.")
        page_content = guide_section(language)
    elif view == "approval":
        page_title = _ui(language, "JY 核准", "JY approval")
        page_intro = _ui(language, "此頁只處理待核准動作；新提案出現時才會自動更新。", "This page is only for pending approvals and refreshes when a proposal changes.")
        page_content = approval_html
    elif view == "usage":
        page_title = _ui(language, "JY Token 用量", "JY token usage")
        page_intro = _ui(language, "各晶片與此晶片每個實驗使用的模型、token 數與估計費用，可選時段計算；從本機 Claude Code 對話紀錄統計，數字永久存在 JY 資料庫。", "Model, tokens and estimated cost per chip and per experiment on this chip, for any time range, counted from the local Claude Code transcripts; the numbers are kept in the JY database.")
        page_content = _usage_section(service, session_id, language, query or {})
    elif view == "results":
        page_title = _ui(language, "JY 實驗結果", "JY experiment results")
        page_intro = _ui(language, "所有實驗結果會保留到服務關閉；此頁只在結果或判斷改變時更新。暫停、停止與關機控制已移到首頁。", "All results remain until the service closes. This page refreshes only for results or decisions. Pause, stop, and shutdown controls are on the Home page.")
        # Operator request 2026-10-13: the chip's successful results across
        # all workflows lead the page, above this session's results.
        page_content = (
            _operator_notice_section(session_id, review.get("history", []), language)
            + _chip_history_html(service, session_id, language)
            + summary_html
            + history_html
        )
    else:
        page_title = _ui(language, "JY 量測首頁", "JY measurement home")
        page_intro = _ui(language, "手機與電腦共用此工作階段入口；量測控制在本頁，核准與結果使用下方分頁。AI 對話仍在原本的 App。", "Phones and computers share this session entry. Measurement controls are on this page; approval and results use the pages below. AI conversation remains in the original app.")
        page_content = (
            status_html
            + autonomy_controls
            + shutdown_control
            + continuity_html
            + device_html
        )
    html = f"""<!doctype html>
<html lang="{escape(language)}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script defer src="/assets/language.js"></script>
<script defer src="/assets/session-dashboard.js"></script>
<title>{escape(page_title)}</title>
<style>
:root {{ color-scheme:dark; font-family:system-ui,-apple-system,sans-serif; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#08101d; color:#edf4ff; line-height:1.55; }}
main {{ max-width:1040px; margin:24px auto; padding:0 16px 52px; }}
.language-switch {{ position:fixed; top:10px; right:12px; z-index:10; padding:8px 10px; background:#121d31; border:1px solid #2b3b5d; border-radius:10px; }}
.language-switch label {{ display:block; color:#9fb2d7; font-size:.75rem; margin-bottom:3px; }}
.language-switch select {{ width:auto; padding:6px 28px 6px 8px; }}
.card {{ background:#121d31; border:1px solid #2b3b5d; border-radius:14px; padding:18px; margin:16px 0; }}
dl {{ display:grid; grid-template-columns:160px 1fr; gap:8px 14px; }} dd {{ margin:0; overflow-wrap:anywhere; }}
dt,.muted {{ color:#9fb2d7; }}
pre {{ overflow:auto; white-space:pre-wrap; word-break:break-word; background:#070d18; padding:14px; border-radius:8px; }}
.result-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr)); gap:14px; }}
figure {{ margin:0; padding:10px; background:#070d18; border-radius:10px; }}
img {{ display:block; width:100%; height:auto; border-radius:7px; }}
.summary-list {{ margin:0; padding:0; list-style:none; }}
.chip-node {{ border-top:1px solid #21304d; padding:10px 0; }}
.chip-node > summary {{ cursor:pointer; font-size:1.05rem; }}
.chip-qubit {{ margin:10px 0 14px 14px; }}
.chip-qubit h4 {{ margin:6px 0; }}
.chip-values {{ list-style:none; padding:0; margin:4px 0 8px; display:flex; flex-wrap:wrap; gap:4px 18px; }}
.chip-qubit details {{ margin-left:8px; }}
figcaption {{ font-size:.75rem; overflow-wrap:anywhere; }}
.summary-row {{ display:grid; grid-template-columns:minmax(150px,230px) 1fr; gap:6px 16px; align-items:start; padding:12px 0; border-top:1px solid #21304d; }}
.summary-row:first-child {{ border-top:0; padding-top:4px; }}
.summary-name a {{ color:#cfe0ff; font-weight:700; text-decoration:none; overflow-wrap:anywhere; }}
.summary-name a:hover {{ text-decoration:underline; }}
.summary-row p {{ margin:0 0 6px; }}
.summary-ok {{ color:#8ce0b0; overflow-wrap:anywhere; }}
.summary-thumbs {{ display:flex; flex-wrap:wrap; gap:8px; }}
.summary-thumbs a {{ display:block; width:112px; }}
.summary-thumbs img {{ width:112px; border:1px solid #2b3b5d; background:#070d18; }}
select,input,button {{ width:100%; padding:12px; border-radius:8px; font:inherit; }}
select,input {{ color:white; background:#07101f; border:1px solid #6680b5; }}
button {{ margin-top:12px; border:0; color:white; background:#3568e8; font-weight:750; cursor:pointer; }}
.pause {{ background:#9a6700; }} .resume {{ background:#16794a; }} .stop {{ background:#b4232d; }}
.emergency {{ background:#ff1738; border:2px solid white; }}
.shutdown-button {{ background:#b4232d; border:2px solid #ffd6dc; }}
.warning {{ padding:12px; border-radius:8px; background:#5b3a12; color:#ffe7b0; }}
.instrument-alert {{ border-color:#f0a93b; background:#3b2b15; }}
.confirmation {{ display:block; padding:12px; background:#070d18; border-radius:8px; user-select:all; overflow-wrap:anywhere; }}
.lease-options {{ border:1px solid #2b3b5d; border-radius:10px; padding:10px 14px; margin:14px 0; }} .lease-options legend {{ padding:0 6px; font-weight:700; }} .lease-options .choice {{ display:flex; gap:10px; align-items:flex-start; margin:8px 0; }} .lease-options .choice input {{ width:auto; margin-top:5px; flex:none; }} .warning {{ color:#ffd479; }}
.notice {{ padding:12px; border-radius:8px; }} .error {{ background:#5a1e2a; }} .success {{ background:#14532d; }}
.dashboard-nav {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:18px 0; }}
.usage-table {{ width:100%; border-collapse:collapse; font-size:.9rem; font-variant-numeric:tabular-nums; }}
.usage-table th,.usage-table td {{ padding:6px 8px; border-top:1px solid #21304d; text-align:right; white-space:nowrap; }}
.usage-table th:first-child,.usage-table td:first-child,.usage-table td.text,.usage-table th.text {{ text-align:left; }}
.usage-table th {{ color:#9fb2d7; font-weight:600; }}
.table-scroll {{ overflow-x:auto; }}
.usage-totals {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }}
.usage-totals div {{ background:#070d18; border-radius:10px; padding:10px 12px; }}
.usage-totals strong {{ display:block; font-size:1.25rem; }}
.usage-totals small {{ display:block; color:#9fb2d7; }}
.usage-table tr.usage-day td {{ background:#16223a; font-weight:700; border-top:2px solid #2b3b5d; }}
.usage-table tr.usage-current td {{ background:#1b3263; }}
.usage-table details summary {{ cursor:pointer; list-style:none; }}
.usage-table details span {{ display:block; color:#9fb2d7; font-size:.8rem; }}
.usage-presets {{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 12px; }}
.usage-presets a {{ color:#cfe0ff; background:#121d31; border:1px solid #2b3b5d; border-radius:999px; padding:6px 14px; text-decoration:none; }}
.usage-presets a.active {{ background:#234b9d; border-color:#7ba2ff; color:white; }}
.usage-range {{ display:grid; grid-template-columns:1fr 1fr auto; gap:10px; align-items:end; }}
.usage-range button {{ margin-top:0; width:auto; }}
@media(max-width:600px) {{ .usage-range {{ grid-template-columns:1fr; }} }}
.dashboard-nav a {{ color:#cfe0ff; background:#121d31; border:1px solid #2b3b5d; border-radius:10px; padding:12px; text-align:center; text-decoration:none; }}
.dashboard-nav a.active {{ background:#234b9d; border-color:#7ba2ff; color:white; }}
.badge {{ display:inline-block; min-width:1.5em; margin-left:4px; padding:0 .4em; border-radius:999px; background:#07101f; }}
[hidden] {{ display:none !important; }}
@media(max-width:600px) {{ dl {{ grid-template-columns:1fr; gap:2px; }} dd {{ margin-bottom:8px; }} .dashboard-nav {{ grid-template-columns:1fr; }} .summary-row {{ grid-template-columns:1fr; }} }}
</style></head>
<body data-session-events="{escape(events_url)}">{_language_switch(language)}<main>
<h1>{escape(page_title)}</h1>
<p class="muted">{escape(page_intro)}</p>
<p>{_ui(language, '審閱者', 'Reviewer')}：{escape(_principal_label(principal, language))}</p>{notice}
{nav}
{operational_notice}
{page_content}
</main></body></html>"""
    return HTMLResponse(html, status_code=status_code, headers=SECURITY_HEADERS)


MODEL_LABELS = {
    "claude-opus-5-5": "Opus 5.5",
    "claude-sonnet-5-5": "Sonnet 5.5",
    "claude-fable-5-1": "Fable 5.1",
    "claude-haiku-4-5": "Haiku 4.5",
}


USAGE_PRESETS = ("chip", "session", "today", "7d", "month")


def _usage_range(
    query: dict[str, str], now: datetime
) -> tuple[str, datetime | None, datetime | None, str | None]:
    """(preset, start, end, error) from the page query, in Taipei civil time.

    Operator choice 2026-10-14: presets for the whole chip, this session,
    today, the last 7 days and this month, or a custom from/to pair. The
    "to" minute is included.
    """
    raw_from = (query.get("from") or "").strip()
    raw_to = (query.get("to") or "").strip()
    preset = (query.get("range") or "").strip()
    if preset not in USAGE_PRESETS:
        preset = "custom" if raw_from or raw_to else "chip"
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if preset == "today":
        return preset, midnight, None, None
    if preset == "7d":
        return preset, now - timedelta(days=7), None, None
    if preset == "month":
        return preset, midnight.replace(day=1), None, None
    if preset != "custom":
        return preset, None, None, None

    def parse(value: str) -> datetime | None:
        if not value:
            return None
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=TAIPEI_TIMEZONE)

    try:
        start = parse(raw_from)
        end = parse(raw_to)
    except ValueError:
        return "chip", None, None, "time"
    if end is not None:
        end += timedelta(minutes=1)
    if start is not None and end is not None and end <= start:
        return "chip", None, None, "order"
    return preset, start, end, None


def _usage_section(
    service: AgentService,
    session_id: str,
    language: str,
    query: dict[str, str] | None = None,
) -> str:
    """Chip-wide model, token and estimated-cost page (2026-10-03, 2026-10-14)."""
    from .usage import total_tokens

    now = datetime.now(TAIPEI_TIMEZONE)
    preset, start, end, range_error = _usage_range(query or {}, now)
    try:
        usage = service.chip_usage(
            session_id,
            start,
            end,
            session_only=preset == "session",
            day_zone=TAIPEI_TIMEZONE,
        )
    except Exception as exc:  # the page must never break the Dashboard
        return f'<section class="card"><p class="notice error">{escape(str(exc))}</p></section>'
    if usage.get("error"):
        return f'<section class="card"><p class="notice error">{escape(str(usage["error"]))}</p></section>'
    overall, ranged = usage["overall"], usage["range"]

    def number(value: Any) -> str:
        return f"{int(value):,}"

    def dollars(bucket: dict[str, Any]) -> str:
        text = f"${float(bucket.get('cost_usd', 0.0)):.2f}"
        if bucket.get("unpriced_calls"):
            text += "*"
        return text

    def models(bucket: dict[str, Any]) -> str:
        return ", ".join(
            MODEL_LABELS.get(name, name) for name in sorted(bucket.get("models", {}))
        ) or "-"

    def tokens(bucket: dict[str, Any]) -> str:
        writes = bucket.get("cache_write_5m", 0) + bucket.get("cache_write_1h", 0)
        detail = (
            f"Input {number(bucket.get('input', 0))} · "
            f"{_ui(language, 'Cache 寫入', 'Cache write')} {number(writes)} · "
            f"{_ui(language, 'Cache 讀取', 'Cache read')} {number(bucket.get('cache_read', 0))} · "
            f"Output {number(bucket.get('output', 0))} · "
            f"{_ui(language, '呼叫', 'Calls')} {number(bucket.get('calls', 0))}"
        )
        return (
            f"<details><summary>{number(total_tokens(bucket))}</summary>"
            f"<span>{escape(detail)}</span></details>"
        )

    def cells(bucket: dict[str, Any]) -> str:
        return (
            f'<td class="text">{escape(models(bucket))}</td>'
            f"<td>{tokens(bucket)}</td>"
            f"<td>{number(bucket.get('output', 0))}</td>"
            f"<td>{dollars(bucket)}</td>"
        )

    def tile(label: str, value: str, note: str = "") -> str:
        small = f"<small>{escape(note)}</small>" if note else ""
        return f'<div><span class="muted">{escape(label)}</span><strong>{value}</strong>{small}</div>'

    def experiments_cost(result: dict[str, Any]) -> float:
        return sum(float(row["cost_usd"]) for row in result["runs"])

    def local(value: Any) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(TAIPEI_TIMEZONE)

    head = (
        f'<th class="text">{_ui(language, "模型", "Model")}</th>'
        f"<th>{_ui(language, '總 token', 'Total tokens')}</th>"
        f"<th>Output</th>"
        f"<th>{_ui(language, '估計費用', 'Est. cost')}</th>"
    )

    # Operator request 2026-10-15: every chip measured so far, newest first.
    def civil_date(value: Any) -> str:
        parsed = local(value)
        return parsed.strftime("%Y-%m-%d") if parsed else "-"

    chip_rows = usage.get("chips") or []
    chip_lines = "".join(
        f'<tr class="{"usage-current" if row["device_key"] == usage.get("device_key") else ""}">'
        f'<td class="text">{escape(str(row["chip_name"]))}</td>'
        f'<td class="text">{escape(civil_date(row["first_at"]))}</td>'
        f'<td class="text">{escape(civil_date(row["last_at"]))}</td>'
        f'<td>{number(row["workflows"])}</td><td>{number(row["runs"])}</td>{cells(row)}</tr>'
        for row in chip_rows
    )
    all_chips: dict[str, Any] = {
        key: sum(row[key] for row in chip_rows)
        for key in ("calls", "cost_usd", "unpriced_calls", "input", "cache_write_5m", "cache_write_1h", "cache_read", "output")
    }
    all_chips["models"] = {name: 1 for row in chip_rows for name in row["models"]}
    chip_total = (
        f'<tr class="usage-day"><td class="text" colspan="3">{_ui(language, "全部晶片", "All chips")}</td>'
        f'<td>{number(sum(row["workflows"] for row in chip_rows))}</td>'
        f'<td>{number(sum(row["runs"] for row in chip_rows))}</td>{cells(all_chips)}</tr>'
        if len(chip_rows) > 1
        else ""
    )
    chips = f"""
    <section class="card"><h2>{_ui(language, '各晶片總用量', 'Usage by chip')}</h2>
    <div class="table-scroll"><table class="usage-table">
    <tr><th class="text">{_ui(language, '晶片', 'Chip')}</th><th class="text">{_ui(language, '第一次', 'First')}</th><th class="text">{_ui(language, '最近一次', 'Latest')}</th><th>Workflows</th><th>Runs</th>{head}</tr>
    {chip_lines}{chip_total}
    </table></div>
    <p class="muted">{_ui(language, '目前這個 session 的晶片以底色標示；下方明細只算這顆晶片。', "This session's chip is highlighted; the details below cover this chip only.")}</p>
    </section>""" if chip_rows else ""

    chip_started = local(overall.get("chip_started_at"))
    chip = f"""
    <section class="card"><h2>{_ui(language, '晶片', 'Chip')} {escape(str(usage.get('chip_name')))}</h2>
    <div class="usage-totals">
      {tile(_ui(language, '開始時間', 'Started'), escape(chip_started.strftime('%Y-%m-%d %H:%M') if chip_started else '-'), 'Asia/Taipei')}
      {tile(_ui(language, '估計總費用', 'Estimated total'), dollars(overall['total']))}
      {tile(_ui(language, '總 token', 'Total tokens'), number(total_tokens(overall['total'])), f"Output {number(overall['total']['output'])}")}
      {tile(_ui(language, '實驗', 'Experiments'), f"${experiments_cost(overall):.2f}", _ui(language, f"{len(overall['runs'])} 次", f"{len(overall['runs'])} runs"))}
      {tile(_ui(language, '量測前準備', 'Before runs'), dollars(overall['setup']))}
      {tile(_ui(language, '程式開發', 'Development'), dollars(overall['development']))}
      {tile(_ui(language, '對話／其他', 'Conversation'), dollars(overall['conversation']))}
      {tile(_ui(language, '模型', 'Models'), escape(models(overall['total'])), _ui(language, f"{overall['workflows']} 個 workflow", f"{overall['workflows']} workflows"))}
    </div></section>"""

    preset_labels = {
        "chip": _ui(language, "整個晶片", "Whole chip"),
        "session": _ui(language, "本 session", "This session"),
        "today": _ui(language, "今天", "Today"),
        "7d": _ui(language, "近 7 天", "Last 7 days"),
        "month": _ui(language, "本月", "This month"),
    }
    presets = "".join(
        f'<a class="{"active" if key == preset else ""}" '
        f'href="?lang={escape(language)}&amp;range={key}">{label}</a>'
        for key, label in preset_labels.items()
    )
    field_from = start.astimezone(TAIPEI_TIMEZONE).strftime("%Y-%m-%dT%H:%M") if start else ""
    field_to = (
        (end - timedelta(minutes=1)).astimezone(TAIPEI_TIMEZONE).strftime("%Y-%m-%dT%H:%M")
        if end
        else ""
    )
    error_text = {
        "time": _ui(language, "時間格式無法辨識，改顯示整個晶片。", "The time could not be read; showing the whole chip."),
        "order": _ui(language, "結束時間必須晚於開始時間，改顯示整個晶片。", "The end must be after the start; showing the whole chip."),
    }.get(range_error or "")
    error_html = f'<p class="notice error">{escape(error_text)}</p>' if error_text else ""
    if preset == "chip":
        range_text = _ui(language, "整個晶片（從第一個 workflow 開始）", "The whole chip (since its first workflow)")
    elif preset == "session":
        range_text = _ui(language, "本 session 的 workflow", "This session's workflow")
    else:
        range_text = (
            f"{field_from.replace('T', ' ') or _ui(language, '最早', 'earliest')} → "
            f"{(end.astimezone(TAIPEI_TIMEZONE).strftime('%Y-%m-%d %H:%M') if end else _ui(language, '現在', 'now'))}"
            " (Asia/Taipei)"
        )
    everything = ranged["all_usage"]
    range_card = f"""
    <section class="card"><h2>{_ui(language, '時段', 'Time range')}</h2>
    <nav class="usage-presets">{presets}</nav>
    <form method="get" class="usage-range">
      <input type="hidden" name="lang" value="{escape(language)}">
      <label>{_ui(language, '起', 'From')}<input type="datetime-local" name="from" value="{escape(field_from)}"></label>
      <label>{_ui(language, '迄（含該分鐘，空白＝現在）', 'To (inclusive, blank = now)')}<input type="datetime-local" name="to" value="{escape(field_to)}"></label>
      <button type="submit">{_ui(language, '計算', 'Calculate')}</button>
    </form>
    {error_html}
    <p class="muted">{escape(range_text)}</p>
    <div class="usage-totals">
      {tile(_ui(language, '此時段（此晶片）', 'In range (this chip)'), dollars(ranged['total']), _ui(language, f"{number(total_tokens(ranged['total']))} tokens", f"{number(total_tokens(ranged['total']))} tokens"))}
      {tile(_ui(language, '其中實驗', 'Of which experiments'), f"${experiments_cost(ranged):.2f}", _ui(language, f"{len(ranged['runs'])} 次", f"{len(ranged['runs'])} runs"))}
      {tile(_ui(language, '此時段全部用量', 'All usage in range'), dollars(everything['total']), _ui(language, '含其他晶片、程式開發與對話', 'including other chips, development and conversation'))}
      {tile(_ui(language, '全部用量中的程式開發', 'Development (all)'), dollars(everything['development']))}
    </div></section>"""

    by_day: dict[str, list[dict[str, Any]]] = {}
    for row in ranged["runs"]:
        begun = local(row.get("started_at"))
        by_day.setdefault(begun.date().isoformat() if begun else "-", []).append(row)
    day_keys = sorted(set(by_day) | set(ranged["days"]), reverse=True)
    empty = {"calls": 0, "cost_usd": 0.0, "models": {}}
    day_rows = []
    for day in day_keys:
        rows = sorted(by_day.get(day, []), key=lambda item: str(item["started_at"]), reverse=True)
        label = _ui(language, f"{day} · {len(rows)} 個實驗", f"{day} · {len(rows)} runs")
        day_rows.append(
            f'<tr class="usage-day"><td class="text" colspan="3">{escape(label)}</td>'
            f"{cells(ranged['days'].get(day, empty))}</tr>"
        )
        for row in rows:
            begun = local(row.get("started_at"))
            day_rows.append(
                f'<tr><td class="text">{escape(begun.strftime("%H:%M:%S") if begun else "-")}</td>'
                f'<td class="text">{escape(row["node_id"])}</td>'
                f'<td class="text">{escape(",".join(row["qubits"]))}</td>{cells(row)}</tr>'
            )
    other_rows = "".join(
        f'<tr class="usage-day"><td class="text" colspan="3">{label}</td>{cells(ranged[key])}</tr>'
        for key, label in (
            ("setup", _ui(language, "量測前準備（進入、等待核准）", "Before runs (entry, approval)")),
            ("development", _ui(language, "程式開發（修改檔案的呼叫）", "Development (calls that edit files)")),
            ("conversation", _ui(language, "對話／其他", "Conversation / other")),
        )
    )
    runs = f"""
    <section class="card"><h2>{_ui(language, '依日期／實驗', 'By date and experiment')}</h2>
    <div class="table-scroll"><table class="usage-table">
    <tr><th class="text">{_ui(language, '開始時間', 'Started')}</th><th class="text">{_ui(language, '節點', 'Node')}</th><th class="text">Qubits</th>{head}</tr>
    {''.join(day_rows)}{other_rows}
    </table></div></section>"""

    node_rows = "".join(
        f'<tr><td class="text">{escape(node)}</td><td>{number(bucket["runs"])}</td>{cells(bucket)}</tr>'
        for node, bucket in ranged["nodes"].items()
    )
    nodes = f"""
    <section class="card"><h2>{_ui(language, '依節點', 'By node')}</h2>
    <div class="table-scroll"><table class="usage-table">
    <tr><th class="text">{_ui(language, '節點', 'Node')}</th><th>Runs</th>{head}</tr>
    {node_rows}
    </table></div></section>"""

    pricing = usage.get("pricing", {}).get("models", {})
    price_text = "; ".join(
        f"{MODEL_LABELS.get(name, name)} ${float(p['input']):g}/${float(p['output']):g} (cache read ${float(p['cache_read']):g})"
        for name, p in pricing.items()
    )
    notes = f"""
    <section class="card"><h2>{_ui(language, '計算方式', 'How this is counted')}</h2>
    <ul class="muted">
      <li>{_ui(language, '晶片用量是這顆晶片所有 workflow 期間內的呼叫：從 workflow 建立到它結束或下一個 workflow 建立為止。workflow 之間的空檔與沒有記錄晶片的舊 workflow 只算在「此時段全部用量」。', "Chip usage is every call inside this chip's workflows, from a workflow's creation to its close or the next workflow's creation. Gaps between workflows and old workflows without a recorded chip count only in All usage in range.")}</li>
      <li>{_ui(language, '一次呼叫歸給它發生時正在進行的實驗（從該實驗開始到下一個實驗開始），包含啟動、等待、分析、決策與 commit；修改檔案的呼叫算程式開發。從前一個 workflow 繼承的 run 沒有重新量測，不列出。', 'A call belongs to the experiment in progress when it happened (from that run start to the next), covering start, waiting, analysis, decision and commit; calls that edit files count as development. Runs inherited from an earlier workflow were not measured again and are not listed.')}</li>
      <li>{_ui(language, '日期列是該日此晶片的全部用量（含量測前準備、程式開發、對話），實驗列只算該實驗。總 token = input + cache 寫入 + cache 讀取 + output，點數字可看拆分。', "A date row is all of this chip's usage that day (including before-run, development and conversation); an experiment row is that experiment only. Total tokens = input + cache write + cache read + output; click a number for the split.")}</li>
      <li>{_ui(language, '估計費用採 Anthropic API 公開價格（每百萬 token，USD）：', 'Estimated cost uses Anthropic API list prices (USD per million tokens): ')}{escape(price_text)}{_ui(language, '；cache 寫入為 input 價格的 1.25 倍（5 分鐘）或 2 倍（1 小時）。訂閱方案的實際計費方式不同，這裡是 API 等值費用。', '; cache writes cost 1.25x (5 min) or 2x (1 h) the input price. Subscription plans are billed differently; this is the API-equivalent cost.')}</li>
      <li>{_ui(language, 'Cache 讀取通常最大：每次呼叫都會重新送出整段對話。標 * 的列含沒有價格的模型。', 'Cache reads are usually the largest part because every call resends the whole conversation. Rows marked * include a model without a price.')}</li>
      <li>{_ui(language, '用量數字（不含對話內容）會存進 JY 資料庫，Claude Code 刪掉舊對話紀錄後仍保留；每次開啟此頁與進入量測模式時更新。保存的資料從', 'Usage numbers (no conversation text) are saved in the JY database and stay after Claude Code deletes old transcripts; they update whenever this page opens or measurement mode is entered. Saved data starts at ')}{escape(civil_date(usage.get('saved_since')))}{_ui(language, ' 開始。', '.')}</li>
      <li>{_ui(language, '資料來源', 'Source')}: {escape(str(usage.get('transcript_dir')))}</li>
    </ul></section>"""
    return chips + chip + range_card + runs + nodes + notes


def _instrument_pause_notice(
    review: dict[str, Any], language: str = "zh-Hant"
) -> str:
    session = review.get("session") or {}
    authorization = session.get("authorization") or {}
    currently_paused = session.get("status") == "paused" or authorization.get(
        "status"
    ) == "paused"
    if not currently_paused:
        return ""
    for item in reversed(review.get("history", [])):
        analysis = item.get("analysis") or {}
        category = analysis.get("failure_category")
        # A probe-confirmed submission timeout only reaches this banner when
        # it escalated, which is the single-target case that does pause.
        if category == "program_submission_timeout":
            if not analysis.get("measurement_paused"):
                continue
        elif category != "instrument_unreachable":
            continue
        messages = analysis.get("operator_message") or {}
        message = messages.get(language) or messages.get("en") or _ui(
            language,
            "儀器連線失敗，本次實驗與後續排程已暫停。",
            "Instrument connectivity failed; this run and new scheduling are paused.",
        )
        recovery_required = bool(
            analysis.get("hardware_lock_recovery_required")
        )
        quarantine = (
            f"<p>{_ui(language, '本次失聯發生在硬體執行階段，軟體無法證明儀器已完全停止，因此 lock 只保留在本機安全隔離；首頁的完整關機仍可關閉 Dashboard、MCP 與 tunnel。', 'Connectivity was lost during hardware execution, so software cannot prove the instrument is fully idle. The lock remains only in local safety quarantine; Home full shutdown can still close the Dashboard, MCP service, and tunnel.')}</p>"
            if recovery_required
            else ""
        )
        return f"""
        <section class="card instrument-alert">
          <h2>{_ui(language, '儀器錯誤：量測已暫停', 'Instrument error: measurement paused')}</h2>
          <p>{escape(str(message))}</p>
          {quarantine}
          <p>{_ui(language, '現場確認儀器可連線後，請用此頁「繼續排程」，或回 AI 對話輸入「恢復量測」／“Resume measurement”；目前節點會以新 run 從頭執行。', 'After connectivity is restored, use Resume scheduling on this page, or send “Resume measurement” in the AI conversation. The current node restarts as a new run.')}</p>
        </section>"""
    return ""


def _natural_name_key(name: str) -> tuple[Any, ...]:
    """Sort q2 before q10 instead of lexicographically."""
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(name))
    )


def _successful_qubits(item: dict[str, Any]) -> list[str]:
    """Qubits this run proved successful.

    Judged per target, not per run. `passing_targets` is JY's own per-qubit
    verdict and is exactly what gates a state commit, so a run that is
    `needs_review` overall still reports the qubits that passed inside it -
    otherwise a node like 05, where nine of ten qubits passed and committed,
    would read as a failure on the summary.

    Operator instruction 2026-10-10: a failed qubit is never shown as
    successful. That rules out listing every target of a single-pass node,
    the node's own `outcomes` (Qualibrate marks every qubit "successful"),
    03a coarse/refinement candidates, and a passing qubit whose values the
    recorded decision left out of the state patch.
    """

    run = item.get("run") or {}
    analysis = item.get("analysis") or {}
    passing = analysis.get("passing_targets")
    if not isinstance(passing, list):
        return []
    if str(run.get("node_id")) == "03a" and analysis.get("03a_stage") != "fine":
        return []
    names = {str(name) for name in passing}
    decision = item.get("decision")
    if isinstance(decision, dict):
        patch = decision.get("state_patch")
        patch = patch if isinstance(patch, list) else []
        in_patch = _patch_qubits(patch)
        proposed = _patch_qubits(analysis.get("candidate_state_patch") or [])
        names -= proposed - in_patch
    return sorted(names, key=_natural_name_key)


def _summary_successes(history: list[dict[str, Any]]) -> list[list[str]]:
    """Per-run successful qubits, each qubit listed once per node.

    Operator instruction 2026-10-10: a qubit succeeds at a node once. When a
    later run of the same node passes it again (an active-reset repeat or
    verification), only the latest run lists it.
    """
    successes = [_successful_qubits(item) for item in history]
    seen: dict[str, set[str]] = {}
    for index in range(len(history) - 1, -1, -1):
        node = str((history[index].get("run") or {}).get("node_id"))
        shown = seen.setdefault(node, set())
        successes[index] = [name for name in successes[index] if name not in shown]
        shown.update(successes[index])
    return successes


def _patch_qubits(patch: Any) -> set[str]:
    names: set[str] = set()
    for item in patch if isinstance(patch, list) else []:
        match = re.match(r"^/qubits/(q[0-9]+)/", str((item or {}).get("path", "")))
        if match:
            names.add(match.group(1))
    return names


def _session_summary_section(
    session_id: str,
    history: list[dict[str, Any]],
    language: str = "zh-Hant",
) -> str:
    """List every experiment of this session with its successful qubits."""
    heading = _ui(language, "結果摘要", "Result summary")
    if not history:
        return (
            f'<section class="card"><h2>{heading}</h2>'
            f'<p>{_ui(language, "本次啟動後尚未執行任何實驗。", "No experiment has run since this session started.")}</p></section>'
        )
    separator = ", " if language == "en" else "、"
    rows: list[str] = []
    successes = _summary_successes(history)
    for ordinal, item in enumerate(history, start=1):
        run = item.get("run") or {}
        label = f'{_ui(language, "實驗", "Experiment")} {ordinal}: {run.get("node_id")}'
        target = f'?lang={escape(language)}&amp;experiment={ordinal}'
        qubits = successes[ordinal - 1]
        if qubits:
            asset_indices = list(item.get("asset_indices", []))
            shown = asset_indices[:SUMMARY_THUMBNAIL_LIMIT]
            thumbnails = "".join(
                f'<a href="{target}" data-experiment-link="{ordinal}">'
                f'<img src="{_asset_url(session_id, item, plot_number - 1, thumb=True)}" '
                f'alt="{escape(_ui(language, "實驗結果圖", "Experiment result plot"))} {ordinal}-{plot_number}" '
                f'loading="lazy"></a>'
                for plot_number, _asset_index in enumerate(shown, start=1)
            )
            if thumbnails:
                remaining = len(asset_indices) - len(shown)
                extra = (
                    f'<p class="muted">{_ui(language, "另有", "Plus")} {remaining} '
                    f'{_ui(language, "張結果圖", "more plots")}</p>'
                    if remaining > 0
                    else ""
                )
                figures = f'<div class="summary-thumbs">{thumbnails}</div>{extra}'
            else:
                figures = (
                    f'<p class="muted">{_ui(language, "此筆實驗沒有可用的結果圖。", "No result plot is available for this experiment.")}</p>'
                )
            outcome = (
                f'<p class="summary-ok">{_ui(language, "成功：", "Successful: ")}'
                f'{escape(separator.join(qubits))}</p>{figures}'
            )
        elif str(run.get("status")) in {"starting", "running", "stopping"}:
            outcome = (
                f'<p class="muted">{escape(_localized_enum(language, run.get("status")))}</p>'
            )
        else:
            outcome = f'<p class="muted">{_ui(language, "未成功", "Not successful")}</p>'
        rows.append(
            f'<li class="summary-row"><div class="summary-name">'
            f'<a href="{target}" data-experiment-link="{ordinal}">{escape(label)}</a>'
            f'</div><div>{outcome}</div></li>'
        )
    return (
        f'<section class="card"><h2>{heading}</h2>'
        f'<p class="muted">{_ui(language, "只列出 JY 判定通過且已寫入 state 的 qubit，每個 node 每顆 qubit 只列一次；點任一列可跳到該次實驗的完整結果。", "Only qubits that JY passed and that were committed to the state are listed, once per node. Select a row to open that experiment in full.")}</p>'
        f'<ol class="summary-list">{"".join(rows)}</ol></section>'
    )


def _operator_notice_section(
    session_id: str,
    history: list[dict[str, Any]],
    language: str = "zh-Hant",
) -> str:
    """Findings that need a person, such as a 02b avoided crossing.

    Operator request 2026-10-08: show the picture and say what to do.
    """

    cards: list[str] = []
    for ordinal, item in enumerate(history, start=1):
        analysis = item.get("analysis") or {}
        notices = analysis.get("operator_notices") or []
        plots = [str(path) for path in item.get("plots") or []]
        for notice in notices:
            if not isinstance(notice, dict):
                continue
            qubit = str(notice.get("qubit") or "")
            messages = notice.get("message") or {}
            message = messages.get(language) or messages.get("en") or ""
            figures = "".join(
                f'<figure><img src="{_asset_url(session_id, item, position)}" '
                f'alt="{escape(qubit)} avoided crossing" loading="lazy"></figure>'
                for position, path in enumerate(plots)
                if "avoided-crossing" in Path(path).name
                and f"-{qubit}-" in Path(path).name
                and position < len(item.get("asset_indices", []))
            )
            cards.append(
                f"""<section class="card instrument-alert">
                  <h2>{_ui(language, '需要人工處理', 'Needs a person')} · {_ui(language, '實驗', 'Experiment')} {ordinal} · {escape(qubit)}</h2>
                  <p>{escape(str(message))}</p>
                  <div class="result-grid">{figures}</div>
                </section>"""
            )
    return "".join(cards)


def _session_history_section(
    session_id: str,
    history: list[dict[str, Any]],
    language: str = "zh-Hant",
) -> str:
    if not history:
        return (
            f'<section class="card"><h2>{_ui(language, "實驗結果", "Experiment results")}</h2>'
            f'<p>{_ui(language, "目前尚無結果；有新實驗或新分析時頁面才會更新。", "No result is available yet. The page updates when a new experiment or analysis is ready.")}</p></section>'
        )
    total = len(history)
    options = "".join(
        f'<option value="{ordinal}">{_ui(language, "實驗", "Experiment")} {ordinal}</option>'
        for ordinal in range(1, total + 1)
    )
    rendered = [
        f'<section class="card"><h2>{_ui(language, "實驗結果", "Experiment results")}</h2>'
        f'<label for="experiment-selector">{_ui(language, "選擇本次啟動後的實驗序號", "Select an experiment from this session")}</label>'
        f'<select id="experiment-selector" data-experiment-selector>{options}</select>'
        '</section>'
    ]
    for ordinal, item in enumerate(history, start=1):
        run = item.get("run") or {}
        analysis = item.get("analysis") or {}
        decision = item.get("decision") or {}
        figures = "".join(
            f'<figure><img src="{_asset_url(session_id, item, plot_number - 1)}" '
            f'alt="{escape(_ui(language, "實驗結果圖", "Experiment result plot"))} {ordinal}-{plot_number}" loading="lazy">'
            f'<figcaption>{_ui(language, "結果圖", "Plot")} {plot_number}</figcaption></figure>'
            for plot_number, _asset_index in enumerate(
                item.get("asset_indices", []), start=1
            )
        )
        if not figures:
            figures = f"<p>{_ui(language, '此筆實驗尚無可用的 snapshot 結果圖。', 'No snapshot result plot is available for this experiment.')}</p>"
        evidence = {
            "analysis_status": analysis.get("analysis_status"),
            "failure_reasons": analysis.get("failure_reasons", []),
            "warnings": analysis.get("warnings", []),
            "fit_quality": analysis.get("fit_quality"),
            "dataset_metrics": analysis.get("dataset_metrics"),
        }
        next_action = {
            "next_node": decision.get("next_node"),
            "new_parameters": decision.get("next_parameters", {}),
        }
        rendered.append(f"""
        <section class="card" data-experiment-panel="{ordinal}" hidden>
          <h2>{_ui(language, '實驗', 'Experiment')} {ordinal} / {total} &middot; {escape(str(run.get('node_id')))}</h2>
          <dl>
            <dt>{_ui(language, '執行編號', 'Run')}</dt><dd>{escape(str(run.get('id')))}</dd>
            <dt>{_ui(language, '狀態', 'Status')}</dt><dd>{escape(_localized_enum(language, run.get('status')))}</dd>
            <dt>{_ui(language, '快照', 'Snapshot')}</dt><dd>{escape(str(run.get('snapshot_id')))}</dd>
            <dt>{_ui(language, '耗時', 'Elapsed')}</dt><dd>{escape(_format_seconds(run.get('elapsed_seconds')))}</dd>
          </dl>
          {_json_details(_ui(language, '時間預測', 'Prediction'), item.get('duration_prediction'))}
          {_json_details(_ui(language, '參數', 'Parameters'), run.get('parameters', {}))}
          <div class="result-grid">{figures}</div>
          <h3>{_ui(language, '判斷與理由', 'Decision and reason')}</h3>
          <dl>
            <dt>{_ui(language, '判斷', 'Decision')}</dt><dd>{escape(_localized_enum(language, decision.get('decision') or _ui(language, '待判斷', 'Pending')))}</dd>
            <dt>{_ui(language, '理由', 'Reason')}</dt><dd>{escape(str(decision.get('reason') or _ui(language, '尚未記錄判斷理由。', 'No decision reason has been recorded.')))}</dd>
          </dl>
          {_json_details(_ui(language, '分析', 'Analysis'), evidence)}
          {_json_details(_ui(language, '下一步', 'Next action'), next_action)}
        </section>""")
    return "".join(rendered)


def _autonomy_page(
    service: AgentService,
    status: dict[str, Any],
    principal: ApprovalPrincipal,
    *,
    error: str | None = None,
    success: str | None = None,
    status_code: int = 200,
    language: str = "zh-Hant",
) -> HTMLResponse:
    lease_id = str(status["id"])
    token = service.autonomy_csrf_token(lease_id)
    notice = ""
    if error:
        notice = f'<p class="notice error">{escape(error)}</p>'
    elif success:
        notice = f'<p class="notice success">{escape(success)}</p>'
    controls = ""
    if status["status"] == "active":
        controls = f"""
        <form method="post" action="/autonomy/{escape(lease_id)}?lang={escape(language)}">
          <input type="hidden" name="csrf_token" value="{escape(token)}">
          <button name="action" value="pause" class="pause">{_ui(language, '本次完成後暫停', 'Pause after current run')}</button>
          <button name="action" value="stop" class="stop">{_ui(language, '安全停止', 'Graceful stop')}</button>
          <button name="action" value="emergency_stop" class="emergency">{_ui(language, '緊急停止', 'Emergency stop')}</button>
        </form>"""
    elif status["status"] == "paused":
        controls = f"""
        <form method="post" action="/autonomy/{escape(lease_id)}?lang={escape(language)}">
          <input type="hidden" name="csrf_token" value="{escape(token)}">
          <button name="action" value="resume" class="resume">{_ui(language, '繼續排程', 'Resume scheduling')}</button>
          <button name="action" value="stop" class="stop">{_ui(language, '撤銷授權', 'Revoke authorization')}</button>
          <button name="action" value="emergency_stop" class="emergency">{_ui(language, '緊急停止', 'Emergency stop')}</button>
        </form>"""
    payload = json.dumps(status, ensure_ascii=False, indent=2, default=str)
    event_revision = service.autonomy_ui_revision(lease_id)
    review_html = _autonomy_history_section(service, lease_id, language)
    events_url = (
        f"/autonomy/{lease_id}/events?after={event_revision}"
        if status["status"] in {"active", "paused"}
        else ""
    )
    event_script = (
        '<script defer src="/assets/autonomy-events.js"></script>'
        if events_url
        else ""
    )
    html = f"""<!doctype html>
<html lang="{escape(language)}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script defer src="/assets/language.js"></script>
{event_script}
<title>{_ui(language, 'JY 有限自動量測控制', 'JY bounded automatic measurement control')}</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
body {{ margin:0; background:#08101d; color:#edf4ff; line-height:1.5; }}
main {{ max-width:900px; margin:24px auto; padding:0 16px 48px; }}
.language-switch {{ position:fixed; top:10px; right:12px; z-index:10; padding:8px 10px; background:#121d31; border:1px solid #2b3b5d; border-radius:10px; }}
.language-switch label {{ display:block; color:#9fb2d7; font-size:.75rem; }}
.language-switch select {{ width:auto; padding:6px; color:white; background:#07101f; border:1px solid #6680b5; border-radius:7px; }}
.card {{ background:#121d31; border:1px solid #2b3b5d; border-radius:14px;
padding:18px; margin:16px 0; }}
dl {{ display:grid; grid-template-columns:160px 1fr; gap:8px 14px; }} dd {{ margin:0; }}
button {{ width:100%; margin-top:12px; padding:14px; border:0; border-radius:9px;
font:inherit; font-weight:750; color:white; cursor:pointer; }}
.pause {{ background:#9a6700; }} .resume {{ background:#16794a; }} .stop {{ background:#b4232d; }}
.emergency {{ background:#ff1738; border:2px solid white; }}
pre {{ overflow:auto; white-space:pre-wrap; background:#070d18; padding:14px;
border-radius:8px; }} .notice {{ padding:12px; border-radius:8px; }}
.result-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr)); gap:14px; }}
figure {{ margin:0; padding:10px; background:#070d18; border-radius:10px; }}
img {{ display:block; width:100%; height:auto; border-radius:7px; }}
.error {{ background:#5a1e2a; }} .success {{ background:#14532d; }}
@media(max-width:600px) {{ dl {{ grid-template-columns:1fr; }} }}
</style></head><body data-autonomy-events="{escape(events_url)}">{_language_switch(language)}<main>
<h1>{_ui(language, 'JY 有限自動量測控制', 'JY bounded automatic measurement control')}</h1>
<p>{_ui(language, '審閱者', 'Reviewer')}：{escape(_principal_label(principal, language))}</p>{notice}
<section class="card"><dl>
<dt>{_ui(language, '授權', 'Authorization')}</dt><dd>{escape(lease_id)}</dd>
<dt>{_ui(language, '狀態', 'Status')}</dt><dd>{escape(_localized_enum(language, status['status']))}</dd>
<dt>{_ui(language, '工作流程', 'Workflow')}</dt><dd>{escape(str(status['workflow_id']))}</dd>
<dt>{_ui(language, '目標', 'Targets')}</dt><dd>{escape(', '.join(status['targets']))}</dd>
<dt>{_ui(language, '節點', 'Nodes')}</dt><dd>{escape(', '.join(status['allowed_nodes']))}</dd>
<dt>{_ui(language, '到期時間', 'Expires')}</dt><dd>{escape(_format_taipei_time(status.get('expires_at')))}</dd>
<dt>{_ui(language, '每個節點／qubit 上限', 'Per node/qubit limit')}</dt><dd>{escape(str(status['max_attempts_per_node_qubit']))}</dd>
<dt>{_ui(language, '整體執行次數上限', 'Total run limit')}</dt><dd>{escape(_limit_label(language, status.get('max_total_runs')))}</dd>
<dt>{_ui(language, '已完成目標', 'Resolved targets')}</dt><dd>{escape(json.dumps(status.get('resolved_targets_by_node', {}), ensure_ascii=False))}</dd>
<dt>{_ui(language, '未完成目標', 'Incomplete targets')}</dt><dd>{escape(json.dumps(status.get('incomplete_targets_by_node', {}), ensure_ascii=False))}</dd>
</dl></section>
{review_html}
<section class="card"><h2>{_ui(language, '手動控制', 'Manual controls')}</h2>{controls or f'<p>{_ui(language, "目前沒有可用的控制。", "No active controls.")}</p>'}</section>
<details class="card"><summary>{_ui(language, '完整授權與計數資料', 'Complete authorization and counters')}</summary>
<pre>{escape(payload)}</pre></details>
</main></body></html>"""
    return HTMLResponse(html, status_code=status_code, headers=SECURITY_HEADERS)


def _autonomy_history_section(
    service: AgentService, lease_id: str, language: str = "zh-Hant"
) -> str:
    """Render the append-only run history for one bounded authorization."""
    try:
        review = service.autonomy_review(lease_id)
    except ServiceError as exc:
        return (
            f'<section class="card"><h2>{_ui(language, "量測結果", "Measurement results")}</h2><p>'
            f"{escape(str(exc))}</p></section>"
        )
    history = review.get("history", [])
    if not history:
        return f"""
        <section class="card"><h2>{_ui(language, '本次授權的量測結果', 'Results for this authorization')}</h2>
        <p>{_ui(language, '目前尚無結果。頁面只會在新結果完成分析或手動控制狀態改變時更新。', 'No result is available yet. The page updates only when a new analysis or manual control state is available.')}</p>
        </section>"""

    rendered: list[str] = [
        f'<section class="card"><h2>{_ui(language, "本次授權的完整量測歷史", "Complete measurement history for this authorization")}</h2>'
        f'<p>{_ui(language, "新結果會依時間附加；既有結果不會被下一次量測覆蓋。", "New results are appended over time; later runs do not replace earlier results.")}</p></section>'
    ]
    total = len(history)
    for ordinal, item in enumerate(history, start=1):
        run = item.get("run") or {}
        analysis = item.get("analysis") or {}
        decision = item.get("decision") or {}
        figures = "".join(
            f'<figure><img src="/autonomy/{escape(lease_id)}/assets/{asset_index}" '
            f'alt="{escape(_ui(language, "實驗結果圖", "Experiment result plot"))} {ordinal}-{plot_number}" loading="lazy">'
            f'<figcaption>{_ui(language, "實驗", "Experiment")} {ordinal} &middot; {_ui(language, "結果圖", "Plot")} {plot_number}</figcaption></figure>'
            for plot_number, asset_index in enumerate(
                item.get("asset_indices", []), start=1
            )
        )
        if not figures:
            figures = f"<p>{_ui(language, '此筆量測尚無可用的 snapshot 結果圖。', 'No snapshot result plot is available for this run.')}</p>"
        next_action = {
            "next_node": decision.get("next_node"),
            "new_parameters": decision.get("next_parameters", {}),
        }
        evidence = {
            "analysis_status": analysis.get("analysis_status"),
            "failure_reasons": analysis.get("failure_reasons", []),
            "warnings": analysis.get("warnings", []),
            "fit_quality": analysis.get("fit_quality"),
            "dataset_metrics": analysis.get("dataset_metrics"),
        }
        rendered.append(
            f"""
        <section class="card" id="run-{escape(str(run.get('id')))}">
          <h2>{_ui(language, '實驗', 'Experiment')} {ordinal} / {total} &middot; {escape(str(run.get('node_id')))}</h2>
          <dl>
            <dt>{_ui(language, '執行編號', 'Run')}</dt><dd>{escape(str(run.get('id')))}</dd>
            <dt>{_ui(language, '狀態', 'Status')}</dt><dd>{escape(_localized_enum(language, run.get('status')))}</dd>
            <dt>{_ui(language, '快照', 'Snapshot')}</dt><dd>{escape(str(run.get('snapshot_id')))}</dd>
            <dt>{_ui(language, '耗時', 'Elapsed')}</dt><dd>{escape(_format_seconds(run.get('elapsed_seconds')))}</dd>
          </dl>
          {_json_details(_ui(language, '時間預測', 'Prediction'), item.get('duration_prediction'))}
          {_json_details(_ui(language, '參數', 'Parameters'), run.get('parameters', {}))}
          <div class="result-grid">{figures}</div>
          <h3>{_ui(language, '判斷與理由', 'Decision and reason')}</h3>
          <dl>
            <dt>{_ui(language, '判斷', 'Decision')}</dt><dd>{escape(_localized_enum(language, decision.get('decision') or _ui(language, '待判斷', 'Pending')))}</dd>
            <dt>{_ui(language, '理由', 'Reason')}</dt><dd>{escape(str(decision.get('reason') or _ui(language, '尚未記錄判斷理由。', 'No decision reason has been recorded.')))}</dd>
          </dl>
          {_json_details(_ui(language, '分析', 'Analysis'), evidence)}
          {_json_details(_ui(language, '下一步', 'Next action'), next_action)}
        </section>
        """
        )
    return "".join(rendered)


def _proposal_page(
    service: AgentService,
    proposal: dict[str, Any],
    *,
    principal: ApprovalPrincipal | None = None,
    error: str | None = None,
    success: str | None = None,
    status_code: int = 200,
    language: str = "zh-Hant",
) -> HTMLResponse:
    proposal_id = str(proposal["id"])
    status = str(proposal["status"])
    payload = json.dumps(
        proposal.get("payload", {}),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    confirmation = f"APPROVE {proposal_id}"
    form = ""
    if status == "pending" and _lease_option_offer(proposal) is not None:
        form = (
            f'<section class="card approval"><h2>{_ui(language, "核准", "Approval")}</h2>'
            f'<p>{_ui(language, "自動量測請在量測 Dashboard 的「核准」頁選擇起點與選項、預覽會執行的實驗後再核准。", "Approve automatic measurement on the Dashboard approval page: choose the start point and options, preview the experiments, then approve.")}</p></section>'
        )
    elif status == "pending":
        csrf_token = service.approval_csrf_token(proposal_id)
        form = f"""
        <section class="card approval">
          <h2>{_ui(language, '核准', 'Approval')}</h2>
          <p>{_ui(language, '確認結果、判斷與下一步後，輸入以下完整文字：', 'Review the result, decision, and next action, then enter the complete text below:')}</p>
          <code class="confirmation">{escape(confirmation)}</code>
          <form method="post" action="/approve/{escape(proposal_id)}?lang={escape(language)}">
            <input type="hidden" name="csrf_token" value="{escape(csrf_token)}">
            {_lease_options_fields(proposal, language)}
            <label for="confirmation">{_ui(language, '核准文字', 'Confirmation')}</label>
            <input id="confirmation" name="confirmation" type="text"
                   autocomplete="off" autocapitalize="off" spellcheck="false" required>
            <button type="submit">{_ui(language, '核准此提案', 'Approve this proposal')}</button>
          </form>
        </section>
        """
    notice = ""
    if error:
        notice = f'<p class="notice error">{escape(error)}</p>'
    elif success:
        notice = f'<p class="notice success">{escape(success)}</p>'

    reviewer = (
        _principal_label(principal, language)
        if principal
        else _ui(language, "本機審閱", "local review")
    )
    review_html = _review_section(service, proposal_id, language)
    proposed_action_html = _proposed_action_section(proposal, language)
    control_url = proposal.get("autonomy_status_url")
    autonomy_link = (
        f'<section class="card"><h2>{_ui(language, "自動量測控制", "Automatic measurement controls")}</h2>'
        f'<p><a href="{escape(str(control_url))}">{_ui(language, "開啟暫停、停止與緊急停止控制", "Open pause, stop, and emergency controls")}</a></p></section>'
        if control_url and status == "approved"
        else ""
    )
    html = f"""<!doctype html>
<html lang="{escape(language)}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script defer src="/assets/language.js"></script>
  <title>{_ui(language, 'JY 實驗審閱與核准', 'JY experiment review and approval')}</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, -apple-system, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #08101d; color: #edf4ff; line-height: 1.55; }}
    main {{ max-width: 1040px; margin: 28px auto; padding: 0 16px 52px; }}
    .language-switch {{ position:fixed; top:10px; right:12px; z-index:10; padding:8px 10px; background:#121d31; border:1px solid #2b3b5d; border-radius:10px; }}
    .language-switch label {{ display:block; color:#9fb2d7; font-size:.75rem; margin:0 0 3px; }}
    .language-switch select {{ width:auto; padding:6px 28px 6px 8px; color:white; background:#07101f; border:1px solid #6680b5; border-radius:7px; }}
    h1 {{ margin: 0 0 6px; font-size: clamp(1.55rem, 5vw, 2.25rem); }}
    h2 {{ margin-top: 0; }}
    .subtitle, dt {{ color: #9fb2d7; }}
    .card {{ background: #121d31; border: 1px solid #2b3b5d;
      border-radius: 14px; padding: clamp(15px, 4vw, 22px); margin: 16px 0; }}
    .result-grid {{ display: grid; grid-template-columns: repeat(auto-fit,
      minmax(min(100%, 300px), 1fr)); gap: 14px; }}
    figure {{ margin: 0; padding: 10px; background: #070d18; border-radius: 10px; }}
    img {{ display: block; width: 100%; height: auto; border-radius: 7px; }}
    figcaption {{ color: #9fb2d7; margin-top: 8px; overflow-wrap: anywhere; }}
    dl {{ display: grid; grid-template-columns: minmax(110px, 160px) 1fr;
      gap: 8px 16px; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    pre {{ overflow: auto; padding: 14px; background: #070d18; border-radius: 8px;
      white-space: pre-wrap; word-break: break-word; }}
    code {{ overflow-wrap: anywhere; }}
    .confirmation {{ display: block; padding: 12px; background: #070d18;
      border-radius: 8px; user-select: all; }}
    label {{ display: block; margin: 20px 0 6px; }}
    input {{ width: 100%; padding: 12px; font: inherit; border: 1px solid #6680b5;
      border-radius: 8px; background: #07101f; color: white; }}
    button {{ width: 100%; margin-top: 14px; padding: 13px 18px; font: inherit;
      font-weight: 750; border: 0; border-radius: 9px; color: white;
      background: #3568e8; cursor: pointer; }}
    .notice {{ padding: 14px; border-radius: 8px; }}
    a {{ color: #8ab4ff; }}
    .error {{ background: #5a1e2a; }} .success {{ background: #14532d; }}
    .lease-options {{ border:1px solid #2b3b5d; border-radius:10px; padding:10px 14px; margin:14px 0; }} .lease-options legend {{ padding:0 6px; font-weight:700; }} .lease-options .choice {{ display:flex; gap:10px; align-items:flex-start; margin:8px 0; }} .lease-options .choice input {{ width:auto; margin-top:5px; flex:none; }}
    .warning {{ color: #ffd479; }} .muted {{ color: #9fb2d7; }}
    details summary {{ cursor: pointer; font-weight: 700; }}
    @media (max-width: 600px) {{
      main {{ margin-top: 18px; }}
      dl {{ grid-template-columns: 1fr; gap: 2px; }}
      dd {{ margin-bottom: 8px; }}
    }}
  </style>
</head>
<body>{_language_switch(language)}
<main>
  <h1>{_ui(language, 'JY 實驗審閱與核准', 'JY experiment review and approval')}</h1>
  <p class="subtitle">{_ui(language, '結果圖、判斷、理由、下一步與核准', 'Result snapshots, decision, reason, next action, and approval')}</p>
  <p class="warning">{_ui(language, '請逐項確認。核准後，下一個動作可能啟動硬體量測或套用 state 變更。', 'Review each item. Approval may allow the next action to start a hardware run or apply a state change.')}</p>
  {notice}
  {review_html}
  {proposed_action_html}
  {autonomy_link}
  <section class="card">
    <h2>{_ui(language, '提案', 'Proposal')}</h2>
    <dl>
      <dt>{_ui(language, '提案編號', 'Proposal')}</dt><dd>{escape(proposal_id)}</dd>
      <dt>{_ui(language, '類型', 'Kind')}</dt><dd>{escape(_localized_enum(language, proposal['kind']))}</dd>
      <dt>{_ui(language, '狀態', 'Status')}</dt><dd>{escape(_localized_enum(language, status))}</dd>
      <dt>{_ui(language, '工作流程', 'Workflow')}</dt><dd>{escape(str(proposal['workflow_id']))}</dd>
      <dt>{_ui(language, '來源用戶端', 'Source client')}</dt><dd>{escape(str(proposal['source_client']))}</dd>
      <dt>{_ui(language, '審閱者', 'Reviewer')}</dt><dd>{escape(reviewer)}</dd>
      <dt>{_ui(language, '建立時間', 'Created')}</dt><dd>{escape(_format_taipei_time(proposal['created_at']))}</dd>
      <dt>{_ui(language, '到期時間', 'Expires')}</dt><dd>{escape(_format_taipei_time(proposal['expires_at']))}</dd>
      <dt>{_ui(language, '使用次數', 'Uses')}</dt><dd>{escape(str(proposal['uses']))} / {escape(str(proposal['max_uses']))}</dd>
    </dl>
  </section>
  <details class="card">
    <summary>{_ui(language, '完整提案資料', 'Complete proposal payload')}</summary>
    <pre>{escape(payload)}</pre>
  </details>
  {form}
</main>
</body>
</html>"""
    return HTMLResponse(html, status_code=status_code, headers=SECURITY_HEADERS)


def _review_section(
    service: AgentService, proposal_id: str, language: str = "zh-Hant"
) -> str:
    try:
        review = service.proposal_review(proposal_id)
    except (AttributeError, ServiceError):
        review = {"available": False, "plots": []}
    if not review.get("available"):
        return f"""
        <section class="card">
          <h2>{_ui(language, '結果', 'Result')}</h2>
          <p class="muted">{_ui(language, '這是第一個提案，或尚無可綁定的前次實驗結果。', 'This is the first proposal, or no previous experiment result can be attached yet.')}</p>
        </section>
        """

    plots = review.get("plots", [])
    figures = "".join(
        f"""
        <figure>
          <img src="/approve/{escape(proposal_id)}/assets/{index}"
               alt="{escape(_ui(language, '結果圖', 'Result plot'))} {index + 1}" loading="lazy">
          <figcaption>{_ui(language, '結果圖', 'Result plot')} {index + 1}</figcaption>
        </figure>
        """
        for index, _ in enumerate(plots)
    )
    if not figures:
        figures = f'<p class="muted">{_ui(language, "此筆量測沒有可用的結果圖。", "No result plot was available for this run.")}</p>'

    run = review.get("run") or {}
    analysis = review.get("analysis") or {}
    decision = review.get("decision") or {}
    decision_value = decision.get("decision") or _ui(
        language, "尚未記錄", "Not recorded"
    )
    reason = decision.get("reason") or _ui(
        language, "尚未記錄判斷理由。", "No decision reason has been recorded."
    )
    next_action = {
        "next_node": decision.get("next_node"),
        "new_parameters": decision.get("next_parameters", {}),
    }
    evidence = {
        "analysis_status": analysis.get("analysis_status"),
        "failure_reasons": analysis.get("failure_reasons", []),
        "warnings": analysis.get("warnings", []),
        "fit_quality": analysis.get("fit_quality"),
        "dataset_metrics": analysis.get("dataset_metrics"),
    }
    return f"""
    <section class="card">
      <h2>{_ui(language, '結果', 'Result')}</h2>
      <dl>
        <dt>{_ui(language, '執行編號', 'Run')}</dt><dd>{escape(str(run.get('id', _ui(language, '未知', 'unknown'))))}</dd>
        <dt>{_ui(language, '節點', 'Node')}</dt><dd>{escape(str(run.get('node_id', _ui(language, '未知', 'unknown'))))}</dd>
        <dt>{_ui(language, '快照', 'Snapshot')}</dt><dd>{escape(str(run.get('snapshot_id', _ui(language, '未知', 'unknown'))))}</dd>
        <dt>{_ui(language, '分析狀態', 'Analysis status')}</dt><dd>{escape(_localized_enum(language, run.get('analysis_status', _ui(language, '未知', 'unknown'))))}</dd>
        <dt>{_ui(language, '耗時', 'Elapsed')}</dt><dd>{escape(_format_seconds(run.get('elapsed_seconds')))}</dd>
      </dl>
      {_json_details(_ui(language, '參數', 'Parameters'), run.get('parameters', {}))}
      <div class="result-grid">{figures}</div>
    </section>
    <section class="card">
      <h2>{_ui(language, '判斷與理由', 'Decision and reason')}</h2>
      <dl>
        <dt>{_ui(language, '判斷', 'Decision')}</dt><dd>{escape(_localized_enum(language, decision_value))}</dd>
        <dt>{_ui(language, '理由', 'Reason')}</dt><dd>{escape(str(reason))}</dd>
      </dl>
      {_json_details(_ui(language, '分析', 'Analysis'), evidence)}
      {_json_details(_ui(language, '下一步', 'Next action'), next_action)}
    </section>
    """


def _proposed_action_section(
    proposal: dict[str, Any], language: str = "zh-Hant"
) -> str:
    payload = proposal.get("payload", {})
    kind = str(proposal.get("kind", "unknown"))
    prediction: Any = None
    parameters: Any = None
    if kind == "run":
        prediction = payload.get("duration_prediction")
        parameters = payload.get("parameters", {})
        action = {
            "kind": "run",
            "node": payload.get("node_id"),
            "reason": payload.get("reason"),
            "warnings": payload.get("warnings", []),
            "prior_decision_experience": payload.get(
                "prior_decision_experience", []
            ),
        }
    elif kind == "autonomy_lease":
        action = {
            "kind": kind,
            "lease_id": payload.get("lease_id"),
            "reason": payload.get("reason"),
            "targets": payload.get("targets", []),
            "allowed_nodes": payload.get("allowed_nodes", []),
            "duration_hours": payload.get("duration_hours"),
            "max_total_runs": payload.get("max_total_runs"),
            "max_attempts_per_node_qubit": payload.get(
                "max_attempts_per_node_qubit"
            ),
            "auto_state_commit_analysis_statuses": payload.get(
                "auto_state_commit_analysis_statuses", []
            ),
            "halt_conditions": payload.get("halt_conditions", []),
            "manual_controls": payload.get("manual_controls", []),
        }
    else:
        action = {
            "kind": kind,
            "reason": payload.get("reason"),
            "run_id": payload.get("run_id"),
            "patch": payload.get("patch", []),
        }
    extra_details = ""
    if kind == "run":
        extra_details = (
            _json_details(_ui(language, "時間預測", "Prediction"), prediction)
            + _json_details(_ui(language, "參數", "Parameters"), parameters)
        )
    return f"""
    <section class="card">
      <h2>{_ui(language, '建議的下一步', 'Proposed next action')}</h2>
      {extra_details}
      {_json_details(_ui(language, '下一步', 'Next action'), action)}
    </section>
    """


def _message_page(
    title: str,
    message: str,
    *,
    status_code: int,
    language: str = "zh-Hant",
) -> HTMLResponse:
    html = (
        f'<!doctype html><html lang="{escape(language)}"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<script defer src="/assets/language.js"></script>'
        "<style>body{font-family:system-ui,sans-serif;padding:2rem}.language-switch{position:fixed;top:10px;right:12px}select{padding:.4rem}</style>"
        f"<title>{escape(title)}</title></head><body>{_language_switch(language)}"
        f"<h1>{escape(title)}</h1><p>{escape(message)}</p></body></html>"
    )
    return HTMLResponse(html, status_code=status_code, headers=SECURITY_HEADERS)


def _approval_principal(
    request: Request, service: AgentService
) -> ApprovalPrincipal:
    remote_host = request.client.host if request.client is not None else ""
    settings = service.settings
    if settings.approval_transport == "public":
        if not bool(getattr(request.state, "jy_public_authenticated", False)):
            raise ServiceError("This public dashboard requires its private access link.")
        return ApprovalPrincipal(
            actor="authorized operator via public HTTPS dashboard",
            method="public_token_browser",
            display_name="authorized operator",
        )
    if not _host_in_cidrs(remote_host, settings.approval_trusted_proxy_cidrs):
        raise ServiceError(
            "Approval requests must arrive from the local trusted HTTPS proxy."
        )

    if not _is_loopback(remote_host):
        raise ServiceError("Browser approval is available only through the JY host.")
    return ApprovalPrincipal(
        actor=f"{getpass.getuser()} via local browser ({remote_host})",
        method="local_browser",
        display_name=f"{getpass.getuser()} (local browser)",
    )


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def _host_in_cidrs(host: str, cidrs: Collection[str]) -> bool:
    try:
        address = ipaddress.ip_address(host)
        return any(address in ipaddress.ip_network(value) for value in cidrs)
    except ValueError:
        return host.lower() == "localhost" and any(
            ipaddress.ip_address("127.0.0.1") in ipaddress.ip_network(value)
            for value in cidrs
        )


def _request_origin(request: Request) -> str:
    host = request.headers.get("host", request.url.netloc)
    return f"{request.url.scheme}://{host}".rstrip("/")


def _has_trusted_submission_source(
    request: Request,
    expected_origins: str | Collection[str],
) -> bool:
    """Accept configured same-origin forms plus opaque WebView forms with CSRF."""
    origins = (
        {expected_origins.rstrip("/")}
        if isinstance(expected_origins, str)
        else {value.rstrip("/") for value in expected_origins}
    )
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") in origins:
        return True
    if origin not in {None, "null"}:
        return False

    referer = request.headers.get("referer", "")
    if any(referer.startswith(f"{value}/approve/") for value in origins):
        return True
    fetch_site = request.headers.get("sec-fetch-site", "").lower()
    if fetch_site in {"same-origin", "none"}:
        return True

    return False


# Operator request 2026-10-04: labels for the optional nodes on the lease page.
OPTIONAL_NODE_LABELS = {
    "02": ("02 寬頻共振腔掃描（Broadband spectroscopy，02x 之前）", "02 Broadband spectroscopy (before 02x)"),
    "06a": ("06a Ramsey vs flux（06 之後，微調 flux offset；需選擇進行 flux 校正）", "06a Ramsey vs flux (after 06, fine-tunes the flux offset; needs flux calibration)"),
    "05st": ("05st T1 統計", "05st T1 statistics"),
    "06st_t2star": ("06st T2* 統計", "06st T2* statistics"),
    "06st_t2e": ("06st T2 echo 統計", "06st T2 echo statistics"),
}


def _lease_option_offer(proposal: dict[str, Any]) -> dict[str, Any] | None:
    if str(proposal.get("kind")) != "autonomy_lease":
        return None
    offer = (proposal.get("payload") or {}).get("workflow_options_offer")
    return offer if isinstance(offer, dict) else None


def _lease_options_fields(
    proposal: dict[str, Any],
    language: str,
    selected: dict[str, Any] | None = None,
) -> str:
    """Checkboxes and the measurement-mode choice for an autonomy lease.

    Operator request 2026-10-04: automatic measurement starts only after the
    operator has chosen which optional experiments run and whether the qubits
    are measured one at a time or multiplexed. `selected` re-fills the form
    with the choices a preview was made from (operator request 2026-10-10).
    """

    offer = _lease_option_offer(proposal)
    if offer is None:
        return ""
    parts: list[str] = []
    history = offer.get("start_node")
    if isinstance(history, dict):
        parts.append(_start_node_fieldset(history, language, selected))
    flux = offer.get("flux_calibration")
    flux_optional = (
        set(flux.get("optional") or []) if isinstance(flux, dict) else set()
    )
    flux_boxes: list[str] = []
    optional = [
        item
        for item in offer.get("optional_nodes", [])
        if isinstance(item, dict)
    ]
    for item in [item for item in optional if str(item.get("node")) in flux_optional]:
        node = str(item.get("node"))
        zh, en = OPTIONAL_NODE_LABELS.get(node, (node, node))
        blocked = bool(flux.get("blocked")) or flux.get("locked") is False
        ticked = (
            node in (selected.get("optional_nodes") or [])
            if selected is not None
            else item.get("default", True)
        )
        checked = " checked" if ticked and not blocked else ""
        disabled = " disabled" if blocked else ""
        flux_boxes.append(
            f'<label class="choice flux-dependent"><input type="checkbox" name="optional_node" '
            f'value="{escape(node)}"{checked}{disabled}> {escape(_ui(language, zh, en))}</label>'
        )
    optional = [item for item in optional if str(item.get("node")) not in flux_optional]
    if optional:
        boxes = []
        for item in optional:
            node = str(item.get("node"))
            zh, en = OPTIONAL_NODE_LABELS.get(node, (node, node))
            ticked = (
                node in (selected.get("optional_nodes") or [])
                if selected is not None
                else item.get("default", True)
            )
            checked = " checked" if ticked else ""
            box = (
                f'<label class="choice"><input type="checkbox" name="optional_node" '
                f'value="{escape(node)}"{checked}> {escape(_ui(language, zh, en))}</label>'
            )
            if item.get("required"):
                # A disabled box is not submitted, so a hidden field carries it.
                names = ", ".join(str(name) for name in item.get("required_because") or [])
                note = _ui(
                    language,
                    f"必須執行：{names} 的讀取 IF 為 0 或重複，請填共振腔數量與設計頻率。",
                    f"Required: readout IFs of {names} are zero or duplicated; "
                    "enter the resonator count and design frequencies.",
                )
                box = (
                    f'<input type="hidden" name="optional_node" value="{escape(node)}">'
                    f'<label class="choice"><input type="checkbox" checked disabled> '
                    f"{escape(_ui(language, zh, en))}</label>"
                    f'<p class="warning">{escape(note)}</p>'
                )
            if node == "02":
                # Operator choice 2026-10-11: a later start point skips 02, so
                # the 02 box and its fields disappear (the server ignores them).
                skipped_note = _ui(
                    language,
                    "已選擇從後面的實驗開始：02 不會執行，不需勾選或填寫。",
                    "A later start point is selected: 02 does not run and needs no tick or parameters.",
                )
                box = (
                    "<style>"
                    "form:has(input[name=start_node]:checked:not([value=beginning])) .node-02-block { display:none; }"
                    ".node-02-skipped { display:none; }"
                    "form:has(input[name=start_node]:checked:not([value=beginning])) .node-02-skipped { display:block; }"
                    "</style>"
                    f'<div class="optional-node node-02-block">{box}'
                    f"{_broadband_parameter_fields(((selected or {}).get('node_parameters') or {}).get('02') or item.get('parameters'), language)}</div>"
                    f'<p class="muted node-02-skipped">{escape(skipped_note)}</p>'
                )
            boxes.append(box)
        parts.append(
            f"""<fieldset class="lease-options">
              <legend>{_ui(language, '額外實驗（勾選後才會執行）', 'Optional experiments (run only when ticked)')}</legend>
              {''.join(boxes)}
            </fieldset>"""
        )
    drive = offer.get("drive_mode")
    if isinstance(drive, dict):
        first = escape(str(drive.get("serialize_from_node") or "03a"))
        groups = drive.get("shared_xy_drive_groups") or {}
        recommended = drive.get("recommended")
        locked = drive.get("locked")
        if groups:
            shared = "; ".join(
                f"{escape(str(output).split('/ports/')[-1])} → {escape(', '.join(names))}"
                for output, names in sorted(groups.items())
            )
            detected = _ui(
                language,
                f"偵測到 wiring 中多個 qubit 共用同一個 XY 輸出（共用 LO）：{shared}。{first} 之後建議一顆一顆量。",
                f"The wiring routes several qubits to one XY output (shared LO): {shared}. One qubit at a time is recommended from {first} on.",
            )
        else:
            detected = _ui(
                language,
                f"每個 qubit 都有自己的 XY 輸出；{first} 之後建議 multiplex 量測。",
                f"Every qubit has its own XY output; multiplexed measurement is recommended from {first} on.",
            )
        choices = (
            (
                "per_qubit",
                _ui(language, f"一顆一顆量：每顆 qubit 單獨走完 {first} 之後的全部實驗，再換下一顆", f"One qubit at a time: each qubit goes through every node from {first} on before the next one starts"),
            ),
            (
                "multiplex",
                _ui(language, f"Multiplex：{first} 之後每個實驗同時量所有 qubit", f"Multiplexed: every node from {first} on measures all qubits together"),
            ),
        )
        blocked = set(drive.get("blocked") or [])
        radios = []
        for value, label in choices:
            mark = (
                _ui(language, "（建議）", " (recommended)") if value == recommended else ""
            )
            if value in blocked:
                mark = _ui(language, "（共用 XY 輸出，已停用）", " (disabled: shared XY output)")
            forced = blocked and value not in blocked
            disabled = (
                " disabled"
                if value in blocked or (locked is not None and locked != value)
                else ""
            )
            chosen = (
                selected is not None
                and selected.get("drive_mode") == value
                and not disabled
            )
            checked = (
                " checked"
                if locked == value or (forced and locked is None) or chosen
                else ""
            )
            radios.append(
                f'<label class="choice"><input type="radio" name="drive_mode" '
                f'value="{value}" required{checked}{disabled}> {escape(label)}{mark}</label>'
            )
        locked_note = (
            f'<p class="muted">{_ui(language, "此工作流程已選定量測方式，不能在中途更改。", "This workflow already chose its measurement mode; it cannot change mid-workflow.")}</p>'
            if locked is not None
            else ""
        )
        warning = (
            f'<p class="warning">{_ui(language, "共用 XY 輸出無法 multiplex：同一條線上的 qubit 共用 LO 與驅動，因此只能一次量一顆 qubit。", "A shared XY output cannot be multiplexed: the qubits on that line share one LO and drive, so they are measured one at a time.")}</p>'
            if groups
            else ""
        )
        parts.append(
            f"""<fieldset class="lease-options">
              <legend>{_ui(language, '量測方式（必選）', 'Measurement mode (required)')}</legend>
              <p>{detected}</p>
              {''.join(radios)}
              {locked_note}
              {warning}
            </fieldset>"""
        )
    if isinstance(flux, dict):
        parts.append(
            _flux_calibration_fieldset(
                flux,
                flux_boxes,
                language,
                (selected or {}).get("flux_calibration"),
            )
        )
    return "\n".join(parts)


def _lease_preview(
    service: AgentService, proposal_id: str, lease_options: dict[str, Any]
) -> dict[str, Any]:
    return {
        "options": lease_options,
        "preview": service.preview_lease_options(proposal_id, lease_options),
        "digest": service.lease_preview_digest(proposal_id, lease_options),
    }


def _lease_approval_card(
    service: AgentService,
    session_id: str,
    pending: dict[str, Any],
    language: str,
    lease_preview: dict[str, Any] | None,
) -> str:
    """Two-step automatic-measurement approval.

    Operator request 2026-10-10: the operator makes the choices and previews
    which experiments will run; only then do the agree box and the approval
    code appear, and Approve needs both. Changing a choice after the preview
    needs a new preview.
    """

    proposal_id = str(pending["id"])
    csrf = service.approval_csrf_token(proposal_id)
    selected = (lease_preview or {}).get("options")
    preview = (lease_preview or {}).get("preview")
    digest = (lease_preview or {}).get("digest")
    confirm_html = ""
    if isinstance(preview, dict) and digest:
        confirmation = f"APPROVE {proposal_id}"
        confirm_html = f"""
            {_lease_preview_section(preview, language)}
            <input type="hidden" name="preview_digest" value="{escape(str(digest))}">
            <label class="choice agree"><input type="checkbox" name="agree" value="yes" required>
              {_ui(language, '我已確認以上會執行的實驗、起點與選項。', 'I have checked the experiments, start point and choices above.')}</label>
            <p>{_ui(language, '勾選後輸入以下完整文字：', 'Then enter the complete text below:')}</p>
            <code class="confirmation">{escape(confirmation)}</code>
            <label for="confirmation">{_ui(language, '核准文字', 'Confirmation')}</label>
            <input id="confirmation" name="confirmation" type="text"
                   autocomplete="off" autocapitalize="off" spellcheck="false" required>
            <button type="submit" name="operation" value="approve">{_ui(language, '核准', 'Approve')}</button>"""
    step = (
        _ui(language, "若要更改選項，改完後請重新預覽。", "To change a choice, change it and preview again.")
        if confirm_html
        else _ui(language, "選完後按「預覽」，確認此次會量哪些實驗後才會出現核准碼。", "Make the choices, then press Preview; the approval code appears after you have seen which experiments will run.")
    )
    return f"""
        <section class="card approval">
          <h2>{_ui(language, '待核准動作', 'Action awaiting approval')}</h2>
          {_proposed_action_section(pending, language)}
          <form method="post" action="/session/{escape(session_id)}/approval?lang={escape(language)}">
            <input type="hidden" name="proposal_id" value="{escape(proposal_id)}">
            <input type="hidden" name="csrf_token" value="{escape(csrf)}">
            {_lease_options_fields(pending, language, selected)}
            <p class="muted">{step}</p>
            <button type="submit" name="operation" value="preview" formnovalidate>{_ui(language, '預覽此次量測的實驗', 'Preview the experiments')}</button>
            {confirm_html}
          </form>
        </section>"""


def _start_node_fieldset(
    history: dict[str, Any], language: str, selected: dict[str, Any] | None
) -> str:
    """Where this workflow starts, with the chip's latest workflow beside it.

    Operator request 2026-10-10: start from the beginning or at any node the
    chip's latest workflow reached; the evidence before it is kept. Starting
    after 02b keeps that workflow's flux choice.
    """

    chosen = str((selected or {}).get("start_node") or "beginning")
    flux = bool(history.get("flux_calibration"))
    rows = []
    for item in history.get("nodes") or []:
        if not isinstance(item, dict):
            continue
        status = (
            _ui(language, "完成", "Finished")
            if item.get("finished")
            else _ui(language, "已量但未完成", "Measured, not finished")
            if item.get("ran")
            else _ui(language, "未量", "Not measured")
        )
        passed = ", ".join(str(name) for name in item.get("passed") or []) or "—"
        rows.append(
            f"<tr><td>{escape(str(item.get('node')))}</td><td>{escape(status)}</td>"
            f"<td>{escape(passed)}</td></tr>"
        )
    radios = [
        f'<label class="choice"><input type="radio" name="start_node" value="beginning"'
        f'{" checked" if chosen == "beginning" else ""}> '
        f'{_ui(language, "從頭開始（不沿用上次結果）", "From the beginning (no earlier results kept)")}</label>'
    ]
    for node in history.get("start_nodes") or []:
        node = str(node)
        radios.append(
            f'<label class="choice"><input type="radio" name="start_node" value="{escape(node)}"'
            f'{" checked" if chosen == node else ""}> '
            f'{escape(_ui(language, f"從 {node} 開始（沿用 {node} 之前的結果）", f"Start at {node} (keep the results before {node})"))}</label>'
        )
    warning = (
        ""
        if history.get("state_matches", True)
        else f'<p class="warning">{_ui(language, "注意：目前的 state.json 與上次 workflow 結束時不同（之後有其他量測或修改）。沿用的結果可能已不符合目前狀態。", "Note: the current state.json differs from the one the last workflow left. The kept results may no longer match the current state.")}</p>'
    )
    blocked = [str(name) for name in history.get("start_blocked_by") or []]
    absent = [str(name) for name in history.get("absent_at_02") or []]
    if blocked:
        names = ", ".join(blocked)
        warning += (
            f'<p class="warning">{escape(_ui(language, f"{names} 的讀取 IF 尚未設定，且上次 02 沒有判定為找不到，所以這次只能從頭開始（02 必須執行並填寫）。", f"{names} have no readout IF and were not found absent at 02 last time, so this measurement must start from the beginning (02 runs and needs its parameters)."))}</p>'
        )
    elif absent:
        names = ", ".join(absent)
        warning += (
            f'<p class="muted">{escape(_ui(language, f"上次 02 找不到 {names} 的共振腔；從後面的實驗開始時會沿用這個判定，不需要再做 02。", f"Last time 02 found no resonator for {names}; a later start keeps that and skips 02."))}</p>'
        )
    created = _format_taipei_time(history.get("created_at"))
    return f"""<fieldset class="lease-options">
      <legend>{_ui(language, '起點（必選）', 'Start point (required)')}</legend>
      <p>{escape(_ui(language, f"這顆 chip（{history.get('device_key')}）最近一次的 workflow：{history.get('source_workflow_id')}（{created} 開始），量到 {history.get('reached_node')}；當時 flux 校正：{'是' if flux else '否'}。", f"This chip's ({history.get('device_key')}) latest workflow: {history.get('source_workflow_id')} (started {created}), reached {history.get('reached_node')}; flux calibration then: {'yes' if flux else 'no'}."))}</p>
      <table class="history"><thead><tr><th>Node</th><th>{_ui(language, '狀態', 'Status')}</th><th>{_ui(language, '通過的 qubit', 'Passed qubits')}</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table>
      {''.join(radios)}
      {_qubit_start_fields(history, language, selected)}
      <p class="muted">{_ui(language, '從 02b 之後的 node 開始時，flux 校正固定沿用上次的選擇；上次未選 flux 時不會出現 03b。', 'Starting after 02b keeps the last flux choice; without flux last time, 03b does not appear.')}</p>
      {warning}
    </fieldset>"""


def _qubit_start_fields(
    history: dict[str, Any], language: str, selected: dict[str, Any] | None
) -> str:
    """One start node per qubit, shown when the start is the per-qubit pass.

    Operator request 2026-10-12: on a chip measured one qubit at a time each
    qubit may restart its pass at any node it reached last time.
    """

    offer = history.get("qubit_start_nodes") or {}
    serial = history.get("serialize_from_node")
    if not offer or not serial:
        return ""
    chosen = (selected or {}).get("qubit_start_nodes") or {}
    rows = []
    for qubit, nodes in offer.items():
        options = "".join(
            f'<option value="{escape(str(node))}"'
            f'{" selected" if str(chosen.get(qubit, serial)) == str(node) else ""}>'
            f"{escape(str(node))}</option>"
            for node in nodes
        )
        rows.append(
            f'<label class="qubit-start">{escape(str(qubit))} '
            f'<select name="qubit_start__{escape(str(qubit))}">{options}</select></label>'
        )
    note = _ui(
        language,
        f"從 {serial} 開始時，每顆 qubit 可各自選擇從上次量到的哪個實驗繼續（之前的結果沿用）。",
        f"Starting at {serial}, each qubit may continue from any experiment it reached last time (earlier results are kept).",
    )
    return (
        "<style>"
        ".qubit-starts { display:none; }"
        f"form:has(input[name=start_node][value=\"{escape(str(serial))}\"]:checked) .qubit-starts {{ display:block; }}"
        ".qubit-start { display:inline-block; margin-right:1em; }"
        "</style>"
        f'<div class="qubit-starts"><p class="muted">{escape(note)}</p>{"".join(rows)}</div>'
    )


def _lease_preview_section(preview: dict[str, Any], language: str) -> str:
    """The nodes this approval would run, shown before the agree box."""

    labels = {
        "run": _ui(language, "會量測", "Runs"),
        "inherited": _ui(language, "沿用上次結果", "Kept from last time"),
        "not_run": _ui(language, "不量測", "Not run"),
        "skipped": _ui(language, "略過（未勾選）", "Skipped (unticked)"),
        "skipped_flux": _ui(language, "略過（未選 flux 校正）", "Skipped (no flux calibration)"),
    }
    items = []
    for item in preview.get("nodes") or []:
        status = str(item.get("status"))
        css = "run" if status == "run" else "muted"
        items.append(
            f'<li class="{css}"><strong>{escape(str(item.get("node")))}</strong> — '
            f"{escape(labels.get(status, status))}</li>"
        )
    count = sum(1 for item in preview.get("nodes") or [] if item.get("status") == "run")
    starts = []
    for item in preview.get("qubit_starts") or []:
        if not isinstance(item, dict):
            continue
        node = str(item.get("start_node"))
        label = _ui(language, f"從 {node} 開始", f"starts at {node}")
        starts.append(
            f"<li><strong>{escape(str(item.get('qubit')))}</strong> — {escape(label)}</li>"
        )
    starts_html = (
        f"<h3>{_ui(language, '每顆 qubit 的起點（一次量一顆）', 'Where each qubit starts (one qubit at a time)')}</h3>"
        f'<ul class="preview-qubits">{"".join(starts)}</ul>'
        if starts
        else ""
    )
    return f"""<section class="lease-preview">
      <h3>{_ui(language, f'此次量測會執行的實驗（{count} 個）', f'Experiments this measurement runs ({count})')}</h3>
      <ol class="preview-nodes">{''.join(items)}</ol>
      {starts_html}
    </section>"""


def _flux_calibration_fieldset(
    flux: dict[str, Any],
    extra_boxes: list[str],
    language: str,
    chosen: bool | None = None,
) -> str:
    """Yes/no flux calibration, blocked when the wiring shares a Z output.

    Operator request 2026-10-08.
    """

    nodes = ", ".join(str(node) for node in flux.get("nodes") or [])
    extras = ", ".join(str(node) for node in flux.get("optional") or [])
    groups = flux.get("shared_z_groups") or {}
    blocked = bool(flux.get("blocked"))
    locked = flux.get("locked")
    explanation = _ui(
        language,
        f"選「是」會執行 {nodes}（02b 在 02a 之後：共振腔 vs flux；03b 在 03a 之後："
        "qubit spectroscopy vs flux），把 qubit 移到 sweet spot。"
        + (f"可再勾選 {extras}（06 之後）。" if extras else ""),
        f"Yes runs {nodes} (02b after 02a: resonator vs flux; 03b after 03a: "
        "qubit spectroscopy vs flux) and moves each qubit to its sweet spot."
        + (f" {extras} (after 06) can be ticked as well." if extras else ""),
    )
    if groups:
        shared = "; ".join(
            f"{escape(str(output).split('/ports/')[-1])} → {escape(', '.join(names))}"
            for output, names in sorted(groups.items())
        )
        detected = _ui(
            language,
            f"偵測到 wiring 中多個 qubit 共用同一個 Z 輸出：{shared}。共用 flux 線無法單獨調整一顆 qubit，因此不能進行 flux 校正。",
            f"The wiring routes several qubits to one Z output: {shared}. A shared flux line cannot tune one qubit alone, so flux calibration is disabled.",
        )
    else:
        detected = _ui(
            language,
            "每個 qubit 都有自己的 Z 輸出，可以進行 flux 校正。",
            "Every qubit has its own Z output, so flux calibration is available.",
        )
    choices = (
        ("yes", _ui(language, "是，進行 flux 校正", "Yes, calibrate flux"), True),
        ("no", _ui(language, "否，維持目前的 flux 設定", "No, keep the current flux settings"), False),
    )
    radios = []
    for value, label, meaning in choices:
        disabled = (
            " disabled"
            if (blocked and meaning) or (locked is not None and locked is not meaning)
            else ""
        )
        checked = (
            " checked"
            if locked is meaning
            or (blocked and meaning is False and locked is None)
            or (chosen is meaning and not disabled)
            else ""
        )
        mark = (
            _ui(language, "（共用 Z 輸出，已停用）", " (disabled: shared Z output)")
            if blocked and meaning
            else ""
        )
        radios.append(
            f'<label class="choice"><input type="radio" name="flux_calibration" '
            f'value="{value}" required{checked}{disabled}> {escape(label)}{mark}</label>'
        )
    extras_html = ""
    if extra_boxes:
        extras_html = (
            f'<div class="flux-extras"><p class="muted">'
            f'{_ui(language, "額外 flux 實驗（需選「是」）：", "Extra flux experiment (needs Yes):")}</p>'
            f'{"".join(extra_boxes)}</div>'
        )
    locked_note = (
        f'<p class="muted">{_ui(language, "此工作流程已選定是否進行 flux 校正，不能在中途更改。", "This workflow already chose whether to calibrate flux; it cannot change mid-workflow.")}</p>'
        if locked is not None
        else ""
    )
    return f"""<style>
      .lease-options:has(input[name=flux_calibration][value=no]:checked) .flux-extras {{ opacity:.45; }}
    </style>
    <fieldset class="lease-options">
      <legend>{_ui(language, '是否進行 flux 校正（必選）', 'Calibrate flux? (required)')}</legend>
      <p>{explanation}</p>
      <p>{detected}</p>
      {''.join(radios)}
      {extras_html}
      {locked_note}
    </fieldset>"""


def _broadband_parameter_fields(stored: Any, language: str) -> str:
    """res_num and res_design_freq inputs shown while 02 is ticked.

    Operator request 2026-10-07: 02 needs the expected resonator count and
    design frequencies. The page CSP allows no inline script, so `:has()`
    hides the inputs while 02 is unticked; a browser without `:has()` simply
    always shows them.
    """

    stored = stored if isinstance(stored, dict) else {}
    res_num = stored.get("res_num")
    design = stored.get("res_design_freq") or []
    design_text = ", ".join(
        f"{(float(value) / 1e9 if float(value) >= 1e6 else float(value)):g}"
        for value in design
    )
    return f"""<style>
      .optional-node:not(:has(input[type=checkbox]:checked)) .node-parameters {{ display: none; }}
      .node-parameters {{ margin: .25rem 0 .75rem 1.75rem; display: grid; gap: .4rem; }}
      .node-parameters input[type=text], .node-parameters input[type=number] {{ width: 100%; max-width: 36rem; padding: .35rem; }}
    </style>
    <div class="node-parameters">
      <label>{_ui(language, '共振腔數量 res_num（q1..qn 的 n）', 'Number of resonators res_num (n of q1..qn)')}
        <input type="number" name="node02_res_num" min="1" step="1" value="{escape(str(res_num)) if res_num is not None else ''}"></label>
      <label>{_ui(language, '設計頻率 res_design_freq（GHz，依 q1..qn 順序，以逗號分隔，個數須等於 res_num）', 'Design frequencies res_design_freq (GHz, q1..qn order, comma-separated, res_num entries)')}
        <input type="text" name="node02_res_design_freq" placeholder="5.95, 6.02, 6.10" value="{escape(design_text)}"></label>
      <p class="muted">{_ui(language, '勾選 02 時兩欄都必填。', 'Both fields are required while 02 is ticked.')}</p>
    </div>"""


def _broadband_parameters_from_fields(fields: dict[str, list[str]]) -> dict[str, Any]:
    """Parse the 02 inputs; both are required while 02 is ticked."""

    def one(name: str) -> str:
        values = fields.get(name, [])
        if len(values) > 1:
            raise ServiceError(f"The form sent {name} more than once.")
        return values[0].strip() if values else ""

    num_text = one("node02_res_num")
    design_text = one("node02_res_design_freq")
    if not num_text or not design_text:
        raise ServiceError(
            "02 is ticked: fill in both res_num and res_design_freq, or untick 02."
        )
    try:
        res_num = int(num_text)
    except ValueError:
        raise ServiceError(f"res_num must be a whole number, not {num_text!r}.") from None
    if res_num < 1:
        raise ServiceError("res_num must be at least 1.")
    try:
        design = [
            float(value)
            for value in design_text.replace("，", ",").replace(" ", ",").split(",")
            if value
        ]
    except ValueError:
        raise ServiceError(
            f"res_design_freq must be comma-separated numbers in GHz, not {design_text!r}."
        ) from None
    if not all(math.isfinite(value) and value > 0 for value in design):
        raise ServiceError("Every res_design_freq entry must be a positive frequency.")
    if len(design) != res_num:
        raise ServiceError(
            f"res_design_freq has {len(design)} entries but res_num is {res_num}."
        )
    return {"res_num": res_num, "res_design_freq": design}


def _workflow_options_rows(workflow: Any, language: str) -> str:
    """The lease-page choices in force for this workflow."""

    options = workflow.get("options") if isinstance(workflow, dict) else None
    if not isinstance(options, dict) or not options:
        return ""
    mode = options.get("drive_mode")
    mode_text = {
        "per_qubit": _ui(language, "一顆一顆量", "One qubit at a time"),
        "multiplex": _ui(language, "Multiplex", "Multiplexed"),
    }.get(str(mode), _ui(language, "依 wiring 判斷", "From the wiring"))
    skipped = ", ".join(str(node) for node in options.get("excluded_nodes", [])) or _ui(
        language, "無", "None"
    )
    flux = options.get("flux_calibration")
    flux_text = (
        _ui(language, "是", "Yes")
        if flux is True
        else _ui(language, "否", "No")
    )
    rows = (
        f"<dt>{_ui(language, '量測方式', 'Measurement mode')}</dt><dd>{escape(mode_text)}</dd>"
        f"<dt>{_ui(language, 'Flux 校正', 'Flux calibration')}</dt><dd>{escape(flux_text)}</dd>"
        f"<dt>{_ui(language, '未勾選的實驗', 'Unticked experiments')}</dt><dd>{escape(skipped)}</dd>"
    )
    broadband = (options.get("node_parameters") or {}).get("02")
    if isinstance(broadband, dict):
        design = ", ".join(f"{value:g}" for value in broadband.get("res_design_freq", []))
        rows += (
            f"<dt>{_ui(language, '02 參數', '02 parameters')}</dt>"
            f"<dd>res_num = {escape(str(broadband.get('res_num')))}; "
            f"res_design_freq = [{escape(design)}] GHz</dd>"
        )
    return rows


def _lease_options_from_fields(
    fields: dict[str, list[str]], proposal: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The lease-page choices, or None when this form had none to make."""

    if proposal is None or _lease_option_offer(proposal) is None:
        return None
    modes = fields.get("drive_mode", [])
    if len(modes) > 1:
        raise ServiceError("Choose exactly one measurement mode.")
    flux_values = fields.get("flux_calibration", [])
    if len(flux_values) > 1 or any(value not in {"yes", "no"} for value in flux_values):
        raise ServiceError("Choose yes or no for flux calibration.")
    selected = list(fields.get("optional_node", []))
    start_values = fields.get("start_node", [])
    if len(start_values) > 1:
        raise ServiceError("Choose exactly one start point.")
    offer = _lease_option_offer(proposal) or {}
    if isinstance(offer.get("start_node"), dict) and not start_values:
        raise ServiceError("Choose where this measurement starts.")
    start_node = start_values[0] if start_values else None
    if start_node not in (None, "", "beginning"):
        # Operator choice 2026-10-11: 02 runs only from the beginning; a later
        # start ignores its tick and its (hidden) parameter fields.
        selected = [node for node in selected if node != "02"]
    node_parameters = (
        {"02": _broadband_parameters_from_fields(fields)} if "02" in selected else {}
    )
    history = offer.get("start_node") if isinstance(offer.get("start_node"), dict) else {}
    qubit_start_nodes: dict[str, str] = {}
    if start_node and start_node == history.get("serialize_from_node"):
        # Operator request 2026-10-12: one start node per qubit.
        for qubit in history.get("qubit_start_nodes") or {}:
            values = fields.get(f"qubit_start__{qubit}", [])
            if len(values) > 1:
                raise ServiceError(f"Choose exactly one start point for {qubit}.")
            if values:
                qubit_start_nodes[str(qubit)] = values[0]
    return {
        "optional_nodes": selected,
        "drive_mode": modes[0] if modes else None,
        "flux_calibration": (flux_values[0] == "yes") if flux_values else None,
        "node_parameters": node_parameters,
        "start_node": None if start_node in (None, "", "beginning") else start_node,
        "qubit_start_nodes": qubit_start_nodes,
    }


def _single_field(fields: dict[str, list[str]], name: str) -> str:
    values = fields.get(name)
    if values is None or len(values) != 1:
        raise ValueError(f"Approval form must include one {name} field.")
    return values[0]


def _format_taipei_time(value: Any) -> str:
    """Render an approval timestamp in fixed UTC+08:00 Taipei civil time."""
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    localized = parsed.astimezone(TAIPEI_TIMEZONE)
    return f"{localized.isoformat(timespec='seconds')} (Asia/Taipei)"


def _format_seconds(value: Any) -> str:
    try:
        seconds = max(0.0, float(value))
    except (TypeError, ValueError):
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d} h {minutes:02d} m {remainder:02d} s" if hours else f"{minutes:d} m {remainder:02d} s"
