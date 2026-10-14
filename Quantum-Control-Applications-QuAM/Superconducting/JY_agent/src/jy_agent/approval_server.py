from __future__ import annotations

import hmac
import os
import time
from html import escape
from urllib.parse import parse_qs, urlencode

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from .approval_web import (
    handle_autonomy_control,
    handle_autonomy_control_asset,
    handle_autonomy_events,
    handle_autonomy_events_script,
    handle_browser_approval,
    handle_browser_approval_asset,
    handle_guide_asset,
    handle_guide_page,
    handle_language_script,
    handle_chip_history_asset,
    handle_session_dashboard,
    handle_session_dashboard_asset,
    handle_session_dashboard_events,
    handle_session_dashboard_script,
)
from .config import Settings
from .dashboard_access import DashboardAccessManager
from .service import AgentService


DASHBOARD_PASSWORD_FILENAME = "dashboard-password.txt"
DASHBOARD_COOKIE = "jy_dashboard_access"
LOGIN_WINDOW_SECONDS = 5 * 60
LOGIN_MAX_FAILURES = 5
LOGIN_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
        "base-uri 'none'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


def create_approval_app(settings: Settings | None = None) -> Starlette:
    """Create the review-only web app; it intentionally exposes no MCP route."""
    resolved = settings or Settings.load()
    service = AgentService(resolved)
    dashboard_access = DashboardAccessManager(service.db)
    login_failures: dict[str, list[float]] = {}

    async def approval(request: Request) -> Response:
        return await handle_browser_approval(request, service)

    async def asset(request: Request) -> Response:
        return await handle_browser_approval_asset(request, service)

    async def autonomy(request: Request) -> Response:
        return await handle_autonomy_control(request, service)

    async def autonomy_asset(request: Request) -> Response:
        return await handle_autonomy_control_asset(request, service)

    async def autonomy_events(request: Request) -> Response:
        return await handle_autonomy_events(request, service)

    async def session_dashboard(request: Request) -> Response:
        return await handle_session_dashboard(request, service)

    async def dashboard_entry(request: Request) -> Response:
        return await handle_session_dashboard(request, service, view="home")

    async def session_home(request: Request) -> Response:
        return await handle_session_dashboard(request, service, view="home")

    async def session_approval(request: Request) -> Response:
        return await handle_session_dashboard(request, service, view="approval")

    async def session_results(request: Request) -> Response:
        return await handle_session_dashboard(request, service, view="results")

    async def session_usage(request: Request) -> Response:
        return await handle_session_dashboard(request, service, view="usage")

    async def session_guide(request: Request) -> Response:
        return await handle_session_dashboard(request, service, view="guide")

    async def guide(request: Request) -> Response:
        return await handle_guide_page(request, service)

    async def guide_asset(request: Request) -> Response:
        return await handle_guide_asset(request, service)

    async def session_usage_events(request: Request) -> Response:
        return await handle_session_dashboard_events(request, service, view="usage")

    async def session_asset(request: Request) -> Response:
        return await handle_session_dashboard_asset(request, service)

    async def chip_asset(request: Request) -> Response:
        return await handle_chip_history_asset(request, service)

    async def session_events(request: Request) -> Response:
        return await handle_session_dashboard_events(request, service)

    async def session_home_events(request: Request) -> Response:
        return await handle_session_dashboard_events(request, service, view="home")

    async def session_approval_events(request: Request) -> Response:
        return await handle_session_dashboard_events(request, service, view="approval")

    async def session_results_events(request: Request) -> Response:
        return await handle_session_dashboard_events(request, service, view="results")

    async def healthz(request: Request) -> Response:
        status = service.status()
        return JSONResponse(
            {
                "service": "jy-approval",
                "pid": os.getpid(),
                "instance_nonce": os.getenv("JY_SERVICE_INSTANCE_NONCE", ""),
                "remote_approval_enabled": status["server"][
                    "remote_approval_enabled"
                ],
                "approval_endpoint": status["server"]["approval_endpoint"],
                "recovery_only": status["server"]["recovery_only"],
            }
        )

    async def login(request: Request) -> Response:
        return await _handle_login(
            request,
            resolved,
            dashboard_access,
            login_failures,
        )

    async def logout(request: Request) -> Response:
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(DASHBOARD_COOKIE, path="/")
        return response

    app = Starlette(
        debug=False,
        routes=[
            Route("/", dashboard_entry, methods=["GET"], name="dashboard_entry"),
            Route("/healthz", healthz, methods=["GET"], name="healthz"),
            Route("/login", login, methods=["GET", "POST"], name="login"),
            Route("/logout", logout, methods=["GET", "POST"], name="logout"),
            Route(
                "/approve/{proposal_id}",
                approval,
                methods=["GET", "POST"],
                name="browser_approval",
            ),
            Route(
                "/approve/{proposal_id}/assets/{asset_index:int}",
                asset,
                methods=["GET"],
                name="browser_approval_asset",
            ),
            Route(
                "/autonomy/{lease_id}",
                autonomy,
                methods=["GET", "POST"],
                name="autonomy_control",
            ),
            Route(
                "/autonomy/{lease_id}/assets/{asset_index:int}",
                autonomy_asset,
                methods=["GET"],
                name="autonomy_control_asset",
            ),
            Route(
                "/autonomy/{lease_id}/events",
                autonomy_events,
                methods=["GET"],
                name="autonomy_events",
            ),
            Route(
                "/assets/autonomy-events.js",
                handle_autonomy_events_script,
                methods=["GET"],
                name="autonomy_events_script",
            ),
            Route(
                "/session/{session_id}",
                session_dashboard,
                methods=["GET", "POST"],
                name="session_dashboard",
            ),
            Route(
                "/session/{session_id}/home",
                session_home,
                methods=["GET", "POST"],
                name="session_home",
            ),
            Route(
                "/session/{session_id}/approval",
                session_approval,
                methods=["GET", "POST"],
                name="session_approval",
            ),
            Route(
                "/session/{session_id}/results",
                session_results,
                methods=["GET", "POST"],
                name="session_results",
            ),
            Route(
                "/session/{session_id}/usage",
                session_usage,
                methods=["GET", "POST"],
                name="session_usage",
            ),
            Route(
                "/session/{session_id}/guide",
                session_guide,
                methods=["GET"],
                name="session_guide",
            ),
            Route("/guide", guide, methods=["GET"], name="guide"),
            Route(
                "/guide/assets/{name}",
                guide_asset,
                methods=["GET"],
                name="guide_asset",
            ),
            Route(
                "/session/{session_id}/usage/events",
                session_usage_events,
                methods=["GET"],
                name="session_usage_events",
            ),
            Route(
                "/session/{session_id}/assets/{asset_index:int}",
                session_asset,
                methods=["GET"],
                name="session_dashboard_asset",
            ),
            Route(
                "/session/{session_id}/chip-assets/{run_id}/{position:int}",
                chip_asset,
                methods=["GET"],
                name="session_chip_history_asset",
            ),
            Route(
                "/session/{session_id}/events",
                session_events,
                methods=["GET"],
                name="session_dashboard_events",
            ),
            Route(
                "/session/{session_id}/home/events",
                session_home_events,
                methods=["GET"],
                name="session_home_events",
            ),
            Route(
                "/session/{session_id}/approval/events",
                session_approval_events,
                methods=["GET"],
                name="session_approval_events",
            ),
            Route(
                "/session/{session_id}/results/events",
                session_results_events,
                methods=["GET"],
                name="session_results_events",
            ),
            Route(
                "/assets/language.js",
                handle_language_script,
                methods=["GET"],
                name="language_script",
            ),
            Route(
                "/assets/session-dashboard.js",
                handle_session_dashboard_script,
                methods=["GET"],
                name="session_dashboard_script",
            ),
        ],
    )

    @app.middleware("http")
    async def public_password_access(request: Request, call_next):
        if resolved.approval_transport != "public":
            return await call_next(request)
        expected = resolved.approval_access_token or ""
        health_token = request.headers.get("x-jy-health-token", "")
        if (
            request.url.path == "/healthz"
            and bool(expected)
            and hmac.compare_digest(health_token, expected)
        ):
            return await call_next(request)
        if request.url.path in {"/login", "/logout"}:
            return await call_next(request)
        cookie = request.cookies.get(DASHBOARD_COOKIE, "")
        device = dashboard_access.validate_device(cookie)
        if device is None:
            target = request.url.path
            if request.url.query:
                target += "?" + request.url.query
            return RedirectResponse(
                "/login?" + urlencode({"next": target}),
                status_code=303,
            )
        request.state.jy_public_authenticated = True
        return await call_next(request)

    return app


async def _handle_login(
    request: Request,
    settings: Settings,
    access: DashboardAccessManager,
    failures: dict[str, list[float]],
) -> Response:
    next_path = _safe_next_path(request.query_params.get("next", "/"))
    if request.method == "GET":
        return _login_page(next_path)

    client_key = _login_client_key(request)
    retry_after = _login_retry_after(failures, client_key)
    if retry_after > 0:
        return _login_page(
            next_path,
            error="登入嘗試過多，請稍後再試。 / Too many attempts; try again later.",
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    try:
        if not request.headers.get("content-type", "").lower().startswith(
            "application/x-www-form-urlencoded"
        ):
            raise ValueError("Invalid login form encoding")
        body = await request.body()
        if len(body) > 4096:
            raise ValueError("Login form was too large")
        fields = parse_qs(
            body.decode("utf-8"),
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=3,
        )
        passwords = fields.get("password", [])
        next_values = fields.get("next", [next_path])
        if len(passwords) != 1 or len(next_values) != 1:
            raise ValueError("Invalid login form")
        supplied = passwords[0]
        next_path = _safe_next_path(next_values[0])
    except (UnicodeDecodeError, ValueError):
        return _login_page(
            next_path,
            error="登入資料格式錯誤。 / Invalid login request.",
            status_code=400,
        )
    try:
        expected = _read_dashboard_password(settings)
    except (OSError, ValueError):
        return _login_page(
            next_path,
            error="Dashboard 密碼目前無法使用，請由量測電腦檢查密碼檔。 / "
            "Dashboard password unavailable; inspect the password file on the lab PC.",
            status_code=503,
        )

    if not hmac.compare_digest(supplied, expected):
        failures.setdefault(client_key, []).append(time.monotonic())
        return _login_page(
            next_path,
            error="密碼錯誤。 / Incorrect password.",
            status_code=401,
        )

    failures.pop(client_key, None)
    session = access.issue_password_session(
        actor=f"Dashboard password login from {client_key}"
    )
    response = RedirectResponse(next_path, status_code=303)
    response.set_cookie(
        DASHBOARD_COOKIE,
        str(session["token"]),
        max_age=12 * 60 * 60,
        httponly=True,
        secure=True,
        samesite="strict",
        path="/",
    )
    return response


def _read_dashboard_password(settings: Settings) -> str:
    password_path = settings.runtime / DASHBOARD_PASSWORD_FILENAME
    password = password_path.read_text(encoding="utf-8-sig").strip()
    if not 12 <= len(password) <= 256 or "\n" in password or "\r" in password:
        raise ValueError(
            f"{DASHBOARD_PASSWORD_FILENAME} must contain one 12-256 character password"
        )
    return password


def _login_page(
    next_path: str,
    *,
    error: str | None = None,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> HTMLResponse:
    notice = (
        f'<p class="error" role="alert">{escape(error)}</p>' if error else ""
    )
    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JY Dashboard login</title><style>
:root{{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif}}
*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;display:grid;place-items:center;
background:#08101d;color:#edf4ff;padding:20px}}main{{width:min(420px,100%);
background:#121d31;border:1px solid #2b3b5d;border-radius:16px;padding:24px}}
h1{{margin-top:0}}label{{display:block;margin:18px 0 8px}}input,button{{width:100%;
padding:12px;border-radius:9px;font:inherit}}input{{background:#07101f;color:white;
border:1px solid #6680b5}}button{{margin-top:14px;border:0;background:#3568e8;
color:white;font-weight:750;cursor:pointer}}.muted{{color:#b7c4df}}.error{{background:#5a1e2a;
padding:11px;border-radius:8px}}
</style></head><body><main><h1>JY 量測 Dashboard</h1>
<p class="muted">請輸入實驗室固定密碼。<br>Enter the shared lab password.</p>
{notice}<form method="post" action="/login">
<input type="hidden" name="next" value="{escape(_safe_next_path(next_path))}">
<label for="password">密碼 / Password</label>
<input id="password" name="password" type="password" autocomplete="current-password"
autofocus required maxlength="256"><button type="submit">登入 / Sign in</button>
</form></main></body></html>"""
    response_headers = dict(LOGIN_SECURITY_HEADERS)
    if headers:
        response_headers.update(headers)
    return HTMLResponse(html, status_code=status_code, headers=response_headers)


def _safe_next_path(value: str) -> str:
    candidate = str(value or "/")
    if (
        not candidate.startswith("/")
        or candidate.startswith("//")
        or "\r" in candidate
        or "\n" in candidate
    ):
        return "/"
    return candidate


def _login_client_key(request: Request) -> str:
    return (
        request.headers.get("cf-connecting-ip")
        or (request.client.host if request.client else "unknown")
    ).strip()[:128]


def _login_retry_after(
    failures: dict[str, list[float]], client_key: str
) -> int:
    now = time.monotonic()
    recent = [
        attempted
        for attempted in failures.get(client_key, [])
        if now - attempted < LOGIN_WINDOW_SECONDS
    ]
    if recent:
        failures[client_key] = recent
    else:
        failures.pop(client_key, None)
    if len(recent) < LOGIN_MAX_FAILURES:
        return 0
    return max(1, int(LOGIN_WINDOW_SECONDS - (now - recent[0])))


def run_approval_server(
    settings: Settings | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
) -> None:
    import uvicorn

    resolved = settings or Settings.load()
    uvicorn.run(
        create_approval_app(resolved),
        host=host or resolved.approval_host,
        port=port or resolved.approval_port,
        log_level="info",
        # Avoid persisting login endpoints, client addresses, or dashboard paths.
        access_log=False,
    )


if __name__ == "__main__":
    run_approval_server()
