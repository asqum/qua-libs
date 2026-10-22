"""Static measurement history that survives shutdown (operator 2026-10-22).

`runtime/history_site/` is a self-contained folder of plain HTML pages:

- `index.html`: the chips and every measurement session, newest first, and
  while a session is running a link to its live Dashboard;
- `sessions/<session id>/index.html`: that session's Experiment results page
  (summary, every run with its plots, Decision, Reason and Next action);
- `chips/<chip>/index.html`: the chip history (every successful result of
  the chip across its workflows).

Plots are hard-linked (or copied) next to the pages, so the folder opens from
the lab PC with no JY service running, and it can be copied elsewhere as a
whole. The live service rewrites the affected pages after each analysis and
decision and when a session starts or ends; `jy-agent history-site` rebuilds
every page.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from .chip_history import CHIP_HISTORY_STYLE, link_or_copy, write_static_chip_history
from .util import json_loads, utc_now

LIVE_SESSION_STATUSES = {"active", "paused"}
LIVE_WORKFLOW_STATUSES = {"active", "paused"}

_EXTRA_STYLE = """
table.history-table { width:100%; border-collapse:collapse; }
table.history-table th, table.history-table td { text-align:left; padding:6px 8px; border-top:1px solid #21304d; vertical-align:top; }
table.history-table th { color:#9fb2d7; font-weight:600; }
a { color:#8ab4ff; }
.live { border-color:#3c7a4e; }
"""


def site_root(service: Any) -> Path:
    return Path(service.settings.runtime) / "history_site"


def chip_name(device_key: str | None) -> str:
    return Path(str(device_key or "unknown")).name or "unknown"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _page(title: str, body: str, *, chip_style: bool = False) -> str:
    from .approval_web import DASHBOARD_STYLE

    extra = CHIP_HISTORY_STYLE if chip_style else ""
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
{DASHBOARD_STYLE}
{_EXTRA_STYLE}
{extra}
</style></head><body><main>{body}</main></body></html>"""


def _sessions(service: Any) -> list[dict[str, Any]]:
    return service.db.all(
        "SELECT s.*, w.status AS workflow_status, w.device_key AS device_key, "
        "w.targets_json AS targets_json, w.current_node AS current_node, "
        "(SELECT COUNT(*) FROM runs r WHERE r.workflow_id = s.workflow_id "
        " AND r.inherited_from_run_id IS NULL) AS run_count "
        "FROM measurement_sessions s JOIN workflows w ON w.id = s.workflow_id "
        "ORDER BY s.started_at DESC"
    )


def _is_live(row: dict[str, Any]) -> bool:
    return (
        str(row.get("status")) in LIVE_SESSION_STATUSES
        and str(row.get("workflow_status")) in LIVE_WORKFLOW_STATUSES
    )


def write_session_page(service: Any, session_id: str) -> Path:
    """The static copy of one session's Experiment results page."""

    from .approval_web import (
        _format_taipei_time,
        _localized_enum,
        _operator_notice_section,
        _session_history_section,
        _session_summary_section,
    )

    review = service.dashboard_review(session_id, maintain=False)
    session = review["session"]
    workflow = session.get("workflow") or {}
    folder = site_root(service) / "sessions" / session_id
    images = folder / "images"
    plots = list(review.get("plots") or [])
    names: dict[tuple[str, int], str] = {}
    for item in review.get("history") or []:
        run = item.get("run") or {}
        for position, index in enumerate(item.get("asset_indices") or []):
            if index >= len(plots):
                continue
            source = Path(plots[index])
            if not service._is_allowed_review_asset(source) or not source.is_file():
                continue
            name = f"{run.get('id', 'run')}_{position}{source.suffix.lower()}"
            target = images / name
            if not target.is_file():
                images.mkdir(parents=True, exist_ok=True)
                link_or_copy(source, target)
            names[(str(run.get("id")), position)] = f"images/{quote(name)}"

    def asset_url(_session: str, item: dict[str, Any], position: int, *, thumb: bool = False) -> str:
        return names.get((str((item.get("run") or {}).get("id")), position), "")

    language = "zh-Hant"
    history = review.get("history") or []
    chip = chip_name(workflow.get("device_key"))
    authorization = session.get("authorization") or {}
    rows = [
        ("晶片 Chip", chip),
        ("開始 Started", _format_taipei_time(session.get("started_at"))),
        ("最後更新 Updated", _format_taipei_time(session.get("updated_at"))),
        ("模式 Mode", _localized_enum(language, session.get("mode"))),
        ("狀態 Status", _localized_enum(language, session.get("status"))),
        ("Qubits", ", ".join(str(name) for name in workflow.get("targets") or [])),
        ("Session", session_id),
        ("Workflow", str(session.get("workflow_id"))),
    ]
    if authorization:
        rows.append(
            (
                "授權 Authorization",
                f"{authorization.get('duration_hours')} h · "
                + ("全自動判斷 on" if authorization.get("auto_judgement") else "全自動判斷 off"),
            )
        )
    details = "".join(
        f"<dt>{html.escape(label)}</dt><dd>{html.escape(str(value))}</dd>"
        for label, value in rows
    )
    body = f"""
<p><a href="../../index.html">← 量測歷史 Measurement history</a> ·
<a href="../../chips/{quote(chip)}/index.html">晶片歷史 Chip history</a></p>
<h1>{html.escape(chip)} · {html.escape(_format_taipei_time(session.get('started_at')))}</h1>
<p class="muted">靜態副本，寫於 {html.escape(_format_taipei_time(utc_now()))}；關閉 JY 服務後仍可開啟。
Static copy; it stays readable after JY stops.</p>
<section class="card"><dl>{details}</dl></section>
{_operator_notice_section(session_id, history, language, asset_url=asset_url)}
{_session_summary_section(session_id, history, language, asset_url=asset_url)}
{_session_history_section(session_id, history, language, asset_url=asset_url)}
"""
    target = folder / "index.html"
    _write(target, _page(f"{chip} measurement {session_id}", body))
    return target


def write_chip_page(service: Any, device_key: str) -> Path:
    folder = site_root(service) / "chips" / chip_name(device_key)
    return write_static_chip_history(service, device_key, folder=folder)


def write_index(service: Any) -> Path:
    from .approval_web import _format_taipei_time, _localized_enum

    root = site_root(service)
    language = "zh-Hant"
    sessions = _sessions(service)
    live = [row for row in sessions if _is_live(row)]
    live_items = "".join(
        "<li>"
        + html.escape(chip_name(row.get("device_key")))
        + " · "
        + html.escape(_localized_enum(language, row.get("mode")))
        + " · "
        + html.escape(_localized_enum(language, row.get("status")))
        + f' — <a href="{html.escape(service.session_home_url(str(row["id"])))}">'
        + "開啟即時 Dashboard Open the live Dashboard</a>"
        + (
            f' · <a href="sessions/{quote(str(row["id"]))}/index.html">靜態結果 Static results</a>'
            if (root / "sessions" / str(row["id"]) / "index.html").is_file()
            else ""
        )
        + "</li>"
        for row in live
    )
    live_html = (
        f"<ul>{live_items}</ul><p class=\"muted\">即時 Dashboard 需要 Dashboard 密碼。"
        "The live Dashboard asks for the Dashboard password.</p>"
        if live
        else '<p class="muted">目前沒有進行中的量測。No measurement is running.</p>'
    )
    chips: dict[str, dict[str, Any]] = {}
    for row in sessions:
        key = str(row.get("device_key") or "")
        entry = chips.setdefault(
            key, {"name": chip_name(key), "sessions": 0, "runs": 0, "latest": ""}
        )
        entry["sessions"] += 1
        entry["runs"] += int(row.get("run_count") or 0)
        entry["latest"] = max(entry["latest"], str(row.get("updated_at") or ""))
    chip_rows = "".join(
        "<tr><td>"
        + (
            f'<a href="chips/{quote(entry["name"])}/index.html">{html.escape(entry["name"])}</a>'
            if (root / "chips" / entry["name"] / "index.html").is_file()
            else html.escape(entry["name"])
        )
        + f"</td><td>{entry['sessions']}</td><td>{entry['runs']}</td>"
        + f"<td>{html.escape(_format_taipei_time(entry['latest']))}</td></tr>"
        for entry in sorted(chips.values(), key=lambda item: item["latest"], reverse=True)
    )
    session_rows = "".join(
        "<tr><td>"
        + (
            f'<a href="sessions/{quote(str(row["id"]))}/index.html">'
            f"{html.escape(_format_taipei_time(row.get('started_at')))}</a>"
            if (root / "sessions" / str(row["id"]) / "index.html").is_file()
            else html.escape(_format_taipei_time(row.get("started_at")))
        )
        + f"</td><td>{html.escape(chip_name(row.get('device_key')))}</td>"
        + f"<td>{html.escape(_localized_enum(language, row.get('mode')))}</td>"
        + f"<td>{html.escape(_localized_enum(language, row.get('status')))}</td>"
        + f"<td>{html.escape(', '.join(json_loads(row.get('targets_json'), [])))}</td>"
        + f"<td>{int(row.get('run_count') or 0)}</td></tr>"
        for row in sessions
    )
    body = f"""
<h1>JY 量測歷史 <span class="muted">Measurement history</span></h1>
<p class="muted">靜態網頁，寫於 {html.escape(_format_taipei_time(utc_now()))}；關閉 JY 服務後仍可在量測電腦開啟。
A static page; it opens on the lab PC with no JY service running.</p>
<section class="card live"><h2>進行中 Live</h2>{live_html}</section>
<section class="card"><h2>晶片 Chips</h2>
<table class="history-table"><thead><tr><th>晶片 Chip</th><th>工作階段 Sessions</th><th>實驗 Runs</th><th>最後更新 Updated</th></tr></thead>
<tbody>{chip_rows}</tbody></table></section>
<section class="card"><h2>量測工作階段 Sessions</h2>
<table class="history-table"><thead><tr><th>開始 Started</th><th>晶片 Chip</th><th>模式 Mode</th><th>狀態 Status</th><th>Qubits</th><th>實驗 Runs</th></tr></thead>
<tbody>{session_rows}</tbody></table></section>
"""
    target = root / "index.html"
    _write(target, _page("JY measurement history", body))
    return target


def write_history_site(
    service: Any,
    *,
    session_ids: Iterable[str] | None = None,
    everything: bool = False,
) -> Path:
    """Rewrite the given sessions' pages, their chips' pages and the index.

    `everything` rebuilds every session and chip page (the CLI's full
    rebuild); otherwise only the pages of `session_ids` and their chips are
    rewritten. The index is always rewritten.
    """

    sessions = _sessions(service)
    wanted = {str(item) for item in (session_ids or [])}
    chips: set[str] = set()
    for row in sessions:
        session_id = str(row["id"])
        if everything or session_id in wanted:
            write_session_page(service, session_id)
            if row.get("device_key"):
                chips.add(str(row["device_key"]))
    for device_key in sorted(chips):
        write_chip_page(service, device_key)
    return write_index(service)
