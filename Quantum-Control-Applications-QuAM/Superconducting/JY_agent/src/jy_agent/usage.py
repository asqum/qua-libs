"""Model and token usage per JY run, read from Claude Code session transcripts.

Operator request 2026-10-03: the Dashboard shows which model each experiment
used, how many tokens, and an estimated cost. MCP does not pass token usage to
the server, so this reads the transcripts Claude Code keeps under
``~/.claude/projects/<project>/``. Only numbers are extracted; no conversation
text leaves this module.

One API call can appear on several transcript lines (one per content block),
so calls are de-duplicated by message id. Sub-agent transcripts also write a
streaming call before it finishes, with a partial output count (checked
against ccusage 2026-10-14), so each token count keeps its largest value.
Attribution, per call: a call that uses a JY MCP tool is measurement work and
belongs to the JY run whose window [started_at, next run's started_at)
contains it (before the first run: session setup). A call that edits files
(Edit, Write, NotebookEdit) is development. Any other call (reading, waiting,
answering) takes the label of the next labelled call in the same user turn, or
the previous one; a turn with neither kind is conversation.

Operator request 2026-10-14: the page covers the whole chip, not only the
current session. A workflow owns the calls from its creation to its close or
the next workflow's creation, whichever is first; ``chip_usage`` adds up the
windows of every workflow on one chip, optionally cut to a time range.

Operator request 2026-10-15: usage is kept for good. Claude Code deletes old
transcripts (about 30 days by default), so ``sync_calls`` saves each call's
numbers in the JY database and the page reads the saved calls merged with the
transcripts still on disk. ``chips_summary`` gives one row per chip.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_left
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any, Iterable

DEVELOPMENT_TOOLS = frozenset({"Edit", "Write", "NotebookEdit", "MultiEdit"})
JY_TOOL_PREFIX = "mcp__jy_bringup__"
TOKEN_KEYS = ("input", "cache_write_5m", "cache_write_1h", "cache_read", "output")
# A workflow in one of these states is still open; any other status closed it
# at its updated_at.
OPEN_WORKFLOW_STATUSES = frozenset({"active", "paused", "recovery_required"})

_CACHE: dict[str, tuple[tuple[float, int], list["Call"]]] = {}


@dataclass
class Call:
    message_id: str
    timestamp: datetime
    model: str
    speed: str
    tokens: dict[str, int]
    turn: str
    tools: set[str] = field(default_factory=set)


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def default_transcript_dir(agent_root: Path) -> Path | None:
    """The Claude Code project folder for the repository that holds .mcp.json."""
    repo = next(
        (p for p in [agent_root, *agent_root.parents] if (p / ".mcp.json").is_file()),
        None,
    )
    projects = Path.home() / ".claude" / "projects"
    if repo is None or not projects.is_dir():
        return None
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(repo)).lower()
    for candidate in projects.iterdir():
        if candidate.is_dir() and candidate.name.lower() == slug:
            return candidate
    return None


def load_calls(root: Path) -> list[Call]:
    """Every distinct API call in every transcript below ``root``."""
    calls: dict[str, Call] = {}
    for path in sorted(root.rglob("*.jsonl")):
        if "memory" in path.parts or "tool-results" in path.parts:
            continue
        for call in _file_calls(path):
            known = calls.get(call.message_id)
            if known is None:
                calls[call.message_id] = call
            else:
                _absorb(known, call.tools, call.tokens)
    return sorted(calls.values(), key=lambda item: item.timestamp)


def _file_calls(path: Path) -> list[Call]:
    stat = path.stat()
    key = str(path)
    signature = (stat.st_mtime, stat.st_size)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    by_id: dict[str, Call] = {}
    turn = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            kind = entry.get("type")
            message = entry.get("message") if isinstance(entry.get("message"), dict) else {}
            if kind == "user" and _is_user_input(message.get("content")):
                turn += 1
                continue
            if kind != "assistant" or not isinstance(message.get("usage"), dict):
                continue
            stamp = parse_time(entry.get("timestamp"))
            message_id = str(message.get("id") or entry.get("requestId") or "")
            if stamp is None or not message_id:
                continue
            tools = {
                str(block.get("name"))
                for block in message.get("content") or []
                if isinstance(block, dict) and block.get("type") == "tool_use"
            }
            call = by_id.get(message_id)
            if call is None:
                usage = message["usage"]
                by_id[message_id] = Call(
                    message_id=message_id,
                    timestamp=stamp,
                    model=str(message.get("model") or "unknown"),
                    speed=str(usage.get("speed") or "standard"),
                    tokens=_tokens(usage),
                    turn=f"{path.name}#{turn}",
                    tools=tools,
                )
            else:
                _absorb(call, tools, _tokens(message["usage"]))
    calls = list(by_id.values())
    _CACHE[key] = (signature, calls)
    return calls


def _absorb(call: Call, tools: set[str], tokens: dict[str, int]) -> None:
    """Fold another line of the same API call into ``call``."""
    call.tools |= tools
    for key in TOKEN_KEYS:
        call.tokens[key] = max(call.tokens[key], tokens[key])


def _is_user_input(content: Any) -> bool:
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return any(
            isinstance(block, dict) and block.get("type") != "tool_result"
            for block in content
        )
    return False


def _tokens(usage: dict[str, Any]) -> dict[str, int]:
    detail = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else {}
    write_total = int(usage.get("cache_creation_input_tokens") or 0)
    write_1h = int(detail.get("ephemeral_1h_input_tokens") or 0)
    write_5m = int(detail.get("ephemeral_5m_input_tokens") or (write_total - write_1h))
    return {
        "input": int(usage.get("input_tokens") or 0),
        "cache_write_5m": max(write_5m, 0),
        "cache_write_1h": write_1h,
        "cache_read": int(usage.get("cache_read_input_tokens") or 0),
        "output": int(usage.get("output_tokens") or 0),
    }


def stored_calls(db: Any) -> list[Call]:
    """Every call saved in the JY database."""
    calls = []
    for row in db.all("SELECT * FROM usage_calls"):
        stamp = parse_time(row["timestamp"])
        if stamp is None:
            continue
        tools = json.loads(row["tools_json"] or "[]")
        calls.append(
            Call(
                message_id=str(row["message_id"]),
                timestamp=stamp,
                model=str(row["model"]),
                speed=str(row["speed"]),
                tokens={key: int(row[key] or 0) for key in TOKEN_KEYS},
                turn=str(row["turn"]),
                tools={str(name) for name in tools},
            )
        )
    return calls


def sync_calls(db: Any, root: Path | None) -> list[Call]:
    """Saved calls merged with the transcripts below ``root``; new or grown calls are saved.

    Token counts only grow (a streaming line is replaced by the final one),
    so a saved row keeps the larger of its stored and transcript values.
    """
    known = {call.message_id: call for call in stored_calls(db)}
    changed: list[Call] = []
    if root is not None and root.is_dir():
        for call in load_calls(root):
            saved = known.get(call.message_id)
            if saved is None:
                # A copy: load_calls returns objects its file cache keeps.
                saved = replace(call, tokens=dict(call.tokens), tools=set(call.tools))
                known[call.message_id] = saved
                changed.append(saved)
            elif not call.tools <= saved.tools or any(
                call.tokens[key] > saved.tokens[key] for key in TOKEN_KEYS
            ):
                _absorb(saved, call.tools, call.tokens)
                changed.append(saved)
    if changed:
        grow = ", ".join(f"{key} = MAX({key}, excluded.{key})" for key in TOKEN_KEYS)
        with db.connect() as connection:
            connection.executemany(
                "INSERT INTO usage_calls (message_id, timestamp, model, speed, turn, "
                f"tools_json, {', '.join(TOKEN_KEYS)}) VALUES ({', '.join('?' * (6 + len(TOKEN_KEYS)))}) "
                f"ON CONFLICT(message_id) DO UPDATE SET tools_json = excluded.tools_json, {grow}",
                [
                    (
                        call.message_id,
                        call.timestamp.isoformat(),
                        call.model,
                        call.speed,
                        call.turn,
                        json.dumps(sorted(call.tools)),
                        *(call.tokens[key] for key in TOKEN_KEYS),
                    )
                    for call in changed
                ],
            )
    return sorted(known.values(), key=lambda item: item.timestamp)


def call_cost(call: Call, pricing: dict[str, Any]) -> float | None:
    """Estimated USD at API list prices; None for a model without a price."""
    prices = (pricing.get("models") or {}).get(call.model)
    if not isinstance(prices, dict):
        return None
    rate_in = float(prices["input"])
    cost = (
        call.tokens["input"] * rate_in
        + call.tokens["cache_write_5m"] * rate_in * float(pricing.get("cache_write_5m_multiplier", 1.25))
        + call.tokens["cache_write_1h"] * rate_in * float(pricing.get("cache_write_1h_multiplier", 2.0))
        + call.tokens["cache_read"] * float(prices["cache_read"])
        + call.tokens["output"] * float(prices["output"])
    ) / 1_000_000
    if call.speed == "fast":
        cost *= float(pricing.get("fast_multiplier", 2.0))
    return cost


def _empty() -> dict[str, Any]:
    return {"calls": 0, "cost_usd": 0.0, "unpriced_calls": 0, "models": {}, **{k: 0 for k in TOKEN_KEYS}}


def total_tokens(bucket: dict[str, Any]) -> int:
    return sum(int(bucket.get(key, 0)) for key in TOKEN_KEYS)


def _merge(into: dict[str, Any], bucket: dict[str, Any]) -> None:
    for key in ("calls", "cost_usd", "unpriced_calls", *TOKEN_KEYS):
        into[key] += bucket[key]
    for model, count in bucket["models"].items():
        into["models"][model] = into["models"].get(model, 0) + count


def _add(bucket: dict[str, Any], call: Call, cost: float | None) -> None:
    bucket["calls"] += 1
    for key in TOKEN_KEYS:
        bucket[key] += call.tokens[key]
    if cost is None:
        bucket["unpriced_calls"] += 1
    else:
        bucket["cost_usd"] += cost
    bucket["models"][call.model] = bucket["models"].get(call.model, 0) + 1


def attribute_usage(
    calls: Iterable[Call],
    runs: list[dict[str, Any]],
    pricing: dict[str, Any],
    start: datetime,
    end: datetime | None = None,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate calls in [start, end) into per-run, per-node, setup and development buckets.

    ``labels`` lets a caller label every call once, so a turn cut by a window
    boundary keeps the label it has in the whole transcript.
    """
    ordered = sorted(
        (run for run in runs if parse_time(run.get("started_at")) is not None),
        key=lambda run: parse_time(run["started_at"]),
    )
    starts = [parse_time(run["started_at"]) for run in ordered]
    selected = [
        call
        for call in calls
        if call.timestamp >= start and (end is None or call.timestamp < end)
    ]
    if labels is None:
        labels = _labels(selected)
    per_run = {str(run["id"]): _empty() for run in ordered}
    setup, development, conversation, total = _empty(), _empty(), _empty(), _empty()
    for call in selected:
        cost = call_cost(call, pricing)
        _add(total, call, cost)
        label = labels[call.message_id]
        if label == "development":
            _add(development, call, cost)
            continue
        if label == "conversation":
            _add(conversation, call, cost)
            continue
        index = _window(starts, call.timestamp)
        if index is None:
            _add(setup, call, cost)
        else:
            _add(per_run[str(ordered[index]["id"])], call, cost)
    run_rows = []
    per_node: dict[str, dict[str, Any]] = {}
    for run in ordered:
        bucket = per_run[str(run["id"])]
        node = str(run.get("node_id"))
        run_rows.append(
            {
                "run_id": str(run["id"]),
                "node_id": node,
                "qubits": list(run.get("qubits") or []),
                "started_at": run.get("started_at"),
                **bucket,
            }
        )
        node_bucket = per_node.setdefault(node, {**_empty(), "runs": 0})
        node_bucket["runs"] += 1
        _merge(node_bucket, bucket)
    return {
        "runs": run_rows,
        "nodes": per_node,
        "setup": setup,
        "development": development,
        "conversation": conversation,
        "total": total,
    }


def workflow_windows(
    workflows: Iterable[dict[str, Any]],
) -> list[tuple[dict[str, Any], datetime, datetime | None]]:
    """(workflow, start, end) per workflow; end None while it is still open.

    Workflows of every chip are needed here: the next workflow's creation
    closes the previous one even when it measures another chip.
    """
    ordered = sorted(
        (item for item in workflows if parse_time(item.get("created_at")) is not None),
        key=lambda item: parse_time(item["created_at"]),
    )
    windows = []
    for position, workflow in enumerate(ordered):
        start = parse_time(workflow["created_at"])
        end = (
            parse_time(ordered[position + 1]["created_at"])
            if position + 1 < len(ordered)
            else None
        )
        if workflow.get("status") not in OPEN_WORKFLOW_STATUSES:
            closed = parse_time(workflow.get("updated_at"))
            if closed is not None and closed >= start and (end is None or closed < end):
                end = closed
        windows.append((workflow, start, end))
    return windows


def chip_usage(
    calls: Iterable[Call],
    workflows: Iterable[dict[str, Any]],
    runs_by_workflow: dict[str, list[dict[str, Any]]],
    pricing: dict[str, Any],
    device_key: str,
    start: datetime | None = None,
    end: datetime | None = None,
    day_zone: tzinfo = timezone.utc,
    only_workflow: str | None = None,
) -> dict[str, Any]:
    """Usage of one chip's workflows inside [start, end), plus every call in it.

    ``only_workflow`` keeps a single workflow's window (the session view)
    while the other workflows still decide where that window ends.

    ``all_usage`` counts every call in the range whatever chip or workflow it
    belongs to (operator choice 2026-10-14: workflows without a chip and work
    between workflows appear only there). ``days`` holds each civil date in
    ``day_zone`` with all of the chip's calls on that date.
    """
    calls = sorted(calls, key=lambda item: item.timestamp)
    labels = _labels(calls)
    in_range = [
        call
        for call in calls
        if (start is None or call.timestamp >= start)
        and (end is None or call.timestamp < end)
    ]
    everything = {key: _empty() for key in ("measurement", "development", "conversation", "total")}
    for call in in_range:
        cost = call_cost(call, pricing)
        _add(everything["total"], call, cost)
        _add(everything[labels[call.message_id]], call, cost)

    chip_windows = [
        window for window in workflow_windows(workflows)
        if window[0].get("device_key") == device_key
    ]
    result: dict[str, Any] = {
        "chip_started_at": chip_windows[0][1].isoformat() if chip_windows else None,
        "workflows": len(chip_windows),
        "runs": [],
        "nodes": {},
        "days": {},
        "setup": _empty(),
        "development": _empty(),
        "conversation": _empty(),
        "total": _empty(),
        "all_usage": everything,
    }
    for workflow, opened, closed in chip_windows:
        if only_workflow is not None and str(workflow["id"]) != only_workflow:
            continue
        low =opened if start is None else max(opened, start)
        high = closed if end is None else (end if closed is None else min(closed, end))
        if high is not None and high <= low:
            continue
        part = attribute_usage(
            calls,
            runs_by_workflow.get(str(workflow["id"]), []),
            pricing,
            low,
            high,
            labels=labels,
        )
        for row in part["runs"]:
            begun = parse_time(row.get("started_at"))
            started_inside = begun is not None and begun >= low and (high is None or begun < high)
            if row["calls"] or started_inside:
                result["runs"].append({**row, "workflow_id": str(workflow["id"])})
        for node, bucket in part["nodes"].items():
            node_bucket = result["nodes"].setdefault(node, {**_empty(), "runs": 0})
            node_bucket["runs"] += bucket["runs"]
            _merge(node_bucket, bucket)
        for key in ("setup", "development", "conversation", "total"):
            _merge(result[key], part[key])
        for call in calls:
            if call.timestamp >= low and (high is None or call.timestamp < high):
                day = call.timestamp.astimezone(day_zone).date().isoformat()
                _add(result["days"].setdefault(day, _empty()), call, call_cost(call, pricing))
    result["runs"].sort(key=lambda row: parse_time(row["started_at"]))
    return result


def chips_summary(
    calls: Iterable[Call],
    workflows: Iterable[dict[str, Any]],
    run_counts: dict[str, int],
    pricing: dict[str, Any],
) -> list[dict[str, Any]]:
    """One row per chip, newest activity first (operator request 2026-10-15).

    A chip's usage is every call inside its workflow windows, the same total
    ``chip_usage`` reports. ``last_at`` is its latest call, or its latest
    workflow's creation when no call fell inside.
    """
    ordered = sorted(calls, key=lambda item: item.timestamp)
    stamps = [call.timestamp for call in ordered]
    rows: dict[str, dict[str, Any]] = {}
    for workflow, opened, closed in workflow_windows(workflows):
        key = workflow.get("device_key")
        if not key:
            continue
        row = rows.setdefault(
            str(key),
            {
                "device_key": str(key),
                "chip_name": Path(str(key)).name,
                "first_at": opened,
                "last_at": opened,
                "workflows": 0,
                "runs": int(run_counts.get(str(key), 0)),
                **_empty(),
            },
        )
        row["workflows"] += 1
        row["last_at"] = max(row["last_at"], opened)
        low = bisect_left(stamps, opened)
        high = len(ordered) if closed is None else bisect_left(stamps, closed)
        for call in ordered[low:high]:
            _add(row, call, call_cost(call, pricing))
            row["last_at"] = max(row["last_at"], call.timestamp)
    result = sorted(rows.values(), key=lambda row: row["last_at"], reverse=True)
    for row in result:
        row["first_at"] = row["first_at"].isoformat()
        row["last_at"] = row["last_at"].isoformat()
    return result


def _labels(calls: list[Call]) -> dict[str, str]:
    """measurement / development / conversation for every call."""
    turns: dict[str, list[Call]] = {}
    for call in sorted(calls, key=lambda item: item.timestamp):
        turns.setdefault(call.turn, []).append(call)
    labels: dict[str, str] = {}
    for members in turns.values():
        own: list[str | None] = []
        for call in members:
            if call.tools & DEVELOPMENT_TOOLS:
                own.append("development")
            elif any(name.startswith(JY_TOOL_PREFIX) for name in call.tools):
                own.append("measurement")
            else:
                own.append(None)
        for position, call in enumerate(members):
            label = own[position]
            if label is None:
                later = next((item for item in own[position + 1 :] if item), None)
                earlier = next((item for item in reversed(own[:position]) if item), None)
                label = later or earlier or "conversation"
            labels[call.message_id] = label
    return labels


def _window(starts: list[datetime], stamp: datetime) -> int | None:
    index = None
    for position, begin in enumerate(starts):
        if begin <= stamp:
            index = position
        else:
            break
    return index
