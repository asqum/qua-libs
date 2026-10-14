"""Chip history: every successful result of a chip, across its workflows.

Operator request 2026-10-13: the Experiment results page (and a static HTML
copy that survives shutdown) lists, for the chip in use, each node in sequence
order and under it each qubit that passed, with that result's plots and key
values -- so results stay visible after a lease or session ends. A qubit counts
when it is in the run's `passing_targets`; the newest success per node and
qubit leads, earlier ones fold underneath. Runs a workflow inherited from an
earlier one are skipped (the original is already listed).
"""

from __future__ import annotations

import html
import math
import re
import shutil
from pathlib import Path
from typing import Any, Callable

from .util import json_loads

_QUBIT_IN_NAME = re.compile(r"(?<![A-Za-z0-9])(q[0-9]+)(?![0-9])")
_HIDDEN_PATCH_KEYS = {"intermediate_frequency", "integration_weights_angle"}
_LIFETIME_LABEL = {"05": "T1", "06": "T2*", "06b": "T2E"}
_STATISTICS_LABEL = {"05st": "T1", "06st_t2star": "T2*", "06st_t2e": "T2E"}
# State keys that repeat a value already shown under a fit label.
_PATCH_LABELS = {
    "T1": "T1",
    "T2echo": "T2E",
    "T2": "T2E",
    "T2ramsey": "T2*",
    "idle_freq": "f_q",
    "T1_dev": "T1 σ",
    "T2_dev": "T2E σ",
    "T2ramsey_dev": "T2* σ",
}


def _natural(name: str) -> tuple[Any, ...]:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(name)))


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _format_value(key: str, value: Any) -> str:
    if not _finite(value):
        return str(value)
    number = float(value)
    if abs(number) >= 1e8:
        return f"{number / 1e9:.6g} GHz"
    if key.lower() in {"t1", "t2", "t2echo", "t2ramsey", "t1_dev", "t2_dev", "t2ramsey_dev"} or (
        0 < abs(number) < 1e-3 and key.lower().startswith("t")
    ):
        return f"{number * 1e6:.4g} µs"
    if key == "length":
        return f"{number:g} ns"
    return f"{number:.4g}"


def _key_values(node_id: str, qubit: str, analysis: dict[str, Any]) -> list[tuple[str, str]]:
    """The values worth reading at a glance, in a stable order."""

    values: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(label: str, text: str) -> None:
        if label not in seen:
            seen.add(label)
            values.append((label, text))

    fit = (analysis.get("fit_quality") or {}).get("results") or {}
    fit = fit.get(qubit) if isinstance(fit, dict) else None
    fit = fit if isinstance(fit, dict) else {}
    if _finite(fit.get("drive_freq")):
        add("f_q", f"{float(fit['drive_freq']) / 1e9:.6g} GHz")
    if _finite(fit.get("Pi_amplitude")):
        add("π amplitude", f"{float(fit['Pi_amplitude']):.4g}")
    if _finite(fit.get("t1_seconds")):
        add("T1", f"{float(fit['t1_seconds']) * 1e6:.4g} µs")
    if _finite(fit.get("coherence_seconds")) and float(fit["coherence_seconds"]) > 0:
        add(_LIFETIME_LABEL.get(node_id, "T2"), f"{float(fit['coherence_seconds']) * 1e6:.4g} µs")
    if _finite(fit.get("readout_fidelity")):
        fidelity = float(fit["readout_fidelity"])
        add("readout fidelity", f"{fidelity * 100 if fidelity <= 1 else fidelity:.1f}%")
    if _finite(fit.get("r_squared")):
        add("R²", f"{float(fit['r_squared']):.3f}")
    summaries = (analysis.get("dataset_metrics") or {}).get("protected_node_summaries") or {}
    summary = summaries.get(qubit) if isinstance(summaries, dict) else None
    if isinstance(summary, dict) and _finite(summary.get("mu_us")):
        sigma = summary.get("sigma_us")
        add(
            _STATISTICS_LABEL.get(node_id, "mean"),
            f"{float(summary['mu_us']):.4g} ± {float(sigma):.3g} µs"
            if _finite(sigma)
            else f"{float(summary['mu_us']):.4g} µs",
        )
    if fit.get("summary_robust") is False:
        # Operator choice 2026-10-13: status kept, but the mean is flagged.
        entry = fit.get("robust_summary") or {}
        note = "mean not robust (σ > μ; failed single fits dominate)"
        if all(_finite(entry.get(key)) for key in ("median_us", "q25_us", "q75_us")):
            note += (
                f"; median {float(entry['median_us']):.4g} µs, IQR "
                f"{float(entry['q25_us']):.4g}–{float(entry['q75_us']):.4g} µs "
                f"({entry.get('fits')}/{entry.get('iterations')} fits)"
            )
        add("⚠ note", note)
    for item in analysis.get("candidate_state_patch") or []:
        path = str(item.get("path", ""))
        parts = path.strip("/").split("/")
        if len(parts) < 3 or parts[0] != "qubits" or parts[1] != qubit:
            continue
        key = parts[-1]
        if key in _HIDDEN_PATCH_KEYS:
            continue
        label = _PATCH_LABELS.get(key) or (
            "/".join(parts[-2:]) if parts[-2] not in {qubit, "extras"} else key
        )
        add(label, _format_value(key, item.get("value")))
    return values


def _run_qubits(row: dict[str, Any]) -> list[str]:
    names = json_loads(row.get("parameters_json"), {}).get("qubits", [])
    return [str(name) for name in names] if isinstance(names, list) else []


def collect_chip_history(service: Any, device_key: str) -> dict[str, Any]:
    """Nodes in sequence order -> qubits -> successful results, newest first."""

    rows = service.db.all(
        "SELECT r.id, r.workflow_id, r.node_id, r.parameters_json, r.analysis_json, "
        "r.finished_at, r.started_at, r.rowid AS run_rowid FROM runs r "
        "JOIN workflows w ON w.id = r.workflow_id "
        "WHERE w.device_key = ? AND r.status = 'completed' "
        "AND r.inherited_from_run_id IS NULL "
        "AND r.analysis_status IN ('pass', 'needs_review', 'failed') "
        "ORDER BY r.rowid DESC",
        (device_key,),
    )
    nodes: dict[str, dict[str, Any]] = {}
    for row in rows:
        analysis = json_loads(row.get("analysis_json"), {})
        passing = [str(name) for name in analysis.get("passing_targets") or []]
        if not passing:
            continue
        node_id = str(row["node_id"])
        run_qubits = _run_qubits(row)
        plots = [str(path) for path in analysis.get("plots") or []]
        own: dict[str, list[int]] = {}
        shared: list[int] = []
        for position, path in enumerate(plots):
            named = set(_QUBIT_IN_NAME.findall(Path(path).name))
            if len(run_qubits) == 1:
                own.setdefault(run_qubits[0], []).append(position)
            elif named:
                for name in named:
                    own.setdefault(name, []).append(position)
            else:
                shared.append(position)
        node = nodes.setdefault(node_id, {"node": node_id, "qubits": {}, "shared": []})
        if shared:
            node["shared"].append(
                {
                    "run_id": row["id"],
                    "snapshot_id": analysis.get("snapshot_id"),
                    "finished_at": row.get("finished_at"),
                    "qubits": sorted(passing, key=_natural),
                    "plots": [(position, plots[position]) for position in shared],
                }
            )
        for qubit in passing:
            node["qubits"].setdefault(qubit, []).append(
                {
                    "run_id": row["id"],
                    "snapshot_id": analysis.get("snapshot_id"),
                    "finished_at": row.get("finished_at"),
                    "values": _key_values(node_id, qubit, analysis),
                    "plots": [(position, plots[position]) for position in own.get(qubit, [])],
                }
            )
    sequence = list(service.settings.workflow_sequence)
    order = sorted(
        nodes,
        key=lambda name: (sequence.index(name), "") if name in sequence else (len(sequence), name),
    )
    return {
        "device_key": device_key,
        "chip": Path(str(device_key)).name or str(device_key),
        "nodes": [
            {
                "node": name,
                "shared": nodes[name]["shared"],
                "qubits": [
                    {"qubit": qubit, "results": nodes[name]["qubits"][qubit]}
                    for qubit in sorted(nodes[name]["qubits"], key=_natural)
                ],
            }
            for name in order
        ],
    }


def chip_history_asset(service: Any, device_key: str, run_id: str, position: int) -> Path:
    """One plot of a successful run on this chip; refuses anything else."""

    row = service.db.one(
        "SELECT r.analysis_json FROM runs r JOIN workflows w ON w.id = r.workflow_id "
        "WHERE r.id = ? AND w.device_key = ?",
        (run_id, device_key),
    )
    if row is None:
        raise ValueError("Unknown chip history run")
    plots = json_loads(row.get("analysis_json"), {}).get("plots") or []
    if isinstance(position, bool) or not 0 <= int(position) < len(plots):
        raise ValueError("Unknown chip history plot")
    path = Path(str(plots[int(position)])).resolve()
    if not service._is_allowed_review_asset(path) or not path.is_file():
        raise ValueError("Chip history plot is unavailable")
    return path


def render_chip_history(
    history: dict[str, Any],
    image_url: Callable[[str, int, bool], str],
    language: str = "zh-Hant",
) -> str:
    """The chip history card; `image_url(run_id, position, thumb)` places images."""

    def ui(zh: str, en: str) -> str:
        return en if language == "en" else zh

    def figure(run_id: str, position: int, path: str) -> str:
        full = html.escape(image_url(run_id, position, False))
        thumb = html.escape(image_url(run_id, position, True))
        return (
            f'<figure><a href="{full}" target="_blank" rel="noopener">'
            f'<img loading="lazy" src="{thumb}" alt="{html.escape(Path(path).name)}"></a>'
            f"<figcaption class=\"muted\">{html.escape(Path(path).name)}</figcaption></figure>"
        )

    def entry(result: dict[str, Any]) -> str:
        values = "".join(
            f"<li><span class=\"muted\">{html.escape(label)}</span> {html.escape(text)}</li>"
            for label, text in result["values"]
        )
        meta = ui("快照", "Snapshot") + f" #{html.escape(str(result.get('snapshot_id') or '?'))}"
        when = str(result.get("finished_at") or "")[:16].replace("T", " ")
        figures = "".join(figure(result["run_id"], pos, path) for pos, path in result["plots"])
        return (
            f'<div class="chip-entry"><p class="muted">{meta} · {html.escape(when)} UTC</p>'
            + (f'<ul class="chip-values">{values}</ul>' if values else "")
            + (f'<div class="result-grid">{figures}</div>' if figures else "")
            + "</div>"
        )

    heading = ui("晶片歷史結果", "Chip history")
    intro = ui(
        f"{html.escape(history['chip'])} 所有 workflow 中成功的實驗，依實驗順序與 qubit 排列；每項顯示最新一次，較早的成功結果收在下方。",
        f"Every successful result on {html.escape(history['chip'])} across its workflows, by node and qubit; the newest leads, earlier ones fold underneath.",
    )
    if not history["nodes"]:
        return (
            f'<section class="card"><h2>{heading}</h2><p>{intro}</p>'
            f'<p>{ui("此晶片尚無成功的實驗。", "No successful result on this chip yet.")}</p></section>'
        )
    blocks = []
    for node in history["nodes"]:
        qubit_names = ", ".join(item["qubit"] for item in node["qubits"])
        shared = ""
        if node["shared"]:
            latest = node["shared"][0]
            shared = (
                f'<div class="chip-shared"><p class="muted">{ui("多 qubit 圖", "Multi-qubit plot")}: '
                f'{html.escape(", ".join(latest["qubits"]))} · '
                f'{ui("快照", "Snapshot")} #{html.escape(str(latest.get("snapshot_id") or "?"))}</p>'
                f'<div class="result-grid">'
                + "".join(figure(latest["run_id"], pos, path) for pos, path in latest["plots"])
                + "</div></div>"
            )
        rows = []
        for item in node["qubits"]:
            newest, earlier = item["results"][0], item["results"][1:]
            older = (
                f'<details><summary>{ui("較早的成功結果", "Earlier successes")} ({len(earlier)})</summary>'
                + "".join(entry(result) for result in earlier)
                + "</details>"
                if earlier
                else ""
            )
            rows.append(
                f'<div class="chip-qubit"><h4>{html.escape(item["qubit"])}</h4>{entry(newest)}{older}</div>'
            )
        blocks.append(
            f'<details class="chip-node" open><summary><strong>{html.escape(node["node"])}</strong> '
            f'<span class="muted">{html.escape(qubit_names)}</span></summary>'
            f"{shared}{''.join(rows)}</details>"
        )
    return (
        f'<section class="card chip-history"><h2>{heading}</h2><p class="muted">{intro}</p>'
        + "".join(blocks)
        + "</section>"
    )


CHIP_HISTORY_STYLE = """
.chip-node { border-top:1px solid #21304d; padding:10px 0; }
.chip-node > summary { cursor:pointer; font-size:1.05rem; }
.chip-qubit { margin:10px 0 14px 14px; }
.chip-qubit h4 { margin:6px 0; }
.chip-values { list-style:none; padding:0; margin:4px 0 8px; display:flex; flex-wrap:wrap; gap:4px 18px; }
.chip-entry details, .chip-qubit details { margin-left:8px; }
figcaption { font-size:.75rem; overflow-wrap:anywhere; }
"""


def write_static_chip_history(service: Any, device_key: str) -> Path:
    """Write the chip history as a self-contained folder that survives shutdown."""

    history = collect_chip_history(service, device_key)
    folder = service.settings.runtime / "chip_history" / history["chip"]
    images = folder / "images"
    images.mkdir(parents=True, exist_ok=True)
    names: dict[tuple[str, int], str] = {}
    for node in history["nodes"]:
        groups = [node["shared"][0]] if node["shared"] else []
        groups += [result for item in node["qubits"] for result in item["results"]]
        for result in groups:
            for position, path in result["plots"]:
                source = Path(path)
                if not service._is_allowed_review_asset(source) or not source.is_file():
                    continue
                name = f"{result['run_id']}_{position}{source.suffix.lower()}"
                target = images / name
                if not target.is_file():
                    shutil.copy2(source, target)
                names[(result["run_id"], position)] = f"images/{name}"

    def image_url(run_id: str, position: int, _thumb: bool) -> str:
        return names.get((run_id, position), "")

    body = render_chip_history(history, image_url, "zh-Hant")
    page = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(history['chip'])} chip history</title>
<style>
:root {{ color-scheme:dark; font-family:system-ui,-apple-system,sans-serif; }}
body {{ margin:0; background:#08101d; color:#edf4ff; line-height:1.55; }}
main {{ max-width:1040px; margin:24px auto; padding:0 16px 52px; }}
.card {{ background:#121d31; border:1px solid #2b3b5d; border-radius:14px; padding:18px; margin:16px 0; }}
.muted {{ color:#9fb2d7; }}
.result-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr)); gap:14px; }}
figure {{ margin:0; padding:10px; background:#070d18; border-radius:10px; }}
img {{ display:block; width:100%; height:auto; border-radius:7px; }}
{CHIP_HISTORY_STYLE}
</style></head><body><main>{body}</main></body></html>"""
    target = folder / "index.html"
    temporary = folder / "index.html.tmp"
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(target)
    return target
