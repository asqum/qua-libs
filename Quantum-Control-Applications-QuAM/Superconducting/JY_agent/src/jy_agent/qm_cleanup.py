"""Close this chip's stale quantum machine, ported from `00x_close_specific_qm`.

Operator request 2026-10-20. When a run breaks off (a crash, a lost
connection), the quantum machine it opened can stay open on the QOP with its
job still running, and the next run then fails to open the same ports. The
notebook `calibration_graph/00x_close_specific_qm.ipynb` finds the open
machine whose ports are this chip's and closes it; this module does the same
without a notebook so JY can do it before a retry.

Only a machine whose every port belongs to the active `state.json` is ours
and is closed (its running job is halted first). A machine that merely
shares some ports is reported, never closed: it may be someone else's.

Run it as ``python -m jy_agent.qm_cleanup --state STATE --wiring WIRING`` (add
``--dry-run`` to only list); it prints one JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from .instrument_probe import _coerce_port, read_network_settings


Port = tuple[str, int, int]


def state_ports(state: dict[str, Any]) -> set[Port]:
    """Every (controller, FEM, port) that the chip's `state.json` declares."""

    found: set[Port] = set()
    containers = state.get("ports") if isinstance(state, dict) else None
    if not isinstance(containers, dict):
        return found
    for controllers in containers.values():
        if not isinstance(controllers, dict):
            continue
        for fems in controllers.values():
            if not isinstance(fems, dict):
                continue
            for ports in fems.values():
                if not isinstance(ports, dict):
                    continue
                for details in ports.values():
                    port = _port_from_details(details)
                    if port is not None:
                        found.add(port)
    return found


def config_ports(config: dict[str, Any]) -> set[Port]:
    """Every (controller, FEM, port) an open machine's elements use."""

    found: set[Port] = set()
    elements = config.get("elements") if isinstance(config, dict) else None
    if isinstance(elements, dict):
        for element in elements.values():
            found |= set(_walk_ports(element))
    return found


def classify(machine_ports: set[Port], chip_ports: set[Port]) -> str:
    """`ours` (all its ports are the chip's), `overlapping` or `unrelated`."""

    if machine_ports and chip_ports and machine_ports <= chip_ports:
        return "ours"
    if machine_ports & chip_ports:
        return "overlapping"
    return "unrelated"


def _port_from_details(details: Any) -> Port | None:
    if not isinstance(details, dict):
        return None
    controller = details.get("controller_id")
    fem = details.get("fem_id")
    port = details.get("port_id")
    if not controller or fem is None or port is None:
        return None
    try:
        return (str(controller), int(fem), int(port))
    except (TypeError, ValueError):
        return None


def _walk_ports(value: Any) -> Iterable[Port]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "port" and isinstance(item, (list, tuple)) and len(item) == 3:
                try:
                    yield (str(item[0]), int(item[1]), int(item[2]))
                except (TypeError, ValueError):
                    pass
            else:
                yield from _walk_ports(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_ports(item)


def _is_job_handle(job: Any) -> bool:
    return callable(getattr(job, "cancel", None)) or callable(getattr(job, "halt", None))


def _halt_running_jobs(machine: Any) -> list[str]:
    """Cancel the machine's running or queued job(s); return what was done.

    qm-qua 1.4 on a QOP 3 (`QmApi`) lists jobs as `JobData` records, which
    carry only an id; `get_job(id)` gives the handle whose `cancel()` stops
    it. The older `QuantumMachine` offers `get_running_job()`, whose job is
    stopped with `halt()` (renamed `cancel()` in 1.2).
    """

    actions: list[str] = []
    jobs: list[Any] = []
    get_jobs = getattr(machine, "get_jobs", None)
    if callable(get_jobs):
        try:
            records = list(get_jobs(status=["Running", "In queue"]))
        except Exception:
            records = []
        get_job = getattr(machine, "get_job", None)
        for record in records:
            if _is_job_handle(record):
                jobs.append(record)
                continue
            job_id = getattr(record, "id", None)
            if not job_id or not callable(get_job):
                continue
            try:
                jobs.append(get_job(job_id))
            except Exception as exc:
                actions.append(f"get_job {job_id} failed: {type(exc).__name__}: {exc}")
    if not jobs:
        get_running = getattr(machine, "get_running_job", None)
        if callable(get_running):
            try:
                job = get_running()
            except Exception:
                job = None
            if job is not None:
                jobs = [job]
    for job in jobs:
        job_id = str(getattr(job, "id", "") or "running job")
        for method in ("cancel", "halt"):
            stop = getattr(job, method, None)
            if not callable(stop):
                continue
            try:
                stop()
            except Exception as exc:
                actions.append(f"{method} {job_id} failed: {type(exc).__name__}: {exc}")
                continue
            actions.append(f"{method} {job_id}")
            break
    return actions


def close_stale_qms(
    state_path: Path | str,
    wiring_path: Path | str,
    *,
    close: bool = True,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """List the open machines and close the ones that are this chip's."""

    try:
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"Could not read state.json: {exc}", "machines": []}
    chip = state_ports(state)
    if not chip:
        return {"ok": False, "error": "state.json declares no ports", "machines": []}
    network = read_network_settings(wiring_path)
    host = str(network.get("host") or "").strip()
    if not host:
        return {"ok": False, "error": "wiring.json names no QOP host", "machines": []}
    settings: dict[str, Any] = {"host": host, "timeout": timeout}
    if network.get("port") is not None:
        settings["port"] = _coerce_port(network.get("port"))
    if network.get("cluster_name"):
        settings["cluster_name"] = network["cluster_name"]
    try:
        from qm import QuantumMachinesManager

        manager = QuantumMachinesManager(**settings)
        machine_ids = list(manager.list_open_qms())
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Could not reach the QOP: {type(exc).__name__}: {exc}",
            "machines": [],
        }
    machines: list[dict[str, Any]] = []
    for machine_id in machine_ids:
        entry: dict[str, Any] = {"qm_id": str(machine_id)}
        try:
            machine = manager.get_qm(machine_id)
            ports = config_ports(machine.get_config())
        except Exception as exc:
            entry.update(verdict="unreadable", error=f"{type(exc).__name__}: {exc}")
            machines.append(entry)
            continue
        entry["verdict"] = classify(ports, chip)
        entry["ports"] = sorted(ports)
        if entry["verdict"] == "ours" and close:
            entry["jobs"] = _halt_running_jobs(machine)
            try:
                machine.close()
                entry["closed"] = True
            except Exception as exc:
                entry["closed"] = False
                entry["error"] = f"{type(exc).__name__}: {exc}"
        machines.append(entry)
    return {
        "ok": all(item.get("closed", True) for item in machines),
        "closed": [item["qm_id"] for item in machines if item.get("closed")],
        "overlapping_not_closed": [
            item["qm_id"] for item in machines if item.get("verdict") == "overlapping"
        ],
        "machines": machines,
        "dry_run": not close,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", required=True)
    parser.add_argument("--wiring", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    report = close_stale_qms(
        args.state, args.wiring, close=not args.dry_run, timeout=args.timeout
    )
    sys.stdout.write(json.dumps(report, ensure_ascii=False) + "\n")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
