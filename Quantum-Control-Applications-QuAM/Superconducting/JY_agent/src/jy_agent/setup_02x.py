"""Deterministic setup before 02x (operator debug request 2026-10-11).

Two state problems made CRSQ_26009_die(5,-1)'s 02x and 02c unreadable:

* `02x_Bare_Resonator_Spectroscopy.py` probes every resonator at a fixed
  -10 dBm (0.1 V) and plays all of a line's tones at once. `set_output_power`
  only raises the port's full scale, so at a -11 dBm full scale each tone
  needs amplitude ~0.79; six tones summed to ~4.7 and the output clipped, so
  every qubit's trace showed the same compressed signal (run #385). Raising
  the line's full scale until the summed probe fits keeps 02x linear; each
  qubit's readout amplitude is scaled so its delivered power is unchanged.
* Several qubits shared one `grid_location`, so every node's grid plot drew
  them on top of each other and only q6 and q4 were visible.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .state_patch_02c import _readout_line, _readout_operation

# 02x's hard-coded probe: u.volts2dBm(0.1) with 50 ohm.
PROBE_02X_DBM = -10.0
# MW-FEM full scale moves in 3 dB steps from -41 dBm; +10 dBm is the maximum
# `set_output_power` accepts.
FULL_SCALE_GRID_DBM = list(range(-41, 11, 3))


def _probe_amplitude(full_scale_dbm: float, probe_dbm: float = PROBE_02X_DBM) -> float:
    """The amplitude 02x's `set_output_power` ends at from this full scale."""
    scale = float(full_scale_dbm)
    while 10.0 ** ((probe_dbm - scale) / 20.0) > 1.0:
        scale += 3.0
    return 10.0 ** ((probe_dbm - scale) / 20.0)


def probe_02x_headroom_patch(
    state: dict[str, Any],
    wiring: dict[str, Any],
    targets: list[str],
    *,
    max_line_sum: float = 1.0,
    min_full_scale_dbm: float = -11.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Raise each readout line's full scale so 02x's summed probe fits."""

    lines: dict[str, dict[str, Any]] = {}
    for name in sorted(targets):
        try:
            line = _readout_line(state, wiring, name)
        except Exception:
            continue
        if line.get("path") is None:
            continue
        entry = lines.setdefault(line["key"], {**line, "probed": []})
        entry["probed"].append(name)
    patch: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for key, line in sorted(lines.items()):
        old = float(line["full_scale"])
        count = len(line["probed"])
        if count * _probe_amplitude(old) <= max_line_sum:
            continue
        candidates = [
            value
            for value in FULL_SCALE_GRID_DBM
            if value >= max(old, min_full_scale_dbm)
        ]
        if not candidates:
            continue
        chosen = next(
            (
                value
                for value in candidates
                if count * _probe_amplitude(value) <= max_line_sum
            ),
            candidates[-1],
        )
        if float(chosen) == old:
            continue
        patch.append({"op": "replace", "path": line["path"], "value": int(chosen)})
        factor = 10.0 ** ((old - chosen) / 20.0)
        amplitudes: dict[str, list[float]] = {}
        for name in line["members"]:
            operations = state["qubits"][name]["resonator"]["operations"]
            op = "readout"
            if isinstance(operations.get(op), str) and operations[op].startswith("#./"):
                op = operations[op][3:]
            readout = _readout_operation(state["qubits"][name])
            amplitude = readout.get("amplitude") if isinstance(readout, dict) else None
            if not isinstance(amplitude, (int, float)) or not math.isfinite(amplitude):
                continue
            new = float(amplitude) * factor
            patch.append(
                {
                    "op": "replace",
                    "path": f"/qubits/{name}/resonator/operations/{op}/amplitude",
                    "value": new,
                }
            )
            amplitudes[name] = [float(amplitude), new]
        details[key] = {
            "old_full_scale_dbm": old,
            "new_full_scale_dbm": int(chosen),
            "probed_qubits": line["probed"],
            "summed_probe_amplitude": [
                count * _probe_amplitude(old),
                count * _probe_amplitude(chosen),
            ],
            "readout_amplitudes": amplitudes,
        }
    return patch, details


def unique_thread_patch(
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One QUA thread per qubit when several readout elements share one.

    Operator decision 2026-10-11: on CRSQ_26009_die(5,-1) and (-1,3) every
    resonator used thread "a" (most XY elements "b"). Elements on one thread
    run on one core, and multiplexed 02x then reported the last element's
    response (q6) on every channel. Each qubit gets its own thread, shared by
    its resonator and XY element, as on as-qpu-10qV2.
    """

    qubits = state.get("qubits", {})
    threads = [
        (qubit.get("resonator") or {}).get("thread")
        for qubit in qubits.values()
        if isinstance(qubit, dict)
    ]
    used = [value for value in threads if value is not None]
    if len(used) == len(set(used)):
        return [], {}
    patch: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for index, name in enumerate(sorted(qubits, key=_natural)):
        thread = chr(ord("a") + index) if index < 26 else f"t{index}"
        changed: dict[str, list[Any]] = {}
        for element in ("resonator", "xy"):
            current = (qubits[name].get(element) or {})
            if not isinstance(current, dict) or not current:
                continue
            if current.get("thread") != thread:
                patch.append(
                    {
                        "op": "add" if current.get("thread") is None else "replace",
                        "path": f"/qubits/{name}/{element}/thread",
                        "value": thread,
                    }
                )
                changed[element] = [current.get("thread"), thread]
        if changed:
            details[name] = changed
    return patch, details


def _natural(name: str) -> tuple[Any, ...]:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(name)))


def unique_grid_location_patch(
    state: dict[str, Any], columns: int = 4
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Give every qubit its own plot cell when some share one."""

    qubits = state.get("qubits", {})
    locations = {name: qubit.get("grid_location") for name, qubit in qubits.items()}
    used = [value for value in locations.values() if value is not None]
    if len(used) == len(set(used)):
        return [], {}
    patch: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for index, name in enumerate(sorted(qubits, key=_natural)):
        value = f"{index % columns},{index // columns}"
        if locations.get(name) != value:
            patch.append(
                {
                    "op": "add" if locations.get(name) is None else "replace",
                    "path": f"/qubits/{name}/grid_location",
                    "value": value,
                }
            )
            details[name] = [locations.get(name), value]
    return patch, details
