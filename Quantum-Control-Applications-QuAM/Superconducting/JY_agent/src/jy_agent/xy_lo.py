"""Drive-LO bookkeeping for qubits that share one XY output.

Operator instruction 2026-10-12. On a chip whose qubits share an XY output the
port LO is one knob for all of them, so each qubit records the LO it uses in
`extras.xy_lo_freq` and later nodes reuse it:

* when one LO on the 100 MHz grid keeps every measured qubit within
  `common_max_if_hz` (room for a +-100 MHz scan inside the 400 MHz IF limit),
  all of them share it, preferring IFs at least `min_if_hz` from the LO
  leakage;
* otherwise the qubit whose pass is running uses its own LO, and a qubit the
  LO cannot reach is parked at the IF limit -- it is not played -- while its
  real frequency stays in `extras.idle_freq`.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Collection

from .state import StateError, pointer_get

_OUTPUT = re.compile(r"#/ports/mw_outputs/([^/]+)/([0-9]+)/([0-9]+)")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def xy_output(state: dict[str, Any], wiring: dict[str, Any], name: str) -> str | None:
    """The MW output reference a qubit's XY channel is wired to."""

    wiring_qubits = wiring.get("wiring", {}).get("qubits", {})
    reference = (
        wiring_qubits.get(name, {}).get("xy", {}).get("opx_output")
        if isinstance(wiring_qubits, dict)
        else None
    )
    if not isinstance(reference, str):
        direct = state.get("qubits", {}).get(name, {}).get("xy", {}).get("opx_output")
        reference = direct if isinstance(direct, str) else None
    return reference if reference and _OUTPUT.fullmatch(reference) else None


def output_members(
    state: dict[str, Any], wiring: dict[str, Any], reference: str
) -> list[str]:
    names = [
        name
        for name in state.get("qubits", {})
        if xy_output(state, wiring, name) == reference
    ]
    return sorted(names, key=_natural)


def _natural(name: str) -> tuple[Any, ...]:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(name)))


def recorded_lo(state: dict[str, Any], name: str) -> float | None:
    extras = state.get("qubits", {}).get(name, {}).get("extras", {}) or {}
    return _number(extras.get("xy_lo_freq"))


def intended_rf(state: dict[str, Any], name: str, lo_hz: float) -> float | None:
    """A qubit's real drive frequency: `extras.idle_freq`, else LO + IF."""

    qubit = state.get("qubits", {}).get(name, {})
    idle = _number((qubit.get("extras") or {}).get("idle_freq"))
    if idle is not None:
        return idle
    intermediate = _number((qubit.get("xy") or {}).get("intermediate_frequency"))
    return None if intermediate is None else lo_hz + intermediate


def xy_lo_plan_patch(
    state: dict[str, Any],
    wiring: dict[str, Any],
    focus: str,
    measured: Collection[str],
    *,
    grid_hz: float,
    max_if_hz: float,
    common_max_if_hz: float,
    min_if_hz: float,
    band_window: Callable[[Any], tuple[float, float] | None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The LO and IFs the shared output needs before `focus` is measured."""

    reference = xy_output(state, wiring, focus)
    if reference is None:
        raise StateError(f"{focus} has no resolvable MW XY output")
    port_path = reference[1:]
    lo_path = f"{port_path}/upconverter_frequency"
    current_lo = _number(pointer_get(state, lo_path))
    if current_lo is None:
        raise StateError(f"{focus} XY LO is not numeric")
    members = output_members(state, wiring, reference)
    rf = {
        name: value
        for name in members
        if (value := intended_rf(state, name, current_lo)) is not None
    }
    if focus not in rf:
        raise StateError(f"{focus} has no idle frequency to place its LO around")
    known = sorted(
        (set(measured) | {focus}) & set(rf) & set(members), key=_natural
    )
    window = None
    if band_window is not None:
        try:
            band = pointer_get(state, f"{port_path}/band")
        except (KeyError, IndexError, StateError):
            band = None
        window = band_window(band)

    def on_grid(low: float, high: float) -> list[float]:
        first = math.ceil(low / grid_hz)
        last = math.floor(high / grid_hz)
        values = [index * grid_hz for index in range(first, last + 1)]
        if window is not None:
            values = [value for value in values if window[0] <= value <= window[1]]
        return values

    def clear_of_leakage(lo: float, names: Collection[str]) -> bool:
        return all(abs(rf[name] - lo) >= min_if_hz for name in names)

    def worst_if(lo: float, names: Collection[str]) -> float:
        return max(abs(rf[name] - lo) for name in names)

    common = [
        lo
        for lo in on_grid(
            min(rf[name] for name in known) - common_max_if_hz,
            max(rf[name] for name in known) + common_max_if_hz,
        )
        if worst_if(lo, known) <= common_max_if_hz
    ]
    own = recorded_lo(state, focus)
    if common:
        pool = [lo for lo in common if clear_of_leakage(lo, known)] or common
        if any(math.isclose(current_lo, lo) for lo in pool):
            chosen = current_lo
        else:
            chosen = min(pool, key=lambda lo: (worst_if(lo, known), abs(lo - current_lo)))
        mode = "common"
        recorded_for = known
    else:
        focus_pool = on_grid(rf[focus] - common_max_if_hz, rf[focus] + common_max_if_hz)
        if own is not None and abs(rf[focus] - own) <= common_max_if_hz:
            chosen = own
        elif focus_pool:
            clear = [lo for lo in focus_pool if clear_of_leakage(lo, [focus])]
            chosen = min(clear or focus_pool, key=lambda lo: abs(rf[focus] - lo))
        else:
            raise StateError(f"No usable XY LO near {rf[focus] / 1e9:g} GHz for {focus}")
        mode = "own"
        recorded_for = [focus]

    patch: list[dict[str, Any]] = []
    if not math.isclose(chosen, current_lo, abs_tol=0.5):
        patch.append({"op": "replace", "path": lo_path, "value": chosen})
    intermediate: dict[str, float] = {}
    parked: list[str] = []
    for name in members:
        if name not in rf:
            continue
        value = rf[name] - chosen
        if abs(value) > max_if_hz:
            value = math.copysign(max_if_hz, value)
            parked.append(name)
        intermediate[name] = value
        path = f"/qubits/{name}/xy/intermediate_frequency"
        old = _number(state["qubits"][name].get("xy", {}).get("intermediate_frequency"))
        if old is None or not math.isclose(old, value, abs_tol=0.5):
            patch.append({"op": "replace", "path": path, "value": value})
    for name in recorded_for:
        old = recorded_lo(state, name)
        if old is None or not math.isclose(old, chosen, abs_tol=0.5):
            extras = state["qubits"][name].get("extras")
            patch.append(
                {
                    "op": "add" if not isinstance(extras, dict) or "xy_lo_freq" not in extras else "replace",
                    "path": f"/qubits/{name}/extras/xy_lo_freq",
                    "value": chosen,
                }
            )
    details = {
        "focus": focus,
        "xy_output": reference,
        "mode": mode,
        "lo_hz": chosen,
        "previous_lo_hz": current_lo,
        "measured": known,
        "recorded_for": list(recorded_for),
        "intermediate_frequency_hz": intermediate,
        "parked": parked,
        "focus_rf_hz": rf[focus],
    }
    return patch, details
