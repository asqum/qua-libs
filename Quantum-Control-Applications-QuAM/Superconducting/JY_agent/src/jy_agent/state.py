from __future__ import annotations

import copy
import json
import math
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Iterable

from .util import atomic_write_json, exclusive_file_lock, sha256_file


class StateError(ValueError):
    pass


def load_state(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise StateError(f"State root must be an object: {path}")
    return value


def pointer_parts(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise StateError(f"Invalid JSON pointer: {pointer}")
    if pointer == "/":
        return [""]
    return [
        part.replace("~1", "/").replace("~0", "~")
        for part in pointer[1:].split("/")
    ]


def pointer_get(document: Any, pointer: str) -> Any:
    current = document
    for part in pointer_parts(pointer):
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise StateError(f"Cannot traverse {pointer}")
    return current


def _pointer_parent(document: Any, pointer: str) -> tuple[Any, str]:
    parts = pointer_parts(pointer)
    if not parts:
        raise StateError("Cannot modify the document root")
    current = document
    for part in parts[:-1]:
        if isinstance(current, dict):
            if part not in current:
                current[part] = {}
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise StateError(f"Cannot traverse {pointer}")
    return current, parts[-1]


def apply_json_patch(
    document: dict[str, Any], patch: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    result = copy.deepcopy(document)
    for item in patch:
        operation = item.get("op")
        pointer = item.get("path")
        if operation not in {"add", "replace"} or not isinstance(pointer, str):
            raise StateError(f"Unsupported patch item: {item}")
        parent, key = _pointer_parent(result, pointer)
        if isinstance(parent, dict):
            if operation == "replace" and key not in parent:
                raise StateError(f"replace target does not exist: {pointer}")
            parent[key] = copy.deepcopy(item.get("value"))
        elif isinstance(parent, list):
            index = int(key)
            if operation == "replace":
                parent[index] = copy.deepcopy(item.get("value"))
            else:
                parent.insert(index, copy.deepcopy(item.get("value")))
        else:
            raise StateError(f"Cannot modify {pointer}")
    return result


def json_diff(before: Any, after: Any, pointer: str = "") -> list[dict[str, Any]]:
    """Return scalar add/replace operations; list structure changes are not proposed."""
    changes: list[dict[str, Any]] = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(after.keys() | before.keys()):
            escaped = key.replace("~", "~0").replace("/", "~1")
            child = f"{pointer}/{escaped}"
            if key not in before:
                changes.append({"op": "add", "path": child, "value": after[key]})
            elif key not in after:
                continue
            else:
                changes.extend(json_diff(before[key], after[key], child))
        return changes
    if isinstance(before, list) or isinstance(after, list):
        return changes if before == after else [
            {"op": "replace", "path": pointer, "value": after}
        ]
    if before != after:
        changes.append({"op": "replace", "path": pointer, "value": after})
    return changes


def recorded_updates_to_patch(
    updates: dict[str, Any], base_state: dict[str, Any]
) -> list[dict[str, Any]]:
    """Convert Qualibrate's recorded QuAM updates into JSON Patch operations."""
    if not isinstance(updates, dict):
        raise StateError("Recorded state updates must be an object")
    patch: list[dict[str, Any]] = []
    for reference in sorted(updates):
        item = updates[reference]
        if not isinstance(item, dict):
            raise StateError(f"Invalid recorded state update at {reference!r}")
        key = item.get("key", reference)
        if not isinstance(key, str):
            raise StateError(f"Recorded state update has no reference at {reference!r}")
        if key.startswith("#/"):
            pointer = key[1:]
        elif key.startswith("/"):
            pointer = key
        else:
            raise StateError(f"Unsupported recorded state reference: {key!r}")
        try:
            pointer_get(base_state, pointer)
        except (KeyError, IndexError, StateError):
            operation = "add"
        else:
            operation = "replace"
        patch.append(
            {"op": operation, "path": pointer, "value": copy.deepcopy(item.get("new"))}
        )
    return patch


def filter_patch(
    patch: Iterable[dict[str, Any]], allowed_patterns: Iterable[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    patterns = [re.compile(pattern) for pattern in allowed_patterns]
    allowed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in patch:
        destination = str(item.get("path", ""))
        (allowed if any(p.fullmatch(destination) for p in patterns) else rejected).append(item)
    return allowed, rejected


_QUBIT_POINTER = re.compile(r"^/qubits/(q[0-9]+)/")


def patch_qubit_targets(patch: Iterable[dict[str, Any]]) -> set[str]:
    """Return the qubit names a calibration state patch writes to."""

    return {
        match.group(1)
        for item in patch
        if (match := _QUBIT_POINTER.match(str(item.get("path", ""))))
    }


def split_patch_by_target(
    patch: Iterable[dict[str, Any]], allowed: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a patch into the entries owned by an allowed qubit and the rest.

    A pointer that names no qubit is never "allowed": per-target evidence can
    only authorize a per-target change.
    """

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for item in patch:
        match = _QUBIT_POINTER.match(str(item.get("path", "")))
        (kept if match is not None and match.group(1) in allowed else dropped).append(
            item
        )
    return kept, dropped


def operation_backing_name(qubit: dict[str, Any], operation: str) -> str:
    operations = qubit["xy"]["operations"]
    entry = operations.get(operation)
    if isinstance(entry, str) and entry.startswith("#./"):
        return entry[3:]
    if isinstance(entry, dict):
        return operation
    raise StateError(f"Cannot resolve XY operation {operation!r}")


def operation_amplitude(qubit: dict[str, Any], operation: str) -> float:
    backing = operation_backing_name(qubit, operation)
    value = qubit["xy"]["operations"][backing].get("amplitude")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise StateError(f"{operation} amplitude is not numeric")
    return float(value)


def xy_drive_groups(wiring: dict[str, Any]) -> dict[str, list[str]]:
    """Qubit names grouped by the XY drive output each one is wired to."""

    qubits = wiring.get("wiring", wiring).get("qubits", {})
    if not isinstance(qubits, dict):
        return {}
    groups: dict[str, list[str]] = {}
    for name, item in qubits.items():
        if not isinstance(item, dict):
            continue
        xy = item.get("xy")
        output = xy.get("opx_output") if isinstance(xy, dict) else None
        if not isinstance(output, str):
            continue
        groups.setdefault(output, []).append(str(name))
    return {output: sorted(names) for output, names in groups.items()}


def shared_xy_drive_groups(wiring: dict[str, Any]) -> dict[str, list[str]]:
    """Only the XY drive outputs that more than one qubit is wired to.

    A chip with a single drive line reaching several qubits cannot address
    them independently, so every node that uses the drive line has to measure
    one qubit per run. See `PolicyEngine.shared_xy_drive_cap`.
    """

    return {
        output: names
        for output, names in xy_drive_groups(wiring).items()
        if len(names) > 1
    }


def z_line_groups(wiring: dict[str, Any]) -> dict[str, list[str]]:
    """Qubit names grouped by the Z (flux) output each one is wired to."""

    qubits = wiring.get("wiring", wiring).get("qubits", {})
    if not isinstance(qubits, dict):
        return {}
    groups: dict[str, list[str]] = {}
    for name, item in qubits.items():
        if not isinstance(item, dict):
            continue
        z = item.get("z")
        output = z.get("opx_output") if isinstance(z, dict) else None
        if not isinstance(output, str):
            continue
        groups.setdefault(output, []).append(str(name))
    return {output: sorted(names) for output, names in groups.items()}


def shared_z_line_groups(wiring: dict[str, Any]) -> dict[str, list[str]]:
    """Only the Z outputs that more than one qubit is wired to.

    Operator request 2026-10-08: a shared flux line cannot move one qubit
    without moving the others, so any such output on the chip disables flux
    calibration on the lease page.
    """

    return {
        output: names
        for output, names in z_line_groups(wiring).items()
        if len(names) > 1
    }


def z_offset(qubit: dict[str, Any], key: str) -> float | None:
    """A qubit's z offset, following QuAM's `#./other_key` references."""

    z = qubit.get("z") if isinstance(qubit, dict) else None
    if not isinstance(z, dict):
        return None
    value: Any = z.get(key)
    seen = {key}
    while isinstance(value, str) and value.startswith("#./"):
        target = value[3:]
        if target in seen:
            return None
        seen.add(target)
        value = z.get(target)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def channel_kind(qubit: dict[str, Any]) -> str:
    class_name = str(qubit.get("xy", {}).get("__class__", ""))
    if class_name.endswith("MWChannel"):
        return "mw"
    if class_name.endswith("IQChannel"):
        return "iq"
    raise StateError(f"Unsupported XY channel class: {class_name or '<missing>'}")


def bootstrap_patch(
    state: dict[str, Any],
    qubits: list[str],
    policies: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    limits = policies["instrument_limits"]
    x180_fraction = float(limits["bootstrap_x180_fraction"])
    x90_fraction = float(limits["bootstrap_x90_fraction_of_x180"])
    patch: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for name in qubits:
        qubit = state["qubits"][name]
        kind = channel_kind(qubit)
        x180_limit = float(limits[kind]["max_x180_wf_amplitude"])
        initial_x180 = x180_limit * x180_fraction
        initial_x90 = initial_x180 * x90_fraction
        x180_backing = operation_backing_name(qubit, "x180")
        x90_backing = operation_backing_name(qubit, "x90")
        old_x180 = operation_amplitude(qubit, "x180")
        old_x90 = operation_amplitude(qubit, "x90")
        changed: dict[str, Any] = {}
        if old_x180 == 0:
            path = f"/qubits/{name}/xy/operations/{x180_backing}/amplitude"
            patch.append({"op": "replace", "path": path, "value": initial_x180})
            changed["x180"] = {"old": old_x180, "new": initial_x180}
        if old_x90 == 0:
            # Use the bootstrapped x180 value only when x180 itself was missing/zero.
            x90_value = initial_x90 if old_x180 == 0 else old_x180 * x90_fraction
            path = f"/qubits/{name}/xy/operations/{x90_backing}/amplitude"
            patch.append({"op": "replace", "path": path, "value": x90_value})
            changed["x90"] = {"old": old_x90, "new": x90_value}
        details[name] = {
            "channel_kind": kind,
            "max_x180_wf_amplitude": x180_limit,
            "changes": changed,
        }
    return patch, details


def fresh_qubit_defaults_patch(
    state: dict[str, Any],
    qubits: list[str],
    defaults: dict[str, Any],
    shared_xy: Collection[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Starting values for qubits measured from scratch (operator 2026-10-14).

    Empties `extras`, writes the configured T1/T2/anharmonicity, saturation and
    readout pulses, resets the readout discrimination and the flux offsets,
    and sets the x180/x90 gate: amplitude, x90 at half of it, DRAG terms 0,
    and a length that follows the XY wiring. Values held as `#` references are
    left alone; they follow their target.
    """

    patch: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    def put(path: str, value: Any) -> None:
        patch.append({"op": "add", "path": path, "value": value})

    def numeric(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    for name in qubits:
        qubit = state["qubits"][name]
        base = f"/qubits/{name}"
        shared = name in shared_xy
        gate_ns = int(defaults["x180_length_ns"]["shared_xy" if shared else "private_xy"])
        x180_amplitude = float(defaults["x180_amplitude"])
        put(f"{base}/extras", {})
        put(f"{base}/T1", float(defaults["T1_s"]))
        put(f"{base}/T2ramsey", float(defaults["T2ramsey_s"]))
        put(f"{base}/T2echo", float(defaults["T2echo_s"]))
        put(f"{base}/anharmonicity", float(defaults["anharmonicity_hz"]))
        if "freq_vs_flux_01_quad_term" in qubit:
            put(f"{base}/freq_vs_flux_01_quad_term", 0.0)
        z = qubit.get("z")
        if isinstance(z, dict):
            for key in ("joint_offset", "independent_offset"):
                if key in z and not isinstance(z[key], str):
                    put(f"{base}/z/{key}", 0.0)
        operations = qubit["xy"]["operations"]
        x180 = operation_backing_name(qubit, "x180")
        x90 = operation_backing_name(qubit, "x90")
        old_length = operations[x180].get("length")
        for op_name, operation in operations.items():
            if not isinstance(operation, dict):
                continue
            lowered = str(op_name).lower()
            length = operation.get("length")
            if op_name == x180 or (
                numeric(length)
                and numeric(old_length)
                and float(length) == float(old_length)
                and not lowered.startswith(("ef_", "saturation"))
            ):
                put(f"{base}/xy/operations/{op_name}/length", gate_ns)
        for op_name, amplitude in ((x180, x180_amplitude), (x90, x180_amplitude / 2)):
            operation = operations[op_name]
            if not isinstance(operation.get("amplitude"), str):
                put(f"{base}/xy/operations/{op_name}/amplitude", amplitude)
            for key in ("alpha", "detuning"):
                if numeric(operation.get(key)):
                    put(f"{base}/xy/operations/{op_name}/{key}", 0.0)
        saturation = operations.get("saturation")
        if isinstance(saturation, dict):
            put(
                f"{base}/xy/operations/saturation/amplitude",
                float(defaults["saturation_amplitude"]),
            )
            put(
                f"{base}/xy/operations/saturation/length",
                int(defaults["saturation_length_ns"]),
            )
        resonator = qubit.get("resonator")
        readout = (
            (resonator.get("operations") or {}).get("readout")
            if isinstance(resonator, dict)
            else None
        )
        if isinstance(readout, dict):
            rr = f"{base}/resonator"
            put(f"{rr}/operations/readout/length", int(defaults["readout_length_ns"]))
            put(f"{rr}/depletion_time", int(defaults["depletion_time_ns"]))
            put(f"{rr}/operations/readout/integration_weights_angle", 0.0)
            put(f"{rr}/operations/readout/threshold", 0.0)
            put(f"{rr}/operations/readout/rus_exit_threshold", 0.0)
            put(f"{rr}/confusion_matrix", [[1.0, 0.0], [0.0, 1.0]])
        details[name] = {
            "xy_line": "shared" if shared else "private",
            "x180_length_ns": gate_ns,
            "x180_amplitude": x180_amplitude,
            "cleared_extras": sorted((qubit.get("extras") or {}).keys()),
        }
    return patch, details


def drive_lo_recenter_patch(
    state: dict[str, Any],
    wiring: dict[str, Any],
    qubits: list[str],
    lo_grid_hz: float = 100_000_000,
    *,
    force_zero_if: bool = False,
    target_lo_hz: dict[str, float] | None = None,
    target_rf_hz: dict[str, float] | None = None,
    allow_shared_output: bool = False,
    band_window: Callable[[Any], tuple[float, float] | None] | None = None,
    max_if_abs_hz: float | None = None,
    park_out_of_range: bool = False,
    focus_rf_hz: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build XY LO/IF patches for 03a coarse-search placement.

    A private XY output moves on its own. A shared one may only move with
    `allow_shared_output`, for a single qubit at a time, and then every other
    qubit on that port has its IF compensated so its RF is preserved -- the
    port LO is one knob for all of them. `band_window` maps a port's band to
    the LO range that band allows; an LO outside it is refused, except for an
    explicit coarse-window shift, which is clamped to the edge so the 03a
    search keeps moving instead of stalling.

    With `park_out_of_range` (operator instruction 2026-10-12) a companion the
    new LO cannot reach within `max_if_abs_hz` is parked at that limit instead
    of refusing the move: a per-qubit pass never plays it, and the XY LO plan
    puts it back from `extras.idle_freq` when its own turn comes.
    `focus_rf_hz` gives a selected qubit's real RF when its own IF is parked,
    so its coarse window is centred where it belongs."""
    if not qubits or len(set(qubits)) != len(qubits):
        raise StateError("LO recenter qubits must be a non-empty unique list")
    if (
        not isinstance(lo_grid_hz, (int, float))
        or isinstance(lo_grid_hz, bool)
        or not math.isfinite(float(lo_grid_hz))
        or float(lo_grid_hz) <= 0
    ):
        raise StateError("LO grid must be finite and positive")
    if target_lo_hz is not None and target_rf_hz is not None:
        raise StateError("Explicit LO and RF targets are mutually exclusive")
    if target_lo_hz is not None and set(target_lo_hz) != set(qubits):
        raise StateError("Explicit LO centers must exactly match selected qubits")
    if target_rf_hz is not None and set(target_rf_hz) != set(qubits):
        raise StateError("Explicit RF centers must exactly match selected qubits")
    wiring_qubits = wiring.get("wiring", {}).get("qubits", {})
    if not isinstance(wiring_qubits, dict):
        raise StateError("Wiring does not contain a qubit map")

    patch: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for name in qubits:
        try:
            qubit = state["qubits"][name]
            xy = qubit["xy"]
        except (KeyError, TypeError) as exc:
            raise StateError(f"Unknown qubit for LO recenter: {name}") from exc
        if not str(xy.get("__class__", "")).endswith("MWChannel"):
            raise StateError(f"{name} XY channel is not an MWChannel")

        output_ref = wiring_qubits.get(name, {}).get("xy", {}).get("opx_output")
        if not isinstance(output_ref, str):
            direct_ref = xy.get("opx_output")
            output_ref = direct_ref if isinstance(direct_ref, str) else None
        match = re.fullmatch(
            r"#/ports/mw_outputs/([^/]+)/([0-9]+)/([0-9]+)",
            output_ref or "",
        )
        if match is None:
            raise StateError(f"{name} has no resolvable MW XY output")

        owners = sorted(
            owner
            for owner, item in wiring_qubits.items()
            if isinstance(item, dict)
            and item.get("xy", {}).get("opx_output") == output_ref
        )
        companions: list[str] = []
        if owners != [name]:
            if not allow_shared_output:
                raise StateError(
                    f"{name} XY output is shared by {owners or ['unknown']}; "
                    "automatic LO recenter requires a private output"
                )
            if len(qubits) != 1:
                raise StateError(
                    f"{name} XY output is shared by {owners or ['unknown']}; "
                    "a shared LO moves for one qubit at a time, so recenter "
                    "the qubit whose pass is starting on its own"
                )
            companions = [owner for owner in owners if owner != name]

        port_path = output_ref[1:]
        lo_path = f"{port_path}/upconverter_frequency"
        if_path = f"/qubits/{name}/xy/intermediate_frequency"
        try:
            old_lo = pointer_get(state, lo_path)
            old_if = pointer_get(state, if_path)
        except (KeyError, IndexError, StateError) as exc:
            raise StateError(f"Cannot resolve LO/IF state for {name}") from exc
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (old_lo, old_if)
        ):
            raise StateError(f"{name} LO and IF must be numeric")

        old_rf = float(old_lo) + float(old_if)
        if focus_rf_hz is not None and name in focus_rf_hz:
            old_rf = float(focus_rf_hz[name])
        grid = float(lo_grid_hz)
        if target_rf_hz is not None:
            requested_rf = target_rf_hz[name]
            if (
                not isinstance(requested_rf, (int, float))
                or isinstance(requested_rf, bool)
                or not math.isfinite(float(requested_rf))
            ):
                raise StateError(f"{name} explicit RF center must be finite")
            new_lo = math.floor(float(requested_rf) / grid + 0.5) * grid
            new_if = float(requested_rf) - new_lo
            mode = "candidate_center_preserve_rf"
        elif target_lo_hz is not None:
            requested_lo = target_lo_hz[name]
            if (
                not isinstance(requested_lo, (int, float))
                or isinstance(requested_lo, bool)
                or not math.isfinite(float(requested_lo))
            ):
                raise StateError(f"{name} explicit LO center must be finite")
            grid_units = float(requested_lo) / grid
            if not math.isclose(grid_units, round(grid_units), abs_tol=1e-9):
                raise StateError(
                    f"{name} explicit LO center must lie on the {grid:g} Hz grid"
                )
            new_lo = float(requested_lo)
            new_if = 0.0
            mode = "shifted_coarse_window"
        elif force_zero_if:
            new_lo = old_rf
            new_if = 0.0
            mode = "preserve_rf_zero_if"
        else:
            new_lo = math.floor(old_rf / grid + 0.5) * grid
            new_if = old_rf - new_lo
            mode = "nearest_grid_preserve_rf"
        window = None
        if band_window is not None:
            try:
                band = pointer_get(state, f"{port_path}/band")
            except (KeyError, IndexError, StateError):
                # A port with no declared band keeps the global LO limits.
                band = None
            window = band_window(band)
        clamped = False
        if window is not None and not window[0] <= new_lo <= window[1]:
            if mode != "shifted_coarse_window":
                raise StateError(
                    f"{name} needs an LO of {new_lo / 1e9:g} GHz, which is "
                    f"outside the usable part of its band "
                    f"({window[0] / 1e9:g}-{window[1] / 1e9:g} GHz)"
                )
            new_lo = min(max(new_lo, window[0]), window[1])
            new_lo = math.floor(new_lo / grid + 0.5) * grid
            new_lo = min(max(new_lo, window[0]), window[1])
            clamped = True

        compensated: dict[str, float] = {}
        parked: dict[str, float] = {}
        for companion in companions:
            companion_if_path = f"/qubits/{companion}/xy/intermediate_frequency"
            try:
                companion_if = pointer_get(state, companion_if_path)
            except (KeyError, IndexError, StateError) as exc:
                raise StateError(
                    f"Cannot resolve the XY IF of {companion}, which shares "
                    f"the LO being moved for {name}"
                ) from exc
            if not isinstance(companion_if, (int, float)) or isinstance(
                companion_if, bool
            ):
                raise StateError(f"{companion} XY IF must be numeric")
            preserved = float(old_lo) + float(companion_if) - new_lo
            if (
                max_if_abs_hz is not None
                and abs(preserved) > float(max_if_abs_hz)
                and park_out_of_range
            ):
                parked[companion] = float(old_lo) + float(companion_if)
                preserved = math.copysign(float(max_if_abs_hz), preserved)
            elif max_if_abs_hz is not None and abs(preserved) > float(max_if_abs_hz):
                raise StateError(
                    f"Moving the shared LO to {new_lo / 1e9:g} GHz for {name} "
                    f"would put {companion} at an IF of {preserved / 1e6:g} MHz, "
                    f"beyond the {float(max_if_abs_hz) / 1e6:g} MHz limit"
                )
            compensated[companion] = preserved
            patch.append(
                {
                    "op": "replace",
                    "path": companion_if_path,
                    "value": preserved,
                }
            )

        patch.extend(
            [
                {"op": "replace", "path": lo_path, "value": new_lo},
                {"op": "replace", "path": if_path, "value": new_if},
            ]
        )
        details[name] = {
            "mode": mode,
            "shared_with": companions,
            "compensated_if_hz": compensated,
            "parked_rf_hz": parked,
            "band_window_hz": list(window) if window is not None else None,
            "clamped_to_band": clamped,
            "xy_output": output_ref,
            "lo_grid_hz": grid,
            "old_lo_hz": float(old_lo),
            "new_lo_hz": new_lo,
            "old_if_hz": float(old_if),
            "new_if_hz": new_if,
            "old_rf_hz": old_rf,
            "new_rf_hz": new_lo + new_if,
            "preserved_rf_hz": old_rf,
            "target_rf_hz": (
                float(target_rf_hz[name]) if target_rf_hz is not None else None
            ),
        }
    return patch, details


def commit_state(
    state_path: Path,
    patch: list[dict[str, Any]],
    backup_dir: Path,
    expected_hash: str,
) -> dict[str, str]:
    lock_path = backup_dir.parent / "state-commit.lock"
    try:
        with exclusive_file_lock(lock_path, "active state commit"):
            actual_hash = sha256_file(state_path)
            if actual_hash != expected_hash:
                raise StateError(
                    "Active state changed after this proposal was created; request a new proposal."
                )
            original = load_state(state_path)
            updated = apply_json_patch(original, patch)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / (
                f"{timestamp}-{actual_hash[:12]}-{uuid.uuid4().hex[:12]}-state.json"
            )
            shutil.copy2(state_path, backup_path)
            atomic_write_json(state_path, updated)
            after_hash = sha256_file(state_path)
    except RuntimeError as exc:
        raise StateError(str(exc)) from exc
    return {
        "before_hash": actual_hash,
        "after_hash": after_hash,
        "backup_path": str(backup_path),
    }


def snapshot_state_path(snapshot_path: Path) -> Path | None:
    candidates = [
        snapshot_path / "quam_state" / "state.json",
        snapshot_path / "quam_state.json",
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)
