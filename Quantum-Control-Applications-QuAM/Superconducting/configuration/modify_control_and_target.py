# %%
"""
Align qubit_pair control/target with physics-based moving qubit roles, and
pre-compute CZ (or iSWAP) flux detuning for node 61 (use_saved_detuning=True).

Convention: moving qubit == control qubit (flux pulse is applied on control).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional

import numpy as np

from quam_libs.components import QuAM
from quam_libs.quam_builder.machine import save_machine

# %% {User settings}
# Process all pairs when None; otherwise only the listed pair names.
qubit_pair_names: Optional[list[str]] = None  # e.g. ["coupler_q1_q2", "coupler_q2_q3"]

gate_type: Literal["cz", "iswap"] = "cz"

# Swap wiring control_qubit / target_qubit when moving != current control.
swap_control_target: bool = True

# Write computed flux centre to qubit_pair.detuning (for CZ gate).
update_detuning: bool = False

# Print planned changes only; do not save state.json / wiring.json.
dry_run: bool = False

# %%


class QubitRoles(NamedTuple):
    """Resolved qubit roles for a two-qubit gate."""

    moving: object
    stationary: object
    leakage: object
    high: object
    low: object


def qubit_frequency_hz(qubit) -> float:
    """Return f_01 in Hz, falling back to xy.RF_frequency."""
    f_01 = getattr(qubit, "f_01", None)
    if isinstance(f_01, (int, float)):
        return float(f_01)
    return float(qubit.xy.RF_frequency)


def high_low_qubits(qp) -> tuple[object, object]:
    if qubit_frequency_hz(qp.qubit_control) >= qubit_frequency_hz(qp.qubit_target):
        return qp.qubit_control, qp.qubit_target
    return qp.qubit_target, qp.qubit_control


def resolve_qubit_roles(qp, gate: Literal["cz", "iswap"] = "cz") -> QubitRoles:
    """
    Resolve moving / stationary / leakage / high / low for the interaction.

    iSWAP: high-frequency qubit tunes down to the low-frequency partner.
    CZ (|11> <-> |20>): if delta > |alpha_high|, high qubit moves; else low qubit moves.
    |alpha_high| is the anharmonicity of the higher-frequency qubit in the pair.
    Leakage qubit is always the high-frequency qubit.
    """
    high_q, low_q = high_low_qubits(qp)

    if gate == "iswap":
        return QubitRoles(moving=high_q, stationary=low_q, leakage=high_q, high=high_q, low=low_q)

    if gate == "cz":
        delta = qubit_frequency_hz(high_q) - qubit_frequency_hz(low_q)
        moving = high_q if delta > abs(high_q.anharmonicity) else low_q
        stationary = low_q if moving is high_q else high_q
        return QubitRoles(moving=moving, stationary=stationary, leakage=high_q, high=high_q, low=low_q)

    raise ValueError(f"Invalid gate_type: {gate!r}")


def estimate_flux_detuning(qp, roles: QubitRoles, gate: Literal["cz", "iswap"] = "cz") -> float:
    """
    Flux bias centre (V) for the moving qubit at the interaction point.

    flux^2 = -detuning_hz / freq_vs_flux_01_quad_term  (quad < 0 at upper sweet spot).
    """
    quad = roles.moving.freq_vs_flux_01_quad_term
    if quad == 0:
        raise ValueError(
            f"Pair {qp.name}: moving qubit {roles.moving.name} has "
            "freq_vs_flux_01_quad_term=0. Run Ramsey-vs-flux calibration first."
        )

    if gate == "iswap":
        detuning_hz = qubit_frequency_hz(roles.moving) - qubit_frequency_hz(roles.stationary)
        if detuning_hz < 0:
            raise ValueError(
                f"Pair {qp.name} [iSWAP]: moving qubit {roles.moving.name} is below "
                f"partner {roles.stationary.name}."
            )
    elif gate == "cz":
        alpha_high = abs(roles.high.anharmonicity)
        delta_rf = qubit_frequency_hz(roles.high) - qubit_frequency_hz(roles.low)
        if roles.moving is roles.high:
            detuning_hz = delta_rf - alpha_high
        else:
            detuning_hz = alpha_high - delta_rf
    else:
        raise ValueError(f"Invalid gate_type: {gate!r}")

    ratio = -detuning_hz / quad
    if ratio < 0:
        raise ValueError(
            f"Pair {qp.name}: cannot estimate flux detuning (sqrt of negative ratio {ratio:.3g}). "
            "Check frequencies, anharmonicity, and freq_vs_flux_01_quad_term."
        )
    return float(np.sqrt(ratio))


def _qubit_name_from_wiring_ref(ref) -> str:
    """Return the qubit name from a wiring entry.

    After QuAM.load(), the wiring reference may already be resolved into a
    Transmon object, so handle both the raw "#/qubits/qX" string and the object.
    """
    if isinstance(ref, str):
        return ref.rstrip("/").split("/")[-1]
    return getattr(ref, "name", str(ref))


def find_wiring_pair_key(machine: QuAM, qp) -> Optional[str]:
    pair_qubits = {qp.qubit_control.name, qp.qubit_target.name}
    for key, pair_data in machine.wiring.get("qubit_pairs", {}).items():
        channels = pair_data.get("c", {})
        ctrl = _qubit_name_from_wiring_ref(channels.get("control_qubit", ""))
        tgt = _qubit_name_from_wiring_ref(channels.get("target_qubit", ""))
        if {ctrl, tgt} == pair_qubits:
            return key
    return None


def _qubit_wiring_ref(qubit) -> str:
    """Canonical wiring reference string for a qubit."""
    name = qubit if isinstance(qubit, str) else qubit.name
    return f"#/qubits/{name}"


def swap_control_target_wiring(machine: QuAM, qp) -> None:
    wiring_key = find_wiring_pair_key(machine, qp)
    if wiring_key is None:
        raise ValueError(f"Pair {qp.name}: could not find matching entry in machine.wiring['qubit_pairs'].")

    channels = machine.wiring["qubit_pairs"][wiring_key]["c"]
    ctrl_name = _qubit_name_from_wiring_ref(channels["control_qubit"])
    tgt_name = _qubit_name_from_wiring_ref(channels["target_qubit"])
    # Wiring fields are QuAM references; assign "#/qubits/qX" strings, not resolved objects.
    channels["control_qubit"] = _qubit_wiring_ref(tgt_name)
    channels["target_qubit"] = _qubit_wiring_ref(ctrl_name)


def swap_gate_role_fields(qp) -> None:
    for gate in qp.gates.values():
        if not hasattr(gate, "phase_shift_control") or not hasattr(gate, "phase_shift_target"):
            continue
        gate.phase_shift_control, gate.phase_shift_target = (
            gate.phase_shift_target,
            gate.phase_shift_control,
        )


def swap_mutual_flux_bias(qp) -> None:
    if qp.mutual_flux_bias and len(qp.mutual_flux_bias) >= 2:
        qp.mutual_flux_bias = [qp.mutual_flux_bias[1], qp.mutual_flux_bias[0]]


@dataclass
class PairUpdateReport:
    pair_name: str
    current_control: str
    current_target: str
    moving: str
    stationary: str
    high: str
    low: str
    delta_mhz: float
    alpha_high_mhz: float
    needs_swap: bool
    swapped: bool
    old_detuning: Optional[float]
    new_detuning: Optional[float]
    detuning_updated: bool
    error: Optional[str] = None


def process_pair(qp, machine: QuAM) -> PairUpdateReport:
    roles = resolve_qubit_roles(qp, gate_type)
    needs_swap = roles.moving is not qp.qubit_control
    delta_mhz = (qubit_frequency_hz(roles.high) - qubit_frequency_hz(roles.low)) / 1e6
    alpha_high_mhz = roles.high.anharmonicity / 1e6

    report = PairUpdateReport(
        pair_name=qp.name,
        current_control=qp.qubit_control.name,
        current_target=qp.qubit_target.name,
        moving=roles.moving.name,
        stationary=roles.stationary.name,
        high=roles.high.name,
        low=roles.low.name,
        delta_mhz=delta_mhz,
        alpha_high_mhz=alpha_high_mhz,
        needs_swap=needs_swap,
        swapped=False,
        old_detuning=getattr(qp, "detuning", None),
        new_detuning=None,
        detuning_updated=False,
    )

    try:
        if needs_swap and swap_control_target and not dry_run:
            swap_control_target_wiring(machine, qp)
            swap_gate_role_fields(qp)
            swap_mutual_flux_bias(qp)
            report.swapped = True

        if update_detuning:
            report.new_detuning = estimate_flux_detuning(qp, roles, gate_type)
            if not dry_run:
                qp.detuning = report.new_detuning
            report.detuning_updated = True
    except Exception as exc:
        report.error = str(exc)

    return report


def print_report(report: PairUpdateReport) -> None:
    print(f"\n=== {report.pair_name} ({gate_type}) ===")
    print(
        f"  frequencies: high={report.high}, low={report.low} | "
        f"Δ={report.delta_mhz:.1f} MHz, |α_high|={abs(report.alpha_high_mhz):.1f} MHz (α of {report.high})"
    )
    print(
        f"  current:  control={report.current_control}, target={report.current_target}"
    )
    print(
        f"  resolved: moving={report.moving} (-> control), stationary={report.stationary} (-> target)"
    )

    if report.needs_swap:
        if report.swapped:
            print(f"  swap:     control/target swapped in wiring (+ gate phase shifts)")
        elif not swap_control_target:
            print("  swap:     needed but swap_control_target=False")
        elif report.error:
            print("  swap:     needed but failed (see ERROR)")
        elif dry_run:
            print("  swap:     needed; planned only (dry_run, not applied)")
        else:
            print("  swap:     needed but not applied")
    else:
        print("  swap:     already aligned")

    if report.detuning_updated and report.new_detuning is not None:
        old = report.old_detuning
        if old is None:
            print(f"  detuning: set to {report.new_detuning:.6f} V")
        else:
            print(f"  detuning: {old:.6f} -> {report.new_detuning:.6f} V")
    elif update_detuning:
        print("  detuning: not updated")

    if report.error:
        print(f"  ERROR:    {report.error}")


# %% {Run}
machine = QuAM.load()
state_path = machine.get_quam_state_path()

if qubit_pair_names is None:
    pairs = list(machine.qubit_pairs.values())
else:
    pairs = [machine.qubit_pairs[name] for name in qubit_pair_names]

print(f"Loaded state from {state_path}")
print(f"Processing {len(pairs)} qubit pair(s), gate_type={gate_type!r}, dry_run={dry_run}")

reports = [process_pair(qp, machine) for qp in pairs]
for report in reports:
    print_report(report)

if dry_run:
    print("\n[dry_run] No files written. Set dry_run=False to save the updated QuAM state.")
else:
    save_machine(machine, state_path)
    print(f"\nSaved updated state to {state_path}")

# %%
