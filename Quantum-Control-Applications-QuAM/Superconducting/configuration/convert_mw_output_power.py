"""
Scale readout and/or XY drive powers on OPX1000 MW-FEM channels.

This script keeps the physical output power while re-splitting it into the SNR-optimal
combination of the lowest allowed full_scale_power_dbm and the highest normalized
waveform amplitude.

Examples:
    readout_amplitude_scale = 2.0  -> double readout pulse amplitude (+6 dBm output power)
    xy_amplitude_scale = 0.5       -> halve XY pulse amplitude (-6 dBm output power)

Readout uses a hard-coded multiplex assumption: 5 qubits share one MW readout port,
so each readout pulse is limited to normalized amplitude 0.2.
"""

from __future__ import annotations

from typing import Optional

from quam_libs.components import QuAM
from quam_libs.lib.mw_power_utils import (
    MAX_READOUT_WF_AMPLITUDE,
    NUM_QUBITS_SHARING_MW_READOUT,
    format_power_settings,
    get_channel_output_power,
    scale_readout_powers,
    scale_xy_powers,
)

# %% Input zone
state_PATH: Optional[str] = None

# Set to 1.0 to leave that channel unchanged.
readout_amplitude_scale: float = 1.0
xy_amplitude_scale: float = 1.0

# XY pulses to rescale together. x180 is always included when scaling XY.
xy_operations: tuple[str, ...] = ("x180", "x90", "saturation")

# Preview only; set False to write changes back to the QuAM state.
dry_run: bool = True

# Which qubits to update. None means machine.active_qubits.
qubit_names: Optional[list[str]] = None
# %% Input zone end


def _capture_readout_powers(qubits) -> dict[str, float]:
    return {
        qubit.name: get_channel_output_power(qubit.resonator, "readout")
        for qubit in qubits
    }


def _capture_xy_powers(qubits, operations) -> dict[str, dict[str, float]]:
    captured = {}
    for qubit in qubits:
        captured[qubit.name] = {}
        for operation in operations:
            if operation in qubit.xy.operations:
                captured[qubit.name][operation] = get_channel_output_power(qubit.xy, operation)
    return captured


def _print_readout_summary(qubits, settings_by_qubit, before_powers) -> None:
    print("\nReadout settings (shared MW port):")
    if not settings_by_qubit:
        print("  skipped")
        return

    shared_full_scale = next(iter(settings_by_qubit.values())).full_scale_power_dbm
    print(
        f"  multiplex: {NUM_QUBITS_SHARING_MW_READOUT} qubits, "
        f"max amplitude per qubit = {MAX_READOUT_WF_AMPLITUDE:.3f}"
    )
    print(f"  shared full_scale_power_dbm = {shared_full_scale} dBm")
    for qubit in qubits:
        settings = settings_by_qubit[qubit.name]
        before = before_powers[qubit.name]
        print(
            f"  {qubit.name}: before={before:.2f} dBm -> {format_power_settings(settings)}"
        )


def _print_xy_summary(qubits, settings_by_qubit, before_powers) -> None:
    print("\nXY settings:")
    if not any(settings_by_qubit.values()):
        print("  skipped")
        return

    for qubit in qubits:
        operation_settings = settings_by_qubit.get(qubit.name, {})
        if not operation_settings:
            continue
        print(f"  {qubit.name}:")
        for operation, settings in operation_settings.items():
            before = before_powers[qubit.name][operation]
            print(f"    {operation}: before={before:.2f} dBm -> {format_power_settings(settings)}")


def main() -> None:
    if readout_amplitude_scale <= 0 or xy_amplitude_scale <= 0:
        raise ValueError("Scale factors must be positive.")

    machine = QuAM.load(state_PATH) if state_PATH is not None else QuAM.load()
    qubits = (
        [machine.qubits[name] for name in qubit_names]
        if qubit_names is not None
        else machine.active_qubits
    )

    readout_before = _capture_readout_powers(qubits)
    xy_before = _capture_xy_powers(qubits, xy_operations)

    readout_settings = {}
    if readout_amplitude_scale != 1.0:
        readout_settings = scale_readout_powers(
            qubits,
            readout_amplitude_scale,
            dry_run=dry_run,
        )

    xy_settings = {}
    if xy_amplitude_scale != 1.0:
        xy_settings = scale_xy_powers(
            qubits,
            xy_amplitude_scale,
            operations=xy_operations,
            dry_run=dry_run,
        )

    _print_readout_summary(qubits, readout_settings, readout_before)
    _print_xy_summary(qubits, xy_settings, xy_before)

    if dry_run:
        print("\nDry run only. Set dry_run = False to save the updated QuAM state.")
    else:
        machine.save()
        print("\nSaved updated QuAM state.")


if __name__ == "__main__":
    main()

# %%
