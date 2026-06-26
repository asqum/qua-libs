"""Utilities for OPX1000 MW-FEM output power optimization.

On MW-FEM channels, output power is set by two knobs:
    P_out [dBm] = full_scale_power_dbm + 20 * log10(wf_amplitude)

For best SNR, use the lowest allowed full_scale_power_dbm and compensate with
a higher normalized waveform amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from quam.components.channels import IQChannel, MWChannel

from quam_libs.components import Transmon

# Hard-coded multiplex setup: five qubits share one MW readout port.
NUM_QUBITS_SHARING_MW_READOUT = 5
MAX_READOUT_WF_AMPLITUDE = 1.0 / NUM_QUBITS_SHARING_MW_READOUT  # 0.2

OPX1000_FULL_SCALE_POWERS_DBM: tuple[int, ...] = tuple(range(-11, 17, 3))
FULL_SCALE_POWER_STEP_DBM = 3


@dataclass(frozen=True)
class MWPowerSettings:
    full_scale_power_dbm: int
    amplitude: float
    output_power_dbm: float


def power_dbm_from_settings(full_scale_power_dbm: float, amplitude: float) -> float:
    """Return output power in dBm for normalized amplitude at a given full-scale power."""
    if amplitude <= 0:
        raise ValueError(f"Amplitude must be positive, got {amplitude}.")
    return full_scale_power_dbm + 20 * np.log10(amplitude)


def amplitude_from_power(full_scale_power_dbm: float, target_power_dbm: float) -> float:
    """Return normalized amplitude needed to reach target power at a fixed full-scale power."""
    return 10 ** ((target_power_dbm - full_scale_power_dbm) / 20)


def peak_voltage(
    full_scale_power_dbm: float,
    amplitude: float,
    impedance_ohm: float = 50,
) -> float:
    """Peak voltage into a resistive load for MW-FEM settings."""
    power_mw = 10 ** (full_scale_power_dbm / 10)
    full_scale_voltage = np.sqrt(2 * impedance_ohm * power_mw / 1000)
    return amplitude * full_scale_voltage


def scaled_power_dbm(current_power_dbm: float, amplitude_scale_factor: float) -> float:
    """Return the output power after scaling normalized amplitude by a linear factor."""
    if amplitude_scale_factor <= 0:
        raise ValueError(f"Scale factor must be positive, got {amplitude_scale_factor}.")
    return current_power_dbm + 20 * np.log10(amplitude_scale_factor)


def optimal_mw_power_settings(
    target_power_dbm: float,
    max_amplitude: float = 1.0,
    *,
    min_full_scale_power_dbm: Optional[int] = None,
) -> MWPowerSettings:
    """Pick the lowest full_scale_power_dbm that reaches target power within max_amplitude."""
    allowed = [
        power
        for power in OPX1000_FULL_SCALE_POWERS_DBM
        if min_full_scale_power_dbm is None or power >= min_full_scale_power_dbm
    ]
    for full_scale_power_dbm in sorted(allowed):
        amplitude = amplitude_from_power(full_scale_power_dbm, target_power_dbm)
        if amplitude <= max_amplitude:
            return MWPowerSettings(
                full_scale_power_dbm=full_scale_power_dbm,
                amplitude=amplitude,
                output_power_dbm=target_power_dbm,
            )
    raise ValueError(
        f"Cannot reach {target_power_dbm:.2f} dBm with max_amplitude={max_amplitude:.4f}."
    )


def get_channel_output_power(
    channel: Union[MWChannel, IQChannel],
    operation: str,
    impedance_ohm: float = 50,
) -> float:
    """Return current output power in dBm for a pulse operation on an MW or IQ channel."""
    if isinstance(channel, MWChannel):
        full_scale_power_dbm = channel.opx_output.full_scale_power_dbm
        amplitude = channel.operations[operation].amplitude
        return power_dbm_from_settings(full_scale_power_dbm, amplitude)

    from qualang_tools.units import unit

    u = unit(coerce_to_integer=True)
    amplitude = channel.operations[operation].amplitude
    gain = channel.frequency_converter_up.gain
    return gain + u.volts2dBm(amplitude, Z=impedance_ohm)


def apply_mw_power_settings(
    channel: MWChannel,
    operation: str,
    settings: MWPowerSettings,
) -> MWPowerSettings:
    """Write optimized MW-FEM settings to a channel operation."""
    channel.opx_output.full_scale_power_dbm = settings.full_scale_power_dbm
    channel.operations[operation].amplitude = settings.amplitude
    return settings


def apply_shared_readout_port_settings(
    qubits: Sequence[Transmon],
    target_powers_dbm: dict[str, float],
    *,
    max_amplitude: float = MAX_READOUT_WF_AMPLITUDE,
    operation: str = "readout",
) -> dict[str, MWPowerSettings]:
    """Apply SNR-optimal readout settings for qubits sharing one MW readout port."""
    per_qubit_optimal = {
        qubit.name: optimal_mw_power_settings(
            target_powers_dbm[qubit.name],
            max_amplitude=max_amplitude,
        )
        for qubit in qubits
    }
    shared_full_scale_power_dbm = max(
        settings.full_scale_power_dbm for settings in per_qubit_optimal.values()
    )

    applied_settings = {}
    for qubit in qubits:
        amplitude = amplitude_from_power(
            shared_full_scale_power_dbm,
            target_powers_dbm[qubit.name],
        )
        if amplitude > max_amplitude:
            raise ValueError(
                f"{qubit.name} needs amplitude {amplitude:.4f} > {max_amplitude:.4f} "
                f"with shared full_scale_power_dbm={shared_full_scale_power_dbm} dBm."
            )
        settings = MWPowerSettings(
            full_scale_power_dbm=shared_full_scale_power_dbm,
            amplitude=amplitude,
            output_power_dbm=target_powers_dbm[qubit.name],
        )
        apply_mw_power_settings(qubit.resonator, operation, settings)
        applied_settings[qubit.name] = settings
    return applied_settings


def scale_readout_powers(
    qubits: Sequence[Transmon],
    amplitude_scale_factor: float,
    *,
    operation: str = "readout",
    max_amplitude: float = MAX_READOUT_WF_AMPLITUDE,
    dry_run: bool = False,
) -> dict[str, MWPowerSettings]:
    """Scale all readout pulse amplitudes and re-optimize shared MW port settings."""
    target_powers_dbm = {
        qubit.name: scaled_power_dbm(
            get_channel_output_power(qubit.resonator, operation),
            amplitude_scale_factor,
        )
        for qubit in qubits
    }
    if dry_run:
        per_qubit_optimal = {
            qubit.name: optimal_mw_power_settings(
                target_powers_dbm[qubit.name],
                max_amplitude=max_amplitude,
            )
            for qubit in qubits
        }
        shared_full_scale_power_dbm = max(
            settings.full_scale_power_dbm for settings in per_qubit_optimal.values()
        )
        return {
            qubit.name: MWPowerSettings(
                full_scale_power_dbm=shared_full_scale_power_dbm,
                amplitude=amplitude_from_power(
                    shared_full_scale_power_dbm,
                    target_powers_dbm[qubit.name],
                ),
                output_power_dbm=target_powers_dbm[qubit.name],
            )
            for qubit in qubits
        }
    return apply_shared_readout_port_settings(
        qubits,
        target_powers_dbm,
        max_amplitude=max_amplitude,
        operation=operation,
    )


def apply_xy_port_settings(
    qubit: Transmon,
    target_powers_dbm: dict[str, float],
    *,
    max_amplitude: float = 1.0,
    dry_run: bool = False,
) -> dict[str, MWPowerSettings]:
    """Apply SNR-optimal settings for multiple pulses on one XY MW port."""
    per_operation_optimal = {
        operation: optimal_mw_power_settings(power_dbm, max_amplitude=max_amplitude)
        for operation, power_dbm in target_powers_dbm.items()
    }
    shared_full_scale_power_dbm = max(
        settings.full_scale_power_dbm for settings in per_operation_optimal.values()
    )

    applied_settings = {}
    for operation, target_power_dbm in target_powers_dbm.items():
        amplitude = amplitude_from_power(shared_full_scale_power_dbm, target_power_dbm)
        if amplitude > max_amplitude:
            raise ValueError(
                f"{qubit.name}.{operation} needs amplitude {amplitude:.4f} > "
                f"{max_amplitude:.4f} with shared full_scale_power_dbm="
                f"{shared_full_scale_power_dbm} dBm."
            )
        settings = MWPowerSettings(
            full_scale_power_dbm=shared_full_scale_power_dbm,
            amplitude=amplitude,
            output_power_dbm=target_power_dbm,
        )
        if not dry_run:
            qubit.xy.opx_output.full_scale_power_dbm = settings.full_scale_power_dbm
            qubit.xy.operations[operation].amplitude = settings.amplitude
        applied_settings[operation] = settings
    return applied_settings


def scale_xy_powers(
    qubits: Sequence[Transmon],
    amplitude_scale_factor: float,
    *,
    operations: Iterable[str] = ("x180", "x90", "saturation"),
    max_amplitude: float = 1.0,
    dry_run: bool = False,
) -> dict[str, dict[str, MWPowerSettings]]:
    """Scale qubit drive pulse amplitudes and re-optimize per-qubit MW settings."""
    results: dict[str, dict[str, MWPowerSettings]] = {}
    for qubit in qubits:
        target_powers_dbm = {}
        for operation in operations:
            if operation not in qubit.xy.operations:
                continue
            current_power_dbm = get_channel_output_power(qubit.xy, operation)
            target_powers_dbm[operation] = scaled_power_dbm(
                current_power_dbm,
                amplitude_scale_factor,
            )
        if not target_powers_dbm:
            continue
        results[qubit.name] = apply_xy_port_settings(
            qubit,
            target_powers_dbm,
            max_amplitude=max_amplitude,
            dry_run=dry_run,
        )
    return results


def format_power_settings(settings: MWPowerSettings) -> str:
    return (
        f"full_scale={settings.full_scale_power_dbm} dBm, "
        f"amplitude={settings.amplitude:.4f}, "
        f"output={settings.output_power_dbm:.2f} dBm"
    )


def is_mw_fem_channel(channel: Union[MWChannel, IQChannel]) -> bool:
    return isinstance(channel, MWChannel)


def is_mw_fem_readout(resonator) -> bool:
    return hasattr(resonator, "opx_output") and hasattr(
        resonator.opx_output, "full_scale_power_dbm"
    )


def optimize_operation_amplitude(
    channel: MWChannel,
    operation: str,
    target_amplitude: float,
    *,
    max_amplitude: float = 1.0,
) -> MWPowerSettings:
    """Re-split a target normalized amplitude into SNR-optimal MW-FEM settings."""
    target_power_dbm = power_dbm_from_settings(
        channel.opx_output.full_scale_power_dbm,
        target_amplitude,
    )
    settings = optimal_mw_power_settings(
        target_power_dbm,
        max_amplitude=max_amplitude,
    )
    apply_mw_power_settings(channel, operation, settings)
    return settings


def optimize_xy_operation_amplitude(
    qubit: Transmon,
    operation: str,
    target_amplitude: float,
    *,
    max_amplitude: float = 1.0,
) -> MWPowerSettings:
    return optimize_operation_amplitude(
        qubit.xy,
        operation,
        target_amplitude,
        max_amplitude=max_amplitude,
    )


def optimize_xy_sweep_headroom(
    qubit: Transmon,
    operation: str,
    max_amp_factor: float,
    *,
    max_wf_amplitude: float = 1.0,
) -> Optional[MWPowerSettings]:
    """Ensure amplitude_scale * base_amplitude stays within max_wf_amplitude during sweeps."""
    if max_amp_factor <= 0:
        raise ValueError(f"max_amp_factor must be positive, got {max_amp_factor}.")

    current_amplitude = qubit.xy.operations[operation].amplitude
    if current_amplitude * max_amp_factor <= max_wf_amplitude:
        return None

    target_power_dbm = get_channel_output_power(qubit.xy, operation)
    max_base_amplitude = max_wf_amplitude / max_amp_factor
    settings = optimal_mw_power_settings(
        target_power_dbm,
        max_amplitude=max_base_amplitude,
    )
    apply_mw_power_settings(qubit.xy, operation, settings)
    return settings


def mw_power_settings_to_dict(settings: MWPowerSettings) -> dict[str, float]:
    return {
        "full_scale_power_dbm": settings.full_scale_power_dbm,
        "amplitude": settings.amplitude,
        "output_power_dbm": settings.output_power_dbm,
    }


def apply_fitted_pi_amplitude(
    qubit: Transmon,
    operation: str,
    fitted_pi_amplitude: float,
    *,
    max_amplitude: float,
    update_x90: bool = False,
    dry_run: bool = False,
) -> dict[str, float | dict[str, float]]:
    """Apply fitted pi amplitude using SNR-optimal MW settings when available."""
    result: dict[str, float | dict[str, float]] = {"Pi_amplitude": fitted_pi_amplitude}

    if is_mw_fem_channel(qubit.xy):
        target_power_dbm = power_dbm_from_settings(
            qubit.xy.opx_output.full_scale_power_dbm,
            fitted_pi_amplitude,
        )
        xy_settings = apply_xy_port_settings(
            qubit,
            {operation: target_power_dbm},
            max_amplitude=max_amplitude,
            dry_run=dry_run,
        )
        result["Pi_amplitude"] = xy_settings[operation].amplitude
        result["power_settings"] = {
            op_name: mw_power_settings_to_dict(settings)
            for op_name, settings in xy_settings.items()
        }
        if (
            not dry_run
            and operation == "x180"
            and update_x90
            and "x90" in qubit.xy.operations
        ):
            qubit.xy.operations["x90"].amplitude = xy_settings[operation].amplitude / 2
        return result

    if fitted_pi_amplitude > max_amplitude:
        fitted_pi_amplitude = max_amplitude
    result["Pi_amplitude"] = fitted_pi_amplitude
    if not dry_run:
        qubit.xy.operations[operation].amplitude = fitted_pi_amplitude
        if operation == "x180" and update_x90 and "x90" in qubit.xy.operations:
            qubit.xy.operations["x90"].amplitude = fitted_pi_amplitude / 2
    return result

