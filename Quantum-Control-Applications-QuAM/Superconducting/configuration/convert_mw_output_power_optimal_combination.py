"""
Optimize the OPX1000 MW-FEM power combination without changing output power.

MW-FEM output power is set by two knobs:
    P_out [dBm] = full_scale_power_dbm + 20 * log10(wf_amplitude)

For a fixed P_out, many (full_scale_power_dbm, amplitude) pairs give the same power.
The optimal combination is: pick the *lowest allowed* full_scale_power_dbm and use the
*highest amplitude within your cap* to reach the same P_out.

This script reads the current settings, keeps each operation's output power unchanged, and
re-splits that power into this optimal combination.

You set the maximum normalized amplitude each channel may use:
    xy_max_amplitude          for XY drive
    1 / num_qubits_sharing_readout   for readout (multiplex cap)

The script then finds the lowest full_scale_power_dbm whose required amplitude still fits
within that cap. When several operations share one port (x180/x90/saturation on XY), the
most power-hungry operation sets full_scale_power_dbm for the whole port.

Readout multiplex: num_qubits_sharing_readout qubits share one MW readout port.
"""

from __future__ import annotations

from typing import Optional

from quam_libs.components import QuAM
from quam_libs.lib.mw_power_utils import (
    apply_shared_readout_port_settings,
    apply_xy_port_settings,
    format_power_settings,
    get_channel_output_power,
    is_mw_fem_channel,
)

# %% Input zone
# None uses the current/default QuAM state path (QuAM.load()).
state_PATH: Optional[str] = None

# Which qubits to update. None means machine.active_qubits.
qubit_names: Optional[list[str]] = None

# Re-optimize the XY drive channels.
optimize_xy: bool = True
# XY operations sharing each drive port. The most power-hungry one sets full_scale_power_dbm.
xy_operations: tuple[str, ...] = ("x180", "x90")  # ("x180", "x90", "saturation")
# Max normalized base amplitude the XY waveforms may reach. The script lowers
# full_scale_power_dbm until the most power-hungry operation hits this cap.
# Must be <= the hardware limit (MW-FEM max_wf_amplitude = 1.0).
xy_max_amplitude: float = 0.5

# Re-optimize the readout channels (shared MW port).
optimize_readout: bool = True
# Number of qubits sharing one MW readout port. Change this if the readout wiring changes.
# Each readout pulse amplitude cap is 1.0 / num_qubits_sharing_readout.
num_qubits_sharing_readout: int = 5

# Preview only; set False to write changes back to the QuAM state.
dry_run: bool = False
# %% Input zone end


def _capture_channel_operation(channel, operation) -> dict[str, float]:
    info = {
        "power": get_channel_output_power(channel, operation),
        "amplitude": channel.operations[operation].amplitude,
    }
    if is_mw_fem_channel(channel):
        info["full_scale"] = channel.opx_output.full_scale_power_dbm
    return info


def _capture_readout_powers(qubits) -> dict[str, dict[str, float]]:
    return {
        qubit.name: _capture_channel_operation(qubit.resonator, "readout")
        for qubit in qubits
    }


def _capture_xy_powers(qubits, operations) -> dict[str, dict[str, dict[str, float]]]:
    captured = {}
    for qubit in qubits:
        captured[qubit.name] = {}
        for operation in operations:
            if operation in qubit.xy.operations:
                captured[qubit.name][operation] = _capture_channel_operation(qubit.xy, operation)
    return captured


def _optimize_xy(qubits, operations, max_amplitude, *, dry_run):
    """Lower each XY port's full_scale_power_dbm until the worst-case op hits max_amplitude."""
    results: dict[str, dict] = {}
    for qubit in qubits:
        if not is_mw_fem_channel(qubit.xy):
            continue
        target_powers_dbm = {
            operation: get_channel_output_power(qubit.xy, operation)
            for operation in operations
            if operation in qubit.xy.operations
        }
        if not target_powers_dbm:
            continue
        results[qubit.name] = apply_xy_port_settings(
            qubit,
            target_powers_dbm,
            max_amplitude=max_amplitude,
            dry_run=dry_run,
        )
    return results


def _optimize_readout(qubits, max_amplitude, *, dry_run):
    """Lower the shared readout port's full_scale_power_dbm until a qubit hits max_amplitude."""
    mw_qubits = [q for q in qubits if is_mw_fem_channel(q.resonator)]
    if not mw_qubits:
        return {}
    target_powers_dbm = {
        qubit.name: get_channel_output_power(qubit.resonator, "readout")
        for qubit in mw_qubits
    }
    if dry_run:
        from quam_libs.lib.mw_power_utils import (
            MWPowerSettings,
            amplitude_from_power,
            optimal_mw_power_settings,
        )

        per_qubit_optimal = {
            qubit.name: optimal_mw_power_settings(
                target_powers_dbm[qubit.name],
                max_amplitude=max_amplitude,
            )
            for qubit in mw_qubits
        }
        shared_full_scale = max(s.full_scale_power_dbm for s in per_qubit_optimal.values())
        return {
            qubit.name: MWPowerSettings(
                full_scale_power_dbm=shared_full_scale,
                amplitude=amplitude_from_power(shared_full_scale, target_powers_dbm[qubit.name]),
                output_power_dbm=target_powers_dbm[qubit.name],
            )
            for qubit in mw_qubits
        }
    return apply_shared_readout_port_settings(
        mw_qubits,
        target_powers_dbm,
        max_amplitude=max_amplitude,
    )


def _print_xy_summary(qubits, settings_by_qubit, before_powers, max_amplitude) -> None:
    print("\nXY settings (optimal power combination, output power preserved):")
    if not any(settings_by_qubit.values()):
        print("  skipped")
        return
    print(f"  max base amplitude = {max_amplitude}")
    for qubit in qubits:
        operation_settings = settings_by_qubit.get(qubit.name, {})
        if not operation_settings:
            continue
        print(f"  {qubit.name}:")
        for operation, settings in operation_settings.items():
            before = before_powers[qubit.name][operation]
            delta_full_scale = settings.full_scale_power_dbm - before["full_scale"]
            print(
                f"    {operation}: before=full_scale={before['full_scale']} dBm, "
                f"amplitude={before['amplitude']:.4f}, output={before['power']:.2f} dBm "
                f"-> {format_power_settings(settings)} "
                f"(delta full_scale={delta_full_scale:+d} dB)"
            )


def _print_readout_summary(qubits, settings_by_qubit, before_powers, num_sharing) -> None:
    print("\nReadout settings (shared MW port, optimal power combination, output power preserved):")
    if not settings_by_qubit:
        print("  skipped")
        return
    shared_full_scale = next(iter(settings_by_qubit.values())).full_scale_power_dbm
    max_amplitude = 1.0 / num_sharing
    print(
        f"  multiplex: {num_sharing} qubits, max amplitude per qubit = {max_amplitude:.3f}"
    )
    print(f"  shared full_scale_power_dbm = {shared_full_scale} dBm")
    for qubit in qubits:
        settings = settings_by_qubit.get(qubit.name)
        if settings is None:
            continue
        before = before_powers[qubit.name]
        delta_full_scale = settings.full_scale_power_dbm - before["full_scale"]
        print(
            f"  {qubit.name}: before=full_scale={before['full_scale']} dBm, "
            f"amplitude={before['amplitude']:.4f}, output={before['power']:.2f} dBm "
            f"-> {format_power_settings(settings)} "
            f"(delta full_scale={delta_full_scale:+d} dB)"
        )


def main() -> None:
    if xy_max_amplitude <= 0:
        raise ValueError("xy_max_amplitude must be positive.")
    if num_qubits_sharing_readout <= 0:
        raise ValueError("num_qubits_sharing_readout must be positive.")

    readout_max_amplitude = 1.0 / num_qubits_sharing_readout
    machine = QuAM.load(state_PATH) if state_PATH is not None else QuAM.load()
    qubits = (
        [machine.qubits[name] for name in qubit_names]
        if qubit_names is not None
        else machine.active_qubits
    )

    xy_before = _capture_xy_powers(qubits, xy_operations)
    readout_before = _capture_readout_powers(qubits)

    xy_settings = {}
    if optimize_xy:
        xy_settings = _optimize_xy(
            qubits,
            xy_operations,
            xy_max_amplitude,
            dry_run=dry_run,
        )

    readout_settings = {}
    if optimize_readout:
        readout_settings = _optimize_readout(
            qubits,
            readout_max_amplitude,
            dry_run=dry_run,
        )

    _print_xy_summary(qubits, xy_settings, xy_before, xy_max_amplitude)
    _print_readout_summary(qubits, readout_settings, readout_before, num_qubits_sharing_readout)

    if dry_run:
        print("\nDry run only. Set dry_run = False to save the updated QuAM state.")
    else:
        machine.save()
        print("\nSaved updated QuAM state.")


if __name__ == "__main__":
    main()

# %%
