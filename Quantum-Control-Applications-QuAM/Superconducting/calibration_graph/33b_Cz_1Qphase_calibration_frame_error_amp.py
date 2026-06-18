# %%
"""
CZ SINGLE-QUBIT PHASE COMPENSATION WITH ERROR AMPLIFICATION

This node calibrates residual single-qubit phase shifts accumulated during CZ gate execution.
Compared with 33a_Cz_1Qphase_calibration_frame, it repeats the CZ operation a variable number
of times and applies a swept virtual frame correction after each CZ to amplify small phase errors.

For each selected qubit pair:
    1. Prepare either the control or target qubit in a Ramsey-like sequence.
    2. Apply 1..N CZ operations with per-CZ virtual-Z (stored compensation + swept frame).
    3. Find the swept frame that maximizes the mean return probability across the amplification axis.
    4. Add the fitted frame increment to the stored phase_shift_control / phase_shift_target.

State update:
    - qp.gates[operation].phase_shift_control
    - qp.gates[operation].phase_shift_target
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubit_pairs_from_node,
)
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr


# %% {Node_parameters}
qubit_pair_indexes = [4]


class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s" % (i, i + 1) for i in qubit_pair_indexes]
    num_averages: int = 100
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal["active", "thermal"] = "active"
    use_state_discrimination: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 10_000
    timeout: int = 100
    num_frames: int = 80
    frame_span: float = 0.4
    number_of_operations: int = 30
    load_data_id: Optional[int] = None
    operation: Literal["Cz", "Cz_flattop", "Cz_unipolar", "Cz_bipolar", "Cz_flattop_erf"] = "Cz_flattop"


node = QualibrationNode(name="33b_Cz_1Qphase_calibration_frame_error_amp", parameters=Parameters())
assert not (
    node.parameters.simulate and node.parameters.load_data_id is not None
), "If simulate is True, load_data_id must be None, and vice versa."


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
num_qubit_pairs = len(qubit_pairs)
node.namespace["qubit_pairs"] = qubit_pairs

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()


# %% {QUA_program_parameters}
n_avg = node.parameters.num_averages
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type
operation_name = node.parameters.operation
num_operations = node.parameters.number_of_operations
half_span = node.parameters.frame_span / 2
frames = np.arange(-half_span, half_span, node.parameters.frame_span / node.parameters.num_frames)
operation_axis = np.arange(1, num_operations + 1)

gate_refs = {}
for qp in qubit_pairs:
    gate = qp.gates[operation_name]
    gate_refs[qp.name] = {
        "qubit_amplitude": gate.flux_pulse_control.amplitude,
        "coupler_amplitude": gate.coupler_flux_pulse.amplitude,
        "phase_shift_control": gate.phase_shift_control,
        "phase_shift_target": gate.phase_shift_target,
    }
node.namespace["gate_refs"] = gate_refs


# %% {Utility_functions}
def find_phase_from_max_mean(signal: xr.DataArray):
    mean_vs_frame = signal.mean(dim="number_of_operations")
    best_idx = int(mean_vs_frame.argmax(dim="frame").values)
    fitted_phase = float(mean_vs_frame.frame.values[best_idx])
    peak_mean = float(mean_vs_frame.values[best_idx])
    return fitted_phase, peak_mean, mean_vs_frame


def analyze_phase_compensation(ds: xr.Dataset, qubit_pairs):
    fit_datasets = []
    fit_results = {}

    for qp in qubit_pairs:
        try:
            ds_qp = ds.sel(qubit=qp.name)
            signal_control = ds_qp.state_control if "state_control" in ds_qp.data_vars else ds_qp.I_control
            signal_target = ds_qp.state_target if "state_target" in ds_qp.data_vars else ds_qp.I_target

            control_phase, control_peak_mean, control_mean = find_phase_from_max_mean(signal_control)
            target_phase, target_peak_mean, target_mean = find_phase_from_max_mean(signal_target)

            fit_ds = xr.Dataset(
                {
                    "control_mean_vs_frame": control_mean,
                    "target_mean_vs_frame": target_mean,
                    "fitted_control_phase": xr.DataArray(control_phase),
                    "fitted_target_phase": xr.DataArray(target_phase),
                    "control_mean_at_peak": xr.DataArray(control_peak_mean),
                    "target_mean_at_peak": xr.DataArray(target_peak_mean),
                    "success": xr.DataArray(True),
                }
            ).expand_dims(qubit=[qp.name])
            fit_results[qp.name] = {
                "fitted_control_phase": control_phase,
                "fitted_target_phase": target_phase,
                "control_mean_at_peak": control_peak_mean,
                "target_mean_at_peak": target_peak_mean,
                "success": True,
            }
        except Exception as exc:
            fit_ds = xr.Dataset({"success": xr.DataArray(False)}).expand_dims(qubit=[qp.name])
            fit_results[qp.name] = {
                "fitted_control_phase": np.nan,
                "fitted_target_phase": np.nan,
                "control_mean_at_peak": np.nan,
                "target_mean_at_peak": np.nan,
                "success": False,
                "fit_error": str(exc),
            }

        fit_datasets.append(fit_ds)

    return xr.concat(fit_datasets, dim="qubit"), fit_results


def normalize_pair_dataset(ds: xr.Dataset) -> xr.Dataset:
    if "qubit_pair" in ds.coords and "qubit" not in ds.coords:
        ds = ds.rename({"qubit_pair": "qubit"})
    return ds


def convert_pair_IQ_to_V(ds: xr.Dataset, qubit_pairs) -> xr.Dataset:
    """Convert control/target IQ streams to volts for each qubit pair."""
    if "I_control" not in ds.data_vars:
        return ds
    out = ds.copy()
    for var, qubit_attr in [
        ("I_control", "qubit_control"),
        ("Q_control", "qubit_control"),
        ("I_target", "qubit_target"),
        ("Q_target", "qubit_target"),
    ]:
        scales = xr.DataArray(
            [
                2**12 / getattr(qp, qubit_attr).resonator.operations["readout"].length
                for qp in qubit_pairs
            ],
            coords=[("qubit", [qp.name for qp in qubit_pairs])],
        )
        out[var] = out[var] * scales
    return out


def plot_phase_compensation_with_fit(ds_raw: xr.Dataset, qubit_pairs, ds_fit: xr.Dataset):
    """Plot mean signal vs frame for control and target (2 rows per qubit pair)."""
    n_pairs = len(qubit_pairs)
    fig, axes = plt.subplots(2, n_pairs, figsize=(6 * n_pairs, 8), squeeze=False)

    for i, qp in enumerate(qubit_pairs):
        qp_name = qp.name
        if not bool(ds_fit.sel(qubit=qp_name).success.values):
            for row in range(2):
                axes[row, i].text(
                    0.5,
                    0.5,
                    "fit failed",
                    ha="center",
                    va="center",
                    transform=axes[row, i].transAxes,
                )
                axes[row, i].set_title(f"{qp_name} - {'control' if row == 0 else 'target'}")
                axes[row, i].set_xlabel("Frame rotation [2π]")
                axes[row, i].set_ylabel("Mean signal")
            continue

        qp_fit = ds_fit.sel(qubit=qp_name)
        frames = ds_raw.sel(qubit=qp_name).frame.values

        for row, (label, color) in enumerate([("control", "tab:blue"), ("target", "tab:red")]):
            ax = axes[row, i]
            mean_var = f"{label}_mean_vs_frame"
            phase_var = f"fitted_{label}_phase"
            peak_mean_var = f"{label}_mean_at_peak"

            mean_curve = qp_fit[mean_var].values
            fitted_phase = float(qp_fit[phase_var].values)
            peak_mean = float(qp_fit[peak_mean_var].values)

            ax.plot(frames, mean_curve, "o-", color=color)
            ax.axvline(fitted_phase, color="k", ls="--", alpha=0.7, label=f"phase = {fitted_phase:.4f}")
            ax.plot(fitted_phase, peak_mean, "*", color="gold", ms=15, zorder=5, label=f"peak mean = {peak_mean:.4f}")
            ax.set_title(f"{qp_name} - {label}")
            ax.set_xlabel("Frame rotation [2π]")
            ax.set_ylabel("Mean signal")
            ax.grid(alpha=0.3)
            ax.legend(loc="best")

    fig.suptitle("CZ phase compensation - error amplification (max-mean method)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# %% {QUA_program}
# CZ executed at nominal gate parameters (amplitude_scale = 1.0); frame sweep amplifies phase error.
with program() as CZ_1Q_phase_error_amp:
    frame = declare(fixed)
    n = declare(int)
    n_op = declare(int)
    count = declare(int)
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    I_control = [declare(fixed) for _ in range(num_qubit_pairs)]
    Q_control = [declare(fixed) for _ in range(num_qubit_pairs)]
    I_target = [declare(fixed) for _ in range(num_qubit_pairs)]
    Q_target = [declare(fixed) for _ in range(num_qubit_pairs)]
    I_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    Q_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    I_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    Q_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    extra_phase_c = declare(fixed)
    extra_phase_t = declare(fixed)

    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the desired frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(n_op, 1, n_op <= num_operations, n_op + 1):
                with for_(*from_array(frame, frames)):
                    assign(extra_phase_c, qp.gates[operation_name].phase_shift_control)
                    assign(extra_phase_t, qp.gates[operation_name].phase_shift_target)
                    for qubit, state_q, state_st, I, I_st, Q, Q_st in [
                        (
                            qp.qubit_control,
                            state_control[i],
                            state_st_control[i],
                            I_control[i],
                            I_st_control[i],
                            Q_control[i],
                            Q_st_control[i],
                        ),
                        (
                            qp.qubit_target,
                            state_target[i],
                            state_st_target[i],
                            I_target[i],
                            I_st_target[i],
                            Q_target[i],
                            Q_st_target[i],
                        ),
                    ]:
                        # Initialize both qubits before each Ramsey trace.
                        if reset_type == "active":
                            active_reset(qp.qubit_control)
                            qp.align()
                            active_reset(qp.qubit_target)
                            qp.align()
                        else:
                            wait(qp.qubit_control.thermalization_time * u.ns)
                        qp.align()
                        reset_frame(qp.qubit_control.xy.name)
                        reset_frame(qp.qubit_target.xy.name)

                        qubit.xy.play("x90")
                        qp.align()

                        with for_(count, 0, count < n_op, count + 1):
                            if qubit is qp.qubit_control:
                                qp.gates[operation_name].execute_dgx(
                                    phase_shift_control=extra_phase_c + frame
                                )
                            elif qubit is qp.qubit_target:
                                qp.gates[operation_name].execute_dgx(
                                    phase_shift_target=extra_phase_t + frame
                                )
                            qp.align()

                        qubit.xy.play("x90")
                        qp.align()

                        if node.parameters.use_state_discrimination:
                            readout_state(qubit, state_q)
                            save(state_q, state_st)
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I, Q))
                            save(I, I_st)
                            save(Q, Q_st)
                        qp.align()

        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_st_control[i].buffer(len(frames)).buffer(num_operations).average().save(
                    f"state_control{i + 1}"
                )
                state_st_target[i].buffer(len(frames)).buffer(num_operations).average().save(
                    f"state_target{i + 1}"
                )
            else:
                I_st_control[i].buffer(len(frames)).buffer(num_operations).average().save(f"I_control{i + 1}")
                Q_st_control[i].buffer(len(frames)).buffer(num_operations).average().save(f"Q_control{i + 1}")
                I_st_target[i].buffer(len(frames)).buffer(num_operations).average().save(f"I_target{i + 1}")
                Q_st_target[i].buffer(len(frames)).buffer(num_operations).average().save(f"Q_target{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)
    job = qmm.simulate(config, CZ_1Q_phase_error_amp, simulation_config)
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(CZ_1Q_phase_error_amp)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)


# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_results_as_xarray(
            job.result_handles,
            qubit_pairs,
            {"frame": frames, "number_of_operations": operation_axis},
        )
        ds = normalize_pair_dataset(ds)
        node.results = {"ds": ds, "gate_refs": gate_refs}
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = normalize_pair_dataset(node.results["ds"])
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
        operation_name = node.parameters.operation
        gate_refs = {}
        for qp in qubit_pairs:
            gate = qp.gates[operation_name]
            gate_refs[qp.name] = {
                "qubit_amplitude": gate.flux_pulse_control.amplitude,
                "coupler_amplitude": gate.coupler_flux_pulse.amplitude,
                "phase_shift_control": gate.phase_shift_control,
                "phase_shift_target": gate.phase_shift_target,
            }
        node.namespace["qubit_pairs"] = qubit_pairs
        node.namespace["gate_refs"] = gate_refs
        node.results = {"ds": ds, "gate_refs": gate_refs}

    if not node.parameters.use_state_discrimination:
        ds = convert_pair_IQ_to_V(ds, qubit_pairs)
        node.results["ds"] = ds

    # %% {Data_analysis}
    ds_fit, fit_results = analyze_phase_compensation(ds, qubit_pairs)
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = fit_results

    for qp in qubit_pairs:
        fit_result = fit_results[qp.name]
        if fit_result["success"]:
            print(
                f"{qp.name}: control phase correction = {fit_result['fitted_control_phase']:.6f}, "
                f"target phase correction = {fit_result['fitted_target_phase']:.6f}"
            )
        else:
            print(f"{qp.name}: 1Q phase compensation fit failed: {fit_result['fit_error']}")

    # %% {Plotting}
    fig_phase = plot_phase_compensation_with_fit(ds, qubit_pairs, ds_fit)
    plt.show()
    node.results["figure_phase"] = fig_phase

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qp in qubit_pairs:
                if not fit_results[qp.name]["success"]:
                    continue
                qp.gates[operation_name].phase_shift_control = (
                    qp.gates[operation_name].phase_shift_control + fit_results[qp.name]["fitted_control_phase"]
                ) % 1.0
                qp.gates[operation_name].phase_shift_target = (
                    qp.gates[operation_name].phase_shift_target + fit_results[qp.name]["fitted_target_phase"]
                ) % 1.0

    # %% {Save_results}
    node.outcomes = {
        qp.name: ("successful" if fit_results[qp.name]["success"] else "failed") for qp in qubit_pairs
    }
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%
