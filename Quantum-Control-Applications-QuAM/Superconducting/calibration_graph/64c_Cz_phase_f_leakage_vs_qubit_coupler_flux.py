# %%
"""
Calibration of the Controlled-Phase (CPhase) of the CZ Gate

This sequence calibrates the CPhase of the CZ gate by scanning the pulse amplitude and measuring the resulting phase of the target qubit. The calibration compares two scenarios:

1. Control qubit in the ground state
2. Control qubit in the excited state

For each amplitude, we measure:
1. The phase difference of the target qubit between the two scenarios
2. The amount of leakage to the |f> state when the control qubit is in the excited state

The calibration process involves:
1. Applying a CZ gate with varying amplitudes
2. Measuring the phase of the target qubit for both control qubit states
3. Calculating the phase difference
4. Measuring the population in the |f> state to quantify leakage

The optimal CZ gate amplitude is determined by finding the point where:
1. The phase difference is closest to π (0.5 in normalized units)
2. The leakage to the |f> state is minimized

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits
- Initial estimate of the CZ gate amplitude

Outcomes:
- Optimal CZ gate amplitude for achieving a π phase shift
- Leakage characteristics across the amplitude range
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef, active_reset_simple
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
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
import matplotlib.pyplot as plt
import numpy as np
import warnings
from qualang_tools.bakery import baking
from quam_libs.lib.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse

# %% {Node_parameters}
qubit_pair_indexes = [2]  # The indexes of the qubit pair to calibrate


class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s" % (i, i + 1) for i in qubit_pair_indexes]
    """List of qubit pair names to calibrate. If None or empty, all active qubit pairs will be used."""
    num_averages: int = 20
    """Number of averages to perform. Default is 100."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    """Flux point setting strategy: 'joint' or 'independent'. Default is 'joint'."""
    reset_type: Literal["active", "thermal"] = "active"
    """Type of reset to use between experiments. Options are 'active' or 'thermal'. Default is 'active'."""
    simulate: bool = False
    """If True, simulates the QUA program instead of executing it on hardware. Default is False."""
    timeout: int = 200
    """Timeout for the QOP session in seconds. Default is 100 seconds."""
    coupler_flux_min : float = -0.086 #relative to the coupler set point
    """Minimum of the coupler flux sweep"""
    coupler_flux_max : float =  -0.074 #relative to the coupler set point
    """Maximum of the coupler flux sweep"""
    coupler_flux_num_points : int = 51
    """Length of the flux sweep for coupler fluxes."""    
    qubit_flux_min: float = -0.02  #relative to the qubit pair detuning
    """Minimum of the qubit flux sweep"""
    qubit_flux_max: float = 0.01  #relative to the qubit pair detuning
    """Maximum of the qubit flux sweep"""
    qubit_flux_num_points : int = 51
    """Length of the flux sweep for qubit fluxes."""
    num_frames: int = 10
    """Number of frames to sample the oscillation."""
    load_data_id: Optional[int] = None  # 92417
    """If provided, loads data from a previous calibration with this ID instead of executing the experiment."""
    plot_raw: bool = False
    """If True, plots the raw data after fetching."""
    measure_leak: bool = True
    """If True, measures leakage to the |f> state of the control qubit."""
    operation: Literal["Cz_unipolar", "Cz_flattop", "Cz_bipolar", "Cz_slepian", "Cz_slepian_flattop"] = "Cz_unipolar"
    """Type of CZ operation to perform."""



node = QualibrationNode(name="64c_Cz_phase_f_leakage_vs_qubit_coupler_flux", parameters=Parameters())
assert not (
    node.parameters.simulate and node.parameters.load_data_id is not None
), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

num_qubit_pairs = len(qubit_pairs)

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

####################
# Helper functions #
####################


def tanh_fit(x, a, b, c, d):
    return a * np.tanh(b * x + c) + d


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

# Loop parameters
# Loop parameters
fluxes_coupler = np.linspace(node.parameters.coupler_flux_min, node.parameters.coupler_flux_max+0.0001, node.parameters.coupler_flux_num_points)

fluxes_qubit = np.linspace(node.parameters.qubit_flux_min,node.parameters.qubit_flux_max , node.parameters.qubit_flux_num_points)
fluxes_qp = {}
for qp in qubit_pairs:
    # estimate the flux shift to get the control qubit to the target qubit frequency
    fluxes_qp[qp.name] = fluxes_qubit + qp.detuning
frames = np.arange(0, 1, 1 / node.parameters.num_frames)
operation_name = node.parameters.operation

with program() as CPhase_Oscillations:
    amp = declare(fixed)
    frame = declare(fixed)
    control_initial = declare(int)
    flux_coupler = declare(float)
    flux_qubit = declare(float)
    comp_flux_qubit = declare(float)
    n = declare(int)
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]

    for i, qp in enumerate(qubit_pairs):
        qp.gates["Cz"].phase_shift_control = 0.0
        qp.gates["Cz"].phase_shift_target = 0.0
        # Bring the active qubits to the minimum frequency point
        if not node.parameters.simulate:
            if flux_point == "independent":
                machine.apply_all_flux_to_min()
                # qp.apply_mutual_flux_point()
            elif flux_point == "joint":
                machine.apply_all_flux_to_joint_idle()
            else:
                machine.apply_all_flux_to_zero()
            wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(flux_coupler, fluxes_coupler)):
                with for_(*from_array(flux_qubit, fluxes_qp[qp.name])):
                    with for_(*from_array(frame, frames)):
                        with for_(*from_array(control_initial, [0, 1])):
                            # reset
                            if not node.parameters.simulate:
                                if node.parameters.reset_type == "active":
                                    active_reset_gef(qp.qubit_control)
                                    active_reset(qp.qubit_target)
                                    # active_reset_simple(qp.qubit_control)
                                    # active_reset_simple(qp.qubit_target)
                                else:
                                    wait(qp.qubit_control.thermalization_time * u.ns)
                            qp.align()
                            if "coupler_qubit_crosstalk" in qp.extras:
                                assign(comp_flux_qubit, flux_qubit + qp.extras["coupler_qubit_crosstalk"] * flux_coupler)
                            else:
                                print("No crosstalk compensated")
                                assign(comp_flux_qubit, flux_qubit)

                            reset_frame(qp.qubit_target.xy.name)
                            reset_frame(qp.qubit_control.xy.name)
                            # setting both qubits ot the initial state
                            # qp.qubit_control.xy.play("x180", condition=control_initial==1)
                            with if_(control_initial == 1):
                                qp.qubit_control.xy.play("x180")
                            qp.qubit_target.xy.play("x90")
                            qp.align()

                            # play the CZ gate
                            qp.gates[operation_name].execute(
                                amplitude_scale=comp_flux_qubit / qp.gates[operation_name].flux_pulse_control.amplitude, 
                                coupler_amplitude_scale=flux_coupler/qp.gates[operation_name].coupler_flux_pulse.amplitude
                                )
                            align()
                            # rotate the frame
                            frame_rotation_2pi(frame, qp.qubit_target.xy.name)

                            # return the target qubit before measurement
                            qp.qubit_target.xy.play("x90")

                            # measure both qubits
                            if node.parameters.measure_leak:
                                readout_state_gef(qp.qubit_control, state_control[i])
                            else:
                                readout_state(qp.qubit_control, state_control[i])
                            readout_state(qp.qubit_target, state_target[i])
                            save(state_control[i], state_st_control[i])
                            save(state_target[i], state_st_target[i])
        align()

    with stream_processing():
        n_st.save("n")
        state_st_control[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).buffer(
            n_avg
        ).save(f"state_control{i + 1}")
        state_st_target[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).buffer(
            n_avg
        ).save(f"state_target{i + 1}")

# %% {Simulate_or_execute
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000 // 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)

    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.results = {"figure": plt.gcf()}
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(CPhase_Oscillations)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(
            job.result_handles,
            qubit_pairs,
            {
                "control_axis": [0, 1],
                "frame": frames,
                "flux_qubit": fluxes_qubit,
                "flux_coupler": fluxes_coupler,
                "N": np.linspace(1, n_avg, n_avg),
            },
        )
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
    node.results = {"ds": ds}

# %% {Data prcessing}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        detuning = np.array(
            [-fluxes_qp[qp.name] ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term for qp in qubit_pairs]
        )
        ds = ds.assign_coords({"detuning": (["qubit", "flux_qubit"], detuning)})
        flux_coupler_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
        ds = ds.assign_coords({"flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)})
        flux_qubit_full = np.array([fluxes_qp[qp.name] for qp in qubit_pairs])
        ds = ds.assign_coords({"flux_qubit_full": (["qubit", "flux_qubit"], flux_qubit_full)})
    else:
        if "detuning" not in ds.coords or "flux_coupler_full" not in ds.coords or "flux_qubit_full" not in ds.coords:
            fluxes_coupler = np.linspace(
                node.parameters.coupler_flux_min,
                node.parameters.coupler_flux_max + 0.0001,
                node.parameters.coupler_flux_num_points,
            )
            fluxes_qubit = np.linspace(
                node.parameters.qubit_flux_min,
                node.parameters.qubit_flux_max,
                node.parameters.qubit_flux_num_points,
            )
            fluxes_qp = {qp.name: fluxes_qubit + qp.detuning for qp in qubit_pairs}
            if "detuning" not in ds.coords:
                detuning = np.array(
                    [-fluxes_qp[qp.name] ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term for qp in qubit_pairs]
                )
                ds = ds.assign_coords({"detuning": (["qubit", "flux_qubit"], detuning)})
            if "flux_coupler_full" not in ds.coords:
                flux_coupler_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
                ds = ds.assign_coords({"flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)})
            if "flux_qubit_full" not in ds.coords:
                flux_qubit_full = np.array([fluxes_qp[qp.name] for qp in qubit_pairs])
                ds = ds.assign_coords({"flux_qubit_full": (["qubit", "flux_qubit"], flux_qubit_full)})


# %% # %% Data analysis
import xarray as xr
node.results["results"] = {}
if not node.parameters.simulate:
    for qp in qubit_pairs:
        fit_data = fit_oscillation(ds.state_target.mean(dim="N"), "frame")
        ds = ds.assign(
            {
                "fitted": oscillation(
                    ds.frame,
                    fit_data.sel(fit_vals="a"),
                    fit_data.sel(fit_vals="f"),
                    fit_data.sel(fit_vals="phi"),
                    fit_data.sel(fit_vals="offset"),
                )
            }
        )

        phase = fix_oscillation_phi_2pi(fit_data)
        phase_diff = ((phase.sel(control_axis=0) - phase.sel(control_axis=1)) % 1).rename("phase_diff")
        ds = xr.merge([ds, phase_diff])

        try:
            flux_coupler_opt = qp.gates[operation_name].coupler_flux_pulse.amplitude 
            fc_idx = np.abs(ds.flux_coupler - flux_coupler_opt).argmin(dim="flux_coupler")
            phase_line = phase_diff.sel(qubit=qp.name).isel(flux_coupler=fc_idx)
            fq_idx = np.abs(phase_line - 0.5).argmin(dim="flux_qubit")
            optimal_qubit_flux_shift = ds.sel(qubit=qp.name).flux_qubit_full.isel(flux_qubit=fq_idx).item()
            print(f"Optimal coupler flux shift for {qp.name}: {flux_coupler_opt*1e3:.2f} mV")
            print(f"Optimal qubit flux shift for {qp.name}: {optimal_qubit_flux_shift*1e3:.2f} mV")
        except:
            print(f"Fitting failed for {qp.name}")
            

        node.results["results"][qp.name] = {
            "optimal_qubit_flux_shift": optimal_qubit_flux_shift,
            "optimal_coupler_flux_shift": flux_coupler_opt,}

        if node.parameters.measure_leak:
            all_counts = (ds.state_control < 3).sum(dim="N").sel(control_axis=1).sum(dim="frame")
            leak_counts = (ds.state_control == 2).sum(dim="N").sel(control_axis=1).sum(dim="frame")
            leakage = leak_counts / all_counts
            ds = xr.merge([ds, leakage.rename("leakage")])

# %% Plotting
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp in grid_iter(grid):
        qp_name = qp["qubit"]
        qubit_pair = machine.qubit_pairs[qp_name]
        leakage = ds.leakage.sel(qubit=qp_name)
        values_to_plot = leakage
        values_to_plot.assign_coords(
            {
                "flux_qubit_mV": 1e3 * values_to_plot.flux_qubit_full,
                "flux_coupler_mV": 1e3 * values_to_plot.flux_coupler,
            }
        ).plot(ax=ax, cmap="viridis", x="flux_qubit_mV", y="flux_coupler_mV")
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize=10)
        # --- Optional analysis plotted if it exists ---
        legend_entries = []
        try:
            res = node.results["results"].get(qp_name, {})
            # Extract (may be missing or NaN)
            optimal_qubit_flux_shift_mV      = 1e3 * res.get("optimal_qubit_flux_shift", np.nan)
            optimal_coupler_flux_shift_mV = 1e3 * res.get("optimal_coupler_flux_shift", np.nan)
            if np.isfinite(optimal_qubit_flux_shift_mV):
                ax.axvline(optimal_qubit_flux_shift_mV, color="black", lw=1.0, ls=":")
            if np.isfinite(optimal_coupler_flux_shift_mV):
                ax.axhline(optimal_coupler_flux_shift_mV, color="black", lw=1.0, ls=":")
            if np.isfinite(optimal_qubit_flux_shift) and np.isfinite(optimal_coupler_flux_shift_mV):
                ax.plot(
                    optimal_qubit_flux_shift_mV,
                    optimal_coupler_flux_shift_mV,
                    marker="+",
                    color="blue",
                    markersize=10,
                    mew=2.0,
                    label="Optimal",
                )
                legend_entries.append("Optimal")

        except Exception as e:
            print(f"[WARN] Annotations failed for {qp_name}: {e}")
       
        try:
            sel = ds.sel(qubit=qp_name)
            flux_qubit_data = (sel.flux_qubit_full.values * 1e3).ravel()
            detuning_data   = (sel.detuning.values * 1e-6).ravel()

            mask = np.isfinite(flux_qubit_data) & np.isfinite(detuning_data)
            flux_qubit_data = flux_qubit_data[mask]
            detuning_data   = detuning_data[mask]

            order = np.argsort(flux_qubit_data)
            x_sorted = flux_qubit_data[order]
            y_sorted = detuning_data[order]

            x_unique, unique_idx = np.unique(x_sorted, return_index=True)
            y_unique = y_sorted[unique_idx]
            
            def flux_to_detuning(x):
                return np.interp(np.asarray(x), x_unique, y_unique)

            def detuning_to_flux(y):
                return np.interp(np.asarray(y), y_unique, x_unique)

            sec_ax = ax.secondary_xaxis("top", functions=(flux_to_detuning, detuning_to_flux))
            sec_ax.set_xlabel("Detuning [MHz]")

        except Exception as e:
            print(f"[WARN] Secondary axis failed for {qp_name}: {e}")

    
        ax.set_xlabel("Qubit flux shift [mV]")
        ax.set_ylabel("Coupler flux shift [mV]")
        if legend_entries:
            ax.legend(fontsize=7, loc="upper right", frameon=True)
    grid.fig.suptitle(f"Leakage to control f state  \n {operation_name}")
    plt.tight_layout()
    plt.show()
    node.results["figure_leakage"] = grid.fig

    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp in grid_iter(grid):
        qp_name = qp["qubit"]
        phase_diff = ds.phase_diff.sel(qubit=qp_name)
        values_to_plot = phase_diff

        img = values_to_plot.assign_coords(
            {
                "flux_qubit_mV": 1e3 * values_to_plot.flux_qubit_full,
                "flux_coupler_mV": 1e3 * values_to_plot.flux_coupler,
            }
        ).plot(ax=ax, x="flux_qubit_mV", y="flux_coupler_mV", cmap="RdBu_r", add_colorbar=True)

        # img is a xarray plot object → it contains the colorbar
        cbar = img.colorbar
        cbar.set_label("Phase diff (in unitns of 2π)")
        qubit_pair = machine.qubit_pairs[qp_name]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset*1e3:.0f}mV", fontsize=10)
        # --- Optional analysis plotted if it exists ---
        legend_entries = []
        try:
            res = node.results["results"].get(qp_name, {})
            # Extract (may be missing or NaN)
            optimal_qubit_flux_shift_mV      = 1e3 * res.get("optimal_qubit_flux_shift", np.nan)
            optimal_coupler_flux_shift_mV = 1e3 * res.get("optimal_coupler_flux_shift", np.nan)
            if np.isfinite(optimal_qubit_flux_shift_mV):
                ax.axvline(optimal_qubit_flux_shift_mV, color="black", lw=1.0, ls=":")
            if np.isfinite(optimal_coupler_flux_shift_mV):
                ax.axhline(optimal_coupler_flux_shift_mV, color="black", lw=1.0, ls=":")
            if np.isfinite(optimal_qubit_flux_shift) and np.isfinite(optimal_coupler_flux_shift_mV):
                ax.plot(
                    optimal_qubit_flux_shift_mV,
                    optimal_coupler_flux_shift_mV,
                    marker="+",
                    color="blue",
                    markersize=10,
                    mew=2.0,
                    label="Optimal",
                )
                legend_entries.append("Optimal")

        except Exception as e:
            print(f"[WARN] Annotations failed for {qp_name}: {e}")
        
        try:
            sel = ds.sel(qubit=qp_name)
            flux_qubit_data = (sel.flux_qubit_full.values * 1e3).ravel()
            detuning_data   = (sel.detuning.values * 1e-6).ravel()

            mask = np.isfinite(flux_qubit_data) & np.isfinite(detuning_data)
            flux_qubit_data = flux_qubit_data[mask]
            detuning_data   = detuning_data[mask]

            order = np.argsort(flux_qubit_data)
            x_sorted = flux_qubit_data[order]
            y_sorted = detuning_data[order]

            x_unique, unique_idx = np.unique(x_sorted, return_index=True)
            y_unique = y_sorted[unique_idx]
            
            def flux_to_detuning(x):
                return np.interp(np.asarray(x), x_unique, y_unique)

            def detuning_to_flux(y):
                return np.interp(np.asarray(y), y_unique, x_unique)

            sec_ax = ax.secondary_xaxis("top", functions=(flux_to_detuning, detuning_to_flux))
            sec_ax.set_xlabel("Detuning [MHz]")

        except Exception as e:
            print(f"[WARN] Secondary axis failed for {qp_name}: {e}")
        ax.set_xlabel("Qubit flux shift [mV]")
        ax.set_ylabel("Coupler flux shift [mV]")
        if legend_entries:
            ax.legend(fontsize=7, loc="upper right", frameon=True)
    grid.fig.suptitle(f"Conditional phase $\phi$  \n {operation_name}")
    plt.tight_layout()
    plt.minorticks_on()
    plt.show()
    node.results["figure_phase"] = grid.fig

# %% {Update_state}
if not node.parameters.simulate and node.parameters.load_data_id is None:
    with node.record_state_updates():
        for qp in qubit_pairs:
            qp.extras["CZ_qubit_flux"] = node.results["results"][qp.name]["optimal_qubit_flux_shift"]
            qp.gates[operation_name].flux_pulse_control.amplitude = node.results["results"][qp.name]["optimal_qubit_flux_shift"]
# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.results["ds"] = ds
    node.save()

# %%
