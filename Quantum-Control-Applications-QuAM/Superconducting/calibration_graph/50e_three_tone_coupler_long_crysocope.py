"""
        RESONATOR SPECTROSCOPY VERSUS FLUX
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various readout intermediate dfs and flux biases.
The resonator frequency as a function of flux bias is then extracted and fitted so that the parameters can be stored in the state.

This information can then be used to adjust the readout frequency for the maximum and minimum frequency points.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the relevant flux biases in the state.
    - Save the current state
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset_simple, readout_state
from quam_libs.lib.qua_datasets import convert_IQ_to_V
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
import numpy as np
from dataclasses import asdict
from typing import Dict, List
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
from typing import List, Tuple, Optional, Any

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["coupler_q2_q3"]
    """List of qubit pair names to be measured. If None or empty, all active qubit pairs are measured."""
    num_averages: int = 50
    """Number of averages for the measurement."""
    frequency_span_in_mhz: float = 400
    """Frequency span around the coupler RF frequency for the scan."""
    frequency_step_in_mhz: float = 1.0
    """Frequency step size for the scan."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    """Whether to set flux point jointly for all qubits or independently."""
    duration_in_ns: Optional[int] = 300_000
    """Total duration of the flux pulse in ns."""
    time_axis: Literal["linear", "log"] = "linear"
    """Type of time axis for the flux pulse duration sweep."""
    time_step_num: Optional[int] = 101 
    """Number of time steps for logarithmic time axis."""
    min_wait_time_in_ns: Optional[int] = 16
    """Minimum wait time in ns for the flux pulse duration sweep."""
    use_state_discrimination: bool = False
    """Whether to use state discrimination in readout. Defaults to False."""
    coupler_flux: float = 0.09
    """Coupler flux value set  """
    detuning_in_mhz: Optional[float] = 0.0
    """Detuning of the coupler from its RF frequency in MHz."""
    control_drive_operation: Literal["x180_Square", "x180"] = "x180_Square"
    """Type of control qubit drive operation."""
    control_pulse_duration_in_ns: int = 500 
    """Duration of the control qubit pulse in ns."""
    control_pulse_amplitude: float = 0.08
    """Amplitude scale for the control qubit pulse."""
    target_drive_operation: str = "saturation"
    """Type of operation to perform on the target qubit (e.g., "saturation", "x180"). Defaults to "saturation"."""
    target_pulse_amplitude: Optional[float] = 0.02  # 0.05  # 0.004, 0.02
    """Relative amplitude factor for the target drive pulse. Defaults to 0.005."""
    target_pulse_duration_in_ns: Optional[int] = 400
    """Duration of the target qubit pulse in nanoseconds. Defaults to 1000 ns."""
    simulate: bool = False
    """Whether to simulate the QUA program instead of executing it."""
    simulation_duration_ns: int = 2500
    """Duration of the simulation in ns."""
    timeout: int = 100
    """Timeout for the QOP session in seconds."""
    load_data_id: Optional[int] = None
    """If provided, load data from the specified node ID instead of executing the program."""
    reset_type: Literal["active", "thermal"] = "thermal"
    """Type of qubit reset to use: 'active' or 'thermal'."""
    wait_extra_time: Optional[bool] = True
    """Whether to wait extra time after flux pulse before control pulse."""
    fitting_base_fractions: List[float] = [0.4, 0.15, 0.05]


node = QualibrationNode(name="50e_three_tone_coupler_long_crysocope", parameters=Parameters())


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
qubit_pair_names = [qp.name for qp in qubit_pairs]

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
# dfs = np.arange(-span / 2, +span / 2, step)
dfs = np.arange(-180e6, -150e6, step)  # Fixing the frequency range for better fitting

# Flux bias sweep
if node.parameters.time_axis == "linear":
    times = np.linspace(
        node.parameters.min_wait_time_in_ns // 4,
        node.parameters.duration_in_ns // 4,
        node.parameters.time_step_num,
        dtype=np.int32,
    )
elif node.parameters.time_axis == "log":
    times = np.logspace(
        np.log10(node.parameters.min_wait_time_in_ns // 4),
        np.log10(node.parameters.duration_in_ns // 4),
        node.parameters.time_step_num,
        dtype=np.int32,
    )
    # Remove repetitions from times
times = np.unique(times)

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

detuning = node.parameters.detuning_in_mhz * 1e6
coupler_IFs = {
    qp.name: qp.coupler.RF_frequency - detuning - qp.qubit_control.xy.opx_output.upconverter_frequency
    for qp in qubit_pairs
}

with program() as multi_res_spec_vs_flux:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubit_pairs)
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_stream_target = [declare_stream() for _ in range(num_qubit_pairs)]
    df = declare(int)  # QUA variable for the readout frequency
    t_delay = declare(int)  # QUA variable for delay time scan
    duration = node.parameters.duration_in_ns * u.ns

    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit_pairs[0].qubit_target)
        for qp in qubit_pairs:
            qp.coupler.set_dc_offset(0.0)
            wait(1000)
        align()

    align()
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        for i, qp in enumerate(qubit_pairs):
            with for_(*from_array(df, dfs)):  # type: ignore
                with for_each_(t_delay, times):
                    # Qubit initialization
                    qubit_control = qp.qubit_control
                    qubit_target = qp.qubit_target

                    # Update the qubit frequency
                    # qubit_control.xy.update_frequency(qubit_control.xy.intermediate_frequency)
                    
                    if node.parameters.reset_type == "active":
                        active_reset_simple(qubit_control)
                        active_reset_simple(qubit_target)
                        qp.align()

                    else:
                        qubit_control.wait(qubit_control.thermalization_time * u.ns)
                        qubit_target.wait(qubit_target.thermalization_time * u.ns)
                        qp.align()

                    if node.parameters.wait_extra_time:
                        qubit_control.xy.wait(node.parameters.duration_in_ns // 4)
                    qp.align()
                    
                    # update the frequency of the control qubit
                    qubit_control.xy.update_frequency(df + coupler_IFs[qp.name])
                    
                    target_pulse_duration = (
                        node.parameters.target_pulse_duration_in_ns * u.ns
                        if node.parameters.target_pulse_duration_in_ns is not None
                        else qubit_target.xy.operations[node.parameters.target_drive_operation].length * u.ns
                    )
                    
                    control_pulse_duration = (
                        node.parameters.control_pulse_duration_in_ns * u.ns
                        if node.parameters.control_pulse_duration_in_ns is not None
                        else qubit_control.xy.operations[node.parameters.control_drive_operation].length * u.ns
                    )
                    qp.align()
                    
                    qp.coupler.play(
                        "const",
                        amplitude_scale=node.parameters.coupler_flux / qp.coupler.operations["const"].amplitude,
                        duration=t_delay + 400,
                    )
                    qubit_control.xy.wait(t_delay)
                    
                    qubit_control.xy.play(
                        node.parameters.control_drive_operation,
                        amplitude_scale=node.parameters.control_pulse_amplitude,
                        duration=control_pulse_duration,
                    )
                    
                    qp.align()

                    qubit_target.xy.play(
                        node.parameters.target_drive_operation,
                        amplitude_scale=node.parameters.target_pulse_amplitude,
                        duration=target_pulse_duration,
                    )
                    qp.align()

                    # Measure target qubit and save data
                    if node.parameters.use_state_discrimination:
                        readout_state(qp.qubit_target, state_target[i])
                        save(state_target[i], state_stream_target[i])
                    else:
                        qp.qubit_target.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

        # Measure sequentially
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_stream_target[i].buffer(len(times)).buffer(len(dfs)).average().save(f"state_target{i + 1}")
            else:
                I_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"Q{i + 1}")
        


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_flux, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(multi_res_spec_vs_flux)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    if node.parameters.load_data_id is not None:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
    else:
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"time": times * 4, "detuning": dfs})

    # %% {Process_raw}
    if not node.parameters.use_state_discrimination:
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
    if node.parameters.load_data_id is None:
        detuning_axis = dfs
        RF_freq = np.array([detuning_axis + qp.coupler.RF_frequency - detuning for qp in qubit_pairs])
        ds = ds.assign_coords({"freq_full_control": (["qubit", "detuning"], RF_freq)})
        detuned_freq = np.array([detuning_axis - detuning for qp in qubit_pairs]) * 1e-6
        ds = ds.assign_coords({"detunings": (["qubit", "detuning"], detuned_freq)})
    elif "freq_full_control" not in ds.coords or "detunings" not in ds.coords:
        detuning_axis = ds.coords["detuning"].values
        detuning_hz = node.parameters.detuning_in_mhz * 1e6
        RF_freq = np.array([detuning_axis + qp.coupler.RF_frequency - detuning_hz for qp in qubit_pairs])
        ds = ds.assign_coords({"freq_full_control": (["qubit", "detuning"], RF_freq)})
        detuned_freq = np.array([detuning_axis - detuning_hz for qp in qubit_pairs]) * 1e-6
        ds = ds.assign_coords({"detunings": (["qubit", "detuning"], detuned_freq)})
    else:
        detuning_axis = ds.coords["detuning"].values
    ds.freq_full_control.attrs["long_name"] = "Frequency"
    ds.freq_full_control.attrs["units"] = "GHz"
    ds.detunings.attrs["long_name"] = "Detuning"
    ds.detunings.attrs["units"] = "MHz"

    # %% {Analyze_data}
    import xarray as xr
    from calibration_utils.pi_flux.analysis import fit_gaussian, PiFluxParameters, optimize_start_fractions
    
    node.parameters.fitting_base_fractions = [0.4, 0.1, 0.005]
    if node.parameters.use_state_discrimination and "state_target" in ds.data_vars:
        state_da = ds["state_target"].transpose("qubit", "time", "detuning")
        center_freqs = xr.apply_ufunc(
            lambda states: fit_gaussian(detuning_axis, states),
            state_da,
            input_core_dims=[["detuning"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
    else:
        # center_freqs = extract_center_freqs_iq(ds, dfs)
        stacked = ds.IQ_abs.transpose("qubit", "time", "detuning")
        center_freqs = xr.apply_ufunc(
            lambda iq_slice: fit_gaussian(detuning_axis, iq_slice),
            stacked,
            input_core_dims=[["detuning"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
    
    # Add flux-induced frequency shift to center frequencies
    center_freqs = center_freqs - node.parameters.detuning_in_mhz * 1e6
    
    # Calculate flux response from frequency shifts
    flux_response = np.sqrt(-1*center_freqs )
    flux_response_normalized = flux_response / flux_response.isel(time = slice(0, 5)).mean(dim = "time")
    # Store results in dataset
    ds['center_freqs'] = center_freqs
    ds['flux_response'] = flux_response
    ds['flux_response_normalized'] = flux_response_normalized
    
    fit_results = {}
    for q in qubit_pairs:
        t_data = flux_response_normalized.sel(qubit=q.name).time.values
        y_data = flux_response_normalized.sel(qubit=q.name).values
        fit_successful, best_fractions, best_components, best_a_dc, best_rms = optimize_start_fractions(
            t_data, y_data, node.parameters.fitting_base_fractions, bounds_scale=0.5, a_dc=flux_response_normalized.min().values
        )
        fit_results[q.name] = PiFluxParameters(
            fit_successful=fit_successful,
            optimized_fractions=best_fractions,
            a_tau_tuple=best_components,
            a_dc=best_a_dc,
            rms_error=best_rms,
        )
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    
    #%% {Plotting}

    def plot_fit(ds: xr.Dataset, qubits: List[Any], fit_results: Dict):
        """
        Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset containing the quadrature data.
        qubits : list of AnyTransmon
            A list of qubits to plot.
        fits : xr.Dataset
            The dataset containing the fit parameters.

        Returns
        -------
        Figure
            The matplotlib figure object containing the plots.

        Notes
        -----
        - The function creates a grid of subplots, one for each qubit.
        - Each subplot contains the raw data and the fitted curve.
        """
        # grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for q in qubits:
            t_data = ds.time.values
            y_data = ds.flux_response_normalized.sel(qubit=q.name).values

            components = fit_results[q.name]["a_tau_tuple"]
            a_dc = fit_results[q.name]["a_dc"]
            if a_dc is None or (isinstance(a_dc, (float, np.floating)) and np.isnan(a_dc)):
                a_dc = float(y_data[-5:].mean()) if len(y_data) >= 5 else float(y_data.mean())

            fig, _ = plot_individual_fit(
                t_data, y_data, components=components, a_dc=a_dc,
                qubit_name=q.name, ds=ds
            )

        return fig


    def plot_individual_fit(t_data: np.ndarray, y_data: np.ndarray,
                            components: List[Tuple[float, float]],
                            a_dc: float, qubit_name: str,
                            ds: Optional[xr.Dataset] = None):
        """Plot exponential fit results plus dataset amplitude (IQ_abs or I)."""

        # --- Build fit ---
        fit_text = f"a_dc = {a_dc:.3f}\n"
        y_fit = np.ones_like(t_data, dtype=float) * a_dc
        for i, (amp, tau) in enumerate(components):
            y_fit += amp * np.exp(-t_data / tau)
            fit_text += f"a{i + 1} = {amp / a_dc:.3f}, τ{i + 1} = {tau:.0f}ns\n"

        # --- Create grid layout: top full-width, bottom with 2 plots ---
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2)
        fig.suptitle(f"Long time Cryoscope: {qubit_name}")

        # --- Top plot: raw dataset (spans both columns) ---
        ax_top = fig.add_subplot(gs[0, :])  # span both columns
        if ds is not None:
            if "IQ_abs" in ds.data_vars:
                ds.IQ_abs.sel(qubit=qubit_name).plot(ax=ax_top, label="IQ_abs", cmap = "viridis")
            elif "I" in ds.data_vars:
                ds.I.sel(qubit=qubit_name).plot(ax=ax_top, label="I", cmap = "viridis")
            elif "state_target" in ds.data_vars:
                ds.state_target.sel(qubit=qubit_name).plot(ax=ax_top, label="State Target", cmap = "viridis")
            else:
                ax_top.text(0.5, 0.5, "No IQ_abs or I found",
                            ha="center", va="center", transform=ax_top.transAxes)
        ax_top.plot(ds.sel(qubit=qubit_name).time.values, ds.sel(qubit=qubit_name).center_freqs.values, "r-", label="Peak frequencies", linewidth=2)
        ax_top.set_title("Raw Dataset")
        ax_top.set_xlabel("Time [ns]")
        ax_top.set_ylabel("Detuning [Hz]")
        ax_top.legend()


        # --- Bottom-left: linear fit ---
        ax_lin = fig.add_subplot(gs[1, 0])
        ax_lin.plot(t_data, y_data, ".--", label="Data")
        ax_lin.plot(t_data, y_fit, label="Fit")
        ax_lin.text(0.98, 0.5, fit_text, transform=ax_lin.transAxes,
                    fontsize=10, ha="right", va="center")
        ax_lin.set_xlabel("Time (ns)")
        ax_lin.set_ylabel("Flux Response")
        ax_lin.legend()
        ax_lin.grid(True)
        ax_lin.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        # --- Bottom-right: log fit ---
        ax_log = fig.add_subplot(gs[1, 1])
        ax_log.plot(t_data, y_data, ".--", label="Data")
        ax_log.plot(t_data, y_fit, label="Fit")
        ax_log.text(0.98, 0.5, fit_text, transform=ax_log.transAxes,
                    fontsize=10, ha="right", va="center")
        ax_log.set_xlabel("Time (ns)")
        ax_log.set_ylabel("Flux Response")
        ax_log.set_xscale("log")
        ax_log.set_yscale("log")
        ax_log.legend(loc="best")
        ax_log.grid(True)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig, [ax_top, ax_lin, ax_log]


    
    fig = plot_fit(ds, qubit_pairs, node.results.get("fit_results"))
    plt.show()
    node.results["fitted_data"] = fig

    # %% {Update_state}

    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for i, qp in enumerate(qubit_pairs):
                z_out = qubit_pairs[i].coupler.opx_output
            if z_out.exponential_filter is None:
                z_out.exponential_filter = []
        
        with node.record_state_updates():
            for i, qp in enumerate(qubit_pairs):
                res = node.results["fit_results"][qp.name]
                # Support dict or dataclass
                fit_success = res["fit_successful"]
                if not fit_success:
                    continue
                best_a_dc = res["a_dc"]
                components = res["a_tau_tuple"]
                A_list = [amp / best_a_dc for amp, _ in components]
                tau_list = [tau for _, tau in components]
                qubit_pairs[i].coupler.opx_output.exponential_filter.extend(list(zip(A_list, tau_list)))
                print(f"Updated {qp.name} filter to: {qubit_pairs[i].coupler.opx_output.exponential_filter}")

    # %% {Save_results}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%
