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
from quam_libs.macros import qua_declaration, active_reset_simple, readout_state_coupler
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.save_utils import fetch_results_as_xarray
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

    coupler: str = "coupler_q4_q5"
    """List of qubit pair names to be measured. If None or empty, all active qubit pairs are measured."""
    num_averages: int = 300
    """Number of averages for the measurement."""
    frequency_span_in_mhz: float = 48
    """Frequency span around the coupler RF frequency for the scan."""
    frequency_step_in_mhz: float = 0.8
    """Frequency step size for the scan."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    """Whether to set flux point jointly for all qubits or independently."""
    duration_in_ns: Optional[int] = 2_016
    """Total duration of the flux pulse in ns."""
    time_axis: Literal["linear", "log"] = "linear"
    """Type of time axis for the flux pulse duration sweep."""
    time_step_num: Optional[int] = 101
    """Number of time steps for logarithmic time axis."""
    min_wait_time_in_ns: Optional[int] = 16
    """Minimum wait time in ns for the flux pulse duration sweep."""
    coupler_flux: float = 0.1
    """Coupler flux value set  """
    pi_pulse_duration_scale: int = 5 
    """Duration of the control qubit pulse in ns."""
    simulate: bool = False
    """Whether to simulate the QUA program instead of executing it."""
    simulation_duration_ns: int = 2500
    """Duration of the simulation in ns."""
    timeout: int = 100
    """Timeout for the QOP session in seconds."""
    load_data_id: Optional[int] = None
    """If provided, load data from the specified node ID instead of executing the program."""

    fitting_base_fractions: List[float] = [0.4, 0.15, 0.05]
    """Fitting coefs and can be editted later"""

    separate_avg_mode:bool = False
    """Separately measure then combiine to averge. If true, separate num_average by 100"""


node = QualibrationNode(name="50x_coupler_long_crysocope", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
coupler = [machine.qubit_pairs[node.parameters.coupler]] # currently supports 1 coupler a time only.
drive_q = [machine.qubits[coupler[0].extras["RD"]["driven_q"]]]
detector_q = [machine.qubits[coupler[0].extras["RD"]["readout_q"]]]

# Change driving LO
if not node.parameters.simulate and node.parameters.load_data_id is None:
    drive_LO_original = {drive_q[0].name: drive_q[0].xy.opx_output.upconverter_frequency}
    drive_q[0].xy.opx_output.upconverter_frequency = coupler[0].extras["RD"]["LO"]
    if "swap_direction" in coupler[0].extras["RD"]:
        detector_q[0].z.operations['aSWAP'].slope_direction = coupler[0].extras["RD"]["swap_direction"]


# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {QUA_program}
n_avg = node.parameters.num_averages if not node.parameters.separate_avg_mode else 100  # The number of averages
N_AVG = node.parameters.num_averages//n_avg
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)


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


with program() as multi_res_spec_vs_flux:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=len(detector_q))
    state_target = [declare(int) for _ in range(len(detector_q))]
    state_stream_target = [declare_stream() for _ in range(len(detector_q))]
    df = declare(int)  # QUA variable for the readout frequency
    t_delay = declare(int)  # QUA variable for delay time scan
    duration = node.parameters.duration_in_ns * u.ns
    qp = coupler[0]
    
    if not node.parameters.simulate:
        machine.apply_all_couplers_to_min()

    align()
    for i, qubit in enumerate(drive_q):
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
        new_du = node.parameters.pi_pulse_duration_scale*qubit.xy.operations['x180_cp'].length//4
        align()
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):  # type: ignore
                with for_each_(t_delay, times):
                    
                    # update the frequency of the control qubit
                    qubit.xy.update_frequency(df + coupler[0].extras["RD"]["IF"])
                    
                    if not node.parameters.simulate:
                        if qubit.thermalization_time//5 > coupler[0].extras['T1']*1e9:
                            wait(qubit.thermalization_time * u.ns)
                        else:
                            wait(5*coupler[0].extras['T1']*1e9 * u.ns)
                    align()
                    
                    qp.coupler.play(
                        "const",
                        amplitude_scale=node.parameters.coupler_flux / qp.coupler.operations["const"].amplitude,
                        duration=t_delay,
                    )
                    align()
                    qubit.xy.play(
                        'x180_cp',
                        amplitude_scale=1/node.parameters.pi_pulse_duration_scale,
                        duration=new_du,
                    )
                    
                    readout_state_coupler(detector_q[0], state_target[i], method='aswap')
                    save(state_target[i], state_stream_target[i])
        

    with stream_processing():
        n_st.save("n")
        for i in range(len(drive_q)):
            state_stream_target[i].buffer(len(times)).buffer(len(dfs)).average().save(f"state_target{i + 1}")
            
        


# %% {Simulate_or_execute}
import time
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

else:
    if node.parameters.load_data_id is None:
        dss = []
        for i in range(N_AVG):
            with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                job = qm.execute(multi_res_spec_vs_flux)
                results = fetching_tool(job, ["n"], mode="live")
                while results.is_processing():
                    # Fetch results
                    n = results.fetch_all()[0]
                    # Progress bar
                    # progress_counter(n, n_avg, start_time=results.start_time)
                    progress_counter(n, n_avg, start_time=results.start_time)

            # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
            ds = fetch_results_as_xarray(job.result_handles, coupler, {"time": times * 4, "detuning": dfs})
            dss.append(ds)
            
            # for _ in range(5):
            #     print(".", end='')
            #     time.sleep(1)
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]

        

# %%
if  node.parameters.load_data_id is None and not node.parameters.simulate:
    import xarray as xr
    combined_ds = xr.concat(dss, dim='iteration')
    combined_ds = combined_ds.assign_coords(iteration=np.arange(len(dss)))
    ds = combined_ds.mean(dim='iteration')

    RF_freq = np.array([dfs + c.extras["RD"]["LO"] + c.extras["RD"]["LO"] for c in coupler])
    ds = ds.assign_coords({"freq_full_control": (["qubit", "freq"], RF_freq)})
    detuned_freq = np.array([dfs for c in coupler]) * 1e-6
    ds = ds.assign_coords({"detunings": (["qubit", "freq"], detuned_freq)})
    ds.freq_full_control.attrs["long_name"] = "Frequency"
    ds.freq_full_control.attrs["units"] = "GHz"
    ds.detunings.attrs["long_name"] = "Detuning"
    ds.detunings.attrs["units"] = "MHz"


    # %% {Analyze_data}

    from calibration_utils.pi_flux.analysis import fit_gaussian, PiFluxParameters, optimize_start_fractions
    
    node.parameters.fitting_base_fractions = [0.04, 0.1, 0.001]
    if "state_target" in ds.data_vars:
        state_da = ds["state_target"].transpose("qubit", "time", "detuning")
        center_freqs = xr.apply_ufunc(
            lambda states: fit_gaussian(dfs, states),
            state_da,
            input_core_dims=[["detuning"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
    
    
    # Calculate flux response from frequency shifts
    flux_response = np.sqrt(-1*center_freqs )
    flux_response_normalized = flux_response / flux_response.isel(time = slice(0, 5)).mean(dim = "time")
    # Store results in dataset
    ds['center_freqs'] = center_freqs
    ds['flux_response'] = flux_response
    ds['flux_response_normalized'] = flux_response_normalized
    
    fit_results = {}
    for q in coupler:
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

        can_log_y = np.all(y_data > 0)
        can_log_x = np.all(t_data > 0)

        if can_log_y and can_log_x:
            ax_log.set_xscale("log")
            ax_log.set_yscale("log")
        else:
            print("axis log scale failed... ")

        ax_log.legend(loc="best")
        ax_log.grid(True)

        # fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig, [ax_top, ax_lin, ax_log]


    
    fig = plot_fit(ds, coupler, node.results.get("fit_results"))

    plt.show()
    node.results["fitted_data"] = fig


    # %% {Update_state}

    if node.parameters.load_data_id is None and not node.parameters.simulate:
        with node.record_state_updates():
            for i, qp in enumerate(coupler):
                z_out = qp.coupler.opx_output
                if z_out.exponential_filter is None:
                    z_out.exponential_filter = []
                

                res = node.results["fit_results"][qp.name]
                # Support dict or dataclass
                fit_success = res["fit_successful"]
                if not fit_success:
                    continue
                best_a_dc = res["a_dc"]
                components = res["a_tau_tuple"]
                A_list = [amp / best_a_dc for amp, _ in components]
                tau_list = [tau for _, tau in components]
                qp.coupler.opx_output.exponential_filter.extend(list(zip(A_list, tau_list)))
                print(f"Updated {qp.name} filter to: {qp.coupler.opx_output.exponential_filter}")

        # %% {Save_results}
        for q in drive_q:
            q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
        for q in detector_q:
            q.z.operations['aSWAP'].slope_direction = -1 # always at -1
        node.outcomes = {q.name: "successful" for q in drive_q}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()

# %%
