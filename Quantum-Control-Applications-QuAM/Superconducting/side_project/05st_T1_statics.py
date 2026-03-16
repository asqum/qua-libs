"""
        T1 MEASUREMENT
The sequence consists in putting the qubit in the excited stated by playing the x180 pulse and measuring the resonator
after a varying time. The qubit T1 is extracted by fitting the exponential decay of the measured quadratures.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the qubit T1 in the state.
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state, active_reset_simple
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_decay_exp, decay_exp
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import norm
from time import time

# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 300
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 200016
    wait_time_step_in_ns: int = 1600
    flux_point_joint_or_independent_or_arbitrary: Literal["joint", "independent"] = "independent"
    reset_type: Literal["active", "thermal"] = "active"
    use_state_discrimination: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    histo_num:int = 5 # 
node = QualibrationNode(name="05st_T1_histogram", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    
# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    node.parameters.wait_time_step_in_ns // 4,
)

flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary  # 'independent' or 'joint'
if flux_point == "arbitrary":
    detunings = {q.name: q.arbitrary_intermediate_frequency for q in qubits}
    arb_flux_bias_offset = {q.name: q.z.arbitrary_offset for q in qubits}
else:
    arb_flux_bias_offset = {q.name: 0.0 for q in qubits}
    detunings = {q.name: 0.0 for q in qubits}

with program() as t1:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the idle time
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

    machine.apply_all_couplers_to_min()
    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
        qubit.z.settle()
        qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(t, idle_times)):
                if node.parameters.reset_type == "active":
                    # active_reset(qubit, "readout")
                    active_reset_simple(qubit, "readout")
                else:
                    qubit.resonator.wait(qubit.thermalization_time * u.ns)
                    qubit.align()

                qubit.xy.play("x180")
                qubit.align()
                # qubit.z.wait(20)
                # qubit.z.play(
                #     "const",
                #     amplitude_scale=arb_flux_bias_offset[qubit.name] / qubit.z.operations["const"].amplitude,
                #     duration=t,
                # )
                # qubit.z.wait(20)
                # qubit.align()
                qubit.wait(t)
               

                # Measure the state of the resonators
                if node.parameters.use_state_discrimination:
                    readout_state(qubit, state[i])
                    save(state[i], state_st[i])
                else:
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            if node.parameters.use_state_discrimination:
                state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")

# from qm import generate_qua_script
# sourceFile = open('T1debug.py', 'w')
# print(generate_qua_script(t1, config), file=sourceFile)

# %%
# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, t1, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples, plot=True)
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
        start = time()
        for jj in range(node.parameters.histo_num):
            try: # prevent getting a unpredictable error like connection failed or qmm closed.
                with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                    job = qm.execute(t1)
                    results = fetching_tool(job, ["n"], mode="live")
                    while results.is_processing():
                        # Fetch results
                        n = results.fetch_all()[0]
                        # Progress bar
                        progress_counter(n, n_avg, start_time=results.start_time)
                
                ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times})
                # Convert IQ data into volts
                if not node.parameters.use_state_discrimination:
                    ds = convert_IQ_to_V(ds, qubits)
                print(f"Counts: {jj+1}\n")
                dss.append(ds)
            except:
                print("Error got, break the loop and step into analysis.")
                break
        end = time()
        print(f"Total {round(end-start,1)} sec for {node.parameters.histo_num} counts")
        ds = xr.concat(dss, dim='iteration')
        # Convert time into µs
        ds = ds.assign_coords(idle_time=4 * ds.idle_time / u.us)  # convert to µs
        ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
        node.results = {"ds": ds}
    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)


# %% {Data_analysis}
if not node.parameters.simulate:
    
    qbs = ds.qubit.values
    iterations = ds.iteration.values

    
    t1_collection = {q: [] for q in qbs}
    fit_collection = {q: [] for q in qbs}

    for q_name in qbs:
        for iter_val in iterations:
            
            ds_sub = ds.sel(qubit=q_name, iteration=iter_val)
            
            
            if node.parameters.use_state_discrimination:
                fit_data = fit_decay_exp(ds_sub.state, "idle_time")
            else:
                fit_data = fit_decay_exp(ds_sub.I, "idle_time")

            fit_collection[q_name] = fit_data
            
            
            decay_val = fit_data.sel(fit_vals="decay").values
            tau_val = -1 / decay_val
            t1_collection[q_name].append(tau_val)

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    mu_collection, sig_collection = {}, {}
    for ax, qubit in grid_iter(grid):
        if node.parameters.histo_num > 1:
            data = np.array(t1_collection[qubit['qubit']])
            counts, bins, _ = ax.hist(data, bins=15, alpha=0.7, color='skyblue', edgecolor='white', label='Counts')
            ### Normal distribution
            mu, sigma = norm.fit(data)
            bin_width = bins[1] - bins[0]
            scaling_factor = len(data) * bin_width
            x = np.linspace(min(data), max(data), 100)
            p = norm.pdf(x, mu, sigma) * scaling_factor  
            mu_collection[qubit['qubit']], sig_collection[qubit['qubit']] = mu, sigma
        
            ### Plot
            ax.plot(x, p, 'r-', lw=2, label='Normal Fit')
            ax.set_title(f"Qubit {qubit['qubit']} T1 Statistics")
            ax.set_xlabel("T1 (µs)")
            ax.set_ylabel("Counts")
            ax.grid(axis='y', alpha=0.3)
            
            stats_text = (
                f"$\mu = {mu:.2f}$ µs\n"
                f"$\sigma = {sigma:.2f}$ µs\n"
                f"Total # = {len(data)}"
            )
            ax.text(
                0.05, 0.95, stats_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
            )
        else:
            fitted = decay_exp(
                ds.idle_time,
                fit_collection[qubit['qubit']].sel(fit_vals="a"),
                fit_collection[qubit['qubit']].sel(fit_vals="offset"),
                fit_collection[qubit['qubit']].sel(fit_vals="decay"),
            )
            decay = fit_collection[qubit['qubit']].sel(fit_vals="decay")
            decay_res = fit_collection[qubit['qubit']].sel(fit_vals="decay_decay")
            tau = -1 / fit_collection[qubit['qubit']].sel(fit_vals="decay")
            tau_error = -tau * (np.sqrt(decay_res) / decay)
            if node.parameters.use_state_discrimination:
                ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
                ax.set_ylabel("State")
            else:
                ds.sel(qubit=qubit["qubit"]).I.plot(ax=ax)
                ax.set_ylabel("I (V)")
            ax.plot(ds.idle_time, fitted, "r--")
            ax.set_title(qubit["qubit"])
            ax.set_xlabel("Idle_time (uS)")
            ax.text(
                0.1,
                0.9,
                f'T1 = {tau.values:.1f} ± {tau_error.values:.1f} µs',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.5),
            )
            mu_collection[qubit['qubit']], sig_collection[qubit['qubit']] = float(tau.values), float(tau_error.values)


    plt.tight_layout()
    plt.show()

    node.results["t1_stats"] = {q: {"mu_us": mu_collection[q], "sigma_us": sig_collection[q]} for q in qbs}
    node.results["figure_histogram"] = grid.fig


    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for index, q in enumerate(qubits):
                if mu_collection[q.name]> 0:
                    q.T1 = float(mu_collection[q.name]) * 1e-6

        # %% {Save_results}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()


# %%
