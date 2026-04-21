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
    num_averages: int = 150
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 250016
    flux_point_joint_or_independent_or_arbitrary: Literal["joint", "independent"] = "independent"
    reset_type: Literal["active", "thermal"] = "active"
    time_scale:Literal["log"] = "log"
    use_state_discrimination: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True
    histo_num:int = 1 # 
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
qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles

idle_times = np.unique(
    np.geomspace(
        node.parameters.min_wait_time_in_ns,
        node.parameters.max_wait_time_in_ns,
        100,
    )//4
).astype(int)

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
    for multiplexed_qubits in qubits.batch():

        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=list(multiplexed_qubits.values())[0])
        # if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
        for i, qubit in multiplexed_qubits.items():
            qubit.z.settle()
            qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_each_(t, idle_times):
                for i, qubit in multiplexed_qubits.items():
                    if not node.parameters.simulate:
                    # measure ground-state IQ blob for all qubits
                        if node.parameters.reset_type == "active":
                            # active_reset(qubit, "readout")
                            active_reset_simple(qubit, "readout")
                        elif node.parameters.reset_type == "thermal":
                            qubit.wait(2 * qubit.thermalization_time * u.ns)
                        else:
                            raise ValueError(f"Unrecognized reset type {node.parameters.reset_type}.")
                
                align(*[q.xy.name for q in multiplexed_qubits.values()] +
                   [q.resonator.name for q in multiplexed_qubits.values()] +
                   [q.z.name for q in multiplexed_qubits.values()])
                
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("x180")
                    qubit.align()
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
                if node.parameters.multiplexed:
                    align(*[q.xy.name for q in multiplexed_qubits.values()] +
                   [q.resonator.name for q in multiplexed_qubits.values()] +
                   [q.z.name for q in multiplexed_qubits.values()])
                else:
                    align()
                
        # # Measure sequentially
        # if not node.parameters.multiplexed:
        #     align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            if node.parameters.use_state_discrimination:
                state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")



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
        
        target_counts = node.parameters.histo_num
        current_success = 0
        max_retries = target_counts + 5  # 設定一個總嘗試上限，避免無限迴圈
        attempts = 0
    

        while current_success < target_counts and attempts < max_retries:
            attempts += 1
            try: # prevent getting a unpredictable error like connection failed or qmm closed.
                with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                    job = qm.execute(t1)
                    results = fetching_tool(job, ["n"], mode="live")
                    while results.is_processing():
                        # Fetch results
                        n = results.fetch_all()[0]
                        # Progress bar
                        if target_counts <= 5:
                            progress_counter(n, n_avg, start_time=results.start_time)
                
                ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times})
                # Convert IQ data into volts
                if not node.parameters.use_state_discrimination:
                    ds = convert_IQ_to_V(ds, qubits)

                dss.append(ds)
                current_success += 1
                print(f"Counts: {current_success} (Total attempts: {attempts})")
            except Exception as e:
                print(f"Attempt {attempts} failed: {e}. Skipping...")
                if (attempts - current_success) > 5:
                    print("Too many consecutive failures. Stopping experiment.")
                    break
             
        end = time()
        print(f"Total {round(end-start,1)} sec for {current_success} counts")
        ds = xr.concat(dss, dim='iteration')
        # Convert time into µs
        ds = ds.assign_coords(idle_time=4 * ds.idle_time / u.us)  # convert to µs
        ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
        node.results = {"ds": ds}
        reload_qbs = False
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"] 
        machine = node.machine
        reload_qbs = True




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
    
    mu_collection, sig_collection = {}, {}
    if reload_qbs:
        qubits = [machine.qubits[q] for q in qbs]
    

    if node.parameters.histo_num > 1:
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        tot_c = 0
        for ax, qubit in grid_iter(grid):
        
            data = np.array(t1_collection[qubit['qubit']])
            lower_bound = np.percentile(data, 1)   # 下界
            upper_bound = np.percentile(data, 99)  # 上界
            data = data[(data >= lower_bound) & (data <= upper_bound)]
            tot_c = len(data)
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
            ax.set_title(f"{qubit['qubit']}")
            ax.set_xlabel("T1 (µs)")
            ax.set_ylabel("Counts")
            ax.grid(axis='y', alpha=0.3)
            
            stats_text = (
                f"$\mu = {mu:.1f} \pm {sigma:.2f}$ µs\n"
            )
            ax.text(
                0.05, 0.95, stats_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
            )
        plt.suptitle(f" T1 Statistics, #={tot_c}", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        node.results["figure_histogram"] = grid.fig

        iterations = ds.iteration.values
        from numpy.random import randint
        iter = randint(max(iterations))
        grid_2 = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid_2):
            sub_ds = ds.sel(iteration=iter)
            fitted = decay_exp(
                sub_ds.idle_time,
                fit_collection[qubit['qubit']].sel(fit_vals="a"),
                fit_collection[qubit['qubit']].sel(fit_vals="offset"),
                fit_collection[qubit['qubit']].sel(fit_vals="decay"),
            )
            decay = fit_collection[qubit['qubit']].sel(fit_vals="decay")
            decay_res = fit_collection[qubit['qubit']].sel(fit_vals="decay_decay")
            tau = -1 / fit_collection[qubit['qubit']].sel(fit_vals="decay")
            tau_error = -tau * (np.sqrt(decay_res) / decay)
            if node.parameters.use_state_discrimination:
                sub_ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax, marker='o', linestyle='', alpha=0.5)
                ax.set_ylabel("State")
            else:
                sub_ds.sel(qubit=qubit["qubit"]).I.plot(ax=ax, marker='o', linestyle='', alpha=0.5)
                ax.set_ylabel("I (V)")
            ax.plot(sub_ds.idle_time, fitted, "r--")
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
        plt.suptitle("T1")
        plt.tight_layout()
        plt.show()
        node.results[f"figure_idx_{iter}"] = grid_2.fig
        
    else:
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
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
                ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax, marker='o', linestyle='', alpha=0.5)
                ax.set_ylabel("State")
            else:
                ds.sel(qubit=qubit["qubit"]).I.plot(ax=ax, marker='o', linestyle='', alpha=0.5)
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

        plt.suptitle("T1")
        plt.tight_layout()
        plt.show()
        node.results["figure"] = grid.fig
        
    node.results["t1_stats"] = {q: {"mu_us": mu_collection[q], "sigma_us": sig_collection[q]} for q in mu_collection}
    


    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for index, q in enumerate(qubits):
                if mu_collection[q.name]> 0:
                    q.T1 = float(mu_collection[q.name]) * 1e-6
                if node.parameters.histo_num >= 100:
                    if sig_collection[q.name]> 0:
                        q.extras["T1_dev"] = float(sig_collection[q.name]) * 1e-6
                    if mu_collection[q.name]> 0:
                        q.extras["T1"] = float(mu_collection[q.name]) * 1e-6
                    

        # %% {Save_results}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()


# %%
