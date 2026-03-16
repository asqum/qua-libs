"""
The T2 echo for the target coupler.

Prerequisites:
    - pi_pulse for the coupler.
"""

# %%
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state_coupler, active_reset, readout_state

import matplotlib.pyplot as plt
import numpy as np
from time import time
import xarray as xr
from scipy.stats import norm
import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_decay_exp, decay_exp


# %% {Node_parameters}
class Parameters(NodeParameters):
    coupler: str = 'coupler_q4_q5'
    num_averages: int = 500
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 2508
    wait_time_step_in_ns: int = 24
    flux_point_joint_or_independent_or_arbitrary: Literal['joint', 'independent'] = 'independent'   
    simulate: bool = False
    timeout: int = 100
    histo_num:int = 5

node = QualibrationNode(
    name="06x_coupler_T2echo",
    parameters=Parameters()
)


# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
coupler = [machine.qubit_pairs[node.parameters.coupler]] # currently supports 1 coupler a time only.
drive_q = [machine.qubits[coupler[0].extras["RD"]["driven_q"]]]
detector_q = [machine.qubits[coupler[0].extras["RD"]["readout_q"]]]

# Change driving LO
if not node.parameters.simulate:
    drive_LO_original = {drive_q[0].name: drive_q[0].xy.opx_output.upconverter_frequency}
    drive_q[0].xy.opx_output.upconverter_frequency = coupler[0].extras["RD"]["LO"]
    if "swap_direction" in coupler[0].extras["RD"]:
        detector_q[0].z.operations['aSWAP'].slope_direction = coupler[0].extras["RD"]["swap_direction"]

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    node.parameters.wait_time_step_in_ns // 4,
)

flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary  # 'independent' or 'joint'
# if flux_point == "arbitrary":
#     detunings = {q.name : q.arbitrary_intermediate_frequency for q in qubits}
#     arb_flux_bias_offset = {q.name: q.z.arbitrary_offset for q in qubits}
# else:
arb_flux_bias_offset = {q.name: 0.0 for q in drive_q}
detunings = {q.name: 0.0 for q in drive_q}

with program() as t2echo:
    _, _, _, _, n, n_st = qua_declaration(num_qubits=len(detector_q))
    t = declare(int)  # QUA variable for the idle time
    
    state = [declare(int) for _ in range(len(detector_q))]
    state_st = [declare_stream() for _ in range(len(detector_q))]
    
    for i, qubit in enumerate(drive_q):

        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            machine.apply_all_couplers_to_min()
            qubit.z.to_independent_idle()
        elif flux_point == "joint" or "arbitrary":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()

        # Wait for the flux bias to settle
        
        qubit.z.settle()
        qubit.xy.update_frequency(coupler[0].extras["RD"]["IF"])
        align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(t, idle_times)):
                
                if not node.parameters.simulate:
                    if qubit.thermalization_time//5 > coupler[0].extras['T1']*1e9:
                        wait(qubit.thermalization_time * u.ns)
                    else:
                        wait(5*coupler[0].extras['T1']*1e9 * u.ns)
                align()
                
                    
                qubit.xy.play("x90_cp")
                # qubit.align()
                # qubit.z.wait(20)
                # qubit.z.play("const", amplitude_scale=arb_flux_bias_offset[qubit.name]/qubit.z.operations["const"].amplitude, duration=t)
                # qubit.z.wait(20)
                # qubit.align()
                qubit.wait(t)
                qubit.xy.play("x180_cp")
                # qubit.align()
                # qubit.z.wait(20)
                # qubit.z.play("const", amplitude_scale=arb_flux_bias_offset[qubit.name]/qubit.z.operations["const"].amplitude, duration=t)
                # qubit.z.wait(20)
                # qubit.align()
                qubit.wait(t)
                qubit.xy.play("x90_cp", amplitude_scale=-1.0)
                qubit.align()

                
                # Measure the state of the resonators
                readout_state_coupler(detector_q[i], state[i], method='aswap')
                save(state[i], state_st[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(drive_q):
            state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
            


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, t2echo, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    dss = []
    start = time()
    for jj in range(node.parameters.histo_num):
        try:
            with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                job = qm.execute(t2echo)
                # Get results from QUA program
                for i in range(len(drive_q)):
                    print(f"Fetching results for qubit {drive_q[i].name}")
                    data_list = ["n"]
                    results = fetching_tool(job, data_list, mode="live")
                # Live plotting
                # fig, axes = plt.subplots(2, num_qubits, figsize=(4 * num_qubits, 8))
                # interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
                    while results.is_processing():
                    # Fetch results
                        fetched_data = results.fetch_all()
                        n = fetched_data[0]

                        progress_counter(n, n_avg, start_time=results.start_time)
            ds = fetch_results_as_xarray(job.result_handles, coupler, {"idle_time": idle_times})
            dss.append(ds)
        except:
            print("Error got, break the loop and step into analysis.")
            break
    end = time()
    print(f"Total {round(end-start,1)} sec for {node.parameters.histo_num} counts")
    ds = xr.concat(dss, dim='iteration')
    ds = ds.assign_coords(idle_time=8*ds.idle_time/1e3)  # convert to usec
    ds.idle_time.attrs = {'long_name': 'idle time', 'units': 'usec'}
    node.results = {"ds": ds}
        

# %% {Analysis}
if not node.parameters.simulate:
    
    qbs = ds.qubit.values
    iterations = ds.iteration.values

    
    t2_collection = {q: [] for q in qbs}
    fit_collection = {q: [] for q in qbs}

    for q_name in qbs:
        for iter_val in iterations:
            
            ds_sub = ds.sel(qubit=q_name, iteration=iter_val)
            
            
            
            fit_data = fit_decay_exp(ds_sub.state, "idle_time")
            

            fit_collection[q_name] = fit_data
            
            
            decay_val = fit_data.sel(fit_vals="decay").values
            tau_val = -1 / decay_val
            t2_collection[q_name].append(tau_val)


    # %% {Plot}
    grid_names = [q.grid_location for q in drive_q]
    grid = QubitGrid(ds, grid_names)
    mu_collection, sig_collection = {}, {}
    for ax, qubit in grid_iter(grid):
        ## HARDcoded:
        qubit['qubit'] = coupler[0].name
        if node.parameters.histo_num > 1:
            data = np.array(t2_collection[qubit['qubit']])
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
            ax.set_title(f"Qubit {qubit['qubit']} T2 Statistics")
            ax.set_xlabel("T2 (µs)")
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
            
            ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax, marker='o', linestyle='', alpha=0.5)
            ax.set_ylabel("State")
            
            ax.plot(ds.idle_time, fitted, "r--")
            ax.set_title(qubit["qubit"])
            ax.set_xlabel("Idle_time (uS)")
            ax.text(
                0.1,
                0.9,
                f'T2 = {tau.values:.1f} ± {tau_error.values:.1f} µs',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.5),
            )
            mu_collection[qubit['qubit']], sig_collection[qubit['qubit']] = float(tau.values), float(tau_error.values)


    plt.tight_layout()
    plt.show()

    node.results["t2_stats"] = {q: {"mu_us": mu_collection[q], "sigma_us": sig_collection[q]} for q in qbs}
    node.results["figure_histogram"] = grid.fig


# %% {update state}
if not node.parameters.simulate:
    with node.record_state_updates():
        for c in coupler:
            c.extras['T2'] = float(mu_collection[c.name]) * 1e-6

    # %% {save data}
    for q in drive_q:
        q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
    for q in detector_q:
        q.z.operations['aSWAP'].slope_direction = -1
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%
