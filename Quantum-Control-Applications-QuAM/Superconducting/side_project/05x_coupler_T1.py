"""
The T1 for the target coupler.

Prerequisites:
    - pi_pulse for the coupler.

"""
## hihihi 
# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state, active_reset_simple, readout_state_coupler
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
from time import time
import xarray as xr
from scipy.stats import norm


# %% {Node_parameters}
class Parameters(NodeParameters):
    coupler: str = 'coupler_q4_q5'
    readout_strategy: Literal['zz-pi', 'aswap'] = 'aswap'
    num_averages: int = 500
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 160016
    wait_time_step_in_ns: int = 1600
    flux_point_joint_or_independent_or_arbitrary: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    simulation_duration_ns: int = 10000
    timeout: int = 100
    load_data_id: Optional[int] = None
    debug: bool = False
    histo_num:int = 1


node = QualibrationNode(name="05x_coupler_T1", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handle units and conversions.
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
n_avg = node.parameters.num_averages  # The number of averages
# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    node.parameters.wait_time_step_in_ns // 4,
)

flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary  # 'independent' or 'joint'
# if flux_point == "arbitrary":
#     detunings = {q.name: q.arbitrary_intermediate_frequency for q in qubits}
#     arb_flux_bias_offset = {q.name: q.z.arbitrary_offset for q in qubits}
# else:
arb_flux_bias_offset = {q.name: 0.0 for q in drive_q}
detunings = {q.name: 0.0 for q in drive_q}

with program() as t1:
    _, _, _, _, n, n_st = qua_declaration(num_qubits=len(detector_q))
    t = declare(int)  # QUA variable for the idle time
    
    state = [declare(int) for _ in range(len(detector_q))]
    flip_state = declare(int) 
    state_st = [declare_stream() for _ in range(len(detector_q))]
    if not node.parameters.simulate:
        machine.apply_all_couplers_to_min()
    for i, qubit in enumerate(drive_q):

        # Bring the active qubits to the desired frequency point
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
        qubit.xy.update_frequency(coupler[0].extras["RD"]["IF"])
        qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(t, idle_times)):
                # if node.parameters.reset_type == "active":
                #     # active_reset(qubit, "readout")
                #     active_reset_simple(qubit, "readout")
                # else:
                
                if not node.parameters.simulate:
                    if qubit.thermalization_time//5 > coupler[0].extras['T1']*1e9:
                        wait(qubit.thermalization_time * u.ns)
                    else:
                        wait(5*coupler[0].extras['T1']*1e9 * u.ns)

                align()
                if not node.parameters.debug:
                    qubit.xy.play("x180_cp")
                align()
                # qubit.z.wait(20)
                # qubit.z.play(
                #     "const",
                #     amplitude_scale=arb_flux_bias_offset[qubit.name] / qubit.z.operations["const"].amplitude,
                #     duration=t,
                # )
                # qubit.z.wait(20)
                # qubit.align()
                wait(t)
               

                # Measure the state of the resonators
                readout_state_coupler(detector_q[i], state[i], method=node.parameters.readout_strategy, zz_pi_pulse_duration_scale=100)
                save(state[i], state_st[i])


    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(drive_q):
            state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
            

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
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.save()

else:
    if node.parameters.load_data_id is None:
        dss = []
        start = time()
        for jj in range(node.parameters.histo_num):
            try:
                with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                    job = qm.execute(t1)
                    results = fetching_tool(job, ["n"], mode="live")
                    while results.is_processing():
                        # Fetch results
                        n = results.fetch_all()[0]
                        # Progress bar
                        progress_counter(n, n_avg, start_time=results.start_time)
                
                ds = fetch_results_as_xarray(job.result_handles, coupler, {"idle_time": idle_times})
                print(f"Counts: {jj+1}\n")
                dss.append(ds)

            except:
                print("Error got, break the loop and step into analysis.")
                break
        end = time()
        print(f"Total {round(end-start,1)} sec for {node.parameters.histo_num} counts")
        ds = xr.concat(dss, dim='iteration')
        ds = ds.assign_coords(idle_time=4 * ds.idle_time / u.us)  # convert to µs
        ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
        node.results = {"ds": ds}
    
    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:

    qbs = ds.qubit.values
    iterations = ds.iteration.values

    
    t1_collection = {q: [] for q in qbs}
    fit_collection = {q: [] for q in qbs}

    for q_name in qbs:
        for iter_val in iterations:
            
            ds_sub = ds.sel(qubit=q_name, iteration=iter_val)
            
            fit_data = fit_decay_exp(ds_sub.state, "idle_time")

            fit_collection[q_name] = fit_data
            
            
            decay_val = fit_data.sel(fit_vals="decay").values
            tau_val = -1 / decay_val
            t1_collection[q_name].append(tau_val)

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in drive_q])
    mu_collection, sig_collection = {}, {}
    for ax, qubit in grid_iter(grid):
        ## HARDcoded:
        qubit['qubit'] = coupler[0].name
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
            ax.set_title(f"{qubit['qubit']} T1 Statistics")
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
            
            ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
            ax.set_ylabel("State")
            
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

    # %% {update state}
    if not node.parameters.simulate and node.parameters.load_data_id is None:
        with node.record_state_updates():
            if not node.parameters.simulate:
                for c in coupler:
                    c.extras["T1"] = float(mu_collection[c.name]) * 1e-6

        #%% {save data}
        for q in drive_q:
            q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
        for q in detector_q:
            q.z.operations['aSWAP'].slope_direction = -1
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()


# %%
