# %%
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state, active_reset_simple
from quam_libs.lib.qua_datasets import convert_IQ_to_V
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from time import time
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.qubit_thermometer import repetition_data
from quam_libs.lib.parity_switching import RamseyAnalysis
from scipy.stats import norm

# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None #The qubit to be measured. If None, all active qubits will be measured
    num_averages: int = 5000
    frequency_detuning_in_mhz:float=0.2
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 20016
    flux_point_joint_or_independent_or_arbitrary: Literal['joint', 'independent'] = 'independent'   
    simulate: bool = False
    timeout: int = 100
    use_state_discrimination: bool = True
    time_scale:Literal["linear",'log'] = "linear"
    reset_type: Literal['active', 'thermal'] = "active"
    histo_num:int = 1

node = QualibrationNode(
    name="06st_T2star_histogram",
    parameters=Parameters()
)


# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
detuning = node.parameters.frequency_detuning_in_mhz * u.MHz

# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
time_points:int = 150
if node.parameters.time_scale == 'linear':
    idle_times = np.unique(
        np.linspace(
            node.parameters.min_wait_time_in_ns,
            node.parameters.max_wait_time_in_ns,
            time_points,
        )//4
    ).astype(int)
else:
    idle_times = np.unique(
        np.geomspace(
            node.parameters.min_wait_time_in_ns,
            node.parameters.max_wait_time_in_ns,
            time_points,
        )//4
    ).astype(int)



flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary  # 'independent' or 'joint'
if flux_point == "arbitrary":
    detunings = {q.name : q.arbitrary_intermediate_frequency for q in qubits}
    arb_flux_bias_offset = {q.name: q.z.arbitrary_offset for q in qubits}
else:
    arb_flux_bias_offset = {q.name: 0.0 for q in qubits}
    detunings = {q.name: 0.0 for q in qubits}

with program() as t1:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the idle time
    virtual_detuning_phases = [declare(fixed) for _ in range(num_qubits)]
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
    for multiplexed_qubits in qubits.batch():
        for i, qubit in multiplexed_qubits.items():

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
            wait(1000, qubit.z.name)

            qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_each_(t, idle_times):
                assign(virtual_detuning_phases[i], Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
                # align()
                for i, qubit in multiplexed_qubits.items():
                    if not node.parameters.simulate:
                        if node.parameters.reset_type == "active":
                            # active_reset(qubit, "readout")
                            active_reset_simple(qubit, "readout")
                        elif node.parameters.reset_type == "thermal":
                            qubit.wait(2 * qubit.thermalization_time * u.ns)
                        else:
                            raise ValueError(f"Unrecognized reset type {node.parameters.reset_type}.")

                    reset_frame(qubit.xy.name)
                    qubit.xy.play("x90")
                    qubit.wait(t)
                    qubit.xy.frame_rotation_2pi(virtual_detuning_phases[i])
                    qubit.xy.play("x90")

                    # Measure the state of the resonators
                    if node.parameters.use_state_discrimination:
                        readout_state(qubit, state[i])
                        save(state[i], state_st[i])
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            if node.parameters.use_state_discrimination:
                state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, t1, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    dss = []
    start = time()
    target_counts = node.parameters.histo_num
    current_success = 0
    max_retries = target_counts + 5  # 設定一個總嘗試上限，避免無限迴圈
    attempts = 0

    while current_success < target_counts and attempts < max_retries:
        attempts += 1
        try:
            with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                job = qm.execute(t1)
                # Get results from QUA program
                data_list = ["n"]
                results = fetching_tool(job, data_list, mode="live")
                # Live plotting
                # fig, axes = plt.subplots(2, num_qubits, figsize=(4 * num_qubits, 8))
                # interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
                while results.is_processing():
                # Fetch results
                    fetched_data = results.fetch_all()
                    n = fetched_data[0]
                    if target_counts <= 5:
                        progress_counter(n, n_avg, start_time=results.start_time)
            
            ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times})

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
    print(f"Total {round(end-start,1)} sec for {node.parameters.histo_num} counts")
    ds = xr.concat(dss, dim='iteration')

    ds = ds.assign_coords(idle_time=4*ds.idle_time/1e3)  # convert to usec
    ds.idle_time.attrs = {'long_name': 'idle time', 'units': 'usec'}
    node.results = {"ds": ds}


# %%
if not node.parameters.simulate:
    dd = ds
    models = {q.name:[] for q in qubits}
    for iteration in range(dd.dims['iteration']):
        signal_name = 'state' if node.parameters.use_state_discrimination else 'I'
        dss = dd.isel(iteration=iteration).rename({signal_name: "signal"})
        sep_data = repetition_data(dss, repetition_dim="qubit")
        for sq_data in sep_data:
            qubit_name = sq_data["qubit"].values.item()
            analysis = RamseyAnalysis(sq_data)
            models[qubit_name].append(analysis)
            # figs = analysis._plot_results()


    # %% {Plotting}
    mu_collection, sig_collection = {}, {}
    if node.parameters.histo_num > 1:
        tot_c = 1
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            t2_collection = []
            for iter in range(len(models[qubit['qubit']])):
                _, tau, _ = models[qubit['qubit']][iter].plot(None, qubit['qubit'], node.parameters.use_state_discrimination)
                t2_collection.append(tau)

            data = np.array(t2_collection)
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
            ax.set_xlabel("T2* (µs)")

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
        plt.suptitle(f" T2* Statistics, #={tot_c}", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        node.results["figure_histogram"] = grid.fig

        from numpy.random import randint
        c = randint(len(models[qubit['qubit']]))
        grid_2 = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid_2):
            ax, tau, tau_err = models[qubit['qubit']][c].plot(ax, qubit['qubit'],  node.parameters.use_state_discrimination)
        plt.suptitle("T2*")
        plt.tight_layout()
        plt.show()
        node.results[f"figure_idx_{c}"] = grid_2.fig


    else:
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            ax, tau, tau_err = models[qubit['qubit']][0].plot(ax, qubit['qubit'],  node.parameters.use_state_discrimination)
            mu_collection[qubit['qubit']], sig_collection[qubit['qubit']] = tau, tau_err
    
        plt.suptitle("T2*")
        plt.tight_layout()
        plt.show()
        node.results["figure"] = grid.fig
    

    node.results["t2*_stats"] = {q: {"mu_us": mu_collection[q], "sigma_us": sig_collection[q]} for q in mu_collection}
    


# %% {Update state}
if not node.parameters.simulate:
    with node.record_state_updates():
        for q in qubits:
            if mu_collection[q.name] > 0:
                q.T2ramsey = mu_collection[q.name] * 1e-6
            if node.parameters.histo_num >= 100:
                if mu_collection[q.name] > 0:
                    q.extras["T2ramsey"] = mu_collection[q.name] * 1e-6
                if sig_collection[q.name] > 0:
                    q.extras["T2ramsey_dev"] = float(sig_collection[q.name]) * 1e-6
    # %% {Save data}          
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%
