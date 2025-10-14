# %%
from __future__ import annotations

from dataclasses import asdict
from typing import List, Literal, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit
from quam_libs.macros import qua_declaration, active_reset

from qualibrate import QualibrationNode, NodeParameters
from quam_libs.trackable_object import tracked_updates
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.components import QuAM
import warnings

from calibration_utils.pi_flux import (
    process_raw_dataset,
    fit_raw_data,
    plot_fit,
)


description = """
Pi vs Flux :
"""

class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q5"]
    num_shots: int = 100
    operation: str = "x180"
    operation_amplitude_factor: float = 1.0
    duration_in_ns: int = 9000
    time_axis: Literal["linear", "log"] = "linear"
    time_step_in_ns: int = 48
    time_step_num: int = 200
    frequency_span_in_mhz: float = 300
    frequency_step_in_mhz: float = 0.4
    fitting_base_fractions: List[float] = [0.4, 0.15, 0.05]
    update_state: bool = False
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    timeout: int = 100
    multiplexed: bool = False
    reset_type_active_or_thermal: Literal["active", "thermal"] = "thermal"
    thermal_reset_extra_time_in_us: int = 10_000
    min_wait_time_in_ns: int = 32
    use_state_discrimination: bool = False
    load_data_id:str = None
    simulate:str = None 
    detuning_in_mhz: int = 450.0

node = QualibrationNode(
    name="16a_pi_vs_flux_long_distortions",
    description=description,
    parameters=Parameters(),
)
# Instantiate machine
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)

# Instantiate the QUAM class from the state file
machine = QuAM.load()
node.machine = machine

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

node.namespace["qubits"] = qubits
flux_point = node.parameters.flux_point_joint_or_independent


# %% {Create_qua_program}


operation_name = node.parameters.operation

# Ensure operation exists
for qubit in qubits:
    if hasattr(qubit.xy.operations, operation_name):
        continue
    qubit.xy.operations[operation_name] = qubit.xy.operations["x180"]

operation_amp_scale = node.parameters.operation_amplitude_factor or 1.0
n_avg = node.parameters.num_shots

# Frequency sweep
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, span // 2, step, dtype=np.int32)

# Time sweep
if node.parameters.time_axis == "linear":
    times = np.arange(
        node.parameters.min_wait_time_in_ns // 4,
        node.parameters.duration_in_ns // 4,
        max(node.parameters.time_step_in_ns, 4) // 4,
        dtype=np.int32,
    )
else:
    times = np.logspace(
        np.log10(max(node.parameters.min_wait_time_in_ns // 4, 1)),
        np.log10(max(node.parameters.duration_in_ns // 4, 2)),
        max(node.parameters.time_step_num, 3),
        dtype=np.int32,
    )
    times = np.unique(times)
    
flux_amps = [np.sqrt(-node.parameters.detuning_in_mhz * 1e6 / q.freq_vs_flux_01_quad_term) for q in qubits]

# LO retune for headroom
tracked_qubits = []
if_update = []

node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray([qubit.id for q in qubits]),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "time": xr.DataArray(4 * times, attrs={"long_name": "Flux pulse duration", "units": "ns"}),
        }

for q in qubits:
    # Decide if updating the LO is needed depending on the detuning request
    if (
        q.xy.intermediate_frequency
        - node.parameters.detuning_in_mhz * 1e6
        - node.parameters.frequency_span_in_mhz * 1e6 / 2
    ) < -400e6:
        node.parameters.reset_type_active_or_thermal = "thermal"  # Active reset will not work if the lo is changed
        warnings.warn(
            "Qubit LO has been changed to reach desired detuning, active reset will not work. Reset type changed to thermal."
        )
        if_update.append(0)
        # track the LO and IF changes to revert later
        with tracked_updates(q, auto_revert=True, dont_assign_to_none=False) as q_upd:
            rf_frequency = q_upd.xy.intermediate_frequency + q_upd.xy.opx_output.upconverter_frequency
            lo_frequency = q_upd.xy.opx_output.upconverter_frequency - node.parameters.detuning_in_mhz * 1e6
            if (q_upd.xy.opx_output.band == 3) and (lo_frequency < 6.5e9):
                raise ValueError("Requested detuning is too large for the given MW FEM band")
            elif (q_upd.xy.opx_output.band == 2) and (lo_frequency < 4.5e9):
                raise ValueError("Requested detuning is too large for the given MW FEM band")
            print(f"Updating {q_upd.name} LO to {lo_frequency}")
            q_upd.xy.opx_output.upconverter_frequency = lo_frequency
            q_upd.xy.intermediate_frequency -= node.parameters.detuning_in_mhz * 1e6
            tracked_qubits.append(q_upd)
    else:
        if_update.append(int(node.parameters.detuning_in_mhz))

node.namespace["if_update"] = if_update
node.namespace["tracked_qubits"] = tracked_qubits


with program() as qua_prog:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)
    t_delay = declare(int)
    for qubit in qubits:
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
    align()

    for i, qubit in enumerate(qubits):
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            # Qubit spectroscopy frequency loop
            with for_(*from_array(df, dfs)):
                # Time delay loop
                with for_each_(t_delay, times):
                    # Step the qubit spectroscopy tone frequency
                    qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency - if_update[i])
                    qubit.align()
                    # Play the flux pulse
                    qubit.z.play(
                        "const",
                        amplitude_scale=flux_amps[i] / qubit.z.operations["const"].amplitude,
                        duration=t_delay + 200,
                    )
                    # Wait for a variable time
                    qubit.xy.wait(t_delay)
                    # Play the qubit spectroscopy pulse
                    qubit.xy.play(operation_name, amplitude_scale=operation_amp_scale)
                    qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                    qubit.align()
                    qubit.wait(200)
                    # Measure
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
        # Measure sequentially
        if not node.parameters.multiplexed:
            align()
        
    with stream_processing():
        n_st.save("n")
        for i, _ in enumerate(qubits):
            I_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"Q{i + 1}")

node.namespace["qua_program"] = qua_prog

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, qua_prog, simulation_config)
    # Plot the simulated samples
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Update the node & save
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    # Open a quantum machine to execute the QUA program
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(qua_prog)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)
# %% {Data_fetching_and_dataset_creation}

if not node.parameters.simulate:
    if node.parameters.load_data_id is not None:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"time": 4 * times, "detuning": dfs})
    # Add the dataset to the node
    node.results = {"ds_raw": ds}

# %% {Process_raw}
from calibration_utils.pi_flux import  process_raw_dataset
ds_raw = node.results["ds_raw"]
ds_proc = process_raw_dataset(ds_raw, node)
node.results["ds_proc_input"] = ds_proc


# %% {Analyze_data}
ds_in = node.results["ds_proc_input"]
ds, fit_results = fit_raw_data(ds_in, node)
node.results["ds_proc"] = ds
node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
# log_fitted_results(fit_results, log_callable=node.log)


# %% {Plot}
ds = node.results["ds_proc"]
fig = plot_fit(ds, qubits, node.results.get("fit_results"))
plt.show()
plt.show()
node.results["fitted_data"] = fig


 #%% {Update_state}
if node.parameters.load_data_id is  None:
    with node.record_state_updates():
        for q in qubits:
            z_out = node.machine.qubits[q.name].z.opx_output
        if z_out.exponential_filter is None:
            z_out.exponential_filter = []

    with node.record_state_updates():
        for q in qubits:
            res = node.results["fit_results"][q.name]
            # Support dict or dataclass
            fit_success = res["fit_successful"]
            if not fit_success:
                continue
            best_a_dc = res["a_dc"]
            components = res["a_tau_tuple"]
            A_list = [amp / best_a_dc for amp, _ in components]
            tau_list = [tau for _, tau in components]
            node.machine.qubits[q.name].z.opx_output.exponential_filter.extend(list(zip(A_list, tau_list)))
            print(f"Updated {q.name} filter to: {node.machine.qubits[q.name].z.opx_output.exponential_filter}")


# %% {Save_results}
node.outcomes = {q.name: "successful" for q in qubits}
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
node.save()


# %%
