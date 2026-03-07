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
from quam_libs.macros import qua_declaration, active_reset, readout_state, active_reset_simple, readout_state_coupler
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


# %% {Node_parameters}
class Parameters(NodeParameters):
    coupler: str = 'coupler_q3_q4'
    coupler_flux_amp:float = -0.235
    detector_qb:str = 'q4'
    driver_qb:str = 'q3'
    num_averages: int = 3000
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 150016
    wait_time_step_in_ns: int = 1500
    flux_point_joint_or_independent_or_arbitrary: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    simulation_duration_ns: int = 5000
    timeout: int = 100
    load_data_id: Optional[int] = None
    debug: bool = False


node = QualibrationNode(name="05x_coupler_T1", parameters=Parameters())


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
drive_q = [machine.qubits[node.parameters.driver_qb]]
detector_q = [machine.qubits[node.parameters.detector_qb]]
coupler = [machine.qubit_pairs[node.parameters.coupler]] # currently supports 1 coupler a time only.

# 
detuned_qs = [machine.qubits[q] for q in node.parameters.coupler.split("_")[1:]]



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
    state_st = [declare_stream() for _ in range(len(detector_q))]
    if not node.parameters.simulate:
        machine.apply_all_couplers_to_min()
    for i, qubit in enumerate(drive_q):

        # Bring the active qubits to the desired frequency point
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
        qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(t, idle_times)):
                # if node.parameters.reset_type == "active":
                #     # active_reset(qubit, "readout")
                #     active_reset_simple(qubit, "readout")
                # else:
                if not node.parameters.simulate:
                    qubit.resonator.wait(qubit.thermalization_time * u.ns)
                else:
                    qubit.resonator.wait(16 * u.ns)
                qubit.align()
                if not node.parameters.debug:
                    qubit.xy.play("x180_cp")
                align()
                coupler[0].coupler.play("const", amplitude_scale=node.parameters.coupler_flux_amp/coupler[0].coupler.operations["const"].amplitude, duration=t)
                for q in detuned_qs:
                    q.z.play("const", amplitude_scale=0.15 / q.z.operations["const"].amplitude, duration=t)
                # # qubit.z.wait(20)
                # # qubit.z.play(
                # #     "const",
                # #     amplitude_scale=arb_flux_bias_offset[qubit.name] / qubit.z.operations["const"].amplitude,
                # #     duration=t,
                # # )
                # # qubit.z.wait(20)
                # # qubit.align()
                align()
                wait(500) # 2µs settle down time after the flux pulse, please increase the shots
                

                # Measure the state of the resonators
                readout_state_coupler(detector_q[i], state[i], method='aswap')
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
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(t1)
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
        ds = fetch_results_as_xarray(job.result_handles, drive_q, {"idle_time": idle_times})

        # Convert time into µs
        ds = ds.assign_coords(idle_time=4 * ds.idle_time / u.us)  # convert to µs
        ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # Fit the exponential decay
    
    fit_data = fit_decay_exp(ds.state, "idle_time")
    
    fit_data.attrs = {"long_name": "time", "units": "µs"}
    # Fitted decay
    fitted = decay_exp(
        ds.idle_time,
        fit_data.sel(fit_vals="a"),
        fit_data.sel(fit_vals="offset"),
        fit_data.sel(fit_vals="decay"),
    )
    # Decay rate and its uncertainty
    decay = fit_data.sel(fit_vals="decay")
    decay.attrs = {"long_name": "decay", "units": "ns"}
    decay_res = fit_data.sel(fit_vals="decay_decay")
    decay_res.attrs = {"long_name": "decay", "units": "ns"}
    # T1 and its uncertainty
    tau = -1 / fit_data.sel(fit_vals="decay")
    tau.attrs = {"long_name": "T1", "units": "µs"}
    tau_error = -tau * (np.sqrt(decay_res) / decay)
    tau_error.attrs = {"long_name": "T1 error", "units": "µs"}

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in drive_q])
    for ax, qubit in grid_iter(grid):
        
        ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
        ax.set_ylabel("State")
        
        ax.plot(ds.idle_time, fitted.loc[qubit], "r--")
        ax.set_title(f"prepare {node.parameters.coupler} at {'|1>' if not node.parameters.debug else '|0>'}")
        ax.set_xlabel("Idle_time (uS)")
        ax.text(
            0.1,
            0.9,
            f'T1 = {tau.sel(qubit = qubit["qubit"]).values:.1f} ± {tau_error.sel(qubit = qubit["qubit"]).values:.1f} µs',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )
    grid.fig.suptitle("T1")
    plt.tight_layout()
    plt.show()
    node.results["figure_raw"] = grid.fig

    # %% {Save_results}
    if node.parameters.load_data_id is None:
    
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()


# %%
