# %%
"""
The Power Rabi for the target coupler.

Prerequisites:
    - the driving frequency for the target coupler.

Updates:
    - the pi pulse amplitude for this coupler.

Next step:
    - You may check the population again for the case in aSWAP's slope direction to +1.

"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state_coupler
from quam_libs.lib.instrument_limits import instrument_limits
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_oscillation, oscillation
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

    coupler: str = 'coupler_q5_q6'
    num_averages: int = 500
    operation_x180_or_any_90: Literal["x180", "x90"] = "x180"
    min_amp_factor: float = 0.0 #0.001
    max_amp_factor: float = 1.79 #2.0
    amp_factor_step: float = 0.018 #005
    max_number_rabi_pulses_per_sweep: int = 1 #1, 40
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    update_x90: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 8000
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False

node = QualibrationNode(name="04x_coupler_PowerRabi", parameters=Parameters())


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
    if 'strategy' not in coupler[0].extras["RD"]:
        readout_strategy = 'aswap'
    else:
        readout_strategy = coupler[0].extras["RD"]["strategy"]

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()



# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
N_pi = node.parameters.max_number_rabi_pulses_per_sweep  # Number of applied Rabi pulses sweep
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = 'thermal' #node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
state_discrimination = True
operation = node.parameters.operation_x180_or_any_90  # The qubit operation to play
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(
    node.parameters.min_amp_factor,
    node.parameters.max_amp_factor,
    node.parameters.amp_factor_step,
)

# Number of applied Rabi pulses sweep
if N_pi > 1:
    if operation in ["x180"]:

        N_pi_vec = np.arange(1, N_pi, 2).astype("int")
    elif operation in ["x90"]:
        N_pi_vec = np.arange(2, N_pi, 4).astype("int")
    else:
        raise ValueError(f"Unrecognized operation {operation}.")
else:
    N_pi_vec = np.linspace(1, N_pi, N_pi).astype("int")[::2]

operation_exact_name = f"{operation}_{coupler[0].name}"

with program() as power_rabi:
    _, _, _, _, n, n_st = qua_declaration(num_qubits=len(detector_q))
    
    state = [declare(int) for _ in range(len(detector_q))]
    state_stream = [declare_stream() for _ in range(len(detector_q))]
    
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    npi = declare(int)  # QUA variable for the number of qubit pulses
    count = declare(int)  # QUA variable for counting the qubit pulses

    if not node.parameters.simulate:
        machine.apply_all_couplers_to_min()
    for i, qubit in enumerate(drive_q):
        # Bring the active qubits to the minimum frequency point
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
        qubit.xy.update_frequency(coupler[0].extras["RD"]["IF"])
        # update LO
        qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(npi, N_pi_vec)):
                with for_(*from_array(a, amps)):
                    # Initialize the qubits
                    
                    if not node.parameters.simulate:
                        if qubit.thermalization_time//5 > coupler[0].extras['T1']*1e9:
                            wait(qubit.thermalization_time * u.ns)
                        else:
                            wait(5*coupler[0].extras['T1']*1e9 * u.ns)

                    # for a better RO fidelity
                    # align()
                    # active_reset(detector_q[i], "readout")
                    # align()

                    # Loop for error amplification (perform many qubit pulses)
                    with for_(count, 0, count < npi, count + 1):
                        qubit.xy.play(operation_exact_name, amplitude_scale=a)
                    qubit.align()

                    readout_state_coupler(detector_q[i], state[i], method=readout_strategy)
                    save(state[i], state_stream[i])

        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(drive_q):
            if operation in ["x180"]:
                
                state_stream[i].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save(
                    f"state{i + 1}"
                )
            elif operation in ["x90"]:
                
                state_stream[i].buffer(len(amps)).buffer(np.ceil(N_pi / 4)).average().save(
                    f"state{i + 1}"
                )
            else:
                raise ValueError(f"Unrecognized operation {operation}.")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, power_rabi, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(power_rabi)
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
        ds = fetch_results_as_xarray(job.result_handles, drive_q, {"amp": amps, "N": N_pi_vec})

        # Add the qubit pulse absolute amplitude to the dataset
        ds = ds.assign_coords(
        {
            "abs_amp": (
                ["qubit", "amp"],
                np.array([q.xy.operations[operation_exact_name].amplitude * amps for q in drive_q]),
            )
            }
        )
    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    fit_results = {}
    if N_pi == 1:
        # Fit the power Rabi oscillations
        fit = fit_oscillation(ds.state, "amp")
        fit_evals = oscillation(
            ds.amp,
            fit.sel(fit_vals="a"),
            fit.sel(fit_vals="f"),
            fit.sel(fit_vals="phi"),
            fit.sel(fit_vals="offset"),
        )

        # Save fitting results
        for q in drive_q:
            fit_results[q.name] = {}
            f_fit = fit.loc[q.name].sel(fit_vals="f")
            phi_fit = fit.loc[q.name].sel(fit_vals="phi")
            phi_fit = phi_fit - np.pi * (phi_fit > np.pi / 2)
            factor = float(1.0 * (np.pi - phi_fit) / (2 * np.pi * f_fit))
            new_pi_amp = q.xy.operations[operation_exact_name].amplitude * factor
            limits = instrument_limits(q.xy)
            if new_pi_amp < limits.max_x180_wf_amplitude:
                print(f"amplitude for Pi pulse is modified by a factor of {factor:.2f}")
                print(f"new amplitude is {1e3 * new_pi_amp:.2f} {limits.units} \n")
                fit_results[q.name]["Pi_amplitude"] = new_pi_amp
            else:
                print(f"Fitted amplitude too high, new amplitude is {limits.max_x180_wf_amplitude} \n")
                fit_results[q.name]["Pi_amplitude"] = limits.max_x180_wf_amplitude
        node.results["fit_results"] = fit_results

    elif N_pi > 1:
        # Get the average along the number of pulses axis to identify the best pulse amplitude
        I_n = ds.state.mean(dim="N")
        if (N_pi_vec[0] % 2 == 0 and operation == "x180") or (N_pi_vec[0] % 2 != 0 and operation != "x180"):
            data_max_idx = I_n.argmin(dim="amp")
        else:
            data_max_idx = I_n.argmax(dim="amp")

        # Save fitting results
        for q in drive_q:
            new_pi_amp = float(ds.abs_amp.sel(qubit=q.name)[data_max_idx.sel(qubit=q.name)].data)
            fit_results[q.name] = {}
            limits = instrument_limits(q.xy)
            if new_pi_amp < limits.max_x180_wf_amplitude:
                fit_results[q.name]["Pi_amplitude"] = new_pi_amp
                print(
                    f"amplitude for Pi pulse is modified by a factor of {I_n.idxmax(dim='amp').sel(qubit = q.name):.2f}"
                )
                print(f"new amplitude is {1e3 * new_pi_amp:.2f} {limits.units} \n")
            else:
                print(f"Fitted amplitude too high, new amplitude is {limits.max_x180_wf_amplitude} \n")
                fit_results[q.name]["Pi_amplitude"] = limits.max_x180_wf_amplitude

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in drive_q])
    for ax, qubit in grid_iter(grid):
        if N_pi == 1:
            
            ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qubit].state.plot(ax=ax, x="amp_mV")
            ax.plot(ds.abs_amp.loc[qubit] * 1e3, fit_evals.loc[qubit][0])
            ax.set_ylabel("Qubit state")

        elif N_pi > 1:
            
            ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qubit].state.plot(ax=ax, x="amp_mV", y="N")
            
            ax.set_ylabel("num. of pulses")
            ax.axvline(1e3 * ds.abs_amp.loc[qubit][data_max_idx.loc[qubit]], color="r")
        ax.set_xlabel("Amplitude [mV]")
        ax.set_title(node.parameters.coupler)
    grid.fig.suptitle("Rabi : State vs. amplitude")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    if not node.parameters.simulate:
        with node.record_state_updates():
            for q in drive_q:
                q.xy.operations[operation_exact_name].amplitude = fit_results[q.name]["Pi_amplitude"]
                if operation == "x180" and node.parameters.update_x90:
                    q.xy.operations[f"x90_{coupler[0].name}"].amplitude = fit_results[q.name]["Pi_amplitude"] / 2

        # %% {Save_results}
        if node.parameters.load_data_id is None:
            for q in drive_q:
                q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
            for q in detector_q:
                q.z.operations['aSWAP'].slope_direction = -1 # always at -1
        node.outcomes = {q.name: "successful" for q in drive_q}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()

# %%
