# %%
"""
POWER RABI WITH ERROR AMPLIFICATION
This sequence involves repeatedly executing the qubit pulse (such as x180, square_pi, or similar) 'N' times and
measuring the state of the resonator across different qubit pulse amplitudes and number of pulses.
By doing so, the effect of amplitude inaccuracies is amplified, enabling a more precise measurement of the pi pulse
amplitude. The results are then analyzed to determine the qubit pulse amplitude suitable for the selected duration.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port)
    - Having found the rough qubit frequency and pi pulse duration (rabi_chevron_duration or time_rabi).
    - Set the qubit frequency, desired pi pulse duration and rough pi pulse amplitude in the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the qubit pulse amplitude (pi_amp) in the state.
    - Save the current state by calling machine.save("quam")
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state_coupler, active_reset_coupler
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray
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

    coupler: str = 'coupler_q4_q5'
    num_averages: int = 200 #10
    operation_x180_or_any_90: Literal["x180", "x90"] = "x180"
    update_x90:bool = True
    min_amp_factor: float = 0.8
    max_amp_factor: float = 1.2
    amp_factor_step: float = 0.004
    max_number_rabi_pulses_per_sweep: int = 44
    reset_type: Literal['active', 'thermal'] = 'active'
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    timeout: int = 100
    load_data_id:int|None = None


node = QualibrationNode(name="04xx_coupler_Power_Rabi_State", parameters=Parameters())


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
    aswap_dir_update_is_q = True
    drive_LO_original = {drive_q[0].name: drive_q[0].xy.opx_output.upconverter_frequency}
    drive_q[0].xy.opx_output.upconverter_frequency = coupler[0].extras["RD"]["LO"]
    if "swap_direction" in coupler[0].extras["RD"]:
        detector_q[0].z.operations['aSWAP'].slope_direction = coupler[0].extras["RD"]["swap_direction"]
    if 'strategy' not in coupler[0].extras["RD"]:
        readout_strategy = 'aswap'
    else:
        readout_strategy = coupler[0].extras["RD"]["strategy"]
    if coupler[0].extras["RD"]["aswap_supplier"].lower() == 'c':
        print("*** aSWAP is applied on coupler itself !")
        if not hasattr(coupler[0].coupler.operations, "aSWAP"):
            raise  LookupError(f"aSWAP operation now is not in {coupler[0].name}.coupler.operation, please add it to unlock the ability for coupler's measurement!")
        aswaper = coupler[0]
        coupler[0].coupler.operations['aSWAP'].slope_direction = coupler[0].extras["RD"]["swap_direction"]
        aswap_dir_update_is_q = False
    else:
        aswaper = None


# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()

num_qubits = len(drive_q)


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
N_pi = (
    node.parameters.max_number_rabi_pulses_per_sweep
)  # Number of applied Rabi pulses sweep
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = 'thermal' #node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
operation = node.parameters.operation_x180_or_any_90  # The qubit operation to play
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(
    node.parameters.min_amp_factor,
    node.parameters.max_amp_factor,
    node.parameters.amp_factor_step,
)

if operation == "x180":
    N_pi_vec = np.arange(1, N_pi, 2).astype("int")
elif operation in ["x90"]:
    N_pi_vec = np.arange(2, N_pi, 4).astype("int")
else:
    raise ValueError(f"Unrecognized operation {operation}.")

operation_exact_name = f"{operation}_{coupler[0].name}"

with program() as power_rabi:
    I, _, Q, _, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    state_stream = [declare_stream() for _ in range(num_qubits)]
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    npi = declare(int)  # QUA variable for the number of qubit pulses
    count = declare(int)  # QUA variable for counting the qubit pulses

    if not node.parameters.simulate:
        machine.apply_all_couplers_to_min()
    for i, qubit in enumerate(drive_q):
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
        qubit.align()
        qubit.xy.update_frequency(coupler[0].extras["RD"]["IF"])
        align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(npi, N_pi_vec)):
                with for_(*from_array(a, amps)):
                    if node.parameters.reset_type == "active":
                        active_reset_coupler(qubit, detector_q[i], f"x180_{coupler[0].name}", flux_applied_target=aswaper, method='aswap')
                        
                    else:
                        if not node.parameters.simulate:
                            if qubit.thermalization_time//5 > coupler[0].extras['T1']*1e9:
                                wait(qubit.thermalization_time * u.ns)
                            else:
                                wait(5*coupler[0].extras['T1']*1e9 * u.ns)

                    align()
                    # Loop for error amplification (perform many qubit pulses)
                    with for_(count, 0, count < npi, count + 1):
                        qubit.xy.play(operation_exact_name, amplitude_scale=a)
                    align()
                    
                    readout_state_coupler(detector_q[i], state[i], flux_applied_target=aswaper, method=readout_strategy)
                    save(state[i], state_stream[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(drive_q):
            if operation == "x180":
                state_stream[i].buffer(len(amps)).buffer(
                    np.ceil(N_pi / 2)
                ).average().save(f"state{i + 1}")
            elif operation in ["x90"]:
                state_stream[i].buffer(len(amps)).buffer(
                    np.ceil(N_pi / 4)
                ).average().save(f"state{i + 1}")
            else:
                raise ValueError(f"Unrecognized operation {operation}.")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, power_rabi, simulation_config)
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(power_rabi)

        # %% {Live_plot}
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

    # %% {Data_fetching_and_dataset_creation}
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(
        job.result_handles, coupler, {"amp": amps, "N": N_pi_vec}
    )
    # Add the qubit pulse absolute amplitude to the dataset
    ds = ds.assign_coords(
        {
            "abs_amp": (
                ["qubit", "amp"],
                np.array([q.xy.operations[operation_exact_name].amplitude * amps for q in drive_q]),
            )
        }
    )
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
        for q in coupler:
            fit_results[q.name] = {}
            f_fit = fit.loc[q.name].sel(fit_vals="f")
            phi_fit = fit.loc[q.name].sel(fit_vals="phi")
            phi_fit = phi_fit - np.pi * (phi_fit > np.pi / 2)
            factor = float(1.0 * (np.pi - phi_fit) / (2 * np.pi * f_fit))
            new_pi_amp = drive_q[0].xy.operations[operation_exact_name].amplitude * factor
            if new_pi_amp < 0.3:  # TODO: 1 for OPX1000 MW
                print(f"amplitude for Pi pulse is modified by a factor of {factor:.2f}")
                print(
                    f"new amplitude is {1e3 * new_pi_amp:.2f} mV \n"
                )  # TODO: 1 for OPX1000 MW
                fit_results[q.name]["Pi_amplitude"] = float(new_pi_amp)
            else:
                print(f"Fitted amplitude too high, new amplitude is 300 mV \n")
                fit_results[q.name]["Pi_amplitude"] = 0.3  # TODO: 1 for OPX1000 MW
        node.results["fit_results"] = fit_results

    elif N_pi > 1:
        # Get the average along the number of pulses axis to identify the best pulse amplitude
        I_n = ds.state.mean(dim="N")
        data_max_idx = I_n.argmax(dim="amp")
        
    # Save fitting results
        for q in coupler:
            new_pi_amp = ds.abs_amp.sel(qubit=q.name)[data_max_idx.sel(qubit=q.name)]
            fit_results[q.name] = {}
            if new_pi_amp < 1:  # TODO: 1 for OPX1000 MW
                fit_results[q.name]["Pi_amplitude"] = float(new_pi_amp)
                print(
                    f"amplitude for Pi pulse is modified by a factor of {I_n.idxmax(dim='amp').sel(qubit = q.name):.2f}"
                )
                print(
                    f"new amplitude is {1e3 * new_pi_amp:.2f} mV \n"
                )  # TODO: 1 for OPX1000 MW
            else:
                print(f"Fitted amplitude too high, new amplitude is 1000 mV \n")
                fit_results[q.name]["Pi_amplitude"] = 1  # TODO: 1 for OPX1000 MW
        node.results["fit_results"] = fit_results

    # %% {Plotting}
    grid_names, qubit_pair_names = grid_pair_names(coupler)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit in grid_iter(grid):
        if N_pi == 1:
            ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qubit].state.plot(
                ax=ax, x="amp_mV"
            )
            ax.plot(ds.abs_amp.loc[qubit] * 1e3, 1e3 * fit_evals.loc[qubit][0])
            ax.set_ylabel("Trans. amp. I [mV]")
        elif N_pi > 1:
            ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qubit].state.plot(
                ax=ax, x="amp_mV", y="N"
            )
            ax.axvline(1e3 * ds.abs_amp.loc[qubit][data_max_idx.loc[qubit]], color="r")
            ax.set_ylabel("num. of pulses")
        ax.set_xlabel("Amplitude [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle(f"{operation_exact_name} Power Rabi State")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    if not node.parameters.simulate:
        with node.record_state_updates():
            for q in coupler:
                drive_q[0].xy.operations[operation_exact_name].amplitude = fit_results[q.name]["Pi_amplitude"]
                if operation == 'x180' and node.parameters.update_x90:
                    drive_q[0].xy.operations[f'x90_{coupler[0].name}'].amplitude = fit_results[q.name]["Pi_amplitude"]/2

        # %% {Save_results}
        if node.parameters.load_data_id is None:
            for q in drive_q:
                q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
            if aswap_dir_update_is_q:
                for q in detector_q:
                    q.z.operations['aSWAP'].slope_direction = -1
            else:
                for c in coupler:
                    c.coupler.operations['aSWAP'].slope_direction = -1
        node.outcomes = {q.name: "successful" for q in coupler}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()

# %%
