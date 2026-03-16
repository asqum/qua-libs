"""
Ramsey experiment for a coupler
Prerequisites:
    - pi_pulse for the target qubit and the coupler. Note that the duration of the pi pulse for the coupler and the qubit should be same so that the sequence can be well aligned.
Updates:
    - Driving IF for this coupler.
"""

# %%
from qualibrate import QualibrationNode, NodeParameters
from typing import Literal
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state_coupler
import matplotlib.pyplot as plt
import numpy as np
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.experiments.ramsey.analysis.fitting import fit_ramsey_oscillations_with_exponential_decay, extract_relevant_fit_parameters, calculate_fit_results, RamseyFit
from quam_libs.experiments.ramsey.plotting import add_fit_text
import matplotlib.pyplot as plt


# %% {Node_parameters}
class Parameters(NodeParameters):
    coupler: str = 'coupler_q4_q5'

    num_averages: int = 5000
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 516
    frequency_detuning_in_mhz: float = 5.0
    wait_time_step_in_ns: int = 4
    flux_point_joint_or_independent_or_arbitrary: Literal['joint', 'independent'] = 'independent'   
    simulate: bool = False
    timeout: int = 100
    debug: bool = False

node = QualibrationNode(
    name="06x_coupler_Ramsey",
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



# pi pulse duration check
if not node.parameters.simulate:
    if drive_q[0].xy.operations['x180_cp'].length != detector_q[0].xy.operations['x180'].length:
        raise ValueError(f"The duration of the pi pulse for the coupler and the qubit should be same for the echo sequence to be well aligned. Currently, the duration of the x180_cp pulse for the coupler {coupler[0].name} is {drive_q[0].xy.operations['x180_cp'].length} ns, while the duration of the x180 pulse for the qubit {detector_q[0].name} is {detector_q[0].xy.operations['x180'].length} ns. Please calibrate the pulses such that they have the same duration.")

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

detuning = node.parameters.frequency_detuning_in_mhz * u.MHz
detuning_signs = [-1, 1]
with program() as Ramsey:
    _, _, _, _, n, n_st = qua_declaration(num_qubits=len(drive_q))
    t = declare(int)  # QUA variable for the idle time
    
    state = [declare(int) for _ in range(len(drive_q))]
    state_st = [declare_stream() for _ in range(len(drive_q))]
    detuning_sign = declare(int)
    virtual_detuning_phases = [declare(fixed) for _ in range(len(drive_q))]

    for i, qubit in enumerate(drive_q):
        if not node.parameters.simulate:
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
            with for_(*from_array(detuning_sign, detuning_signs)):
                with for_(*from_array(t, idle_times)):

                    with if_(detuning_sign == 1):
                        assign(virtual_detuning_phases[i], Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
                    with else_():
                        assign(virtual_detuning_phases[i], Cast.mul_fixed_by_int(-detuning * 1e-9, 4 * t))
                    align()
                    
                    if not node.parameters.simulate:
                        if qubit.thermalization_time//5 > coupler[0].extras['T1']*1e9:
                            wait(qubit.thermalization_time * u.ns)
                        else:
                            wait(5*coupler[0].extras['T1']*1e9 * u.ns)
                    
                    align()  
                    qubit.xy.play("x90_cp")
                    wait(t)
                    qubit.xy.frame_rotation_2pi(virtual_detuning_phases[i])
                    qubit.xy.play("x90_cp")
                    align()

                    # Measure the state of the resonators
                    readout_state_coupler(detector_q[i], state[i], method='aswap')
                    save(state[i], state_st[i])

            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(drive_q):
            state_st[i].buffer(len(idle_times)).buffer(len(detuning_signs)).average().save(f"state{i + 1}")
            


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=2500)  # In clock cycles = 4ns
    job = qmm.simulate(config, Ramsey, simulation_config)
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(Ramsey)
        # Get results from QUA program
        for i in range(len(detector_q)):
            print(f"Fetching results for qubit {detector_q[i].name}")
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


# %%
if not node.parameters.simulate:
    # {Data_fetching_and_dataset_creation}
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, drive_q, {"time": idle_times, "sign":detuning_signs})
    ds = ds.assign_coords({"time": (["time"], 4 * idle_times)})
    ds.time.attrs["long_name"] = "idle_time"
    ds.time.attrs["units"] = "ns"
    node.results = {"ds": ds}


# %% {Data_analysis_and_plotting}


def oscillation_decay_exp(t, a, f, phi, offset, decay):
    return a * np.exp(-t * decay) * np.cos(2 * np.pi * f * t + phi) + offset

def plot_ramsey_data_with_fit(ax, ds, qubit, fit, c_name):
    """
    Plot individual qubit data on a given axis.

    """
    fitted_ramsey_data = oscillation_decay_exp(
        ds.time,
        fit.raw_fit_results.sel(fit_vals="a"),
        fit.raw_fit_results.sel(fit_vals="f"),
        fit.raw_fit_results.sel(fit_vals="phi"),
        fit.raw_fit_results.sel(fit_vals="offset"),
        fit.raw_fit_results.sel(fit_vals="decay"),
    )

    
    plot_state(ax, ds, qubit, fitted_ramsey_data)
    ax.set_ylabel("State")
    

    ax.set_xlabel("Idle time [ns]")
    ax.set_title(c_name)
    add_fit_text(ax, fit)
    ax.legend()


def plot_state(ax, ds, qubit, fitted):
    """Plot state data for a qubit."""
    ds.sel(sign=1).loc[qubit].state.plot(
        ax=ax, x="time", c="C0", marker=".", ms=5.0, ls="", label="$\Delta$ = +"
    )
    ds.sel(sign=-1).loc[qubit].state.plot(
        ax=ax, x="time", c="C1", marker=".", ms=5.0, ls="", label="$\Delta$ = -"
    )
    ax.plot(ds.time, fitted.fit.loc[qubit].sel(sign=1), c="C0", ls="-", lw=1)
    ax.plot(ds.time, fitted.fit.loc[qubit].sel(sign=-1), c="C1", ls="-", lw=1)


# %%
if not node.parameters.simulate:
    fit = fit_ramsey_oscillations_with_exponential_decay(ds, True)

    frequency, decay, tau, tau_error = extract_relevant_fit_parameters(fit)

    detuning = int(node.parameters.frequency_detuning_in_mhz * 1e6)

    freq_offset, decay, decay_error = calculate_fit_results(
        frequency, tau, tau_error, fit, detuning
    )
    fits = {
            q.name: RamseyFit(
                qubit_name = q.name,
                freq_offset=1e9 * freq_offset.loc[q.name].values,
                decay=decay.loc[q.name].values,
                decay_error=decay_error.loc[q.name].values,
                raw_fit_results=fit.to_dataset(name="fit")
            )

            for q in drive_q
        }

    grid = QubitGrid(ds, [q.grid_location for q in drive_q])
    for ax, qubit in grid_iter(grid):
        plot_ramsey_data_with_fit(ax, ds, qubit, fits[qubit['qubit']], coupler[0].name)
    node.results["figure"] = grid.fig
    
# %%{Update state}
if not node.parameters.simulate :
    with node.record_state_updates():
        for q in drive_q:
            coupler[0].extras["RD"]["IF"] -= float(fits[q.name].freq_offset)

    # %%{Save data}
    for q in drive_q:
        q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
    for q in detector_q:
        q.z.operations['aSWAP'].slope_direction = -1
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%
