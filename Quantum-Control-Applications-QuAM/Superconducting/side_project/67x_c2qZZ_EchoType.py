"""
ZZ interaction from coupler to qubit, use echo sequence to extract the ZZ interaction strength. The coupler is put at |1> state to see the maximum ZZ interaction strength. The qubit frequency shift can be observed from the Ramsey oscillation frequency, and the decay of the Ramsey oscillation can give information about the coherence of the qubit under the influence of the coupler.

Prerequisites:
    - pi_pulse for the target qubit and the coupler. Note that the duration of the pi pulse for the coupler and the qubit should be same so that the sequence can be well aligned.
"""

# %%
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List, Dict
from qm.qua import *
from qm import SimulationConfig
from quam_libs.experiments.ramsey.analysis.fetch_dataset import fetch_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state_coupler, active_reset, readout_state
from dataclasses import asdict
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.experiments.ramsey.analysis.fitting import Transmon, fit_ramsey_oscillations_with_exponential_decay, extract_relevant_fit_parameters, calculate_fit_results, RamseyFit
from quam_libs.experiments.ramsey.plotting import plot_ramsey_data_with_fit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq

def fit_ramsey(ds: xr.Dataset, qubits: List[Transmon]):
    fit = fit_ramsey_oscillations_with_exponential_decay(ds, True)
    frequency, decay, tau, tau_error = extract_relevant_fit_parameters(fit)

    freq_offset, decay, decay_error = calculate_fit_results(
        frequency, tau, tau_error, fit, 0
    )

    fits = {
        q.name: RamseyFit(
            qubit_name = q.name,
            freq_offset=1e9 * freq_offset.loc[q.name].values,
            decay=decay.loc[q.name].values,
            decay_error=decay_error.loc[q.name].values,
            raw_fit_results=fit.to_dataset(name="fit")
        )

        for q in qubits
    }

    return fits

def plot_ramsey_fit(ds: xr.Dataset, qubits: List[Transmon], fits: Dict[str, RamseyFit]):
    """
    Plot qubit data for Ramsey experiments.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        plot_ramsey_data_with_fit(ax, ds, qubit, True, fits[qubit["qubit"]])

    grid.fig.suptitle("ZZ interaction: State vs. idle time")

    return grid.fig
    

# %% {Node_parameters}
class Parameters(NodeParameters):
    coupler: str = 'coupler_q4_q5'

    num_averages: int = 1000
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 2008
    wait_time_step_in_ns: int = 40
    flux_point_joint_or_independent_or_arbitrary: Literal['joint', 'independent'] = 'independent'   
    simulate: bool = False
    timeout: int = 100
    debug: bool = False

node = QualibrationNode(
    name="67x_coupler2qubit_ZZintteraction",
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
arb_flux_bias_offset = {q.name: 0.0 for q in drive_q}
detunings = {q.name: 0.0 for q in drive_q}

with program() as t2echo:
    _, _, _, _, n, n_st = qua_declaration(num_qubits=len(detector_q))
    t = declare(int)  # QUA variable for the idle time
    
    state = [declare(int) for _ in range(len(detector_q))]
    state_st = [declare_stream() for _ in range(len(detector_q))]
    
    for i, qubit in enumerate(detector_q):
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
        drive_q[0].xy.update_frequency(coupler[0].extras["RD"]["IF"])
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
                
                    
                qubit.xy.play("x90")
                # qubit.align()
                # qubit.z.wait(20)
                # qubit.z.play("const", amplitude_scale=arb_flux_bias_offset[qubit.name]/qubit.z.operations["const"].amplitude, duration=t)
                # qubit.z.wait(20)
                # qubit.align()
                wait(t)
                align()
                qubit.xy.play("x180")
                if not node.parameters.debug:
                    drive_q[0].xy.play("x180_cp")
                # qubit.align()
                # qubit.z.wait(20)
                # qubit.z.play("const", amplitude_scale=arb_flux_bias_offset[qubit.name]/qubit.z.operations["const"].amplitude, duration=t)
                # qubit.z.wait(20)
                # qubit.align()
                wait(t)
                qubit.xy.play("x90", amplitude_scale=-1.0)
                align()

                
                # Measure the state of the resonators
                readout_state(detector_q[i], state[i])
                save(state[i], state_st[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(detector_q):
            state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
            


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=2500)  # In clock cycles = 4ns
    job = qmm.simulate(config, t2echo, simulation_config)
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(t2echo)
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
    ds = fetch_results_as_xarray(job.result_handles, detector_q, {"time": idle_times})
    ds = ds.assign_coords({"time": (["time"], 4 * idle_times)})
    ds.time.attrs["long_name"] = "idle_time"
    ds.time.attrs["units"] = "ns"
    node.results = {"ds": ds}

# %% {Data_analysis_and_plotting}

def guess_initial_parameters(t, y):

    # 1. Offset
    offset_guess = np.mean(y)
    
    # 2. Amplitude:
    y_detrend = y - offset_guess
    a_guess = (np.max(y) - np.min(y)) / 2
    
    # 3. Frequency : use FFT to find the dominant frequency component as an initial guess for the oscillation frequency
    N = len(t)
    dt = t[1] - t[0]
    yf = rfft(y_detrend)
    xf = rfftfreq(N, dt)
    f_guess = xf[np.argmax(np.abs(yf))]
    
    # 4. Decay 
    decay_guess = 1 / (t[-1] * 0.5) 
    
    # 5. Phase
    phi_guess = 0.0
    
    return [a_guess, f_guess, phi_guess, offset_guess, decay_guess]
def oscillation_decay_exp(t, a, f, phi, offset, decay):
    return a * np.exp(-t * decay) * np.cos(2 * np.pi * f * t + phi) + offset
def apply_fit(x, y, a, f, phi, offset, decay):
    try:
        p0 = guess_initial_parameters(x, y)
        return curve_fit(oscillation_decay_exp, x, y, p0=p0)
    except RuntimeError as e:
        print(f"{a=}, {f=}, {phi=}, {offset=}, {decay=}")
        plt.plot(x, oscillation_decay_exp(x, a, f, phi, offset, decay))
        plt.plot(x, y)
        plt.show()
        # raise e

if not node.parameters.simulate:
    plt.scatter(ds.time, ds.state.values[0], label='Raw Data')
    if not node.parameters.debug:
        fit, _ = apply_fit(ds.time.values, ds.state.values[0], a=0.5, f=1e6, phi=0, offset=0.5, decay=1e-6)
        plt.plot(ds.time, oscillation_decay_exp(ds.time.values, *fit), 'r-', label=f'ZZ = {round(fit[1] * 1000, 2)} MHz', linewidth=2)
        plt.title(f"{coupler[0].name} to {detector_q[0].name} ZZ Interaction Experiment (Coupler at |1>)")
    else:
        plt.title(f"{coupler[0].name} to {detector_q[0].name} ZZ Interaction Experiment (Coupler at |0>)")
    plt.xlabel("Idle time (ns)")
    plt.ylabel("Measured state")
    
    plt.grid()
    plt.legend()
    node.results["figure"] = plt.gcf()
    plt.show()
    


# %% {save data}
if not node.parameters.simulate:
    
    for q in drive_q:
        q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
    for q in detector_q:
        q.z.operations['aSWAP'].slope_direction = -1
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%
