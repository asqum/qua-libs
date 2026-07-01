# %% {Imports}
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt
from calibration_utils.cryoscope import (
    log_fitted_results,
    plot_fit,
)
from calibration_utils.cryoscope.analysis import unwrap_phase, fit_oscillation, optimize_start_fractions, _extract_relevant_fit_parameters, diff_savgol
from qualang_tools.results import fetching_tool, progress_counter
from qm.qua import *
from qualang_tools.bakery import baking
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qm import SimulationConfig
from quam_libs.components import QuAM, Transmon, TransmonPair
from typing import List, Literal, Optional
from quam_libs.macros import qua_declaration, active_reset, readout_state_coupler
import xarray as xr


# %% {Node_parameters}
description = """
CRYOSCOPE
The goal of this protocol is to measure the step response of the flux line and design
proper FIR and IIR filters (implemented on the OPX) to pre-distort the flux pulses and
improve the two-qubit gates fidelity. Since the flux line ends on the qubit chip, it is
not possible to measure the flux pulse after propagation through the fridge. The idea is
to exploit the flux dependency of the qubit frequency, measured with a modified Ramsey
sequence, to estimate the flux amplitude received by the qubit as a function of time.

The sequence consists of a Ramsey sequence ("x90" - idle time - "x90" or "y90") with a
fixed dephasing time. A flux pulse with varying duration is played during the idle time.
The Sx and Sy components of the Bloch vector are measured by alternatively closing the
Ramsey sequence with a "x90" or "y90" gate in order to extract the qubit dephasing as a
function of the flux pulse duration.

The results are then post-processed to retrieve the step function of the flux line which
is fitted with an exponential function. The corresponding exponential parameters are
then used to derive the FIR and IIR filter taps that will compensate for the distortions
introduced by the flux line (wiring, bias-tee...). Such digital filters are then
implemented on the OPX. Note that these filters will introduce a global delay on all the
output channels that may rotate the IQ blobs so that you may need to recalibrate them for
state discrimination or active reset protocols. More details on these filters:
https://docs.quantum-machines.co/0.1/qm-qua-sdk/docs/Guides/output_filter/?h=filter#hardware-implementation

The protocol is inspired from https://doi.org/10.1063/1.5133894, which contains more
details about the sequence and the post-processing of the data.

This version sweeps the flux pulse duration using the baking tool, which means that the
flux pulse can be scanned with a 1ns resolution, but must be shorter than ~260ns. For
longer pulses either reduce the resolution (2ns steps) or use the 4ns version
(`cryoscope_4ns.py`).

Prerequisites:
        - Resonator spectroscopy performed.
        - Qubit gates (x90, y90) calibrated: spectroscopy, rabi_chevron, power_rabi, Ramsey
            and configuration updated.

Next steps before going to the next node:
        - Update the FIR and IIR filter taps in the configuration:
                - OPX+: config/controllers/con1/analog_outputs/"filter": {"feedforward": fir,
                    "feedback": iir}
                - OPX1000: config/controllers/con1/analog_outputs/"filter": {"feedforward": [],
                    "exponential": [(A, tau)]}
        - WARNING: digital filters add a global delay: recalibrate IQ blobs (rotation_angle &
            ge_threshold).
"""
# Be sure to include [Parameters, Quam] so the node has proper type hinting

class Parameters(NodeParameters):
    coupler:str = 'coupler_q4_q5' # A coupler only
    num_shots: int = 500
    """Number of averages to perform. Default is 50."""
    freq_detuning_MHz: float = 10
    """Since now coupler is away from sweet spot, very sensitive to flux. 10 MHz"""
    cryoscope_len: int = 200
    """Length of the cryoscope operation in microseconds. Default is 240."""
    num_frames: int = 17*3
    """Number of frames to use in the cryoscope experiment. Default is 17."""
    exponential_fit_time_fractions: List[float] = [0.5, 0.01]
    """List of time fractions for the exponential fit. Default is [0.5, 0.01]."""
    update_state_from_GUI: bool = False
    """Whether to update the state from the GUI. Default is False."""
    update_state: bool = False
    """Whether to update the state. Default is False."""
    load_data_id:str = None
    """ID of the data to be loaded"""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    """Flux point configurations"""
    t_guard: int = 60
    """A gaurd time (ns) to make sure second xy pulse play after the flux pulse has ended. Dont exeed ~20% of T2*"""
    timeout: int = 100
    """Mesurement timeout"""
    simulate:bool = False 
    "Whether to simulate or not"
    simulation_duration_ns:int = 5_000
    """ Simulation time"""
    

# %% 
def baked_waveform(config, waveform_amp: float, qubit, max_length: int = 16):
    """Create baked pulse segments with 1ns granularity up to ``max_length`` ns.

    This mirrors the previous inline implementation inside ``12b_cryoscope.py`` and is
    extracted here so it can be shared / unit tested. Each index ``i`` (1..max_length)
    produces a baking object that plays a constant waveform of ``i`` ns with amplitude
    ``waveform_amp`` on the qubit flux line.

    Parameters
    ----------
    config : dict
        Configuration dictionary (typically produced by ``machine.generate_config()``)
        that the baking context mutates.
    waveform_amp : float
        The absolute amplitude to use for the flux pulse.
    qubit : Any
        QUAM qubit object containing the ``z`` or ``coupler`` element name.
    max_length : int, optional
        Maximum pulse length in ns to bake (default 16 to keep within baking memory limits).

    Returns
    -------
    list
        A list of baking objects; element ``i-1`` corresponds to a pulse of length ``i`` ns.
    """
    pulse_segments = []
    # Create the base waveform (1ns resolution). Represent as list of samples.
    waveform = [waveform_amp] * max_length
    if isinstance(qubit, Transmon):
        for i in range(1, max_length + 1):  # inclusive
            with baking(config, padding_method="right") as b:
                wf = waveform[:i]
                b.add_op(f"flux_pulse{i}", qubit.z.name, wf)
                b.play(f"flux_pulse{i}", qubit.z.name)
            pulse_segments.append(b)
    elif isinstance(qubit, TransmonPair):
        for i in range(1, max_length + 1):  # inclusive
            with baking(config, padding_method="right") as b:
                wf = waveform[:i]
                b.add_op(f"flux_pulse{i}", qubit.coupler.name, wf)
                b.play(f"flux_pulse{i}", qubit.coupler.name)
            pulse_segments.append(b)
    else:
        raise TypeError(f"The given qubit object does not have z or coupler element.")
    return pulse_segments

def fit_raw_data(ds: xr.Dataset, node: QualibrationNode):
    """
    Fit raw cryoscope data with exponential models.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset containing I/Q or state data
    node : QualibrationNode
        Node containing parameters and configuration

    Returns
    -------
    tuple
        (fitted_dataset, fit_results_dict)
    """

    def _cryoscope_frequency(ds, stable_time_indices, slope, intercept, sg_range=3, sg_order=2):
        ds = ds.copy()

        freq_cryoscope = diff_savgol(ds, "time", range=sg_range, order=sg_order)

        ds["freq"] = freq_cryoscope

        flux_cryoscope = (1e9 * freq_cryoscope - intercept ) / slope

        baseline = flux_cryoscope.sel(
            time=slice(stable_time_indices[0], stable_time_indices[1])
        ).mean(dim="time")

        ds["flux"] = flux_cryoscope 

        return ds

    if hasattr(ds, "I"):
        data = "I"
    elif hasattr(ds, "state"):
        data = "state"
    else:
        raise ValueError("Dataset must contain either 'I' or 'state' data")

    dafit = fit_oscillation(ds[data], "frame")

    daphi = unwrap_phase(dafit.sel(fit_vals="phi"), "time")
    sg_order = 2
    sg_range = 3

    qubit_name = node.parameters.coupler
    qp = node.machine.qubit_pairs[qubit_name]

    ds_fit = _cryoscope_frequency(
        daphi,
        slope=float(qp.extras["Fx"]["linear_fit_coef"][0]),
        intercept = float(qp.extras["Fx"]["linear_fit_coef"][1]),
        stable_time_indices=(node.parameters.cryoscope_len - 20, node.parameters.cryoscope_len),
        sg_order=sg_order,
        sg_range=sg_range,
    )

    qubit = node.namespace["qubits"][0].name

    # Find the index where ds_fit.flux is closest to 1/e
    qubit_flux = ds_fit.flux.sel(qubit=qubit)
    flux_vals = qubit_flux.values
    time_vals = ds_fit.time.values

    fitting_start_fractions = node.parameters.exponential_fit_time_fractions
    success, best_fractions, components, a_dc, best_rms = optimize_start_fractions(
        time_vals, flux_vals, fitting_start_fractions
    )

    ds_fit.attrs["fit_success"] = success
    if components is not None:
        try:
            amps = [float(a) for a, _ in components]
            taus = [float(t) for _, t in components]
        except Exception:
            amps, taus = [], []
        ds_fit.attrs["fit_component_amps"] = np.array(amps)
        ds_fit.attrs["fit_component_taus_ns"] = np.array(taus)
    ds_fit.attrs["fit_a_dc"] = float(a_dc) if a_dc is not None else np.nan

    ds["fit_results"] = ds_fit

    fit, fit_results = _extract_relevant_fit_parameters(ds, node)

    return fit, fit_results


#%%

node = QualibrationNode(
    name="50xx_coupler_short_cryoscope",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Instantiate machine
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)


# Instantiate the QUAM class from the state file
machine  = QuAM.load()
node.machine = machine

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


num_qubits = len(coupler)

node.namespace["qubits"] = coupler

loaded_fractions = node.parameters.exponential_fit_time_fractions


amplitude = ((u.MHz*node.parameters.freq_detuning_MHz) - float(coupler[0].extras["Fx"]["linear_fit_coef"][1])) / float(coupler[0].extras["Fx"]["linear_fit_coef"][0])

print(amplitude)


# %% {Create_QUA_program}
# Get the active qubits from the node and organize them by batches

n_avg = node.parameters.num_shots  # The number of averages
cryoscope_len = node.parameters.cryoscope_len  # The length of the cryoscope in nanoseconds

cryoscope_time = np.arange(1, cryoscope_len + 1, 1)  # x-axis for plotting - must be in ns

frames = np.linspace(0, 1, node.parameters.num_frames)

baked_config = config

baked_signals = {qubit.name: baked_waveform(baked_config, amplitude, qubit, max_length=16) for qubit in coupler}

node.namespace["baked_config"] = baked_config
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = 'thermal'
    
with program() as qua_prog:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    t_left_ns = declare(int)  # QUA variable for the remainding ns to add to the flux pulse multiple of 4
    t_cycles = declare(int)  # QUA variable for the flux pulse multiple of 4
    idx = declare(int)
    frame = declare(fixed)

    # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)

    for qubit in drive_q:
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        qubit.z.settle()
    align()

    qd = drive_q[0]
    c = coupler[0]
    qd.xy.update_frequency(c.extras["RD"]["IF"])
    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        # Loop over the cryoscope pulse time duration (idx represents the duration in ns)
        with for_(idx, 1, idx <= cryoscope_len, idx + 1):
            # Loop over the phase of the second ramsey x90 pulse to reconstruct the qubit phase
            with for_each_(frame, frames):
                # Qubit initialization
                if not node.parameters.simulate:
                    if qd.thermalization_time//5 > c.extras['T1']*1e9:
                        wait(qd.thermalization_time * u.ns)
                    else:
                        wait(5*c.extras['T1']*1e9 * u.ns)
                
                align()
                ################################################################################################
                # The duration argument in the play command can only produce pulses with duration multiple of  #
                # 4ns. To overcome this limitation we use the baking tool from the qualang-tools package to    #
                # generate pulses with 1ns granularity. To avoid creating custom waveforms for each iteration  #
                # we combine baked pulses with dynamically stretched (multiple of 4ns) pulses.                 #
                ################################################################################################
                # For the first 16ns we play baked pulses exclusively. Loop the time idx counter until 16.
                with if_(idx <= 16):
                    # Swich case to select the baked pulse with duration idx ns
                    with switch_(idx):
                        for j in range(1, 17):
                            # The Ramsey sequence is embedded in the switch case to allow gapless execution
                            with case_(j):
                                align()
                                qd.xy.play("x90_cp")
                                c.coupler.wait((qd.xy.operations["x90_cp"].length + 16) // 4)
                                baked_signals[c.name][j - 1].run()  # Play the baked pulse
                                qd.xy.wait((cryoscope_len + node.parameters.t_guard) >> 2)  # 16ns buffer between pulses
                                qd.xy.frame_rotation_2pi(frame)
                                qd.xy.play("x90_cp")
                # For pulse durations above 16ns we combine baking with regular play statements.
                with else_():
                    # We calculate the closest lower multiple of 4 of the time index
                    assign(t_cycles, idx >> 2)  # Right shift by 2 is a quick way to divide by 4
                    # Calculate the duration to add to pulse multiple of 4.
                    assign(t_left_ns, idx - (t_cycles << 2))  # left shift by 2 is a quick way to multiply by 4
                    # Switch case with the 4 possible sequences:
                    with switch_(t_left_ns):
                        # Play only the pulse multiple of 4
                        with case_(0):
                            align()
                            qd.xy.play("x90_cp")
                            c.coupler.wait((qd.xy.operations["x90_cp"].length + 16) // 4)
                            c.coupler.play(
                                "const",
                                duration=t_cycles,
                                amplitude_scale=amplitude / c.coupler.operations["const"].amplitude,
                            )
                            qd.xy.wait((cryoscope_len + node.parameters.t_guard) // 4)
                            qd.xy.frame_rotation_2pi(frame)
                            qd.xy.play("x90_cp")
                        # Play the pulse multiple of 4 followed by the baked pulse of the missing duration
                        for j in range(1, 4):
                            with case_(j):
                                align()
                                qd.xy.play("x90_cp")
                                c.coupler.wait((qd.xy.operations["x90_cp"].length + 16) // 4)
                                c.coupler.play(
                                    "const",
                                    duration=t_cycles,
                                    amplitude_scale=amplitude / c.coupler.operations["const"].amplitude,
                                )
                                baked_signals[c.name][j - 1].run()
                                qd.xy.wait((cryoscope_len + node.parameters.t_guard) // 4)
                                qd.xy.frame_rotation_2pi(frame)
                                qd.xy.play("x90_cp")
                # Wait for the idle time set slightly above the maximum flux pulse duration
                # to ensure that the 2nd x90 pulse arrives after the longest flux pulse

                # Measure resonator state after the sequence
                align()
                readout_state_coupler(detector_q[0], state[0], method='aswap')
                save(state[0], state_st[0])
                
                align()

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        
        state_st[0].buffer(len(frames)).buffer(cryoscope_len).average().save("state1")



# %% {Simulate or execule}
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
    node.save()

else:
    if node.parameters.load_data_id is None:
        # Open a quantum machine to execute the QUA program
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                # Fetch results
                n = results.fetch_all()[0]
                # Progress bar
                progress_counter(n, n_avg, start_time=results.start_time)

        ds = fetch_results_as_xarray(job.result_handles, coupler, {"frame": frames, "time": cryoscope_time})
        node.results = {"ds_raw": ds}
    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)


# %% {Analyse_data}
node.parameters.exponential_fit_time_fractions = [0.5, 0.02]
node.results["ds_raw"] = node.results["ds_raw"]
node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)

# Log the relevant information extracted from the data analysis
log_fitted_results(fit_results, log_callable=node.log)
# Convert to dict format for storage and create outcomes
node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
node.outcomes = {
    qubit_name: ("successful" if fit_result.success else "failed") for qubit_name, fit_result in fit_results.items()
}


# %% {Plot_data}
fig_flux = plot_fit(node.results["ds_fit"], node.namespace["qubits"], fits=node.results["ds_fit"])

node.results["figure_flux"] = fig_flux


# %% {Update_state}
if not node.parameters.simulate:
    with node.record_state_updates():
        for q in coupler:
            components = node.results["fit_results"][q.name]["components"]
            a_dc = node.results["fit_results"][q.name]["a_dc"]
            A_list = [amp / a_dc for amp, _ in components]
            tau_list = [tau for _, tau in components]
            if not q.z.opx_output.exponential_filter:
                    q.z.opx_output.exponential_filter = []
            q.z.opx_output.exponential_filter.extend(list(zip(A_list, tau_list)))


# %% {Save_results}
for q in drive_q:
    q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
for q in detector_q:
    q.z.operations['aSWAP'].slope_direction = -1 # always at -1
node.outcomes = {q.name: "successful" for q in qubits}
node.results["initial_parameters"] = node.parameters.model_dump()
node.save()


# %%
