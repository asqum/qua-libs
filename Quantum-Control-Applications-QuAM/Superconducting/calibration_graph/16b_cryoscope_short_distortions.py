# %% {Imports}
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt
from calibration_utils.cryoscope import (
    fit_raw_data,
    log_fitted_results,
    plot_fit,
    process_raw_dataset,
)
from qualang_tools.results import fetching_tool, progress_counter
from qm.qua import *
from qualang_tools.bakery import baking
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubits_from_node,
)
from qm import SimulationConfig
from quam_libs.components import QuAM
from typing import List, Literal, Optional
from quam_libs.macros import qua_declaration, active_reset, readout_state


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
    qubits: Optional[List[str]] = ["q4"]
    num_shots: int = 500
    """Number of averages to perform. Default is 50."""
    detuning_target_in_MHz: int = 500
    """Target detuning from sweetspot for the cryoscope pulse in MHz. Default is 350."""
    cryoscope_len: int = 200
    """Length of the cryoscope operation in microseconds. Default is 240."""
    num_frames: int = 17
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
    reset_type_active_or_thermal: Literal["active", "thermal"] = "active"
    """Reset type"""
    use_state_discrimination: bool = "False"
    """Whether to use state discrimination"""
    t_guard: int = 300
    """A gaurd time to make sure second xy pulse play after the flux pulse has ended. Dont exeed ~20% of T2*"""
    timeout: int = 100
    """Mesurement timeout"""
    simulate:str = None 
    "Whether to simulate or not"
    
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
        QUAM qubit object containing the ``z`` element name.
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
    for i in range(1, max_length + 1):  # inclusive
        with baking(config, padding_method="right") as b:
            wf = waveform[:i]
            b.add_op(f"flux_pulse{i}", qubit.z.name, wf)
            b.play(f"flux_pulse{i}", qubit.z.name)
        pulse_segments.append(b)
    return pulse_segments

node = QualibrationNode(
    name="16b_cryoscope_short_distortions",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Instantiate machine
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)


# Instantiate the QUAM class from the state file
machine  = QuAM.load()
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

loaded_fractions = node.parameters.exponential_fit_time_fractions




# %% {Create_QUA_program}
# Get the active qubits from the node and organize them by batches
qubit = qubits[0]

assert num_qubits == 1, "This node only supports one qubit at the time."

n_avg = node.parameters.num_shots  # The number of averages
cryoscope_len = node.parameters.cryoscope_len  # The length of the cryoscope in nanoseconds

# Absolute amplitude of the Cryoscope pulse
amplitude = float(np.sqrt(-node.parameters.detuning_target_in_MHz * 1e6 / qubits[0].freq_vs_flux_01_quad_term))

cryoscope_time = np.arange(1, cryoscope_len + 1, 1)  # x-axis for plotting - must be in ns

frames = np.linspace(0, 1, node.parameters.num_frames)

baked_config = config

baked_signals = {qubit.name: baked_waveform(baked_config, amplitude, qubit, max_length=16) for qubit in qubits}

node.namespace["baked_config"] = baked_config
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_active_or_thermal
    
with program() as qua_prog:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    if node.parameters.use_state_discrimination:
        state = [declare(bool) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
    t_left_ns = declare(int)  # QUA variable for the remainding ns to add to the flux pulse multiple of 4
    t_cycles = declare(int)  # QUA variable for the flux pulse multiple of 4
    idx = declare(int)
    frame = declare(fixed)

    # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)

    for qubit in qubits:
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
    align()

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        # Loop over the cryoscope pulse time duration (idx represents the duration in ns)
        with for_(idx, 1, idx <= cryoscope_len, idx + 1):
            # Loop over the phase of the second ramsey x90 pulse to reconstruct the qubit phase
            with for_each_(frame, frames):
                # Qubit initialization
                if reset_type == "active":
                    active_reset(qubit)
                elif reset_type == "thermal":
                    qubit.wait(qubit.thermalization_time * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")
                
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
                                qubit.xy.play("x90")
                                qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                baked_signals[qubit.name][j - 1].run()  # Play the baked pulse
                                qubit.xy.wait((cryoscope_len + node.parameters.t_guard) >> 2)  # 16ns buffer between pulses
                                qubit.xy.frame_rotation_2pi(frame)
                                qubit.xy.play("x90")
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
                            qubit.xy.play("x90")
                            qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                            qubit.z.play(
                                "const",
                                duration=t_cycles,
                                amplitude_scale=amplitude / qubit.z.operations["const"].amplitude,
                            )
                            qubit.xy.wait((cryoscope_len + node.parameters.t_guard) // 4)
                            qubit.xy.frame_rotation_2pi(frame)
                            qubit.xy.play("x90")
                        # Play the pulse multiple of 4 followed by the baked pulse of the missing duration
                        for j in range(1, 4):
                            with case_(j):
                                align()
                                qubit.xy.play("x90")
                                qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                qubit.z.play(
                                    "const",
                                    duration=t_cycles,
                                    amplitude_scale=amplitude / qubit.z.operations["const"].amplitude,
                                )
                                baked_signals[qubit.name][j - 1].run()
                                qubit.xy.wait((cryoscope_len + node.parameters.t_guard) // 4)
                                qubit.xy.frame_rotation_2pi(frame)
                                qubit.xy.play("x90")
                # Wait for the idle time set slightly above the maximum flux pulse duration
                # to ensure that the 2nd x90 pulse arrives after the longest flux pulse

                # Measure resonator state after the sequence
                align()
                qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))

                if node.parameters.use_state_discrimination:
                    assign(state[0], I[0] > qubit.resonator.operations["readout"].threshold)
                    save(state[0], state_st[0])
                else:
                    save(I[0], I_st[0])
                    save(Q[0], Q_st[0])
                align()

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        if node.parameters.use_state_discrimination:
            state_st[0].boolean_to_int().buffer(len(frames)).buffer(cryoscope_len).average().save("state1")
        else:
            I_st[0].buffer(len(frames)).buffer(cryoscope_len).average().save("I1")
            Q_st[0].buffer(len(frames)).buffer(cryoscope_len).average().save("Q1")


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
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds_raw"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubits = resolve_qubits_from_node(machine, node)
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"frame": frames, "time": cryoscope_time})
        # Convert IQ data into volts
        # ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        # ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
    # Add the dataset to the node
    node.results = {"ds_raw": ds}

# %% {Analyse_data}
node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
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
        for q in qubits:
            components = node.results["fit_results"][q.name]["components"]
            a_dc = node.results["fit_results"][q.name]["a_dc"]
            A_list = [amp / a_dc for amp, _ in components]
            tau_list = [tau for _, tau in components]
            if not q.z.opx_output.exponential_filter:
                    q.z.opx_output.exponential_filter = []
            q.z.opx_output.exponential_filter.extend(list(zip(A_list, tau_list)))


# %% {Save_results}
node.outcomes = {q.name: "successful" for q in qubits}
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
node.save()


# %%
