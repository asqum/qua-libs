# %%
"""
        RESONATOR SPECTROSCOPY MULTIPLEXED
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to extract the
'I' and 'Q' quadratures across varying readout intermediate frequencies for all resonators simultaneously.
The data is then post-processed to determine the resonator resonance frequency.
This frequency can be used to update the readout frequency in the state.

Prerequisites:
    - Ensure calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibrate the IQ mixer connected to the readout line (whether it's an external mixer or an Octave port).
    - Define the desired readout pulse amplitude and duration in the state.
    - Specify the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the readout frequency, in the state for all resonators.
    - Save the current state
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q1']
    num_averages: int = 300
    frequency_span_in_mhz: float = 40 #20.0
    frequency_step_in_mhz: float = 0.1
    simulate: bool = True
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False


node = QualibrationNode(name="00_hello_qua_2", parameters=Parameters())
assert not (
    node.parameters.simulate and node.parameters.load_data_id is not None
), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
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
resonators = [qubit.resonator for qubit in qubits]
num_qubits = len(qubits)


# %% {QUA_program}
n_avg = node.parameters.num_averages
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

with program() as multi_res_spec:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    for i, qubit in enumerate(qubits):
        qubit.xy.play("x180")
        qubit.align()
        qubit.z.play("const")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, multi_res_spec, simulation_config)
    # Plot the simulated samples
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples, plot=True)
    # Update the node & save
    node.results = {"figure": plt.gcf(), "waveform_report": waveform_report, "samples": samples}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    # Open a quantum machine to execute the QUA program
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(multi_res_spec)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)
