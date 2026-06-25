# %%
"""
Two-Qubit Readout Confusion Matrix Measurement

This sequence measures the readout error when simultaneously measuring the state of two qubits. The process involves:

1. Preparing the two qubits in all possible combinations of computational basis states (|00⟩, |01⟩, |10⟩, |11⟩)
2. Performing simultaneous readout on both qubits
3. Calculating the confusion matrix based on the measurement results

For each prepared state, we measure:
1. The readout result of the first qubit
2. The readout result of the second qubit

The measurement process involves:
1. Initializing both qubits to the ground state
2. Applying single-qubit gates to prepare the desired input state
3. Performing simultaneous readout on both qubits
4. Repeating the process multiple times to gather statistics

The outcome of this measurement will be used to:
1. Quantify the readout fidelity for two-qubit states
2. Identify and characterize crosstalk effects in the readout process
3. Provide data for readout error mitigation in two-qubit experiments

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits

Outcomes:
- 4x4 confusion matrix representing the probabilities of measuring each two-qubit state given a prepared input state
- Readout fidelity metrics for simultaneous two-qubit measurement
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam.utils.qua_types import QuaVariable, ScalarInt, ScalarFloat
from quam_libs.components import QuAM, Transmon, TransmonPair
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef, active_reset_simple
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import warnings
from qualang_tools.bakery import baking
from quam_libs.lib.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse

# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits:List[str] = ['q4']
    operation_len_ns:int = 104
    pulse_align_debug:bool = False
    neg_pole_amp_ratio:float = 0.5


node = QualibrationNode(
    name="waveform_simulator", parameters=Parameters()
)


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()
# %%

####################
# Helper functions #
####################

def split_bipolar_macro(
        qbORqp:Transmon|TransmonPair,
        amplitude_scale:float|ScalarFloat = 1.0,
        neg_pole_amp_ratio:float|ScalarFloat = 1.0,
        debug:bool=False):

    ### ========== Composing ==========
    if isinstance(qbORqp, Transmon):
        channel = qbORqp.z
    elif isinstance(qbORqp, TransmonPair):
        channel = qbORqp.coupler
    else:
        raise TypeError(f"Target assigned for split_bipolar macro must be Transmon or TransmonPair ! Go checking it plz.")
    
    if not debug:
        ## Half Cosine Raise
        channel.play('flattopV2', amplitude_scale=amplitude_scale)
        ## Half Cosine Fall
        channel.play('flattopV2', amplitude_scale=-1*neg_pole_amp_ratio*amplitude_scale)
    else:
        ## Half Cosine Raise
        channel.play('Cz_flattop', amplitude_scale=amplitude_scale)
        ## Half Cosine Fall
        channel.play('Cz_flattop', amplitude_scale=-1*neg_pole_amp_ratio*amplitude_scale)
        
    # qbORqp.align()

for q in qubits:
    q.z.opx_output.delay = 0.0


# %% {QUA_program}
half_len_clicks_python = node.parameters.operation_len_ns//8 # len calculated in python

with program() as wf_simu:
    amp_sca, neg_amp_ratio = declare(fixed, value=1.0), declare(fixed, value=node.parameters.neg_pole_amp_ratio)

    for i, qb in enumerate(qubits): 
        qb.xy.play("x180",amplitude_scale=1.79)
        align()
        # split_bipolar_macro(qb, amplitude_scale=0.8, neg_pole_amp_ratio=neg_amp_ratio, debug=node.parameters.pulse_align_debug)
        qb.z.play("aSWAP", amplitude_scale=1.0)
    
        

# %% {Simulate}

# Simulates the QUA program for the specified duration
simulation_config = SimulationConfig(duration=3_000//4)  # In clock cycles = 4ns
job = qmm.simulate(config, wf_simu, simulation_config)
samples = job.get_simulated_samples()
samples.con1.plot()
node.results = {"figure": plt.gcf()}
wf_report = job.get_simulated_waveform_report()
wf_report.create_plot(samples, plot=True, save_path=None)
node.save()


# %% {Run all}