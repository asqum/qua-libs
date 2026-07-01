"""
        TWO-QUBIT INTERLEAVED RANDOMIZED BENCHMARKING
The program consists in playing random sequences of Clifford gates interleaved with a target two-qubit gate and measuring 
the state of the resonators afterward. Each random sequence is generated for the maximum depth (specified as an input) 
and played for each depth asked by the user (the sequence is truncated to the desired depth). Each truncated sequence 
ends with the recovery gate that will bring the qubits back to their ground state.

The random circuits are generated offline and transpiled to a basis gate set (default is ['rz', 'sx', 'x', 'cz']). 
The circuits are executed per two-qubit layer using a switch_case block structure, allowing for efficient execution 
of the quantum circuits.

The program supports two types of target gates: 'idle_2q' and 'cz'. The 'idle_2q' gate is implemented as a hardcoded 
wait time of T1/50 for each qubit. The interleaved RB protocol allows for direct measurement of the fidelity of the 
target gate by comparing the decay rates of the interleaved sequences with reference sequences.

Key Features:
    - reduce_to_1q_cliffords: When enabled (default), the Clifford gates are sampled as 1q Cliffords per qubit 
      (this is of course a much smaller subset of the whole 2q Clifford group).
    - use_input_stream: When enabled (default), the circuit sequences are streamed to the OPX in using the 
      input stream feature. This allows for dynamic circuit execution and reduces memory usage on the OPX.

Each sequence is played multiple times for averaging, and multiple random sequences are generated for each depth to 
improve statistical significance. The data is then post-processed to extract both the two-qubit Clifford fidelity and 
the specific target gate fidelity.

Prerequisites:
    - Having calibrated both qubits' single-qubit gates (resonator_spectroscopy, qubit_spectroscopy, rabi_chevron, power_rabi).
    - Having calibrated the two-qubit gate (cz or idle_2q) that will be benchmarked.
    - Having calibrated the readout for both qubits (readout_frequency, amplitude, duration_optimization IQ_blobs).
    - Having set the appropriate flux bias points for the qubit pair.
    - Having calibrated the qubit frequencies and coupling strength.
"""

# %%

from datetime import datetime, timezone, timedelta
from typing import List, Literal, Optional
from more_itertools import flatten
from quam_libs.experiments.rb_standard.data_utils import RBResult, InterleavedRBResult
import xarray as xr


from qm.qua import *
from qm import SimulationConfig
from qualang_tools.multi_user import qm_session
import numpy as np

from qualang_tools.results import progress_counter, fetching_tool

from qualibrate  import NodeParameters, QualibrationNode
from quam_libs.experiments.rb_standard.circuit_utils import layerize_quantum_circuit, process_circuit_to_integers
from quam_libs.experiments.rb_standard.qua_utils import QuaProgramHandler
from quam_libs.lib.plot_utils import plot_samples
from quam_libs.lib.save_utils import fetch_results_as_xarray

from quam_libs.components import QuAM
from quam_libs.experiments.rb_standard.cloud_utils import write_sync_hook
from quam_libs.experiments.rb_standard.rb_utils import InterleavedRB, StandardRB
from quam_libs.experiments.rb_standard.data_utils import plot_combined_rb
from quam_libs.experiments.rb_standard.plot_utils import gate_mapping


# Question data S-11094 I-11095 -> CZ 100%

# %% {Node_parameters}

class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q1_q2"]#None
    circuit_lengths: tuple[int] = (0,) # in number of cliffords
    num_circuits_per_length: int = 1
    num_averages: int =1
    target_gate: str = "" # "idle_2q" or "cz" supported 
    basis_gates: list[str] = [] 
    readout_mode: Literal["ge", "gef"] = "ge"
    reset_type_thermal_or_active: Literal["thermal", "active", "active_gef"] = "active"
    load_data_id_SRB: Optional[int] = 112 #put the data id of the standard RB here
    load_data_id_IRB: Optional[int] = 113 #put the data id of the interleaved RB here
    reduce_to_1q_cliffords: bool = False
    timeout: int = 100
    seed: int = 0

node = QualibrationNode(name="70d_combine_rb", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}

# Instantiate the QuAM class from the state file
node.machine = QuAM.load()

# # Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = node.machine.active_qubit_pairs
else:
    qubit_pairs = [node.machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

if len(qubit_pairs) == 0:
    raise ValueError("No qubit pairs selected")


config = node.machine.generate_config()

#%% {Load_data_IRB}
node_IRB = node.load_from_id(node.parameters.load_data_id_IRB)
ds_IRB = node_IRB.results["ds"]
target_gate = node_IRB.parameters.target_gate
circuit_depths_IRB = list(node_IRB.parameters.circuit_lengths)
num_repeats_IRB = node_IRB.parameters.num_circuits_per_length
num_averages_IRB = node_IRB.parameters.num_averages 
node.results["ds_IRB"]= ds_IRB
ds_transposed_IRB = ds_IRB.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
ds_transposed_IRB = ds_transposed_IRB.transpose("qubit", "repeat", "circuit_depth", "average")

#%% {Load_data_SRB}
node_SRB = node.load_from_id(node.parameters.load_data_id_SRB)
ds_SRB = node_SRB.results["ds"]
circuit_depths_SRB = list(node_SRB.parameters.circuit_lengths)
num_repeats_SRB = node_SRB.parameters.num_circuits_per_length
num_averages_SRB = node_SRB.parameters.num_averages 
node.results["ds_SRB"]= ds_SRB

standard_RB = StandardRB(
    amplification_lengths=node_SRB.parameters.circuit_lengths,
    num_circuits_per_length=node_SRB.parameters.num_circuits_per_length,
    basis_gates=node_SRB.parameters.basis_gates,
    reduce_to_1q_cliffords=node_SRB.parameters.reduce_to_1q_cliffords,
    num_qubits=2,
    seed=node.parameters.seed
)

transpiled_circuits = standard_RB.transpiled_circuits
transpiled_circuits_as_ints = {}
layerized_circuits = {}
for l, circuits in transpiled_circuits.items():
    layerized_circuits[l] = [layerize_quantum_circuit(qc) for qc in circuits]
    transpiled_circuits_as_ints[l] = [process_circuit_to_integers(qc) for qc in layerized_circuits[l]]

# to calculate the average number of 2q layers per Clifford
average_layers_per_clifford = np.mean([np.mean([len(circ) for circ in circuits])/np.array(length+1) for length, circuits in transpiled_circuits_as_ints.items() if length > 0])

ds_transposed_SRB = ds_SRB.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
ds_transposed_SRB = ds_transposed_SRB.transpose("qubit", "repeat", "circuit_depth", "average")

rb_result_IRB = {}
rb_result_SRB = {}

# %% {Data_analysis and plotting}
for qp in qubit_pairs:

    rb_result_SRB[qp.id] = RBResult(
        circuit_depths=circuit_depths_SRB,
        num_repeats=num_repeats_SRB,
        num_averages=num_averages_SRB,
        state=ds_transposed_SRB.sel(qubit=qp.name).state.data
    )
    
    # Fit the data and calculate all error and fidelity metrics
    rb_result_SRB[qp.id].fit(
        average_layers_per_clifford=average_layers_per_clifford,
        average_gates_per_2q_layer=None
    )
    
    # Plot the results
    fig_SRB = rb_result_SRB[qp.id].plot_with_fidelity()
    
    fig_SRB.suptitle(f"2Q Randomized Benchmarking - {qp.name}")
    # node.add_node_info_subtitle(fig)
    fig_SRB.show()
    
    node.results[f"{qp.id}_figure_RB_decay"] = fig_SRB

    rb_result_IRB[qp.id] = InterleavedRBResult(
        # standard_rb_alpha=node.machine.qubit_pairs[qp.id].macros["cz"].fidelity.get("StandardRB", 1).get("alpha", 1),
        standard_rb_alpha=rb_result_SRB[qp.id].alpha, # if "StandardRB" in qp.extras else 1,
        circuit_depths=circuit_depths_IRB,
        num_repeats=num_repeats_IRB ,
        num_averages=num_averages_IRB ,
        state=ds_transposed_IRB.sel(qubit=qp.name).state.data
    )

    # Fit the data and calculate all error and fidelity metrics
    rb_result_IRB[qp.id].fit()
    
    # Plot the results
    fig_IRB = rb_result_IRB[qp.id].plot_with_fidelity()
    fig_IRB.suptitle(f"2Q Interleaved Randomized Benchmarking - {qp.name}")
    # node.add_node_info_subtitle(fig)
    fig_IRB.show()
    
    node.results[f"{qp.id}_figure_IRB_decay"] = fig_IRB

    fig_combined = plot_combined_rb(
        qp.name,
        rb_result_SRB[qp.id],
        rb_result_IRB[qp.id],
        target_gate=target_gate.upper()
    )

    fig_combined.show()

    node.results[f"{qp.id}_figure_combined_RB_decay"] = fig_combined
# %% {Update_state}
# with node.record_state_updates():
#     for qp in qubit_pairs:
#         qp.extras['Interleaved_RB'] = rb_result[qp.id].fidelity
# %% {Save_results}
node.save()
# %%
