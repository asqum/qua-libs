"""
        TWO-QUBIT STANDARD RANDOMIZED BENCHMARKING
The program consists in playing random sequences of Clifford gates and measuring the state of the resonators afterward. 
Each random sequence is generated for the maximum depth (specified as an input) and played for each depth asked by the 
user (the sequence is truncated to the desired depth). Each truncated sequence ends with the recovery gate that will 
bring the qubits back to their ground state.

The random circuits are generated offline and transpiled to a basis gate set (default is ['rz', 'sx', 'x', 'cz']). 
The circuits are executed per two-qubit layer using a switch_case block structure, allowing for efficient execution 
of the quantum circuits.

Readout is GEF. States 0-3 are the computational subspace and state 4 is leakage. Fidelity is the
leakage post-selection estimate from Chen & Baldwin, arXiv:2502.00154 Sec. IV.2.3: retention gives the
leakage rate τ, the post-selected |00> decay gives the computational error λ, and
F = 1 - (3/4)λ - τ. The single-exponential fit of P(|00>) is reported alongside it.

The same shots are also fit to the Wood & Gambetta model, arXiv:1704.03081. Retention is
A + B λ1^m, which gives the leakage rate L1 = (1 - A)(1 - λ1) and the seepage rate
L2 = A(1 - λ1). P(|00>) is A0 + B0 λ1^m + C0 λ2^m, and
F = [(d - 1) λ2 + 1 - L1] / d. That fit is a separate figure. L1 and L2 are only
separately determined once retention has approached its asymptote.

Key Features:
    - reduce_to_1q_cliffords: When enabled (default), the Clifford gates are sampled as 1q Cliffords per qubit 
      (this is of course a much smaller subset of the whole 2q Clifford group).
    - use_input_stream: When enabled, the circuit sequences are streamed to the OPX chunk-by-chunk (one chunk
      per circuit depth) using the QUA input-stream feature (advance_input_stream) instead of being declared
      as a single large array. This bypasses the OPX's QUA variable budget cap on declared arrays, enabling
      longer circuit depths and/or more circuits per depth than the without-input-stream path can support.

Each sequence is played multiple times for averaging, and multiple random sequences are generated for each depth to 
improve statistical significance. The data is then post-processed to extract the two-qubit Clifford fidelity.

Prerequisites:
    - Having calibrated both qubits' single-qubit gates (resonator_spectroscopy, qubit_spectroscopy, rabi_chevron, power_rabi).
    - Having calibrated the two-qubit gate (cz) that will be used in the Clifford sequences.
    - Having calibrated the readout for both qubits (readout_frequency, amplitude, duration_optimization IQ_blobs).
    - Having set the appropriate flux bias points for the qubit pair.
    - Having calibrated the qubit frequencies and coupling strength.
"""

# %%

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Literal, Optional
from matplotlib import pyplot as plt
from more_itertools import flatten
import numpy as np
from quam_libs.experiments.rb_standard.data_utils import LeakageRBResult
import xarray as xr
from tqdm.auto import tqdm


from qm.qua import *
from qm import SimulationConfig
from qualang_tools.multi_user import qm_session

from qualang_tools.results import progress_counter, fetching_tool

from qualibrate  import NodeParameters, QualibrationNode
from quam_libs.experiments.rb_standard.circuit_utils import circuit_to_layer_ints
from quam_libs.experiments.rb_standard.qua_utils import QuaProgramHandler
from quam_libs.lib.plot_utils import plot_samples
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubit_pairs_from_node,
)

from quam_libs.components import QuAM
from quam_libs.experiments.rb_standard.cloud_utils import write_sync_hook
from quam_libs.experiments.rb_standard.rb_utils import StandardRB, rb_cache_key, rb_save, rb_try_load
from quam_libs.experiments.rb_standard.plot_utils import gate_mapping
from numpy import arange

# %% {Node_parameters}

class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q4_q5"]#None
    circuit_lengths: tuple[int] = (0, 1, 2 ,3, 4,5, 6, 7, 8, 12, 16, 20, 25, 30, 40, 50, 60,70) # in number of cliffords
    num_circuits_per_length: int = 30
    num_averages: int = 200
    basis_gates: list[str] = ['rz', 'sx', 'x', 'cz'] 
    readout_mode: Literal["ge", "gef"] = "gef"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type_thermal_or_active: Literal["thermal", "active", "active_gef"] = "thermal"
    reduce_to_1q_cliffords: bool = False
    use_input_stream: bool = True
    max_chunk_ints: int = 16000
    simulate: bool = False
    simulation_duration_ns: int = 10000
    load_data_id: Optional[int] = None
    timeout: int = 600
    seed: int = 0

node = QualibrationNode[Parameters, QuAM](name="71a_two_qubit_standard_rb", parameters=Parameters())


# Average gates per 2q layer calculation:
# - Cases with non-Z gates (X/Y via .play()): assign value 2
# - Cases with only Z gates (via .frame_rotation()): assign value 0
# - Case 64 (CZ gate): assign value 1
# Average number of gate per layer ≈ 1.51
average_gates_per_2q_layer = None


def build_two_qubit_state_probabilities(ds: xr.Dataset) -> xr.DataArray:
    state = ds["state"]
    outcome_probs = xr.concat([(state == i) for i in range(4)], dim="outcome").astype(float)
    outcome_probs = outcome_probs.assign_coords(outcome=("outcome", ["00", "01", "10", "11"]))
    leakage_probs = (state >= 4).astype(float)
    leakage_probs = leakage_probs.expand_dims(outcome=["leakage"])
    return xr.concat([outcome_probs, leakage_probs], dim="outcome")


def plot_two_qubit_state_probabilities(probabilities: xr.DataArray, qubit_pair_name: str):
    fig, ax = plt.subplots()

    style_map = {
        "00": {"color": "tab:blue", "linewidth": 2.5, "marker": "o"},
        "01": {"color": "tab:orange", "linewidth": 1.5, "marker": "s"},
        "10": {"color": "tab:green", "linewidth": 1.5, "marker": "^"},
        "11": {"color": "tab:red", "linewidth": 1.5, "marker": "d"},
        "leakage": {"color": "black", "linewidth": 2.5, "marker": "x", "linestyle": "--"},
    }

    for outcome in probabilities.outcome.values:
        outcome_probabilities = probabilities.sel(outcome=outcome)
        sequence_means = outcome_probabilities.mean(dim="average")
        mean_prob = sequence_means.mean(dim="repeat")
        error_bars = sequence_means.std(dim="repeat")
        ax.errorbar(
            mean_prob.circuit_depth,
            mean_prob,
            yerr=error_bars,
            label=outcome,
            capsize=3,
            elinewidth=1.0,
            **style_map[str(outcome)],
        )

    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("Probability")
    ax.set_title(f"2Q State Distribution - {qubit_pair_name}")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0, title="State")

    return fig


# %% {Initialize_QuAM_and_QOP}

# Instantiate the QuAM class from the state file
node.machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = node.machine.active_qubit_pairs
else:
    qubit_pairs = [node.machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

if len(qubit_pairs) == 0:
    raise ValueError("No qubit pairs selected")

# Generate the OPX and Octave configurations

# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = node.machine.connect(timeout=node.parameters.timeout)

config = node.machine.generate_config()


# %% {Random circuit generation}

READOUT_OPCODE = 66
cache_key = rb_cache_key(
    node.parameters.seed,
    node.parameters.circuit_lengths,
    node.parameters.num_circuits_per_length,
)
cache_dir = Path(__file__).resolve().parent.parent / ".rb_cache"
cached = rb_try_load(cache_dir, cache_key)

total_cliffords = node.parameters.num_circuits_per_length * sum(node.parameters.circuit_lengths)
print(
    f"RB config: {len(node.parameters.circuit_lengths)} depths, "
    f"{node.parameters.num_circuits_per_length} circuits/depth, "
    f"~{total_cliffords} Cliffords to generate+transpile (first run only; cached after)"
)

if cached is not None:
    circuits_as_ints = cached["circuits_as_ints"]
    average_layers_per_clifford = cached["average_layers_per_clifford"]
    print(f"Loaded {len(circuits_as_ints)} cached RB circuits (key {cache_key[:12]})")
else:
    standard_RB = StandardRB(
        amplification_lengths=node.parameters.circuit_lengths,
        num_circuits_per_length=node.parameters.num_circuits_per_length,
        basis_gates=node.parameters.basis_gates,
        reduce_to_1q_cliffords=node.parameters.reduce_to_1q_cliffords,
        num_qubits=2,
        seed=node.parameters.seed,
    )

    transpiled_circuits = standard_RB.transpiled_circuits
    transpiled_circuits_as_ints = {}
    total_circuits_to_encode = sum(len(circuits) for circuits in transpiled_circuits.values())
    with tqdm(total=total_circuits_to_encode, desc="Encoding RB circuits to ints", unit="circ") as pbar:
        for length, circuits in transpiled_circuits.items():
            encoded = []
            for qc in circuits:
                encoded.append(circuit_to_layer_ints(qc))
                pbar.update(1)
            transpiled_circuits_as_ints[length] = encoded

    # to calculate the average number of 2q layers per Clifford
    average_layers_per_clifford = np.mean(
        [
            np.mean([len(circ) for circ in circuits]) / np.array(length + 1)
            for length, circuits in transpiled_circuits_as_ints.items()
            if length > 0
        ]
    )

    circuits_as_ints = []
    for circuits_per_len in transpiled_circuits_as_ints.values():
        for circuit in circuits_per_len:
            circuits_as_ints.append(circuit + [READOUT_OPCODE])

    rb_save(
        cache_dir,
        cache_key,
        {
            "circuits_as_ints": circuits_as_ints,
            "average_layers_per_clifford": float(average_layers_per_clifford),
        },
    )
    print(f"Computed and cached {len(circuits_as_ints)} RB circuits (key {cache_key[:12]})")

# %% {QUA_program}

num_pairs = len(qubit_pairs)

qua_program_handler = QuaProgramHandler(node, num_pairs, circuits_as_ints, node.machine, qubit_pairs)

rb = qua_program_handler.get_qua_program()
node.namespace = {"qua_program" : rb}

if node.parameters.use_input_stream:
    print(f"Total input-stream pushes (all qubit pairs): {qua_program_handler.n_sub_chunks} x {num_pairs} = {qua_program_handler.total_pushes}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns//4)  # in clock cycles
    job = qmm.simulate(config, rb, simulation_config)
    samples = job.get_simulated_samples()

elif node.parameters.load_data_id is None:
    # Prepare data for saving
    node.results = {}
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    
    with qm_session(node.machine.qmm, config, timeout=node.parameters.timeout) as qm:
        if node.parameters.use_input_stream:
            job = qm.execute(rb)
            qua_program_handler.push_all_chunks(job)
        else:
            job = qm.execute(rb)

        results = fetching_tool(job, ["iteration"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar: with input streams, "iteration" counts host pushes
            # (advance_input_stream calls, i.e. depth/sub-chunk boundaries) across all
            # qubit pairs. Without input streams, it counts completed averages.
            if node.parameters.use_input_stream:
                progress_counter(n, qua_program_handler.total_pushes, start_time=results.start_time)
            else:
                progress_counter(n, node.parameters.num_averages, start_time=results.start_time)

# %% {Plot_sequence}

for num in flatten(circuits_as_ints):
    print(gate_mapping[num])
    
# %% {Plot and save if simulation}
if node.parameters.simulate:
    qubit_names = [qubit_pair.qubit_control.name for qubit_pair in qubit_pairs] + [qubit_pair.qubit_target.name for qubit_pair in qubit_pairs]
    readout_lines = set([q[1] for q in qubit_names])
    fig = plot_samples(samples, qubit_names, readout_lines=list(readout_lines), xlim=(0,10000))
    
    # node.results["figure"] = fig
    # node.save()

 # %% {Data_fetching_and_dataset_creation}
if node.parameters.load_data_id is None:
    if node.parameters.use_input_stream:
        # With input streams, the QUA program advances one (depth) chunk at a time and
        # replays every shot before advancing, so the raw save order is depth-major,
        # then shot, then sequence (see QuaProgramHandler._get_qua_program_with_input_stream).
        # fetch_results_as_xarray reverses the dict order, so pass it reversed here too.
        measurement_axis = {
            "sequence": range(node.parameters.num_circuits_per_length),
            "shots": range(node.parameters.num_averages),
            "depths": list(node.parameters.circuit_lengths),
        }
    else:
        measurement_axis = {
            "sequence": range(node.parameters.num_circuits_per_length),
            "depths": list(node.parameters.circuit_lengths),
            "shots": range(node.parameters.num_averages),
        }
    ds = fetch_results_as_xarray(
        job.result_handles,
        qubit_pairs,
        measurement_axis,
    )
else:
    load_data_id = node.parameters.load_data_id
    node = node.load_from_id(load_data_id)
    ds = node.results["ds"]
    restore_load_data_id(node, load_data_id)
    machine = node.machine
    qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
# Add the dataset to the node
node.results = {"ds": ds}
# %% {Data_analysis and plotting}

state_probabilities = build_two_qubit_state_probabilities(ds)
state_probabilities = state_probabilities.rename(
    {"shots": "average", "sequence": "repeat", "depths": "circuit_depth"}
)
state_probabilities = state_probabilities.transpose("qubit", "outcome", "repeat", "circuit_depth", "average")

ds_transposed = ds.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
ds_transposed = ds_transposed.transpose("qubit", "repeat", "circuit_depth", "average")

node.results["state_probabilities"] = state_probabilities.to_dataset(name="probability")

rb_result = {}

for qp in qubit_pairs:
    rb_result[qp.id] = LeakageRBResult(
        circuit_depths=list(node.parameters.circuit_lengths),
        num_repeats=node.parameters.num_circuits_per_length,
        num_averages=node.parameters.num_averages,
        state=ds_transposed.sel(qubit=qp.name).state.data,
    )
    rb_result[qp.id].fit()

    fig = rb_result[qp.id].plot(title=f"2Q leakage RB - {qp.name}")
    fig.show()
    node.results[f"{qp.id}_figure_RB_decay"] = fig

    lrb_fig = rb_result[qp.id].plot_lrb(title=f"Wood-Gambetta LRB - {qp.name}")
    lrb_fig.show()
    node.results[f"{qp.id}_figure_LRB"] = lrb_fig

    state_distribution_fig = plot_two_qubit_state_probabilities(
        state_probabilities.sel(qubit=qp.name),
        qp.name,
    )
    state_distribution_fig.show()
    node.results[f"{qp.id}_figure_state_distribution"] = state_distribution_fig

# %% {Update_state}
with node.record_state_updates():
    for qp in qubit_pairs:
        result = rb_result[qp.id]
        qp.extras["LeakageRB"] = {
            "fidelity": float(result.fidelity),
            "fidelity_err": float(result.fidelity_err),
            "error_per_clifford": float(1.0 - result.fidelity),
            "lambda": float(result.lam),
            "lambda_err": float(result.lam_err),
            "tau": float(result.tau),
            "tau_err": float(result.tau_err),
            "naive_fidelity": float(result.naive_fidelity),
            "naive_alpha": float(result.naive_alpha),
            "wood_L1": float(result.L1),
            "wood_L1_err": float(result.L1_err),
            "wood_L2": float(result.L2),
            "wood_L2_err": float(result.L2_err),
            "wood_lambda1": float(result.lrb_lam1),
            "wood_lambda2": float(result.lrb_lam2),
            "wood_fidelity": float(result.lrb_fidelity),
            "wood_fidelity_err": float(result.lrb_fidelity_err),
            "wood_plateau_resolved": bool(result.lrb_plateau_resolved),
        }
# %% {Save_results}
node.save()

# %%
