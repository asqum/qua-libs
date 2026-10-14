"""
        TWO-QUBIT LEAKAGE RB, AVERAGE OVER MEASUREMENT BASIS
Each random Clifford sequence is measured four times. After the recovery gate, a final
II, X(target), X(control), or XX maps the ideal |00> onto |00>, |01>, |10>, or |11>.
p_avg is the average probability of landing on that state. GEF readout separates leakage:
retention P(state < 4) gives the leakage rate τ.

This is the computational-dominant Avg. MB analysis of Chen & Baldwin, arXiv:2502.00154
Sec. IV.2.2. p_avg = A r^ℓ + B with r = 1 - λ - τ, retention = A (1 - τ ℓ), and
F = 1 - (3/4)λ - τ. The formula assumes τℓ ≪ 1.

Readout is GEF. The program plays four times as many circuits as num_random_circuits.
"""

# %%

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Literal, Optional
import numpy as np
from tqdm.auto import tqdm

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter, fetching_tool

from qualibrate import NodeParameters, QualibrationNode
from quam_libs.experiments.rb_standard.circuit_utils import circuit_to_layer_ints
from quam_libs.experiments.rb_standard.qua_utils import QuaProgramHandler
from quam_libs.experiments.rb_standard.data_utils import AvgMBLeakageRBResult
from quam_libs.lib.plot_utils import plot_samples
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubit_pairs_from_node,
)
from quam_libs.components import QuAM
from quam_libs.experiments.rb_standard.rb_utils import StandardRB, rb_cache_key, rb_save, rb_try_load

# %% {Node_parameters}

class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q4_q5"]
    circuit_lengths: tuple[int] = (0, 1, 2 ,3, 4,5, 6, 7, 8, 12, 16, 20, 25, 30, 40)
    num_random_circuits: int = 30
    # Set from num_random_circuits * 4 before the QUA program is built.
    num_circuits_per_length: int = 120
    num_averages: int = 100
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

node = QualibrationNode[Parameters, QuAM](name="71b_two_qubit_avg_mb_leakage_rb", parameters=Parameters())


# state = state_control * 2 + state_target. A final X flips |0> to |1> in the computational basis.
# II -> |00> = 0, X(target) -> |01> = 1, X(control) -> |10> = 2, XX -> |11> = 3.
READOUT_OPCODE = 66
FINAL_BASIS_LAYERS = (
    [],
    [57],
    [15],
    [9],
)


def expand_with_final_bases(circuits_by_length, circuit_lengths) -> list[list[int]]:
    expanded = []
    for length in circuit_lengths:
        for circuit in circuits_by_length[int(length)]:
            body = list(circuit)
            if body and body[-1] == READOUT_OPCODE:
                body = body[:-1]
            for layers in FINAL_BASIS_LAYERS:
                expanded.append(body + list(layers) + [READOUT_OPCODE])
    return expanded


# %% {Initialize_QuAM_and_QOP}

node.machine = QuAM.load()

if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = node.machine.active_qubit_pairs
else:
    qubit_pairs = [node.machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

if len(qubit_pairs) == 0:
    raise ValueError("No qubit pairs selected")

if node.parameters.load_data_id is None:
    qmm = node.machine.connect(timeout=node.parameters.timeout)

config = node.machine.generate_config()

# %% {Random circuit generation}

n_bases = len(FINAL_BASIS_LAYERS)
n_random = node.parameters.num_random_circuits
node.parameters.num_circuits_per_length = n_random * n_bases
circuit_lengths = list(node.parameters.circuit_lengths)

cache_key = rb_cache_key(
    node.parameters.seed,
    circuit_lengths,
    n_random,
    target_gate="avg_mb",
)
cache_dir = Path(__file__).resolve().parent.parent / ".rb_cache"
cached = rb_try_load(cache_dir, cache_key)

print(
    f"Avg. MB config: {len(circuit_lengths)} depths, {n_random} random circuits/depth, "
    f"{n_bases} final bases -> {node.parameters.num_circuits_per_length} played circuits/depth"
)

if cached is not None:
    circuits_by_length = {int(k): v for k, v in cached["circuits_by_length"].items()}
    average_layers_per_clifford = cached["average_layers_per_clifford"]
    print(f"Loaded cached RB circuits (key {cache_key[:12]})")
else:
    standard_RB = StandardRB(
        amplification_lengths=circuit_lengths,
        num_circuits_per_length=n_random,
        basis_gates=node.parameters.basis_gates,
        reduce_to_1q_cliffords=node.parameters.reduce_to_1q_cliffords,
        num_qubits=2,
        seed=node.parameters.seed,
    )
    circuits_by_length = {}
    total_circuits_to_encode = sum(len(circuits) for circuits in standard_RB.transpiled_circuits.values())
    with tqdm(total=total_circuits_to_encode, desc="Encoding RB circuits to ints", unit="circ") as pbar:
        for length, circuits in standard_RB.transpiled_circuits.items():
            encoded = []
            for qc in circuits:
                encoded.append(circuit_to_layer_ints(qc))
                pbar.update(1)
            circuits_by_length[int(length)] = encoded

    average_layers_per_clifford = float(np.mean(
        [
            np.mean([len(circ) for circ in circuits]) / np.array(length + 1)
            for length, circuits in circuits_by_length.items()
            if length > 0
        ]
    ))
    rb_save(
        cache_dir,
        cache_key,
        {
            "circuits_by_length": circuits_by_length,
            "average_layers_per_clifford": average_layers_per_clifford,
        },
    )
    print(f"Computed and cached RB circuits (key {cache_key[:12]})")

circuits_as_ints = expand_with_final_bases(circuits_by_length, circuit_lengths)
if len(circuits_as_ints) != len(circuit_lengths) * node.parameters.num_circuits_per_length:
    raise ValueError(
        f"Expanded circuit count {len(circuits_as_ints)} does not match "
        f"{len(circuit_lengths)} depths x {node.parameters.num_circuits_per_length} circuits"
    )

print(f"Layers per Clifford (before the final basis gate): {average_layers_per_clifford:.2f}")
print("Final layers, in order, and the accepted computational state:")
print("  II -> |00> (0),  X(target) -> |01> (1),  X(control) -> |10> (2),  XX -> |11> (3)")

# %% {QUA_program}

num_pairs = len(qubit_pairs)
qua_program_handler = QuaProgramHandler(node, num_pairs, circuits_as_ints, node.machine, qubit_pairs)
rb = qua_program_handler.get_qua_program()
node.namespace = {"qua_program": rb}

if node.parameters.use_input_stream:
    print(
        f"Total input-stream pushes (all qubit pairs): "
        f"{qua_program_handler.n_sub_chunks} x {num_pairs} = {qua_program_handler.total_pushes}"
    )

# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)
    job = qmm.simulate(config, rb, simulation_config)
    samples = job.get_simulated_samples()

elif node.parameters.load_data_id is None:
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
            n = results.fetch_all()[0]
            if node.parameters.use_input_stream:
                progress_counter(n, qua_program_handler.total_pushes, start_time=results.start_time)
            else:
                progress_counter(n, node.parameters.num_averages, start_time=results.start_time)

# %% {Plot and save if simulation}
if node.parameters.simulate:
    qubit_names = [qubit_pair.qubit_control.name for qubit_pair in qubit_pairs] + [qubit_pair.qubit_target.name for qubit_pair in qubit_pairs]
    readout_lines = set([q[1] for q in qubit_names])
    fig = plot_samples(samples, qubit_names, readout_lines=list(readout_lines), xlim=(0, 10000))

# %% {Data_fetching_and_dataset_creation}
if node.parameters.load_data_id is None:
    if node.parameters.use_input_stream:
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

node.results = {"ds": ds}

# %% {Data_analysis and plotting}

ds_transposed = ds.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
ds_transposed = ds_transposed.transpose("qubit", "repeat", "circuit_depth", "average")

rb_result = {}
for qp in qubit_pairs:
    rb_result[qp.id] = AvgMBLeakageRBResult(
        circuit_depths=list(node.parameters.circuit_lengths),
        num_repeats=node.parameters.num_circuits_per_length,
        num_averages=node.parameters.num_averages,
        state=ds_transposed.sel(qubit=qp.name).state.data,
    )
    rb_result[qp.id].fit()
    fig = rb_result[qp.id].plot(title=f"2Q Avg. MB leakage RB - {qp.name}")
    fig.show()
    node.results[f"{qp.id}_figure_RB_decay"] = fig

# %% {Update_state}
with node.record_state_updates():
    for qp in qubit_pairs:
        result = rb_result[qp.id]
        qp.extras["AvgMBLeakageRB"] = {
            "fidelity": float(result.fidelity),
            "fidelity_err": float(result.fidelity_err),
            "error_per_clifford": float(1.0 - result.fidelity),
            "r": float(result.r),
            "r_err": float(result.r_err),
            "lambda": float(result.lam),
            "lambda_err": float(result.lam_err),
            "tau": float(result.tau),
            "tau_err": float(result.tau_err),
            "bound_infidelity": float(result.bound_infidelity),
        }

# %% {Save_results}
node.save()

# %%
