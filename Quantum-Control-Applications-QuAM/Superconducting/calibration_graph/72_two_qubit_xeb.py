# %%
"""
Two-qubit XEB node.

This node wraps the XEB workflow from the notebook into a QualibrationNode so
that all saved outputs go through `node.results` / `node.save()`.
"""

# %%
from typing import Literal, Optional, List

import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qm.qua import align, wait
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode

from quam_libs.components import QuAM, TransmonPair
from quam_libs.experiments.two_qubit_xeb import (
    QUAGate,
    XEB,
    XEBConfig,
    backend as fake_backend,
)


u = unit(coerce_to_integer=True)


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q4_q5"]
    readout_qubits: Optional[List[str]] = None
    seqs: int = 50
    depths: list[int] = list(range(1, 40, 1))
    n_shots: int = 1000
    baseline_gate_name: str = "x90"
    gate_set_choice: Literal["sw", "t"] = "sw"
    two_qubit_gate: Literal["none", "cz"] = "cz"
    reset_method: Literal["active", "cooldown"] = "active"
    reset_cooldown_time_ns: int = 400000
    reset_max_tries: int = 3
    reset_pi_pulse: str = "x180"
    execution_mode: Literal["hardware", "hardware_simulation", "qiskit_simulation"] = "qiskit_simulation"
    simulation_duration_ns: int = 200000
    generate_new_data: bool = True
    disjoint_processing: bool = False
    apply_confusion_matrix: bool = True
    seed: int = 1234
    timeout: int = 100
    load_data_id: Optional[int] = None
    targets_name = "qubit_pairs"


node = QualibrationNode(name="72_two_qubit_xeb", parameters=Parameters())


# %% {Helpers}
def cz_gate(qubit_pair: TransmonPair):
    qubit_pair.align()
    # wait(120 * u.ns)
    qubit_pair.gates["Cz"].execute()
    # wait(120 * u.ns)
    align()


def unique_qubits_from_pairs(qubit_pairs: list[TransmonPair]):
    qubits = []
    for pair in qubit_pairs:
        for qubit in (pair.qubit_control, pair.qubit_target):
            if qubit not in qubits:
                qubits.append(qubit)
    return qubits


def serialize_linear_fidelities(result):
    if result.xeb_config.disjoint_processing:
        serialized = {}
        for qubit_name, fidelity in zip(result.qubit_names, result.linear_fidelities):
            serialized[qubit_name] = {
                "depth": np.asarray(fidelity["depth"]),
                "fidelity": np.asarray(fidelity["fidelity"]),
            }
        return serialized

    return {
        "depth": np.asarray(result.linear_fidelities["depth"]),
        "fidelity": np.asarray(result.linear_fidelities["fidelity"]),
    }


def add_result_entry(results_dict, key_prefix, value):
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            add_result_entry(results_dict, f"{key_prefix}_{sub_key}", sub_value)
    elif isinstance(value, (np.ndarray, np.generic, int, float, str, bool)):
        results_dict[key_prefix] = value
    elif value is None:
        results_dict[key_prefix] = value


def qubit_names(qubits):
    return [qubit.name if hasattr(qubit, "name") else str(qubit) for qubit in qubits]


def print_xeb_setup(xeb_config: XEBConfig, qubit_pairs: list[TransmonPair], target_qubits):
    print("target_qubits: %s" % qubit_names(target_qubits))

    tot_pts = xeb_config.seqs * len(xeb_config.depths) * xeb_config.n_shots
    print("Number of points: %s" % tot_pts)

    if xeb_config.reset_method == "cooldown":
        base_runtime_min = tot_pts / (150 * 27 * 700) * 19.35
        cooldown_factor = node.parameters.reset_cooldown_time_ns / 40000
        print("time required: %s min" % (cooldown_factor * base_runtime_min))
    else:
        expected_run_time_sec = (tot_pts * (max(1, len(target_qubits)) / 2) / 48640000) * 632.3
        print("time required roughly: %s min" % (expected_run_time_sec / 60))

    for qubit_pair in qubit_pairs:
        print("qubit_control: %s" % qubit_pair.qubit_control)
        print("qubit_target: %s" % qubit_pair.qubit_target)
        if hasattr(qubit_pair.qubit_target, "id"):
            print(qubit_pair.qubit_target.id)

    print("Qubits: %s" % ", ".join(qubit_names(target_qubits)))
    print("sequences: %s" % xeb_config.seqs)
    print("depths: %s" % xeb_config.depths)
    print("shots: %s" % xeb_config.n_shots)
    print("CZ: %s" % xeb_config.two_qb_gate)
    print("apply_confusion_matrix: %s" % xeb_config.apply_confusion_matrix)
    print("XEB raw data saving: %s" % xeb_config.should_save_data)
    if xeb_config.should_save_data:
        print("saving data under: %s" % xeb_config.save_dir)
    else:
        print("Qualibrate node results will be saved with node.save()")


def print_xeb_result_summary(xeb_config: XEBConfig, result):
    print("sequences: %s" % xeb_config.seqs)
    print("shots: %s" % xeb_config.n_shots)
    print("Depth: %s" % xeb_config.depths)
    print("CZ: %s" % xeb_config.two_qb_gate)
    print("apply_confusion_matrix: %s" % xeb_config.apply_confusion_matrix)
    print(result.singularities)
    print("Singularities: {:.3f}%".format(len(result.singularities) / (xeb_config.seqs * len(xeb_config.depths)) * 100))
    print(result.linear_fidelities)


# %% {Initialize_QuAM}
node.machine = QuAM.load()

if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = node.machine.active_qubit_pairs
else:
    qubit_pairs = [node.machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

if len(qubit_pairs) == 0:
    raise ValueError("No qubit pairs selected")

target_qubits = unique_qubits_from_pairs(qubit_pairs)
if node.parameters.readout_qubits is None or node.parameters.readout_qubits == "":
    readout_qubits = node.machine.active_qubits
else:
    readout_qubits = [node.machine.qubits[q] for q in node.parameters.readout_qubits]

two_qubit_gate = QUAGate("cz", cz_gate) if node.parameters.two_qubit_gate == "cz" else None

reset_kwargs = {
    "max_tries": node.parameters.reset_max_tries,
    "pi_pulse": node.parameters.reset_pi_pulse,
}
if node.parameters.reset_method == "cooldown":
    reset_kwargs["cooldown_time"] = node.parameters.reset_cooldown_time_ns

xeb_config = XEBConfig(
    seqs=node.parameters.seqs,
    depths=node.parameters.depths,
    n_shots=node.parameters.n_shots,
    readout_qubits=readout_qubits,
    qubits=target_qubits,
    qubit_pairs=qubit_pairs,
    baseline_gate_name=node.parameters.baseline_gate_name,
    gate_set_choice=node.parameters.gate_set_choice,
    two_qb_gate=two_qubit_gate,
    save_dir="",
    should_save_data=False,
    data_folder_name=None,
    generate_new_data=node.parameters.generate_new_data,
    disjoint_processing=node.parameters.disjoint_processing,
    apply_confusion_matrix=node.parameters.apply_confusion_matrix,
    reset_method=node.parameters.reset_method,
    reset_kwargs=reset_kwargs,
    seed=node.parameters.seed,
)

node.results = {
    "initial_parameters": node.parameters.model_dump(),
    "xeb_config": xeb_config.as_dict(),
    "qubit_names": np.array([q.name for q in target_qubits]),
    "readout_qubit_names": np.array([q.name for q in readout_qubits]),
    "qubit_pair_names": np.array([qp.name for qp in qubit_pairs]),
}

print_xeb_setup(xeb_config, qubit_pairs, target_qubits)


# %% {Run_or_load}
if node.parameters.load_data_id is not None:
    print("Loading from previous data")
    loaded_node = node.load_from_id(node.parameters.load_data_id)
    if loaded_node is None:
        raise ValueError(f"Could not load node with id {node.parameters.load_data_id}")
    node = loaded_node
else:
    print("Loading from new data")
    print("Qubits: %s" % ", ".join(qubit_names(target_qubits)))
    xeb = XEB(xeb_config, machine=node.machine)
    node.namespace = {"xeb": xeb}

    if node.parameters.execution_mode == "qiskit_simulation":
        job = xeb.simulate(backend=fake_backend)
    elif node.parameters.execution_mode == "hardware_simulation":
        job = xeb.run(
            simulate=True,
            simulation_config=SimulationConfig(duration=node.parameters.simulation_duration_ns),
        )
        job.plot_simulated_samples()
        node.results["figure_simulated_samples"] = plt.gcf()
    else:
        job = xeb.run(simulate=False)

    result = job.result(disjoint_processing=node.parameters.disjoint_processing)
    print_xeb_result_summary(xeb_config, result)

    node.results["log_fidelities"] = result.log_fidelities
    node.results["singularities"] = result.singularities
    node.results["outliers"] = result.outliers
    node.results["linear_fidelities"] = serialize_linear_fidelities(result)
    node.results["layer_fidelity_linear"] = result.get_layer_fidelity("linear")
    node.results["layer_fidelity_log"] = result.get_layer_fidelity("log")

    for key, value in result.data.items():
        add_result_entry(node.results, key, value)

    for i, fig in enumerate(result.plot_state_heatmap()):
        node.results[f"figure_state_heatmap_{i}"] = fig

    for i, fig in enumerate(result.plot_records()):
        node.results[f"figure_records_{i}"] = fig

    for i, fig in enumerate(result.plot_fidelities(fit_linear=False, fit_log_entropy=True, separate_plots=True)):
        node.results[f"figure_fidelity_log_{i}"] = fig

    for i, fig in enumerate(result.plot_fidelities(fit_linear=True, fit_log_entropy=False, separate_plots=True)):
        node.results[f"figure_fidelity_linear_{i}"] = fig

    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.save()

#%%
