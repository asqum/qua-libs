# %%
"""
Multi-Qubit GHZ State Measurement in Z Basis

This sequence measures the state distribution of N-qubit GHZ states (3, 4, or 5 qubits) in the Z basis. The process involves:

1. Preparing N qubits in a GHZ state (|00...0⟩ + |11...1⟩)/√2 or equivalent up to global phase
2. Performing simultaneous readout on all qubits
3. Calculating the probability distribution of measurement outcomes with readout error mitigation

For the prepared GHZ state, we measure:
1. The readout result of each qubit
2. The combined state

The measurement process involves:
1. Initializing all qubits to the ground state
2. Applying single-qubit gates and controlled-phase gates to prepare the GHZ state
3. Performing simultaneous readout on all qubits
4. Repeating the process multiple times to gather statistics

The outcome of this measurement will be used to:
1. Quantify the fidelity of multi-qubit GHZ states
2. Identify and characterize multi-qubit readout errors and crosstalk
3. Provide data for error mitigation in multi-qubit experiments

Prerequisites:
- Calibrated single-qubit gates for all qubits in the chain
- Calibrated controlled-phase gates for adjacent qubit pairs
- Calibrated readout for all qubits

Outcomes:
- Probability distribution over all possible N-qubit states (2^N states)
- Fidelity metrics for the N-qubit GHZ state preparation and measurement
- Comparison between Kronecker product and direct NQ confusion matrix mitigation
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubit_pairs_from_node,
)

from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List, Dict, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
from calibration_utils.ghz_zbasis import compute_zbasis_mitigated_results, plot_zbasis_distributions


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_groups: List[List[str]] = [["q1","q2", "q3"]]  # List of lists, each containing 3, 4, or 5 qubit names
    num_shots: int = 4000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = 'active'
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None


node = QualibrationNode(
    name="41a_GHZ_Zbasis", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine

# Get the relevant QuAM components

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
# %%

# %%

# Create qubit groups from Parameters.qubit_groups
# node.parameters.qubit_groups is List[List[str]], e.g., [["qD4","qD3","qC4","qC2","qC1"]]
qubit_objects_raw = [[machine.qubits[qubit] for qubit in group] for group in node.parameters.qubit_groups]
num_qubit_groups = len(qubit_objects_raw)

# Validate that all groups have the same number of qubits
if len(set(len(group) for group in qubit_objects_raw)) > 1:
    raise ValueError("All qubit groups must have the same number of qubits")
num_qubits = len(qubit_objects_raw[0])
num_states = 2 ** num_qubits

# Validate number of qubits
if num_qubits < 2 or num_qubits > 5:
    raise ValueError(f"Number of qubits must be between 3 and 5, got {num_qubits}")

@dataclass
class QubitGroup:
    qubits: List
    num_qubits: int
    name: str
    max_thermalization_time: int
    qubit_pairs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_qubits(cls, qubits, machine):
        qg = cls(
            qubits=qubits,
            num_qubits=len(qubits),
            name="-".join([q.name for q in qubits]),
            max_thermalization_time=max(q.thermalization_time for q in qubits),
        )
        for i in range(qg.num_qubits - 1):
            q1 = qubits[i]
            q2 = qubits[i + 1]
            pair_key = f"pair_{i}{i+1}"
            pair_name_1 = f"{q1.name}-{q2.name}"
            pair_name_2 = f"{q2.name}-{q1.name}"

            if pair_name_1 in machine.qubit_pairs:
                qg.qubit_pairs[pair_key] = machine.qubit_pairs[pair_name_1]
            elif pair_name_2 in machine.qubit_pairs:
                qg.qubit_pairs[pair_key] = machine.qubit_pairs[pair_name_2]
            else:
                for qp in machine.qubit_pairs.values():
                    if qp.qubit_control in [q1, q2] and qp.qubit_target in [q1, q2]:
                        qg.qubit_pairs[pair_key] = qp
                        break
        return qg


# Create qubit group structures for QUA program (needed for gate operations)
qubit_groups_for_qua = [QubitGroup.from_qubits(qubits, machine) for qubits in qubit_objects_raw]

# %% {QUA_program}
n_shots = node.parameters.num_shots  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

# Declare state variables for up to 5 qubits
with program() as GHZ_Zbasis_least_squares:
    n = declare(int)
    n_st = declare_stream()
    state_vars = [declare(int) for _ in range(num_qubits)]
    state = [declare(int) for _ in range(num_qubit_groups)]
    state_st = [declare_stream() for _ in range(num_qubit_groups)]
    
    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.apply_all_flux_to_joint_idle()
    wait(1000)
    
    for i, qg in enumerate(qubit_groups_for_qua):
        # Bring the active qubits to the minimum frequency point
        if flux_point != "joint":
            machine.apply_all_flux_to_min()
        align()
        
        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st)         
            # reset
            if node.parameters.reset_type == "active":
                for q in qg.qubits:
                    active_reset(q)
            else:
                wait(5 * qg.max_thermalization_time * u.ns)
            align()
            
            # GHZ chain preparation generalized for 3-5 qubits.
            qg.qubits[1].xy.play("y90")
            qg.qubits[0].xy.play("y90")
            qg.qubit_pairs["pair_01"].macros["cz"].apply()
            qg.qubits[0].xy.play("-y90")
            for qubit_idx in range(2, num_qubits):
                align()
                qg.qubits[qubit_idx].xy.play("y90")
                qg.qubit_pairs[f"pair_{qubit_idx - 1}{qubit_idx}"].macros["cz"].apply()
                qg.qubits[qubit_idx].xy.play("-y90")
            
            align()
            
            # Readout all qubits
            for idx, q in enumerate(qg.qubits):
                readout_state(q, state_vars[idx])
            
            # Compute combined state value from all readout bits.
            state_expr = state_vars[0]
            for idx in range(1, num_qubits):
                state_expr = state_expr * 2 + state_vars[idx]
            assign(state[i], state_expr)
            
            save(state[i], state_st[i])
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_groups):
            state_st[i].buffer(n_shots).save(f"state{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, GHZ_Zbasis_least_squares, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
        job = qm.execute(GHZ_Zbasis_least_squares)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_shots, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes
        ds = fetch_results_as_xarray(job.result_handles, qubit_groups_for_qua, {"N": np.linspace(1, n_shots, n_shots)})
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
    node.results = {"ds": ds}
    
# %% Analysis
if not node.parameters.simulate:
    states = list(range(num_states))

    raw_results, corrected_results_by_method, fidelities_by_method, fidelity_differences = compute_zbasis_mitigated_results(
        ds=ds,
        qubit_groups_for_qua=qubit_groups_for_qua,
        qubit_group_names=node.parameters.qubit_groups,
        machine=machine,
        num_states=num_states,
        num_shots=node.parameters.num_shots,
    )

    all_0_label = "0" * num_qubits
    all_1_label = "1" * num_qubits
    for qg in qubit_groups_for_qua:
        print(
            f"{qg.name} (Kron): Z-basis population fidelity ({all_0_label}+{all_1_label}): "
            f"{fidelities_by_method['kron'][qg.name]:.4f}"
        )
        if qg.name in fidelities_by_method["nq"]:
            print(
                f"{qg.name} ({num_qubits}Q): Z-basis population fidelity ({all_0_label}+{all_1_label}): "
                f"{fidelities_by_method['nq'][qg.name]:.4f}, Difference: {fidelity_differences[qg.name]:.4f}"
            )
        else:
            print(f"Warning: {num_qubits}Q confusion matrix not found for {qg.name}")

    # Backward-compatible aliases.
    corrected_results = corrected_results_by_method["kron"]
    corrected_results_nq = corrected_results_by_method["nq"]
    fidelities = fidelities_by_method["kron"]
    fidelities_nq = fidelities_by_method["nq"]


# %% Plotting
if not node.parameters.simulate:
    # Generate state labels dynamically
    state_labels = [format(s, f'0{num_qubits}b') for s in states]
    all_0_label = '0' * num_qubits
    all_1_label = '1' * num_qubits
    
    # Adaptive figure size based on number of states
    if num_qubits == 3:
        fig_width = 6
    elif num_qubits == 4:
        fig_width = 10
    else:  # 5 qubits
        fig_width = 12

    fig_kron = plot_zbasis_distributions(
        groups=qubit_groups_for_qua,
        corrected_data=corrected_results_by_method["kron"],
        fidelities=fidelities_by_method["kron"],
        state_labels=state_labels,
        num_qubits=num_qubits,
        all_0_label=all_0_label,
        all_1_label=all_1_label,
        title_fn=lambda qg: f"{qg.name} (Kronecker correction LS)",
        bar_color="skyblue",
        edge_color="navy",
        fidelity_box_facecolor="wheat",
        fig_width=fig_width,
    )
    plt.show()
    node.results["figure"] = fig_kron

    groups_with_nq = [qg for qg in qubit_groups_for_qua if qg.name in corrected_results_nq]
    if groups_with_nq:
        fig_nq = plot_zbasis_distributions(
            groups=groups_with_nq,
            corrected_data=corrected_results_by_method["nq"],
            fidelities=fidelities_by_method["nq"],
            state_labels=state_labels,
            num_qubits=num_qubits,
            all_0_label=all_0_label,
            all_1_label=all_1_label,
            title_fn=lambda qg: f"{qg.name} ({num_qubits}Q confusion matrix correction LS)",
            bar_color="moccasin",
            edge_color="orange",
            fidelity_box_facecolor="moccasin",
            fig_width=fig_width,
            fidelity_differences=fidelity_differences,
            delta_text_fn=lambda diff: (
                f"Δ Z-basis population fidelity ({num_qubits}Q - Kron): "
                f"{'+' if diff > 0 else ''}{diff:.4f}"
            ),
        )
        plt.show()
        node.results["figure_nq"] = fig_nq
    
    # Store results in node.results
    node.results["corrected_results"] = {k: v.tolist() for k, v in corrected_results.items()}
    node.results["fidelities"] = fidelities
    if corrected_results_nq:
        node.results["corrected_results_nq"] = {k: v.tolist() for k, v in corrected_results_nq.items()}
        node.results["fidelities_nq"] = fidelities_nq
        node.results["fidelity_differences"] = fidelity_differences

# %% {Update_state}

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()
