# %%
"""
Multi-Qubit Readout Confusion Matrix Measurement

This sequence measures the readout error when simultaneously measuring the state of N qubits. The process involves:

1. Preparing the qubits in all possible combinations of computational basis states (|00...0⟩ to |11...1⟩)
2. Performing simultaneous readout on all qubits
3. Calculating the confusion matrix based on the measurement results

For each prepared state, we measure:
- The readout result of each qubit

The measurement process involves:
1. Initializing all qubits to the ground state
2. Applying single-qubit gates to prepare the desired input state
3. Performing simultaneous readout on all qubits
4. Repeating the process multiple times to gather statistics

The outcome of this measurement will be used to:
1. Quantify the readout fidelity for multi-qubit states
2. Identify and characterize crosstalk effects in the readout process
3. Provide data for readout error mitigation in multi-qubit experiments

Prerequisites:
- Calibrated single-qubit gates for all qubits in the group
- Calibrated readout for all qubits

Outcomes:
- N×N confusion matrix (where N = 2^num_qubits) representing the probabilities of measuring each state given a prepared input state
- Readout fidelity metrics for simultaneous multi-qubit measurement
"""

# %% {Imports}

from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state

from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubits_from_node,
)
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from calibration_utils.confusion_matrix import (
    nested_binary_loops,
    state_to_label,
    compute_confusion_matrix,
    compute_kron_confusion_matrix,
    plot_matrix_figure,
)

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_groups: Optional[List[List[str]]] = [["q1", "q2", "q3"]]  # List of lists, each containing qubit names (can be any number of qubits)
    num_shots: int = 2000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    plot_raw: bool = False
    measure_leak: bool = False
    targets_name: str = "qubit_groups"


node = QualibrationNode(
    name="40d_confusion_matrix_general", parameters=Parameters()
)

assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubit_groups is None or node.parameters.qubit_groups == "":
    raise ValueError("qubit_groups must be provided")
else:
    qubit_objects = [[machine.qubits[q] for q in group] for group in node.parameters.qubit_groups]

# Define a helper class for qubit groups (generalized for N qubits).
@dataclass
class QubitGroup:
    qubits: List
    num_qubits: int
    name: str
    max_thermalization_time: int

    @classmethod
    def from_qubits(cls, qubits):
        if len(qubits) == 0:
            raise ValueError("Each qubit group must contain at least one qubit")
        return cls(
            qubits=qubits,
            num_qubits=len(qubits),
            name="-".join([q.name for q in qubits]),
            max_thermalization_time=max(q.thermalization_time for q in qubits),
        )

qubit_groups = [QubitGroup.from_qubits(q) for q in qubit_objects]
num_qubit_groups = len(qubit_groups)

# Validate that all groups have the same number of qubits
if len(set(qg.num_qubits for qg in qubit_groups)) > 1:
    raise ValueError("All qubit groups must have the same number of qubits")
num_qubits = qubit_groups[0].num_qubits
num_states = 2 ** num_qubits

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
# %%

####################
# Helper functions #
####################


# %% {QUA_program}
n_shots = node.parameters.num_shots  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

# Validate number of qubits
if num_qubits < 1 or num_qubits > 5:
    raise ValueError(f"Number of qubits ({num_qubits}) must be between 1 and 5")


with program() as ConfusionMatrixNQ:
    n = declare(int)
    n_st = declare_stream()

    init_vars = [declare(int) for _ in range(num_qubits)]
    state_vars = [declare(int) for _ in range(num_qubits)]
    state = [declare(int) for _ in range(num_qubit_groups)]
    state_st = [declare_stream() for _ in range(num_qubit_groups)]

    for i, qg in enumerate(qubit_groups):
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()
        wait(1000)

        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st)

            with nested_binary_loops(init_vars):
                if node.parameters.reset_type == "active":
                    for q in qg.qubits:
                        active_reset(q)
                else:
                    wait(5 * qg.max_thermalization_time * u.ns)
                align()

                for idx, q in enumerate(qg.qubits):
                    with if_(init_vars[idx] == 1):
                        q.xy.play("x180")
                align()

                for idx, q in enumerate(qg.qubits):
                    readout_state(q, state_vars[idx])

                state_expr = state_vars[0]
                for idx in range(1, num_qubits):
                    state_expr = state_expr * 2 + state_vars[idx]
                assign(state[i], state_expr)
                save(state[i], state_st[i])
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_groups):
            state_stream = state_st[i]
            for _ in range(num_qubits):
                state_stream = state_stream.buffer(2)
            state_stream.buffer(n_shots).save(f"state{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ConfusionMatrixNQ, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
        job = qm.execute(ConfusionMatrixNQ)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_shots, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        axes_dict = {f"init_{q_idx}": [0, 1] for q_idx in range(num_qubits - 1, -1, -1)}
        axes_dict["N"] = np.linspace(1, n_shots, n_shots)
        
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes
        # Only fetch the 'state' handles, not individual qubit states
        ds = fetch_results_as_xarray(job.result_handles, qubit_groups, axes_dict)
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_objects = [[machine.qubits[q] for q in group] for group in node.parameters.qubit_groups]
        qubit_groups = [QubitGroup.from_qubits(q) for q in qubit_objects]
    node.results = {"ds": ds}

if not node.parameters.simulate:
    states = np.arange(num_states)
    state_labels = [state_to_label(s, num_qubits) for s in states]

    confusions = {}
    kron_confs = {}
    for qg in qubit_groups:
        conf = compute_confusion_matrix(
            ds=ds,
            qg_name=qg.name,
            n_qubits=num_qubits,
            n_states=num_states,
            n_shots=node.parameters.num_shots,
        )
        confusions[qg.name] = conf
        kron_confs[qg.name] = compute_kron_confusion_matrix(qg.qubits)
        node.results[f"{qg.name}_mean_assignment_fidelity"] = np.trace(conf) / num_states

# %% {Plot_results}
if not node.parameters.simulate:
    # Show per-cell percentages only when still readable.
    annotate_cells = num_qubits <= 3
    text_fontsize = 11 if num_qubits <= 2 else 9

    fig_confusion = plot_matrix_figure(
        qubit_groups=qubit_groups,
        state_labels=state_labels,
        matrix_by_group=confusions,
        title_fn=lambda group_name, nq: (
            f"Confusion matrix {group_name} ({nq}Q) \n reset type = {node.parameters.reset_type}"
        ),
        num_qubits=num_qubits,
        annotate_cells=annotate_cells,
        text_fontsize=text_fontsize,
    )
    node.results["figure_confusion"] = fig_confusion

    # Kronecker product confusion matrix plot
    fig_kron = plot_matrix_figure(
        qubit_groups=qubit_groups,
        state_labels=state_labels,
        matrix_by_group=kron_confs,
        title_fn=lambda group_name, nq: (
            f"Kronecker Confusion matrix {group_name} ({nq}Q) \n reset type = {node.parameters.reset_type}"
        ),
        num_qubits=num_qubits,
        annotate_cells=annotate_cells,
        text_fontsize=text_fontsize,
    )
    node.results["figure_kron"] = fig_kron

    # Subtraction (difference) matrix plot
    diff_confs = {qg.name: confusions[qg.name] - kron_confs[qg.name] for qg in qubit_groups}
    fig_diff = plot_matrix_figure(
        qubit_groups=qubit_groups,
        state_labels=state_labels,
        matrix_by_group=diff_confs,
        title_fn=lambda group_name, nq: (
            f"Difference (Direct - Kron) {group_name} ({nq}Q) \n reset type = {node.parameters.reset_type}"
        ),
        num_qubits=num_qubits,
        annotate_cells=annotate_cells,
        text_fontsize=text_fontsize,
        cmap="RdBu",
        is_difference=True,
    )
    node.results["figure_diff"] = fig_diff

# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qg in qubit_groups:
                # Only save if we have at least 2 qubits (for qubit pair lookup)
                if qg.num_qubits >= 2:
                    # Get the first two qubits to determine the qubit pair
                    q1_name = qg.qubits[0].name
                    q2_name = qg.qubits[1].name
                    
                    # Try both possible pair name orders
                    pair_name_1 = f"coupler_{q1_name}_{q2_name}"
                    pair_name_2 = f"coupler_{q2_name}_{q1_name}"
                    
                    # Find the qubit pair in the machine
                    qp = None
                    if pair_name_1 in machine.qubit_pairs:
                        qp = machine.qubit_pairs[pair_name_1]
                    elif pair_name_2 in machine.qubit_pairs:
                        qp = machine.qubit_pairs[pair_name_2]
                    
                    if qp is not None:
                        # Initialize extras if it doesn't exist
                        if not hasattr(qp, 'extras') or qp.extras is None:
                            qp.extras = {}

                        group_name = qg.name
                        if group_name not in qp.extras:
                            qp.extras[group_name] = {}

                        confusion_key = f"confusion_{qg.num_qubits}q"
                        qp.extras[group_name][confusion_key] = confusions[qg.name].tolist()
                    else:
                        print(f"Warning: Qubit pair {pair_name_1} or {pair_name_2} not found in machine.qubit_pairs. Skipping confusion matrix save.")
                else:
                    print(f"Warning: Qubit group {qg.name} has less than 2 qubits. Cannot save to qubit pair.")
# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qg.name: "successful" for qg in qubit_groups}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
    node.machine.save()
# %%