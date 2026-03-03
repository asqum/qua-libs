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
    qubit_groups_raw = [[machine.qubits[q] for q in group] for group in node.parameters.qubit_groups]

# Define a helper class for qubit groups (generalized for N qubits)
class QubitGroup:
    def __init__(self, qubits):
        self.qubits = qubits  # List of qubit objects
        self.num_qubits = len(qubits)
        self.name = "-".join([q.name for q in qubits])
        # For backward compatibility and state saving, expose first two qubits
        if len(qubits) >= 1:
            self.qubit_A = qubits[0]
        if len(qubits) >= 2:
            self.qubit_B = qubits[1]
        if len(qubits) >= 3:
            self.qubit_C = qubits[2]

qubit_groups = [QubitGroup(q) for q in qubit_groups_raw]
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

# Helper function to generate state label from integer
def state_to_label(state_int, num_qubits):
    """Convert integer state to binary string label."""
    return format(state_int, f'0{num_qubits}b')

# Create QUA program dynamically based on number of qubits
# Note: QUA doesn't support truly dynamic variable creation, so we'll use explicit
# nested structures for common cases (1-5 qubits)

# Validate number of qubits
if num_qubits < 1 or num_qubits > 5:
    raise ValueError(f"Number of qubits ({num_qubits}) must be between 1 and 5")


with program() as ConfusionMatrixNQ:
    n = declare(int)
    n_st = declare_stream()
    
    # Declare init variables for each qubit position (support up to 5 qubits)
    init_vars = [declare(int) for _ in range(5)]
    
    # Declare state variables and streams for each group
    state_vars = [[declare(int) for _ in range(5)] for _ in range(num_qubit_groups)]
    state_st_vars = [[declare_stream() for _ in range(5)] for _ in range(num_qubit_groups)]
    state = [declare(int) for _ in range(num_qubit_groups)]
    state_st = [declare_stream() for _ in range(num_qubit_groups)]
    
    for i, qg in enumerate(qubit_groups):
        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            # qp.apply_mutual_flux_point()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()
        wait(1000)

        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st)
            
            # Create nested loops based on number of qubits (explicit for 1-5 qubits)
            if num_qubits == 1:
                with for_(*from_array(init_vars[0], [0, 1])):
                    if node.parameters.reset_type == "active":
                        active_reset(qg.qubits[0])
                        align()
                    else:
                        wait(5 * qg.qubits[0].thermalization_time * u.ns)
                    align()
                    with if_(init_vars[0] == 1):
                        qg.qubits[0].xy.play("x180")
                    align()
                    readout_state(qg.qubits[0], state_vars[i][0])
                    save(state_vars[i][0], state_st_vars[i][0])
                    assign(state[i], state_vars[i][0])
                    save(state[i], state_st[i])
            elif num_qubits == 2:
                with for_(*from_array(init_vars[0], [0, 1])):
                    with for_(*from_array(init_vars[1], [0, 1])):
                        if node.parameters.reset_type == "active":
                            active_reset(qg.qubits[0])
                            active_reset(qg.qubits[1])
                            align()
                        else:
                            wait(5 * qg.qubits[0].thermalization_time * u.ns)
                        align()
                        with if_(init_vars[0] == 1):
                            qg.qubits[0].xy.play("x180")
                        with if_(init_vars[1] == 1):
                            qg.qubits[1].xy.play("x180")
                        align()
                        readout_state(qg.qubits[0], state_vars[i][0])
                        readout_state(qg.qubits[1], state_vars[i][1])
                        save(state_vars[i][0], state_st_vars[i][0])
                        save(state_vars[i][1], state_st_vars[i][1])
                        assign(state[i], state_vars[i][0] * 2 + state_vars[i][1])
                        save(state[i], state_st[i])
            elif num_qubits == 3:
                with for_(*from_array(init_vars[0], [0, 1])):
                    with for_(*from_array(init_vars[1], [0, 1])):
                        with for_(*from_array(init_vars[2], [0, 1])):
                            if node.parameters.reset_type == "active":
                                active_reset(qg.qubits[0])
                                active_reset(qg.qubits[1])
                                active_reset(qg.qubits[2])
                                align()
                            else:
                                wait(5 * qg.qubits[0].thermalization_time * u.ns)
                            align()
                            with if_(init_vars[0] == 1):
                                qg.qubits[0].xy.play("x180")
                            with if_(init_vars[1] == 1):
                                qg.qubits[1].xy.play("x180")
                            with if_(init_vars[2] == 1):
                                qg.qubits[2].xy.play("x180")
                            align()
                            readout_state(qg.qubits[0], state_vars[i][0])
                            readout_state(qg.qubits[1], state_vars[i][1])
                            readout_state(qg.qubits[2], state_vars[i][2])
                            save(state_vars[i][0], state_st_vars[i][0])
                            save(state_vars[i][1], state_st_vars[i][1])
                            save(state_vars[i][2], state_st_vars[i][2])
                            assign(state[i], state_vars[i][0] * 4 + state_vars[i][1] * 2 + state_vars[i][2])
                            save(state[i], state_st[i])
            elif num_qubits == 4:
                with for_(*from_array(init_vars[0], [0, 1])):
                    with for_(*from_array(init_vars[1], [0, 1])):
                        with for_(*from_array(init_vars[2], [0, 1])):
                            with for_(*from_array(init_vars[3], [0, 1])):
                                if node.parameters.reset_type == "active":
                                    for q in qg.qubits:
                                        active_reset(q)
                                    align()
                                else:
                                    wait(5 * qg.qubits[0].thermalization_time * u.ns)
                                align()
                                for idx, q in enumerate(qg.qubits):
                                    with if_(init_vars[idx] == 1):
                                        q.xy.play("x180")
                                align()
                                for idx, q in enumerate(qg.qubits):
                                    readout_state(q, state_vars[i][idx])
                                    save(state_vars[i][idx], state_st_vars[i][idx])
                                state_val = state_vars[i][0] * 8 + state_vars[i][1] * 4 + state_vars[i][2] * 2 + state_vars[i][3]
                                assign(state[i], state_val)
                                save(state[i], state_st[i])
            elif num_qubits == 5:
                with for_(*from_array(init_vars[0], [0, 1])):
                    with for_(*from_array(init_vars[1], [0, 1])):
                        with for_(*from_array(init_vars[2], [0, 1])):
                            with for_(*from_array(init_vars[3], [0, 1])):
                                with for_(*from_array(init_vars[4], [0, 1])):
                                    if node.parameters.reset_type == "active":
                                        for q in qg.qubits:
                                            active_reset(q)
                                        align()
                                    else:
                                        wait(5 * qg.qubits[0].thermalization_time * u.ns)
                                    align()
                                    for idx, q in enumerate(qg.qubits):
                                        with if_(init_vars[idx] == 1):
                                            q.xy.play("x180")
                                    align()
                                    for idx, q in enumerate(qg.qubits):
                                        readout_state(q, state_vars[i][idx])
                                        save(state_vars[i][idx], state_st_vars[i][idx])
                                    state_val = (state_vars[i][0] * 16 + state_vars[i][1] * 8 + 
                                               state_vars[i][2] * 4 + state_vars[i][3] * 2 + state_vars[i][4])
                                    assign(state[i], state_val)
                                    save(state[i], state_st[i])
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_groups):
            qg = qubit_groups[i]
            # Create buffer chain: buffer(2) for each qubit, then buffer(n_shots)
            buffer_chain = state_st[i]
            for _ in range(qg.num_qubits):
                buffer_chain = buffer_chain.buffer(2)
            buffer_chain.buffer(n_shots).save(f"state{i + 1}")

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
        # Build the axes dictionary dynamically based on number of qubits
        # Reverse order: innermost to outermost (matching QUA loop order)
        # For 3 qubits: init_C (innermost), init_B, init_A (outermost)
        axes_dict = {}
        for q_idx in range(num_qubits - 1, -1, -1):
            axes_dict[f"init_{q_idx}"] = [0, 1]
        axes_dict["N"] = np.linspace(1, n_shots, n_shots)
        
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes
        # Only fetch the 'state' handles, not individual qubit states
        ds = fetch_results_as_xarray(job.result_handles, qubit_groups, axes_dict)
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}
    
# %%
if not node.parameters.simulate:
    # Generate states and labels dynamically
    states = list(range(num_states))
    state_labels = [state_to_label(s, num_qubits) for s in states]

    confusions = {}
    kron_confs = {}
    for qg in qubit_groups:
        conf = []
        
        # Generate all possible prepared states
        def generate_prepared_states(level, current_state, init_values):
            """Recursively generate all prepared states."""
            if level == num_qubits:
                # Calculate prepared state integer
                prepared_state = 0
                for idx, val in enumerate(init_values):
                    prepared_state = prepared_state * 2 + val
                
                # Count measurements for this prepared state
                row = []
                sel_dict = {f"init_{idx}": val for idx, val in enumerate(init_values)}
                for measured_state in states:
                    count = (ds.sel(qubit=qg.name).state.sel(**sel_dict) == measured_state).sum().values
                    row.append(count)
                conf.append(row)
            else:
                for val in [0, 1]:
                    generate_prepared_states(level + 1, current_state, init_values + [val])
        
        generate_prepared_states(0, 0, [])
        confusions[qg.name] = np.array(conf) / node.parameters.num_shots
        
        # import itertools
        # import numpy as np

        # states = np.arange(2**num_qubits)

        # conf = []
        # for init_values in itertools.product([0, 1], repeat=num_qubits):
        #     sel_dict = {f"init_{idx}": val for idx, val in enumerate(init_values)}
        #     measured = ds.sel(qubit=qg.name).state.sel(**sel_dict).values  # shape (n_shots,)

        #     # histogram counts of 0..(2^N-1)
        #     row = np.bincount(measured.astype(int), minlength=2**num_qubits)
        #     conf.append(row)

        # confusions = np.array(conf) / n_shots

        # Compute Kronecker product confusion matrix
        conf_mat = np.array([[1.0]])
        for q in qg.qubits:
            conf_mat = np.kron(conf_mat, q.resonator.confusion_matrix)
        kron_confs[qg.name] = conf_mat

# %% {Plot_results}
if not node.parameters.simulate:
    # Organize plots in 3-column grid
    num_groups = len(qubit_groups)
    num_cols = 3
    num_rows = int(np.ceil(num_groups / num_cols))
    
    # Adaptive font size based on number of qubits
    # Smaller font for more qubits to prevent crowding
    if num_qubits <= 2:
        text_fontsize = 10
    elif num_qubits == 3:
        text_fontsize = 8
    elif num_qubits == 4:
        text_fontsize = 6
    else:  # 5+ qubits
        text_fontsize = 4
    
    fig_confusion = plt.figure(figsize=(5 * num_cols, 5 * num_rows))
    for idx, qg in enumerate(qubit_groups):
        ax = fig_confusion.add_subplot(num_rows, num_cols, idx + 1)
        print(qg.name)
        conf = confusions[qg.name]
        ax.pcolormesh(state_labels, state_labels, conf)
        for i in range(num_states):
            for j in range(num_states):
                if i == j:
                    ax.text(j, i, f"{100 * conf[i][j]:.1f}%", ha="center", va="center", color="k", fontsize=text_fontsize)
                else:
                    ax.text(j, i, f"{100 * conf[i][j]:.1f}%", ha="center", va="center", color="w", fontsize=text_fontsize)
        ax.set_ylabel('prepared')
        ax.set_xlabel('measured')
        # ax.set_title(f"Confusion matrix {qg.name} ({num_qubits}Q) \n {node.date_time} GMT+{node.time_zone} #{node.node_id} \n reset type = {node.parameters.reset_type}")
        ax.set_title(f"Confusion matrix {qg.name} ({num_qubits}Q) \n reset type = {node.parameters.reset_type}")
    fig_confusion.tight_layout()
    fig_confusion.show()
    node.results["figure_confusion"] = fig_confusion

    # Kronecker product confusion matrix plot
    fig_kron = plt.figure(figsize=(5 * num_cols, 5 * num_rows))
    for idx, qg in enumerate(qubit_groups):
        ax = fig_kron.add_subplot(num_rows, num_cols, idx + 1)
        conf = kron_confs[qg.name]
        ax.pcolormesh(state_labels, state_labels, conf)
        for i in range(num_states):
            for j in range(num_states):
                if i == j:
                    ax.text(j, i, f"{100 * conf[i][j]:.1f}%", ha="center", va="center", color="k", fontsize=text_fontsize)
                else:
                    ax.text(j, i, f"{100 * conf[i][j]:.1f}%", ha="center", va="center", color="w", fontsize=text_fontsize)
        ax.set_ylabel('prepared')
        ax.set_xlabel('measured')
        ax.set_title(f"Kronecker Confusion matrix {qg.name} ({num_qubits}Q) \n reset type = {node.parameters.reset_type}")
        # ax.set_title(f"Kronecker Confusion matrix {qg.name} ({num_qubits}Q) \n {node.date_time} GMT+{node.time_zone} #{node.node_id} \n reset type = {node.parameters.reset_type}")
    fig_kron.tight_layout()
    fig_kron.show()
    node.results["figure_kron"] = fig_kron

    # Subtraction (difference) matrix plot
    fig_diff = plt.figure(figsize=(5 * num_cols, 5 * num_rows))
    for idx, qg in enumerate(qubit_groups):
        ax = fig_diff.add_subplot(num_rows, num_cols, idx + 1)
        diff = confusions[qg.name] - kron_confs[qg.name]
        max_abs = np.max(np.abs(diff))
        ax.pcolormesh(state_labels, state_labels, diff, cmap='RdBu', vmin=-max_abs, vmax=max_abs)
        for i in range(num_states):
            for j in range(num_states):
                val = diff[i][j]
                if abs(val) > 0.01:  # Annotate only significant differences
                    ax.text(j, i, f"{100 * val:.1f}%", ha="center", va="center", color="k" if abs(val) < 0.5 * max_abs else "w", fontsize=text_fontsize)
        ax.set_ylabel('prepared')
        ax.set_xlabel('measured')
        # ax.set_title(f"Difference (Direct - Kron) {qg.name} ({num_qubits}Q) \n {node.date_time} GMT+{node.time_zone} #{node.node_id} \n reset type = {node.parameters.reset_type}")
        ax.set_title(f"Difference (Direct - Kron) {qg.name} ({num_qubits}Q) \n reset type = {node.parameters.reset_type}")
    fig_diff.tight_layout()
    fig_diff.show()
    node.results["figure_diff"] = fig_diff
# %%

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
                        
                        # Initialize the group entry if it doesn't exist
                        group_name = qg.name
                        # Also try sorted version
                        qubit_names = [q.name for q in qg.qubits]
                        group_name_sorted = "-".join(sorted(qubit_names))
                        
                        # Try both names
                        for name_to_try in [group_name, group_name_sorted]:
                            if name_to_try not in qp.extras:
                                qp.extras[name_to_try] = {}
                            
                            # Save the confusion matrix with key based on number of qubits
                            confusion_key = f"confusion_{qg.num_qubits}q"
                            qp.extras[name_to_try][confusion_key] = confusions[qg.name].tolist()
                            break  # Only save once
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