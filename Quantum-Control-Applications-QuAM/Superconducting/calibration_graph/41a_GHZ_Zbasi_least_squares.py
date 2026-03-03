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
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.readout_mitigation import get_nq_confusion_matrix, least_squares_mitigation

from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_groups: List[List[str]] = [["q1","q2","q3"]]  # List of lists, each containing 3, 4, or 5 qubit names
    num_shots: int = 2000
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
if num_qubits < 3 or num_qubits > 5:
    raise ValueError(f"Number of qubits must be between 3 and 5, got {num_qubits}")

# Create qubit group structures for QUA program (needed for gate operations)
qubit_groups_for_qua = []
for qubits in qubit_objects_raw:
    # Create a simple object to hold qubit group data
    qg = type('QubitGroup', (), {})()
    qg.qubits = qubits
    qg.num_qubits = len(qubits)
    qg.name = "-".join([q.name for q in qubits])
    
    # Expose individual qubits for backward compatibility
    qg.qubit_A = qubits[0]
    qg.qubit_B = qubits[1]
    qg.qubit_C = qubits[2]
    if num_qubits >= 4:
        qg.qubit_D = qubits[3]
    if num_qubits >= 5:
        qg.qubit_E = qubits[4]
    
    # Find qubit pairs for adjacent qubits
    # Check both orderings of pair names (e.g., "qD4-qD3" and "qD3-qD4")
    qg.qubit_pairs = {}
    for i in range(num_qubits - 1):
        q1 = qubits[i]
        q2 = qubits[i + 1]
        pair_key = f"pair_{i}{i+1}"
        # Try both orderings of the pair name
        pair_name_1 = f"{q1.name}-{q2.name}"
        pair_name_2 = f"{q2.name}-{q1.name}"
        
        # Check if either pair name exists in machine.qubit_pairs
        if pair_name_1 in machine.qubit_pairs:
            qg.qubit_pairs[pair_key] = machine.qubit_pairs[pair_name_1]
        elif pair_name_2 in machine.qubit_pairs:
            qg.qubit_pairs[pair_key] = machine.qubit_pairs[pair_name_2]
        else:
            # Fallback: search by qubit objects (in case pair structure is different)
            for qp_id, qp in machine.qubit_pairs.items():
                if (qp.qubit_control in [q1, q2] and 
                    qp.qubit_target in [q1, q2]):
                    qg.qubit_pairs[pair_key] = qp
                    break
    
    # For backward compatibility
    if "pair_01" in qg.qubit_pairs:
        qg.qubit_pair_AB = qg.qubit_pairs["pair_01"]
    if "pair_12" in qg.qubit_pairs:
        qg.qubit_pair_BC = qg.qubit_pairs["pair_12"]
    if "pair_23" in qg.qubit_pairs:
        qg.qubit_pair_CD = qg.qubit_pairs["pair_23"]
    if "pair_34" in qg.qubit_pairs:
        qg.qubit_pair_DE = qg.qubit_pairs["pair_34"]
    
    qubit_groups_for_qua.append(qg)

# %% {QUA_program}
n_shots = node.parameters.num_shots  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

# Declare state variables for up to 5 qubits
with program() as CPhase_Oscillations:
    n = declare(int)
    n_st = declare_stream()
    # State variables for each qubit (support up to 5 qubits)
    state_vars = [[declare(int) for _ in range(5)] for _ in range(num_qubit_groups)]
    state_st_vars = [[declare_stream() for _ in range(5)] for _ in range(num_qubit_groups)]
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
                wait(5*qg.qubit_A.thermalization_time * u.ns)
            align()
            
            # GHZ state preparation - generalized for 3, 4, or 5 qubits
            if num_qubits == 3:
                qg.qubit_B.xy.play("y90")
                qg.qubit_A.xy.play("y90")
                qg.qubit_pair_AB.macros['cz'].apply()
                qg.qubit_A.xy.play("-y90")
                align()
                qg.qubit_C.xy.play("y90")
                qg.qubit_pair_BC.macros['cz'].apply()
                qg.qubit_C.xy.play("-y90")
            elif num_qubits == 4:
                qg.qubit_B.xy.play("y90")
                qg.qubit_A.xy.play("y90")
                qg.qubit_pair_AB.macros['cz'].apply()
                qg.qubit_A.xy.play("-y90")
                align()
                qg.qubit_C.xy.play("y90")
                qg.qubit_pair_BC.macros['cz'].apply()
                qg.qubit_C.xy.play("-y90")
                align()
                qg.qubit_D.xy.play("y90")
                qg.qubit_pair_CD.macros['cz'].apply()
                qg.qubit_D.xy.play("-y90")
            elif num_qubits == 5:
                qg.qubit_B.xy.play("y90")
                qg.qubit_A.xy.play("y90")
                qg.qubit_pair_AB.macros['cz'].apply()
                qg.qubit_A.xy.play("-y90")
                align()
                qg.qubit_C.xy.play("y90")
                qg.qubit_pair_BC.macros['cz'].apply()
                qg.qubit_C.xy.play("-y90")
                align()
                qg.qubit_D.xy.play("y90")
                qg.qubit_pair_CD.macros['cz'].apply()
                qg.qubit_D.xy.play("-y90")
                align()
                qg.qubit_E.xy.play("y90")
                qg.qubit_pair_DE.macros['cz'].apply()
                qg.qubit_E.xy.play("-y90")
            
            align()
            
            # Readout all qubits
            for idx, q in enumerate(qg.qubits):
                readout_state(q, state_vars[i][idx])
                save(state_vars[i][idx], state_st_vars[i][idx])
            
            # Compute combined state based on number of qubits
            if num_qubits == 3:
                assign(state[i], state_vars[i][0]*4 + state_vars[i][1]*2 + state_vars[i][2])
            elif num_qubits == 4:
                assign(state[i], state_vars[i][0]*8 + state_vars[i][1]*4 + state_vars[i][2]*2 + state_vars[i][3])
            elif num_qubits == 5:
                assign(state[i], state_vars[i][0]*16 + state_vars[i][1]*8 + state_vars[i][2]*4 + state_vars[i][3]*2 + state_vars[i][4])
            
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
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
        job = qm.execute(CPhase_Oscillations)

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
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}
    
# %%
if not node.parameters.simulate:
    states = list(range(num_states))

    results = {}
    corrected_results = {}
    corrected_results_nq = {}
    fidelities = {}
    fidelities_nq = {}
    fidelity_differences = {}
    
    for i, qg in enumerate(qubit_groups_for_qua):
        results[qg.name] = []
        for state in states:
            results[qg.name].append((ds.sel(qubit = qg.name).state == state).sum().values)
        results[qg.name] = np.array(results[qg.name])/node.parameters.num_shots
        
        # Original correction using Kronecker product of single-qubit confusion matrices
        conf_mat = np.array([[1]])
        for q in qg.qubits:
            conf_mat = np.kron(conf_mat, q.resonator.confusion_matrix)
        
        # Least-squares mitigation instead of inverse
        corrected_results[qg.name] = least_squares_mitigation(conf_mat, results[qg.name])
        # Calculate fidelity as sum of all-0 and all-1 state probabilities
        fidelities[qg.name] = corrected_results[qg.name][0] + corrected_results[qg.name][num_states - 1]
        
        # New correction using NQ confusion matrix if available
        # Pass qubit names directly from Parameters and machine object
        conf_mat_nq = get_nq_confusion_matrix(node.parameters.qubit_groups[i], machine)
        if conf_mat_nq is not None:
            # Least-squares mitigation instead of inverse
            corrected_results_nq[qg.name] = least_squares_mitigation(conf_mat_nq, results[qg.name])
            # Calculate fidelity as sum of all-0 and all-1 state probabilities
            fidelities_nq[qg.name] = corrected_results_nq[qg.name][0] + corrected_results_nq[qg.name][num_states - 1]
            # Calculate fidelity difference (NQ - Kronecker)
            fidelity_differences[qg.name] = fidelities_nq[qg.name] - fidelities[qg.name]
            all_0_label = '0' * num_qubits
            all_1_label = '1' * num_qubits
            print(f"{qg.name} (Kron): Z-basis population fidelity ({all_0_label}+{all_1_label}): {fidelities[qg.name]:.4f}")
            print(f"{qg.name} ({num_qubits}Q): Z-basis population fidelity ({all_0_label}+{all_1_label}): {fidelities_nq[qg.name]:.4f}, Difference: {fidelity_differences[qg.name]:.4f}")
        else:
            all_0_label = '0' * num_qubits
            all_1_label = '1' * num_qubits
            print(f"{qg.name} (Kron): Z-basis population fidelity ({all_0_label}+{all_1_label}): {fidelities[qg.name]:.4f}")
            print(f"Warning: {num_qubits}Q confusion matrix not found for {qg.name}")



# %%
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
    
    # Plot with Kronecker product correction
    if num_qubit_groups == 1:
        f,axs = plt.subplots(1,figsize=(fig_width,3))
    else:
        f,axs = plt.subplots(num_qubit_groups,1,figsize=(fig_width,3*num_qubit_groups))
    
    for i, qg in enumerate(qubit_groups_for_qua):
        if num_qubit_groups == 1:
            ax = axs
        else:
            ax = axs[i]
        ax.bar(state_labels, corrected_results[qg.name], color='skyblue', edgecolor='navy')
        ax.set_ylim(0, 1)
        # Only show text labels for significant probabilities to avoid clutter
        for j, v in enumerate(corrected_results[qg.name]):
            if v > 0.01:
                rotation = 90 if num_qubits >= 4 else 0
                ax.text(j, v, f'{v:.2f}', ha='center', va='bottom', rotation=rotation, fontsize=8 if num_qubits >= 4 else 10)
        ax.set_ylabel('Probability')
        # Only add x-axis label to the last subplot
        if i == num_qubit_groups - 1:
            ax.set_xlabel('State')
        if num_qubits >= 4:
            ax.tick_params(axis='x', rotation=90)
        fidelity = fidelities[qg.name]
        ax.set_title(f"{qg.name} (Kronecker correction LS)")
        # Add fidelity as text annotation inside the plot
        ax.text(0.02, 0.98, f'Z-basis population fidelity ({all_0_label}+{all_1_label}): {fidelity:.4f}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Add more space between subplots
    f.tight_layout(pad=2.0)
    f.subplots_adjust(hspace=0.4)
    plt.show()
    node.results["figure"] = f
    
    # Plot with NQ confusion matrix correction if available
    if corrected_results_nq:
        groups_with_nq = [qg for qg in qubit_groups_for_qua if qg.name in corrected_results_nq]
        num_groups_nq = len(groups_with_nq)
        if num_groups_nq == 1:
            f_nq,axs_nq = plt.subplots(1,figsize=(fig_width,3))
        else:
            f_nq,axs_nq = plt.subplots(num_groups_nq,1,figsize=(fig_width,3*num_groups_nq))
        
        for i, qg in enumerate(groups_with_nq):
            if num_groups_nq == 1:
                ax = axs_nq
            else:
                ax = axs_nq[i]
            ax.bar(state_labels, corrected_results_nq[qg.name], color='moccasin', edgecolor='orange')
            ax.set_ylim(0, 1)
            # Only show text labels for significant probabilities to avoid clutter
            for j, v in enumerate(corrected_results_nq[qg.name]):
                if v > 0.01:
                    rotation = 90 if num_qubits >= 4 else 0
                    ax.text(j, v, f'{v:.2f}', ha='center', va='bottom', rotation=rotation, fontsize=8 if num_qubits >= 4 else 10)
            ax.set_ylabel('Probability')
            # Only add x-axis label to the last subplot
            if i == num_groups_nq - 1:
                ax.set_xlabel('State')
            if num_qubits >= 4:
                ax.tick_params(axis='x', rotation=90)
            fidelity_nq = fidelities_nq[qg.name]
            fidelity_diff = fidelity_differences[qg.name]
            ax.set_title(f"{qg.name} ({num_qubits}Q confusion matrix correction LS)")
            # Add fidelity as text annotation inside the plot
            ax.text(0.02, 0.98, f'Z-basis population fidelity ({all_0_label}+{all_1_label}): {fidelity_nq:.4f}', 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.5))
            # Add fidelity difference legend
            diff_color = 'orange' if fidelity_diff > 0 else 'red'
            diff_sign = '+' if fidelity_diff > 0 else ''
            ax.text(0.02, 0.88, f'Δ Z-basis population fidelity ({num_qubits}Q - Kron): {diff_sign}{fidelity_diff:.4f}', 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=diff_color, linewidth=2))
        # Add more space between subplots
        f_nq.tight_layout(pad=2.0)
        f_nq.subplots_adjust(hspace=0.4)
        plt.show()
        node.results["figure_nq"] = f_nq
    
    # Store results in node.results
    node.results["corrected_results"] = {k: v.tolist() for k, v in corrected_results.items()}
    node.results["fidelities"] = fidelities
    if corrected_results_nq:
        node.results["corrected_results_nq"] = {k: v.tolist() for k, v in corrected_results_nq.items()}
        node.results["fidelities_nq"] = fidelities_nq
        node.results["fidelity_differences"] = fidelity_differences
    
    # grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    # grid = QubitPairGrid(grid_names, qubit_pair_names)
    # for ax, qubit_pair in grid_iter(grid):
    #     print(qubit_pair['qubit'])
    #     corrected_res = corrected_results[qubit_pair['qubit']]
    #     ax.bar(['00', '01', '10', '11'], corrected_res, color='skyblue', edgecolor='navy')
    #     ax.set_ylim(0, 1)
    #     for i, v in enumerate(corrected_res):
    #         ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    #     ax.set_ylabel('Probability')
    #     ax.set_xlabel('State')
    #     ax.set_title(qubit_pair['qubit'])
    # plt.show()
    # node.results["figure"] = grid.fig
# %%

# %% {Update_state}

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
        
# %%
