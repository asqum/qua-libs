import numpy as np
from scipy.optimize import minimize
def least_squares_mitigation(confusion_matrix, measured_probs):
    """
    Constrained Least-Squares estimation for readout error mitigation.
    
    Minimizes: ||p_meas - Cρ||
    Subject to: ρ ≥ 0, ∑ρ = 1
    
    Args:
        confusion_matrix: confusion matrix C
        measured_probs: Array of measured probabilities p_meas for each state
        
    Returns:
        Estimated true population ρ
    
    Example:
        from iqcc_calibration_tools.analysis.readout_mitigation import get_nq_confusion_matrix, least_squares_mitigation
        from iqcc_calibration_tools.quam_config.components import Quam
        
        # Load machine and get confusion matrix
        machine = Quam.load()
        qubit_names = ["qD4", "qD3", "qC4", "qC2"]
        conf_mat_nq = get_nq_confusion_matrix(qubit_names, machine)
        
        # Apply least-squares mitigation to measured results
        if conf_mat_nq is not None:
            # results[qg.name] is array of measured probabilities for each state
            corrected_results = least_squares_mitigation(conf_mat_nq, results[qg.name])
    """
    n_states = len(measured_probs)
    
    # Objective function: L2 norm of residual
    def objective(rho):
        predicted = confusion_matrix @ rho
        residual = measured_probs - predicted
        return np.sum(residual ** 2)
    
    # Constraint: sum of probabilities equals 1
    def constraint_sum(rho):
        return np.sum(rho) - 1.0
    
    # Initial guess: uniform distribution
    rho_init = np.ones(n_states) / n_states
    
    # Bounds: all probabilities must be non-negative
    bounds = [(0.0, 1.0) for _ in range(n_states)]
    
    # Constraints
    constraints = {'type': 'eq', 'fun': constraint_sum}
    
    # Minimize
    result = minimize(
        objective,
        rho_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        print(f"Warning: Least-squares optimization did not converge: {result.message}")
    
    return result.x
def reorder_confusion_matrix(confusion_matrix, source_order, target_order):
    """
    Reorder a confusion matrix to match a different qubit order.
    
    Args:
        confusion_matrix: NxN confusion matrix (N = 2^num_qubits)
        source_order: List of qubit names in the order used when measuring the confusion matrix
        target_order: List of qubit names in the order used in the experiment(eg: GHZ state)
        
    Returns:
        Reordered confusion matrix
    """
    confusion_matrix = np.asarray(confusion_matrix)
    
    # Validate inputs
    if len(source_order) != len(target_order):
        raise ValueError(f"source_order and target_order must have the same length: {len(source_order)} != {len(target_order)}")
    
    if set(source_order) != set(target_order):
        raise ValueError(f"source_order and target_order must contain the same qubits: {set(source_order)} != {set(target_order)}")
    
    # Infer number of qubits from matrix size (more reliable than trusting the order length)
    matrix_size = confusion_matrix.shape[0]
    if confusion_matrix.shape[1] != matrix_size:
        raise ValueError(f"Confusion matrix must be square, got shape {confusion_matrix.shape}")
    
    # Calculate expected number of qubits from matrix size
    num_qubits_from_matrix = int(np.log2(matrix_size))
    if 2 ** num_qubits_from_matrix != matrix_size:
        raise ValueError(f"Confusion matrix size {matrix_size} is not a power of 2 (expected 2^n for n qubits)")
    
    # Use the inferred number of qubits, but validate against order length
    num_qubits = num_qubits_from_matrix
    if len(source_order) != num_qubits:
        raise ValueError(
            f"Mismatch between confusion matrix size and qubit order length: "
            f"matrix size {matrix_size} implies {num_qubits} qubits, "
            f"but source_order has {len(source_order)} qubits"
        )
    
    num_states = matrix_size  # Use actual matrix size instead of 2^num_qubits
    
    # Create mapping from source state index to target state index
    # For each state in source order, find the corresponding state in target order
    state_mapping = {}
    for source_state_idx in range(num_states):
        # Convert state index to binary representation
        source_binary = format(source_state_idx, f'0{num_qubits}b')
        # Map each bit to the corresponding qubit value
        source_state_dict = {source_order[i]: int(source_binary[i]) for i in range(num_qubits)}
        # Reconstruct state index in target order
        target_binary = ''.join(str(source_state_dict[q]) for q in target_order)
        target_state_idx = int(target_binary, 2)
        state_mapping[source_state_idx] = target_state_idx
    # Reorder both rows and columns of the confusion matrix
    reordered = np.zeros_like(confusion_matrix)
    for source_row in range(num_states):
        target_row = state_mapping[source_row]
        for source_col in range(num_states):
            target_col = state_mapping[source_col]
            reordered[target_row, target_col] = confusion_matrix[source_row, source_col]
    
    return reordered

def get_nq_confusion_matrix(qubit_names, machine_obj=None):
    """
    Search for N-Qubits confusion matrix in all possiblequbit pairs.
    Checks in qubit pairs.extras to find a match based on qubit set (order-independent).
    Only matches if the qubit sets are having the same qubits, any order.
    Automatically reorders the confusion matrix if qubit order differs from the target qubit order.
    
    Args:
        qubit_names: List of qubit name strings (e.g., ["qD4","qD3","qC4","qC2","qC1"])
        machine_obj: Machine object with qubits and qubit_pairs. If None, uses global 'machine'.
    
    Returns:
        Confusion matrix as numpy array, or None if not found.
    
    Example:
        from iqcc_calibration_tools.analysis.readout_mitigation import get_nq_confusion_matrix, least_squares_mitigation
        from iqcc_calibration_tools.quam_config.components import Quam
        
        # Load machine from state.json
        machine = Quam.load()
        
        # Get confusion matrix for a qubit group (from Parameters.qubit_groups)
        qubit_names = ["qD4", "qD3", "qC4", "qC2"]  # 4-qubit group
        conf_mat_nq = get_nq_confusion_matrix(qubit_names, machine)
        
        # Use with least-squares mitigation (as in 41a_GHZ_Zbasi_least squares.py)
        if conf_mat_nq is not None:
            # results[qg.name] contains measured probabilities for each state
            corrected_results = least_squares_mitigation(conf_mat_nq, results[qg.name])
            # Calculate fidelity as sum of all-0 and all-1 state probabilities
            num_states = len(corrected_results)
            fidelity = corrected_results[0] + corrected_results[num_states - 1]
        else:
            print("Warning: NQ confusion matrix not found, using Kronecker product instead")
    """
    # Get machine object (use provided one or global)
    if machine_obj is None:
        try:
            machine_obj = globals().get('machine')
            if machine_obj is None:
                raise AttributeError("machine not available")
        except (AttributeError, KeyError):
            print("Error: machine object not available. Please provide machine_obj or inject 'machine' into module globals.")
            return None
    
    target_qubit_order = qubit_names
    qubit_group_name = "-".join(target_qubit_order)
    
    # Get the set of qubit names we're looking for (order-independent)
    target_qubit_set = set(target_qubit_order)
    num_qubits = len(target_qubit_set)
    confusion_key = f"confusion_{num_qubits}q"

    def check_in_pair(qp, pair_name=""):
        """Check if confusion matrix exists in a qubit pair, matching by qubit set (order-independent)."""
        if qp is None:
            return None, None
        
        # Special case for 2-qubit confusion matrices: stored directly in qp.confusion
        if num_qubits == 2:
            if hasattr(qp, 'confusion') and qp.confusion is not None:
                # Get the qubit names from the pair
                pair_qubit_names = [qp.qubit_control.name, qp.qubit_target.name]
                pair_qubit_set = set(pair_qubit_names)
                
                # Check if the pair qubits match the target qubits
                if pair_qubit_set == target_qubit_set:
                    confusion_matrix = np.array(qp.confusion)
                    
                    # Reorder the confusion matrix if qubit order is different
                    if pair_qubit_names != target_qubit_order:
                        confusion_matrix = reorder_confusion_matrix(
                            confusion_matrix, pair_qubit_names, target_qubit_order
                        )
                    
                    return confusion_matrix, f"pair {pair_name} (qp.confusion)"
        
        # For 3+ qubit confusion matrices, check in extras
        if not hasattr(qp, 'extras') or qp.extras is None:
            return None, None

        # Iterate through all entries in extras, entry name format: "q1-q2-q3-..." (any order)
        for entry_name, entry_data in qp.extras.items():
            # Check if it's dict-like by checking for keys() method
            if not hasattr(entry_data, 'keys') or not hasattr(entry_data, '__getitem__'):
                continue
            # Check if the entry match the target qubits number (confusion_3q, confusion_4q, etc.)
            if confusion_key in entry_data:
                # Check if the entry qubits set match the target qubits set
                try:
                    # Parse the entry name to get the qubit set and order
                    # Entry name format: "q1-q2-q3-..." (any order)
                    entry_qubit_list = entry_name.split("-")
                    entry_qubit_set = set(entry_qubit_list)
                    # Check if the qubit sets match exactly (same qubits, order-independent)
                    if entry_qubit_set == target_qubit_set:
                        confusion_matrix = np.array(entry_data[confusion_key])
                        
                        # Reorder the confusion matrix if qubit order is different
                        if entry_qubit_list != target_qubit_order:
                            confusion_matrix = reorder_confusion_matrix(
                                confusion_matrix, entry_qubit_list, target_qubit_order
                            )
                        
                        return confusion_matrix, f"pair {pair_name}, entry '{entry_name}'"
                except Exception as e:
                    # If parsing fails, skip this entry
                    continue
        
        return None, None
    
    # Search all possible pair combinations from the qubit group (order matters)
    # Generate all combinations: (i, j) where i != j, checking both orderings explicitly
    pairs_checked = 0
    checked_pairs_set = set()  # Track which physical pairs we've already checked to avoid duplicates
    found_location = None
    
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i == j:
                continue  # Skip same qubit pairs
            
            q1_name = target_qubit_order[i]
            q2_name = target_qubit_order[j]
            
            # Create a canonical pair key (sorted) to track if we've checked this physical pair
            canonical_pair = tuple(sorted([q1_name, q2_name]))
            
            # Try to find the pair in machine using the specific ordering
            pair_name = f"{q1_name}-{q2_name}"
            qp = None
            
            # Check if this exact ordering exists
            if pair_name in machine_obj.qubit_pairs:
                qp = machine_obj.qubit_pairs[pair_name]
            else:
                # Try reverse ordering
                reverse_pair_name = f"{q2_name}-{q1_name}"
                if reverse_pair_name in machine_obj.qubit_pairs:
                    qp = machine_obj.qubit_pairs[reverse_pair_name]
                else:
                    # Fallback: search by qubit objects
                    q1_obj = machine_obj.qubits[q1_name]
                    q2_obj = machine_obj.qubits[q2_name]
                    for qp_candidate in machine_obj.qubit_pairs.values():
                        if (qp_candidate.qubit_control in [q1_obj, q2_obj] and 
                            qp_candidate.qubit_target in [q1_obj, q2_obj]):
                            qp = qp_candidate
                            break
            
            if qp is None:
                # Pair doesn't exist in machine, skip gracefully
                continue
            
            # Only check each physical pair once (even if we check both orderings)
            if canonical_pair not in checked_pairs_set:
                checked_pairs_set.add(canonical_pair)
            
            pairs_checked += 1
            result, location = check_in_pair(qp, pair_name)
            if result is not None:
                found_location = location
                print(f"Found {num_qubits}Q confusion matrix for {qubit_group_name} in {found_location}")
                return result
    
    # Print final summary
    print(f"{num_qubits}Q confusion matrix not found for {qubit_group_name} (qubits: {sorted(target_qubit_set)})")
    return None