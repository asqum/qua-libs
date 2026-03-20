import itertools
import numpy as np


def compute_confusion_matrix(ds, qg_name, n_qubits, n_states, n_shots):
    """Compute direct confusion matrix from measured integer states."""
    conf_rows = []
    for init_values in itertools.product([0, 1], repeat=n_qubits):
        sel_dict = {f"init_{idx}": val for idx, val in enumerate(init_values)}
        measured = ds.sel(qubit=qg_name).state.sel(**sel_dict).values
        row = np.bincount(measured.astype(int), minlength=n_states)
        conf_rows.append(row)
    return np.array(conf_rows) / n_shots


def compute_kron_confusion_matrix(qubits):
    """Compute Kronecker-product reference confusion matrix from per-qubit readout matrices."""
    conf_mat = np.array([[1.0]])
    for q in qubits:
        conf_mat = np.kron(conf_mat, q.resonator.confusion_matrix)
    return conf_mat
