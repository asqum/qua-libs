import itertools
import numpy as np


def generate_pauli_basis(n_qubits: int):
    """Generate the full Pauli index basis for N qubits."""
    return list(itertools.product(range(4), repeat=n_qubits))


def gen_inverse_hadamard(n_qubits: int):
    """Build the inverse N-qubit tomography transform matrix."""
    h_mat = np.array([[1, 1], [1, -1]]) / 2
    hn_mat = h_mat
    for _ in range(n_qubits - 1):
        hn_mat = np.kron(hn_mat, h_mat)
    return np.linalg.inv(hn_mat)


def get_kron_confusion_matrix(qg):
    """Construct a group confusion matrix via Kronecker product."""
    conf_mat = np.array([[1]])
    for q in qg.qubits:
        conf_mat = np.kron(conf_mat, q.resonator.confusion_matrix)
    return conf_mat


def ghz_density_matrix(num_qubits: int, sign: int = +1):
    """Build the ideal GHZ density matrix."""
    dim = 2**num_qubits
    rho = np.zeros((dim, dim), dtype=complex)

    rho[0, 0] = 0.5
    rho[-1, -1] = 0.5
    rho[0, -1] = 0.5 * sign
    rho[-1, 0] = 0.5 * sign

    return rho


def ghz_state_vector(num_qubits: int, sign: int = +1):
    """Build the ideal GHZ state vector."""
    psi = np.zeros(2**num_qubits, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[-1] = sign / np.sqrt(2)
    return psi


def fidelity_with_pure_target(rho: np.ndarray, psi_target: np.ndarray):
    """Compute fidelity against a pure target state."""
    fidelity = np.real(np.vdot(psi_target, rho @ psi_target))
    return float(np.clip(fidelity, 0.0, 1.0))
