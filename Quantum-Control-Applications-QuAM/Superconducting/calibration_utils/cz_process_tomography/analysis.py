"""
Two-qubit quantum process tomography (QPT) analysis for 90a_cz_process_tomography.

Every function in this module uses the conventions below. They are also exported as
``CONVENTIONS`` and saved with the node results so that data can be re-analysed offline.

Qubit and tensor-product ordering
    Qubit 0 = ``qubit_pair.qubit_control``, qubit 1 = ``qubit_pair.qubit_target``.
    |c t> has computational index 2*c + t, i.e. operators are kron(control, target).
    This is the ordering of ``state_control * 2 + state_target`` in 34_2Q_confusion_matrix,
    40b_Bell_state_tomography and 100b_two_qubit_gate_set_tomography_ais.

Readout assignment
    ``readout_state``: 0 -> |0> (Z = +1), 1 -> |1> (Z = -1). With GEF readout, 2 -> |2> (leaked).

Single-qubit gate convention (same mapping as the pyGSTi GST node: Gxpi2 -> x90, Gypi2 -> y90)
    x90 = exp(-i pi/4 X), y90 = exp(-i pi/4 Y), -x90 / -y90 are their inverses, x180 = exp(-i pi/2 X).

Input states, per qubit (index 0..3); two-qubit index k = 4*prep_control + prep_target
    0: |0>   no pulse
    1: |1>   x180
    2: |+>   y90    -> (|0> + |1>)/sqrt(2)
    3: |+i>  -x90   -> (|0> + i|1>)/sqrt(2)

Tomography pre-rotations, per qubit (index 0..2), followed by Z readout; setting m = 3*tomo_control + tomo_target
    0: X  -y90  (U^dag Z U = +X, so P(0) - P(1) = <X>)
    1: Y  +x90  (U^dag Z U = +Y)
    2: Z  none
    Note: 40b/41b use +y90 for X, i.e. they measure -X. That sign cancels in even-weight
    correlators (XX) but not in single-qubit terms, so it is not reused here.

Pauli operator basis
    sigma = [I, X, Y, Z];  P_{4a+b} = sigma_a (control) (x) sigma_b (target)
    Labels: II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ (first letter = control).

Choi matrix
    J = sum_ij |i><j| (x) Lambda(|i><j|)   (input (x) output, unnormalised, Tr J = d for TP maps)
    Lambda(rho) = Tr_in[(rho^T (x) 1) J]

Chi (process) matrix
    Lambda(rho) = sum_mn chi_mn P_m rho P_n^dag, unnormalised Paulis, Tr chi = Tr J / d (= 1 for TP maps)
    chi = B^dag J B / d^2 with B[:, m] = vec(P_m) (column stacking).
    For a unitary U = sum_m u_m P_m:  chi_mn = u_m conj(u_n), u_m = Tr(P_m U) / d.

Pauli transfer matrix (PTM)
    R_ij = Tr[P_i Lambda(P_j)] / d   (real for Hermiticity-preserving maps; R_00 = 1 and R_0j = 0 for TP maps)

Fidelity
    Process fidelity = entanglement fidelity with a unitary target U:
        F_pro = <<U| J |U>> / d^2 = Tr[chi_U chi] = Tr[R_U^T R] / d^2
    Average gate fidelity:
        F_avg = (d F_pro + Tr[J]/d) / (d + 1)
    which equals (4 F_pro + 1) / 5 for trace-preserving maps (d = 4).
"""

import itertools
import warnings
from typing import Dict, Optional

import numpy as np
import xarray as xr
from scipy.optimize import minimize

from calibration_utils.ghz_tomography import get_density_matrix, get_pauli_data_nq

N_QUBITS = 2
D = 2**N_QUBITS

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_1Q = (I2, X2, Y2, Z2)
PAULI_1Q_LABELS = ("I", "X", "Y", "Z")
PAULI_2Q_LABELS = tuple(a + b for a in PAULI_1Q_LABELS for b in PAULI_1Q_LABELS)
PAULI_2Q = np.array([np.kron(a, b) for a in PAULI_1Q for b in PAULI_1Q])

PREP_LABELS = ("0", "1", "+", "+i")
PREP_PULSES = (None, "x180", "y90", "-x90")
TOMO_LABELS = ("X", "Y", "Z")
TOMO_PULSES = ("-y90", "x90", None)
OUTCOME_LABELS = ("00", "01", "10", "11")
INPUT_LABELS = tuple(f"{c},{t}" for c in PREP_LABELS for t in PREP_LABELS)

CONVENTIONS = {
    "qubit_order": "qubit 0 = qubit_control, qubit 1 = qubit_target; |c t> -> index 2*c + t (kron(control, target))",
    "readout_assignment": "state 0 -> |0> (Z=+1), 1 -> |1> (Z=-1), 2 -> |2> (GEF readout only)",
    "gate_convention": "x90 = exp(-i pi/4 X), y90 = exp(-i pi/4 Y); -x90/-y90 inverses; x180 = exp(-i pi/2 X)",
    "input_states": "per qubit: 0:|0> (none), 1:|1> (x180), 2:|+> (y90), 3:|+i> (-x90); k = 4*prep_control + prep_target",
    "tomography_rotations": "per qubit: X: -y90, Y: +x90, Z: none, then Z readout; m = 3*tomo_control + tomo_target",
    "pauli_basis": "P_(4a+b) = sigma_a(control) (x) sigma_b(target), sigma = [I, X, Y, Z]; labels " + ",".join(PAULI_2Q_LABELS),
    "choi": "J = sum_ij |i><j| (x) Lambda(|i><j|) (input (x) output), Tr J = d for TP maps",
    "chi": "Lambda(rho) = sum_mn chi_mn P_m rho P_n^dag, unnormalised Paulis, Tr chi = 1 for TP maps",
    "ptm": "R_ij = Tr[P_i Lambda(P_j)] / d",
    "process_fidelity": "entanglement fidelity F_pro = <<U|J|U>>/d^2 = Tr[chi_ideal chi]",
    "average_gate_fidelity": "F_avg = (d F_pro + Tr[J]/d)/(d+1) = (4 F_pro + 1)/5 for TP maps",
    "cz_model": "U_eff = [Z(phi1) (x) Z(phi2)] fSim(theta, phi_CZ), Z(phi) = diag(1, exp(-i phi)), "
    "fSim = [[1,0,0,0],[0,cos,-i sin,0],[0,-i sin,cos,0],[0,0,0,exp(-i phi_CZ)]]",
}


# ---------------------------------------------------------------------------
# Single-qubit rotations, input states and measurement operators
# ---------------------------------------------------------------------------
def rx(theta: float) -> np.ndarray:
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * X2


def ry(theta: float) -> np.ndarray:
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * Y2


PREP_UNITARIES_1Q = (I2, rx(np.pi), ry(np.pi / 2), rx(-np.pi / 2))
TOMO_UNITARIES_1Q = (ry(-np.pi / 2), rx(np.pi / 2), I2)


def input_states() -> np.ndarray:
    """Ideal input density matrices, shape (16, 4, 4), index k = 4*prep_control + prep_target."""
    ket0 = np.array([1, 0], dtype=complex)
    rhos_1q = []
    for u in PREP_UNITARIES_1Q:
        psi = u @ ket0
        rhos_1q.append(np.outer(psi, psi.conj()))
    return np.array([np.kron(rc, rt) for rc in rhos_1q for rt in rhos_1q])


def measurement_projectors() -> np.ndarray:
    """POVM elements Pi[m, o] = U_m^dag |o><o| U_m, shape (9, 4, 4, 4); m = 3*tomo_control + tomo_target, o = 2*c + t."""
    projectors = np.zeros((len(TOMO_LABELS) ** 2, D, D, D), dtype=complex)
    for m, (uc, ut) in enumerate(itertools.product(TOMO_UNITARIES_1Q, repeat=2)):
        u = np.kron(uc, ut)
        for o in range(D):
            ket = np.zeros(D)
            ket[o] = 1
            projectors[m, o] = u.conj().T @ np.outer(ket, ket) @ u
    return projectors


def pauli_expectations_of_states(rhos: np.ndarray) -> np.ndarray:
    """E[k, i] = Tr(P_i rho_k) for a stack of density matrices, shape (n, 16)."""
    return np.real(np.einsum("iab,kba->ki", PAULI_2Q, rhos))


# ---------------------------------------------------------------------------
# Raw data -> populations -> readout mitigation
# ---------------------------------------------------------------------------
def counts_from_shots(shots: np.ndarray, readout_base: int):
    """Histogram single shots into computational-subspace counts.

    ``shots`` has shape (N, 4, 4, 3, 3) = (shot, prep_control, prep_target, tomo_control, tomo_target)
    and holds ``readout_base * state_control + state_target`` (readout_base = 2 for g/e readout,
    3 for g/e/f readout).

    Returns (counts, leak_counts, n_total):
        counts      (16, 9, 4) counts of outcomes 00, 01, 10, 11
        leak_counts (16, 9)    shots where at least one qubit was assigned to |2>
        n_total     number of shots per setting (including leaked shots)
    """
    shots = np.asarray(shots, dtype=int)
    n_total = shots.shape[0]
    state_control, state_target = shots // readout_base, shots % readout_base
    counts = np.zeros((16, 9, D))
    for c in range(2):
        for t in range(2):
            counts[:, :, 2 * c + t] = ((state_control == c) & (state_target == t)).sum(axis=0).reshape(16, 9)
    leak_counts = ((state_control == 2) | (state_target == 2)).sum(axis=0).reshape(16, 9).astype(float)
    return counts, leak_counts, n_total


def mitigate_populations(populations: np.ndarray, confusion: np.ndarray) -> np.ndarray:
    """Unconstrained linear-inversion readout mitigation, p_true = M^-1 p_meas.

    ``confusion`` is M[measured, prepared] (columns sum to 1). No clipping or renormalisation is
    applied: small negative quasi-probabilities are kept so that the linear-inversion estimate stays
    unbiased (physicality is imposed separately and explicitly by ``fit_physical_process``).
    Because the columns of M sum to 1, the total population of each setting is preserved.
    """
    return populations @ np.linalg.inv(confusion).T


# ---------------------------------------------------------------------------
# Density-matrix reconstruction (reuses the GHZ state-tomography helpers)
# ---------------------------------------------------------------------------
def reconstruct_output_states(populations: np.ndarray):
    """Linear-inversion state tomography of the 16 output states.

    ``populations`` has shape (16, 9, 4). For every input state the 9 tomography settings are passed
    to ``get_pauli_data_nq`` (same estimator as 41b_GHZ_tomography: every Pauli expectation value is
    averaged over all settings that measure it) and ``get_density_matrix`` builds
    rho = sum_i <P_i> P_i / d.

    Returns (expectations (16, 16), rhos (16, 4, 4)). Density matrices are not forced to be
    positive: this is the raw linear-inversion estimate.
    """
    pauli_keys = [f"{a},{b}" for a in range(4) for b in range(4)]
    expectations = np.zeros((16, 16))
    rhos = np.zeros((16, D, D), dtype=complex)
    for k in range(16):
        da = xr.DataArray(
            populations[k].reshape(3, 3, D),
            dims=["tomo_axis_0", "tomo_axis_1", "state"],
            coords={"tomo_axis_0": [0, 1, 2], "tomo_axis_1": [0, 1, 2], "state": list(OUTCOME_LABELS)},
        ).stack(tomo_axis=["tomo_axis_0", "tomo_axis_1"])
        paulis = get_pauli_data_nq(da, N_QUBITS)
        expectations[k] = paulis.sel(pauli_op=pauli_keys).values
        rhos[k] = get_density_matrix(paulis, N_QUBITS)
    return expectations, rhos


def pauli_sign_check(expectations: np.ndarray, expectations_ideal: np.ndarray) -> Dict[str, float]:
    """Least-squares slope of measured vs ideal expectation values for the single-qubit Paulis.

    A negative slope for an X or Y term means the tomography/preparation pulse signs do not match
    the convention above. A global mirror of the Y axis (y90 <-> -y90 on hardware) is not detected
    by this check: it complex-conjugates the reconstructed process (see ``fit_cz_model``).
    """
    slopes = {}
    for label in ("XI", "YI", "ZI", "IX", "IY", "IZ"):
        i = PAULI_2Q_LABELS.index(label)
        ideal, meas = expectations_ideal[:, i], expectations[:, i]
        norm = float(ideal @ ideal)
        slopes[label] = float(ideal @ meas / norm) if norm > 0 else float("nan")
    return slopes


# ---------------------------------------------------------------------------
# Process representations and conversions
# ---------------------------------------------------------------------------
def vec(a: np.ndarray) -> np.ndarray:
    """Column-stacking vectorisation, |A>> = sum_i |i> (x) A|i>."""
    return a.reshape(-1, order="F")


PAULI_VEC_BASIS = np.array([vec(p) for p in PAULI_2Q]).T  # B[:, m] = vec(P_m)


def apply_ptm(ptm: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Lambda(rho) = sum_ij R_ij Tr(P_j rho)/d P_i (valid for any operator rho)."""
    coefficients = np.einsum("jab,ba->j", PAULI_2Q, rho) / D
    return np.einsum("i,iab->ab", ptm @ coefficients, PAULI_2Q)


def apply_choi(choi: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Lambda(rho) = Tr_in[(rho^T (x) 1) J]."""
    return np.einsum("ki,kaib->ab", rho, choi.reshape(D, D, D, D))


def ptm_to_choi(ptm: np.ndarray) -> np.ndarray:
    choi = np.zeros((D * D, D * D), dtype=complex)
    for i in range(D):
        for j in range(D):
            e_ij = np.zeros((D, D), dtype=complex)
            e_ij[i, j] = 1
            choi += np.kron(e_ij, apply_ptm(ptm, e_ij))
    return choi


def choi_to_ptm(choi: np.ndarray) -> np.ndarray:
    return np.array(
        [[np.real(np.trace(p_i @ apply_choi(choi, p_j))) / D for p_j in PAULI_2Q] for p_i in PAULI_2Q]
    )


def choi_to_chi(choi: np.ndarray) -> np.ndarray:
    return PAULI_VEC_BASIS.conj().T @ choi @ PAULI_VEC_BASIS / D**2


def chi_to_choi(chi: np.ndarray) -> np.ndarray:
    return PAULI_VEC_BASIS @ chi @ PAULI_VEC_BASIS.conj().T


def unitary_to_choi(u: np.ndarray) -> np.ndarray:
    v = vec(u)
    return np.outer(v, v.conj())


def partial_trace_output(choi: np.ndarray) -> np.ndarray:
    """Tr_out J (an operator on the input space). Equals the identity for trace-preserving maps."""
    return np.trace(choi.reshape(D, D, D, D), axis1=1, axis2=3)


def ptm_from_pauli_expectations(expectations_out: np.ndarray, rhos_in: np.ndarray) -> np.ndarray:
    """Linear-inversion PTM from the output Pauli expectation values.

    With e_in[k] = Tr(P rho_in^k) and e_out[k] = Tr(P Lambda(rho_in^k)) we have e_out[k] = R e_in[k],
    hence R = E_out^T (E_in^T)^-1. The 16 product inputs {0, 1, +, +i}^(x)2 are tomographically
    complete, so E_in is invertible. The inputs are assumed to be ideal (SPAM enters the estimate).
    """
    expectations_in = pauli_expectations_of_states(rhos_in)
    return expectations_out.T @ np.linalg.inv(expectations_in.T)


# ---------------------------------------------------------------------------
# Ideal CZ and figures of merit
# ---------------------------------------------------------------------------
def ideal_cz_unitary() -> np.ndarray:
    return np.diag([1, 1, 1, -1]).astype(complex)


def process_fidelity(choi: np.ndarray, u_target: np.ndarray) -> float:
    v = vec(u_target)
    return float(np.real(v.conj() @ choi @ v)) / D**2


def average_gate_fidelity(choi: np.ndarray, u_target: np.ndarray) -> float:
    return (D * process_fidelity(choi, u_target) + float(np.real(np.trace(choi))) / D) / (D + 1)


def process_metrics(choi: np.ndarray, u_target: np.ndarray) -> Dict[str, float]:
    """Figures of merit of a reconstructed Choi matrix against a unitary target."""
    chi = choi_to_chi(choi)
    chi_target = choi_to_chi(unitary_to_choi(u_target))
    choi_eigenvalues = np.linalg.eigvalsh((choi + choi.conj().T) / 2) / D
    return {
        "process_fidelity": process_fidelity(choi, u_target),
        "average_gate_fidelity": average_gate_fidelity(choi, u_target),
        "chi_frobenius_distance": float(np.linalg.norm(chi - chi_target)),
        "trace_preservation_error": float(np.linalg.norm(partial_trace_output(choi) - np.eye(D))),
        "mean_output_trace": float(np.real(np.trace(choi))) / D,
        "choi_min_eigenvalue": float(choi_eigenvalues.min()),
        "choi_negativity": float(-choi_eigenvalues[choi_eigenvalues < 0].sum()),
    }


def error_channel_pauli_probabilities(ptm: np.ndarray, ptm_target: np.ndarray) -> np.ndarray:
    """Diagonal of the chi matrix of the error channel Lambda_err = Lambda o U_target^-1.

    Entry 0 (II) equals the process fidelity; the others are the Pauli-error probabilities of the
    Pauli-twirled error channel.
    """
    ptm_error = ptm @ ptm_target.T
    return np.real(np.diag(choi_to_chi(ptm_to_choi(ptm_error))))


# ---------------------------------------------------------------------------
# Physical (CP + TP / trace non-increasing) least-squares fit
# ---------------------------------------------------------------------------
def project_cp(choi: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((choi + choi.conj().T) / 2)
    return (v * np.clip(w, 0, None)) @ v.conj().T


def project_tp(choi: np.ndarray) -> np.ndarray:
    """Orthogonal projection onto {J : Tr_out J = 1}."""
    t = partial_trace_output(choi)
    return choi - np.kron((t - np.eye(D)) / D, np.eye(D))


def project_tni(choi: np.ndarray) -> np.ndarray:
    """Orthogonal projection onto {J : Tr_out J <= 1} (trace non-increasing)."""
    t = partial_trace_output(choi)
    w, v = np.linalg.eigh((t + t.conj().T) / 2)
    t_clipped = (v * np.minimum(w, 1)) @ v.conj().T
    return choi + np.kron((t_clipped - t) / D, np.eye(D))


def project_physical(choi: np.ndarray, constraint: str = "TP", max_iter: int = 1000, tol: float = 1e-12) -> np.ndarray:
    """Euclidean projection onto CP ∩ TP (or CP ∩ TNI) with Dykstra's alternating projections."""
    project_trace = project_tp if constraint == "TP" else project_tni
    x = choi
    p = np.zeros_like(choi)
    q = np.zeros_like(choi)
    for _ in range(max_iter):
        y = project_trace(x + p)
        p = x + p - y
        x_new = project_cp(y + q)
        q = y + q - x_new
        if np.linalg.norm(x_new - x) < tol and np.linalg.norm(x_new - y) < np.sqrt(tol):
            return x_new
        x = x_new
    return x


def build_forward_model(rhos_in: np.ndarray, projectors: np.ndarray, confusion: np.ndarray) -> np.ndarray:
    """Linear map A with p_measured = A @ vec_row(J), shape (16*9*4, 256).

    p_ideal[k, m, o] = Tr[(rho_k^T (x) Pi_mo) J]; readout errors are included as
    p_measured[k, m, :] = M @ p_ideal[k, m, :] with M[measured, prepared].
    """
    ops = np.einsum("kab,mocd->kmoacbd", np.transpose(rhos_in, (0, 2, 1)), projectors)
    ops = ops.reshape(rhos_in.shape[0], projectors.shape[0], D, D * D, D * D)
    # Tr(M J) = sum(M^T * J) -> rows are M^T flattened (row-major, matching J.reshape(-1))
    rows = np.transpose(ops, (0, 1, 2, 4, 3)).reshape(rhos_in.shape[0], projectors.shape[0], D, -1)
    rows = np.einsum("ab,kmbj->kmaj", confusion, rows)
    return rows.reshape(-1, D**4)


def fit_physical_process(
    counts: np.ndarray,
    n_total: int,
    rhos_in: np.ndarray,
    projectors: np.ndarray,
    confusion: np.ndarray,
    constraint: str = "TP",
    choi_init: Optional[np.ndarray] = None,
    max_iter: int = 3000,
    tol: float = 1e-10,
) -> Dict:
    """Constrained least-squares process fit directly on the measured frequencies.

    Minimises sum_(k,m,o) (p_model - f_meas)^2 with p_model = M Tr[(rho_k^T (x) Pi_mo) J] over Choi
    matrices J that are completely positive and trace preserving (constraint="TP") or trace
    non-increasing (constraint="TNI", used when leakage is measured so that leaked population is not
    renormalised away). The readout confusion matrix M is part of the forward model, so no clipping
    of mitigated probabilities is needed.

    Solver: accelerated projected gradient (FISTA with restart) with step 1/L, L = 2 ||A||_2^2,
    where each projection onto CP ∩ TP/TNI is computed with Dykstra's algorithm (cf. Knee et al.,
    PRA 98, 062336 (2018) for projected-gradient CPTP tomography).
    """
    a = build_forward_model(rhos_in, projectors, confusion)
    y = (counts / n_total).reshape(-1)
    lipschitz = 2 * np.linalg.norm(a, 2) ** 2

    def residual(choi):
        return np.real(a @ choi.reshape(-1)) - y

    def objective(choi):
        r = residual(choi)
        return float(r @ r)

    def gradient(choi):
        g = 2 * (a.conj().T @ residual(choi)).reshape(D * D, D * D)
        return (g + g.conj().T) / 2

    if choi_init is None:
        choi_init = np.eye(D * D, dtype=complex) / D
    choi = project_physical(choi_init, constraint)
    momentum_point = choi.copy()
    t = 1.0
    f_old = objective(choi)
    converged = False
    restarted = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        choi_new = project_physical(momentum_point - gradient(momentum_point) / lipschitz, constraint)
        f_new = objective(choi_new)
        if f_new > f_old:
            # A plain projected-gradient step from the current iterate cannot decrease the objective any
            # further (up to the accuracy of the projection): the fit has stagnated at the optimum.
            if restarted:
                converged = True
                break
            momentum_point, t, restarted = choi.copy(), 1.0, True
            continue
        restarted = False
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        momentum_point = choi_new + ((t - 1) / t_new) * (choi_new - choi)
        step = np.linalg.norm(choi_new - choi)
        choi, t, f_old = choi_new, t_new, f_new
        if step < tol * max(1.0, np.linalg.norm(choi)):
            converged = True
            break
    if not converged:
        warnings.warn(f"Physical process fit did not reach tol={tol} within {max_iter} iterations.")
    return {
        "choi": choi,
        "objective": f_old,
        "residual_rms": float(np.sqrt(f_old / y.size)),
        "iterations": iteration,
        "converged": converged,
        "constraint": constraint,
    }


def model_residual_rms(choi: np.ndarray, counts: np.ndarray, n_total: int, rhos_in, projectors, confusion) -> float:
    """RMS difference between the frequencies predicted by ``choi`` and the measured ones."""
    a = build_forward_model(rhos_in, projectors, confusion)
    r = np.real(a @ choi.reshape(-1)) - (counts / n_total).reshape(-1)
    return float(np.sqrt(np.mean(r**2)))


# ---------------------------------------------------------------------------
# CZ-specific coherent error model
# ---------------------------------------------------------------------------
def fsim(theta: float, phi: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, np.exp(-1j * phi)]], dtype=complex
    )


def local_z(phi: float) -> np.ndarray:
    return np.diag([1, np.exp(-1j * phi)])


def cz_model_unitary(theta: float, phi_cz: float, phi_control: float, phi_target: float) -> np.ndarray:
    """U_eff = [Z(phi_control) (x) Z(phi_target)] fSim(theta, phi_cz); ideal CZ: (0, pi, 0, 0)."""
    return np.kron(local_z(phi_control), local_z(phi_target)) @ fsim(theta, phi_cz)


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def fit_cz_model(choi: np.ndarray, n_grid: int = 8) -> Dict[str, float]:
    """Fit U_eff = [Z(phi1) (x) Z(phi2)] fSim(theta, phi_CZ) by maximising F_pro(Lambda, U_eff).

    Assumptions / limitations:
    - The dominant coherent error is captured by the 4-parameter model. Other coherent errors
      (single-qubit X/Y rotations, phases inside the |01>,|10> subspace) are not modelled and end up
      in 1 - F_model together with incoherent errors.
    - The reconstructed process includes SPAM. Z errors of the preparation/pre-rotation pulses are
      indistinguishable from gate Z errors (gauge freedom), which biases phi1, phi2.
    - A global mirror of the Y axis on hardware complex-conjugates the reconstructed process, which
      flips the signs of theta, phi1, phi2 and phi_CZ - pi but leaves all fidelities unchanged.
    - phi1, phi2 are the model's Z(phi) = diag(1, exp(-i phi)) angles of the full gate (including the
      virtual-Z phase_shift_* already applied by the gate); they are not written back to the state.
    Returned angles are in radians; theta is folded into [-pi/2, pi/2) and phi_CZ into [0, 2pi).
    """

    def infidelity(x):
        return 1.0 - process_fidelity(choi, cz_model_unitary(*x))

    grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    seeds = sorted((infidelity([0.0, np.pi, a, b]), a, b) for a in grid for b in grid)[:4]
    options = {"xatol": 1e-9, "fatol": 1e-13, "maxiter": 20000, "maxfev": 20000}

    best = None
    for _, a, b in seeds:
        for theta0 in (0.0, 0.1, -0.1):
            res = minimize(infidelity, [theta0, np.pi, a, b], method="Nelder-Mead", options=options)
            if best is None or res.fun < best.fun:
                best = res
    theta, phi_cz, phi_control, phi_target = best.x
    # fSim(theta + k pi) = [Z(pi) (x) Z(pi)]^k fSim(theta): fold theta and move k*pi into the local phases
    k = int(np.round(theta / np.pi))
    theta -= k * np.pi
    phi_control += k * np.pi
    phi_target += k * np.pi

    res_z = min(
        (
            minimize(lambda z: infidelity([0.0, np.pi, z[0], z[1]]), [a, b], method="Nelder-Mead", options=options)
            for _, a, b in seeds
        ),
        key=lambda r: r.fun,
    )

    phi_cz = float(phi_cz % (2 * np.pi))
    return {
        "theta": float(theta),
        "phi_cz": phi_cz,
        "cz_phase_error": wrap_to_pi(phi_cz - np.pi),
        "phi_control": wrap_to_pi(phi_control),
        "phi_target": wrap_to_pi(phi_target),
        "process_fidelity_model": 1.0 - float(best.fun),
        "process_fidelity_ideal_cz": process_fidelity(choi, ideal_cz_unitary()),
        "process_fidelity_local_z_corrected": 1.0 - float(res_z.fun),
        "local_z_only_phi_control": wrap_to_pi(res_z.x[0]),
        "local_z_only_phi_target": wrap_to_pi(res_z.x[1]),
    }
