from typing import Callable, Sequence

import itertools
import numpy as np
import xarray as xr

from quam_libs.lib.readout_mitigation import least_squares_mitigation
from .helpers import (
    generate_pauli_basis,
    gen_inverse_hadamard,
)


def get_pauli_data_nq(results_xr: xr.DataArray, n_qubits: int):
    """Estimate N-qubit Pauli coefficients from tomography outcomes.

    Parameters
    ----------
    results_xr : xarray.DataArray
        Tomography probabilities with a stacked ``tomo_axis`` coordinate and ``state`` data.
    n_qubits : int
        Number of qubits.

    Returns
    -------
    xarray.Dataset
        Dataset containing averaged Pauli coefficients under key ``value`` and
        appearance counts under key ``appearances``.
    """
    pauli_basis = generate_pauli_basis(n_qubits)
    inverse_hadamard = gen_inverse_hadamard(n_qubits)

    labels = [",".join(map(str, op)) for op in pauli_basis]
    paulis_data = xr.Dataset(
        {
            "value": (["pauli_op"], np.zeros(len(labels))),
            "appearances": (["pauli_op"], np.zeros(len(labels), dtype=int)),
        },
        coords={"pauli_op": labels},
    )

    tomo_axes = list(results_xr.coords["tomo_axis"].values)

    for tomo_axis in tomo_axes:
        tomo_data = results_xr.sel(tomo_axis=tomo_axis).data
        pauli_data = inverse_hadamard @ tomo_data

        local_paulis = []
        for bits in itertools.product([0, 1], repeat=n_qubits):
            label = []
            for q in range(n_qubits):
                if bits[q] == 0:
                    label.append(0)
                else:
                    label.append(tomo_axis[q] + 1)
            local_paulis.append(",".join(map(str, label)))

        for i, pauli in enumerate(local_paulis):
            paulis_data.value.loc[{"pauli_op": pauli}] += pauli_data[i]
            paulis_data.appearances.loc[{"pauli_op": pauli}] += 1

    paulis_data = xr.where(
        paulis_data.appearances != 0,
        paulis_data.value / paulis_data.appearances,
        paulis_data.value,
    )

    return paulis_data


def get_density_matrix(paulis_data: xr.Dataset, n_qubits: int):
    """Reconstruct a density matrix from N-qubit Pauli coefficients.

    Parameters
    ----------
    paulis_data : xarray.Dataset
        Dataset containing Pauli coefficients indexed by ``pauli_op``.
    n_qubits : int
        Number of qubits.

    Returns
    -------
    numpy.ndarray
        Complex ``(2**n_qubits, 2**n_qubits)`` density matrix.
    """
    i_mat = np.array([[1, 0], [0, 1]], dtype=complex)
    x_mat = np.array([[0, 1], [1, 0]], dtype=complex)
    y_mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z_mat = np.array([[1, 0], [0, -1]], dtype=complex)

    pauli_matrices = [i_mat, x_mat, y_mat, z_mat]

    dim = 2**n_qubits
    rho = np.zeros((dim, dim), dtype=complex)

    for op in itertools.product(range(4), repeat=n_qubits):
        p_mat = pauli_matrices[op[0]]
        for k in op[1:]:
            p_mat = np.kron(p_mat, pauli_matrices[k])

        key = ",".join(map(str, op))
        coeff = paulis_data.sel(pauli_op=key).item()
        rho += coeff * p_mat

    rho /= 2**n_qubits
    return rho


def build_corrected_results_xr(
    results_xr: xr.DataArray,
    qubit_groups_for_qua: Sequence,
    tomo_axis_names: Sequence[str],
    tomo_combinations: Sequence[tuple],
    state_labels: Sequence[str],
    num_qubits: int,
    confusion_matrix_provider: Callable,
):
    """Apply readout mitigation across all tomography settings.

    Parameters
    ----------
    results_xr : xarray.DataArray
        Unmitigated tomography probabilities indexed by qubit group and tomography axes.
    qubit_groups_for_qua : Sequence
        Sequence of qubit-group objects. Each object must provide ``name``.
    tomo_axis_names : Sequence[str]
        Names of tomography-axis coordinates.
    tomo_combinations : Sequence[tuple]
        All tomography-axis index combinations (typically product of ``[0,1,2]``).
    state_labels : Sequence[str]
        Computational basis labels for the ``state`` axis.
    num_qubits : int
        Number of qubits.
    confusion_matrix_provider : Callable
        Callable ``f(i, qg) -> np.ndarray`` returning the mitigation matrix for group ``i``.

    Returns
    -------
    xarray.DataArray
        Mitigated probabilities with a stacked ``tomo_axis`` coordinate.
    """
    corrected_results = []

    for i, qg in enumerate(qubit_groups_for_qua):
        conf_mat = confusion_matrix_provider(i, qg)
        corrected_results_qg = []

        for tomo_axes in tomo_combinations:
            sel_dict = {"qubit": qg.name}
            for axis_name, axis in zip(tomo_axis_names, tomo_axes):
                sel_dict[axis_name] = axis

            results = results_xr.sel(**sel_dict)

            corrected = least_squares_mitigation(conf_mat, results.data)
            corrected = corrected * (corrected > 0)
            corrected = corrected / corrected.sum()
            corrected_results_qg.append(corrected)

        corrected_results.append(corrected_results_qg)

    corrected_results = np.array(corrected_results).reshape(
        len(qubit_groups_for_qua), *([3] * num_qubits), 2**num_qubits
    )

    corrected_results_xr = xr.DataArray(
        corrected_results,
        dims=["qubit", *tomo_axis_names, "state"],
        coords={
            "qubit": [qg.name for qg in qubit_groups_for_qua],
            **{axis_name: [0, 1, 2] for axis_name in tomo_axis_names},
            "state": state_labels,
        },
    )
    return corrected_results_xr.stack(tomo_axis=tomo_axis_names)
