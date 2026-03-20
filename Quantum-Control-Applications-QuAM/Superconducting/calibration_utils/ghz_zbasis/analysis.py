from typing import Sequence

import numpy as np
import xarray as xr

from quam_libs.lib.readout_mitigation import least_squares_mitigation, get_nq_confusion_matrix
from calibration_utils.ghz_tomography.helpers import get_kron_confusion_matrix


def compute_zbasis_mitigated_results(
    ds: xr.Dataset,
    qubit_groups_for_qua: Sequence,
    qubit_group_names: Sequence[Sequence[str]],
    machine,
    num_states: int,
    num_shots: int,
):
    """Compute Z-basis mitigated state distributions for Kron and NQ methods.

    Returns
    -------
    tuple[dict, dict, dict, dict]
        raw_results, corrected_results_by_method, fidelities_by_method, fidelity_differences
    """
    raw_results = {}
    corrected_results_by_method = {"kron": {}, "nq": {}}
    fidelities_by_method = {"kron": {}, "nq": {}}
    fidelity_differences = {}

    for i, qg in enumerate(qubit_groups_for_qua):
        measured_states = ds.sel(qubit=qg.name).state.values.astype(int)
        raw_results[qg.name] = np.bincount(measured_states, minlength=num_states) / num_shots

        method_conf_mats = {"kron": get_kron_confusion_matrix(qg)}
        conf_mat_nq = get_nq_confusion_matrix(qubit_group_names[i], machine)
        if conf_mat_nq is not None:
            method_conf_mats["nq"] = conf_mat_nq

        for method_name, conf_mat in method_conf_mats.items():
            corrected = least_squares_mitigation(conf_mat, raw_results[qg.name])
            corrected_results_by_method[method_name][qg.name] = corrected
            fidelities_by_method[method_name][qg.name] = corrected[0] + corrected[num_states - 1]

        if qg.name in fidelities_by_method["nq"]:
            fidelity_differences[qg.name] = (
                fidelities_by_method["nq"][qg.name] - fidelities_by_method["kron"][qg.name]
            )

    return raw_results, corrected_results_by_method, fidelities_by_method, fidelity_differences
