# %%
"""
Multi-Qubit GHZ State Tomography

This calibration script prepares an N-qubit GHZ state (N >= 2) and performs full
tomography by sweeping local tomography axes on each qubit.

Sequence overview:
1. Reset all qubits (active or thermal reset).
2. Prepare a GHZ chain using nearest-neighbor CZ gates and single-qubit rotations.
3. Apply tomography pre-rotations (X/Y/Z basis mapping via x90/y90/no-op choices).
4. Read out all qubits and accumulate state populations over many shots.

Analysis overview:
1. Build tomography probability distributions for all axis combinations.
2. Apply readout mitigation in two ways:
   - `kron`: tensor product of per-qubit resonator confusion matrices.
   - `nq`: full N-qubit confusion matrix from `get_nq_confusion_matrix(...)`.
3. Reconstruct Pauli coefficients and density matrices for both methods.
4. Report fidelity and purity relative to the ideal GHZ target state.

Prerequisites:
- Calibrated single-qubit control and readout for all qubits in each group.
- Available nearest-neighbor CZ operations along the selected qubit chain.
- Valid readout confusion matrices for mitigation (`kron` and/or `nq`).

Outputs:
- Mitigated tomography datasets (for both `kron` and `nq` confusion matrices).
- Reconstructed density matrices and per-group metrics (fidelity, purity).
- Visualization of real/imaginary density matrix components (3D and heatmaps).
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
import itertools
import xarray as xr
from quam_libs.lib.readout_mitigation import get_nq_confusion_matrix
from contextlib import contextmanager
from calibration_utils.ghz_tomography import (
    build_corrected_results_xr,
    fidelity_with_pure_target,
    get_density_matrix,
    get_kron_confusion_matrix,
    get_pauli_data_nq,
    ghz_density_matrix,
    ghz_state_vector,
)
from calibration_utils.ghz_tomography.plotting import plot_3d_component, plot_density_heatmap


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_groups: List[List[str]] = [["q1", "q2", "q3"]]
    # List of qubit chains to characterize (all groups must have same length).
    num_shots: int = 1000
    # Number of averages per tomography setting.
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    # Flux-bias mode during experiment.
    reset_type: Literal["active", "thermal"] = "active"
    # Qubit reset strategy before each shot.
    simulate: bool = False
    # If True, run OPX simulation instead of hardware execution.
    timeout: int = 100
    # QOP session timeout in seconds.
    load_data_id: Optional[int] = None
    # If set, skip execution and load a previously saved dataset id.
    plot_level: Literal["full", "minimal"] = "minimal"
    # full: both methods with bars+heatmaps, minimal: only nq with 3D bars.


node = QualibrationNode(name="41b_GHZ_tomography", parameters=Parameters())
assert not (
    node.parameters.simulate and node.parameters.load_data_id is not None
), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine

# Create qubit groups from parameters.qubit_groups
# node.parameters.qubit_groups is List[List[str]], e.g., [["qD4","qD3","qC4","qC2","qC1"]]
qubit_objects_raw = [[machine.qubits[qubit] for qubit in group] for group in node.parameters.qubit_groups]
num_qubit_groups = len(qubit_objects_raw)

# Validate that all groups have the same number of qubits
if len(set(len(group) for group in qubit_objects_raw)) > 1:
    raise ValueError("All qubit groups must have the same number of qubits")
num_qubits = len(qubit_objects_raw[0])
num_states = 2**num_qubits

# Validate number of qubits
if num_qubits < 2:
    raise ValueError(f"Number of qubits must be at least 2, got {num_qubits}")


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

        # Find qubit pairs for adjacent qubits.
        # Check both orderings of pair names (e.g., "qD4-qD3" and "qD3-qD4").
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
                # Fallback: search by qubit objects (in case pair structure is different).
                for qp in machine.qubit_pairs.values():
                    if qp.qubit_control in [q1, q2] and qp.qubit_target in [q1, q2]:
                        qg.qubit_pairs[pair_key] = qp
                        break
        return qg


# Create qubit group structures for QUA program (needed for gate operations)
qubit_groups_for_qua = [QubitGroup.from_qubits(qubits, machine) for qubits in qubit_objects_raw]

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

####################
# Helper functions #
####################


@contextmanager
def nested_tomo_loops(tomo_axes, idx=0):
    """Recursively emit nested QUA tomography loops over all declared axes.

    This helper is used as:
        with nested_tomo_loops(tomo_axes):
            ...
    and expands into one nested QUA ``for_`` loop per entry in ``tomo_axes``,
    each iterating over tomography bases 0, 1, 2.
    """
    if idx == len(tomo_axes):
        yield
        return

    with for_(tomo_axes[idx], 0, tomo_axes[idx] < 3, tomo_axes[idx] + 1):
        with nested_tomo_loops(tomo_axes, idx + 1):
            yield


# %% {QUA_program}
n_shots = node.parameters.num_shots  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as GHZ_tomography:
    n = declare(int)
    n_st = declare_stream()
    state_vars = [declare(int) for _ in range(num_qubits)]
    state_st_vars = [declare_stream() for _ in range(num_qubits)]
    state = [declare(int) for _ in range(num_qubit_groups)]
    state_st = [declare_stream() for _ in range(num_qubit_groups)]
    tomo_axes = [declare(int) for _ in range(num_qubits)]

    for i, qg in enumerate(qubit_groups_for_qua):
        # Bring the active qubits to the minimum frequency point
        machine.apply_all_flux_to_joint_idle()
        wait(1000)

        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st)
            with nested_tomo_loops(tomo_axes):
                if node.parameters.reset_type == "active":
                    for q in qg.qubits:
                        active_reset(q)
                else:
                    wait(5 * qg.max_thermalization_time * u.ns)
                align()

                # GHZ state preparation for any number of qubits >= 2.
                qg.qubits[0].xy.play("-y90")
                qg.qubits[1].xy.play("y90")
                qg.qubit_pairs["pair_01"].macros["cz"].apply()
                qg.qubits[0].xy.play("y90")
                for qubit_idx in range(2, num_qubits):
                    align()
                    qg.qubits[qubit_idx].xy.play("y90")
                    qg.qubit_pairs[f"pair_{qubit_idx - 1}{qubit_idx}"].macros["cz"].apply()
                    qg.qubits[qubit_idx].xy.play("-y90")

                for idx, q in enumerate(qg.qubits):
                    with if_(tomo_axes[idx] == 0):
                        q.xy.play("y90")
                    with if_(tomo_axes[idx] == 1):
                        q.xy.play("x90")
                align()

                # Readout all qubits
                for idx, q in enumerate(qg.qubits):
                    readout_state(q, state_vars[idx])
                    save(state_vars[idx], state_st_vars[idx])
                state_expr = 0
                for idx in range(num_qubits):
                    state_expr += state_vars[idx] * (2 ** (num_qubits - idx - 1))
                assign(state[i], state_expr)
                save(state[i], state_st[i])
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_groups):
            state_stream = state_st[i]
            for _ in range(num_qubits):
                state_stream = state_stream.buffer(3)
            state_stream.buffer(n_shots).save(f"state{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, GHZ_tomography, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(GHZ_tomography)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_shots, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    tomo_axis_names = [f"tomo_axis_{idx}" for idx in range(num_qubits)]
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        fetch_axes = {axis_name: [0, 1, 2] for axis_name in reversed(tomo_axis_names)}
        fetch_axes["N"] = np.linspace(1, n_shots, n_shots)
        ds = fetch_results_as_xarray(
            job.result_handles,
            qubit_groups_for_qua,
            fetch_axes,
        )
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
    node.results = {"ds": ds}

# %% {Data analysis}
# Apply readout error mitigation
if not node.parameters.simulate:
    states = list(range(2**num_qubits))
    results = []
    for state in states:
        results.append((ds.state == state).sum(dim="N") / node.parameters.num_shots)

results_xr = xr.concat(results, dim=xr.DataArray(states, name="state"))
results_xr = results_xr.rename({"dim_0": "state"})
results_xr = results_xr.stack(tomo_axis=tomo_axis_names)
state_labels = [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)]
tomo_combinations = list(itertools.product([0, 1, 2], repeat=num_qubits))

corrected_results_xr_kron = build_corrected_results_xr(
    results_xr=results_xr,
    qubit_groups_for_qua=qubit_groups_for_qua,
    tomo_axis_names=tomo_axis_names,
    tomo_combinations=tomo_combinations,
    state_labels=state_labels,
    num_qubits=num_qubits,
    confusion_matrix_provider=lambda _, qg: get_kron_confusion_matrix(qg),
)
corrected_results_xr_nq = build_corrected_results_xr(
    results_xr=results_xr,
    qubit_groups_for_qua=qubit_groups_for_qua,
    tomo_axis_names=tomo_axis_names,
    tomo_combinations=tomo_combinations,
    state_labels=state_labels,
    num_qubits=num_qubits,
    confusion_matrix_provider=lambda i, _: get_nq_confusion_matrix(node.parameters.qubit_groups[i], machine),
)

# Calculate density matrices for both confusion-matrix constructions
paulis_data_by_method = {"kron": {}, "nq": {}}
rhos_by_method = {"kron": {}, "nq": {}}
for qg in qubit_groups_for_qua:
    paulis_data_by_method["kron"][qg.name] = get_pauli_data_nq(corrected_results_xr_kron.sel(qubit=qg.name), num_qubits)
    rhos_by_method["kron"][qg.name] = get_density_matrix(paulis_data_by_method["kron"][qg.name], num_qubits)

    paulis_data_by_method["nq"][qg.name] = get_pauli_data_nq(corrected_results_xr_nq.sel(qubit=qg.name), num_qubits)
    rhos_by_method["nq"][qg.name] = get_density_matrix(paulis_data_by_method["nq"][qg.name], num_qubits)


ideal_dat = ghz_density_matrix(num_qubits)
ideal_psi = ghz_state_vector(num_qubits)

for method_name, rhos in rhos_by_method.items():
    for qg in qubit_groups_for_qua:
        fidelity = fidelity_with_pure_target(rhos[qg.name], ideal_psi)
        purity = np.abs(np.trace(rhos[qg.name] @ rhos[qg.name]))
        print(f"[{method_name}] Fidelity of {qg.name}: {fidelity:.3f}")
        print(f"[{method_name}] Purity of {qg.name}: {purity:.3f}")
        print()
        node.results[f"{qg.name}_{method_name}_fidelity"] = fidelity
        node.results[f"{qg.name}_{method_name}_purity"] = purity

        # Backward-compatible result keys keep the previous (kron) behavior.
        if method_name == "kron":
            node.results[f"{qg.name}_fidelity"] = fidelity
            node.results[f"{qg.name}_purity"] = purity

# %% {Plotting}
figures = {}
plot_level = node.parameters.plot_level
methods_to_plot = ["kron", "nq"] if plot_level == "full" else ["nq"]

for method_name in methods_to_plot:
    for qg in qubit_groups_for_qua:
        plot_fidelity = node.results[f"{qg.name}_{method_name}_fidelity"]
        plot_title = f"{qg.name} [{method_name}], GHZ state fidelity={plot_fidelity:.3f}"

        fig1 = plot_3d_component(
            rhos_by_method[method_name][qg.name],
            ideal_dat,
            num_qubits,
            title=plot_title,
            component="real",
        )
        figures[f"{qg.name}_{method_name}_3d_real"] = fig1
        plt.show()

        fig2 = plot_3d_component(
            rhos_by_method[method_name][qg.name],
            ideal_dat,
            num_qubits,
            title=plot_title,
            component="imag",
        )
        figures[f"{qg.name}_{method_name}_3d_imag"] = fig2
        plt.show()

        if plot_level == "full":
            fig3 = plot_density_heatmap(
                rhos_by_method[method_name][qg.name],
                num_qubits,
                title=plot_title,
                component="real",
                annotate_values=num_qubits <= 3,
            )
            figures[f"{qg.name}_{method_name}_heatmap_real"] = fig3
            plt.show()

            fig4 = plot_density_heatmap(
                rhos_by_method[method_name][qg.name],
                num_qubits,
                title=plot_title,
                component="imag",
                annotate_values=num_qubits <= 3,
            )
            figures[f"{qg.name}_{method_name}_heatmap_imag"] = fig4
            plt.show()

node.results["figures"] = figures

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qg.name: "successful" for qg in qubit_groups_for_qua}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()
# %%
