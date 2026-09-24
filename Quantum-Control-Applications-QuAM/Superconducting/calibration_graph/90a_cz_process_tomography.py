# %%
"""
Two-Qubit Quantum Process Tomography of the CZ Gate

Characterizes an existing, calibrated CZ gate of a qubit pair with standard two-qubit quantum process
tomography (QPT) and compares the reconstructed process with the ideal CZ = diag(1, 1, 1, -1).

Sequence (every shot loops over 16 input states x 9 tomography settings):
1. Reset both qubits (active or thermal reset).
2. Prepare the input state |c> (x) |t> with c, t in {|0>, |1>, |+>, |+i>}.
3. Apply qp.gates[operation] exactly once.
4. Apply the tomography pre-rotations (X: -y90, Y: +x90, Z: none) and read out both qubits.

Analysis:
1. Raw populations and (optionally) readout-mitigated populations (linear inversion of the confusion matrix).
2. Linear-inversion state tomography of the 16 output states (density matrices + Pauli expectation values).
3. Raw linear-inversion process reconstruction (PTM -> Choi -> chi), without any physicality constraint.
4. (Optional) A separate physical fit: least-squares fit of a completely positive, trace-preserving
   process (trace non-increasing when leakage is measured) directly to the measured frequencies,
   with the readout confusion matrix included in the forward model.
5. Comparison with the ideal CZ: process (entanglement) fidelity, average gate fidelity,
   ||chi_exp - chi_ideal||_F, trace-preservation error and Choi positivity, for both estimates.
6. (Optional) Coherent CZ error model U_eff = [Z(phi1) (x) Z(phi2)] fSim(theta, phi_CZ).

Conventions (qubit/tensor/basis ordering, pulse signs, readout assignment, chi normalisation, fidelity
definitions) are documented in calibration_utils/cz_process_tomography/analysis.py and saved in
node.results["conventions"]. In short: qubit 0 = control, |c t> -> 2c + t, Paulis ordered II, IX, ..., ZZ
(first letter = control), Tr chi = 1, F_pro = Tr[chi_ideal chi], F_avg = (4 F_pro + 1)/5 for TP maps.

Important limitations:
- QPT is sensitive to state-preparation and measurement (SPAM) errors: preparation, pre-rotation and readout
  errors all enter the reconstructed process. QPT fidelities are therefore not equivalent to randomized
  benchmarking fidelities.
- The linear-inversion estimate is unbiased but can be unphysical (negative Choi eigenvalues). The physical
  fit is always physical but biased at finite shot numbers. Both are reported; neither replaces the other.
- Without GEF readout (measure_leakage=False) the process is reconstructed within the computational subspace
  and does not independently quantify leakage.

Prerequisites:
- Calibrated single-qubit gates (x180, x90, -x90, y90, -y90) and readout of both qubits.
- The calibrated CZ gate qp.gates[operation] (including its single-qubit phase corrections).
- For readout mitigation: qp.confusion (34_2Q_confusion_matrix) or resonator.confusion_matrix (07b_IQ_Blobs).
- For measure_leakage=True: calibrated GEF readout (08d_IQ_Blobs_G_E_F) and EF_x180 pulses on both qubits.

Outcomes:
- Output density matrices, Choi / chi / PTM (linear inversion and physical fit), fidelities, CZ error parameters.
- Characterization only: the QuAM state is not updated.
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, active_reset_gef, readout_state, readout_state_gef
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
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import warnings
import xarray as xr
from calibration_utils.confusion_matrix import compute_kron_confusion_matrix
from calibration_utils.cz_process_tomography import (
    CONVENTIONS,
    OUTCOME_LABELS,
    PAULI_2Q_LABELS,
    PREP_LABELS,
    PREP_PULSES,
    TOMO_LABELS,
    TOMO_PULSES,
    choi_to_chi,
    choi_to_ptm,
    counts_from_shots,
    error_channel_pauli_probabilities,
    fit_cz_model,
    fit_physical_process,
    ideal_cz_unitary,
    input_states,
    measurement_projectors,
    mitigate_populations,
    model_residual_rms,
    pauli_expectations_of_states,
    pauli_sign_check,
    process_metrics,
    ptm_from_pauli_expectations,
    ptm_to_choi,
    reconstruct_output_states,
    unitary_to_choi,
    plot_chi_city,
    plot_pauli_expectations,
    plot_ptm,
    plot_summary,
)

# %% {Node_parameters}
qubit_pair_indexes = [4]  # The indexes of the qubit pairs to characterize
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s"%(i,i+1) for i in qubit_pair_indexes]
    num_shots: int = 5000
    """Single shots per (input state, tomography setting); 144 settings per shot loop."""
    operation: str = "Cz_flattop"
    """Name of the CZ gate in qp.gates to characterize."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    readout_mitigation: Literal["none", "pair_confusion", "kron_resonator"] = "pair_confusion"
    """none: raw populations; pair_confusion: qp.confusion (34_2Q_confusion_matrix);
    kron_resonator: kron of the resonator.confusion_matrix of both qubits (07b_IQ_Blobs)."""
    measure_leakage: bool = False
    """Use GEF readout (and GEF active reset) on both qubits to measure leakage to |2>."""
    fit_physical: bool = True
    """Also fit a physical (CP + trace-preserving / trace non-increasing) process to the data."""
    fit_cz_model: bool = True
    """Fit U_eff = [Z(phi1) x Z(phi2)] fSim(theta, phi_CZ) to the reconstructed process."""
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None


node = QualibrationNode(
    name="90a_cz_process_tomography", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

num_qubit_pairs = len(qubit_pairs)
operation_name = node.parameters.operation
for qp in qubit_pairs:
    if operation_name not in qp.gates:
        raise KeyError(f"{qp.name} has no gate '{operation_name}'. Available gates: {list(qp.gates.keys())}")

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

LEAKAGE_NOT_MEASURED = (
    "Process tomography is reconstructed within the computational subspace and does not independently quantify leakage."
)


def get_readout_confusion(qp, method):
    """Readout confusion matrix M[measured, prepared] for the outcomes |control target> = 00, 01, 10, 11."""
    if method == "none":
        return np.eye(4)
    if method == "pair_confusion":
        if qp.confusion is None:
            raise ValueError(f"{qp.name}.confusion is empty: run 34_2Q_confusion_matrix or choose another readout_mitigation.")
        # 34_2Q_confusion_matrix stores confusion[measured][prepared], index 2*control + target
        confusion = np.array(qp.confusion, dtype=float)
    else:
        for q in (qp.qubit_control, qp.qubit_target):
            if q.resonator.confusion_matrix is None:
                raise ValueError(f"{q.name}.resonator.confusion_matrix is empty: run 07b_IQ_Blobs or choose another readout_mitigation.")
        # 07b_IQ_Blobs stores resonator.confusion_matrix[prepared][measured]
        confusion = compute_kron_confusion_matrix([qp.qubit_control, qp.qubit_target]).T
    if not np.allclose(confusion.sum(axis=0), 1, atol=1e-2):
        warnings.warn(
            f"{qp.name}: columns of the '{method}' confusion matrix do not sum to 1 "
            f"({confusion.sum(axis=0)}); expected M[measured, prepared]."
        )
    return confusion


def prepare_input_state(qubit, prep_index):
    """Real-time input-state selection: PREP_LABELS[i] (|0>, |1>, |+>, |+i>) is prepared with PREP_PULSES[i]."""
    for idx, pulse in enumerate(PREP_PULSES):
        if pulse is not None:
            with if_(prep_index == idx):
                qubit.xy.play(pulse)


def apply_tomography_rotation(qubit, tomo_index):
    """Real-time pre-rotation mapping the TOMO_LABELS[i] (X, Y, Z) eigenbasis onto Z: X: -y90, Y: +x90, Z: none."""
    for idx, pulse in enumerate(TOMO_PULSES):
        if pulse is not None:
            with if_(tomo_index == idx):
                qubit.xy.play(pulse)


# %% {QUA_program}
n_shots = node.parameters.num_shots  # The number of shots per setting

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
measure_leakage = node.parameters.measure_leakage
readout_base = 3 if measure_leakage else 2  # saved state = readout_base * state_control + state_target

with program() as CZ_process_tomography:
    n = declare(int)
    n_st = declare_stream()
    prep_control = declare(int)
    prep_target = declare(int)
    tomo_control = declare(int)
    tomo_target = declare(int)
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state = [declare(int) for _ in range(num_qubit_pairs)]
    state_st = [declare_stream() for _ in range(num_qubit_pairs)]

    for i, qp in enumerate(qubit_pairs):
        if not node.parameters.simulate:
            # Bring the active qubits to the minimum frequency point
            if flux_point == "independent":
                machine.apply_all_flux_to_min()
            elif flux_point == "joint":
                machine.apply_all_flux_to_joint_idle()
            else:
                machine.apply_all_flux_to_zero()
            wait(1000)

        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st)
            with for_(prep_control, 0, prep_control < len(PREP_PULSES), prep_control + 1):
                with for_(prep_target, 0, prep_target < len(PREP_PULSES), prep_target + 1):
                    with for_(tomo_control, 0, tomo_control < len(TOMO_PULSES), tomo_control + 1):
                        with for_(tomo_target, 0, tomo_target < len(TOMO_PULSES), tomo_target + 1):
                            # reset
                            if not node.parameters.simulate:
                                if node.parameters.reset_type == "active":
                                    if measure_leakage:
                                        active_reset_gef(qp.qubit_control)
                                        active_reset_gef(qp.qubit_target)
                                    else:
                                        active_reset(qp.qubit_control)
                                        active_reset(qp.qubit_target)
                                else:
                                    wait(5 * max(qp.qubit_control.thermalization_time, qp.qubit_target.thermalization_time) * u.ns)
                            qp.align()
                            # input state
                            prepare_input_state(qp.qubit_control, prep_control)
                            prepare_input_state(qp.qubit_target, prep_target)
                            qp.align()
                            # gate under test, applied exactly once
                            qp.gates[operation_name].execute()
                            qp.align()
                            # tomography pre-rotations
                            apply_tomography_rotation(qp.qubit_control, tomo_control)
                            apply_tomography_rotation(qp.qubit_target, tomo_target)
                            qp.align()
                            # readout
                            if measure_leakage:
                                readout_state_gef(qp.qubit_control, state_control[i])
                                readout_state_gef(qp.qubit_target, state_target[i])
                            else:
                                readout_state(qp.qubit_control, state_control[i])
                                readout_state(qp.qubit_target, state_target[i])
                            assign(state[i], state_control[i] * readout_base + state_target[i])
                            save(state[i], state_st[i])
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st[i].buffer(len(TOMO_PULSES)).buffer(len(TOMO_PULSES)).buffer(len(PREP_PULSES)).buffer(len(PREP_PULSES)).buffer(n_shots).save(f"state{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, CZ_process_tomography, simulation_config)
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    node.results = {"figure": plt.gcf()}
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
        job = qm.execute(CZ_process_tomography)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_shots, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the single shots from the OPX and convert them into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(
            job.result_handles,
            qubit_pairs,
            {
                "tomo_target": list(TOMO_LABELS),
                "tomo_control": list(TOMO_LABELS),
                "prep_target": list(PREP_LABELS),
                "prep_control": list(PREP_LABELS),
                "N": np.linspace(1, n_shots, n_shots),
            },
        )
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
    node.results = {"ds": ds}

# %% {Data_analysis}
if not node.parameters.simulate:
    # Parameters that describe the stored data (taken from the loaded node when load_data_id is set)
    measure_leakage = node.parameters.measure_leakage
    readout_base = 3 if measure_leakage else 2
    mitigation = node.parameters.readout_mitigation
    if measure_leakage and mitigation != "none":
        warnings.warn(
            "measure_leakage=True: the confusion matrix was measured with g/e readout and is applied to the "
            "computational-subspace outcomes of the GEF readout only (approximate mitigation)."
        )

    # Ideal model in the reconstruction basis
    rhos_in = input_states()
    projectors = measurement_projectors()
    u_cz = ideal_cz_unitary()
    choi_ideal = unitary_to_choi(u_cz)
    chi_ideal = choi_to_chi(choi_ideal)
    ptm_ideal = choi_to_ptm(choi_ideal)
    expectations_ideal = pauli_expectations_of_states(np.array([u_cz @ rho @ u_cz.conj().T for rho in rhos_in]))

    tomography_data = {key: [] for key in ("population_raw", "population_mitigated", "expectation_raw", "expectation_mitigated", "leaked_population")}
    qpt_results = {}
    estimates_by_pair = {}
    for qp in qubit_pairs:
        # 1. Populations (raw and readout-mitigated)
        shots = ds.state.sel(qubit=qp.name).transpose("N", "prep_control", "prep_target", "tomo_control", "tomo_target").values
        counts, leak_counts, n_total = counts_from_shots(shots, readout_base)
        confusion = get_readout_confusion(qp, mitigation)
        populations_raw = counts / n_total
        populations = mitigate_populations(populations_raw, confusion)

        # 2. Output density matrices (raw linear-inversion state tomography)
        expectations_raw, _ = reconstruct_output_states(populations_raw)
        expectations, rhos_out = reconstruct_output_states(populations)
        sign_slopes = pauli_sign_check(expectations, expectations_ideal)
        if any(slope < 0 for slope in sign_slopes.values()):
            warnings.warn(f"{qp.name}: negative measured-vs-ideal slope for single-qubit Paulis {sign_slopes}; check pulse sign conventions.")

        # 3. Raw linear-inversion process reconstruction (no physicality constraint)
        ptm_li = ptm_from_pauli_expectations(expectations, rhos_in)
        choi_li = ptm_to_choi(ptm_li)
        estimates = {"linear_inversion": {"choi": choi_li, "chi": choi_to_chi(choi_li), "ptm": ptm_li}}

        # 4. Separate physical fit (CP + TP, or CP + trace non-increasing when leakage is measured)
        physical_fit_info = None
        if node.parameters.fit_physical:
            fit = fit_physical_process(
                counts, n_total, rhos_in, projectors, confusion,
                constraint="TNI" if measure_leakage else "TP", choi_init=choi_li,
            )
            estimates["physical_fit"] = {"choi": fit["choi"], "chi": choi_to_chi(fit["choi"]), "ptm": choi_to_ptm(fit["choi"])}
            physical_fit_info = {k: v for k, v in fit.items() if k != "choi"}

        # 5. Comparison with the ideal CZ
        metrics = {}
        error_probabilities = {}
        for name, est in estimates.items():
            est["chi_diff"] = est["chi"] - chi_ideal
            metrics[name] = process_metrics(est["choi"], u_cz)
            metrics[name]["residual_rms"] = model_residual_rms(est["choi"], counts, n_total, rhos_in, projectors, confusion)
            error_probabilities[name] = error_channel_pauli_probabilities(est["ptm"], ptm_ideal)
        primary = "physical_fit" if "physical_fit" in estimates else "linear_inversion"

        # 6. CZ-specific coherent error model
        cz_model = None
        if node.parameters.fit_cz_model:
            cz_model = fit_cz_model(estimates[primary]["choi"])
            cz_model["estimate"] = primary

        # 7. Leakage (only when independently measured with GEF readout)
        if measure_leakage:
            leakage = {
                "measured": True,
                "mean_leaked_population": float(leak_counts.sum() / (n_total * leak_counts.size)),
                "leaked_population_per_input": leak_counts.mean(axis=1) / n_total,
                **{f"leakage_{name}": 1 - metrics[name]["mean_output_trace"] for name in estimates},
            }
            leakage_text = (
                f"Leakage (GEF readout, includes SPAM): 1 - Tr Λ(1/4) = {leakage['leakage_linear_inversion']:.4f} (linear inversion); "
                f"mean leaked population = {leakage['mean_leaked_population']:.4f}"
            )
        else:
            leakage = {"measured": False, "note": LEAKAGE_NOT_MEASURED}
            leakage_text = LEAKAGE_NOT_MEASURED

        print(f"\n===== {qp.name}: process tomography of gates['{operation_name}'] =====")
        for name, m in metrics.items():
            print(
                f"{name:17s} F_pro={m['process_fidelity']:.4f}  F_avg={m['average_gate_fidelity']:.4f}  "
                f"||Δχ||_F={m['chi_frobenius_distance']:.4f}  TP err={m['trace_preservation_error']:.1e}  "
                f"min eig(J/d)={m['choi_min_eigenvalue']:+.1e}"
            )
        if cz_model is not None:
            print(
                f"CZ model ({primary}): φ_CZ={np.degrees(cz_model['phi_cz']):.2f}° (error {np.degrees(cz_model['cz_phase_error']):+.2f}°), "
                f"φ1={np.degrees(cz_model['phi_control']):+.2f}°, φ2={np.degrees(cz_model['phi_target']):+.2f}°, "
                f"θ={np.degrees(cz_model['theta']):+.2f}°, F_model={cz_model['process_fidelity_model']:.4f}"
            )
        print(leakage_text)

        qpt_results[qp.name] = {
            "confusion_matrix": confusion,
            "input_states": rhos_in,
            "output_density_matrices": rhos_out,
            "chi_ideal": chi_ideal,
            "ptm_ideal": ptm_ideal,
            "choi_ideal": choi_ideal,
            **{
                f"{key}_{name}": est[key]
                for name, est in estimates.items()
                for key in ("chi", "chi_diff", "ptm", "choi")
            },
            "metrics": metrics,
            "error_pauli_probabilities": error_probabilities,
            "primary_estimate": primary,
            "physical_fit_info": physical_fit_info,
            "cz_model": cz_model,
            "pauli_sign_check": sign_slopes,
            "leakage": leakage,
        }
        node.results[f"{qp.name}_process_fidelity"] = metrics[primary]["process_fidelity"]
        node.results[f"{qp.name}_average_gate_fidelity"] = metrics[primary]["average_gate_fidelity"]
        estimates_by_pair[qp.name] = {
            "estimates": estimates, "metrics": metrics, "error_probabilities": error_probabilities,
            "cz_model": cz_model, "leakage_text": leakage_text, "expectations": expectations,
        }

        tomography_data["population_raw"].append(populations_raw.reshape(4, 4, 3, 3, 4))
        tomography_data["population_mitigated"].append(populations.reshape(4, 4, 3, 3, 4))
        tomography_data["expectation_raw"].append(expectations_raw.reshape(4, 4, 16))
        tomography_data["expectation_mitigated"].append(expectations.reshape(4, 4, 16))
        tomography_data["leaked_population"].append((leak_counts / n_total).reshape(4, 4, 3, 3))

    population_dims = ["qubit", "prep_control", "prep_target", "tomo_control", "tomo_target", "outcome"]
    expectation_dims = ["qubit", "prep_control", "prep_target", "pauli"]
    ds_tomography = xr.Dataset(
        {
            "population_raw": (population_dims, np.array(tomography_data["population_raw"])),
            "population_mitigated": (population_dims, np.array(tomography_data["population_mitigated"])),
            "expectation_raw": (expectation_dims, np.array(tomography_data["expectation_raw"])),
            "expectation_mitigated": (expectation_dims, np.array(tomography_data["expectation_mitigated"])),
            "expectation_ideal": (expectation_dims[1:], expectations_ideal.reshape(4, 4, 16)),
        },
        coords={
            "qubit": [qp.name for qp in qubit_pairs],
            "prep_control": list(PREP_LABELS),
            "prep_target": list(PREP_LABELS),
            "tomo_control": list(TOMO_LABELS),
            "tomo_target": list(TOMO_LABELS),
            "outcome": list(OUTCOME_LABELS),
            "pauli": list(PAULI_2Q_LABELS),
        },
    )
    if measure_leakage:
        ds_tomography["leaked_population"] = (population_dims[:-1], np.array(tomography_data["leaked_population"]))
    node.results["ds_tomography"] = ds_tomography
    node.results["process_tomography"] = qpt_results
    node.results["conventions"] = CONVENTIONS

# %% {Plotting}
if not node.parameters.simulate:
    figures = {}
    for qp in qubit_pairs:
        pair = estimates_by_pair[qp.name]
        prefix = f"{qp.name} gates['{operation_name}']"

        figures[f"{qp.name}_pauli_expectations"] = plot_pauli_expectations(
            pair["expectations"], expectations_ideal, f"{prefix}: output-state Pauli expectation values (readout mitigation: {mitigation})"
        )
        plt.show()
        for name, est in pair["estimates"].items():
            m = pair["metrics"][name]
            title = f"{prefix}, {name}: F_pro = {m['process_fidelity']:.4f}, F_avg = {m['average_gate_fidelity']:.4f}"
            figures[f"{qp.name}_chi_city_{name}"] = plot_chi_city(est["chi"], chi_ideal, f"χ matrix, {title}")
            plt.show()
            figures[f"{qp.name}_ptm_{name}"] = plot_ptm(est["ptm"], ptm_ideal, f"Pauli transfer matrix, {title}")
            plt.show()
        figures[f"{qp.name}_summary"] = plot_summary(
            pair["metrics"], pair["error_probabilities"], pair["cz_model"], pair["leakage_text"], f"{prefix}: CZ process tomography summary"
        )
        plt.show()
    node.results["figures"] = figures

# %% {Update_state}
# Characterization only: the QuAM state is not modified.

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()

# %%
