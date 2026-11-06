# %%
"""
Two-qubit Cross-Entropy Benchmarking (XEB).

Random single-qubit circuits interleaved with a two-qubit gate; measured bitstring
probabilities are compared to the ideal evolution to estimate the entangling gate fidelity.

Set execution_mode="hardware" to talk to the OPX; "qiskit_simulation" runs a purely classical
offline simulation (the current default in this file).

Error analysis (see the Purity and 2Q_unitary_estimation sections):
    total error        = 1 - F_total     (linear XEB decay base)
    incoherent error   = 1 - f_cycle     (speckle purity decay base, sqrt of the purity base)
    coherent error     = f_cycle - F_total
All numbers are per cycle (two single-qubit gates + one CZ), not per CZ gate; use two_qubit_gate="none"
as a reference run to isolate the CZ. Shots where a qubit is read in |f> are discarded and the bitstring
probabilities are normalized over the remaining shots; leakage is reported separately.

node.save() writes this node's results to the Qualibrate data folder and also rewrites
the full in-memory QuAM back to state.json under configuration/quam_state. This node does
not change any calibration field (frequencies, confusion matrices, gef_centers, ...), but
the file is still rewritten in full.
"""

# %%
from typing import ClassVar, Literal, Optional, List

import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qm.qua import align
from qualibrate import NodeParameters, QualibrationNode
from quam_libs.lib.save_utils import restore_load_data_id, resolve_qubit_pairs_from_node

from quam_libs.components import QuAM, TransmonPair
from quam_libs.experiments.two_qubit_xeb import (
    QUAGate,
    XEB,
    XEBConfig,
    XEBResult,
    backend as fake_backend,
)


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q4_q5"]
    """Qubit pair names to run on; None or "" uses machine.active_qubit_pairs."""

    readout_qubits: Optional[List[str]] = ["q4", "q5"]
    """Qubits to also read out (for multiplexed-readout conditions); None or "" uses machine.active_qubits."""

    seqs: int = 50
    """Number of random circuits per depth."""

    depths: list[int] = list(range(1, 40, 1))
    """Circuit depths (number of two-qubit-gate layers) to run and fit fidelity decay over."""

    n_shots: int = 500
    """Number of shots per circuit."""

    baseline_gate_name: str = "x90"
    """Name of the calibrated pi/2 pulse used to play the single-qubit gate set."""

    two_qubit_gate: Literal["none", "cz"] = "cz"
    """"cz" inserts cz_macro_name at every depth; "none" runs single-qubit circuits only."""

    cz_macro_name: str = "Cz"
    """Name of the two-qubit gate macro under qubit_pair.gates to use when two_qubit_gate="cz"."""

    control_readout_mode: Literal[2, 3] = 2
    """Readout levels for the control qubit: 2 (G/E) or 3 (G/E/F, tracks leakage separately)."""

    target_readout_mode: Literal[2, 3] = 2
    """Readout levels for the target qubit, same meaning as control_readout_mode."""

    reset_method: Literal["active", "active_gef", "cooldown"] = "active"
    """Reset between shots: "active" (G/E), "active_gef" (G/E/F, needs gef_centers), or "cooldown"."""

    reset_cooldown_time_ns: int = 200000
    """Cooldown wait time in ns, used only when reset_method="cooldown"."""

    reset_max_tries: int = 15
    """Maximum reset attempts, used when reset_method is "active" or "active_gef"."""

    execution_mode: Literal["hardware", "hardware_simulation", "qiskit_simulation"] =  "hardware"
    """"hardware" runs on the OPX; "hardware_simulation" simulates the QUA program; "qiskit_simulation" runs offline."""

    simulation_duration_ns: int = 200000
    """Simulated waveform duration in ns, used only when execution_mode="hardware_simulation"."""

    generate_new_data: bool = True
    """If False with execution_mode="hardware", no new job is run; use load_data_id to reload past results instead."""

    disjoint_processing: bool = False
    """If True, compute each qubit's XEB fidelity independently instead of jointly over both qubits."""

    apply_confusion_matrix: bool = True
    """If True, correct measured probabilities using each qubit's resonator.confusion_matrix."""

    estimate_2q_unitary: bool = True
    """If True, fit the CZ conditional phase, iSWAP angle, and local Z phases from the measured data."""

    unitary_fit_max_depth: Optional[int] = 15
    """Only depths up to this value enter the 2Q unitary fit (deep circuits are close to uniform and carry little
    information about the unitary); None uses all depths."""

    fit_min_depth: int = 6
    """Shallowest depth used for the F_total / f_cycle decay fits (dropped when fewer than 4 deeper points remain)."""

    correct_purity_shot_noise: bool = True
    """If True, subtract the expected finite-shot noise from the speckle-purity variance."""

    seed: int = 1234
    """Random seed for generating the single-qubit gate sequences."""

    timeout: int = 100
    """Reserved; not currently used."""

    load_data_id: Optional[int] = None
    """If set, reloads a previous node's saved data instead of running a new experiment."""

    targets_name: ClassVar[str] = "qubit_pairs"
    """Qualibrate target field name; not an experiment parameter."""


node = QualibrationNode(name="96_XEB", parameters=Parameters())


# %% {Helpers}
def cz_gate(qubit_pair: TransmonPair):
    qubit_pair.align()
    qubit_pair.gates[node.parameters.cz_macro_name].execute()
    align()


def unique_qubits_from_pairs(qubit_pairs: list[TransmonPair]):
    qubits = []
    for pair in qubit_pairs:
        for qubit in (pair.qubit_control, pair.qubit_target):
            if qubit not in qubits:
                qubits.append(qubit)
    return qubits


def serialize_linear_fidelities(result):
    if result.xeb_config.disjoint_processing:
        serialized = {}
        for qubit_name, fidelity in zip(result.qubit_names, result.linear_fidelities):
            serialized[qubit_name] = {
                "depth": np.asarray(fidelity["depth"]),
                "fidelity": np.asarray(fidelity["fidelity"]),
            }
        return serialized

    return {
        "depth": np.asarray(result.linear_fidelities["depth"]),
        "fidelity": np.asarray(result.linear_fidelities["fidelity"]),
    }


def add_result_entry(results_dict, key_prefix, value):
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            add_result_entry(results_dict, f"{key_prefix}_{sub_key}", sub_value)
    elif isinstance(value, (np.ndarray, np.generic, int, float, str, bool)):
        results_dict[key_prefix] = value
    elif value is None:
        results_dict[key_prefix] = value


def qubit_names(qubits):
    return [qubit.name if hasattr(qubit, "name") else str(qubit) for qubit in qubits]


def print_xeb_setup(xeb_config: XEBConfig, qubit_pairs: list[TransmonPair], target_qubits):
    print("target_qubits: %s" % qubit_names(target_qubits))

    tot_pts = xeb_config.seqs * len(xeb_config.depths) * xeb_config.n_shots
    print("Number of points: %s" % tot_pts)

    if xeb_config.reset_method == "cooldown":
        base_runtime_min = tot_pts / (150 * 27 * 700) * 19.35
        cooldown_factor = node.parameters.reset_cooldown_time_ns / 40000
        print("time required: %s min" % (cooldown_factor * base_runtime_min))
    else:
        expected_run_time_sec = (tot_pts * (max(1, len(target_qubits)) / 2) / 48640000) * 632.3
        print("time required roughly: %s min" % (expected_run_time_sec / 60))

    for qubit_pair in qubit_pairs:
        print("qubit_control: %s" % qubit_pair.qubit_control)
        print("qubit_target: %s" % qubit_pair.qubit_target)
        if hasattr(qubit_pair.qubit_target, "id"):
            print(qubit_pair.qubit_target.id)

    print("Qubits: %s" % ", ".join(qubit_names(target_qubits)))
    print("sequences: %s" % xeb_config.seqs)
    print("depths: %s" % xeb_config.depths)
    print("shots: %s" % xeb_config.n_shots)
    print("CZ: %s" % xeb_config.two_qb_gate)
    print("CZ macro: %s" % node.parameters.cz_macro_name)
    print("control_readout_mode: %s, target_readout_mode: %s" % (xeb_config.control_readout_mode, xeb_config.target_readout_mode))
    print("apply_confusion_matrix: %s" % xeb_config.apply_confusion_matrix)
    print("XEB raw data saving: %s" % xeb_config.should_save_data)
    if xeb_config.should_save_data:
        print("saving data under: %s" % xeb_config.save_dir)
    else:
        print("Qualibrate node results will be saved with node.save()")


def print_xeb_result_summary(xeb_config: XEBConfig, result):
    print("sequences: %s" % xeb_config.seqs)
    print("shots: %s" % xeb_config.n_shots)
    print("Depth: %s" % xeb_config.depths)
    print("CZ: %s" % xeb_config.two_qb_gate)
    print("apply_confusion_matrix: %s" % xeb_config.apply_confusion_matrix)
    print(result.singularities)
    print("Singularities: {:.3f}%".format(len(result.singularities) / (xeb_config.seqs * len(xeb_config.depths)) * 100))
    print(result.linear_fidelities)
    if result.leak_counts:
        for qubit_name, probs in result.leakage_probs.items():
            print("Leakage to |f> for %s: mean=%.4f%%, max=%.4f%%" % (qubit_name, np.mean(probs) * 100, np.max(probs) * 100))


# %% {Initialize_QuAM}
node.machine = QuAM.load()

if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = node.machine.active_qubit_pairs
else:
    qubit_pairs = [node.machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

if len(qubit_pairs) == 0:
    raise ValueError("No qubit pairs selected")

if node.parameters.two_qubit_gate == "cz":
    for qubit_pair in qubit_pairs:
        if node.parameters.cz_macro_name not in qubit_pair.gates:
            available = ", ".join(qubit_pair.gates.keys())
            raise KeyError(
                f"{qubit_pair.name} has no gate {node.parameters.cz_macro_name!r}. Available gates: {available}"
            )

if len(qubit_pairs) == 1 and len(unique_qubits_from_pairs(qubit_pairs)) == 2:
    pair = qubit_pairs[0]
    for qubit, mode in (
        (pair.qubit_control, node.parameters.control_readout_mode),
        (pair.qubit_target, node.parameters.target_readout_mode),
    ):
        centers = getattr(qubit.resonator, "gef_centers", None)
        if mode == 3 and (centers is None or len(centers) < 3):
            raise ValueError(
                f"{qubit.name} readout mode is 3, but resonator.gef_centers is missing. "
                "Run 08d_IQ_Blobs_G_E_F for this qubit first."
            )
elif node.parameters.control_readout_mode == 3 or node.parameters.target_readout_mode == 3:
    print("GEF readout modes apply only to a single pair. This run uses 2-level readout.")

if (
    node.parameters.disjoint_processing
    and len(qubit_pairs) == 1
    and len(unique_qubits_from_pairs(qubit_pairs)) == 2
    and (node.parameters.control_readout_mode == 3 or node.parameters.target_readout_mode == 3)
):
    raise ValueError(
        "disjoint_processing=True is not supported with 3-level (G/E/F) readout. "
        "Set control_readout_mode and target_readout_mode to 2, or disable disjoint_processing."
    )

target_qubits = unique_qubits_from_pairs(qubit_pairs)
if node.parameters.reset_method == "active_gef":
    for qubit in target_qubits:
        centers = getattr(qubit.resonator, "gef_centers", None)
        if centers is None or len(centers) < 3:
            raise ValueError(
                f"{qubit.name} reset_method is active_gef, but resonator.gef_centers is missing. "
                "Run 08d_IQ_Blobs_G_E_F for this qubit first."
            )
if node.parameters.readout_qubits is None or node.parameters.readout_qubits == "":
    readout_qubits = node.machine.active_qubits
else:
    readout_qubits = [node.machine.qubits[q] for q in node.parameters.readout_qubits]

two_qubit_gate = QUAGate("cz", cz_gate) if node.parameters.two_qubit_gate == "cz" else None

reset_kwargs = {
    "max_tries": node.parameters.reset_max_tries,
}
if node.parameters.reset_method == "cooldown":
    reset_kwargs["cooldown_time"] = node.parameters.reset_cooldown_time_ns

xeb_config = XEBConfig(
    seqs=node.parameters.seqs,
    depths=node.parameters.depths,
    n_shots=node.parameters.n_shots,
    readout_qubits=readout_qubits,
    qubits=target_qubits,
    qubit_pairs=qubit_pairs,
    baseline_gate_name=node.parameters.baseline_gate_name,
    gate_set_choice="sw",  # "t" gate set fails to compile with the current quam version
    two_qb_gate=two_qubit_gate,
    save_dir="",
    should_save_data=False,
    data_folder_name=None,
    generate_new_data=node.parameters.generate_new_data,
    disjoint_processing=node.parameters.disjoint_processing,
    apply_confusion_matrix=node.parameters.apply_confusion_matrix,
    reset_method=node.parameters.reset_method,
    reset_kwargs=reset_kwargs,
    seed=node.parameters.seed,
    control_readout_mode=node.parameters.control_readout_mode,
    target_readout_mode=node.parameters.target_readout_mode,
)

node.results = {
    "initial_parameters": node.parameters.model_dump(),
    "xeb_config": xeb_config.as_dict(),
    "qubit_names": np.array([q.name for q in target_qubits]),
    "readout_qubit_names": np.array([q.name for q in readout_qubits]),
    "qubit_pair_names": np.array([qp.name for qp in qubit_pairs]),
}

print_xeb_setup(xeb_config, qubit_pairs, target_qubits)


# %% {Run_or_load}
# The analysis options (apply_confusion_matrix, disjoint_processing, fit_min_depth, ...) always come from the
# parameters set now, also when an old node is loaded, so a saved run can be re-analyzed with different options.
analysis_parameters = node.parameters.model_copy()

if node.parameters.load_data_id is not None:
    print("Loading from previous data and re-running the analysis")
    load_data_id = node.parameters.load_data_id
    node = node.load_from_id(load_data_id)
    if node is None:
        raise ValueError(f"Could not load node with id {load_data_id}")
    restore_load_data_id(node, load_data_id)
    machine = node.machine
    qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
    target_qubits = unique_qubits_from_pairs(qubit_pairs)
    readout_qubits = target_qubits
    # Probabilities, fidelities, purity, ... are recomputed from the saved raw counts/states.
    result = XEBResult.from_node_results(
        node.results,
        machine,
        apply_confusion_matrix=analysis_parameters.apply_confusion_matrix,
        disjoint_processing=analysis_parameters.disjoint_processing,
    )
else:
    print("Loading from new data")
    print("Qubits: %s" % ", ".join(qubit_names(target_qubits)))
    xeb = XEB(xeb_config, machine=node.machine)
    node.namespace = {"xeb": xeb}

    if node.parameters.execution_mode == "qiskit_simulation":
        job = xeb.simulate(backend=fake_backend)
    elif node.parameters.execution_mode == "hardware_simulation":
        job = xeb.run(
            simulate=True,
            simulation_config=SimulationConfig(duration=node.parameters.simulation_duration_ns),
        )
        job.plot_simulated_samples()
        node.results["figure_simulated_samples"] = plt.gcf()
    else:
        job = xeb.run(simulate=False)

    if job is None:
        raise RuntimeError(
            "No XEB job was started. generate_new_data is False, so hardware execution was skipped. "
            "Set generate_new_data=True, or set load_data_id to open a previous node."
        )

    result = job.result(disjoint_processing=node.parameters.disjoint_processing)


# %% {Analysis}
# Runs for both a new measurement and a loaded one; `result` comes from the block above.
xeb_config = result.xeb_config
has_two_qubit_gate = xeb_config.two_qb_gate is not None
print_xeb_result_summary(xeb_config, result)

node.results["log_fidelities"] = result.log_fidelities
node.results["singularities"] = result.singularities
node.results["outliers"] = result.outliers
node.results["linear_fidelities"] = serialize_linear_fidelities(result)
node.results["layer_fidelity_linear"] = result.get_layer_fidelity("linear")
node.results["layer_fidelity_log"] = result.get_layer_fidelity("log")

for key, value in result.data.items():
    add_result_entry(node.results, key, value)

for i, fig in enumerate(result.plot_state_heatmap()):
    node.results[f"figure_state_heatmap_{i}"] = fig

for i, fig in enumerate(result.plot_records()):
    node.results[f"figure_records_{i}"] = fig

for i, fig in enumerate(result.plot_fidelities(fit_linear=False, fit_log_entropy=True, separate_plots=True)):
    node.results[f"figure_fidelity_log_{i}"] = fig

for i, fig in enumerate(result.plot_fidelities(fit_linear=True, fit_log_entropy=False, separate_plots=True)):
    node.results[f"figure_fidelity_linear_{i}"] = fig

# %% {Leakage_analysis}
# Only meaningful when control_readout_mode/target_readout_mode == 3 for at least one qubit.
if result.leak_counts:
    for qubit_name, probs in result.leakage_probs.items():
        node.results[f"leakage_probs_{qubit_name}"] = probs

    leakage_fig = result.plot_leakage()
    if leakage_fig is not None:
        node.results["figure_leakage_vs_depth"] = leakage_fig

    for i, fig in enumerate(result.plot_leakage_heatmap()):
        node.results[f"figure_leakage_heatmap_{i}"] = fig

# %% {Purity}
# Speckle purity decays with incoherent error. Linear XEB decays with the total error,
# so their difference is the coherent error (e.g. a wrong CZ conditional phase).
if not xeb_config.disjoint_processing:
    try:
        comparison, fig_purity = result.coherence_limit_comparison(
            min_depth=analysis_parameters.fit_min_depth,
            correct_shot_noise=analysis_parameters.correct_purity_shot_noise,
        )
        print(
            "Purity: F_total = %.5f +/- %.1e, f_cycle = %.5f +/- %.1e"
            % (
                comparison["F_total"],
                comparison["F_total_stderr"],
                comparison["f_cycle"],
                comparison["f_cycle_stderr"],
            )
        )
        print(
            "  error per cycle: total = %.2e, incoherent = %.2e, coherent = %+.2e +/- %.1e"
            % (
                comparison["total_error_per_cycle"],
                comparison["incoherent_error_per_cycle"],
                comparison["coherent_error_per_cycle"],
                comparison["coherent_error_stderr"],
            )
        )
        node.results["purity_avg_arute"] = comparison["purity"]
        node.results["F_total"] = comparison["F_total"]
        node.results["F_total_stderr"] = comparison["F_total_stderr"]
        node.results["f_cycle"] = comparison["f_cycle"]
        node.results["f_cycle_stderr"] = comparison["f_cycle_stderr"]
        node.results["f_cycle_minus_F_total"] = comparison["f_cycle_minus_F_total"]
        node.results["total_error_per_cycle"] = comparison["total_error_per_cycle"]
        node.results["incoherent_error_per_cycle"] = comparison["incoherent_error_per_cycle"]
        node.results["incoherent_error_stderr"] = comparison["incoherent_error_stderr"]
        node.results["coherent_error_per_cycle"] = comparison["coherent_error_per_cycle"]
        node.results["coherent_error_stderr"] = comparison["coherent_error_stderr"]
        node.results["purity_fit_depths"] = comparison["fit_depths"]
        node.results["figure_coherence_limit"] = fig_purity
    except Exception as e:
        print(f"Purity analysis failed: {e}")

# %% {2Q_unitary_estimation}
# Fits an effective (theta_iswap, phi_cphase, phi_rz1, phi_rz2) 2-qubit unitary from the measured
# data via Nelder-Mead. Ideal CZ corresponds to (0, pi, 0, 0); a significant deviation indicates a
# coherent miscalibration of the CZ gate on top of (separate from) incoherent error sources.
if analysis_parameters.estimate_2q_unitary and has_two_qubit_gate and not xeb_config.disjoint_processing:
    try:
        unitary_fit = result.estimate_2q_unitary(
            max_depth=analysis_parameters.unitary_fit_max_depth,
            min_fit_depth=analysis_parameters.fit_min_depth,
        )
        print("2Q unitary fit (fSim + local Z, ideal CZ = theta=0, phi=180, rz1=rz2=0):")
        print(
            "  theta_iswap = %.2f deg, phi_CZ = %.2f deg, rz1 = %.2f deg, rz2 = %.2f deg"
            % (
                unitary_fit["theta_iswap_deg"],
                np.degrees(unitary_fit["phi_cphase"]),
                unitary_fit["rz1_deg"],
                unitary_fit["rz2_deg"],
            )
        )
        print(
            "  average cross-entropy loss: nominal CZ = %.4f, fitted unitary = %.4f"
            % (unitary_fit["loss_nominal"], unitary_fit["loss_fitted"])
        )
        print(
            "  layer fidelity: nominal CZ = %.5f, fitted unitary = %.5f, removable coherent error = %+.2e"
            % (
                unitary_fit["layer_fid_nominal"],
                unitary_fit["layer_fid_fitted"],
                unitary_fit["coherent_error_unitary_fit"],
            )
        )
        print(
            "  fit converged: %s, nominal simulation consistency error: %.1e"
            % (unitary_fit["success"], unitary_fit["nominal_consistency_error"])
        )
        node.results["unitary_fit_layer_fid_nominal"] = unitary_fit["layer_fid_nominal"]
        node.results["unitary_fit_layer_fid_fitted"] = unitary_fit["layer_fid_fitted"]
        node.results["unitary_fit_coherent_error"] = unitary_fit["coherent_error_unitary_fit"]
        node.results["unitary_fit_success"] = unitary_fit["success"]
        node.results["unitary_fit_nominal_consistency_error"] = unitary_fit["nominal_consistency_error"]
        node.results["unitary_fit_theta_iswap_deg"] = unitary_fit["theta_iswap_deg"]
        node.results["unitary_fit_phi_cz_deg"] = float(np.degrees(unitary_fit["phi_cphase"]))
        node.results["unitary_fit_cphase_error_deg"] = unitary_fit["cphase_error_deg"]
        node.results["unitary_fit_rz1_deg"] = unitary_fit["rz1_deg"]
        node.results["unitary_fit_rz2_deg"] = unitary_fit["rz2_deg"]
        node.results["unitary_fit_loss_nominal"] = unitary_fit["loss_nominal"]
        node.results["unitary_fit_loss_fitted"] = unitary_fit["loss_fitted"]
        node.results["figure_unitary_fit"] = result.plot_fidelity_with_unitary_fit(unitary_fit)
    except Exception as e:
        print(f"2Q unitary estimation failed: {e}")

# %% {Save_results}
# A loaded run is only re-analyzed on screen; a new run is saved.
if analysis_parameters.load_data_id is None:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    # Also rewrites configuration/quam_state/.../state.json from the in-memory QuAM.
    # Calibration fields are left unchanged; the file itself is still saved again.
    node.save()

# %%
