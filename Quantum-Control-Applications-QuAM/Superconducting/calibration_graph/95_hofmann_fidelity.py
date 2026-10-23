# %%
"""
Hofmann Fidelity of the CZ Gate (complementary classical fidelities, arXiv:quant-ph/0409083)

Measures an existing, calibrated CZ gate of a qubit pair with eight product-state inputs that form two
mutually unbiased sets, and reports the two classical fidelities F_A and F_B together with the bounds they
give on the process fidelity of the gate with respect to the ideal CZ = diag(1, 1, 1, -1):

    F_A + F_B - 1  <=  F_pro  <=  min(F_A, F_B)

Sequence (every shot loops over the 8 inputs, so slow drift is shared equally by all of them):
1. Reset both qubits (active or thermal reset) and, optionally, reset both qubit frames.
2. Prepare the input state with single-qubit pulses (|0>: none, |1>: x180, |+>: y90, |->: -y90).
3. Apply qp.gates[operation] exactly once.
4. Rotate into the measurement basis (Z: none, X: -y90) and read out both qubits.

    set A (measure X on the control, Z on the target):  |+0>  |-0>  |+1>  |-1>
    set B (measure Z on the control, X on the target):  |0+>  |0->  |1+>  |1->

For an ideal CZ each input ends in exactly one computational outcome of its measurement basis, so the fidelity
of an input is the fraction of shots that land on that outcome. F_A and F_B are the equal-weight means of
set A and set B. The score C_ML = (F_A + F_B) / 2 is the cost that the DGX retuning loop maximises
(DGXQ_AS/applications/client/qm_code/Hofmann_fidelity), so this node measures the same quantity offline.

Analysis (see calibration_utils/hofmann_fidelity/analysis.py for all conventions):
1. Raw single shots -> joint-outcome histograms (8 inputs x 4 outcomes) -> F_k with binomial standard errors.
   The raw values are the primary result.
2. Optionally the same quantities after readout-error correction with qp.confusion (34_2Q_confusion_matrix),
   reported next to the raw ones and never mixed with them.
3. Optionally (measure_leakage) GEF readout: shots with a qubit in |f> are counted as failures, not discarded,
   and the leakage probability is reported separately.

Important limitations:
- F_lower and F_upper are bounds on F_pro, they are not F_pro. The bounds assume that the inputs are prepared and
  the outcomes measured as specified: state-preparation and readout errors lower the measured fidelities and
  therefore also the lower bound. The binomial errors are statistical uncertainties of the measured fractions
  and are not part of the bounds. Use 2Q randomized benchmarking (70b) or process tomography (90) to
  confirm the gate fidelity independently.
- F_lower is reported unclipped (it can be negative for a very poor gate).
- Every shot resets both qubit frames (reset_frame_each_shot) because CZGate.execute accumulates
  frame_rotation_2pi on every call. Switch it off only to reproduce a loop that does not reset the frames.

Prerequisites:
- Calibrated single-qubit gates (x180, y90, -y90) and readout of both qubits.
- The calibrated CZ gate qp.gates[operation] (including its single-qubit phase corrections).
- For the readout-corrected values: qp.confusion (34_2Q_confusion_matrix).
- For measure_leakage=True: calibrated GEF readout (08d_IQ_Blobs_G_E_F) and EF_x180 pulses on both qubits.

Outcomes:
- F_A, F_B, F_lower, F_upper, C_ML with binomial errors, the eight individual fidelities and the outcome histograms.
- Characterization only: the QuAM state is only touched if save_to_state is set, and then only qp.extras["Hofmann"].
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, active_reset_gef, readout_state, readout_state_gef
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
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
from calibration_utils.hofmann_fidelity import (
    CONVENTIONS,
    HOFMANN_INPUTS,
    INPUT_LABELS,
    MEAS_PULSES,
    OUTCOME_LABELS,
    PREP_PULSES,
    analyse_shots,
    plot_input_fidelities,
    plot_outcome_probabilities,
    result_to_flat_dict,
)

# %% {Node_parameters}
qubit_pair_indexes = [4]  # The indexes of the qubit pairs to characterize
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s"%(i,i+1) for i in qubit_pair_indexes]
    num_shots: int = 1000
    """Shots for each of the eight inputs; the inputs are interleaved in every shot loop."""
    operation: str = "Cz"
    """Name of the CZ gate in qp.gates to characterize."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = 'active'
    reset_frame_each_shot: bool = True
    """Reset the frames of both qubits before every shot (CZGate.execute accumulates frame_rotation_2pi)."""
    readout_mitigation: Literal["none", "pair_confusion"] = "pair_confusion"
    """pair_confusion: also report values corrected with qp.confusion (34_2Q_confusion_matrix); the raw values stay the primary result."""
    measure_leakage: bool = False
    """Use GEF readout (and GEF active reset) on both qubits; shots with a qubit in |f> count as failures."""
    save_to_state: bool = False
    """Propose the raw F_A, F_B, F_lower, F_upper, C_ML as qp.extras["Hofmann"] (state update, off by default)."""
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None


node = QualibrationNode(
    name="95_hofmann_fidelity", parameters=Parameters()
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

# %% {QUA_program}
n_shots = node.parameters.num_shots  # The number of shots per input
n_inputs = len(HOFMANN_INPUTS)

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
measure_leakage = node.parameters.measure_leakage

with program() as hofmann_fidelity:
    n = declare(int)
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]

    for i, qp in enumerate(qubit_pairs):
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point, qp.qubit_control)

        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st)
            # The eight inputs are played one after the other inside every shot loop (interleaved in time)
            for entry in HOFMANN_INPUTS:
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
                if node.parameters.reset_frame_each_shot:
                    # CZGate.execute adds frame_rotation_2pi on every call: start each shot from a clean frame
                    reset_frame(qp.qubit_control.xy.name, qp.qubit_target.xy.name)
                qp.align()
                # input state
                for qubit, letter in zip((qp.qubit_control, qp.qubit_target), entry.prep):
                    if PREP_PULSES[letter] is not None:
                        qubit.xy.play(PREP_PULSES[letter])
                qp.align()
                # gate under test, applied exactly once
                qp.gates[operation_name].execute()
                qp.align()
                # rotate into the measurement basis of the set
                for qubit, letter in zip((qp.qubit_control, qp.qubit_target), entry.measure):
                    if MEAS_PULSES[letter] is not None:
                        qubit.xy.play(MEAS_PULSES[letter])
                qp.align()
                # readout
                if measure_leakage:
                    readout_state_gef(qp.qubit_control, state_control[i])
                    readout_state_gef(qp.qubit_target, state_target[i])
                else:
                    readout_state(qp.qubit_control, state_control[i])
                    readout_state(qp.qubit_target, state_target[i])
                save(state_control[i], state_st_control[i])
                save(state_target[i], state_st_target[i])
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(n_inputs).buffer(n_shots).save(f"state_control{i + 1}")
            state_st_target[i].buffer(n_inputs).buffer(n_shots).save(f"state_target{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, hofmann_fidelity, simulation_config)
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
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(hofmann_fidelity)

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
            {"input": list(INPUT_LABELS), "N": np.linspace(1, n_shots, n_shots)},
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
    operation_name = node.parameters.operation
    mitigation = node.parameters.readout_mitigation
    if measure_leakage and mitigation != "none":
        warnings.warn("measure_leakage=True: the 2-state confusion matrix cannot correct GEF readout, only raw values are reported.")

    analyses = {}
    for qp in qubit_pairs:
        shots_control = ds.state_control.sel(qubit=qp.name).transpose("N", "input").values
        shots_target = ds.state_target.sel(qubit=qp.name).transpose("N", "input").values

        confusion = None
        if mitigation == "pair_confusion" and not measure_leakage:
            if getattr(qp, "confusion", None) is None:
                warnings.warn(f"{qp.name}.confusion is empty (run 34_2Q_confusion_matrix): only raw values are reported.")
            else:
                confusion = np.array(qp.confusion, dtype=float)

        analysis = analyse_shots(shots_control, shots_target, confusion=confusion)
        analyses[qp.name] = analysis
        raw = analysis["raw"]

        print(f"\n===== {qp.name}: Hofmann fidelity of gates['{operation_name}'] "
              f"({analysis['shots_per_state']} shots per input, raw) =====")
        print(f"F_A = {raw['F_A']:.4f} ± {raw['sigma_F_A']:.4f}    F_B = {raw['F_B']:.4f} ± {raw['sigma_F_B']:.4f}")
        print(f"{raw['F_lower']:.4f} (± {raw['sigma_F_lower']:.4f}, lower, unclipped) <= F_pro <= "
              f"{raw['F_upper']:.4f} (± {raw['sigma_F_upper']:.4f}, upper)")
        print(f"C_ML = (F_A + F_B) / 2 = {raw['C_ML']:.4f} ± {raw['sigma_C_ML']:.4f}")
        print("per input: " + "  ".join(f"{label}={f:.3f}" for label, f in zip(INPUT_LABELS, raw["F"])))
        if "corrected" in analysis:
            corrected = analysis["corrected"]
            print(f"readout-corrected: F_A = {corrected['F_A']:.4f}, F_B = {corrected['F_B']:.4f}, "
                  f"{corrected['F_lower']:.4f} <= F_pro <= {corrected['F_upper']:.4f}, C_ML = {corrected['C_ML']:.4f}")
        if measure_leakage:
            print(f"leakage (|f> on either qubit, counted as failure): {analysis['leakage_probability']:.4f}")

        node.results[f"{qp.name}_F_A"] = raw["F_A"]
        node.results[f"{qp.name}_F_B"] = raw["F_B"]
        node.results[f"{qp.name}_F_lower"] = raw["F_lower"]
        node.results[f"{qp.name}_F_upper"] = raw["F_upper"]
        node.results[f"{qp.name}_C_ML"] = raw["C_ML"]

    # Everything in one dataset (and plain floats) so that it can be reloaded and compared offline
    ds_hofmann = xr.Dataset(
        {
            "histogram": (["qubit", "input", "outcome"], np.array([analyses[qp.name]["histograms"] for qp in qubit_pairs])),
            "leaked": (["qubit", "input"], np.array([analyses[qp.name]["leaked"] for qp in qubit_pairs])),
            "F": (["qubit", "input"], np.array([analyses[qp.name]["raw"]["F"] for qp in qubit_pairs])),
            "sigma": (["qubit", "input"], np.array([analyses[qp.name]["raw"]["sigma"] for qp in qubit_pairs])),
        },
        coords={
            "qubit": [qp.name for qp in qubit_pairs],
            "input": list(INPUT_LABELS),
            "outcome": list(OUTCOME_LABELS),
        },
    )
    if all("corrected" in analyses[qp.name] for qp in qubit_pairs):
        ds_hofmann["F_corrected"] = (["qubit", "input"], np.array([analyses[qp.name]["corrected"]["F"] for qp in qubit_pairs]))
        ds_hofmann["sigma_corrected"] = (["qubit", "input"], np.array([analyses[qp.name]["corrected"]["sigma"] for qp in qubit_pairs]))
    node.results["ds_hofmann"] = ds_hofmann
    node.results["hofmann"] = {
        qp.name: {
            **result_to_flat_dict(analyses[qp.name]["raw"]),
            **(result_to_flat_dict(analyses[qp.name]["corrected"], "corrected_") if "corrected" in analyses[qp.name] else {}),
            "shots_per_state": analyses[qp.name]["shots_per_state"],
            "leakage_probability": analyses[qp.name]["leakage_probability"],
        }
        for qp in qubit_pairs
    }
    node.results["conventions"] = CONVENTIONS

# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        plot_input_fidelities(ax, analyses[qubit_pair["qubit"]], qubit_pair["qubit"])
    grid.fig.suptitle(f"Hofmann fidelities of gates['{operation_name}']")
    plt.tight_layout()
    plt.show()
    node.results["figure_fidelities"] = grid.fig

    for qp in qubit_pairs:
        fig = plot_outcome_probabilities(analyses[qp.name], f"{qp.name}: outcome probabilities of the eight inputs")
        plt.show()
        node.results[f"{qp.name}_figure_outcomes"] = fig

# %% {Update_state}
if not node.parameters.simulate and node.parameters.load_data_id is None and node.parameters.save_to_state:
    with node.record_state_updates():
        for qp in qubit_pairs:
            raw = analyses[qp.name]["raw"]
            qp.extras["Hofmann"] = {key: float(raw[key]) for key in ("F_A", "F_B", "F_lower", "F_upper", "C_ML")}

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()

# %%
