# %% {Imports}
from dataclasses import asdict
from typing import Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.units import unit
from qualibrate import QualibrationNode, NodeParameters

from calibration_utils.cz_cafe import (
    NUM_STATES,
    build_circuit_angles,
    fit_raw_data,
    log_fitted_results,
    plot_leakage,
    plot_raw_data_with_fit,
    primary_variant,
    process_raw_dataset,
    reference_gates_per_pair,
)
from calibration_utils.cz_cafe.qua_utils import play_layer_cases
from quam_libs.components import QuAM
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubit_pairs_from_node,
)
from quam_libs.macros import active_reset, active_reset_gef, readout_state, readout_state_gef

# %% {Initialisation}
description = """
        CZ CONTEXT AWARE FIDELITY ESTIMATION (CAFE)
Measures the average gate fidelity of the CZ gate and splits its error into coherent and
incoherent parts, following D. M. Debroy et al., arXiv:2303.17565.

Protocol
--------
For each of the 16 states of a two-qubit SIC (a 2-design) and each repetition count n:
1. Prepare the state from |00> with one CZ and single-qubit rotations (Appendix A).
2. Apply the cycle n times. A cycle is qp.gates[operation]; with the 'decaf' variant it is
   followed by X on both qubits, which echoes out single-qubit phase errors and low-frequency
   Z noise.
3. Undo the state the reference cycle would have produced, again with one CZ, and measure
   P(|00>).
Single-qubit rotations use only y90 pulses and virtual Z, which makes P(|00>) independent of
the sign conventions of both.

Analysis
--------
P(|00>) averaged over the 16 states is the average gate fidelity of n cycles. Coherent errors
grow as n^2 and incoherent errors as n, so fitting Eq. (B9) of the paper versus n separates
them. The node reports, per pair and variant:
- F: cycle fidelity against the reference unitary, corrected for SPAM.
- incoherent error: the infidelity left when coherent errors are removed.
- coherent error: the infidelity left when incoherent errors are removed.
A quadratic fit over the shallow depths (Eq. 8) is logged as a cross-check.

With reference_unitary='characterized', the undo step uses an fSim unitary built from the
per-pair angles [Δθ, Δγ, Δφ]; the coherent error then measures how well that model predicts
the gate.

Prerequisites:
- Calibrated single-qubit gates (x180, y90) on both qubits.
- Calibrated CZ gate qp.gates[operation], including its single-qubit phase corrections
  (32a/32x/32y, 33a/33b).
- Calibrated readout with state discrimination (08d_IQ_Blobs_G_E_F if record_leakage=True).

State update:
- qp.gates[operation].extras["CAFE"] (and ["DECAF"] if measured) =
  {"fidelity", "incoherent_error", "coherent_error", "spam"}. With a characterized reference
  the keys are "CAFE_characterized_reference" / "DECAF_characterized_reference".
"""

qubit_pair_indexes = [4]  # The indexes of the qubit pairs to characterize


class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s" % (i, i + 1) for i in qubit_pair_indexes]
    num_shots: int = 400
    """Number of shots per circuit (one circuit per state, depth and variant). Default is 100."""
    operation: str = "Cz"
    """Name of the CZ gate in qp.gates to characterize. Default is 'Cz'."""
    max_depth: int = 40
    """Largest number of cycle repetitions n. Default is 16."""
    depth_step: int = 1
    """Step between repetition counts, starting at n = 0. The paper uses even n only (step 2). Default is 2."""
    variants: List[Literal["cafe", "decaf"]] = ["decaf"]
    """Cycle variants to measure. 'cafe' repeats the CZ; 'decaf' adds X on both qubits after each CZ,
    which echoes out single-qubit phase errors and low-frequency Z noise. Default is ['cafe']."""
    reference_unitary: Literal["ideal", "characterized"] = "ideal"
    """Reference unitary used to undo the state: the ideal CZ, or a characterized CZ built from
    ``characterized_angles``. Default is 'ideal'."""
    characterized_angles: Dict[str, List[float]] = {}
    """Per qubit pair name, [Δθ, Δγ, Δφ] in radians: swap angle, single-qubit phase and conditional-phase
    error of the characterized CZ (Eq. 2 of arXiv:2303.17565). Pairs not listed use the ideal CZ."""
    quadratic_fit_max_depth: int = 4
    """Largest n used in the quadratic cross-check fit (Eq. 8). Default is 4."""
    record_leakage: bool = False
    """If True, read out (and actively reset) with the GEF discriminator and record the |f> population
    of each qubit. Leakage is not used in the fit. Requires a calibrated GEF readout. Default is False."""
    flux_point_joint_or_independent: Literal["joint", "pairwise"] = "joint"
    reset_type: Literal["thermal", "active"] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None


node = QualibrationNode(name="91_cz_cafe", description=description, parameters=Parameters())
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), (
    "If simulate is True, load_data_id must be None, and vice versa."
)

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
if not node.parameters.variants:
    raise ValueError("At least one variant ('cafe' or 'decaf') must be selected.")

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {Circuit_compilation}
n_shots = node.parameters.num_shots
depths = list(range(0, node.parameters.max_depth + 1, node.parameters.depth_step))
variants = list(dict.fromkeys(node.parameters.variants))
record_leakage = node.parameters.record_leakage
flux_point = node.parameters.flux_point_joint_or_independent

# Precompute the single-qubit layer angles of every circuit, per pair (the undo step
# depends on the pair's reference unitary)
reference_gates = reference_gates_per_pair(
    [qp.name for qp in qubit_pairs], node.parameters.reference_unitary, node.parameters.characterized_angles
)
node.namespace["qubit_pairs"] = qubit_pairs
node.namespace["reference_gates"] = reference_gates
circuit_angles = {name: build_circuit_angles(variants, depths, gate) for name, gate in reference_gates.items()}
# Every angle is played as a compile-time constant, one switch case per circuit
preparation_layers = next(iter(circuit_angles.values())).preparation_2pi()
undo_tables = {name: angles.undo_table_2pi() for name, angles in circuit_angles.items()}

# %% {QUA_program}
with program() as cz_cafe:
    n = declare(int)
    n_st = declare_stream()
    state_idx = declare(int)
    depth_idx = declare(int)
    depth = declare(int)
    count = declare(int)
    undo_case = declare(int)
    flag = declare(bool)
    result = declare(int)
    depths_qua = declare(int, value=depths)
    undo_index_qua = [declare(int, value=undo_tables[qp.name][1].ravel().tolist()) for qp in qubit_pairs]
    state_c = [declare(int) for _ in range(num_qubit_pairs)]
    state_t = [declare(int) for _ in range(num_qubit_pairs)]
    p_return_st = [declare_stream() for _ in range(num_qubit_pairs)]
    if record_leakage:
        f_control_st = [declare_stream() for _ in range(num_qubit_pairs)]
        f_target_st = [declare_stream() for _ in range(num_qubit_pairs)]

    def save_flag(condition, stream):
        assign(flag, condition)
        assign(result, Cast.to_int(flag))
        save(result, stream)

    for i, qp in enumerate(qubit_pairs):
        undo_layers = undo_tables[qp.name][0]
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point, qp)

        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st)
            for iv, variant in enumerate(variants):
                with for_(depth_idx, 0, depth_idx < len(depths), depth_idx + 1):
                    assign(depth, depths_qua[depth_idx])
                    with for_(state_idx, 0, state_idx < NUM_STATES, state_idx + 1):
                        # Select the undo circuit before the sequence starts, so that no
                        # real-time computation happens between pulses
                        assign(undo_case, undo_index_qua[i][(iv * len(depths) + depth_idx) * NUM_STATES + state_idx])
                        # Reset the qubits and their frames
                        if not node.parameters.simulate:
                            if node.parameters.reset_type == "active":
                                if record_leakage:
                                    active_reset_gef(qp.qubit_control)
                                    active_reset_gef(qp.qubit_target)
                                else:
                                    active_reset(qp.qubit_control)
                                    active_reset(qp.qubit_target)
                            else:
                                wait(
                                    max(qp.qubit_control.thermalization_time, qp.qubit_target.thermalization_time)
                                    * u.ns
                                )
                        qp.align()
                        reset_frame(qp.qubit_control.xy.name, qp.qubit_target.xy.name)

                        # 1. Prepare the SIC state with one CZ
                        play_layer_cases(qp, state_idx, preparation_layers[:, 0])
                        qp.gates[operation_name].execute()
                        play_layer_cases(qp, state_idx, preparation_layers[:, 1])
                        # 2. Repeat the cycle
                        with for_(count, 0, count < depth, count + 1):
                            qp.gates[operation_name].execute()
                            if variant == "decaf":
                                qp.qubit_control.xy.play("x180")
                                qp.qubit_target.xy.play("x180")
                        # 3. Undo the state expected from the reference cycle
                        play_layer_cases(qp, undo_case, undo_layers[:, 0])
                        qp.gates[operation_name].execute()
                        play_layer_cases(qp, undo_case, undo_layers[:, 1])
                        qp.align()

                        # Measure both qubits and record whether they returned to |00>
                        if record_leakage:
                            readout_state_gef(qp.qubit_control, state_c[i])
                            readout_state_gef(qp.qubit_target, state_t[i])
                        else:
                            readout_state(qp.qubit_control, state_c[i])
                            readout_state(qp.qubit_target, state_t[i])
                        save_flag((state_c[i] == 0) & (state_t[i] == 0), p_return_st[i])
                        if record_leakage:
                            save_flag(state_c[i] == 2, f_control_st[i])
                            save_flag(state_t[i] == 2, f_target_st[i])
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            p_return_st[i].buffer(NUM_STATES).buffer(len(depths)).buffer(len(variants)).average().save(
                f"p_return{i + 1}"
            )
            if record_leakage:
                f_control_st[i].buffer(NUM_STATES).buffer(len(depths)).buffer(len(variants)).average().save(
                    f"f_control{i + 1}"
                )
                f_target_st[i].buffer(NUM_STATES).buffer(len(depths)).buffer(len(variants)).average().save(
                    f"f_target{i + 1}"
                )

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, cz_cafe, simulation_config)
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
        job = qm.execute(cz_cafe)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_shots, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Convert the results into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(
            job.result_handles,
            qubit_pairs,
            {"state": np.arange(NUM_STATES), "depth": depths, "variant": variants},
        )
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds_raw"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
        operation_name = node.parameters.operation
        node.namespace["qubit_pairs"] = qubit_pairs
        node.namespace["reference_gates"] = reference_gates_per_pair(
            [qp.name for qp in qubit_pairs], node.parameters.reference_unitary, node.parameters.characterized_angles
        )

    if "qubit" in ds.dims:
        ds = ds.rename({"qubit": "qubit_pair"})
    node.results = {"ds_raw": ds}

# %% {Data_analysis}
if not node.parameters.simulate:
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {
        qp_name: {variant: asdict(fr) for variant, fr in per_variant.items()}
        for qp_name, per_variant in fit_results.items()
    }
    log_fitted_results(fit_results, log_callable=node.log)
    # The node outcome follows the CAFE variant when it was measured
    main_variant = primary_variant(node.results["ds_fit"].variant.values.tolist())
    node.outcomes = {
        qp_name: ("successful" if per_variant[main_variant].success else "failed")
        for qp_name, per_variant in fit_results.items()
    }

# %% {Plotting}
if not node.parameters.simulate:
    qubit_pairs = node.namespace["qubit_pairs"]
    ds_fit = node.results["ds_fit"]

    fig = plot_raw_data_with_fit(ds_fit, qubit_pairs)
    plt.show()
    node.results["figure"] = fig

    if "f_control" in ds_fit.data_vars:
        fig_leakage = plot_leakage(ds_fit, qubit_pairs)
        plt.show()
        node.results["leakage_figure"] = fig_leakage

# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        suffix = "" if node.parameters.reference_unitary == "ideal" else "_characterized_reference"
        with node.record_state_updates():
            for qp in node.namespace["qubit_pairs"]:
                if node.outcomes[qp.name] == "failed":
                    print(f"Skipping state update for {qp.name}: fit flagged unsuccessful.")
                    continue
                for variant, fr in node.results["fit_results"][qp.name].items():
                    if not fr["success"]:
                        continue
                    qp.gates[operation_name].extras[f"{variant.upper()}{suffix}"] = {
                        "fidelity": fr["fidelity"],
                        "incoherent_error": fr["incoherent_error"],
                        "coherent_error": fr["coherent_error"],
                        "spam": fr["spam"],
                    }

# %% {Save_results}
if not node.parameters.simulate:
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()

# %%
