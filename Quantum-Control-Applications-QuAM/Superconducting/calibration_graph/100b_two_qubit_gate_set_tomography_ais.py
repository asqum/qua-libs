"""
        TWO-QUBIT GATE SET TOMOGRAPHY (Advance Input Stream)
2Q variant of 100a_Gate_Set_Tomography_AIS using the pyGSTi smq2Q_XYICPHASE model pack
(falling back to smq2Q_XYCPHASE), as in CQT_2Q_GST.ipynb.

Native gates: I, x90, y90 on both qubits, plus the calibrated CZ gate of the pair.
Every pyGSTi circuit is tokenized per layer and streamed to the OPX with
advance_input_stream, so the OPX1000 static gate-table limit (~16000 ints) is never hit.
Only max_germs_depth ints live on the OPX at compile time instead of
total_germs_num * max_germs_depth.

Prerequisites:
    - Calibrated readout with state discrimination on both qubits.
    - Calibrated single-qubit gates (x90, y90) on both qubits.
    - Calibrated CZ gate on the selected qubit pair.
    - Calibrated flux bias points for the qubit pair.

Note:
    Two-qubit GST circuit counts grow very quickly. The default max length is 4
    (max_circuit_depth_in_power = 2). Run one qubit pair at a time.

State update:
    - None (analysis-only characterization).
"""

# %% {Imports}
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode

from calibration_utils.gate_set_tomography import (
    GERM_TOKENS_STREAM_NAME,
    analyse_gst_data_2q,
    build_raw_dataset_2q,
    log_gst_design_summary,
    play_tokenized_gst_circuits_2q,
    setup_gst_experiment_2q,
    start_push_gst_germs_in_background,
    write_gst_html_report,
)
from quam_libs.components import QuAM
from quam_libs.lib.save_utils import restore_load_data_id, resolve_qubit_pairs_from_node
from quam_libs.macros import active_reset, qua_declaration, readout_state


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q3_q4"]
    max_circuit_depth_in_power: int = 2
    """Maximum circuit length as a power of two: lengths are 2**0 .. 2**N. Default is 2 (max length 4).
    2Q GST circuit counts grow quickly; keep this small unless you know the runtime."""
    num_runs: int = 100
    """Number of repetitions per GST circuit. Default is 100."""
    use_fiducial_pair_reduction: bool = True
    """Use pyGSTi fiducial-pair reduction when the model pack supports it. Default is True."""
    operation: Literal["Cz", "Cz_unipolar", "Cz_flattop", "Cz_bipolar"] = "Cz"
    """Name of the CZ gate on the qubit pair. Default is 'Cz_unipolar'."""
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 5000
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False


node = QualibrationNode[Parameters, QuAM](
    name="100b_two_qubit_gate_set_tomography_ais", parameters=Parameters()
)

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)
machine = QuAM.load()
node.machine = machine
config = machine.generate_config()
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

if len(qubit_pairs) != 1:
    raise ValueError(
        "2Q GST currently supports exactly one qubit pair per run. "
        f"Got {len(qubit_pairs)} pairs: {[qp.name for qp in qubit_pairs]}."
    )
node.namespace["qubit_pairs"] = qubit_pairs

qp = qubit_pairs[0]
operation_name = node.parameters.operation
if operation_name not in qp.gates:
    raise ValueError(
        f"Qubit pair {qp.name!r} has no gate {operation_name!r}. "
        f"Available gates: {sorted(qp.gates.keys())}"
    )

# %% {QUA_program_parameters}
n_runs = node.parameters.num_runs
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active

# %% {Setup the GST model}
design = setup_gst_experiment_2q(
    node.parameters.max_circuit_depth_in_power,
    use_fiducial_pair_reduction=node.parameters.use_fiducial_pair_reduction,
)
node.namespace["gst_design"] = design
log_gst_design_summary(design, n_runs)

tokenized_germs = design.all_germs_to_qua_tokenized_labels
max_germs_depth = design.max_germs_depth
total_germs_num = design.total_germs_num

# %% {QUA_program}
if node.parameters.simulate:
    # Simulation does not support input streams reliably; use a static gate table.
    with program() as GST_2Q:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=2)
        state_control = declare(int)
        state_target = declare(int)
        state = declare(int)
        state_st = declare_stream()

        single_germ_order = declare(int)
        native_gate_order = declare(int)
        tokenized_germs_list = declare(int, value=np.array(tokenized_germs).flatten())
        single_germ_list = declare(int, size=max_germs_depth)
        germ_idx = declare(int)

        machine.set_all_fluxes(flux_point, qp)

        with for_(germ_idx, 0, germ_idx < total_germs_num, germ_idx + 1):
            save(germ_idx, n_st)
            assign(single_germ_order, germ_idx * max_germs_depth)
            with for_(
                native_gate_order, 0, native_gate_order < max_germs_depth, native_gate_order + 1
            ):
                assign(
                    single_germ_list[native_gate_order],
                    tokenized_germs_list[single_germ_order + native_gate_order],
                )
            with for_(n, 0, n < n_runs, n + 1):
                qp.align()
                play_tokenized_gst_circuits_2q(
                    single_germ_list,
                    depth=max_germs_depth,
                    qubit_pair=qp,
                    cz_operation=operation_name,
                )
                qp.align()
                readout_state(qp.qubit_control, state_control)
                readout_state(qp.qubit_target, state_target)
                assign(state, state_control * 2 + state_target)
                save(state, state_st)

        if not node.parameters.multiplexed:
            align()

        with stream_processing():
            n_st.save("n")
            state_st.buffer(n_runs).buffer(total_germs_num).save("state2q")
else:
    with program() as GST_2Q:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=2)
        state_control = declare(int)
        state_target = declare(int)
        state = declare(int)
        state_st = declare_stream()

        germ_idx = declare(int)
        germ_tokens_is = declare_input_stream(
            int, name=GERM_TOKENS_STREAM_NAME, size=max_germs_depth
        )

        machine.set_all_fluxes(flux_point, qp)

        with for_(germ_idx, 0, germ_idx < total_germs_num, germ_idx + 1):
            save(germ_idx, n_st)
            advance_input_stream(germ_tokens_is)
            with for_(n, 0, n < n_runs, n + 1):
                if reset_type == "active":
                    active_reset(qp.qubit_control)
                    active_reset(qp.qubit_target)
                else:
                    wait(qp.qubit_control.thermalization_time * u.ns)
                qp.align()
                play_tokenized_gst_circuits_2q(
                    germ_tokens_is,
                    depth=max_germs_depth,
                    qubit_pair=qp,
                    cz_operation=operation_name,
                )
                qp.align()
                readout_state(qp.qubit_control, state_control)
                readout_state(qp.qubit_target, state_target)
                assign(state, state_control * 2 + state_target)
                save(state, state_st)

        if not node.parameters.multiplexed:
            align()

        with stream_processing():
            n_st.save("n")
            state_st.buffer(n_runs).buffer(total_germs_num).save("state2q")

# %% {Simulate_or_execute}
job = None

if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)
    job = qmm.simulate(config, GST_2Q, simulation_config)
    samples = job.get_simulated_samples()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples, plot=True, save_path="./")
    node.results = {"figure": plt.gcf()}
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(GST_2Q)
        push_thread = start_push_gst_germs_in_background(job, tokenized_germs)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            germ_idx = results.fetch_all()[0]
            progress_counter(germ_idx, total_germs_num, start_time=results.start_time)
        push_thread.join()
        job.result_handles.wait_for_all_values()
    node.log(job.execution_report())

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = build_raw_dataset_2q(job.result_handles, design, n_runs, qp.name)
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds_raw"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
        node.namespace["qubit_pairs"] = qubit_pairs
        node.namespace["gst_design"] = design
        qp = qubit_pairs[0]

    node.results = {"ds_raw": ds}

# %% {Data_analysis}
if not node.parameters.simulate:
    gst_result_objects = analyse_gst_data_2q(node, node.results["ds_raw"], design, qubit_pairs)
    node.namespace["gst_result_objects"] = gst_result_objects
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}

# %% {Save_results}
if not node.parameters.simulate:
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()

    for qp in qubit_pairs:
        gst_results = node.namespace["gst_result_objects"].get(qp.name)
        if gst_results is None:
            continue
        report_dirname, report_main = write_gst_html_report(
            gst_results, qp.name, node.snapshot_idx
        )
        node.results["gst_report_dir"] = report_dirname
        node.results["gst_report_main"] = report_main
    node.save()

# %%
