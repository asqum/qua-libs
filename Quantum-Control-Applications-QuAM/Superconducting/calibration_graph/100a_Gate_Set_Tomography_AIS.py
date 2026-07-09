"""
    Gate Set Tomography (Advance Input Stream)
    ==========================================
    Variant of 100_Gate_Set_Tomography that streams germ sequences to the OPX via
    ``advance_input_stream``, avoiding the OPX1000 static gate-table limit (~16000 ints).

    Germ sequences are pushed from Python with ``job.push_to_input_stream`` once per
    circuit while the QUA program repeats each circuit ``num_runs`` times before
    advancing to the next. Only ``max_germs_depth`` ints live on the OPX at compile
    time instead of ``total_germs_num * max_germs_depth``.
"""
# %% {Imports}
import re
import threading
from typing import Literal, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pygsti
import xarray as xr
from pygsti.modelpacks import smq1Q_XYI as std
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode

from quam_libs.components import QuAM, Transmon
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.macros import active_reset, qua_declaration, readout_state

OPX1000_GATE_TABLE_LIMIT = 16000
GERM_TOKENS_STREAM_NAME = "germ_tokens"

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ["q1"]
    max_circuit_depth_in_power: int = 9
    num_runs: int = 100
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 5000
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False

node = QualibrationNode(name="100a_Gate_Set_Tomography_AIS", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)
machine = QuAM.load()
node.machine = machine
config = machine.generate_config()
if node.parameters.load_data_id is None:
    qmm = machine.connect()

if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

# %% {QUA_program_parameters}
max_circuit_length = [
    2**i for i in range(max(node.parameters.max_circuit_depth_in_power, 0) + 1)
]
n_runs = node.parameters.num_runs
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active

# %% {Utility functions}
def parse_gst_circuit_string(circuit_str):
    """Parse a pyGSTi circuit string into a list of pulse labels."""
    clean_str = circuit_str.split()[0]
    clean_str = clean_str.replace("@(0)", "")
    clean_str = clean_str.replace("({})", "(I)")
    clean_str = clean_str.replace("{}", "(I)")
    clean_str = clean_str.replace("([])", "(I)")

    token_map = {
        "Gxpi2:0": "X",
        "Gypi2:0": "Y",
        "I": "I",
    }
    temp_str = clean_str
    for key, token in token_map.items():
        temp_str = temp_str.replace(key, token)

    while "^" in temp_str:
        def expand_match(match):
            content = match.group(1)
            power = int(match.group(2))
            return content * power

        temp_str = re.sub(r"\(([^)]+)\)\^(\d+)", expand_match, temp_str)

    temp_str = temp_str.replace("(", "").replace(")", "")

    final_map = {
        "X": "x90",
        "Y": "y90",
        "I": "I",
    }
    return [final_map[ch] for ch in temp_str if ch in final_map]


def tokenize_gst_circuits(gst_str):
    """Convert GST circuit strings into tokenized format for QUA execution."""
    gate_set_token = {
        "I": 0,
        "x90": 1,
        "y90": 2,
        "x180": 3,
        "y180": 4,
    }
    gate_set_length_list = [len(g) for g in gst_str]
    max_gate_set_length = max(gate_set_length_list)
    tokenized_circuits = []

    for germ in gst_str:
        g_len = len(germ)
        tokenized = [gate_set_token[g] for g in germ]
        if g_len < max_gate_set_length:
            tokenized.extend([-1] * (max_gate_set_length - g_len))
        tokenized_circuits.append(tokenized)

    return tokenized_circuits, gate_set_length_list


def play_tokenized_gst_circuits(tokenized_germ, depth, qubit: Transmon):
    i = declare(int)
    with for_(i, 0, i < depth, i + 1):
        with switch_(tokenized_germ[i], unsafe=True):
            with case_(0):
                qubit.xy.wait(4)
            with case_(1):
                qubit.xy.play("x90")
            with case_(2):
                qubit.xy.play("y90")
            with case_(3):
                qubit.xy.play("x180")
            with case_(4):
                qubit.xy.play("y180")
            with case_(-1):
                pass


def push_gst_germs_to_input_stream(job, tokenized_germs):
    """Push each germ sequence once; QUA repeats it num_runs times before advancing."""
    total_pushes = len(tokenized_germs)
    for push_count, tokens in enumerate(tokenized_germs, start=1):
        job.push_to_input_stream(GERM_TOKENS_STREAM_NAME, tokens)
        if push_count % 100 == 0 or push_count == total_pushes:
            print(f"Input stream: pushed {push_count}/{total_pushes} germ sequences")


def start_push_gst_germs_in_background(job, tokenized_germs):
    thread = threading.Thread(
        target=push_gst_germs_to_input_stream,
        args=(job, tokenized_germs),
        daemon=True,
    )
    thread.start()
    return thread


def fetch_gst_state_averaged(handles, qubit, total_germs_num, n_runs):
    """Fetch 2D results (germs x runs) and average over runs per circuit."""
    try:
        ds = fetch_results_as_xarray(
            handles,
            [qubit],
            {"runs": np.arange(n_runs), "germs": np.arange(total_germs_num)},
        )
        if "runs" in ds.dims and "germs" in ds.dims:
            return ds.mean(dim="runs")
    except ValueError:
        pass

    flat = np.array(handles.get("state1").fetch_all(), dtype=float).ravel()
    if flat.size == total_germs_num:
        return xr.Dataset(
            {"state": (("qubit", "germs"), flat.reshape(1, -1))},
            coords={"qubit": [qubit.name], "germs": np.arange(total_germs_num)},
        )
    if flat.size == n_runs:
        raise ValueError(
            f"Got {n_runs} values (one per run, not per circuit). The compiled QUA "
            "program likely still has runs as the outer loop. Re-run the QUA program "
            "cell so the loop order is: outer germs, inner runs."
        )
    raise ValueError(
        f"Expected shape ({total_germs_num}, {n_runs}) from "
        f"buffer({n_runs}).buffer({total_germs_num}), got {flat.size} values. "
        "Re-run the QUA program cell after code changes."
    )


# %% Setup the GST model
std_model = std.target_model()
exp_design = pygsti.protocols.StandardGSTDesign(
    std_model,
    std.prep_fiducials(),
    std.meas_fiducials(),
    std.germs(),
    max_circuit_length,
)

all_germs_from_gst_model = [s.str for s in exp_design.all_circuits_needing_data]
all_germs_to_qua_labels = [parse_gst_circuit_string(s) for s in all_germs_from_gst_model]
all_germs_to_qua_tokenized_labels, all_germs_depth = tokenize_gst_circuits(all_germs_to_qua_labels)
max_germs_depth = max(all_germs_depth)
total_germs_num = len(all_germs_to_qua_tokenized_labels)
static_gate_table_size = total_germs_num * max_germs_depth

print("=== GST experiment design summary ===")
print(f"max_circuit_length passed to pyGSTi: {max_circuit_length}")
print(f"Longest germ sequence (max_germs_depth): {max_germs_depth}")
print(f"Total number of gate lists (circuits/germs): {total_germs_num}")
print(
    f"Static gate-table size if compiled inline: "
    f"{total_germs_num} x {max_germs_depth} = {static_gate_table_size}"
)
print(f"OPX1000 gate-table limit: {OPX1000_GATE_TABLE_LIMIT}")
if static_gate_table_size > OPX1000_GATE_TABLE_LIMIT:
    print(
        f"Static table EXCEEDS limit by {static_gate_table_size - OPX1000_GATE_TABLE_LIMIT}; "
        "this node streams germs via advance_input_stream."
    )
else:
    print("Static table fits in OPX memory; AIS still used to avoid future depth scaling issues.")
print(f"Input stream pushes (one per circuit): {total_germs_num}")
print(f"Stream processing: buffer({n_runs}).buffer({total_germs_num})  [inner=runs, outer=germs]")
print(
    f"Total measurements: {n_runs} runs x {total_germs_num} germs = "
    f"{n_runs * total_germs_num}"
)
print("=====================================")

qubit = qubits[0]

# %% {QUA_program}
if node.parameters.simulate:
    # Simulation does not support input streams reliably; use static gate table.
    with program() as GST:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = [declare(int) for _ in range(1)]
        state_st = [declare_stream() for _ in range(1)]

        single_germ_order = declare(int)
        native_gate_order = declare(int)
        tokenized_germs_list = declare(
            int, value=np.array(all_germs_to_qua_tokenized_labels).flatten()
        )
        single_germ_list = declare(int, size=max_germs_depth)
        germ_idx = declare(int)

        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

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
                if reset_type == "active":
                    active_reset(qubit)
                elif reset_type == "thermal":
                    qubit.wait(qubit.thermalization_time * u.ns)
                play_tokenized_gst_circuits(
                    single_germ_list, depth=max_germs_depth, qubit=qubit
                )
                align()
                readout_state(qubit, state[0])
                save(state[0], state_st[0])

        if not node.parameters.multiplexed:
            align()

        with stream_processing():
            n_st.save("n")
            state_st[0].buffer(n_runs).buffer(total_germs_num).save("state1")
else:
    with program() as GST:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = [declare(int) for _ in range(1)]
        state_st = [declare_stream() for _ in range(1)]

        germ_idx = declare(int)
        germ_tokens_is = declare_input_stream(
            int, name=GERM_TOKENS_STREAM_NAME, size=max_germs_depth
        )

        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(germ_idx, 0, germ_idx < total_germs_num, germ_idx + 1):
            save(germ_idx, n_st)
            advance_input_stream(germ_tokens_is)
            with for_(n, 0, n < n_runs, n + 1):
                if reset_type == "active":
                    active_reset(qubit)
                elif reset_type == "thermal":
                    qubit.wait(qubit.thermalization_time * u.ns)
                play_tokenized_gst_circuits(
                    germ_tokens_is, depth=max_germs_depth, qubit=qubit
                )
                align()
                readout_state(qubit, state[0])
                save(state[0], state_st[0])

        if not node.parameters.multiplexed:
            align()

        with stream_processing():
            n_st.save("n")
            state_st[0].buffer(n_runs).buffer(total_germs_num).save("state1")

# %% {Simulate_or_execute}
job = None

if node.parameters.simulate:
    simulation_config = SimulationConfig(
        duration=node.parameters.simulation_duration_ns * 4
    )
    job = qmm.simulate(config, GST, simulation_config)
    samples = job.get_simulated_samples()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples, plot=True, save_path="./")
    node.results = {"figure": plt.gcf()}
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(GST)
        push_thread = start_push_gst_germs_in_background(
            job, all_germs_to_qua_tokenized_labels
        )
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            germ_idx = results.fetch_all()[0]
            progress_counter(germ_idx, total_germs_num, start_time=results.start_time)
        push_thread.join()
        job.result_handles.wait_for_all_values()

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = None
        for i in range(num_qubits):
            ds_ = fetch_gst_state_averaged(
                job.result_handles, qubits[i], total_germs_num, n_runs
            )
            ds = xr.concat([ds, ds_], dim="qubit") if ds is not None else ds_
        count1 = (ds.state.values * 100).astype(int)
        count0 = 100 - count1
        ds["count0"] = (("qubit", "germs"), count0)
        ds["count1"] = (("qubit", "germs"), count1)
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]

    # %% {Data_analysis}

    def transform_dataset_to_gst(ds):
        gst_ds = pygsti.data.DataSet(outcome_labels=["0", "1"])
        for i, crc in enumerate(exp_design.all_circuits_needing_data):
            gst_ds.add_count_dict(
                crc, {"0": ds.count0.values[0, i], "1": ds.count1.values[0, i]}
            )
        return gst_ds

    node.results = {"ds": ds, "figs": {}, "results": {}}
    node.results["ds"] = ds

    gst_ds = transform_dataset_to_gst(ds)
    gst_data = pygsti.protocols.ProtocolData(exp_design, gst_ds)
    gst_protocol = pygsti.protocols.StandardGST()
    gst_results = gst_protocol.run(gst_data, disable_checkpointing=True)

    node.results["results"][qubit.name] = {}
    node.results["results"][qubit.name]["gst_results"] = {
        "TP": {
            "rho0": {"density_mx": None, "fidelity": None},
            "meas_op": {
                "0": {"povm": None, "fidelity": None},
                "1": {"povm": None, "fidelity": None},
            },
            "gate_op": {
                "I": {"choi": None, "fidelity": None, "robustness": None},
                "x90": {"choi": None, "fidelity": None, "robustness": None},
                "y90": {"choi": None, "fidelity": None, "robustness": None},
            },
        },
        "CPTP": {
            "rho0": {"density_mx": None, "fidelity": None},
            "meas_op": {
                "0": {"povm": None, "fidelity": None},
                "1": {"povm": None, "fidelity": None},
            },
            "gate_op": {
                "I": {"choi": None, "fidelity": None, "robustness": None},
                "x90": {"choi": None, "fidelity": None, "robustness": None},
                "y90": {"choi": None, "fidelity": None, "robustness": None},
            },
        },
        "Ideal": {
            "rho0": {"density_mx": None, "fidelity": None},
            "meas_op": {
                "0": {"povm": None, "fidelity": None},
                "1": {"povm": None, "fidelity": None},
            },
            "gate_op": {
                "I": {"choi": None, "fidelity": None, "robustness": None},
                "x90": {"choi": None, "fidelity": None, "robustness": None},
                "y90": {"choi": None, "fidelity": None, "robustness": None},
            },
        },
    }
    estimate_keys = ["full TP", "CPTPLND", "Target"]
    native_gate_keys = [(), ("Gxpi2", 0), ("Gypi2", 0)]
    np.set_printoptions(suppress=True, precision=6)
    for i, cond in enumerate(node.results["results"][qubit.name]["gst_results"].keys()):
        est_model = gst_results.estimates[estimate_keys[i]].models["stdgaugeopt"]
        print(f"\n=== Under {cond} condiction ===")

        rho_vec = est_model.preps["rho0"]
        rho_est_mat = pygsti.tools.vec_to_stdmx(rho_vec, basis="pp")
        rho_std_mat = pygsti.tools.vec_to_stdmx(std_model.preps["rho0"], basis="pp")
        state_fidelity = pygsti.tools.fidelity(rho_est_mat, rho_std_mat)
        print(f"State preparation fidelity for '0' state is {state_fidelity:.6f}.")

        node.results["results"][qubit.name]["gst_results"][cond]["rho0"]["density_mx"] = rho_est_mat
        node.results["results"][qubit.name]["gst_results"][cond]["rho0"]["fidelity"] = state_fidelity

        povm_obj = est_model.povms["Mdefault"]
        print("State measurement fidelity for ", end="")
        for label, effect_vec in povm_obj.items():
            matrix_est_form = pygsti.tools.vec_to_stdmx(effect_vec, basis="pp")
            matri_std_form = pygsti.tools.vec_to_stdmx(
                std_model.povms["Mdefault"][str(label)], basis="pp"
            )
            meas_fidelity = pygsti.tools.fidelity(matrix_est_form, matri_std_form)
            print(f"'{label}' state is {meas_fidelity:.6f} ", end="")

            node.results["results"][qubit.name]["gst_results"][cond]["meas_op"][str(label)][
                "povm"
            ] = matrix_est_form
            node.results["results"][qubit.name]["gst_results"][cond]["meas_op"][str(label)][
                "fidelity"
            ] = meas_fidelity

        for j, gate in enumerate(
            node.results["results"][qubit.name]["gst_results"][cond]["gate_op"].keys()
        ):
            matrix_ptm = est_model.operations[native_gate_keys[j]].to_dense()
            choi = pygsti.tools.jamiolkowski.jamiolkowski_iso(
                matrix_ptm, op_mx_basis="pp", choi_mx_basis="std"
            )
            infidelity = pygsti.tools.entanglement_infidelity(
                matrix_ptm,
                std_model.operations[native_gate_keys[j]].to_dense(),
                "pp",
            )
            node.results["results"][qubit.name]["gst_results"][cond]["gate_op"][gate]["choi"] = choi
            node.results["results"][qubit.name]["gst_results"][cond]["gate_op"][gate][
                "fidelity"
            ] = 1 - infidelity

            print(f"\nGate Infidelity: {infidelity:.6f}")
            evals = np.linalg.eigvals(choi)
            print(f"Choi Eigenvalues: {np.round(evals.real, 6)}")

    # %% {Update_state}
    if node.parameters.reset_type_thermal_or_active == "active":
        for i, j in zip(machine.active_qubit_names, "abcde"):
            machine.qubits[i].xy.core = j
            machine.qubits[i].resonator.core = j

    # %% {Save_results}
    if not node.parameters.simulate:
        from pathlib import Path

        from qualibrate_config.resolvers import get_qualibrate_config, get_qualibrate_config_path
        from quam_libs.compat import get_node_dir_path

        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        gst_report_dirname = f"gst_report_{qubit.name}"
        node.results["gst_report_dir"] = gst_report_dirname
        node.results["gst_report_main"] = f"{gst_report_dirname}/main.html"
        node.save()

        qs = get_qualibrate_config(get_qualibrate_config_path())
        node_dir = Path(get_node_dir_path(node.snapshot_idx, qs.storage.location))
        gst_report_dir = node_dir / gst_report_dirname

        try:
            report = pygsti.report.construct_standard_report(
                gst_results,
                title=f"GST Report - {qubit.name}",
            )
            report.write_html(str(gst_report_dir), auto_open=False)
            main_html = gst_report_dir / "main.html"
            if not main_html.exists():
                raise RuntimeError("pyGSTi finished without creating main.html")
            print(f"GST HTML report saved to {main_html}")
        except Exception as exc:
            raise RuntimeError(
                "Failed to generate GST HTML report. "
                "pyGSTi requires plotly<6 for HTML reports; run `uv sync` to install a compatible plotly version."
            ) from exc

# %%

