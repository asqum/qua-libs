# %%
"""
        CZ Z-BASIS STATE DISTRIBUTION (|00>, |11>, |11>+CZ, |11>+CZ->|00>, |11>->|00>)

This sequence measures the joint two-qubit Z-basis state distribution of a qubit pair in
five scenarios:
    1. Reset both qubits and readout in |00⟩ (no drive pulses).
    2. Reset both qubits, prepare |11⟩ (π pulse on each qubit), then readout.
    3. Reset both qubits, prepare |11⟩, apply a CZ gate, then readout.
    4. Reset both qubits, prepare |11⟩, apply a CZ gate, then a second π pulse on each
       qubit (attempting to return to |00⟩), then readout.
    5. Reset both qubits, prepare |11⟩, then a second π pulse on each qubit (no CZ,
       returning to |00⟩), then readout.

Each shot is classified per qubit into |g>, |e>, |f> using the GEF centers stored from
08d_IQ_Blobs_G_E_F (same Manhattan-distance rule as readout_state_gef). The two single-qubit
labels are combined into one of 9 joint states (gg, ge, gf, eg, ee, ef, fg, fe, ff) and the
resulting probability distribution is plotted for each of the 5 sequences, grouped into a
single bar chart per qubit pair (in the style of 40a_Bell_state_Zbasis).

Prerequisites:
    - Calibrated single-qubit π pulses for both qubits
    - Calibrated GEF readout (run 08d_IQ_Blobs_G_E_F so gef_centers exist)
    - Calibrated CZ gate for the selected qubit pair

Outcomes:
    - Joint |g>/|e>/|f> population percentages for all 5 sequences, grouped per qubit pair
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset_gef
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubit_pairs_from_node,
)
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# %% {Node_parameters}
qubit_pair_indexes = [4]
class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s" % (i, i + 1) for i in qubit_pair_indexes]
    num_runs: int = 3000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal["thermal", "active"] = "thermal"
    operation: Literal["Cz", "Cz_flattop", "Cz_unipolar", "Cz_bipolar"] = "Cz"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None


node = QualibrationNode(name="93_cz_zbasis", parameters=Parameters())
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), (
    "If simulate is True, load_data_id must be None, and vice versa."
)

SEQUENCE_LABELS = {
    0: "|00⟩",
    1: "|11⟩",
    2: "|11⟩ + CZ",
    3: "|11⟩ + CZ → |00⟩",
    4: "|11⟩ → |00⟩",
}
STATE_LABELS_1Q = ["g", "e", "f"]
STATE_LABELS_2Q = [c + t for c in STATE_LABELS_1Q for t in STATE_LABELS_1Q]
SEQ_COLORS = ["C0", "C1", "C2", "C3", "C4"]


# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)
machine = QuAM.load()
node.machine = machine

if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

num_qubit_pairs = len(qubit_pairs)
config = machine.generate_config()
octave_config = machine.get_octave_config()
if node.parameters.load_data_id is None:
    qmm = machine.connect()

node.namespace["qubit_pairs"] = qubit_pairs


def gef_frequency_shift(qubit):
    shift = qubit.resonator.GEF_frequency_shift
    return 0 if shift is None else shift


def measure_iq_gef(qubit, I, Q):
    """Measure IQ at the GEF readout frequency, matching readout_state_gef."""
    qubit.resonator.update_frequency(
        qubit.resonator.intermediate_frequency + gef_frequency_shift(qubit),
        keep_phase=True,
    )
    wait(4)
    qubit.resonator.measure("readout", qua_vars=(I, Q))
    wait(4)
    qubit.resonator.update_frequency(qubit.resonator.intermediate_frequency, keep_phase=True)
    wait(4)
    qubit.wait(qubit.resonator.depletion_time // 4)


def classify_gef(I, Q, centers):
    """Manhattan-distance GEF classification, matching readout_state_gef."""
    dist = np.stack(
        [np.abs(I - centers[p][0]) + np.abs(Q - centers[p][1]) for p in range(3)],
        axis=0,
    )
    return np.argmin(dist, axis=0)


def joint_state_percentages(labels_c, labels_t):
    """Combine per-qubit GEF labels (0:g,1:e,2:f) into the 9 joint states and return
    their fractions, ordered as STATE_LABELS_2Q (gg, ge, gf, eg, ee, ef, fg, fe, ff)."""
    joint = labels_c * 3 + labels_t
    counts = np.array([(joint == k).sum() for k in range(9)], dtype=float)
    return counts / max(len(joint), 1)


def plot_joint_bars(ax, seq_pops, title):
    x = np.arange(len(STATE_LABELS_2Q))
    n_seq = len(seq_pops)
    width = 0.8 / max(n_seq, 1)
    for i_seq, (seq_label, pops) in enumerate(seq_pops):
        offset = (i_seq - (n_seq - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            100 * np.array(pops),
            width,
            label=seq_label,
            color=SEQ_COLORS[i_seq % len(SEQ_COLORS)],
            alpha=0.85,
        )
        for bar, pop in zip(bars, pops):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{100 * pop:.0f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_LABELS_2Q)
    ax.set_ylabel("Probability [%]")
    ax.set_ylim(0, 110)
    ax.set_title(title)
    ax.legend(fontsize=7)


# %% {QUA_program}
n_runs = node.parameters.num_runs
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type
operation_name = node.parameters.operation
sequences = list(SEQUENCE_LABELS.keys())

with program() as cz_zbasis:
    I_c, I_c_st, Q_c, Q_c_st, n, n_st = qua_declaration(num_qubits=num_qubit_pairs)
    I_t, I_t_st, Q_t, Q_t_st, _, _ = qua_declaration(num_qubits=num_qubit_pairs)
    seq = declare(int)

    for i, qp in enumerate(qubit_pairs):
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point, qp)
        wait(1000)

        with for_(n, 0, n < n_runs, n + 1):
            save(n, n_st)
            with for_(*from_array(seq, sequences)):
                if not node.parameters.simulate:
                    if reset_type == "active":
                        active_reset_gef(qp.qubit_control)
                        active_reset_gef(qp.qubit_target)
                    elif reset_type == "thermal":
                        wait(qp.qubit_control.thermalization_time * u.ns)
                        wait(qp.qubit_target.thermalization_time * u.ns)
                    else:
                        raise ValueError(f"Unrecognized reset type {reset_type}.")

                qp.align()
                reset_frame(qp.qubit_control.xy.name)
                reset_frame(qp.qubit_target.xy.name)

                # seq 0: |00⟩, no drive pulses
                # seq 1: |11⟩
                # seq 2: |11⟩ + CZ
                # seq 3: |11⟩ + CZ, then a second π pulse on each qubit back to |00⟩
                # seq 4: |11⟩, then a second π pulse on each qubit back to |00⟩ (no CZ)
                with if_(seq == 0):
                    wait(4)
                with elif_(seq == 1):
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_target.xy.play("x180")
                    qp.align()
                with elif_(seq == 2):
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_target.xy.play("x180")
                    qp.align()
                    qp.gates[operation_name].execute()
                    qp.align()
                with elif_(seq == 3):
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_target.xy.play("x180")
                    qp.align()
                    qp.gates[operation_name].execute()
                    qp.align()
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_target.xy.play("x180")
                    qp.align()
                with elif_(seq == 4):
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_target.xy.play("x180")
                    qp.align()
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_target.xy.play("x180")
                    qp.align()

                measure_iq_gef(qp.qubit_control, I_c[i], Q_c[i])
                qp.align()
                measure_iq_gef(qp.qubit_target, I_t[i], Q_t[i])
                save(I_c[i], I_c_st[i])
                save(Q_c[i], Q_c_st[i])
                save(I_t[i], I_t_st[i])
                save(Q_t[i], Q_t_st[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            I_c_st[i].buffer(len(sequences)).buffer(n_runs).save(f"I_control{i + 1}")
            Q_c_st[i].buffer(len(sequences)).buffer(n_runs).save(f"Q_control{i + 1}")
            I_t_st[i].buffer(len(sequences)).buffer(n_runs).save(f"I_target{i + 1}")
            Q_t_st[i].buffer(len(sequences)).buffer(n_runs).save(f"Q_target{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=10_000)
    job = qmm.simulate(config, cz_zbasis, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(cz_zbasis)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_runs, start_time=results.start_time)


# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_results_as_xarray(
            job.result_handles,
            qubit_pairs,
            {"sequence": sequences, "N": np.arange(n_runs)},
        )
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
        node.namespace["qubit_pairs"] = qubit_pairs

    def extract_value(element):
        if isinstance(element, tuple):
            return element[0]
        return element

    ds = xr.apply_ufunc(
        extract_value,
        ds,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    if "qubit" in ds.dims:
        ds = ds.rename({"qubit": "qubit_pair"})
    node.results = {"ds": ds, "results": {}}


# %% {Analyse_data}
if not node.parameters.simulate:
    qubit_pairs = node.namespace["qubit_pairs"]
    ds = node.results["ds"]

    joint_pops = {}
    for qp in qubit_pairs:
        missing = [
            q.name
            for q in (qp.qubit_control, qp.qubit_target)
            if getattr(q.resonator, "gef_centers", None) is None
        ]
        if missing:
            raise ValueError(
                f"Missing gef_centers for {missing}. Run 08d_IQ_Blobs_G_E_F first."
            )

        centers_c = np.array(qp.qubit_control.resonator.gef_centers, dtype=float)
        centers_t = np.array(qp.qubit_target.resonator.gef_centers, dtype=float)

        pair_results = {}
        seq_pops = []
        for seq, seq_label in SEQUENCE_LABELS.items():
            I_c = ds["I_control"].sel(qubit_pair=qp.name, sequence=seq).values
            Q_c = ds["Q_control"].sel(qubit_pair=qp.name, sequence=seq).values
            I_t = ds["I_target"].sel(qubit_pair=qp.name, sequence=seq).values
            Q_t = ds["Q_target"].sel(qubit_pair=qp.name, sequence=seq).values

            labels_c = classify_gef(I_c, Q_c, centers_c)
            labels_t = classify_gef(I_t, Q_t, centers_t)
            pops = joint_state_percentages(labels_c, labels_t)

            seq_pops.append((seq_label, pops))
            pair_results[seq_label] = {
                state_label: float(pop) for state_label, pop in zip(STATE_LABELS_2Q, pops)
            }
            print(f"{qp.name} {seq_label}: " + ", ".join(
                f"{state_label}={100 * pop:.1f}%" for state_label, pop in zip(STATE_LABELS_2Q, pops)
            ))

        joint_pops[qp.name] = seq_pops
        node.results["results"][qp.name] = pair_results


# %% {Plotting}
if not node.parameters.simulate:
    qubit_pairs = node.namespace["qubit_pairs"]

    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        plot_joint_bars(ax, joint_pops[qubit_pair["qubit"]], qubit_pair["qubit"])
    grid.fig.suptitle("Joint |g>/|e>/|f> distribution: |00⟩, |11⟩, |11⟩+CZ, |11⟩+CZ→|00⟩, |11⟩→|00⟩", fontsize=13)
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig


# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()

# %%
