# %%
"""
        IQ BLOBS AFTER |00>, |11>, |11>+CZ, AND |11> BACK TO |00>

This sequence measures IQ blobs of a qubit pair in four scenarios:
    1. Reset both qubits and readout in |00⟩ (no drive pulses).
    2. Reset both qubits, prepare |11⟩ (π pulse on each qubit), then readout.
    3. Reset both qubits, prepare |11⟩, apply a CZ gate, then readout.
    4. Reset both qubits, prepare |11⟩, then a second π pulse on each qubit
       to return to |00⟩, then readout.

Each shot is classified into |g>, |e>, |f> using the GEF centers stored from
08d_IQ_Blobs_G_E_F (same Manhattan-distance rule as readout_state_gef).
The resulting IQ scatter and three-state percentages are plotted for the
control and target qubits of each pair.

Prerequisites:
    - Calibrated single-qubit π pulses for both qubits
    - Calibrated GEF readout (run 08d_IQ_Blobs_G_E_F so gef_centers exist)
    - Calibrated CZ gate for the selected qubit pair

Outcomes:
    - IQ blobs of |00⟩
    - IQ blobs of |11⟩ vs |11⟩+CZ
    - IQ blobs of |11⟩ driven back to |00⟩
    - |g>, |e>, |f> population percentages for all sequences
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


node = QualibrationNode(name="92_IQ_Blobs_cz", parameters=Parameters())
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), (
    "If simulate is True, load_data_id must be None, and vice versa."
)

SEQUENCE_LABELS = {0: "|00⟩", 1: "|11⟩", 2: "|11⟩ + CZ", 3: "|11⟩→|00⟩"}
STATE_LABELS = ["|g⟩", "|e⟩", "|f⟩"]
STATE_COLORS = ["C0", "C1", "C2"]
BAR_COLORS = ["C2", "C3", "C4", "C5"]


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


def iq_raw_to_v(iq, qubit):
    return np.asarray(iq) * 2**12 / qubit.resonator.operations["readout"].length


def classify_gef(I, Q, centers):
    """Manhattan-distance GEF classification, matching readout_state_gef."""
    dist = np.stack(
        [np.abs(I - centers[p][0]) + np.abs(Q - centers[p][1]) for p in range(3)],
        axis=0,
    )
    return np.argmin(dist, axis=0)


def draw_gef_boundaries(ax, centers, xlim, ylim, n=250):
    """Manhattan nearest-center boundaries, matching readout_state_gef."""
    centers = np.asarray(centers, dtype=float)
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    xx, yy = np.meshgrid(xs, ys)
    dist = np.stack([np.abs(xx - c[0]) + np.abs(yy - c[1]) for c in centers], axis=0)
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        field = np.ma.masked_where(dist[k] < np.minimum(dist[i], dist[j]), dist[i] - dist[j])
        vals = field.compressed()
        if vals.size == 0 or vals.min() > 0 or vals.max() < 0:
            continue
        ax.contour(
            xx, yy, field, levels=[0], colors=[(0.35, 0.35, 0.35, 0.45)], linewidths=1.0, zorder=4
        )


def state_percentages(labels):
    counts = np.array([(labels == p).sum() for p in range(3)], dtype=float)
    return counts / max(len(labels), 1)


def square_limits(xlim, ylim, pad=0.05):
    """Expand limits to a square window so every IQ panel has the same box size."""
    x0, x1 = xlim
    y0, y1 = ylim
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.5 * max(x1 - x0, y1 - y0) * (1 + pad)
    return (cx - half, cx + half), (cy - half, cy + half)


def plot_iq_scatter(ax, I_v, Q_v, labels, pops, centers_v, title):
    for state in range(3):
        mask = labels == state
        ax.plot(
            I_v[mask],
            Q_v[mask],
            ".",
            color=STATE_COLORS[state],
            alpha=0.25,
            markersize=2,
            label=f"{STATE_LABELS[state]} ({100 * pops[state]:.1f}%)",
        )
    for state, center in enumerate(centers_v):
        ax.plot(
            1e3 * center[0],
            1e3 * center[1],
            "o",
            c=STATE_COLORS[state],
            ms=7,
            mec="k",
            zorder=5,
        )
    xlim, ylim = square_limits(ax.get_xlim(), ax.get_ylim())
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    draw_gef_boundaries(ax, 1e3 * np.asarray(centers_v, dtype=float), xlim, ylim)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel("I [mV]")
    ax.set_ylabel("Q [mV]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, markerscale=4)


def plot_gef_bars(ax, seq_pops, title):
    x = np.arange(3)
    n_seq = len(seq_pops)
    width = 0.8 / max(n_seq, 1)
    for i_seq, (seq_label, pops) in enumerate(seq_pops):
        offset = (i_seq - (n_seq - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            100 * np.array(pops),
            width,
            label=seq_label,
            color=BAR_COLORS[i_seq % len(BAR_COLORS)],
            alpha=0.85,
        )
        for bar, pop in zip(bars, pops):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{100 * pop:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_LABELS)
    ax.set_ylabel("Population [%]")
    ax.set_ylim(0, 110)
    ax.set_box_aspect(1)
    ax.set_title(title)
    ax.legend(fontsize=8)


def shot_data(ds, qp_name, qubit, I_name, Q_name, seq):
    I_raw = ds[I_name].sel(qubit_pair=qp_name, sequence=seq).values
    Q_raw = ds[Q_name].sel(qubit_pair=qp_name, sequence=seq).values
    I_v = 1e3 * iq_raw_to_v(I_raw, qubit)
    Q_v = 1e3 * iq_raw_to_v(Q_raw, qubit)
    labels = classify_gef(I_raw, Q_raw, qubit.resonator.gef_centers)
    return I_v, Q_v, labels


# %% {QUA_program}
n_runs = node.parameters.num_runs
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type
operation_name = node.parameters.operation
sequences = [0, 1, 2, 3]

with program() as iq_blobs_cz:
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
                # seq 3: |11⟩, then a second π pulse on each qubit back to |00⟩
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
    job = qmm.simulate(config, iq_blobs_cz, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(iq_blobs_cz)
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

        pair_results = {}
        for role, qubit, I_name, Q_name in (
            ("control", qp.qubit_control, "I_control", "Q_control"),
            ("target", qp.qubit_target, "I_target", "Q_target"),
        ):
            centers = np.array(qubit.resonator.gef_centers, dtype=float)
            role_results = {"centers_raw": centers.tolist(), "sequences": {}}
            for seq, seq_label in SEQUENCE_LABELS.items():
                I = ds[I_name].sel(qubit_pair=qp.name, sequence=seq).values
                Q = ds[Q_name].sel(qubit_pair=qp.name, sequence=seq).values
                labels = classify_gef(I, Q, centers)
                pops = state_percentages(labels)
                role_results["sequences"][seq_label] = {
                    "P_g": float(pops[0]),
                    "P_e": float(pops[1]),
                    "P_f": float(pops[2]),
                }
                print(
                    f"{qp.name} {role} {seq_label}: "
                    f"g={100 * pops[0]:.1f}%, e={100 * pops[1]:.1f}%, f={100 * pops[2]:.1f}%"
                )
            pair_results[role] = role_results
        node.results["results"][qp.name] = pair_results


# %% {Plotting}
if not node.parameters.simulate:
    qubit_pairs = node.namespace["qubit_pairs"]
    ds = node.results["ds"]
    figures = {}
    roles = (
        ("control", "I_control", "Q_control"),
        ("target", "I_target", "Q_target"),
    )

    for qp in qubit_pairs:
        qubits_by_role = {"control": qp.qubit_control, "target": qp.qubit_target}
        fig, axes = plt.subplots(2, 5, figsize=(18, 8), layout="constrained")
        fig.suptitle(f"IQ blobs |00⟩, |11⟩, |11⟩+CZ, |11⟩→|00⟩  ({qp.name})", fontsize=13)

        for row, (role, I_name, Q_name) in enumerate(roles):
            qubit = qubits_by_role[role]
            centers_v = iq_raw_to_v(qubit.resonator.gef_centers, qubit)
            seq_pops = []
            for col, (seq, seq_label) in enumerate(SEQUENCE_LABELS.items()):
                I_v, Q_v, labels = shot_data(ds, qp.name, qubit, I_name, Q_name, seq)
                pops = [
                    node.results["results"][qp.name][role]["sequences"][seq_label][k]
                    for k in ("P_g", "P_e", "P_f")
                ]
                seq_pops.append((seq_label, pops))
                plot_iq_scatter(
                    axes[row, col],
                    I_v,
                    Q_v,
                    labels,
                    pops,
                    centers_v,
                    f"{qubit.name} ({role})  {seq_label}",
                )
            plot_gef_bars(axes[row, 4], seq_pops, f"{qubit.name} GEF %")

        figures[qp.name] = fig
        node.results[f"figure_IQ_blobs_{qp.name}"] = fig
        plt.show()

    node.results["figure_IQ_blobs"] = next(iter(figures.values()))


# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()

# %%
