# %%
"""
Bell-state tomography, two CZ operations compared shot by shot (copy of calibration_graph/40b_Bell_state_tomography.py).

Each shot index n measures, for every one of the 9 tomography bases, BOTH gates in `gate_names`, in the order
A B for even n and B A for odd n (the gate is chosen by a QUA variable, so each gate call site appears once per basis).
Preparation, tomography pulses, binary readout, readout-error correction and the density-matrix reconstruction are
unchanged from 40b (including its qubit-ordering / X-sign conventions), so the two gates are analysed identically and
their DIFFERENCE is meaningful; absolute rho elements keep 40b's conventions.

Outputs per gate: Bell-state fidelity, purity, rho, Pauli expectation values. The shots are split into contiguous
chunks (both gates share the same shot indices) to give a paired fidelity difference with SEM and 95 % CI.

Default: Cz_flattop_gaussian (stored CZ) vs composed_cz (its 1 ns composed copy). composed_cz.execute() takes its
per-ns scales from state extras as compile-time constants, unlike the DGX path (QUA variables + DCT expansion).

dry_run=True builds the program and prints its size without any hardware connection.
compile_only=True opens an own QM, compiles, and closes it (no execution).
"""

# %% {Imports}
import time
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats
from scipy.linalg import sqrtm
from qm import SimulationConfig, generate_qua_script
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode

from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q4_q5"]
    gate_names: List[str] = ["Cz_flattop_gaussian", "composed_cz"]     # exactly two operations in qp.gates
    num_shots: int = 4000
    n_chunks: int = 10                                                   # for the paired fidelity uncertainty
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal["active", "thermal"] = "active"
    dry_run: bool = False
    compile_only: bool = False
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(name="40x_Bell_state_tomography_gate_compare", parameters=Parameters())
assert len(node.parameters.gate_names) == 2, "gate_names must contain exactly two operations"
assert node.parameters.num_shots % node.parameters.n_chunks == 0, "num_shots must be divisible by n_chunks"

# %% {Initialize_QuAM}
u = unit(coerce_to_integer=True)
machine = QuAM.load()
node.machine = machine
qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
num_qubit_pairs = len(qubit_pairs)
gate_names = node.parameters.gate_names
for qp in qubit_pairs:
    missing = [g for g in gate_names if g not in qp.gates]
    assert not missing, f"{missing} not in {qp.name}.gates: {list(qp.gates)}"
    for g in gate_names:
        src = getattr(qp.gates[g], "extras", {}).get("source_operation")
        print(f"{qp.name}.{g}: {type(qp.gates[g]).__name__}" + (f" (1 ns copy of {src})" if src else ""))
config = machine.generate_config()


# %% {Helper_functions}  (unchanged from 40b)
def flatten(data):
    if isinstance(data, tuple):
        if len(data) == 0:
            return ()
        return flatten(data[0]) + flatten(data[1:])
    return (data,)


def generate_pauli_basis(n_qubits):
    pauli = np.array([0, 1, 2, 3])
    paulis = pauli
    for _ in range(n_qubits - 1):
        new_paulis = []
        for ps in paulis:
            for p in pauli:
                new_paulis.append(flatten((ps, p)))
        paulis = new_paulis
    return paulis


def gen_inverse_hadamard(n_qubits):
    H = np.array([[1, 1], [1, -1]]) / 2
    for _ in range(n_qubits - 1):
        H = np.kron(H, H)
    return np.linalg.inv(H)


def get_pauli_data(da):
    pauli_basis = generate_pauli_basis(2)
    inverse_hadamard = gen_inverse_hadamard(2)
    paulis_data = xr.Dataset(
        {"value": (["pauli_op"], np.zeros(len(pauli_basis))),
         "appearances": (["pauli_op"], np.zeros(len(pauli_basis), dtype=int))},
        coords={"pauli_op": [",".join(map(str, op)) for op in pauli_basis]},
    )
    for tomo_axis in da.coords["tomo_axis"].values:
        tomo_data = da.sel(tomo_axis=tomo_axis)
        pauli_data = inverse_hadamard @ tomo_data.data
        paulis = ["0,0", f"{tomo_axis[0]+1},0", f"0,{tomo_axis[1]+1}", f"{tomo_axis[0]+1},{tomo_axis[1]+1}"]
        for i, pauli in enumerate(paulis):
            paulis_data.value.loc[{"pauli_op": pauli}] += pauli_data[i]
            paulis_data.appearances.loc[{"pauli_op": pauli}] += 1
    return xr.where(paulis_data.appearances != 0, paulis_data.value / paulis_data.appearances, paulis_data.value)


def get_density_matrix(paulis_data):
    I = np.array([[1, 0], [0, 1]]); X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]]); Z = np.array([[1, 0], [0, -1]])
    pauli_matrices = [I, X, Y, Z]
    rho = np.zeros((4, 4))
    for i, pauli_i in enumerate(pauli_matrices):
        for j, pauli_j in enumerate(pauli_matrices):
            rho = rho + 0.25 * paulis_data.sel(pauli_op=f"{i},{j}").values * np.kron(pauli_i, pauli_j)
    return rho


IDEAL = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2
S_IDEAL = sqrtm(IDEAL)


def analyze(states, confusion):
    """states: int array [shots, control_axis, target_axis] with values 0..3 (= 2*control + target).
    Same steps as 40b: probabilities -> inv(confusion) -> clip/renormalise -> Pauli -> rho -> fidelity, purity."""
    corrected = np.zeros((3, 3, 4))
    for c in range(3):
        for t in range(3):
            p = np.array([(states[:, c, t] == s).mean() for s in range(4)])
            p = np.linalg.inv(confusion) @ p
            p = p * (p > 0)
            corrected[c, t] = p / p.sum()
    da = xr.DataArray(corrected, dims=["tomo_axis_control", "tomo_axis_target", "state"],
                      coords={"tomo_axis_control": [0, 1, 2], "tomo_axis_target": [0, 1, 2], "state": ["00", "01", "10", "11"]})
    da = da.stack(tomo_axis=["tomo_axis_target", "tomo_axis_control"])
    paulis = get_pauli_data(da)
    rho = get_density_matrix(paulis)
    fidelity = float(np.abs(np.trace(sqrtm(S_IDEAL @ rho @ S_IDEAL))) ** 2)
    purity = float(np.abs(np.trace(rho @ rho)))
    return {"fidelity": fidelity, "purity": purity, "rho": rho, "paulis": paulis}


# %% {QUA_program}
n_shots = node.parameters.num_shots
flux_point = node.parameters.flux_point_joint_or_independent

with program() as gate_compare:
    n, k_ab, arm = declare(int), declare(int), declare(int)
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state = [declare(int) for _ in range(num_qubit_pairs)]
    state_st = [[declare_stream() for _ in range(2)] for _ in range(num_qubit_pairs)]

    for i, qp in enumerate(qubit_pairs):
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point, qp.qubit_control)
        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st)
            for tomo_axis_control in [0, 1, 2]:
                for tomo_axis_target in [0, 1, 2]:
                    with for_(k_ab, 0, k_ab < 2, k_ab + 1):               # A B for even n, B A for odd n
                        assign(arm, (n + k_ab) & 1)
                        if not node.parameters.simulate:
                            if node.parameters.reset_type == "active":
                                active_reset(qp.qubit_control)
                                active_reset(qp.qubit_target)
                            else:
                                wait(5 * qp.qubit_control.thermalization_time * u.ns)
                        qp.align()
                        qp.qubit_control.xy.play("-y90")
                        qp.qubit_target.xy.play("y90")
                        qp.align()
                        with if_(arm == 0):
                            qp.gates[gate_names[0]].execute()
                        with else_():
                            qp.gates[gate_names[1]].execute()
                        qp.align()
                        qp.qubit_control.xy.play("y90")
                        qp.align()
                        if tomo_axis_control == 0:
                            qp.qubit_control.xy.play("y90")
                        if tomo_axis_control == 1:
                            qp.qubit_control.xy.play("x90")
                        if tomo_axis_target == 0:
                            qp.qubit_target.xy.play("y90")
                        if tomo_axis_target == 1:
                            qp.qubit_target.xy.play("x90")
                        qp.align()
                        readout_state(qp.qubit_control, state_control[i])
                        readout_state(qp.qubit_target, state_target[i])
                        assign(state[i], state_control[i] * 2 + state_target[i])
                        with if_(arm == 0):
                            save(state[i], state_st[i][0])
                        with else_():
                            save(state[i], state_st[i][1])
                    align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            for g in range(2):
                state_st[i][g].buffer(3).buffer(3).buffer(n_shots).save(f"state_g{g}_{i + 1}")

# %% {Dry_run_compile_or_execute}
job = None
if node.parameters.dry_run:
    body = generate_qua_script(gate_compare, config).split("\nconfig = ")[0]
    print(f"DRY_RUN: {body.count(chr(10))} lines, {body.count('play(')} play, {body.count('assign(')} assign, "
          f"{body.count('strict_timing_(')} strict_timing blocks, {body.count('measure(')} measure. No hardware connection made.")
else:
    qmm = machine.connect()
    if node.parameters.compile_only:
        qm = qmm.open_qm(config, close_other_machines=False)
        try:
            t0 = time.time()
            print("compiled program id:", qm.compile(gate_compare), f"({time.time() - t0:.1f} s)")
        finally:
            print("own QM closed:", qm.close())
    elif node.parameters.simulate:
        job = qmm.simulate(config, gate_compare, SimulationConfig(duration=10_000))
        job.get_simulated_samples().con1.plot()
        node.results = {"figure": plt.gcf()}
    else:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(gate_compare)
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n_done = results.fetch_all()[0]
                progress_counter(n_done, n_shots, start_time=results.start_time)

run_measurement = job is not None and not node.parameters.simulate

# %% {Data_fetching_and_analysis}
if run_measurement:
    node.results = {}
    summary = {}
    chunk = n_shots // node.parameters.n_chunks
    for i, qp in enumerate(qubit_pairs):
        raw = {g: np.asarray(job.result_handles.get(f"state_g{gi}_{i + 1}").fetch_all(), dtype=int)   # [shots, control, target]
               for gi, g in enumerate(gate_names)}
        node.results[f"{qp.name}_raw"] = xr.Dataset(
            {g: (["N", "tomo_axis_control", "tomo_axis_target"], raw[g]) for g in gate_names},
            coords={"N": np.arange(n_shots), "tomo_axis_control": [0, 1, 2], "tomo_axis_target": [0, 1, 2]})
        full = {g: analyze(raw[g], qp.confusion) for g in gate_names}
        chunk_f = {g: np.array([analyze(raw[g][c * chunk:(c + 1) * chunk], qp.confusion)["fidelity"]
                                for c in range(node.parameters.n_chunks)]) for g in gate_names}
        d = chunk_f[gate_names[1]] - chunk_f[gate_names[0]]
        sem = d.std(ddof=1) / np.sqrt(len(d))
        tval = stats.t.ppf(0.975, len(d) - 1)
        summary[qp.name] = {
            **{f"F_{g}": full[g]["fidelity"] for g in gate_names},
            **{f"purity_{g}": full[g]["purity"] for g in gate_names},
            "dF_full": full[gate_names[1]]["fidelity"] - full[gate_names[0]]["fidelity"],
            "dF_chunks_mean": float(d.mean()), "dF_chunks_sem": float(sem),
            "dF_chunks_ci95": [float(d.mean() - tval * sem), float(d.mean() + tval * sem)],
        }
        for g in gate_names:
            print(f"{qp.name} {g:22s} F = {full[g]['fidelity']:.4f} | purity = {full[g]['purity']:.4f}")
        s = summary[qp.name]
        print(f"{qp.name} dF ({gate_names[1]} - {gate_names[0]}) = {100 * s['dF_full']:+.2f} pp (all shots); "
              f"chunks: {100 * s['dF_chunks_mean']:+.2f} ± {100 * s['dF_chunks_sem']:.2f} pp (sem), "
              f"95% CI [{100 * s['dF_chunks_ci95'][0]:+.2f}, {100 * s['dF_chunks_ci95'][1]:+.2f}] pp")

        # rho (real) of both gates and their difference
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        mats = [np.real(full[g]["rho"]) for g in gate_names] + [np.real(full[gate_names[1]]["rho"] - full[gate_names[0]]["rho"])]
        titles = [f"{g}\nF = {full[g]['fidelity']:.3f}" for g in gate_names] + [f"difference ({gate_names[1]} - {gate_names[0]})"]
        for ax, m, t, vmax in zip(axes, mats, titles, (0.5, 0.5, 0.1)):
            ax.pcolormesh(m, vmin=-vmax, vmax=vmax, cmap="RdBu")
            for a in range(4):
                for b in range(4):
                    ax.text(a + 0.5, b + 0.5, f"{m[a][b]:.2f}", ha="center", va="center", fontsize=8)
            ax.set_xticks(np.arange(4) + 0.5, ["00", "01", "10", "11"]); ax.set_yticks(np.arange(4) + 0.5, ["00", "01", "10", "11"])
            ax.set_title(t, fontsize=10)
        fig.suptitle(f"{qp.name}: Bell-state rho (real part), 40b conventions")
        fig.tight_layout()
        node.results[f"{qp.name}_figure_rho"] = fig

        # Pauli expectation values of both gates
        labels = [lbl.replace(",", "") for lbl in full[gate_names[0]]["paulis"].coords["pauli_op"].values]
        for num, ch in {"0": "I", "1": "X", "2": "Y", "3": "Z"}.items():
            labels = [lbl.replace(num, ch) for lbl in labels]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(12, 4))
        for off, g in zip((-0.2, 0.2), gate_names):
            ax.bar(x + off, full[g]["paulis"].values, 0.4, label=g)
        ax.set_xticks(x, labels); ax.axhline(0, color="0.5", lw=0.8); ax.grid(axis="y", ls=":")
        ax.set_title(f"{qp.name}: Pauli expectation values (40b conventions)"); ax.legend()
        fig.tight_layout()
        node.results[f"{qp.name}_figure_paulis"] = fig
        plt.show()
    node.results["summary"] = summary

# %% {Save_results}
if run_measurement:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()
# %%
