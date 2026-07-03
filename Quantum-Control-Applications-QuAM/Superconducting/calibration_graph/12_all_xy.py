"""
        ALL-XY SEQUENCE
The All-XY sequence consists of 21 pairs of single-qubit gates (I, X, Y, X-Y, Y-X, etc.)
designed to probe gate errors and phase coherence. Each sequence returns the qubit to
the ground, excited or equator state in the ideal case; deviations indicate miscalibration.

Prerequisites:
    - Having calibrated the mixer or the Octave.
    - Having calibrated the qubit x180 and x90 pulse parameters.
    - Having calibrated the qubit frequency.
    - Having calibrated the readout threshold if state discrimination is used.
    - Set the desired flux bias.
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubits_from_node,
)
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
from quam_libs.experiments.node_parameters import (
    QubitsExperimentNodeParameters,
    SimulatableNodeParameters,
    FluxControlledNodeParameters,
    MultiplexableNodeParameters,
    DataLoadableNodeParameters,
    QmSessionNodeParameters,
)
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(
    NodeParameters,
    SimulatableNodeParameters,
    DataLoadableNodeParameters,
    QmSessionNodeParameters,
    FluxControlledNodeParameters,
    MultiplexableNodeParameters,
    QubitsExperimentNodeParameters,
):
    qubits: Optional[List[str]] = None
    num_averages: int = 300
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    use_state_discrimination: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False


node = QualibrationNode(name="12_all_xy", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)


# %% {QUA_program_parameters}
n_avg = node.parameters.num_averages
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
state_discrimination = node.parameters.use_state_discrimination

# All-XY sequences: 21 pairs of single-qubit gates (Reed's thesis / Phys. Rev. A 82).
# "I" = identity (wait). Names must match qubit.xy.operations.
ALL_XY_SEQUENCES = [
    ("I", "I"),
    ("x180", "x180"),
    ("y180", "y180"),
    ("x180", "y180"),
    ("y180", "x180"),
    ("x90", "I"),
    ("y90", "I"),
    ("x90", "y90"),
    ("y90", "x90"),
    ("x90", "y180"),
    ("y90", "x180"),
    ("x180", "y90"),
    ("y180", "x90"),
    ("x90", "x180"),
    ("x180", "x90"),
    ("y90", "y180"),
    ("y180", "y90"),
    ("x180", "I"),
    ("y180", "I"),
    ("x90", "x90"),
    ("y90", "y90"),
]
ALL_XY_LABELS = [",".join(sequence) for sequence in ALL_XY_SEQUENCES]
N_ALL_XY = len(ALL_XY_SEQUENCES)
N_GROUND = 5
N_SUPERPOSITION = 12
N_EXCITED = 4
sequence_indices = np.arange(N_ALL_XY)


# %% {Utility_functions}
def play_identity(qubit):
    qubit.xy.wait(qubit.xy.operations["x90"].length // 4)


def play_all_xy_sequence(sequence_index, multiplexed_qubits):
    with switch_(sequence_index):
        with case_(0):  # I, I
            for qubit in multiplexed_qubits.values():
                play_identity(qubit)
                play_identity(qubit)
        with case_(1):  # x180, x180
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("x180")
                qubit.xy.play("x180")
        with case_(2):  # y180, y180
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("y180")
                qubit.xy.play("y180")
        with case_(3):  # x180, y180
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("x180")
                qubit.xy.play("y180")
        with case_(4):  # y180, x180
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("y180")
                qubit.xy.play("x180")
        with case_(5):  # x90, I
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("x90")
                play_identity(qubit)
        with case_(6):  # y90, I
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("y90")
                play_identity(qubit)
        with case_(7):  # x90, y90
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("x90")
                qubit.xy.play("y90")
        with case_(8):  # y90, x90
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("y90")
                qubit.xy.play("x90")
        with case_(9):  # x90, y180
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("x90")
                qubit.xy.play("y180")
        with case_(10):  # y90, x180
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("y90")
                qubit.xy.play("x180")
        with case_(11):  # x180, y90
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("x180")
                qubit.xy.play("y90")
        with case_(12):  # y180, x90
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("y180")
                qubit.xy.play("x90")
        with case_(13):  # x90, x180
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("x90")
                qubit.xy.play("x180")
        with case_(14):  # x180, x90
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("x180")
                qubit.xy.play("x90")
        with case_(15):  # y90, y180
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("y90")
                qubit.xy.play("y180")
        with case_(16):  # y180, y90
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("y180")
                qubit.xy.play("y90")
        with case_(17):  # x180, I
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("x180")
                play_identity(qubit)
        with case_(18):  # y180, I
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("y180")
                play_identity(qubit)
        with case_(19):  # x90, x90
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("x90")
                qubit.xy.play("x90")
        with case_(20):  # y90, y90
            for qubit in multiplexed_qubits.values():
                qubit.xy.play("y90")
                qubit.xy.play("y90")


def add_sequence_labels(ds):
    ds = ds.assign_coords(sequence_label=("sequence_index", ALL_XY_LABELS))
    ds.sequence_index.attrs = {"long_name": "All-XY sequence index"}
    return ds


def build_expected_state(values):
    vmin = float(np.nanmin(values))
    vmean = float(np.nanmean(values))
    vmax = float(np.nanmax(values))
    return [vmin] * N_GROUND + [vmean] * N_SUPERPOSITION + [vmax] * N_EXCITED


def build_expected_iq(values):
    vmin = float(np.nanmin(values))
    vmean = float(np.nanmean(values))
    vmax = float(np.nanmax(values))
    return [vmax] * N_GROUND + [vmean] * N_SUPERPOSITION + [vmin] * N_EXCITED


def compute_rms(measured, expected):
    measured = np.asarray(measured, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return float(np.sqrt(np.mean((measured - expected) ** 2)))


# %% {QUA_program}
with program() as all_xy:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    sequence_index = declare(int)
    iteration = declare(int)
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]

    if not node.parameters.simulate:
        machine.apply_all_couplers_to_min()
    assign(iteration, 0)

    for multiplexed_qubits in qubits.batch():
        if not node.parameters.simulate:
            if flux_point == "independent":
                machine.apply_all_flux_to_min()
                for qubit in multiplexed_qubits.values():
                    qubit.z.to_independent_idle()
            elif flux_point == "joint":
                machine.apply_all_flux_to_joint_idle()
            else:
                raise ValueError(f"Unrecognized flux point {flux_point}.")

        align(
            *(
                [q.xy.name for q in multiplexed_qubits.values()]
                + [q.resonator.name for q in multiplexed_qubits.values()]
                + [q.z.name for q in multiplexed_qubits.values()]
            )
        )

        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(sequence_index, sequence_indices)):
                for qubit in multiplexed_qubits.values():
                    if not node.parameters.simulate:
                        if reset_type == "active":
                            active_reset(qubit)
                        elif reset_type == "thermal":
                            qubit.wait(qubit.thermalization_time * u.ns)
                        else:
                            raise ValueError(f"Unrecognized reset type {reset_type}.")

                align(
                    *(
                        [q.xy.name for q in multiplexed_qubits.values()]
                        + [q.resonator.name for q in multiplexed_qubits.values()]
                        + [q.z.name for q in multiplexed_qubits.values()]
                    )
                )

                play_all_xy_sequence(sequence_index, multiplexed_qubits)

                align(
                    *(
                        [q.xy.name for q in multiplexed_qubits.values()]
                        + [q.resonator.name for q in multiplexed_qubits.values()]
                        + [q.z.name for q in multiplexed_qubits.values()]
                    )
                )

                for i, qubit in multiplexed_qubits.items():
                    if state_discrimination:
                        readout_state(qubit, state[i])
                        save(state[i], state_st[i])
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

                align(
                    *(
                        [q.xy.name for q in multiplexed_qubits.values()]
                        + [q.resonator.name for q in multiplexed_qubits.values()]
                        + [q.z.name for q in multiplexed_qubits.values()]
                    )
                )

            assign(iteration, iteration + 1)
            save(iteration, n_st)

        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("iteration")
        for i in range(num_qubits):
            if state_discrimination:
                state_st[i].buffer(N_ALL_XY).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(N_ALL_XY).average().save(f"I{i + 1}")
                Q_st[i].buffer(N_ALL_XY).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, all_xy, simulation_config)
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
        job = qm.execute(all_xy)
        results = fetching_tool(job, ["iteration"], mode="live")
        total_iterations = n_avg if node.parameters.multiplexed else n_avg * num_qubits
        while results.is_processing():
            iteration = results.fetch_all()[0]
            progress_counter(iteration, total_iterations, start_time=results.start_time)


# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"sequence_index": sequence_indices})
        ds = add_sequence_labels(ds)
        if not state_discrimination:
            ds = convert_IQ_to_V(ds, qubits)
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubits = resolve_qubits_from_node(machine, node)
        ds = add_sequence_labels(ds)

    node.results = {"ds": ds}

    # %% {Data_analysis}
    fit_results = {}

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    x = np.arange(N_ALL_XY)
    for ax, qubit in grid_iter(grid):
        qubit_name = qubit["qubit"]
        if "state" in ds.data_vars:
            state_values = ds.state.sel(qubit=qubit_name).values
            expected_state = build_expected_state(state_values)
            rms = compute_rms(state_values, expected_state)
            fit_results[qubit_name] = {"success": True, "rms": rms}
            ax.plot(x, state_values, "bo-", label="state")
            ax.plot(x, expected_state, "r-", alpha=0.8, label="expected")
            ax.set_ylabel("State population")
            ax.set_ylim(-0.05, 1.05)
            rms_text = f"RMS = {rms:.4f}"
        else:
            I_values = ds.I.sel(qubit=qubit_name).values
            Q_values = ds.Q.sel(qubit=qubit_name).values
            expected_I = np.array(build_expected_iq(I_values))
            expected_Q = np.array(build_expected_iq(Q_values))
            I_mV = 1e3 * I_values
            Q_mV = 1e3 * Q_values
            rms_I = compute_rms(I_mV, 1e3 * expected_I)
            rms_Q = compute_rms(Q_mV, 1e3 * expected_Q)
            rms = compute_rms(np.concatenate([I_mV, Q_mV]), np.concatenate([1e3 * expected_I, 1e3 * expected_Q]))
            fit_results[qubit_name] = {"success": True, "rms": rms, "rms_I": rms_I, "rms_Q": rms_Q}
            ax.plot(x, I_mV, "bo-", label="I")
            ax.plot(x, Q_mV, "go-", label="Q")
            ax.plot(x, 1e3 * expected_I, "r-", alpha=0.8, label="expected I")
            ax.plot(x, 1e3 * expected_Q, "m-", alpha=0.8, label="expected Q")
            ax.set_ylabel("Quadrature [mV]")
            rms_text = f"RMS = {rms:.2f} mV\n(RMS I = {rms_I:.2f}, RMS Q = {rms_Q:.2f})"

        ax.set_xlabel("Sequence")
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_XY_LABELS, rotation=45, ha="right")
        ax.set_title(qubit_name)
        ax.grid("all")
        ax.legend(framealpha=0)
        ax.text(
            0.02,
            0.98,
            rms_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

    node.results["fit_results"] = fit_results

    grid.fig.suptitle(f"All-XY (multiplexed={node.parameters.multiplexed})")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Save_results}
    if node.parameters.load_data_id is None:
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.save()

# %%
