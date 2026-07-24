# %% {Imports}
from copy import copy

from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef
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
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names

# %% {Description}
description = """
        PARAMETRIC CZ COUPLER CALIBRATION MAP

2D sweep of Cz_Parametric gate coupler amplitude and modulation frequency
(relative to the current gate settings). Prepares |ee>, applies Cz_Parametric,
and measures control / target populations. Outputs mesh plots only.
"""

# %% {Node_parameters}
qubit_pair_indexes = [3]


class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s" % (i, i + 1) for i in qubit_pair_indexes]
    num_averages: int = 100
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal["active", "thermal"] = "active"
    simulate: bool = False
    timeout: int = 200
    load_data_id: Optional[int] = None

    coupler_amp_range: float = 0.5
    coupler_amp_step: float = 0.01
    freq_range: float = 0.1
    freq_step: float = 0.005
    operation: Literal["Cz_Parametric"] = "Cz_Parametric"
    use_state_discrimination: bool = True
    con_tar_flip: bool = False


node = QualibrationNode(name="61xx_coupler_parametric_cz", parameters=Parameters())
assert not (
    node.parameters.simulate and node.parameters.load_data_id is not None
), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)
machine = QuAM.load()
node.machine = machine

if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

num_qubit_pairs = len(qubit_pairs)
operation_name = node.parameters.operation

coupler_amp_scales = np.arange(
    1 - node.parameters.coupler_amp_range,
    1 + node.parameters.coupler_amp_range + node.parameters.coupler_amp_step / 2,
    node.parameters.coupler_amp_step,
)
freq_scales = np.arange(
    1 - node.parameters.freq_range,
    1 + node.parameters.freq_range + node.parameters.freq_step / 2,
    node.parameters.freq_step,
)

parametric_pulse_labels = {}
base_frequencies = {}
base_amplitudes = {}


def register_parametric_frequency_pulses(qp, config, freq_scale_array):
    gate = qp.gates[operation_name]
    base = gate.coupler_flux_pulse
    base_frequencies[qp.name] = float(base.frequency)
    base_amplitudes[qp.name] = float(base.amplitude)
    labels = []
    coupler_elem = config["elements"][qp.coupler.name]
    for i, fs in enumerate(freq_scale_array):
        pulse = copy(base)
        pulse.frequency = float(base.frequency * fs)
        pulse.id = f"coupler_parametric_f{i}"
        pulse.parent = None
        pulse.parent = qp.coupler
        label = f"{operation_name}.coupler_parametric_f{i}"
        pulse.apply_to_config(config)
        coupler_elem["operations"][label] = pulse.pulse_name
        labels.append(label)
    parametric_pulse_labels[qp.name] = labels
    return labels


config = machine.generate_config()
for qp in qubit_pairs:
    register_parametric_frequency_pulses(qp, config, freq_scales)

octave_config = machine.get_octave_config()
if node.parameters.load_data_id is None:
    qmm = machine.connect()


def play_parametric_coupler(qp, labels, freq_idx, amp_scale):
    with switch_(freq_idx):
        for i, label in enumerate(labels):
            with case_(i):
                qp.coupler.play(label, validate=False, amplitude_scale=amp_scale)


def execute_cz_parametric(qp, labels, freq_idx, amp_scale):
    gate = qp.gates[operation_name]
    qp.align()
    qp.qubit_control.z.play(
        gate.flux_pulse_control_label,
        validate=False,
        amplitude_scale=0.0,
    )
    play_parametric_coupler(qp, labels, freq_idx, amp_scale)
    qp.align()
    frame_rotation_2pi(gate.phase_shift_control, qp.qubit_control.xy.name)
    frame_rotation_2pi(gate.phase_shift_target, qp.qubit_target.xy.name)
    qp.qubit_control.xy.play("x180", amplitude_scale=0.0, duration=4)
    qp.qubit_target.xy.play("x180", amplitude_scale=0.0, duration=4)
    qp.align()


# %% {QUA_program}
n_avg = node.parameters.num_averages
flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise

with program() as parametric_cz_map:
    n = declare(int)
    amp_coupler = declare(fixed)
    freq_idx = declare(int)
    n_st = declare_stream()

    if node.parameters.use_state_discrimination:
        state_control = [declare(int) for _ in range(num_qubit_pairs)]
        state_target = [declare(int) for _ in range(num_qubit_pairs)]
        state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    else:
        I_control = [declare(float) for _ in range(num_qubit_pairs)]
        Q_control = [declare(float) for _ in range(num_qubit_pairs)]
        I_target = [declare(float) for _ in range(num_qubit_pairs)]
        Q_target = [declare(float) for _ in range(num_qubit_pairs)]
        I_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        Q_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        I_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
        Q_st_target = [declare_stream() for _ in range(num_qubit_pairs)]

    for i, qp in enumerate(qubit_pairs):
        qp.gates[operation_name].phase_shift_control = 0.0
        qp.gates[operation_name].phase_shift_target = 0.0
        labels = parametric_pulse_labels[qp.name]
        machine.set_all_fluxes(flux_point, qp)
        qp.coupler.to_decouple_idle()
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(amp_coupler, coupler_amp_scales)):
                with for_(freq_idx, 0, freq_idx < len(freq_scales), freq_idx + 1):
                    if not node.parameters.simulate:
                        if node.parameters.reset_type == "active":
                            active_reset(qp.qubit_control)
                            active_reset(qp.qubit_target)
                            qp.align()
                        else:
                            wait(qp.qubit_control.thermalization_time * u.ns)
                            wait(qp.qubit_target.thermalization_time * u.ns)
                    align()
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_target.xy.play("x180")
                    align()
                    execute_cz_parametric(qp, labels, freq_idx, amp_coupler)
                    wait(20)
                    if node.parameters.use_state_discrimination:
                        if not node.parameters.con_tar_flip:
                            readout_state(qp.qubit_control, state_control[i])
                            readout_state(qp.qubit_target, state_target[i])
                        else:
                            readout_state_gef(qp.qubit_control, state_control[i])
                            readout_state(qp.qubit_target, state_target[i])
                        save(state_control[i], state_st_control[i])
                        save(state_target[i], state_st_target[i])
                    else:
                        qp.qubit_control.resonator.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                        qp.qubit_target.resonator.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                        save(I_control[i], I_st_control[i])
                        save(Q_control[i], Q_st_control[i])
                        save(I_target[i], I_st_target[i])
                        save(Q_target[i], Q_st_target[i])
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_st_control[i].buffer(len(freq_scales)).buffer(len(coupler_amp_scales)).average().save(
                    f"state_control{i + 1}"
                )
                state_st_target[i].buffer(len(freq_scales)).buffer(len(coupler_amp_scales)).average().save(
                    f"state_target{i + 1}"
                )
            else:
                I_st_control[i].buffer(len(freq_scales)).buffer(len(coupler_amp_scales)).average().save(
                    f"I_control{i + 1}"
                )
                Q_st_control[i].buffer(len(freq_scales)).buffer(len(coupler_amp_scales)).average().save(
                    f"Q_control{i + 1}"
                )
                I_st_target[i].buffer(len(freq_scales)).buffer(len(coupler_amp_scales)).average().save(
                    f"I_target{i + 1}"
                )
                Q_st_target[i].buffer(len(freq_scales)).buffer(len(coupler_amp_scales)).average().save(
                    f"Q_target{i + 1}"
                )

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=30_000//4)  # In clock cycles = 4ns
    job = qmm.simulate(config, parametric_cz_map, simulation_config)
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)

    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.results = {"figure": plt.gcf()}
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(parametric_cz_map)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_results_as_xarray(
            job.result_handles,
            qubit_pairs,
            {"freq_scale": freq_scales, "amp_coupler": coupler_amp_scales},
        )
        freq_hz = np.array([base_frequencies[qp.name] * freq_scales for qp in qubit_pairs])
        amp_v = np.array([base_amplitudes[qp.name] * coupler_amp_scales for qp in qubit_pairs])
        ds = ds.assign_coords({"frequency_Hz": (["qubit", "freq_scale"], freq_hz)})
        ds = ds.assign_coords({"coupler_amp_V": (["qubit", "amp_coupler"], amp_v)})
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)

    node.results = {"ds": ds}

# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)

    for state_type in ["control", "target"]:
        grid = QubitPairGrid(grid_names, qubit_pair_names)
        for ax, qp in grid_iter(grid):
            qubit_name = qp["qubit"]
            try:
                if node.parameters.use_state_discrimination:
                    values_to_plot = ds[f"state_{state_type}"].sel(qubit=qubit_name)
                else:
                    values_to_plot = ds[f"I_{state_type}"].sel(qubit=qubit_name)

                values_to_plot = values_to_plot.assign_coords(
                    {
                        "frequency_MHz": values_to_plot.frequency_Hz / 1e6,
                        "coupler_amp_mV": 1e3 * values_to_plot.coupler_amp_V,
                    }
                )
                values_to_plot.plot(
                    ax=ax,
                    cmap="viridis",
                    x="frequency_MHz",
                    y="coupler_amp_mV",
                )
            except Exception as e:
                print(f"[WARN] Plot failed for {qubit_name}: {e}")
                ax.set_title(f"{qubit_name} (plot failed)")
                continue

            ax.set_xlabel("Frequency [MHz]")
            ax.set_ylabel("Coupler amplitude [mV]")
            ax.set_title(qubit_name, fontsize=9)

        grid.fig.suptitle(f"{state_type.capitalize()} Qubit — Cz_Parametric", y=0.97, fontsize=12, weight="bold")
        plt.tight_layout()
        plt.show()
        node.results[f"figure_{state_type}"] = grid.fig

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()
# %%