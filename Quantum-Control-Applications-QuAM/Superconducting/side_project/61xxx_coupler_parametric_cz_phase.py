# %% {Imports}
from copy import copy

from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from quam_libs.lib.fit import fit_oscillation, fix_oscillation_phi_2pi
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names

# %% {Description}
description = """
        PARAMETRIC CZ COUPLER CALIBRATION WITH LEAKAGE AND PHASE

2D sweep of Cz_Parametric coupler amplitude and modulation frequency (relative to
the current gate settings). At each grid point:

1. Leakage: prepare |ee>, apply Cz_Parametric, measure populations.
2. Phase: Ramsey-style frame scan with control |g>/|e>, extract conditional phase.

Analysis finds the optimal point via weighted score:
    w_state * |11 state error| + w_phase * |phase_diff - 0.5|

State update:
    - Cz_Parametric coupler_flux_pulse.amplitude
    - Cz_Parametric coupler_flux_pulse.frequency
"""

# %% {Node_parameters}
qubit_pair_indexes = [3]


class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s" % (i, i + 1) for i in qubit_pair_indexes]
    num_averages: int = 200
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal["active", "thermal"] = "active"
    simulate: bool = False
    timeout: int = 200
    load_data_id: Optional[int] = None

    coupler_amp_range: float = 0.5
    coupler_amp_step: float = 0.01
    freq_range: float = 0.1
    freq_step: float = 0.005
    num_frames: int = 10
    operation: Literal["Cz_Parametric"] = "Cz_Parametric"
    score_weight_state: float = 1.0
    score_weight_phase: float = 1.0
    con_tar_flip: bool = False


node = QualibrationNode(name="61xxx_coupler_parametric_cz_phase", parameters=Parameters())
assert not (
    node.parameters.simulate and node.parameters.load_data_id is not None
), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)
machine = QuAM.load()

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
frames = np.arange(0, 1, 1 / node.parameters.num_frames)

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

with program() as parametric_cz_leakage_phase:
    n = declare(int)
    amp_coupler = declare(fixed)
    freq_idx = declare(int)
    frame = declare(fixed)
    control_initial = declare(int)
    n_st = declare_stream()

    leak_state_control = [declare(int) for _ in range(num_qubit_pairs)]
    leak_state_target = [declare(int) for _ in range(num_qubit_pairs)]
    leak_state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    leak_state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]

    phase_state_control = [declare(int) for _ in range(num_qubit_pairs)]
    phase_state_target = [declare(int) for _ in range(num_qubit_pairs)]
    phase_state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    phase_state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]

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
                    # --- Leakage / |11> retention ---
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
                    align()
                    wait(20)
                    if not node.parameters.con_tar_flip:
                        readout_state(qp.qubit_control, leak_state_control[i])
                        readout_state_gef(qp.qubit_target, leak_state_target[i])
                    else:
                        readout_state_gef(qp.qubit_control, leak_state_control[i])
                        readout_state(qp.qubit_target, leak_state_target[i])
                    save(leak_state_control[i], leak_state_st_control[i])
                    save(leak_state_target[i], leak_state_st_target[i])

                    # --- Phase measurement (Ramsey-style) ---
                    with for_(*from_array(frame, frames)):
                        with for_(*from_array(control_initial, [0, 1])):
                            if not node.parameters.simulate:
                                if node.parameters.reset_type == "active":
                                    active_reset(qp.qubit_control)
                                    active_reset(qp.qubit_target)
                                    qp.align()
                                else:
                                    wait(qp.qubit_control.thermalization_time * u.ns)
                                    wait(qp.qubit_target.thermalization_time * u.ns)
                            align()
                            reset_frame(qp.qubit_target.xy.name)
                            reset_frame(qp.qubit_control.xy.name)
                            with if_(control_initial == 1):
                                qp.qubit_control.xy.play("x180")
                            qp.qubit_target.xy.play("x90")
                            align()
                            execute_cz_parametric(qp, labels, freq_idx, amp_coupler)
                            frame_rotation_2pi(frame, qp.qubit_target.xy.name)
                            qp.qubit_target.xy.play("x90")
                            readout_state(qp.qubit_target, phase_state_target[i])
                            readout_state(qp.qubit_control, phase_state_control[i])
                            save(phase_state_control[i], phase_state_st_control[i])
                            save(phase_state_target[i], phase_state_st_target[i])
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            leak_state_st_control[i].buffer(len(freq_scales)).buffer(len(coupler_amp_scales)).average().save(
                f"leak_state_control{i + 1}"
            )
            leak_state_st_target[i].buffer(len(freq_scales)).buffer(len(coupler_amp_scales)).average().save(
                f"leak_state_target{i + 1}"
            )
            phase_state_st_control[i].buffer(2).buffer(len(frames)).buffer(len(freq_scales)).buffer(
                len(coupler_amp_scales)
            ).average().save(f"phase_state_control{i + 1}")
            phase_state_st_target[i].buffer(2).buffer(len(frames)).buffer(len(freq_scales)).buffer(
                len(coupler_amp_scales)
            ).average().save(f"phase_state_target{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=30_000)
    job = qmm.simulate(config, parametric_cz_leakage_phase, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(parametric_cz_leakage_phase)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        leak_handles = {k: v for k, v in job.result_handles.items() if k.startswith("leak_") or k == "n"}
        phase_handles = {k: v for k, v in job.result_handles.items() if k.startswith("phase_") or k == "n"}
        ds_leak = fetch_results_as_xarray(
            leak_handles,
            qubit_pairs,
            {"freq_scale": freq_scales, "amp_coupler": coupler_amp_scales},
        )
        ds_phase = fetch_results_as_xarray(
            phase_handles,
            qubit_pairs,
            {
                "control_axis": [0, 1],
                "frame": frames,
                "freq_scale": freq_scales,
                "amp_coupler": coupler_amp_scales,
            },
        )
    else:
        ds_leak, machine, _, _ = load_dataset(node.parameters.load_data_id, target_filename="ds_leak")
        ds_phase, _, _, _ = load_dataset(node.parameters.load_data_id, target_filename="ds_phase")
        if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
            qubit_pairs = machine.active_qubit_pairs
        else:
            qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
        coupler_amp_scales = ds_leak.amp_coupler.values
        freq_scales = ds_leak.freq_scale.values
        frames = ds_phase.frame.values
        base_frequencies = {
            qp.name: float(qp.gates[operation_name].coupler_flux_pulse.frequency) for qp in qubit_pairs
        }
        base_amplitudes = {
            qp.name: float(qp.gates[operation_name].coupler_flux_pulse.amplitude) for qp in qubit_pairs
        }

    node.results = {"ds_leak": ds_leak, "ds_phase": ds_phase}

# %% {Data_processing}
if not node.parameters.simulate:

    def coupler_amp_full(qp, amp_coupler):
        return amp_coupler * qp.gates[operation_name].coupler_flux_pulse.amplitude

    def frequency_full(qp, freq_scale):
        return freq_scale * qp.gates[operation_name].coupler_flux_pulse.frequency

    if not base_frequencies:
        base_frequencies = {
            qp.name: float(qp.gates[operation_name].coupler_flux_pulse.frequency) for qp in qubit_pairs
        }
    if not base_amplitudes:
        base_amplitudes = {
            qp.name: float(qp.gates[operation_name].coupler_flux_pulse.amplitude) for qp in qubit_pairs
        }

    coupler_full = np.array([base_amplitudes[qp.name] * ds_leak.amp_coupler for qp in qubit_pairs])
    freq_full = np.array([base_frequencies[qp.name] * ds_leak.freq_scale for qp in qubit_pairs])
    ds_leak = ds_leak.assign_coords(
        {
            "coupler_amp_full": (["qubit", "amp_coupler"], coupler_full),
            "frequency_Hz": (["qubit", "freq_scale"], freq_full),
        }
    )
    ds_phase = ds_phase.assign_coords(
        {
            "coupler_amp_full": (["qubit", "amp_coupler"], coupler_full),
            "frequency_Hz": (["qubit", "freq_scale"], freq_full),
        }
    )
    node.results = {"ds_leak": ds_leak, "ds_phase": ds_phase}
    node.results["results"] = {}

    def _select_qp(ds, qp):
        qubit_key = qp.name if qp.name in ds.qubit.values else qp.id
        return ds.sel(qubit=qubit_key)

    def _argmin_2d(da):
        dim_to_pos = dict(zip(da.dims, np.unravel_index(int(np.nanargmin(da.values)), da.shape)))
        return (
            float(da.freq_scale.values[dim_to_pos["freq_scale"]]),
            float(da.amp_coupler.values[dim_to_pos["amp_coupler"]]),
        )

    def _value_at(da, freq_scale, amp_coupler):
        return float(
            da.sel(freq_scale=freq_scale, amp_coupler=amp_coupler, method="nearest").values
        )

    def _plot_coords(ds_qp):
        return {
            "frequency_MHz": ds_qp.frequency_Hz / 1e6,
            "coupler_amp_mV": 1e3 * ds_qp.coupler_amp_full,
        }

    def _mark_optimal(ax, ds_qp, res):
        opt_fs = res.get("freq_scale", np.nan)
        opt_ac = res.get("amp_coupler", np.nan)
        if not (np.isfinite(opt_fs) and np.isfinite(opt_ac)):
            return
        opt_f_mhz = float(ds_qp.frequency_Hz.sel(freq_scale=opt_fs, method="nearest").values) / 1e6
        opt_c_mV = 1e3 * float(ds_qp.coupler_amp_full.sel(amp_coupler=opt_ac, method="nearest").values)
        ax.plot(opt_f_mhz, opt_c_mV, marker="+", color="red", markersize=10, mew=2.0, label="Optimal")
        ax.axhline(opt_c_mV, color="red", lw=1.0, ls="--")
        ax.axvline(opt_f_mhz, color="red", lw=1.0, ls="--")

# %% {Data_analysis}
if not node.parameters.simulate:
    for qp in qubit_pairs:
        try:
            ds_leak_qp = _select_qp(ds_leak, qp)
            ds_phase_qp = _select_qp(ds_phase, qp)

            state_control = ds_leak_qp.leak_state_control.astype(float)
            state_target = ds_leak_qp.leak_state_target.astype(float)
            state_error = (np.abs(state_control - 1) + np.abs(state_target - 1)) / 2

            fit_data = fit_oscillation(ds_phase_qp.phase_state_target, "frame")
            phase = fix_oscillation_phi_2pi(fit_data)
            phase_diff = (phase.isel(control_axis=0) - phase.isel(control_axis=1)) % 1
            phase_error = np.abs(phase_diff - 0.5)

            w_state = node.parameters.score_weight_state
            w_phase = node.parameters.score_weight_phase
            state_error_norm = state_error / state_error.max()
            phase_error_norm = phase_error / phase_error.max()
            score = w_state * state_error_norm + w_phase * phase_error_norm
            opt_freq_scale, opt_amp_coupler = _argmin_2d(score)

            coupler_amp_full_opt = float(
                ds_leak_qp.coupler_amp_full.sel(amp_coupler=opt_amp_coupler, method="nearest").values
            )
            frequency_full_opt = float(
                ds_leak_qp.frequency_Hz.sel(freq_scale=opt_freq_scale, method="nearest").values
            )

            node.results["results"][qp.name] = {
                "freq_scale": opt_freq_scale,
                "amp_coupler": opt_amp_coupler,
                "coupler_amp_full": coupler_amp_full_opt,
                "frequency_full": frequency_full_opt,
                "state_control": _value_at(state_control, opt_freq_scale, opt_amp_coupler),
                "state_target": _value_at(state_target, opt_freq_scale, opt_amp_coupler),
                "phase_diff": _value_at(phase_diff, opt_freq_scale, opt_amp_coupler),
                "phase_error": _value_at(phase_error, opt_freq_scale, opt_amp_coupler),
                "score": _value_at(score, opt_freq_scale, opt_amp_coupler),
            }

            print(
                f"{qp.name}: optimal @ scale=({opt_freq_scale:.4f}, {opt_amp_coupler:.4f}), "
                f"freq={frequency_full_opt / 1e6:.4f} MHz, amp={coupler_amp_full_opt:.5f} V, "
                f"|11> error={_value_at(state_error, opt_freq_scale, opt_amp_coupler):.4f}, "
                f"phase_diff={node.results['results'][qp.name]['phase_diff']:.4f} "
                f"(weights: state={w_state}, phase={w_phase})"
            )
        except Exception as e:
            import traceback

            print(f"[WARN] Analysis failed for {qp.name}: {e}")
            traceback.print_exc()
            node.results["results"][qp.name] = {
                "freq_scale": np.nan,
                "amp_coupler": np.nan,
                "coupler_amp_full": np.nan,
                "frequency_full": np.nan,
                "state_control": np.nan,
                "state_target": np.nan,
                "phase_diff": np.nan,
                "phase_error": np.nan,
            }

# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)

    for state_var, qubit_attr, fig_key, title_suffix in [
        ("leak_state_control", "qubit_control", "figure_leak_control", "Leakage — control qubit state"),
        ("leak_state_target", "qubit_target", "figure_leak_target", "Leakage — target qubit state"),
    ]:
        grid = QubitPairGrid(grid_names, qubit_pair_names)
        for ax, qp_info in grid_iter(grid):
            pair_name = qp_info["qubit"]
            qp = machine.qubit_pairs[pair_name]
            qubit_label = getattr(qp, qubit_attr).name
            res = node.results["results"].get(pair_name, {})

            try:
                ds_qp = _select_qp(ds_leak, qp)
                plot_data = ds_qp[state_var].assign_coords(_plot_coords(ds_qp))
                mesh = plot_data.plot(
                    ax=ax,
                    cmap="viridis",
                    x="frequency_MHz",
                    y="coupler_amp_mV",
                    add_colorbar=False,
                )
                plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.15, aspect=30, label="State")
            except Exception as e:
                print(f"[WARN] Leak plot failed for {pair_name} ({qubit_label}): {e}")
                ax.set_title(f"{qubit_label} (plot failed)")
                continue

            # try:
            #     _mark_optimal(ax, ds_qp, res)
            # except Exception as e:
            #     print(f"[WARN] Leak annotations failed for {qubit_label}: {e}")

            ax.set_xlabel("Frequency [MHz]")
            ax.set_ylabel("Coupler amplitude [mV]")
            ax.set_title(f"{qubit_label}", fontsize=9)

        grid.fig.suptitle(title_suffix, y=0.97, fontsize=12, weight="bold")
        plt.tight_layout()
        plt.show()
        node.results[fig_key] = grid.fig

    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp_info in grid_iter(grid):
        pair_name = qp_info["qubit"]
        qp = machine.qubit_pairs[pair_name]
        res = node.results["results"].get(pair_name, {})

        try:
            ds_phase_qp = _select_qp(ds_phase, qp)
            fit_data = fit_oscillation(ds_phase_qp.phase_state_target, "frame")
            phase = fix_oscillation_phi_2pi(fit_data)
            phase_diff = (phase.isel(control_axis=0) - phase.isel(control_axis=1)) % 1
            phase_error = np.abs(phase_diff - 0.5)

            plot_data = phase_error.assign_coords(_plot_coords(ds_phase_qp))
            mesh = plot_data.plot(
                ax=ax,
                cmap="viridis",
                x="frequency_MHz",
                y="coupler_amp_mV",
                add_colorbar=False,
            )
            plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.15, aspect=30, label="|Phase diff - 0.5|")
        except Exception as e:
            print(f"[WARN] Phase plot failed for {pair_name}: {e}")
            ax.set_title(f"{pair_name} (plot failed)")
            continue

        # try:
        #     ds_leak_qp = _select_qp(ds_leak, qp)
        #     _mark_optimal(ax, ds_leak_qp, res)
        # except Exception as e:
        #     print(f"[WARN] Phase annotations failed for {pair_name}: {e}")

        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Coupler amplitude [mV]")
        ax.set_title(pair_name, fontsize=9)

    grid.fig.suptitle("Phase calibration — |phase diff - 0.5| (π)", y=0.97, fontsize=12, weight="bold")
    plt.tight_layout()
    plt.show()
    node.results["figure_phase"] = grid.fig

# %% {Update_state}
if not node.parameters.simulate:
    with node.record_state_updates():
        for qp in qubit_pairs:
            res = node.results["results"][qp.name]
            if np.isfinite(res["coupler_amp_full"]):
                qp.gates[operation_name].coupler_flux_pulse.amplitude = res["coupler_amp_full"]
            if np.isfinite(res["frequency_full"]):
                qp.gates[operation_name].coupler_flux_pulse.frequency = res["frequency_full"]

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%