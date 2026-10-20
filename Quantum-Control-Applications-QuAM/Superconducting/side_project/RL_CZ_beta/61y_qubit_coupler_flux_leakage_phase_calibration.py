# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef
from quam_libs.lib.save_utils import fetch_results_as_xarray, restore_load_data_id, resolve_qubit_pairs_from_node
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
        COUPLER-QUBIT FLUX CALIBRATION WITH LEAKAGE AND PHASE

Performs a 2D sweep of CZ gate coupler and qubit flux amplitudes (scaled around
the current gate settings using amp_range / amp_num_points). At each grid point:

1. Leakage: prepare |11>, apply CZ. GEF is auto-selected from idle frequencies
   (higher-frequency qubit, 61xx rule): control < target → GEF on target;
   control > target → GEF on control. Stream P(|f>) and P(|11> remains).
2. Phase (optional, meas_phase=True): Ramsey-style frame scan with control |g>/|e>,
   extract conditional phase.

Analysis finds a single optimal point via weighted score:
    w_leak * leak_error + w_phase * |phase_diff - 0.5|
    leak_error = (1 - P(|11>)) + P(|f>)
If meas_phase is False, optimize leakage only (phase terms skipped).

State update:
    - Cz gate coupler_flux_pulse.amplitude
    - Cz gate flux_pulse_control.amplitude
"""

# %% {Node_parameters}
qubit_pair_indexes = [4]


class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s" % (i, i + 1) for i in qubit_pair_indexes]
    num_averages: int = 300
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal["active", "thermal"] = "active"
    simulate: bool = False
    timeout: int = 200
    load_data_id: Optional[int] = None

    qubit_amp_range: float = 0.9999
    coupler_amp_range: float = 0.4
    qubit_amp_num_points: int = 60
    coupler_amp_num_points: int = 50
    num_frames: int = 10
    operation: Literal["composed_Cz"] = "composed_Cz"
    score_weight_control: float = 1.0
    score_weight_phase: float = 3.0
    meas_phase: bool = 1 # If False, skip phase measurement / analysis / plotting


node = QualibrationNode(
    name="61y_qubit_coupler_flux_leakage_phase_calibration", parameters=Parameters()
)
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


def gef_on_target_qubit(qp):
    """GEF the higher-frequency qubit (61xx frequency rule).

    control < target → GEF on target (former con_tar_flip=True)
    control > target → GEF on control (former con_tar_flip=False)
    """
    f_c = float(qp.qubit_control.xy.RF_frequency)
    f_t = float(qp.qubit_target.xy.RF_frequency)
    on_target = f_c < f_t
    gef_qubit = qp.qubit_target if on_target else qp.qubit_control
    print(
        f"{qp.name}: f_control={qp.qubit_control.name} {f_c / 1e9:.4f} GHz, "
        f"f_target={qp.qubit_target.name} {f_t / 1e9:.4f} GHz → "
        f"GEF on {gef_qubit.name} ({'target' if on_target else 'control'})"
    )
    return on_target


gef_on_target = {qp.name: gef_on_target_qubit(qp) for qp in qubit_pairs}

config = machine.generate_config()
octave_config = machine.get_octave_config()
if node.parameters.load_data_id is None:
    qmm = machine.connect(timeout=1500)

# %% {QUA_program}
n_avg = node.parameters.num_averages
flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise

coupler_amp_scales = np.linspace(
    1 - node.parameters.coupler_amp_range,
    1 + node.parameters.coupler_amp_range,
    node.parameters.coupler_amp_num_points,
)
qubit_amp_scales = np.linspace(
    1 - node.parameters.qubit_amp_range,
    1 + node.parameters.qubit_amp_range,
    node.parameters.qubit_amp_num_points,
)
frames = np.arange(0, 1, 1 / node.parameters.num_frames)

with program() as coupler_leakage_phase_calibration:
    n = declare(int)
    amp_coupler = declare(fixed)
    amp_qubit = declare(fixed)
    n_st = declare_stream()
    if node.parameters.meas_phase:
        frame = declare(fixed)
        control_initial = declare(int)

    leak_state_control = [declare(int) for _ in range(num_qubit_pairs)]
    leak_state_target = [declare(int) for _ in range(num_qubit_pairs)]
    leak_state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    leak_state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    leak_pf = [declare(int) for _ in range(num_qubit_pairs)]
    leak_ee = [declare(int) for _ in range(num_qubit_pairs)]
    leak_pf_st = [declare_stream() for _ in range(num_qubit_pairs)]
    leak_ee_st = [declare_stream() for _ in range(num_qubit_pairs)]

    if node.parameters.meas_phase:
        phase_state_control = [declare(int) for _ in range(num_qubit_pairs)]
        phase_state_target = [declare(int) for _ in range(num_qubit_pairs)]
        phase_state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        phase_state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]

    for i, qp in enumerate(qubit_pairs):
        qp.gates[operation_name].phase_shift_control = 0.0
        qp.gates[operation_name].phase_shift_target = 0.0
        machine.set_all_fluxes(flux_point, qp)
        qp.coupler.to_decouple_idle()
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(amp_coupler, coupler_amp_scales)):
                with for_(*from_array(amp_qubit, qubit_amp_scales)):
                    # --- Leakage measurement ---
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
                    qp.align()
                    qp.gates[operation_name].execute(
                        amplitude_scale=amp_qubit,
                        coupler_amplitude_scale=amp_coupler,
                    )
                    align()
                    wait(20)
                    if gef_on_target[qp.name]:
                        readout_state(qp.qubit_control, leak_state_control[i])
                        readout_state_gef(qp.qubit_target, leak_state_target[i])
                        gef_state = leak_state_target[i]
                        partner_state = leak_state_control[i]
                    else:
                        readout_state_gef(qp.qubit_control, leak_state_control[i])
                        readout_state(qp.qubit_target, leak_state_target[i])
                        gef_state = leak_state_control[i]
                        partner_state = leak_state_target[i]
                    save(leak_state_control[i], leak_state_st_control[i])
                    save(leak_state_target[i], leak_state_st_target[i])
                    with if_(gef_state == 2):
                        assign(leak_pf[i], 1)
                    with else_():
                        assign(leak_pf[i], 0)
                    with if_((gef_state == 1) & (partner_state == 1)):
                        assign(leak_ee[i], 1)
                    with else_():
                        assign(leak_ee[i], 0)
                    save(leak_pf[i], leak_pf_st[i])
                    save(leak_ee[i], leak_ee_st[i])

                    # --- Phase measurement (Ramsey-style) ---
                    if node.parameters.meas_phase:
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
                                qp.align()
                                reset_frame(qp.qubit_target.xy.name)
                                reset_frame(qp.qubit_control.xy.name)
                                with if_(control_initial == 1):
                                    qp.qubit_control.xy.play("x180")
                                qp.qubit_target.xy.play("x90")
                                qp.align()
                                qp.gates[operation_name].execute(
                                    amplitude_scale=amp_qubit,
                                    coupler_amplitude_scale=amp_coupler,
                                )
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
            leak_state_st_control[i].buffer(len(qubit_amp_scales)).buffer(len(coupler_amp_scales)).average().save(
                f"leak_state_control{i + 1}"
            )
            leak_state_st_target[i].buffer(len(qubit_amp_scales)).buffer(len(coupler_amp_scales)).average().save(
                f"leak_state_target{i + 1}"
            )
            leak_pf_st[i].buffer(len(qubit_amp_scales)).buffer(len(coupler_amp_scales)).average().save(
                f"leak_pf{i + 1}"
            )
            leak_ee_st[i].buffer(len(qubit_amp_scales)).buffer(len(coupler_amp_scales)).average().save(
                f"leak_ee{i + 1}"
            )
            if node.parameters.meas_phase:
                phase_state_st_control[i].buffer(2).buffer(len(frames)).buffer(len(qubit_amp_scales)).buffer(
                    len(coupler_amp_scales)
                ).average().save(f"phase_state_control{i + 1}")
                phase_state_st_target[i].buffer(2).buffer(len(frames)).buffer(len(qubit_amp_scales)).buffer(
                    len(coupler_amp_scales)
                ).average().save(f"phase_state_target{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=30_000)
    job = qmm.simulate(config, coupler_leakage_phase_calibration, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(coupler_leakage_phase_calibration)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        leak_handles = {k: v for k, v in job.result_handles.items() if k.startswith("leak_") or k == "n"}
        ds_leak = fetch_results_as_xarray(
            leak_handles,
            qubit_pairs,
            {"amp_qubit": qubit_amp_scales, "amp_coupler": coupler_amp_scales},
        )
        ds_phase = None
        if node.parameters.meas_phase:
            phase_handles = {k: v for k, v in job.result_handles.items() if k.startswith("phase_") or k == "n"}
            ds_phase = fetch_results_as_xarray(
                phase_handles,
                qubit_pairs,
                {
                    "control_axis": [0, 1],
                    "frame": frames,
                    "amp_qubit": qubit_amp_scales,
                    "amp_coupler": coupler_amp_scales,
                },
            )
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds_leak = node.results["ds_leak"]
        ds_phase = node.results.get("ds_phase")
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
        gef_on_target = {qp.name: gef_on_target_qubit(qp) for qp in qubit_pairs}
        coupler_amp_scales = ds_leak.amp_coupler.values
        qubit_amp_scales = ds_leak.amp_qubit.values
        if ds_phase is not None and "frame" in ds_phase.coords:
            frames = ds_phase.frame.values

    node.results = {"ds_leak": ds_leak, "gef_on_target": {k: bool(v) for k, v in gef_on_target.items()}}
    if ds_phase is not None:
        node.results["ds_phase"] = ds_phase

# %% {Data_processing}
if not node.parameters.simulate:

    def coupler_amp_full(qp, amp_coupler):
        return amp_coupler * qp.gates[operation_name].coupler_flux_pulse.amplitude

    def qubit_amp_full(qp, amp_qubit):
        return amp_qubit * qp.gates[operation_name].flux_pulse_control.amplitude

    coupler_full = np.array([coupler_amp_full(qp, ds_leak.amp_coupler) for qp in qubit_pairs])
    qubit_full = np.array([qubit_amp_full(qp, ds_leak.amp_qubit) for qp in qubit_pairs])
    ds_leak = ds_leak.assign_coords(
        {
            "coupler_amp_full": (["qubit", "amp_coupler"], coupler_full),
            "qubit_amp_full": (["qubit", "amp_qubit"], qubit_full),
        }
    )
    node.results = {
        "ds_leak": ds_leak,
        "gef_on_target": {k: bool(v) for k, v in gef_on_target.items()},
    }
    if ds_phase is not None:
        ds_phase = ds_phase.assign_coords(
            {
                "coupler_amp_full": (["qubit", "amp_coupler"], coupler_full),
                "qubit_amp_full": (["qubit", "amp_qubit"], qubit_full),
            }
        )
        node.results["ds_phase"] = ds_phase
    node.results["results"] = {}

    def _select_qp(ds, qp):
        qubit_key = qp.name if qp.name in ds.qubit.values else qp.id
        return ds.sel(qubit=qubit_key)

    def _argmin_2d(da):
        dim_to_pos = dict(zip(da.dims, np.unravel_index(int(np.nanargmin(da.values)), da.shape)))
        return (
            float(da.amp_qubit.values[dim_to_pos["amp_qubit"]]),
            float(da.amp_coupler.values[dim_to_pos["amp_coupler"]]),
        )

    def _value_at(da, amp_qubit, amp_coupler):
        return float(
            da.sel(amp_qubit=amp_qubit, amp_coupler=amp_coupler, method="nearest").values
        )

    def _plot_coords(ds_qp):
        if "qubit_amp_full" in ds_qp.coords and "coupler_amp_full" in ds_qp.coords:
            return {
                "qubit_amp_mV": 1e3 * ds_qp.qubit_amp_full,
                "coupler_amp_mV": 1e3 * ds_qp.coupler_amp_full,
            }
        return {
            "qubit_amp_mV": 1e3 * ds_qp.amp_qubit,
            "coupler_amp_mV": 1e3 * ds_qp.amp_coupler,
        }

    def _mark_origin(ax, qp):
        origin_q_amp = 1e3 * qp.gates[operation_name].flux_pulse_control.amplitude
        origin_c_amp = 1e3 * qp.gates[operation_name].coupler_flux_pulse.amplitude
        ax.plot(origin_q_amp, origin_c_amp, marker="+", color="gray", markersize=10, mew=2.0, label="Original")
        ax.axhline(origin_c_amp, color="gray", lw=1.0, ls="--")
        ax.axvline(origin_q_amp, color="gray", lw=1.0, ls="--")

    def _mark_optimal(ax, ds_qp, res, manual_indicate_coords_mV:tuple[float,float]|None=None):
        opt_aq = res.get("amp_qubit", np.nan)
        opt_ac = res.get("amp_coupler", np.nan)
        if not (np.isfinite(opt_aq) and np.isfinite(opt_ac)):
            return
        if "qubit_amp_full" in ds_qp.coords:
            opt_q_mV = 1e3 * float(ds_qp.qubit_amp_full.sel(amp_qubit=opt_aq, method="nearest").values)
            opt_c_mV = 1e3 * float(ds_qp.coupler_amp_full.sel(amp_coupler=opt_ac, method="nearest").values)
        else:
            opt_q_mV = 1e3 * opt_aq
            opt_c_mV = 1e3 * opt_ac

        if manual_indicate_coords_mV is not None:
            opt_q_mV, opt_c_mV = manual_indicate_coords_mV

        ax.plot(opt_q_mV, opt_c_mV, marker="+", color="red", markersize=10, mew=2.0, label="Optimal")
        ax.axhline(opt_c_mV, color="red", lw=1.0, ls="--")
        ax.axvline(opt_q_mV, color="red", lw=1.0, ls="--")

    def _safe_norm(da):
        peak = float(np.nanmax(np.asarray(da.values, dtype=float)))
        if not np.isfinite(peak) or peak <= 0:
            return da * 0.0
        return da / peak

# %% {Data_analysis}
if not node.parameters.simulate:
    for qp in qubit_pairs:
        try:
            ds_leak_qp = _select_qp(ds_leak, qp)

            state_control = ds_leak_qp.leak_state_control.astype(float)
            state_target = ds_leak_qp.leak_state_target.astype(float)
            if "leak_pf" in ds_leak_qp and "leak_ee" in ds_leak_qp:
                p_f = ds_leak_qp.leak_pf.astype(float)
                p_11 = ds_leak_qp.leak_ee.astype(float)
            else:
                raise ValueError(
                    f"{qp.name}: missing leak_pf/leak_ee (re-run 61y; old snapshots only stored mean state)"
                )
            leak_error = (1.0 - p_11) + p_f

            w_control = node.parameters.score_weight_control
            w_phase = node.parameters.score_weight_phase
            leak_error_norm = _safe_norm(leak_error)

            phase_diff = None
            phase_error = None
            if node.parameters.meas_phase and ds_phase is not None:
                ds_phase_qp = _select_qp(ds_phase, qp)
                fit_data = fit_oscillation(ds_phase_qp.phase_state_target, "frame")
                phase = fix_oscillation_phi_2pi(fit_data)
                phase_diff = (phase.isel(control_axis=0) - phase.isel(control_axis=1)) % 1
                phase_error = np.abs(phase_diff - 0.5)
                phase_error_norm = _safe_norm(phase_error)
                score = w_control * leak_error_norm + w_phase * phase_error_norm
            else:
                score = leak_error_norm

            opt_amp_qubit, opt_amp_coupler = _argmin_2d(score)

            qubit_amp_full_opt = float(
                ds_leak_qp.qubit_amp_full.sel(amp_qubit=opt_amp_qubit, method="nearest").values
            )
            coupler_amp_full_opt = float(
                ds_leak_qp.coupler_amp_full.sel(amp_coupler=opt_amp_coupler, method="nearest").values
            )

            result = {
                "amp_qubit": opt_amp_qubit,
                "amp_coupler": opt_amp_coupler,
                "qubit_amp_full": qubit_amp_full_opt,
                "coupler_amp_full": coupler_amp_full_opt,
                "state_control": _value_at(state_control, opt_amp_qubit, opt_amp_coupler),
                "state_target": _value_at(state_target, opt_amp_qubit, opt_amp_coupler),
                "p_f": _value_at(p_f, opt_amp_qubit, opt_amp_coupler),
                "p_11": _value_at(p_11, opt_amp_qubit, opt_amp_coupler),
                "leak_error": _value_at(leak_error, opt_amp_qubit, opt_amp_coupler),
                "gef_on_target": bool(gef_on_target[qp.name]),
                "gef_qubit": qp.qubit_target.name if gef_on_target[qp.name] else qp.qubit_control.name,
                "score": _value_at(score, opt_amp_qubit, opt_amp_coupler),
            }
            if phase_diff is not None and phase_error is not None:
                result["phase_diff"] = _value_at(phase_diff, opt_amp_qubit, opt_amp_coupler)
                result["phase_error"] = _value_at(phase_error, opt_amp_qubit, opt_amp_coupler)
            else:
                result["phase_diff"] = np.nan
                result["phase_error"] = np.nan
            node.results["results"][qp.name] = result

            phase_msg = (
                f"phase_diff={result['phase_diff']:.4f} "
                if np.isfinite(result["phase_diff"])
                else "phase=skipped "
            )
            print(
                f"{qp.name}: optimal @ scale=({opt_amp_qubit:.4f}, {opt_amp_coupler:.4f}), "
                f"amp=({qubit_amp_full_opt:.5f} V, {coupler_amp_full_opt:.5f} V), "
                f"P(f)={result['p_f']:.4f}, P(11)={result['p_11']:.4f}, "
                f"GEF={result['gef_qubit']}, "
                f"{phase_msg}"
                f"(weights: leak={w_control}, phase={w_phase if node.parameters.meas_phase else 0})"
            )
        except Exception as e:
            import traceback

            print(f"[WARN] Analysis failed for {qp.name}: {e}")
            traceback.print_exc()
            node.results["results"][qp.name] = {
                "amp_qubit": np.nan,
                "amp_coupler": np.nan,
                "qubit_amp_full": np.nan,
                "coupler_amp_full": np.nan,
                "state_control": np.nan,
                "state_target": np.nan,
                "p_f": np.nan,
                "p_11": np.nan,
                "leak_error": np.nan,
                "gef_on_target": bool(gef_on_target.get(qp.name, False)),
                "gef_qubit": "",
                "phase_diff": np.nan,
                "phase_error": np.nan,
            }

# %% {Plotting}
manual_indicate_coords_mV = {
    "coupler_q4_q5": None
}



if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)

    for data_var, fig_key, title_suffix, cbar_label in [
        ("leak_pf", "figure_leak_f", "Leakage — P(|f⟩)", "P(|f⟩)"),
        ("leak_ee", "figure_p11", "Leakage — P(|11⟩ remains)", "P(|11⟩)"),
    ]:
        grid = QubitPairGrid(grid_names, qubit_pair_names)
        for ax, qp_info in grid_iter(grid):
            pair_name = qp_info["qubit"]
            qp = machine.qubit_pairs[pair_name]
            res = node.results["results"].get(pair_name, {})
            gef_label = res.get("gef_qubit") or (
                qp.qubit_target.name if gef_on_target.get(pair_name, False) else qp.qubit_control.name
            )

            try:
                ds_qp = _select_qp(ds_leak, qp)
                plot_data = ds_qp[data_var].assign_coords(_plot_coords(ds_qp))
                mesh = plot_data.plot(
                    ax=ax,
                    cmap="viridis",
                    x="qubit_amp_mV",
                    y="coupler_amp_mV",
                    add_colorbar=False,
                )
                plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.15, aspect=30, label=cbar_label)
            except Exception as e:
                print(f"[WARN] Leak plot failed for {pair_name} ({data_var}): {e}")
                ax.set_title(f"{pair_name} (plot failed)")
                continue

            try:
                _mark_optimal(ax, ds_qp, res, manual_indicate_coords_mV.get(pair_name))
                _mark_origin(ax, qp)
            except Exception as e:
                print(f"[WARN] Leak annotations failed for {pair_name}: {e}")

            ax.set_xlabel("Qubit flux amplitude [mV]")
            ax.set_ylabel("Coupler flux amplitude [mV]")
            ax.set_title(f"{pair_name}  GEF={gef_label}", fontsize=9)
            ax.legend()

        grid.fig.suptitle(title_suffix, y=0.97, fontsize=12, weight="bold")
        plt.tight_layout()
        plt.show()
        node.results[fig_key] = grid.fig

        if manual_indicate_coords_mV.get(pair_name) is not None:
            node.results["results"][pair_name]["coupler_amp_full"] = manual_indicate_coords_mV.get(pair_name)[1] / 1e3
            node.results["results"][pair_name]["qubit_amp_full"] = manual_indicate_coords_mV.get(pair_name)[0] / 1e3

    if node.parameters.meas_phase and ds_phase is not None:
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
                    x="qubit_amp_mV",
                    y="coupler_amp_mV",
                    add_colorbar=False,
                )
                plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.15, aspect=30, label="|Phase diff - 0.5|")
            except Exception as e:
                print(f"[WARN] Phase plot failed for {pair_name}: {e}")
                ax.set_title(f"{pair_name} (plot failed)")
                continue

            try:
                ds_leak_qp = _select_qp(ds_leak, qp)
                _mark_optimal(ax, ds_leak_qp, res, manual_indicate_coords_mV.get(pair_name))
                _mark_origin(ax, qp)
            except Exception as e:
                print(f"[WARN] Phase annotations failed for {pair_name}: {e}")

            ax.legend()
            ax.set_xlabel("Qubit flux amplitude [mV]")
            ax.set_ylabel("Coupler flux amplitude [mV]")
            ax.set_title(pair_name, fontsize=9)

        grid.fig.suptitle("Phase calibration — |phase diff - 0.5|", y=0.97, fontsize=12, weight="bold")
        plt.tight_layout()
        plt.show()
        node.results["figure_phase"] = grid.fig


# %% {Update_state}
if not node.parameters.simulate and node.parameters.load_data_id is None:
    with node.record_state_updates():
        for qp in qubit_pairs:
            res = node.results["results"][qp.name]
            if np.isfinite(res["coupler_amp_full"]):
                qp.gates[operation_name].coupler_flux_pulse.amplitude = res["coupler_amp_full"]
            if np.isfinite(res["qubit_amp_full"]):
                qp.gates[operation_name].flux_pulse_control.amplitude = res["qubit_amp_full"]

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()
# %%