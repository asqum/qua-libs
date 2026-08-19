"""
        T1 BAYESIAN FAST TRACKING (u = 1/k)
Real-time adaptive Bayesian estimation of qubit T1, following Berritta et al.,
Phys. Rev. X (2026), arXiv:2506.09576.

Uses u = 1/k reparameterization so posterior shape k can grow beyond the QUA fixed
range limit (~7), enabling credible intervals to shrink (~ 1/sqrt(k)).

Prerequisites:
    - Calibrated readout and state discrimination (07b_IQ_Blobs).
    - Calibrated x180 pulse (04_Power_Rabi).
    - Working active reset.
    - (optional) confusion_matrix in QuAM for SPAM correction.

Next steps:
    - Inspect T1 vs laboratory time trace and posterior evolution plots.
    - Does not write qubit.T1 to state (tracking node).
"""

# %% {Imports}
import logging
import time
from typing import Literal, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.experiments.t1_bayesian import (
    MS_TO_CLK_INT,
    MS_TO_US,
    compute_welch_and_allan,
    fetch_results_as_xarray,
    fetch_t1_datasets,
    plot_bayesian_results,
    posterior_t1_credible_interval,
    resolve_confusion_alpha_beta,
    resolve_t1_prior_us_per_qubit,
)
from quam_libs.experiments.t1_bayesian.analysis import US_TO_MS
from quam_libs.lib.save_utils import restore_load_data_id, resolve_qubits_from_node
from quam_libs.macros import active_reset, active_reset_simple, qua_declaration, readout_state

logger = logging.getLogger(__name__)
u = unit(coerce_to_integer=True)


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q3"]
    num_repetitions: int = 100
    """Number of independent T1 estimation blocks (time-trace length)."""
    num_probes: int = 100
    """Number of adaptive single-shot probes per estimation block."""
    c: float = 0.51
    """Adaptive waiting-time coefficient: tau = c * T1_est."""
    k0: float = 1.0
    """Initial gamma shape parameter for Gamma_1 prior."""
    t1_prior_us: float = 35.0
    """Prior mean T1 in µs."""
    use_quam_t1_prior: bool = False
    """If True and qubit.T1 is set in QuAM, use it instead of t1_prior_us."""
    t1_min_us: float = 1.0
    t1_max_us: float = 100.0
    interleaved_validation: bool = True
    """Add a non-adaptive T1 shot after each adaptive probe."""
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 200_000
    keep_shot_data: bool = True
    credible_interval: float = 0.90
    k_max: float = 100.0
    k_min: float = 0.2
    active_reset_per_probe: bool = False
    reset_type: Literal["active", "active_simple", "thermal"] = "active"
    reset_max_attempts: int = 15
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 600
    load_data_id: Optional[int] = None


node = QualibrationNode(name="05x_T1_fast_tracking", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
machine = QuAM.load()
node.machine = machine
config = machine.generate_config()

qubit_list = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubit_list)
flux_point = node.parameters.flux_point_joint_or_independent

if node.parameters.load_data_id is None:
    qmm = machine.connect()

n_reps = node.parameters.num_repetitions
n_probes = node.parameters.num_probes
k0 = float(node.parameters.k0)
t1_prior_us_list = resolve_t1_prior_us_per_qubit(node.parameters, qubit_list)
t1_prior_ms_list = [t1_us * US_TO_MS for t1_us in t1_prior_us_list]
t1_prior_by_name = dict(zip([q.name for q in qubit_list], t1_prior_us_list))
t1_min_ms = float(node.parameters.t1_min_us) * US_TO_MS
t1_max_ms = float(node.parameters.t1_max_us) * US_TO_MS
tau_min_ms = 0.001
tau_max_ms = t1_max_ms
c_adaptive = float(node.parameters.c)
interleaved = node.parameters.interleaved_validation
active_reset_per_probe = node.parameters.active_reset_per_probe

confusion_alpha = [resolve_confusion_alpha_beta(q)[0] for q in qubit_list]
confusion_beta = [resolve_confusion_alpha_beta(q)[1] for q in qubit_list]

u_max_safe = (7.5 - 1.0) / c_adaptive
k_min_val = max(float(node.parameters.k_min), 1.0 / u_max_safe)
k_max_val = float(node.parameters.k_max)
u_min = 1.0 / k_max_val
u_max = 1.0 / k_min_val
u0 = 1.0 / k0

lin_times_clocks = np.linspace(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    n_probes,
    dtype=int,
)
lin_times_clocks = np.maximum(lin_times_clocks, 4)


# %% {QUA_program}
with program() as t1_bayesian:
    _, _, _, _, n, n_st = qua_declaration(num_qubits=num_qubits)
    rep = declare(int)
    probe = declare(int)

    state = [declare(int) for _ in range(num_qubits)]
    state_lin = [declare(int) for _ in range(num_qubits)]

    u_inv_k = [declare(fixed) for _ in range(num_qubits)]
    t1_est = [declare(fixed) for _ in range(num_qubits)]
    alpha = [declare(fixed) for _ in range(num_qubits)]
    beta = [declare(fixed) for _ in range(num_qubits)]

    tau_ms = [declare(fixed) for _ in range(num_qubits)]
    tau_clocks = [declare(int) for _ in range(num_qubits)]

    z = declare(fixed)
    z2 = declare(fixed)
    phi = declare(fixed)
    lexp = declare(fixed)
    r = declare(fixed)
    r_pow_k = declare(fixed)
    r_pow_k1 = declare(fixed)
    r_pow_k2 = declare(fixed)
    spam = declare(fixed)
    num_k = declare(fixed)
    den_k = declare(fixed)
    num_k1 = declare(fixed)
    den_k1 = declare(fixed)
    ratio_k = declare(fixed)
    ratio_k1 = declare(fixed)
    ratio_ratio = declare(fixed)

    lin_tau = declare(int, value=lin_times_clocks.tolist())

    estimated_t1_st = [declare_stream() for _ in range(num_qubits)]
    u_final_st = [declare_stream() for _ in range(num_qubits)]
    u_evol_st = [declare_stream() for _ in range(num_qubits)]
    t1_evol_st = [declare_stream() for _ in range(num_qubits)]
    time_stamp_st = [declare_stream() for _ in range(num_qubits)]

    state_st = [declare_stream() for _ in range(num_qubits)]
    tau_st = [declare_stream() for _ in range(num_qubits)]
    state_lin_st = [declare_stream() for _ in range(num_qubits)]

    if not node.parameters.simulate:
        machine.apply_all_couplers_to_min()

    for i, qubit in enumerate(qubit_list):
        assign(alpha[i], confusion_alpha[i])
        assign(beta[i], confusion_beta[i])

        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id:
                qubit.z.set_dc_offset(qubit.z.joint_offset)
            qubit.z.settle()
        qubit.align()

        with for_(rep, 0, rep < n_reps, rep + 1):
            if i == 0:
                save(rep, n_st)
            assign(u_inv_k[i], u0)
            assign(t1_est[i], t1_prior_ms_list[i])

            if node.parameters.reset_type == "active":
                active_reset(qubit, readout_pulse_name="readout", max_attempts=node.parameters.reset_max_attempts)
            elif node.parameters.reset_type == "active_simple":
                active_reset_simple(qubit, readout_pulse_name="readout")
            else:
                qubit.resonator.wait(qubit.thermalization_time * u.ns)
            qubit.align()

            with for_(probe, 0, probe < n_probes, probe + 1):
                with if_(rep == n_reps - 1):
                    save(u_inv_k[i], u_evol_st[i])
                    save(t1_est[i], t1_evol_st[i])

                assign(tau_ms[i], c_adaptive * t1_est[i])
                with if_(tau_ms[i] < tau_min_ms):
                    assign(tau_ms[i], tau_min_ms)
                with if_(tau_ms[i] > tau_max_ms):
                    assign(tau_ms[i], tau_max_ms)
                assign(tau_clocks[i], Cast.mul_int_by_fixed(MS_TO_CLK_INT, tau_ms[i]))
                with if_(tau_clocks[i] < 4):
                    assign(tau_clocks[i], 4)

                qubit.xy.play("x180")
                qubit.align()
                qubit.wait(tau_clocks[i])
                qubit.align()
                readout_state(qubit, state[i])
                qubit.align()
                qubit.xy.play("x180", condition=Cast.to_bool(state[i]))
                if node.parameters.keep_shot_data:
                    save(state[i], state_st[i])
                    save(tau_ms[i], tau_st[i])
                align()

                assign(spam, 1.0 - alpha[i] - beta[i])
                assign(z, c_adaptive * u_inv_k[i])
                assign(r, Math.inv(1.0 + z))
                with if_(z < 0.2):
                    assign(z2, z * z)
                    assign(phi, 1.0 - 0.5 * z + z2 * (1.0 / 3.0) - (z2 * z) * (1.0 / 4.0))
                with else_():
                    assign(phi, Math.div(Math.ln(1.0 + z), z))
                assign(lexp, -c_adaptive * phi)
                assign(r_pow_k, Math.exp(lexp))
                assign(r_pow_k1, r_pow_k * r)
                assign(r_pow_k2, r_pow_k1 * r)

                with if_(state[i] == 0):
                    assign(num_k, 1.0 - beta[i] - spam * r_pow_k1)
                    assign(den_k, 1.0 - beta[i] - spam * r_pow_k)
                    assign(num_k1, 1.0 - beta[i] - spam * r_pow_k2)
                    assign(den_k1, 1.0 - beta[i] - spam * r_pow_k1)
                with else_():
                    assign(num_k, beta[i] + spam * r_pow_k1)
                    assign(den_k, beta[i] + spam * r_pow_k)
                    assign(num_k1, beta[i] + spam * r_pow_k2)
                    assign(den_k1, beta[i] + spam * r_pow_k1)

                assign(ratio_k, Math.div(num_k, den_k))
                assign(ratio_k1, Math.div(num_k1, den_k1))

                assign(t1_est[i], Math.div(t1_est[i], ratio_k))
                with if_(t1_est[i] < t1_min_ms):
                    assign(t1_est[i], t1_min_ms)
                with if_(t1_est[i] > t1_max_ms):
                    assign(t1_est[i], t1_max_ms)

                assign(ratio_ratio, Math.div(ratio_k1, ratio_k))
                assign(u_inv_k[i], (1.0 + u_inv_k[i]) * ratio_ratio - 1.0)
                with if_(u_inv_k[i] < u_min):
                    assign(u_inv_k[i], u_min)
                with if_(u_inv_k[i] > u_max):
                    assign(u_inv_k[i], u_max)

                if interleaved:
                    if active_reset_per_probe:
                        if node.parameters.reset_type == "active":
                            active_reset(
                                qubit,
                                readout_pulse_name="readout",
                                max_attempts=node.parameters.reset_max_attempts,
                            )
                        elif node.parameters.reset_type == "active_simple":
                            active_reset_simple(qubit, readout_pulse_name="readout")
                        else:
                            qubit.resonator.wait(qubit.thermalization_time * u.ns)
                        align()

                    assign(tau_clocks[i], lin_tau[probe])
                    qubit.xy.play("x180")
                    align()
                    qubit.wait(tau_clocks[i])
                    align()
                    readout_state(qubit, state_lin[i])
                    save(state_lin[i], state_lin_st[i])
                    with if_(state_lin[i] == 1):
                        qubit.xy.play("x180")
                    align()

            save(rep, time_stamp_st[i])
            save(t1_est[i], estimated_t1_st[i])
            save(u_inv_k[i], u_final_st[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            estimated_t1_st[i].buffer(n_reps).save(f"estimated_t1{i + 1}")
            u_final_st[i].buffer(n_reps).save(f"u_final{i + 1}")
            u_evol_st[i].buffer(n_probes).save(f"u_evol{i + 1}")
            t1_evol_st[i].buffer(n_probes).save(f"t1_evol{i + 1}")
            time_stamp_st[i].buffer(n_reps).save(f"time_stamp{i + 1}")
            if node.parameters.keep_shot_data:
                state_st[i].buffer(n_reps, n_probes).save(f"state{i + 1}")
                tau_st[i].buffer(n_reps, n_probes).save(f"tau_ms{i + 1}")
            if interleaved:
                state_lin_st[i].buffer(n_reps, n_probes).save(f"state_lin{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    samples, fig = simulate_and_plot(qmm, config, t1_bayesian, node.parameters)
    node.results = {"figure": fig}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(t1_bayesian)
        results = fetching_tool(job, ["n"], mode="live")
        wall_clock_start = time.time()
        while results.is_processing():
            progress_counter(results.fetch_all()[0], n_reps, start_time=results.start_time)
        job_total_duration_s = time.time() - wall_clock_start


# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds, time_stamp, ds_state, ds_tau, ds_state_lin = fetch_t1_datasets(
            job.result_handles,
            qubit_list,
            n_reps,
            n_probes,
            interleaved,
            node.parameters.keep_shot_data,
            total_duration_s=locals().get("job_total_duration_s"),
        )
        probe_axis = {"probe": np.arange(n_probes)}
        ds_u_evol = fetch_results_as_xarray(job.result_handles, qubit_list, probe_axis, "u_evol")
        ds_k_evol = xr.Dataset({"k_evol": 1.0 / ds_u_evol.u_evol}, coords=ds_u_evol.coords)
        ds_t1_evol = fetch_results_as_xarray(job.result_handles, qubit_list, probe_axis, "t1_evol")
        ds_t1_evol = ds_t1_evol.assign(t1_evol=ds_t1_evol.t1_evol * MS_TO_US)
        node.results = {"ds": ds}
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_list = resolve_qubits_from_node(machine, node)
        time_stamp = ds.time_stamp
        ds_state = ds.get("state")
        ds_tau = ds.get("tau_us")
        ds_state_lin = ds.get("state_lin")
        ds_k_evol = node.results.get("ds_k_evol")
        ds_t1_evol = node.results.get("ds_t1_evol")

    ds_estimated_t1 = ds
    ds_k = ds
    ds_theta = ds

    # %% {Data_analysis}
    ci = node.parameters.credible_interval
    t1_est = ds_estimated_t1.estimated_t1.values
    k_vals = ds_k.k_final.values
    _, t1_ci_low, t1_ci_high = posterior_t1_credible_interval(k_vals, t1_est, ci=ci)
    ds_ci = xr.Dataset(
        {
            "t1_ci_low": (("qubit", "repetition"), t1_ci_low),
            "t1_ci_high": (("qubit", "repetition"), t1_ci_high),
        },
        coords={"qubit": ds_estimated_t1.qubit.values, "repetition": np.arange(n_reps)},
    )

    ds_welch, ds_allan = compute_welch_and_allan(
        t1_est, time_stamp, ds_estimated_t1.qubit.values
    )

    # %% {Plotting}
    figures, validation_t1_fit_us = plot_bayesian_results(
        node,
        ds,
        ds_estimated_t1,
        ds_k,
        ds_theta,
        ds_ci,
        time_stamp,
        qubit_list,
        ds_state,
        ds_tau,
        ds_state_lin,
        ds_welch,
        ds_allan,
        ds_k_evol,
        ds_t1_evol,
        lin_times_clocks,
        t1_prior_by_name,
    )
    for fig_name, fig in figures.items():
        node.results[f"figure_{fig_name}"] = fig
    for fig in figures.values():
        plt.figure(fig.number)
        plt.show()

    if node.parameters.load_data_id is None:
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.results["t1_prior_us_used"] = t1_prior_by_name
        if validation_t1_fit_us is not None:
            node.results["validation_t1_fit_us"] = validation_t1_fit_us
        if ds_k_evol is not None:
            node.results["ds_k_evol"] = ds_k_evol
        if ds_t1_evol is not None:
            node.results["ds_t1_evol"] = ds_t1_evol
        if ds_welch is not None:
            node.results["ds_welch"] = ds_welch.to_dataset(name="t1_welch_psd")
        if ds_allan is not None:
            node.results["ds_allan"] = ds_allan
        node.machine = machine
        node.save()

# %%
