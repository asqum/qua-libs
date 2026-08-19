"""Analysis helpers for adaptive Bayesian T1 tracking (T1Bayesian_uk)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr
from scipy.stats import norm

logger = logging.getLogger(__name__)

US_TO_MS = 1e-3
MS_TO_US = 1e3
MS_TO_CLK = 250_000.0  # ms -> ns (/1e6) -> 4 ns clock cycles (/4)
MS_TO_CLK_INT = 250_000  # int form for Cast.mul_int_by_fixed


def fetch_results_as_xarray(handles, qubits, measurement_axis, var_name):
    coords = {**measurement_axis, "qubit": [qubit.name for qubit in qubits]}
    coords = {key: coords[key] for key in reversed(coords.keys())}
    dim_names = list(coords.keys())
    per_qubit = [np.squeeze(handles.get(f"{var_name}{i + 1}").fetch_all()) for i in range(len(qubits))]
    values = np.stack(per_qubit, axis=0)
    while values.ndim > len(dim_names):
        values = np.squeeze(values, axis=-1)
    while values.ndim < len(dim_names):
        values = values[np.newaxis, ...]
    return xr.Dataset({var_name: (dim_names, values)}, coords=coords)


def fetch_t1_datasets(
    handles,
    qubit_list,
    n_reps,
    n_probes,
    interleaved,
    keep_shot_data,
    total_duration_s: float | None = None,
):
    rep_axis = {"repetition": np.arange(n_reps)}
    probe_rep_axis = {"probe": np.arange(n_probes), "repetition": np.arange(n_reps)}

    ds_estimated_t1 = fetch_results_as_xarray(handles, qubit_list, rep_axis, "estimated_t1")
    ds_estimated_t1 = ds_estimated_t1.assign(estimated_t1=ds_estimated_t1.estimated_t1 * MS_TO_US)

    ds_u = fetch_results_as_xarray(handles, qubit_list, rep_axis, "u_final")
    k_final = 1.0 / ds_u.u_final
    ds_k = xr.Dataset({"k_final": k_final}, coords=ds_u.coords)
    theta_final = ds_estimated_t1.estimated_t1 * k_final
    ds_theta = xr.Dataset({"theta_final": theta_final}, coords=ds_estimated_t1.coords)

    if handles.get("time_stamp1") is not None:
        ds_time_stamp = fetch_results_as_xarray(handles, qubit_list, rep_axis, "time_stamp")
        rep_index = ds_time_stamp.time_stamp.values.astype(float)
        if total_duration_s is not None and n_reps > 1:
            time_vals = (rep_index / (n_reps - 1)) * total_duration_s
        else:
            time_vals = rep_index - np.min(rep_index)
            if np.max(time_vals) > 0:
                time_vals = time_vals / np.max(time_vals)
        time_stamp = xr.DataArray(
            time_vals,
            coords=ds_time_stamp.time_stamp.coords,
            dims=ds_time_stamp.time_stamp.dims,
            name="time_stamp",
            attrs={"long_name": "Laboratory time", "units": "s"},
        )
    else:
        reps = np.arange(n_reps, dtype=float)
        if total_duration_s is not None and n_reps > 1:
            t_vals = reps / (n_reps - 1) * total_duration_s
        else:
            t_vals = reps
        time_stamp = xr.DataArray(
            np.tile(t_vals, (len(qubit_list), 1)),
            coords={"qubit": [q.name for q in qubit_list], "repetition": np.arange(n_reps)},
            dims=("qubit", "repetition"),
            name="time_stamp",
            attrs={"long_name": "Laboratory time", "units": "s"},
        )

    parts = [ds_estimated_t1, ds_k, ds_theta, time_stamp]
    ds_state = ds_tau = ds_state_lin = None
    if keep_shot_data:
        ds_state = fetch_results_as_xarray(handles, qubit_list, probe_rep_axis, "state")
        ds_tau = fetch_results_as_xarray(handles, qubit_list, probe_rep_axis, "tau_ms")
        ds_tau = ds_tau.assign(tau_us=ds_tau.tau_ms * MS_TO_US).drop_vars("tau_ms")
        parts.extend([ds_state, ds_tau])
    if interleaved:
        ds_state_lin = fetch_results_as_xarray(handles, qubit_list, probe_rep_axis, "state_lin")
        parts.append(ds_state_lin)
    return xr.merge(parts), time_stamp, ds_state, ds_tau, ds_state_lin


def welch_segment_params(n_samples, max_nperseg=1024):
    if n_samples < 4:
        return None
    nperseg = min(max_nperseg, n_samples)
    if nperseg < 8:
        nperseg = n_samples
    noverlap = nperseg // 2
    if noverlap >= nperseg:
        noverlap = nperseg - 1
    return nperseg, noverlap


def posterior_t1_credible_interval(k, t1_est_us, ci=0.9):
    k = np.asarray(k, dtype=float)
    t1_est_us = np.asarray(t1_est_us, dtype=float)
    t1_std = t1_est_us / np.sqrt(np.maximum(k, 1.0))
    z = float(norm.ppf(0.5 + ci / 2))
    t1_ci_low = np.maximum(t1_est_us - z * t1_std, 0.0)
    t1_ci_high = t1_est_us + z * t1_std
    return t1_est_us, t1_ci_low, t1_ci_high


def resolve_t1_prior_us_per_qubit(parameters: Any, qubit_list):
    fallback = float(parameters.t1_prior_us)
    priors = []
    for q in qubit_list:
        t1_s = getattr(q, "T1", None) if parameters.use_quam_t1_prior else None
        priors.append(float(t1_s) * 1e6 if t1_s is not None and t1_s > 0 else fallback)
    return priors


def resolve_confusion_alpha_beta(qubit) -> tuple[float, float]:
    cm = qubit.resonator.confusion_matrix
    if cm is None:
        return 0.0, 0.0
    return float(cm[1][0]), float(cm[0][1])


def overlapping_allan_deviation(trace, dt, max_points=80):
    n = len(trace)
    if n < 4:
        return np.array([]), np.array([])
    max_m = min(n // 2, max_points)
    if max_m < 2:
        return np.array([]), np.array([])
    ms = np.unique(np.geomspace(1, n // 2, num=max_m, dtype=int))
    taus, adevs = [], []
    for m in ms:
        tau = m * dt
        means = np.convolve(trace, np.ones(m) / m, mode="valid")
        if len(means) <= m:
            continue
        deltas = means[m:] - means[:-m]
        adevs.append(np.sqrt(0.5 * np.mean(deltas**2)))
        taus.append(tau)
    return np.array(taus), np.array(adevs)


def compute_welch_and_allan(t1_est, time_stamp, qubit_names):
    from scipy.signal import welch

    t_vals = time_stamp.values
    dt = float(np.mean(np.diff(t_vals, axis=1)))
    fs = 1.0 / dt if dt > 0 else 1.0
    welch_params = welch_segment_params(t1_est.shape[1])
    n_qubits_plot = t1_est.shape[0]

    ds_welch = None
    if welch_params is not None:
        nperseg, noverlap = welch_params
        welch_freqs_list = []
        welch_psd_list = []
        for qidx in range(n_qubits_plot):
            y = t1_est[qidx] - np.nanmean(t1_est[qidx])
            f_welch, pxx = welch(
                y,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="density",
                detrend="constant",
            )
            welch_freqs_list.append(f_welch)
            welch_psd_list.append(pxx)
        welch_psd_arr = np.stack(welch_psd_list, axis=0)
        ds_welch = xr.DataArray(
            welch_psd_arr,
            dims=("qubit", "frequency"),
            coords={"qubit": qubit_names, "frequency": welch_freqs_list[0]},
            name="t1_welch_psd",
        )

    allan_tau_list = []
    allan_dev_list = []
    for qidx in range(n_qubits_plot):
        taus_a, adev_a = overlapping_allan_deviation(t1_est[qidx], dt)
        allan_tau_list.append(taus_a)
        allan_dev_list.append(adev_a)

    max_len = max(len(a) for a in allan_tau_list) if allan_tau_list else 0
    ds_allan = None
    if max_len > 0:
        allan_tau_arr = np.full((n_qubits_plot, max_len), np.nan)
        allan_dev_arr = np.full((n_qubits_plot, max_len), np.nan)
        for qidx in range(n_qubits_plot):
            n_a = len(allan_tau_list[qidx])
            allan_tau_arr[qidx, :n_a] = allan_tau_list[qidx]
            allan_dev_arr[qidx, :n_a] = allan_dev_list[qidx]
        ds_allan = xr.Dataset(
            {
                "allan_tau": (("qubit", "allan_index"), allan_tau_arr),
                "allan_deviation": (("qubit", "allan_index"), allan_dev_arr),
            },
            coords={"qubit": qubit_names, "allan_index": np.arange(max_len)},
        )
    return ds_welch, ds_allan
