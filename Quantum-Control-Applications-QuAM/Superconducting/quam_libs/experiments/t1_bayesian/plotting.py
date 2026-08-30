"""Plotting for adaptive Bayesian T1 tracking."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import gamma as gamma_dist
from scipy.stats import invgamma, norm, lognorm, kstest
from scipy.signal import welch
from scipy.optimize import curve_fit
from quam_libs.lib.fit import decay_exp, fit_decay_exp
from quam_libs.lib.plot_utils import QubitGrid, grid_iter

logger = logging.getLogger(__name__)


def plot_bayesian_results(
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
    fig_size:float=3
):
    parameters = node.parameters
    ci = parameters.credible_interval
    interleaved = parameters.interleaved_validation
    keep_shot_data = parameters.keep_shot_data
    figures = {}

    grid_t1 = QubitGrid(ds, [q.grid_location for q in qubit_list], size=fig_size)
    for ax, qubit in grid_iter(grid_t1):
        qname = qubit["qubit"]
        t_axis = time_stamp.sel(qubit=qname).values
        y = ds_estimated_t1.estimated_t1.sel(qubit=qname).values
        lo = ds_ci.t1_ci_low.sel(qubit=qname).values
        hi = ds_ci.t1_ci_high.sel(qubit=qname).values
        dt_est = np.diff(t_axis)
        est_time_ms = float(np.mean(dt_est)) * 1e3 if dt_est.size else np.nan
        ax.plot(t_axis, y, "o-", alpha=0.6, markersize=3, label="T1 estimate")
        ax.hlines(np.mean(y), min(t_axis), max(t_axis), label="T1 mean", linestyle='-', color='r')
        ax.hlines(np.mean(y)-np.std(y), min(t_axis), max(t_axis), linestyle='--', color='r')
        ax.hlines(np.mean(y)+np.std(y), min(t_axis), max(t_axis), linestyle='--', color='r')
        ax.fill_between(t_axis, lo, hi, alpha=0.25, label=f"{int(ci * 100)}% CI")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("T1 (µs)")
        ax.set_title(f"{qname} T1 ~ {round(np.mean(y),1)} $\pm$ {round(100*np.std(y)/np.mean(y))}%\n #={len(y)} \nestimation time = {est_time_ms:.3f} ms")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    grid_t1.fig.suptitle("Adaptive Bayesian T1 trace (u = 1/k)")
    grid_t1.fig.tight_layout()
    figures["t1_trace"] = grid_t1.fig

    grid_t1_2 = QubitGrid(ds, [q.grid_location for q in qubit_list], size=fig_size)
    for ax, qubit in grid_iter(grid_t1_2):
        qname = qubit["qubit"]
        t_axis = time_stamp.sel(qubit=qname).values
        y = ds_estimated_t1.estimated_t1.sel(qubit=qname).values
        counts, bins, _ = ax.hist(y, bins='auto', color='skyblue', edgecolor='white', label='Counts')
        bin_width = bins[1] - bins[0]
        scaling_factor = len(y) * bin_width
        exp_time = round(t_axis[-1]-t_axis[0],1) if t_axis[-1]-t_axis[0] > 1 else round(t_axis[-1]-t_axis[0],3)

        # --- 1. 自動擬合兩種模型 ---
        mu_norm, sigma_norm = norm.fit(y)
        shape_log, loc_log, scale_log = lognorm.fit(y)
        
        # --- 2. 讓 KS 檢定自動評分 (statistic 越小代表擬合越準確) ---
        ks_norm = kstest(y, 'norm', args=(mu_norm, sigma_norm))
        ks_lognorm = kstest(y, 'lognorm', args=(shape_log, loc_log, scale_log))
        
        # 準備 X 軸
        x = np.linspace(min(y), max(y), 100)
        
        # --- 3. 完全自動判斷邏輯 ---
        if ks_norm.statistic < ks_lognorm.statistic:
            # 判斷結果：常態分佈比較準
            p = norm.pdf(x, mu_norm, sigma_norm)*scaling_factor
            ax.plot(x, p, 'r-', lw=2, label='Normal Fit')
            
            # 取得要顯示的數據
            mu_display, sigma_display = mu_norm, sigma_norm
            fit_type = "Normal"
        else:
            # 判斷結果：對數常態分佈 (Lognormal) 比較準
            p = lognorm.pdf(x, shape_log, loc=loc_log, scale=scale_log)*scaling_factor
            ax.plot(x, p, 'r-', lw=2, label='Lognorm Fit')
            mu_display = loc_log + scale_log * np.exp(-shape_log**2)
            sigma_display = lognorm.std(shape_log, loc=loc_log, scale=scale_log)
            fit_type = "Lognorm"


        ax.set_title(f"{qubit['qubit']}\n $T_{{1}} = {mu_display:.1f} \pm {sigma_display:.2f}  \mu$s")
        ax.set_xlabel("T1 (µs)")
        ax.set_ylabel("Counts")
        ax.grid(axis='y', alpha=0.3)

    grid_t1_2.fig.suptitle(f" T1 Statistics, #={len(y)}\n tracking time={exp_time}s", fontsize=16, y=1.02)
    grid_t1_2.fig.tight_layout()
    figures["figure_histogram"] = grid_t1_2.fig



    if ds_k_evol is not None and ds_t1_evol is not None:
        t1_grid_us = np.linspace(
            max(1.0, 0.5 * float(parameters.t1_min_us)),
            1.2 * float(parameters.t1_max_us),
            400,
        )
        probe_axis = ds_t1_evol.probe.values
        grid_conv = QubitGrid(ds_t1_evol, [q.grid_location for q in qubit_list], size=fig_size)
        for ax, qubit in grid_iter(grid_conv):
            qname = qubit["qubit"]
            k_ev = ds_k_evol.k_evol.sel(qubit=qname).values
            t1_ev_us = ds_t1_evol.t1_evol.sel(qubit=qname).values
            t_exec = probe_axis.astype(float)
            theta_us = k_ev * t1_ev_us
            pdf = np.zeros((t_exec.size, t1_grid_us.size))
            for p in range(t_exec.size):
                if k_ev[p] > 0 and theta_us[p] > 0:
                    pdf[p] = invgamma.pdf(t1_grid_us, a=k_ev[p], scale=theta_us[p])
            pcm = ax.pcolormesh(t1_grid_us, t_exec, pdf, cmap="viridis", shading="auto")
            ax.plot(t1_ev_us, t_exec, "w.-", lw=1, ms=3, label="T1 estimate")
            ax.set_xlabel("T1 (µs)")
            ax.set_ylabel("probe index")
            ax.set_title(f"{qname}\nfinal T1={t1_ev_us[-1]:.1f} µs, k={k_ev[-1]:.2f}")
            ax.legend(fontsize=7, loc="upper right")
            grid_conv.fig.colorbar(pcm, ax=ax, label="P(T1)")
        grid_conv.fig.suptitle("Posterior P(T1) evolution (last repetition)")
        grid_conv.fig.tight_layout()
        figures["gamma_evolution"] = grid_conv.fig

        g1_max = 1.0 / parameters.t1_min_us
        g1_grid = np.linspace(1e-4, g1_max, 500)
        grid_gam = QubitGrid(ds_t1_evol, [q.grid_location for q in qubit_list], size=fig_size)
        for ax, qubit in grid_iter(grid_gam):
            qname = qubit["qubit"]
            k_ev = ds_k_evol.k_evol.sel(qubit=qname).values
            t1_ev_us = ds_t1_evol.t1_evol.sel(qubit=qname).values
            theta_us = k_ev * t1_ev_us
            pdf_g = np.zeros((probe_axis.size, g1_grid.size))
            for p in range(probe_axis.size):
                if k_ev[p] > 0 and theta_us[p] > 0:
                    pdf_g[p] = gamma_dist.pdf(g1_grid, a=k_ev[p], scale=1.0 / theta_us[p])
            pcm = ax.pcolormesh(g1_grid, probe_axis, pdf_g, cmap="viridis", shading="auto")
            ax.plot(1.0 / t1_ev_us, probe_axis, "w.-", lw=1, ms=3, label="Γ₁ estimate")
            ax.set_xlabel("Γ₁ = 1/T1 (1/µs)")
            ax.set_ylabel("Bayesian update step")
            ax.set_title(f"{qname}\nfinal T1={t1_ev_us[-1]:.1f} µs, k={k_ev[-1]:.2f}")
            ax.legend(fontsize=7, loc="upper right")
            grid_gam.fig.colorbar(pcm, ax=ax, label="P(Γ₁)")
        grid_gam.fig.suptitle("Gamma posterior P(Γ₁) evolution within last estimation round")
        grid_gam.fig.tight_layout()
        figures["gamma_pdf_evolution"] = grid_gam.fig

   
    if ds_welch is not None:
        # 1. 定義 Lorentzian + White Noise 模型
        def lorentzian_model(f, A, fc, C):
            return A / (1 + (f / fc)**2) + C
        def log_lorentzian_model(f, A, fc, C):
            val = lorentzian_model(f, A, fc, C)
            return np.log10(np.maximum(val, 1e-12))
            
        grid_welch = QubitGrid(ds_welch.to_dataset(name="t1_welch_psd"), [q.grid_location for q in qubit_list], size=fig_size)
        for ax, qubit in grid_iter(grid_welch):
            qname = qubit["qubit"]
            
            # 繪製原始 PSD 數據
            psd_sub = ds_welch.sel(qubit=qname)
            psd_sub.plot(ax=ax, label='Welch PSD')
            
            # 2. 自動提取 xarray 的頻率座標與 PSD 陣列
            freq_dim = [d for d in psd_sub.dims if d != 'qubit'][0]
            f_data = psd_sub[freq_dim].values
            psd_data = psd_sub.values
            
            # 過濾非正數與 NaN 值
            valid_mask = (f_data > 0) & (~np.isnan(psd_data))
            f_valid = f_data[valid_mask]
            psd_valid = psd_data[valid_mask]
            
            # 3. 進行 Fitting 與定量分析
            sigma_TLS = "NaN"
            tau_ms = np.nan
            try:
                # 2. 對數重採樣：在 Log 頻率軸上均勻取 50 個點 (平衡高低頻點密度)
                f_log = np.logspace(np.log10(f_valid[0]), np.log10(f_valid[-1]), 50)
                psd_log = np.interp(f_log, f_valid, psd_valid) # 內插 PSD 數值

                # initially guess
                C_guess = max(np.median(psd_log[-10:]), 1e-6)
                A_guess = max(psd_log[0] - C_guess, 1e-6)
                mid_psd_log = 10 ** ((np.log10(psd_log[0]) + np.log10(C_guess)) / 2)
                idx_fc = np.argmin(np.abs(psd_log - mid_psd_log))
                fc_guess = f_log[idx_fc]
                
                
                p0 = [A_guess, fc_guess, C_guess]
                bounds = (0, [np.inf, np.max(f_log), np.inf])
                
                # 4. 在 Log 空間進行 Fitting
                popt, _ = curve_fit(
                    log_lorentzian_model, 
                    f_log, 
                    np.log10(psd_log), 
                    p0=p0, 
                    bounds=bounds
                )
                A_fit, fc_fit, C_fit = popt
                
                # 計算 TLS 特徵翻轉時間 tau (ms)
                tau_ms = (1.0 / (2 * np.pi * fc_fit)) * 1000
                # T1 sigma due to TLS
                sigma_TLS = round(np.sqrt(np.pi*A_fit*fc_fit*0.5), 2)

                
                # 生成平滑擬合曲線
                f_dense = np.logspace(np.log10(f_valid[0]), np.log10(f_valid[-1]), 200)
                psd_fit = lorentzian_model(f_dense, *popt)
                
                # 4. 畫出 Fitting 紅色虛線與圖例
                fit_label = f"$f_c={fc_fit:.2f}$ Hz\n$A={A_fit:.3f}$\n$C={C_fit:.3f}$"
                ax.plot(f_dense, psd_fit, 'r--', lw=1.8, label=fit_label)
                ax.legend(fontsize='small', loc='upper right')
            except Exception:
                pass

            tau_display = round(tau_ms, 1) if np.isfinite(tau_ms) else "NaN"

            # 5. 座標軸設定
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid()
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Welch PSD of T1")
            ax.set_title(
                f"{qname}\n"
                f"$\\sigma_{{TLS}} = {sigma_TLS}\\ \\mu s$\n"
                f"$\\tau_{{TLS}} = {tau_display}\\text{{ ms}}$"
            )
            
        grid_welch.fig.suptitle("T1 fluctuation Welch PSD & Lorentzian Fit")
        grid_welch.fig.tight_layout()
        figures["welch_psd"] = grid_welch.fig

    if ds_allan is not None:
        grid_allan = QubitGrid(ds_allan, [q.grid_location for q in qubit_list], size=fig_size)
        for ax, qubit in grid_iter(grid_allan):
            qname = qubit["qubit"]
            tau_a = ds_allan.allan_tau.sel(qubit=qname).values
            adev = ds_allan.allan_deviation.sel(qubit=qname).values
            mask = np.isfinite(tau_a) & np.isfinite(adev) & (tau_a > 0) & (adev > 0)
            ax.loglog(tau_a[mask], adev[mask], "o-")
            ax.set_xlabel("Averaging time τ (s)")
            ax.set_ylabel("Allan deviation σ(T1)")
            ax.set_title(qname)
            ax.grid(True, which="both", alpha=0.3)
        grid_allan.fig.suptitle("T1 Allan deviation")
        grid_allan.fig.tight_layout()
        figures["allan_deviation"] = grid_allan.fig

    if keep_shot_data and ds_state is not None and ds_tau is not None:
        grid_shot = QubitGrid(ds, [q.grid_location for q in qubit_list], size=fig_size)
        for ax, qubit in grid_iter(grid_shot):
            qname = qubit["qubit"]
            tau_vals = np.maximum(ds_tau.tau_us.sel(qubit=qname).values, 0.0)
            states = ds_state.state.sel(qubit=qname).values
            t_axis = time_stamp.sel(qubit=qname).values
            Y = np.repeat(t_axis[:, None], states.shape[1], axis=1)
            pcm = ax.pcolormesh(tau_vals, Y, states, cmap="coolwarm", vmin=0, vmax=1, shading="nearest")
            ax.set_xlabel("adaptive τ (µs)")
            ax.set_ylabel("time (s)")
            ax.set_title(qname)
        grid_shot.fig.colorbar(pcm, ax=ax, label="state")
        grid_shot.fig.suptitle("Single-shot outcomes vs adaptive wait time")
        grid_shot.fig.tight_layout()
        figures["shot_data"] = grid_shot.fig

    validation_t1_fit_us = None
    if interleaved and ds_state_lin is not None:
        try:
            lin_times_us = 4 * lin_times_clocks * 1e-3
            p_exc_lin = (
                ds_state_lin.state_lin.mean(dim="repetition")
                .rename({"probe": "idle_time"})
                .assign_coords(idle_time=("idle_time", lin_times_us))
            )
            fit_data = fit_decay_exp(p_exc_lin, "idle_time")
            decay_vals = fit_data.sel(fit_vals="decay")
            if np.all(~np.isfinite(decay_vals.values)):
                raise ValueError("interleaved validation exponential fit did not converge")
            tau_fit = -1 / decay_vals

            grid_val = QubitGrid(ds_state_lin, [q.grid_location for q in qubit_list], size=fig_size)
            for ax, qubit in grid_iter(grid_val):
                qname = qubit["qubit"]
                p = p_exc_lin.sel(qubit=qname).values
                t1_adaptive = float(ds_estimated_t1.estimated_t1.sel(qubit=qname).mean())
                t1_fit = float(tau_fit.sel(qubit=qname).values)
                ax.plot(lin_times_us, p, "o-", label="interleaved non-adaptive")
                t_fit = np.linspace(lin_times_us.min(), lin_times_us.max(), 200)
                ax.plot(
                    t_fit,
                    decay_exp(
                        t_fit,
                        fit_data.sel(qubit=qname, fit_vals="a").values,
                        fit_data.sel(qubit=qname, fit_vals="offset").values,
                        fit_data.sel(qubit=qname, fit_vals="decay").values,
                    ),
                    "--",
                    color="gray",
                    label=f"exp. fit T1={t1_fit:.1f} µs",
                )
                ax.axhline(0.5, color="k", linestyle=":", alpha=0.3)
                ax.set_xlabel("τ_lin (µs)")
                ax.set_ylabel("P(|1⟩)")
                ax.set_title(f"{qname}\nadaptive mean T1={t1_adaptive:.1f} µs")
                ax.legend(fontsize=8)
            grid_val.fig.suptitle("Interleaved validation: non-adaptive vs adaptive")
            grid_val.fig.tight_layout()
            figures["validation"] = grid_val.fig
            validation_t1_fit_us = {
                q: float(tau_fit.sel(qubit=q).values) for q in ds_estimated_t1.qubit.values
            }
            for q in ds_estimated_t1.qubit.values:
                t1_adaptive = float(ds_estimated_t1.estimated_t1.sel(qubit=q).mean())
                t1_na = validation_t1_fit_us[q]
                if np.isfinite(t1_na) and t1_na > 0 and (
                    t1_adaptive > 3 * t1_na or t1_adaptive < t1_na / 3
                ):
                    logger.warning(
                        "[%s] adaptive mean T1=%.1f µs disagrees with interleaved fit T1=%.1f µs. "
                        "Check t1_prior_us (used %.1f µs) or calibrate qubit.T1 in QuAM.",
                        q,
                        t1_adaptive,
                        t1_na,
                        t1_prior_by_name[q],
                    )
        except Exception as exc:
            logger.warning("Skipping interleaved validation plot: %s", exc)

    return figures, validation_t1_fit_us
