"""Plotting helpers for flux-landscape fitting."""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names

from .analysis import FluxLandscapeFit, _select_flux_coord


def _add_detuning_axis(ax, ds: xr.Dataset, qubit_name: str, flux_qubit_coord: str, detuning_coord: str = "detuning"):
    try:
        flux_qubit_data = (_select_flux_coord(ds, flux_qubit_coord, qubit_name).values * 1e3).ravel()
        detuning_data = (_select_flux_coord(ds, detuning_coord, qubit_name).values * 1e-6).ravel()
        order = np.argsort(flux_qubit_data)
        x_sorted = flux_qubit_data[order]
        y_sorted = detuning_data[order]
        x_unique, unique_idx = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[unique_idx]
        if x_unique.size >= 2:

            def flux_to_detuning(x):
                return np.interp(np.asarray(x), x_unique, y_unique)

            def detuning_to_flux(y):
                return np.interp(np.asarray(y), y_unique, x_unique)

            sec_ax = ax.secondary_xaxis("top", functions=(flux_to_detuning, detuning_to_flux))
            sec_ax.set_xlabel("Detuning [MHz]")
    except Exception:
        pass


def plot_coupler_zeropoint_maps(
    ds: xr.Dataset,
    qubit_pairs: list,
    results: Dict[str, dict],
    *,
    use_state_discrimination: bool,
    fits: Optional[Dict[str, FluxLandscapeFit]] = None,
    analysis_debug: bool = False,
) -> Dict[str, Figure]:
    """Plot control/target 2D maps with optional fit overlays."""
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    figures: Dict[str, Figure] = {}
    machine_pairs = {qp.name: qp for qp in qubit_pairs}

    for state_type in ["control", "target"]:
        grid = QubitPairGrid(grid_names, qubit_pair_names)
        for ax, qp in grid_iter(grid):
            qubit_name = qp["qubit"]
            try:
                if use_state_discrimination:
                    values_to_plot = ds[f"state_{state_type}"].sel(qubit=qubit_name)
                else:
                    values_to_plot = ds[f"I_{state_type}"].sel(qubit=qubit_name)
                values_to_plot = values_to_plot.assign_coords(
                    {
                        "flux_qubit_mV": 1e3 * values_to_plot.flux_qubit_full,
                        "flux_coupler_mV": 1e3 * values_to_plot.flux_coupler_full,
                    }
                )
                values_to_plot.plot(ax=ax, cmap="viridis", x="flux_qubit_mV", y="flux_coupler_mV")
            except Exception as e:
                print(f"[WARN] Plot data failed for {qubit_name}: {e}")
                ax.set_title(f"{qubit_name} (raw plot failed)")
                continue

            res = results.get(qubit_name, {})
            has_marker = False
            if np.isfinite(res.get("flux_coupler_min_full", np.nan)):
                ax.axhline(1e3 * res["flux_coupler_min_full"], color="red", lw=2.0, ls="--", label="Decoupling offset")
                has_marker = True
            qubit_pair = machine_pairs.get(qubit_name)
            if qubit_pair is not None:
                idle = 1e3 * getattr(qubit_pair.coupler, "decouple_offset", np.nan)
                if np.isfinite(idle):
                    ax.axhline(idle, color="blue", lw=0.5, ls="--", label="Current decoupling offset")
            if np.isfinite(res.get("flux_coupler_max_full", np.nan)):
                ax.axhline(1e3 * res["flux_coupler_max_full"], color="black", lw=1.0, ls=":")
                has_marker = True
            if np.isfinite(res.get("flux_qubit_max", np.nan)):
                ax.axvline(1e3 * res["flux_qubit_max"], color="black", lw=1.0, ls=":")
                has_marker = True
            if np.isfinite(res.get("flux_qubit_max", np.nan)) and np.isfinite(res.get("flux_coupler_max_full", np.nan)):
                ax.plot(
                    1e3 * res["flux_qubit_max"],
                    1e3 * res["flux_coupler_max_full"],
                    marker="+",
                    color="black",
                    markersize=10,
                    mew=2.0,
                    label="Gate starting point",
                )
            elif not has_marker and res.get("fit_success") is False:
                ax.text(0.02, 0.98, "fit failed", transform=ax.transAxes, va="top", ha="left", fontsize=8, color="red")
            if has_marker:
                ax.legend(fontsize=7, loc="upper right", frameon=True)

            _add_detuning_axis(ax, ds, qubit_name, "flux_qubit_full")
            ax.set_xlabel("Qubit flux shift [mV]")
            ax.set_ylabel("Coupler flux [mV]")
            ax.set_title(f"{qubit_name}", fontsize=9)

        grid.fig.suptitle(f"{state_type.capitalize()} Qubit", y=0.97, fontsize=12, weight="bold")
        plt.tight_layout()
        figures[f"figure_{state_type}"] = grid.fig

    if analysis_debug and fits:
        figures["contrast_debug"] = plot_contrast_cut_debug(
            fits,
            qubit_pairs,
            ylabel="|contrast| (|control − target|)",
        )
    return figures


def plot_contrast_cut_debug(
    fits: Dict[str, FluxLandscapeFit],
    qubit_pairs: list,
    *,
    ylabel: str = "signal",
) -> Figure:
    """1D cut debug: raw/smoothed trace, flat/oscillation masks, fit markers."""
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)

    for ax, qp in grid_iter(grid):
        qp_name = qp["qubit"]
        fit = fits.get(qp_name)
        if fit is None or fit.contrast_raw is None:
            ax.set_title(f"{qp_name} (no cut data)")
            continue

        x_v = fit.contrast_coupler_full if fit.contrast_coupler_full is not None else fit.contrast_coupler_rel
        x = 1e3 * np.asarray(x_v).ravel()
        y = np.asarray(fit.contrast_raw).ravel()
        smoothed = np.asarray(fit.contrast_smoothed).ravel()
        osc_mask = np.asarray(fit.osc_mask).astype(bool)
        flat_mask = np.asarray(fit.flat_mask).astype(bool)

        ax.plot(x, y, color="steelblue", lw=1.0, alpha=0.4, label="raw")
        ax.plot(x, smoothed, color="steelblue", lw=1.8, label="smoothed")
        ax.fill_between(x, 0, 1, where=osc_mask, alpha=0.12, color="limegreen", transform=ax.get_xaxis_transform(), label="oscillation")
        ax.fill_between(x, 0, 1, where=flat_mask, alpha=0.15, color="tomato", transform=ax.get_xaxis_transform(), label="flat")
        ax.axhline(0, color="gray", ls=":", lw=0.8)

        if np.isfinite(fit.optimal_decouple_offset):
            ax.axvline(1e3 * fit.optimal_decouple_offset, color="red", ls="--", lw=1.5, label="Decouple")
        if np.isfinite(fit.optimal_gate_coupler_flux_total):
            ax.axvline(1e3 * fit.optimal_gate_coupler_flux_total, color="green", ls="--", lw=1.5, label="Gate coupler")
        if np.isfinite(fit.optimal_qubit_flux):
            ax.set_title(f"{qp_name} @ qubit flux {fit.optimal_qubit_flux * 1e3:.1f} mV")
        else:
            ax.set_title(qp_name)

        ax.set_xlabel("Coupler flux [mV]")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6, loc="upper left")

    grid.fig.tight_layout()
    return grid.fig
