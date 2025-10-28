from typing import Dict, List

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from typing import Any, Tuple


def plot_fit(ds: xr.Dataset, qubits: List[Any], fit_results: Dict):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    # grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for q in qubits:
        t_data = ds.time.values
        y_data = ds.flux_response.sel(qubit=q.name).values

        components = fit_results[q.name]["a_tau_tuple"]
        a_dc = fit_results[q.name]["a_dc"]
        if a_dc is None or (isinstance(a_dc, (float, np.floating)) and np.isnan(a_dc)):
            a_dc = float(y_data[-5:].mean()) if len(y_data) >= 5 else float(y_data.mean())

        fig, _ = plot_individual_fit(
            t_data, y_data, components=components, a_dc=a_dc,
            qubit_name=q.name, ds=ds
        )

    return fig


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr
from typing import List, Tuple, Optional


def plot_individual_fit(t_data: np.ndarray, y_data: np.ndarray,
                        components: List[Tuple[float, float]],
                        a_dc: float, qubit_name: str,
                        ds: Optional[xr.Dataset] = None):
    """Plot exponential fit results plus dataset amplitude (IQ_abs or I)."""

    # --- Build fit ---
    fit_text = f"a_dc = {a_dc:.3f}\n"
    y_fit = np.ones_like(t_data, dtype=float) * a_dc
    for i, (amp, tau) in enumerate(components):
        y_fit += amp * np.exp(-t_data / tau)
        fit_text += f"a{i + 1} = {amp / a_dc:.3f}, τ{i + 1} = {tau:.0f}ns\n"

    # --- Create grid layout: top full-width, bottom with 2 plots ---
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2)
    fig.suptitle(f"Long time Cryoscope: {qubit_name}")

    # --- Top plot: raw dataset (spans both columns) ---
    ax_top = fig.add_subplot(gs[0, :])  # span both columns
    if ds is not None:
        if "IQ_abs" in ds.data_vars:
            ds.IQ_abs.sel(qubit=qubit_name).plot(ax=ax_top, label="IQ_abs", cmap = "viridis")
        elif "I" in ds.data_vars:
            ds.I.sel(qubit=qubit_name).plot(ax=ax_top, label="I", cmap = "viridis")
        else:
            ax_top.text(0.5, 0.5, "No IQ_abs or I found",
                        ha="center", va="center", transform=ax_top.transAxes)
    ax_top.plot(ds.sel(qubit=qubit_name).time.values, ds.sel(qubit=qubit_name).center_freqs.values, "r-", label="Peak frequencies", linewidth=2)
    ax_top.set_title("Raw Dataset")
    ax_top.set_xlabel("Time [ns]")
    ax_top.set_ylabel("Detuning [Hz]")
    ax_top.legend()


    # --- Bottom-left: linear fit ---
    ax_lin = fig.add_subplot(gs[1, 0])
    ax_lin.plot(t_data, y_data, ".--", label="Data")
    ax_lin.plot(t_data, y_fit, label="Fit")
    ax_lin.text(0.98, 0.5, fit_text, transform=ax_lin.transAxes,
                fontsize=10, ha="right", va="center")
    ax_lin.set_xlabel("Time (ns)")
    ax_lin.set_ylabel("Flux Response")
    ax_lin.legend()
    ax_lin.grid(True)
    ax_lin.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # --- Bottom-right: log fit ---
    ax_log = fig.add_subplot(gs[1, 1])
    ax_log.plot(t_data, y_data, ".--", label="Data")
    ax_log.plot(t_data, y_fit, label="Fit")
    ax_log.text(0.98, 0.5, fit_text, transform=ax_log.transAxes,
                fontsize=10, ha="right", va="center")
    ax_log.set_xlabel("Time (ns)")
    ax_log.set_ylabel("Flux Response")
    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.legend(loc="best")
    ax_log.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, [ax_top, ax_lin, ax_log]

