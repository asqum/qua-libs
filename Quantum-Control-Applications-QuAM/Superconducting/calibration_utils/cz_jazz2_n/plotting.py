"""Plotting module for the JAZZ2-N CZ amplitude calibration."""

from typing import Any, Dict

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names


def plot_raw_data_with_fit(
    ds_fit: xr.Dataset,
    qubit_pairs: Any,
    title_prefix: str = "JAZZ2-N CZ amplitude calibration",
    abs_amplitude_xlabel: str = "Amplitude (V)",
) -> Dict[str, Figure]:
    """Plot the JAZZ2-N heatmap and N-averaged sinc fit per qubit pair."""
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    figures: Dict[str, Figure] = {}

    map_grid = QubitPairGrid(grid_names, pair_names)
    for ax, qubit in grid_iter(map_grid):
        qp_name = qubit["qubit"]
        plot_individual_map_with_fit(ax, ds_fit, qp_name, abs_amplitude_xlabel=abs_amplitude_xlabel)
    map_grid.fig.suptitle(rf"{title_prefix} — $P_{{|00\rangle}}$ vs $N$ and amplitude")
    map_grid.fig.tight_layout()
    figures["map"] = map_grid.fig

    avg_grid = QubitPairGrid(grid_names, pair_names)
    for ax, qubit in grid_iter(avg_grid):
        qp_name = qubit["qubit"]
        plot_individual_avg_with_fit(ax, ds_fit, qp_name)
    avg_grid.fig.suptitle(rf"{title_prefix} — averaged $P_{{|00\rangle}}$ and sinc fit")
    avg_grid.fig.tight_layout()
    figures["avg"] = avg_grid.fig

    return figures


def plot_individual_map_with_fit(
    ax: Axes,
    ds_fit: xr.Dataset,
    qp_name: str,
    abs_amplitude_xlabel: str = "Amplitude (V)",
) -> None:
    """Plot one qubit-pair JAZZ2-N heatmap of P_|00>."""
    if qp_name not in set(map(str, ds_fit.qubit_pair.values)):
        ax.text(0.5, 0.5, f"{qp_name}: not in dataset", ha="center", va="center")
        ax.set_title(f"{qp_name} [NO DATA]")
        return

    fr = ds_fit.sel(qubit_pair=qp_name)

    if "p" not in fr:
        ax.text(0.5, 0.5, "No P(00) data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(qp_name)
        return

    amps_scale = fr.amp.values
    amps_abs = fr["amp_full"].values if "amp_full" in fr.coords else amps_scale
    n_values = fr.N.values

    p_map = fr["p"].transpose("N", "amp")
    xg, yg = np.meshgrid(amps_scale, n_values)
    pcm = ax.pcolormesh(xg, yg, p_map.values, cmap="magma", shading="auto")

    opt_scale = float(fr.optimal_amplitude_scale.values)
    opt_method = str(fr.fit_method.values)
    success = "success" in fr.coords and bool(fr.success) and np.isfinite(opt_scale)
    if success:
        ax.axvline(opt_scale, color="lime", lw=2, label=f"opt = {opt_scale:.4f} ({opt_method})")

    def amp_scale_to_abs(s, abs_values=amps_abs, scale_values=amps_scale):
        return np.interp(s, scale_values, abs_values)

    def amp_abs_to_scale(a, abs_values=amps_abs, scale_values=amps_scale):
        return np.interp(a, abs_values, scale_values)

    secax = ax.secondary_xaxis("top", functions=(amp_scale_to_abs, amp_abs_to_scale))
    secax.set_xlabel(abs_amplitude_xlabel)
    ax.set_title(qp_name if success else f"{qp_name} — fit failed")
    ax.set_xlabel("Amplitude scale (a.u.)")
    ax.set_ylabel("Repetition N = 2k")
    if success:
        ax.legend(loc="upper right", fontsize=8)
    ax.figure.colorbar(pcm, ax=ax, shrink=0.85).set_label(r"$P_{|00\rangle}$")


def plot_individual_avg_with_fit(ax: Axes, ds_fit: xr.Dataset, qp_name: str) -> None:
    """Plot one qubit-pair averaged P_|00> curve with sinc fit."""
    if qp_name not in set(map(str, ds_fit.qubit_pair.values)):
        ax.text(0.5, 0.5, f"{qp_name}: not in dataset", ha="center", va="center")
        ax.set_title(f"{qp_name} [NO DATA]")
        return

    fr = ds_fit.sel(qubit_pair=qp_name)

    if "p_avg" not in fr:
        ax.text(0.5, 0.5, "No averaged data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(qp_name)
        return

    amps_scale = fr.amp.values
    opt_scale = float(fr.optimal_amplitude_scale.values)
    success = "success" in fr.coords and bool(fr.success) and np.isfinite(opt_scale)

    ax.plot(amps_scale, fr["p_avg"].values, "o", ms=3, color="C0", label=r"$\langle P_{|00\rangle}\rangle_N$")
    if "sinc_fit" in fr:
        fit_vals = fr["sinc_fit"].values
        if np.any(np.isfinite(fit_vals)):
            ax.plot(amps_scale, fit_vals, "-", lw=1.5, color="C3", label="sinc fit")
    if success:
        ax.axvline(opt_scale, color="lime", lw=1.5, label=f"opt = {opt_scale:.4f}")

    ax.set_title(qp_name if success else f"{qp_name} — fit failed")
    ax.set_xlabel("Amplitude scale (a.u.)")
    ax.set_ylabel(r"$\langle P_{|00\rangle}\rangle_N$")
    ax.legend(loc="upper right", fontsize=8)
