"""Plotting utilities for XY-Z delay calibration visualizations."""

from typing import Any, List

import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from quam_libs.lib.plot_utils import QubitGrid, grid_iter


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[Any], fits: xr.Dataset) -> Figure:
    """
    Plots the relative delay scans between the XY and Z pulses with the fitted center.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(
            ax, ds.sel(qubit=qubit["qubit"]), qubit, fits.sel(qubit=qubit["qubit"])
        )

    grid.fig.suptitle("XY-Z delay calibration")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(
    ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None
) -> None:
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        Dataset already selected for one qubit.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        Fit dataset already selected for the same qubit.
    """
    ds.difference.plot(ax=ax)
    if fit is not None and fit.success.data:
        fit.fit.plot(ax=ax)
        ax.axvline(fit.flux_delay.data, color="red", linestyle="--", label="fitted center")
        ax.legend()
