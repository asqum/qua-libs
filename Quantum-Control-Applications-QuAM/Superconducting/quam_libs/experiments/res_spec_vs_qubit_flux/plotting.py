import xarray as xr
import numpy as np
from matplotlib import pyplot as plt

from quam_libs.lib.plot_utils import QubitGrid, grid_iter


def plot_resonator_spectroscopy_vs_qubit_flux(ds: xr.Dataset, fit: dict, qubits: list) -> dict[str, plt.Figure]:
    """
    Plot resonator spectroscopy data for each qubit in the dataset, in separate QubitGrid plots
    for amplitude, phase, average phase, and its derivative.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'I', 'Q', 'flux', 'freq', and 'qubit'.
    fit : dict
        Output from fit_resonator_spectroscopy_vs_flux(), mapping qubits to feature flux values.
    qubits : list
        List of Qubit objects, each with a `grid_location` attribute.
    """
    S = ds.I + 1j * ds.Q

    # Precompute quantities for all qubits
    data_by_qubit = {}
    for qubit in ds.qubit:
        qstr = str(qubit.item())
        flux = ds.sel(qubit=qubit).flux.values
        freq = ds.sel(qubit=qubit).freq.values
        signal = S.sel(qubit=qubit)

        phase = np.unwrap(np.angle(signal))
        avg_phase = np.mean(phase, axis=0)
        derivative = np.gradient(avg_phase)

        for i in range(len(signal)):
            signal[i] -= signal[i].mean()
        amplitude = np.abs(signal)

        data_by_qubit[qstr] = {
            "flux": flux,
            "freq": freq,
            "amplitude": amplitude,
            "phase": phase,
            "avg_phase": avg_phase,
            "derivative": derivative,
            "fit": fit.get(qstr, {})
        }

    # Plot types to generate
    plot_specs = [
        ("Amplitude", lambda d, ax: ax.pcolormesh(d["flux"], d["freq"], d["amplitude"], shading="auto")),
        ("Phase", lambda d, ax: ax.pcolormesh(d["flux"], d["freq"], d["phase"], shading="auto")),
        ("Average Phase", lambda d, ax: ax.plot(d["flux"], d["avg_phase"], 'b')),
        ("Derivative of Average Phase", lambda d, ax: ax.plot(d["flux"], d["derivative"], 'b')),
    ]

    labels_added = set()

    # Utility function for overlays
    def plot_features(ax, flux, fit_data):
        for label, color, linestyle in [
            ("minimum", "y", "--"),
            ("crossing", "red", "--"),
            ("insensitive", "black", "--"),
        ]:
            value = fit_data.get(label, np.nan)
            if not np.isnan(value):
                if label not in labels_added:
                    ax.axvline(value, color=color, linestyle=linestyle, label=label)
                    labels_added.add(label)
                else:
                    ax.axvline(value, color=color, linestyle=linestyle)

    # Loop through each plot type and generate a QubitGrid
    figs = {}
    for title, plot_fn in plot_specs:
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            qubit = qubit["qubit"]
            if qstr not in data_by_qubit:
                continue
            d = data_by_qubit[qubit]
            plot_fn(d, ax)
            plot_features(ax, d["flux"], d["fit"])

            ax.set_title(f"{qubit}")
            ax.set_xlabel("Bias [a.u.]")

            if title in ["Amplitude", "Phase"]:
                ax.scatter(d["fit"]["minimum"], d["fit"]["frequency_at_minimum"], s=100, color="yellow", marker="*")
                ax.scatter(d["fit"]["insensitive"], d["fit"]["frequency_at_insensitive"], s=100, color="black", marker="*")

                yticks = ax.get_yticks()
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks / 1e6)
                ax.set_ylabel("Freq [MHz]")

        grid.fig.suptitle(title, fontsize=16)
        grid.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        grid.fig.legend(loc='lower center')

        figs["fig_" + title.lower().replace(' ', '_')] = grid.fig

        labels_added = set()

    return figs

