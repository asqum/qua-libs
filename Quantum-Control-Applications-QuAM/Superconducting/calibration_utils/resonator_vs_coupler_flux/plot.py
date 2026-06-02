from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from quam_libs.lib.plot_utils import QubitGrid, grid_iter

from .analysis import DecoupleOffsetAnalysis

__all__ = [
    "plot_decouple_offset_maps",
    "plot_resonator_spectroscopy_vs_coupler_flux",
    "plot_multi_pair_resonator_spectroscopy_vs_coupler_flux",
]


def plot_decouple_offset_maps(
    analysis: DecoupleOffsetAnalysis,
    data_id: Optional[int] = None,
    figsize_per_col: float = 4.2,
    row_height: float = 4.0,
    show: bool = True,
):
    """Plot detrended maps with branch offsets (cyan) and selected decouple offset (gold)."""
    n_pairs = len(analysis.pair_order)
    if n_pairs == 0:
        raise ValueError("No pairs to plot.")

    fig, axes = plt.subplots(
        2,
        n_pairs,
        figsize=(figsize_per_col * n_pairs, row_height * 2),
        sharex="col",
        squeeze=False,
    )

    for col, pair_name in enumerate(analysis.pair_order):
        pair = analysis.pairs[pair_name]
        decouple_mV = pair.decouple_offset_mV
        decouple_from = pair.selected_qubit

        for row, qubit in enumerate(pair.qubits):
            ax = axes[row, col]
            branch = pair.branches[qubit]

            im = ax.pcolormesh(
                pair.flux_mV,
                branch.freq_GHz,
                branch.detrended.T,
                shading="auto",
                cmap="viridis",
            )
            plt.colorbar(im, ax=ax, label="|IQ|", pad=0.02)

            ax.axvline(
                branch.offset_mV,
                color="cyan",
                linestyle="--",
                linewidth=1.4,
                label=f"branch {branch.offset_mV:.0f} mV",
            )
            ax.axvline(
                decouple_mV,
                color="gold",
                linestyle="-",
                linewidth=2.4,
                label=f"decouple {decouple_mV:.0f} mV ({decouple_from})",
            )

            ax.set_ylabel("GHz")
            title = pair_name if row == 0 else ""
            ax.set_title(f"{title}\n{qubit}" if title else qubit, fontsize=9)
            ax.legend(loc="upper right", fontsize=6)

    fig.supxlabel("Coupler flux (mV)")
    title = "decouple_offset maps"
    if data_id is not None:
        title = f"data {data_id} — cyan: branch offset | gold: decouple_offset"
    fig.suptitle(title, y=1.01)
    plt.tight_layout()

    if show:
        plt.show()
    return fig


def plot_resonator_spectroscopy_vs_coupler_flux(
    ds,
    qubits,
    qubit_pair,
    fit_results,
    show: bool = True,
):
    """Plot resonator spectroscopy vs coupler flux with detected dips and fits."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    legend_handles = []
    legend_labels = []

    for ax, qubit in grid_iter(grid):
        ds.loc[qubit].IQ_abs.plot(
            ax=ax,
            x="flux_mV",
            y="freq_GHz",
            robust=True,
            add_colorbar=False,
            cmap="viridis",
        )
        qubit_name = qubit["qubit"].item() if hasattr(qubit["qubit"], "item") else qubit["qubit"]

        dip_freq = ds.resonator_dip_freq_abs_GHz.sel(qubit=qubit_name)
        valid_dips = np.isfinite(dip_freq.values)
        if np.any(valid_dips):
            ax.scatter(
                ds.flux_mV.values[valid_dips],
                dip_freq.values[valid_dips],
                s=14,
                c="red",
                edgecolors="white",
                linewidths=0.4,
                label="detected dips",
                zorder=3,
            )

        ax.plot(
            ds.coupler_fit_flux_mV.values,
            ds.hamiltonian_fit_freq_GHz.sel(qubit=qubit_name).values,
            color="white",
            linestyle="--",
            linewidth=2.0,
            label="Hamiltonian fit",
            zorder=4,
        )

        fitted_decouple = float(ds.coupler_decouple_offset.sel(qubit=qubit_name))
        if np.isfinite(fitted_decouple):
            fitted_decouple_freq = float(
                ds.hamiltonian_fit_freq_GHz.sel(qubit=qubit_name).interp(coupler_fit_flux=fitted_decouple)
            )
            ax.axvline(
                fitted_decouple * 1e3,
                color="cyan",
                linestyle=":",
                linewidth=1.6,
                label="fitted decouple",
                zorder=5,
            )
            ax.scatter(
                fitted_decouple * 1e3,
                fitted_decouple_freq,
                marker="*",
                s=100,
                c="cyan",
                edgecolors="black",
                linewidths=0.7,
                zorder=6,
            )

        current_decouple = getattr(qubit_pair.coupler, "decouple_offset", np.nan)
        if np.isfinite(current_decouple):
            ax.axvline(
                current_decouple * 1e3,
                color="orange",
                linestyle="-",
                linewidth=1.4,
                label="current decouple",
                zorder=5,
            )

        selected_decouple = fit_results["selected_decouple_offset"]
        if np.isfinite(selected_decouple):
            ax.axvline(
                selected_decouple * 1e3,
                color="black",
                linestyle="-.",
                linewidth=1.5,
                label="R2 selected decouple",
                zorder=5,
            )

        ax.set_ylabel("Freq (GHz)")
        ax.set_xlabel("Coupler flux (mV)")
        ax.set_title(f"{qubit_name} - {qubit_pair.coupler.name}")

        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)

    grid.fig.suptitle("Resonator spectroscopy vs coupler flux")
    if legend_handles:
        grid.fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(legend_labels),
            fontsize=8,
            frameon=True,
        )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    if show:
        plt.show()
    return grid.fig


def plot_multi_pair_resonator_spectroscopy_vs_coupler_flux(
    ds,
    qubit_pairs,
    fit_results_by_pair,
    show: bool = True,
):
    """Plot analyzed resonator spectroscopy vs coupler flux in a compact pair layout."""
    if not qubit_pairs:
        raise ValueError("No qubit pairs were provided for plotting.")

    pairs_per_row = min(len(qubit_pairs), 3)
    n_rows = int(np.ceil(len(qubit_pairs) / pairs_per_row))
    n_cols = 2 * pairs_per_row
    fig_width = max(7.0, 3.0 * n_cols)
    fig_height = 3.9 * n_rows + 0.9
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    legend_handles = []
    legend_labels = []

    for pair_index, qp in enumerate(qubit_pairs):
        qp_name = qp.name
        pair_ds = ds.sel(qubit_pair=qp_name)
        row = pair_index // pairs_per_row
        col = 2 * (pair_index % pairs_per_row)
        role_axes = axes[row, col : col + 2]

        for role, role_ax in zip(["control", "target"], role_axes):
            role_ds = pair_ds.sel(qubit=role)
            qubit_name = _role_qubit_name(pair_ds, role)
            role_ds.IQ_abs.plot(
                ax=role_ax,
                x="flux_mV",
                y="freq_GHz",
                robust=True,
                add_colorbar=False,
                cmap="viridis",
            )
            _plot_fit_overlays(role_ax, role_ds, qp, fit_results_by_pair[qp_name])

            role_ax.set_ylabel("")
            role_ax.set_xlabel("Coupler flux (mV)")
            role_ax.set_title(f"{qp_name}: {role} {qubit_name}", fontsize=9)
            role_ax.tick_params(axis="both", labelsize=9, pad=2)

            handles, labels = role_ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)

    for ax in axes.flat[2 * len(qubit_pairs) :]:
        ax.set_axis_off()

    fig.suptitle("Resonator spectroscopy vs coupler flux")
    fig.supylabel("Freq (GHz)", x=0.018, fontsize=10)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(legend_labels),
            fontsize=8,
            frameon=True,
        )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.17 if legend_handles else 0.12,
        top=0.88,
        wspace=0.24,
        hspace=0.38,
    )
    if show:
        plt.show()
    return fig


def _plot_fit_overlays(ax, role_ds, qubit_pair, fit_results):
    dip_freq = role_ds.resonator_dip_freq_abs_GHz
    valid_dips = np.isfinite(dip_freq.values)
    if np.any(valid_dips):
        ax.scatter(
            role_ds.flux_mV.values[valid_dips],
            dip_freq.values[valid_dips],
            s=12,
            c="red",
            edgecolors="white",
            linewidths=0.4,
            label="detected dips",
            zorder=3,
        )

    ax.plot(
        role_ds.coupler_fit_flux_mV.values,
        role_ds.hamiltonian_fit_freq_GHz.values,
        color="white",
        linestyle="--",
        linewidth=1.6,
        label="Hamiltonian fit",
        zorder=4,
    )

    fitted_decouple = _finite_float(role_ds.coupler_decouple_offset)
    if np.isfinite(fitted_decouple):
        fitted_decouple_freq = float(role_ds.hamiltonian_fit_freq_GHz.interp(coupler_fit_flux=fitted_decouple))
        ax.axvline(
            fitted_decouple * 1e3,
            color="cyan",
            linestyle=":",
            linewidth=1.4,
            label="fitted decouple",
            zorder=5,
        )
        ax.scatter(
            fitted_decouple * 1e3,
            fitted_decouple_freq,
            marker="*",
            s=80,
            c="cyan",
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
        )

    current_decouple = _finite_float(getattr(qubit_pair.coupler, "decouple_offset", np.nan))
    if np.isfinite(current_decouple):
        ax.axvline(
            current_decouple * 1e3,
            color="orange",
            linestyle="-",
            linewidth=1.2,
            label="current decouple",
            zorder=5,
        )

    selected_decouple = _finite_float(fit_results["selected_decouple_offset"])
    if np.isfinite(selected_decouple):
        ax.axvline(
            selected_decouple * 1e3,
            color="black",
            linestyle="-.",
            linewidth=1.2,
            label="R2 selected decouple",
            zorder=5,
        )


def _role_qubit_name(pair_ds, role):
    if "qubit_name" in pair_ds.coords:
        return str(pair_ds.qubit_name.sel(qubit=role).item())
    return role


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan
