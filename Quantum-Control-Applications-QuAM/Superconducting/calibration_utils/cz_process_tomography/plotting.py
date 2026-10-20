from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .analysis import INPUT_LABELS, PAULI_2Q_LABELS

ESTIMATE_TITLES = {"linear_inversion": "Linear inversion (raw)", "physical_fit": "Physical fit (CP + trace constraint)"}


def _symmetric_limit(*arrays, floor: float = 1e-3) -> float:
    return max(floor, *(float(np.max(np.abs(a))) for a in arrays))


def _city(ax, values: np.ndarray, zlim: float, title: str, labels=PAULI_2Q_LABELS, cmap="RdBu_r"):
    n = values.shape[0]
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    dz = values.ravel()
    colors = plt.get_cmap(cmap)(Normalize(-zlim, zlim)(dz))
    ax.bar3d(xs.ravel(), ys.ravel(), np.zeros_like(dz), 0.6, 0.6, dz, color=colors, shade=True, linewidth=0)
    ax.set_zlim(-zlim, zlim)
    ticks = np.arange(n) + 0.3
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_yticklabels(labels, fontsize=6)
    ax.tick_params(axis="z", labelsize=7)
    ax.set_xlabel("P_m", fontsize=8, labelpad=6)
    ax.set_ylabel("P_n", fontsize=8, labelpad=6)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=30, azim=-55)


def plot_chi_city(chi_exp: np.ndarray, chi_ideal: np.ndarray, title: str):
    """3D bar plots of Re/Im chi for ideal, experimental and their difference.

    Axes are labelled with the two-qubit Pauli basis in the reconstruction order (first letter =
    control). Ideal and experimental panels share one z-scale; the difference panels share their
    own (smaller) symmetric z-scale, printed in the panel titles.
    """
    diff = chi_exp - chi_ideal
    zlim = _symmetric_limit(chi_exp.real, chi_exp.imag, chi_ideal.real, chi_ideal.imag)
    zlim_diff = _symmetric_limit(diff.real, diff.imag)
    panels = [
        (chi_ideal.real, zlim, "Re(χ_ideal)"),
        (chi_exp.real, zlim, "Re(χ_exp)"),
        (diff.real, zlim_diff, f"Re(χ_exp − χ_ideal)  [z: ±{zlim_diff:.3f}]"),
        (chi_ideal.imag, zlim, "Im(χ_ideal)"),
        (chi_exp.imag, zlim, "Im(χ_exp)"),
        (diff.imag, zlim_diff, f"Im(χ_exp − χ_ideal)  [z: ±{zlim_diff:.3f}]"),
    ]
    fig = plt.figure(figsize=(19, 12))
    for idx, (values, lim, panel_title) in enumerate(panels):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
        _city(ax, values, lim, panel_title)
    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.92, wspace=0.05, hspace=0.12)
    return fig


def _heatmap(ax, values, vlim, title, xlabels, ylabels, cmap="RdBu_r"):
    im = ax.imshow(values, cmap=cmap, vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax.set_xticks(range(len(xlabels)))
    ax.set_yticks(range(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_title(title, fontsize=10)
    return im


def plot_ptm(ptm_exp: np.ndarray, ptm_ideal: np.ndarray, title: str):
    """Heatmaps of the ideal PTM, the experimental PTM and their difference (rows: output P_i, columns: input P_j)."""
    diff = ptm_exp - ptm_ideal
    vlim_diff = _symmetric_limit(diff)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, values, vlim, panel_title in zip(
        axes,
        (ptm_ideal, ptm_exp, diff),
        (1.0, 1.0, vlim_diff),
        ("PTM ideal", "PTM experimental", "PTM experimental − ideal"),
    ):
        im = _heatmap(ax, values, vlim, panel_title, PAULI_2Q_LABELS, PAULI_2Q_LABELS)
        ax.set_xlabel("input Pauli P_j")
        ax.set_ylabel("output Pauli P_i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_pauli_expectations(expectations: np.ndarray, expectations_ideal: np.ndarray, title: str):
    """Output-state Pauli expectation values (rows: input state |control,target>, columns: Pauli) vs ideal CZ."""
    diff = expectations - expectations_ideal
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, values, vlim, panel_title in zip(
        axes,
        (expectations_ideal, expectations, diff),
        (1.0, 1.0, _symmetric_limit(diff)),
        ("⟨P⟩ ideal CZ", "⟨P⟩ measured", "measured − ideal"),
    ):
        im = _heatmap(ax, values, vlim, panel_title, PAULI_2Q_LABELS, [f"|{s}⟩" for s in INPUT_LABELS])
        ax.set_xlabel("Pauli (control, target)")
        ax.set_ylabel("input state")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def _format_summary_lines(metrics: Dict[str, Dict[str, float]], cz_model: Optional[Dict], leakage_text: str):
    names = list(metrics)
    header = f"{'':32s}" + "".join(f"{ESTIMATE_TITLES.get(n, n).split(' (')[0]:>20s}" for n in names)
    rows = [
        ("Process fidelity F_pro", "process_fidelity", "{:.4f}"),
        ("Average gate fidelity F_avg", "average_gate_fidelity", "{:.4f}"),
        ("||χ_exp − χ_ideal||_F", "chi_frobenius_distance", "{:.4f}"),
        ("Trace-preservation error", "trace_preservation_error", "{:.2e}"),
        ("Choi min eigenvalue (/d)", "choi_min_eigenvalue", "{:+.2e}"),
        ("Fit residual (rms)", "residual_rms", "{:.2e}"),
    ]
    lines = [header]
    for label, key, fmt in rows:
        lines.append(f"{label:32s}" + "".join(f"{fmt.format(metrics[n][key]):>20s}" for n in names))
    lines.append("")
    if cz_model is not None:
        deg = np.degrees
        lines += [
            f"CZ model on {cz_model['estimate']}:",
            "  U_eff = [Z(φ1)⊗Z(φ2)] · fSim(θ, φ_CZ)",
            f"  Conditional phase φ_CZ         {deg(cz_model['phi_cz']):8.2f}°",
            f"  CZ phase error φ_CZ − π        {deg(cz_model['cz_phase_error']):+8.2f}°",
            f"  Local Z phase control φ1       {deg(cz_model['phi_control']):+8.2f}°",
            f"  Local Z phase target  φ2       {deg(cz_model['phi_target']):+8.2f}°",
            f"  Residual SWAP angle θ          {deg(cz_model['theta']):+8.2f}°",
            f"  F_pro(ideal CZ)                {cz_model['process_fidelity_ideal_cz']:8.4f}",
            f"  F_pro(local-Z corrected CZ)    {cz_model['process_fidelity_local_z_corrected']:8.4f}",
            f"  F_pro(fitted U_eff)            {cz_model['process_fidelity_model']:8.4f}",
            "",
        ]
    lines.append(leakage_text)
    lines.append("QPT includes SPAM errors; F_pro is not equivalent to an RB fidelity.")
    return lines


def plot_summary(
    metrics: Dict[str, Dict[str, float]],
    error_probabilities: Dict[str, np.ndarray],
    cz_model: Optional[Dict],
    leakage_text: str,
    title: str,
):
    """Summary table of all figures of merit next to the error-channel Pauli probabilities."""
    fig, (ax_text, ax_bar) = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax_text.axis("off")
    lines = _format_summary_lines(metrics, cz_model, leakage_text)
    ax_text.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=9.5, va="top", ha="left", transform=ax_text.transAxes)

    labels = PAULI_2Q_LABELS[1:]
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(error_probabilities))
    for idx, (name, probabilities) in enumerate(error_probabilities.items()):
        ax_bar.bar(x + (idx - (len(error_probabilities) - 1) / 2) * width, probabilities[1:], width, label=ESTIMATE_TITLES.get(name, name))
    ax_bar.axhline(0, color="k", lw=0.6)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=90)
    ax_bar.set_ylabel("probability")
    ax_bar.set_title("Error-channel Pauli probabilities  diag χ(Λ ∘ CZ⁻¹), II omitted")
    ax_bar.grid(axis="y", ls=":", alpha=0.7)
    ax_bar.legend(fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig
