from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .analysis import HOFMANN_INPUTS, INPUT_LABELS, OUTCOME_LABELS

COLOR_A = "tab:blue"
COLOR_B = "tab:orange"


def plot_input_fidelities(ax, analysis: Dict[str, object], title: str = ""):
    """Per-input fidelities F_k with binomial 1-sigma bars, the set means F_A / F_B and the Hofmann bounds.

    Bars are the raw values (the primary result). Black diamonds are the readout-corrected values, if present.
    """
    raw = analysis["raw"]
    x = np.arange(len(INPUT_LABELS))
    colors = [COLOR_A if label.startswith("A") else COLOR_B for label in INPUT_LABELS]
    ax.bar(x, raw["F"], yerr=raw["sigma"], color=colors, alpha=0.8, capsize=3, label="raw")
    if "corrected" in analysis:
        corrected = analysis["corrected"]
        ax.errorbar(x, corrected["F"], yerr=corrected["sigma"], fmt="D", color="k", ms=4, capsize=3, label="readout-corrected")
    ax.hlines(raw["F_A"], -0.45, 3.45, color="navy", ls="--", lw=1.5, label=f"F_A = {raw['F_A']:.3f}")
    ax.hlines(raw["F_B"], 3.55, 7.45, color="darkred", ls="--", lw=1.5, label=f"F_B = {raw['F_B']:.3f}")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{entry.label}\n|{entry.ket}>" for entry in HOFMANN_INPUTS], fontsize=8)
    low = float(np.min(raw["F"] - 3 * raw["sigma"]))
    high = 1.0
    if "corrected" in analysis:  # corrected values can leave [0, 1]
        low = min(low, float(np.min(analysis["corrected"]["F"] - 3 * analysis["corrected"]["sigma"])))
        high = max(high, float(np.max(analysis["corrected"]["F"] + 3 * analysis["corrected"]["sigma"])))
    ax.set_ylim(max(0.0, min(0.9, low - 0.05)), high + 0.06)
    ax.set_ylabel("fraction on the ideal CZ outcome")
    ax.set_title(
        f"{title}\n"
        f"{raw['F_lower']:.3f} (±{raw['sigma_F_lower']:.3f}) ≤ F_pro ≤ {raw['F_upper']:.3f} (±{raw['sigma_F_upper']:.3f})"
        f"   C_ML = {raw['C_ML']:.3f}",
        fontsize=9,
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4)
    return ax


def plot_outcome_probabilities(analysis: Dict[str, object], title: str = ""):
    """Measured joint-outcome probabilities of the eight inputs; the outcome of an ideal CZ is framed in red."""
    histograms = np.asarray(analysis["histograms"], dtype=float)
    probabilities = histograms / analysis["shots_per_state"]  # leaked shots (if any) are in no bin
    fig, ax = plt.subplots(figsize=(6.5, 6))
    image = ax.imshow(probabilities, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    for k, entry in enumerate(HOFMANN_INPUTS):
        for j in range(4):
            value = probabilities[k, j]
            ax.text(j, k, f"{value:.3f}", ha="center", va="center", color="w" if value > 0.55 else "k", fontsize=9)
        ax.add_patch(Rectangle((entry.expected_index - 0.5, k - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))
    ax.set_xticks(range(4))
    ax.set_xticklabels(OUTCOME_LABELS)
    ax.set_yticks(range(len(HOFMANN_INPUTS)))
    ax.set_yticklabels([f"{entry.label}  |{entry.ket}> ({entry.basis})" for entry in HOFMANN_INPUTS])
    ax.set_xlabel("measured outcome  (control target)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="probability")
    fig.tight_layout()
    return fig
