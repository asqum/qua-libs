import matplotlib.pyplot as plt


def plot_zbasis_distributions(
    groups,
    corrected_data,
    fidelities,
    state_labels,
    num_qubits,
    all_0_label,
    all_1_label,
    title_fn,
    bar_color,
    edge_color,
    fidelity_box_facecolor,
    fig_width,
    fidelity_differences=None,
    delta_text_fn=None,
):
    """Plot corrected Z-basis state distributions for one mitigation method."""
    num_groups = len(groups)
    if num_groups == 1:
        fig, axes = plt.subplots(1, figsize=(fig_width, 3))
    else:
        fig, axes = plt.subplots(num_groups, 1, figsize=(fig_width, 3 * num_groups))

    for i, qg in enumerate(groups):
        ax = axes if num_groups == 1 else axes[i]
        values = corrected_data[qg.name]

        ax.bar(state_labels, values, color=bar_color, edgecolor=edge_color)
        ax.set_ylim(0, 1)
        for j, v in enumerate(values):
            if v > 0.01:
                rotation = 90 if num_qubits >= 4 else 0
                ax.text(
                    j,
                    v,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    rotation=rotation,
                    fontsize=8 if num_qubits >= 4 else 10,
                )
        ax.set_ylabel("Probability")
        if i == num_groups - 1:
            ax.set_xlabel("State")
        if num_qubits >= 4:
            ax.tick_params(axis="x", rotation=90)

        ax.set_title(title_fn(qg))
        ax.text(
            0.02,
            0.98,
            f"Z-basis population fidelity ({all_0_label}+{all_1_label}): {fidelities[qg.name]:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=fidelity_box_facecolor, alpha=0.5),
        )

        if fidelity_differences is not None and qg.name in fidelity_differences and delta_text_fn is not None:
            diff = fidelity_differences[qg.name]
            diff_color = "orange" if diff > 0 else "red"
            ax.text(
                0.02,
                0.88,
                delta_text_fn(diff),
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor=diff_color, linewidth=2),
            )

    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(hspace=0.4)
    return fig
