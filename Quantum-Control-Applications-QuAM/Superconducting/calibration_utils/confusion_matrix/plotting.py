import matplotlib.pyplot as plt
import numpy as np


def plot_matrix_figure(
    qubit_groups,
    state_labels,
    matrix_by_group,
    title_fn,
    num_qubits,
    annotate_cells,
    text_fontsize,
    cmap=None,
    is_difference=False,
):
    """Plot one matrix figure for all qubit groups using shared styling."""
    num_groups = len(qubit_groups)
    num_cols = 3
    num_rows = int(np.ceil(num_groups / num_cols))
    num_states = len(state_labels)
    fig = plt.figure(figsize=(5 * num_cols, 5 * num_rows))

    for idx, qg in enumerate(qubit_groups):
        ax = fig.add_subplot(num_rows, num_cols, idx + 1)
        matrix = matrix_by_group[qg.name]

        if is_difference:
            max_abs = np.max(np.abs(matrix))
            mesh = ax.pcolormesh(
                state_labels, state_labels, matrix, cmap=cmap or "RdBu", vmin=-max_abs, vmax=max_abs
            )
        else:
            mesh = ax.pcolormesh(state_labels, state_labels, matrix, cmap=cmap)

        if annotate_cells:
            for i in range(num_states):
                for j in range(num_states):
                    val = matrix[i][j]
                    if is_difference and abs(val) <= 0.01:
                        continue
                    if is_difference:
                        color = "k" if abs(val) < 0.5 * max_abs else "w"
                    else:
                        color = "k" if i == j else "w"
                    ax.text(
                        j,
                        i,
                        f"{100 * val:.1f}%",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=text_fontsize,
                    )

        ax.set_ylabel("prepared")
        ax.set_xlabel("measured")
        ax.set_title(title_fn(qg.name, num_qubits))
        fig.colorbar(mesh, ax=ax)

    fig.tight_layout()
    fig.show()
    return fig
