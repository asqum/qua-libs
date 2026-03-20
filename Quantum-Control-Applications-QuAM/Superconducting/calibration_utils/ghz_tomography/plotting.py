import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def _adaptive_tick_spec(n_qubits: int, dim: int):
    """Return tick indices and labels adapted to matrix size."""
    if n_qubits <= 3:
        idx = np.arange(dim)
        labels = [format(i, f"0{n_qubits}b") for i in idx]
        return idx, labels

    # Keep at most ~10 ticks for readability.
    max_ticks = 10
    step = max(1, int(np.ceil(dim / max_ticks)))
    idx = np.arange(0, dim, step)
    if idx[-1] != dim - 1:
        idx = np.append(idx, dim - 1)

    if n_qubits <= 4:
        labels = [format(i, f"0{n_qubits}b") for i in idx]
    else:
        # Compact labels for larger systems.
        labels = [f"0x{i:X}" for i in idx]
    return idx, labels


def plot_3d_component(data, ideal, n_qubits, title="", component="real"):
    """Plot a 3D bar comparison of reconstructed vs ideal matrix component.

    Parameters
    ----------
    data : numpy.ndarray
        Reconstructed density matrix.
    ideal : numpy.ndarray
        Ideal/reference density matrix.
    n_qubits : int
        Number of qubits.
    title : str, optional
        Figure title prefix.
    component : {"real", "imag"}, optional
        Matrix component to plot.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """
    dim = 2**n_qubits
    tick_idx, tick_labels = _adaptive_tick_spec(n_qubits, dim)

    fig = plt.figure(figsize=(max(6, dim), 5))
    ax = fig.add_subplot(111, projection="3d")

    xpos, ypos = np.meshgrid(np.arange(dim) + 0.5, np.arange(dim) + 0.5, indexing="ij")
    xpos, ypos = xpos.ravel(), ypos.ravel()
    zpos = np.zeros_like(xpos)

    if component == "real":
        dz = np.real(data).ravel()
        dzi = np.real(ideal).ravel()
        component_title = "Real part"
    elif component == "imag":
        dz = np.imag(data).ravel()
        dzi = np.imag(ideal).ravel()
        component_title = "Imag part"
    else:
        raise ValueError(f"Unknown component '{component}', expected 'real' or 'imag'.")

    colors = [(0.1, 0.1, 0.6), (0.55, 0.55, 1.0)]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    gmin = min(np.min(dz), np.min(dzi))
    gmax = max(np.max(dz), np.max(dzi))

    ax.bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dz, color=cmap((np.sign(dz) + 1) / 2), alpha=1)
    ax.bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dzi, alpha=0.15, edgecolor="k")

    ax.set_xticks(tick_idx + 1)
    ax.set_yticks(tick_idx + 1)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels, rotation=45)

    ax.set_zlim([gmin, gmax])
    ax.set_title(title + f"\n ({component_title})")
    return fig


def plot_density_heatmap(data, n_qubits, title="", component="real", annotate_values=None):
    """Plot an annotated heatmap of one density-matrix component.

    Parameters
    ----------
    data : numpy.ndarray
        Density matrix to visualize.
    n_qubits : int
        Number of qubits.
    title : str, optional
        Figure title prefix.
    component : {"real", "imag"}, optional
        Matrix component to plot.
    annotate_values : bool, optional
        Whether to annotate each matrix entry value. If None, auto-enables
        for up to 3 qubits and disables for larger systems.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """
    dim = 2**n_qubits
    tick_idx, tick_labels = _adaptive_tick_spec(n_qubits, dim)

    if component == "real":
        rho_component = np.real(data)
        component_title = "Real part"
    elif component == "imag":
        rho_component = np.imag(data)
        component_title = "Imag part"
    else:
        raise ValueError(f"Unknown component '{component}', expected 'real' or 'imag'.")

    if annotate_values is None:
        annotate_values = n_qubits <= 3

    fig, ax = plt.subplots(figsize=(max(6, dim * 0.6), max(5, dim * 0.55)))
    ax.pcolormesh(rho_component, vmin=-0.5, vmax=0.5, cmap="RdBu")

    if annotate_values:
        for i in range(dim):
            for j in range(dim):
                value = rho_component[i][j]
                color = "k" if np.abs(value) < 0.1 else "w"
                ax.text(i + 0.5, j + 0.5, f"{value:.2f}", ha="center", va="center", color=color)

    ax.set_title(title + f"\n({component_title})")
    ax.set_xlabel("Computational basis")
    ax.set_ylabel("Computational basis")
    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    return fig
