# %%
"""
        IQ BLOBS
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit
in the |g> state) and then after applying a pi pulse to the qubit (bringing the qubit to the |e> state) successively.
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the separation between |g> and |e> states
      aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the state.
    - Update the g -> e threshold (ge_threshold) in the state.
    - Save the current state by calling machine.save("quam")
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset_gef, active_reset
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_runs: int = 2000
    max_attempts: int = 30
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    multiplexed: bool = False
    simulate: bool = False
    timeout: int = 100  
    connect_timeout: int = 200


node = QualibrationNode(name="08x_IQ_Blobs_GEF_check_active_reset", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect(timeout=node.parameters.connect_timeout)

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

for q in qubits:
    # Check if an optimized GEF frequency exists
    if not hasattr(q, "GEF_frequency_shift"):
        q.GEF_frequency_shift = 0
    # check if an EF_x180 operation exists
    if "EF_x180" in q.xy.operations:
        GEF_operation = "EF_x180"
    else:
        GEF_operation = "x180"


### Helper functions
def find_biggest_gaussian(da):
    # Define Gaussian function
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

    # Get histogram data
    hist, bin_edges = np.histogram(da, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit multiple Gaussians
    initial_guess = [(hist.max(), bin_centers[hist.argmax()], (bin_centers[-1] - bin_centers[0]) / 4)]
    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)


    # Find the biggest Gaussian
    biggest_gaussian = {'amp': popt[0], 'mu': popt[1], 'sigma': popt[2]}
    
    return biggest_gaussian['mu']


def thermal_blob_circle(I, Q):
    """Center and RMS radius of a thermal blob. Center matches find_biggest_gaussian."""
    I = np.asarray(I, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float).ravel()
    center_I = find_biggest_gaussian(I)
    center_Q = find_biggest_gaussian(Q)
    radius = np.sqrt(np.mean((I - center_I) ** 2 + (Q - center_Q) ** 2))
    return center_I, center_Q, radius


def nearest_center_fractions(I, Q, circles):
    """Fraction of shots closest in Manhattan distance. The three fractions sum to 1."""
    I = np.asarray(I, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float).ravel()
    dist = np.stack(
        [np.abs(I - center_I) + np.abs(Q - center_Q) for center_I, center_Q, *_rest in circles],
        axis=0,
    )
    labels = np.argmin(dist, axis=0)
    return [float(np.mean(labels == state)) for state in range(3)]


def fraction_suffix(fracs):
    return " ".join(f"{name} {100 * frac:.0f}%" for name, frac in zip("gef", fracs))


# %% {QUA_program}
n_runs = node.parameters.num_runs  # Number of runs
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'


with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)
    I_f, I_f_st, Q_f, Q_f_st, _, _ = qua_declaration(num_qubits=num_qubits)
    I_etg, I_etg_st, Q_etg, Q_etg_st, _, _ = qua_declaration(num_qubits=num_qubits)  # e->g thermal
    I_eag, I_eag_st, Q_eag, Q_eag_st, _, _ = qua_declaration(num_qubits=num_qubits)  # e->g active
    I_ftg, I_ftg_st, Q_ftg, Q_ftg_st, _, _ = qua_declaration(num_qubits=num_qubits)  # f->g thermal
    I_fag, I_fag_st, Q_fag, Q_fag_st, _, _ = qua_declaration(num_qubits=num_qubits)  # f->g active
    
    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qubit)
        qubit.resonator.update_frequency(
            qubit.resonator.intermediate_frequency + qubit.resonator.GEF_frequency_shift
        )

        with for_(n, 0, n < n_runs, n + 1):
            # ground iq blobs for all qubits
            align()
            save(n, n_st)
            update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency, keep_phase=True)

            # thermal g — same as 08d: no drive, measure after thermalization
            wait(4 * qubit.thermalization_time * u.ns)
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]))
            qubit.align()
            save(I_g[i], I_g_st[i])
            save(Q_g[i], Q_g_st[i])

            # thermal e — same prep as 08d
            wait(4*qubit.thermalization_time * u.ns)
            qubit.align()
            qubit.xy.play("x180")
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]))
            qubit.align()
            save(I_e[i], I_e_st[i])
            save(Q_e[i], Q_e_st[i])

            # thermal reset e -> g
            wait(4 * qubit.thermalization_time * u.ns)
            qubit.align()
            qubit.xy.play("x180")
            wait(4 * qubit.thermalization_time * u.ns)  # thermalize
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I_etg[i], Q_etg[i]))
            qubit.align()
            save(I_etg[i], I_etg_st[i]); save(Q_etg[i], Q_etg_st[i])

            # thermal f — same prep as 08d
            wait(4*qubit.thermalization_time * u.ns)
            qubit.align()
            qubit.xy.play("x180")
            update_frequency(
                qubit.xy.name, qubit.xy.intermediate_frequency - qubit.anharmonicity, keep_phase=True
            )
            qubit.xy.play(GEF_operation)
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I_f[i], Q_f[i]))
            qubit.align()
            save(I_f[i], I_f_st[i])
            save(Q_f[i], Q_f_st[i])

            # thermal reset f -> g. Previous |f> prep left xy at the EF frequency.
            update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency, keep_phase=True)
            wait(4*qubit.thermalization_time * u.ns)
            qubit.align()
            qubit.xy.play("x180")
            update_frequency(
                qubit.xy.name, qubit.xy.intermediate_frequency - qubit.anharmonicity, keep_phase=True
            )
            qubit.xy.play(GEF_operation)
            wait(4 * qubit.thermalization_time * u.ns)  # thermalize
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I_ftg[i], Q_ftg[i]))
            qubit.align()
            save(I_ftg[i], I_ftg_st[i]); save(Q_ftg[i], Q_ftg_st[i])

            # active reset e -> g. Previous |f> prep left xy at the EF frequency.
            update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency, keep_phase=True)
            wait(4 * qubit.thermalization_time * u.ns)
            qubit.align()
            qubit.xy.play("x180")
            active_reset_gef(qubit, max_attempts=node.parameters.max_attempts)
            qubit.resonator.update_frequency(
                qubit.resonator.intermediate_frequency + qubit.resonator.GEF_frequency_shift,
                keep_phase=True,
            )
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I_eag[i], Q_eag[i]))
            qubit.align()
            save(I_eag[i], I_eag_st[i]); save(Q_eag[i], Q_eag_st[i])

            # active reset f -> g. active_reset_gef leaves xy at the GE frequency.
            wait(4*qubit.thermalization_time * u.ns)
            qubit.align()
            qubit.xy.play("x180")
            update_frequency(
                qubit.xy.name, qubit.xy.intermediate_frequency - qubit.anharmonicity, keep_phase=True
            )
            qubit.xy.play(GEF_operation)
            active_reset_gef(qubit, max_attempts=node.parameters.max_attempts)
            qubit.resonator.update_frequency(
                qubit.resonator.intermediate_frequency + qubit.resonator.GEF_frequency_shift,
                keep_phase=True,
            )
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I_fag[i], Q_fag[i]))
            qubit.align()
            save(I_fag[i], I_fag_st[i]); save(Q_fag[i], Q_fag_st[i])

        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].save_all(f"I_g{i + 1}")
            Q_g_st[i].save_all(f"Q_g{i + 1}")
            I_e_st[i].save_all(f"I_e{i + 1}")
            Q_e_st[i].save_all(f"Q_e{i + 1}")
            I_etg_st[i].save_all(f"I_etg{i + 1}")
            Q_etg_st[i].save_all(f"Q_etg{i + 1}")
            I_eag_st[i].save_all(f"I_eag{i + 1}")
            Q_eag_st[i].save_all(f"Q_eag{i + 1}")
            I_f_st[i].save_all(f"I_f{i + 1}")
            Q_f_st[i].save_all(f"Q_f{i + 1}")
            I_ftg_st[i].save_all(f"I_ftg{i + 1}")
            Q_ftg_st[i].save_all(f"Q_ftg{i + 1}")
            I_fag_st[i].save_all(f"I_fag{i + 1}")
            Q_fag_st[i].save_all(f"Q_fag{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, iq_blobs, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(iq_blobs)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_runs, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(
        job.result_handles, qubits, {"N": np.linspace(1, n_runs, n_runs)}
    )

    # Fix the structure of ds to avoid tuples
    def extract_value(element):
        if isinstance(element, tuple):
            return element[0]
        return element

    ds = xr.apply_ufunc(
        extract_value,
        ds,
        vectorize=True,  # This ensures the function is applied element-wise
        dask="parallelized",  # This allows for parallel processing
        output_dtypes=[float],  # Specify the output data type
    )

    node.results = {"ds": ds, "results": {}}

# %% {Plotting}
if not node.parameters.simulate:
    import matplotlib.pyplot as plt

node.results = {"ds": ds, "figs": {}, "results": {}}

for q in qubits:
    qn = q.name

    # refs
    I_g, Q_g = ds.I_g.sel(qubit=qn), ds.Q_g.sel(qubit=qn)
    I_e, Q_e = ds.I_e.sel(qubit=qn), ds.Q_e.sel(qubit=qn)
    I_f, Q_f = ds.I_f.sel(qubit=qn), ds.Q_f.sel(qubit=qn)

    # return-to-g checks from e
    I_etg, Q_etg = ds.I_etg.sel(qubit=qn), ds.Q_etg.sel(qubit=qn)   # e -> g (thermal)
    I_eag, Q_eag = ds.I_eag.sel(qubit=qn), ds.Q_eag.sel(qubit=qn)   # e -> g (active)
    I_ftg, Q_ftg = ds.I_ftg.sel(qubit=qn), ds.Q_ftg.sel(qubit=qn)   # f -> g (thermal)
    I_fag, Q_fag = ds.I_fag.sel(qubit=qn), ds.Q_fag.sel(qubit=qn)   # f -> g (active)

    fig, axes = plt.subplots(1, 5, figsize=(15, 4.6), layout="constrained")
    fig.suptitle(f"{qn}  active_reset_gef", fontsize=12)

    thermal_circles = [
        (*thermal_blob_circle(I_g, Q_g), "blue", "g"),
        (*thermal_blob_circle(I_e, Q_e), "orange", "e"),
        (*thermal_blob_circle(I_f, Q_f), "green", "f"),
    ]
    populations = {}

    def scatter_population(ax, I, Q, label, color, alpha, key=None):
        if key is None:
            ax.scatter(I, Q, s=5, alpha=alpha, label=label, color=color)
            return
        fracs = nearest_center_fractions(I, Q, thermal_circles)
        populations[key] = fracs
        ax.scatter(
            I, Q, s=5, alpha=alpha, color=color,
            label=f"{label} | {fraction_suffix(fracs)}",
        )

    # 1) Thermal refs (g/e/f)
    scatter_population(axes[0], I_g, Q_g, "g (thermal)", "blue", 0.5, "g_thermal")
    scatter_population(axes[0], I_e, Q_e, "e (thermal)", "orange", 0.5, "e_thermal")
    scatter_population(axes[0], I_f, Q_f, "f (thermal)", "green", 0.5, "f_thermal")
    axes[0].set_title(f"{qn} – Refs (g/e/f)")
    axes[0].set_xlabel("I"); axes[0].set_ylabel("Q")

    # 2) e -> g (thermal reset)
    scatter_population(axes[1], I_g, Q_g, "g (ref)", "blue", 0.2)
    scatter_population(axes[1], I_etg, Q_etg, "e → g (thermal)", "orange", 0.6, "e_to_g_thermal")
    axes[1].set_title(f"{qn} – e → g (Thermal reset)")
    axes[1].set_xlabel("I"); axes[1].set_ylabel("Q")

    # 3) f -> g (thermal reset)
    scatter_population(axes[2], I_g, Q_g, "g (ref)", "blue", 0.2)
    scatter_population(axes[2], I_ftg, Q_ftg, "f → g (thermal)", "green", 0.6, "f_to_g_thermal")
    axes[2].set_title(f"{qn} – f → g (Thermal reset)")
    axes[2].set_xlabel("I"); axes[2].set_ylabel("Q")

    # 4) e -> g (active reset)
    scatter_population(axes[3], I_g, Q_g, "g (ref)", "blue", 0.2)
    scatter_population(axes[3], I_eag, Q_eag, "e → g (active_reset_gef)", "orange", 0.6, "e_to_g_active")
    axes[3].set_title(f"{qn} – e → g (active_reset_gef)")
    axes[3].set_xlabel("I"); axes[3].set_ylabel("Q")

    # 5) f -> g (active reset)
    scatter_population(axes[4], I_g, Q_g, "g (ref)", "blue", 0.2)
    scatter_population(axes[4], I_fag, Q_fag, "f → g (active_reset_gef)", "green", 0.6, "f_to_g_active")
    axes[4].set_title(f"{qn} – f → g (active_reset_gef)")
    axes[4].set_xlabel("I"); axes[4].set_ylabel("Q")

    node.results["results"][qn] = populations

    x0, x1 = axes[0].get_xlim()
    y0, y1 = axes[0].get_ylim()
    half = 0.5 * max(x1 - x0, y1 - y0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    ref_xlim = (cx - half, cx + half)
    ref_ylim = (cy - half, cy + half)

    for ax in axes:
        ax.set_xlim(ref_xlim)
        ax.set_ylim(ref_ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.autoscale(False)
        ax.grid(True)
        n_labels = len(ax.get_legend_handles_labels()[1])
        ax.set_title(ax.get_title(), fontsize=9, pad=4 + 11 * n_labels)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            borderaxespad=0,
            frameon=False,
            fontsize=6,
        )
    node.results["figs"][qn] = fig
    plt.show()


# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
    
# %%# %%
