# %%
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state

import matplotlib.pyplot as plt
import numpy as np

from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_decay_exp, decay_exp


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q3"]
    """Qubits to perform the T1 measurement on. If None or empty, all active qubits will be used."""
    num_averages: int = 50
    """The number of averages to perform."""
    min_wait_time_in_ns: int = 16
    """Minimum wait time in ns."""
    max_wait_time_in_ns: int = 50000
    """Maximum wait time in ns."""
    wait_time_step_in_ns: int = 500
    """Wait time step in ns."""
    qubit_flux_min: float = -0.2
    """Minimum qubit flux pulse amplitude relative to the decoupler offset (V)."""
    qubit_flux_max: float = 0.2
    """Maximum qubit flux pulse amplitude relative to the decoupler offset (V)."""
    qubit_flux_num_points: int = 20
    """Number of qubit flux pulse points."""
    flux_point_joint_or_independent_or_arbitrary: Literal["joint", "independent", "arbitrary"] = "independent"
    """Whether to use joint, independent or arbitrary flux points for the qubits."""
    simulate: bool = False
    """Whether to simulate the QUA program instead of executing it."""
    timeout: int = 100
    """Timeout for the QM session in seconds."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout."""
    reset_type: Literal["active", "thermal"] = "active"
    """Type of reset to use before each measurement."""


node = QualibrationNode(name="05c_T1_vs_qubit_flux", parameters=Parameters())


# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# Idle time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    node.parameters.wait_time_step_in_ns // 4,
)

fluxes_qubit = np.linspace(
    node.parameters.qubit_flux_min,
    node.parameters.qubit_flux_max,
    node.parameters.qubit_flux_num_points,
)

flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary  # 'independent' or 'joint'


with program() as t1_vs_qubit_flux:
    flux_qubit = declare(float)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the idle time
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

    for i, qubit in enumerate(qubits):
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        if "c" in qubit.id:
            qubit.z.set_dc_offset(qubit.z.joint_offset)  # for coupler-test case
        wait(1000)
        qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(flux_qubit, fluxes_qubit)):
                with for_(*from_array(t, idle_times)):
                    if not node.parameters.simulate:
                        if node.parameters.reset_type == "active":
                            active_reset(qubit, "readout")
                            qubit.align()
                        else:
                            qubit.resonator.wait(qubit.thermalization_time * u.ns)
                            qubit.align()

                    qubit.xy.play("x180")
                    qubit.z.wait(qubit.xy.operations["x180"].length // 4)

                    qubit.z.play(
                        "const",
                        amplitude_scale=flux_qubit / qubit.z.operations["const"].amplitude,
                        duration=t,
                    )
                    qubit.xy.wait(t)
                    qubit.align()

                    # Measure the state of the resonators
                    if node.parameters.use_state_discrimination:
                        readout_state(qubit, state[i])
                        save(state[i], state_st[i])
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            if node.parameters.use_state_discrimination:
                state_st[i].buffer(len(idle_times)).buffer(len(fluxes_qubit)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(idle_times)).buffer(len(fluxes_qubit)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(idle_times)).buffer(len(fluxes_qubit)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000 // 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, t1_vs_qubit_flux, simulation_config)
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(t1_vs_qubit_flux)
        # Get results from QUA program
        for i in range(num_qubits):
            print(f"Fetching results for qubit {qubits[i].name}")
            data_list = ["n"]
            results = fetching_tool(job, data_list, mode="live")
            while results.is_processing():
                fetched_data = results.fetch_all()
                n = fetched_data[0]
                progress_counter(n, n_avg, start_time=results.start_time)


# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:

    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times, "flux_qubit": fluxes_qubit})

    ds = ds.assign_coords(idle_time=4 * ds.idle_time / 1e3)  # convert to usec
    ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
    ds.flux_qubit.attrs = {"long_name": "qubit flux pulse relative to decoupler offset", "units": "V"}
    node.results = {"ds": ds}

# %% {Data_analysis}
if not node.parameters.simulate:
    ds = ds.assign_coords(flux_mV=ds.flux_qubit * 1e3)
    ds.flux_mV.attrs = {"long_name": "qubit flux pulse relative to decoupler offset", "units": "mV"}
    fit_results = {}
    fitted = None
    try:
        data = ds.state if node.parameters.use_state_discrimination else ds.I

        fit_dec = fit_decay_exp(data, "idle_time")
        fitted = decay_exp(
            ds.idle_time,
            fit_dec.sel(fit_vals="a"),
            fit_dec.sel(fit_vals="offset"),
            fit_dec.sel(fit_vals="decay"),
        )
        tau = -1 / fit_dec.sel(fit_vals="decay")
        tau_error = -tau * (
            np.sqrt(np.abs(fit_dec.sel(fit_vals="decay_decay"))) / fit_dec.sel(fit_vals="decay")
        )

        tau.attrs = {"long_name": "T1", "units": "µs"}
        tau_error.attrs = {"long_name": "T1 error", "units": "µs"}

        for q in qubits:
            fit_results[q.name] = {
                "T1_vs_qubit_flux": tau.sel(qubit=q.name).values.tolist(),
                "T1err_vs_qubit_flux": tau_error.sel(qubit=q.name).values.tolist(),
            }

        node.results["fit_results"] = fit_results
    except Exception as e:
        print("⚠️ Fit failed:", e)
        print("Proceeding with raw data only.")

# %% {Plotting}
if not node.parameters.simulate:
    flux_xlabel = "Qubit flux pulse relative to decoupler offset (mV)"

    # ---- RAW PLOT ----
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    grid.fig.set_size_inches(12, 3 * len(qubits))

    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]

        if node.parameters.use_state_discrimination:
            im = ds.sel(qubit=qname).state.plot(
                ax=ax, x="flux_mV", y="idle_time", add_colorbar=False, cmap="viridis"
            )
        else:
            im = ds.sel(qubit=qname).I.plot(
                ax=ax, x="flux_mV", y="idle_time", add_colorbar=False, cmap="viridis"
            )

        ax.set_xlabel(flux_xlabel)
        ax.set_ylabel("Idle time (µs)")
        ax.set_title(qname)

        cb = grid.fig.colorbar(im, ax=ax)
        cb.set_label("state" if node.parameters.use_state_discrimination else "I")

    grid.fig.suptitle("Raw")
    plt.tight_layout()
    plt.show()

    node.results["figure_raw_qubit"] = grid.fig

    # ---- FIT PLOT ----
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    grid.fig.set_size_inches(12, 3 * len(qubits))

    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]

        if qname in fit_results and fitted is not None:
            im = fitted.sel(qubit=qname).plot(
                ax=ax, x="flux_mV", y="idle_time",
                add_colorbar=False, cmap="viridis"
            )

            ax.set_xlabel(flux_xlabel)
            ax.set_ylabel("Idle time (µs)")
            ax.set_title(qname)

            cb = grid.fig.colorbar(im, ax=ax)
            cb.set_label("Fitted " + ("state" if node.parameters.use_state_discrimination else "I"))

        else:
            ax.set_title(f"{qname} – no fit")
            ax.text(0.5, 0.5, "No fit data", ha="center", va="center")
            ax.set_axis_off()

    grid.fig.suptitle("Fit (exponential decay)")
    plt.tight_layout()
    plt.show()

    node.results["figure_fit_qubit"] = grid.fig

    # ---- T1 PLOT ----
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    grid.fig.set_size_inches(12, 3 * len(qubits))

    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]

        if qname not in fit_results:
            ax.set_title(f"{qname} – no T1 data")
            ax.set_axis_off()
            continue

        T1 = np.array(fit_results[qname]["T1_vs_qubit_flux"], float)
        T1err = np.array(fit_results[qname]["T1err_vs_qubit_flux"], float)
        flux = ds.flux_mV.values

        mask = (
            np.isfinite(T1) &
            (T1 > 0) &
            (T1err < 0.5 * T1)
        )

        if not np.any(mask):
            ax.set_title(f"{qname} – no valid T1 points")
            ax.set_axis_off()
            continue

        ax.errorbar(
            flux[mask],
            T1[mask],
            yerr=T1err[mask],
            fmt="o-",
            capsize=3,
        )

        ax.set_title(qname)
        ax.set_xlabel(flux_xlabel)
        ax.set_ylabel("T1 (µs)")

    grid.fig.suptitle("T1 vs qubit flux pulse (filtered)")
    plt.tight_layout()
    plt.show()

    node.results["figure_T1_qubit"] = grid.fig


# %% {Save}
node.results["initial_parameters"] = node.parameters.model_dump()
node.save()
# %%
