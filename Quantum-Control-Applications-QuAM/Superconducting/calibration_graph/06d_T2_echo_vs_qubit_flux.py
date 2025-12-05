# %%
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save, active_reset, readout_state

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_decay_exp, decay_exp


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q3"]
    """qubits to perform the T2 echo measurement on. If None or empty, all active qubits will be used."""
    num_averages: int = 100
    """The number of averages to perform."""
    min_wait_time_in_ns: int = 16
    """Minimum wait time in ns."""
    max_wait_time_in_ns: int = 25000
    """Maximum wait time in ns."""
    wait_time_step_in_ns: int = 100
    """Wait time step in ns."""
    qubit_flux_min: float = -0.1 # relative to the coupler set point if reset_coupler_bias is False
    """Minimum coupler flux amplitude (relative to the coupler set point)."""
    qubit_flux_max: float = 0.1 # relative to the coupler set point if reset_coupler_bias is False
    """Maximum coupler flux amplitude (relative to the coupler set point)."""
    qubit_flux_num_points: float = 5 
    """Number of coupler flux points."""
    flux_point_joint_or_independent_or_arbitrary: Literal["joint", "independent", "arbitrary"] = "independent"
    """Whether to use joint, independent or arbitrary flux points for the qubits."""
    simulate: bool = False
    """Whether to simulate the QUA program instead of executing it."""
    timeout: int = 100
    """Timeout for the QM session in seconds."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout."""
    reset_type: Literal["active", "thermal"] = "thermal"
    """Type of reset to use before each measurement."""
    use_qubit_flux_pulse: bool = True
    """Whether to use a coupler flux pulse on the coupler during the idle time."""


node = QualibrationNode(name="06d_T2_echo_vs_qubit_flux", parameters=Parameters())


# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
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

# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    node.parameters.wait_time_step_in_ns // 4,
)

fluxes_qubit = np.linspace(
    node.parameters.qubit_flux_min, 
    node.parameters.qubit_flux_max, 
    node.parameters.qubit_flux_num_points
)


flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary  # 'independent' or 'joint'


with program() as t2_echo_vs_coupler_flux:
    flux_qubit = declare(float)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the idle time
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
    
    for i, qubit in enumerate(qubits):
        XY_delay = qubit.xy.opx_output.delay + 4
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case      
        wait(1000)
        qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(flux_qubit, fluxes_qubit)):
                with for_(*from_array(t, idle_times)):
                    if node.parameters.reset_type == "active":
                        active_reset(qubit, "readout")
                        qubit.align()
                    else:
                        qubit.resonator.wait(qubit.thermalization_time * u.ns)
                        qubit.align()
                    
                    if not node.parameters.use_qubit_flux_pulse:
                        qubit.z.set_dc_offset(flux_qubit)
                        wait(1000)
                        qubit.align()


                    qubit.xy.play("x90")

                    qubit.z.wait(qubit.xy.operations["x90"].length // 4 + XY_delay // 4)
                    if node.parameters.use_qubit_flux_pulse:
                        qubit.z.play(
                            "const",
                            amplitude_scale=flux_qubit / qubit.z.operations["const"].amplitude,
                            duration=t,
                        )

                    qubit.xy.wait(t + 1)
                    qubit.xy.play("x180")
                    qubit.xy.wait(t + 1)

                    qubit.z.wait(duration=qubit.xy.operations["x180"].length // 4)
                    if node.parameters.use_qubit_flux_pulse:
                        qubit.z.play(
                            "const",
                            amplitude_scale=flux_qubit / qubit.z.operations["const"].amplitude,
                            duration=t,
                        )
                        

                    qubit.xy.play("-x90")
                    
                    align()

                    # Measure the state of the resonators
                    if node.parameters.use_state_discrimination:
                        readout_state(qubit, state[i])
                        save(state[i], state_st[i])
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # save data
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
    job = qmm.simulate(config, t2_echo_vs_coupler_flux, simulation_config)
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.machine = machine
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(t2_echo_vs_coupler_flux)
        # Get results from QUA program
        for i in range(num_qubits):
            print(f"Fetching results for qubit {qubits[i].name}")
            data_list = ["n"]
            results = fetching_tool(job, data_list, mode="live")
            # Live plotting
            # fig, axes = plt.subplots(2, num_qubits, figsize=(4 * num_qubits, 8))
            # interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
            while results.is_processing():
                # Fetch results
                fetched_data = results.fetch_all()
                n = fetched_data[0]

                progress_counter(n, n_avg, start_time=results.start_time)


# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:

    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times, "flux_qubit": fluxes_qubit})

    ds = ds.assign_coords(idle_time=8 * ds.idle_time / 1e3)  # convert to usec
    ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
    node.results = {"ds": ds}

# %% {Data_analysis} 
if not node.parameters.simulate:
    ds = ds.assign_coords(flux_mV=ds.flux_qubit * 1e3)
    fit_results = {}
    try:
        for q in qubits:
            # Fit choice
            if node.parameters.use_state_discrimination:
                fit_data = fit_decay_exp(ds.state, "idle_time")
            else:
                fit_data = fit_decay_exp(ds.I, "idle_time")

            fit_data.attrs = {"long_name": "fits", "units": " "}

            # Build fitted decay
            fitted = decay_exp(
                ds.idle_time,
                fit_data.sel(fit_vals="a"),
                fit_data.sel(fit_vals="offset"),
                fit_data.sel(fit_vals="decay"),
            )

            decay = fit_data.sel(fit_vals="decay")
            decay.attrs = {"long_name": "decay", "units": "MHz"}

            decay_res = fit_data.sel(fit_vals="decay_decay")
            decay_res.attrs = {"long_name": "decay error", "units": "MHz"}

            # Compute T2 (you call it T1 but your comment says T2)
            tau = -1 / decay
            tau.attrs = {"long_name": "T2", "units": "µs"}

            tau_error = -tau * (np.sqrt(decay_res) / decay)
            tau_error.attrs = {"long_name": "T2 error", "units": "µs"}

            fit_results[q.name] = {
                "T2_vs_qubit_flux": tau.sel(qubit=q.name).values.tolist(),
                "T2err_vs_qubit_flux": tau_error.sel(qubit=q.name).values.tolist(),
            }

        node.results["fit_results"] = fit_results
    except Exception as e:
        print("⚠️ Fit failed:", e)
        print("Proceeding with raw data only.")

    def t2_is_valid(t2_array):
        """Returns True if T2 contains at least one positive finite value."""
        t2 = np.array(t2_array, dtype=float)
        return np.any(np.isfinite(t2) & (t2 > 0))
    
# %% {Plotting}
if not node.parameters.simulate:
   # ---- RAW PLOT ----
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    grid.fig.set_size_inches(12, 3 * len(qubits))

    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]

        if node.parameters.use_state_discrimination:
            im = ds.sel(qubit=qname).state.plot(
                ax=ax, x="flux_mV", y="idle_time", add_colorbar=False, cmap="viridis"
            )
            ax.set_ylabel("Idle time (µs)")
        else:
            im = ds.sel(qubit=qname).I.plot(
                ax=ax, x="flux_mV", y="idle_time", add_colorbar=False, cmap="viridis"
            )
            ax.set_ylabel("Idle time (µs)")

        xlabel = f"{qname} flux {'shift' if node.parameters.use_qubit_flux_pulse else 'full'} (mV)"
        ax.set_xlabel(xlabel)
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

        # Only plot qubit where fit exists
        if qname in fit_results:

            im = fitted.sel(qubit=qname).plot(
                ax=ax, x="flux_mV", y="idle_time",
                add_colorbar=False, cmap="viridis"
            )

            xlabel = f"{qname} flux {'shift' if node.parameters.use_qubit_flux_pulse else 'full'} (mV)"
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Idle time (µs)")
            ax.set_title(qname)

            cb = grid.fig.colorbar(im, ax=ax)
            cb.set_label("Fitted " + ("state" if node.parameters.use_state_discrimination else "I"))

        else:
            # Fit missing
            ax.set_title(f"{qname} – no fit")
            ax.text(0.5, 0.5, "No fit data", ha="center", va="center")
            ax.set_axis_off()

    grid.fig.suptitle("Fit")
    plt.tight_layout()
    plt.show()

    node.results["figure_fit_qubit"] = grid.fig

    # ---- T2 PLOT ----
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    grid.fig.set_size_inches(12, 3 * len(qubits))

    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]

        if qname not in fit_results:
            ax.set_title(f"{qname} – no T2 data")
            ax.set_axis_off()
            continue

        T2 = np.array(fit_results[qname]["T2_vs_qubit_flux"], float)
        T2err = np.array(fit_results[qname]["T2err_vs_qubit_flux"], float)
        flux = ds.flux_mV.values

        mask = (
        np.isfinite(T2) &
        (T2 > 0) &
        (T2err < 0.5 * T2)   # <-- STD check (relative error))
        )

        # If no good points, skip this qubit
        if not np.any(mask):
            ax.set_title(f"{qname} – no valid T2 points")
            ax.set_axis_off()
            continue

        ax.errorbar(
            flux[mask],
            T2[mask],
            yerr=T2err[mask],
            fmt="o-",
            capsize=3,
        )

        xlabel = f"{qname} flux {'shift' if node.parameters.use_qubit_flux_pulse else 'full'} (mV)"
        ax.set_title(qname)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("T2 (µs)")

    grid.fig.suptitle("T2 vs qubit flux (filtered)")
    plt.tight_layout()
    plt.show()

    node.results["figure_T2_qubit"] = grid.fig



# %% {Save}
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
node.save()
# %%
