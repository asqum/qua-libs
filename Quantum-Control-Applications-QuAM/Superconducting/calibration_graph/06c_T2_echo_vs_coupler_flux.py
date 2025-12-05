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
    qubits: Optional[List[str]] = ["q2"]
    """qubits to perform the T2 echo measurement on. If None or empty, all active qubits will be used."""
    qubit_pair: str = "coupler_q2_q3"
    """The qubit pair containing the coupler to be fluxed during the T2 echo measurement."""
    num_averages: int = 100
    """The number of averages to perform."""
    min_wait_time_in_ns: int = 16
    """Minimum wait time in ns."""
    max_wait_time_in_ns: int = 20000
    """Maximum wait time in ns."""
    wait_time_step_in_ns: int = 100
    """Wait time step in ns."""
    coupler_flux_min: float = -0.5 # relative to the coupler set point if reset_coupler_bias is False
    """Minimum coupler flux amplitude (relative to the coupler set point)."""
    coupler_flux_max: float = 0.5  # relative to the coupler set point if reset_coupler_bias is False
    """Maximum coupler flux amplitude (relative to the coupler set point)."""
    coupler_flux_num_points: float = 51 
    """Number of coupler flux points."""
    flux_point_joint_or_independent_or_arbitrary: Literal["joint", "independent", "arbitrary"] = "independent"
    """Whether to use joint, independent or arbitrary flux points for the qubits."""
    simulate: bool = False
    """Whether to simulate the QUA program instead of executing it."""
    timeout: int = 100
    """Timeout for the QM session in seconds."""
    reset_coupler_bias: bool = True
    """Whether to reset the coupler bias to 0 before each measurement."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout."""
    reset_type: Literal["active", "thermal"] = "thermal"
    """Type of reset to use before each measurement."""
    use_coupler_flux_pulse: bool = False
    """Whether to use a coupler flux pulse on the coupler during the idle time."""


node = QualibrationNode(name="06c_T2_echo_vs_coupler_flux", parameters=Parameters())


# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()

qubit_pair = machine.qubit_pairs[node.parameters.qubit_pair]

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

fluxes_coupler = np.linspace(
    node.parameters.coupler_flux_min, node.parameters.coupler_flux_max, node.parameters.coupler_flux_num_points
)


flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary  # 'independent' or 'joint'
if flux_point == "arbitrary":
    detunings = {q.name: q.arbitrary_intermediate_frequency for q in qubits}
    arb_flux_bias_offset = {q.name: q.z.arbitrary_offset for q in qubits}
else:
    arb_flux_bias_offset = {q.name: 0.0 for q in qubits}
    detunings = {q.name: 0.0 for q in qubits}

reset_coupler_bias = node.parameters.reset_coupler_bias

with program() as t2_echo_vs_coupler_flux:
    flux_coupler = declare(float)
    comp_flux_qubit = declare(float)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the idle time
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
    
    for i, qubit in enumerate(qubits):
        XY_delay = qubit.xy.opx_output.delay + 4
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        
        if reset_coupler_bias:
            qubit_pair.coupler.set_dc_offset(0.0)
        else:
            qubit_pair.coupler.to_decouple_idle()
        wait(1000)
        qubit_pair.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(flux_coupler, fluxes_coupler)):
                with for_(*from_array(t, idle_times)):
                    if node.parameters.reset_type == "active":
                        active_reset(qubit, "readout")
                        qubit_pair.align()
                    else:
                        qubit.resonator.wait(qubit.thermalization_time * u.ns)
                        qubit_pair.align()

                    if "coupler_qubit_crosstalk" in qubit_pair.extras:
                        assign(
                            comp_flux_qubit,
                            arb_flux_bias_offset[qubit.name]
                            + qubit_pair.extras["coupler_qubit_crosstalk"] * flux_coupler,
                        )
                    else:
                        assign(comp_flux_qubit, arb_flux_bias_offset[qubit.name])
                    
                    if not node.parameters.use_coupler_flux_pulse:
                        qubit_pair.coupler.set_dc_offset(flux_coupler)
                        wait(1000)
                        qubit_pair.align()

                    qubit.xy.play("x90")

                    qubit.z.wait(qubit.xy.operations["x90"].length // 4 + XY_delay // 4)
                    qubit_pair.coupler.wait(qubit.xy.operations["x90"].length // 4 + XY_delay // 4)
                    qubit.z.play(
                        "const",
                        amplitude_scale=comp_flux_qubit / qubit.z.operations["const"].amplitude,
                        duration=t,
                    )
                    if node.parameters.use_coupler_flux_pulse:
                        qubit_pair.coupler.play(
                            "const",
                            amplitude_scale=flux_coupler / qubit_pair.coupler.operations["const"].amplitude,
                            duration=t,
                        )

                    qubit.xy.wait(t + 1)
                    qubit.xy.play("x180")
                    qubit.xy.wait(t + 1)

                    qubit_pair.coupler.wait(duration=qubit.xy.operations["x180"].length // 4)
                    qubit.z.wait(duration=qubit.xy.operations["x180"].length // 4)
                    qubit.z.play(
                        "const",
                        amplitude_scale=comp_flux_qubit / qubit.z.operations["const"].amplitude,
                        duration=t,
                    )
                    if node.parameters.use_coupler_flux_pulse:
                        qubit_pair.coupler.play(
                            "const",
                            amplitude_scale=flux_coupler / qubit_pair.coupler.operations["const"].amplitude,
                            duration=t,
                        )
                    

                    qubit.xy.play("-x90")
                    qubit.z.wait(20)
                    qubit_pair.coupler.wait(20)
                    qubit_pair.align()

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
                state_st[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"Q{i + 1}")


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
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times, "flux_coupler": fluxes_coupler})

    ds = ds.assign_coords(idle_time=8 * ds.idle_time / 1e3)  # convert to usec
    ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
    node.results = {"ds": ds}

# %% {Data_analysis} 
if not node.parameters.simulate:
    ds = ds.assign_coords(flux_mV=ds.flux_coupler * 1e3)
    fit_results = {}
    # Fit the exponential decay
    if node.parameters.use_state_discrimination:
        fit_data = fit_decay_exp(ds.state, "idle_time")
    else:
        fit_data = fit_decay_exp(ds.I, "idle_time")
    fit_data.attrs = {"long_name": "fits", "units": " "}
    # Fitted decay
    fitted = decay_exp(
        ds.idle_time,
        fit_data.sel(fit_vals="a"),
        fit_data.sel(fit_vals="offset"),
        fit_data.sel(fit_vals="decay"),
    )
    # Decay rate and its uncertainty
    decay = fit_data.sel(fit_vals="decay")
    decay.attrs = {"long_name": "decay", "units": "MHz"}
    decay_res = fit_data.sel(fit_vals="decay_decay")
    decay_res.attrs = {"long_name": "decay", "units": "MHz"}
    # T1 and its uncertainty
    tau = -1 / fit_data.sel(fit_vals="decay")
    tau.attrs = {"long_name": "T1", "units": "µs"}
    tau_error = -tau * (np.sqrt(decay_res) / decay)
    tau_error.attrs = {"long_name": "T1 error", "units": "µs"}

    for q in qubits:
            fit_results[q.name] = {}
            fit_results[q.name]["T2_vs_coupler_flux"] = tau.sel(qubit=q.name).values.tolist()
            fit_results[q.name]["T2err_vs_coupler_flux"] = tau_error.sel(qubit=q.name).values.tolist()
    node.results["fit_results"] = fit_results
# %% {Plotting}
if not node.parameters.simulate:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    grid.fig.set_size_inches(12, 3 * len(qubits))

    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]

        if node.parameters.use_state_discrimination:
            im = ds.sel(qubit=qname).state.plot(
                ax=ax, x="flux_mV", y="idle_time",
                add_colorbar=False, cmap="viridis"
            )
            ax.set_ylabel("Idle time (µs)")
        else:
            im = ds.sel(qubit=qname).I.plot(
                ax=ax, x="flux_mV", y="idle_time",
                add_colorbar=False, cmap="viridis"
            )
            ax.set_ylabel("Idle time (µs)")

        ax.set_title(f"{qname}")
        ax.set_xlabel("Coupler flux (mV)")
        cb = grid.fig.colorbar(im, ax=ax)
        cb.set_label(("state" if node.parameters.use_state_discrimination else "I"))

    grid.fig.suptitle("Raw")
    plt.tight_layout()
    plt.show()

    node.results["figure_raw_coupler"] = grid.fig

    # ---- FIT PLOT ----
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    grid.fig.set_size_inches(12, 3 * len(qubits))

    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]

        im = fitted.sel(qubit=qname).plot(
            ax=ax, x="flux_mV", y="idle_time", add_colorbar=False,
                cmap="viridis"
        )

        ax.set_title(f"{qname}")
        ax.set_xlabel("Coupler flux (mV)")
        ax.set_ylabel("Idle time (µs)")  
        cb = grid.fig.colorbar(im, ax=ax)
        cb.set_label("Fitted " + ("state" if node.parameters.use_state_discrimination else "I"))

    grid.fig.suptitle("Fit")
    plt.tight_layout()
    plt.show()

    node.results["figure_fit_coupler"] = grid.fig

    # ---- T2 PLOT ----
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    grid.fig.set_size_inches(12, 3 * len(qubits))

    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]

        T2 = tau.sel(qubit=qname).values
        T2err = tau_error.sel(qubit=qname).values

        # Ignore negative/unphysical values
        mask = T2 > 0
        T2 = np.where(mask, T2, np.nan)
        T2err = np.where(mask, T2err, np.nan)

        ax.errorbar(
            ds.flux_mV.values,
            T2,
            yerr=T2err,
            fmt="o-",
            capsize=3,
        )

        ax.set_title(f"{qname}")
        ax.set_xlabel("Coupler flux (mV)")
        ax.set_ylabel("T2 (µs)")

    grid.fig.suptitle("T2 vs Coupler Flux")
    plt.tight_layout()
    plt.show()

    node.results["figure_T2_coupler"] = grid.fig


# %%
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
node.save()
# %%
