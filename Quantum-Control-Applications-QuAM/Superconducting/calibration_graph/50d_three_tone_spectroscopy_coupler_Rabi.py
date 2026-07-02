"""
RABI OSCILLATIONS USING THREE-TONE COUPLER SPECTROSCOPY METHOD

Overview:
    In the absence of a dedicated readout circuit for the coupler, we use three-tone spectroscopy
    with an target qubit to find the coupler frequency. The coupler frequency is set
    by using `set_dc_offset`. Once the coupler frequency is found, we perform Rabi oscillations on the coupler
    by strongly driving it through the control qubit drive line, while monitoring the target qubit state.

Prerequisites:
    - Calibrations of the readout of the target qubit
    - Coupler RF frequency estimate
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset_simple, readout_state
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubit_pairs_from_node,
)
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from quam_libs.lib.fit import fit_oscillation, oscillation
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from quam_libs.lib.instrument_limits import instrument_limits


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q2_q3"]
    """List of qubit pair names to measure."""
    num_averages: int = 1000
    """Number of times to average each measurement point. Defaults to 1000."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    """Whether to apply the same flux bias to all qubits or independently. Can be "joint" or "independent". Defaults to "joint"."""
    control_drive_operation: Literal["x180_Square", "x180"] = "x180"
    """Type of control qubit drive operation."""
    control_pulse_duration_in_ns: int = 64
    """Duration of the control qubit pulse in ns."""
    target_drive_operation: str = "saturation"
    """Type of operation to perform on the target qubit (e.g., "saturation", "x180"). Defaults to "saturation"."""
    target_pulse_amplitude: Optional[float] = 0.005  # 0.05  # 0.004, 0.02
    """Relative amplitude factor for the target drive pulse. Defaults to 0.005."""
    target_pulse_duration_in_ns: Optional[int] = 1000
    """Duration of the target qubit pulse in nanoseconds. Defaults to 1000 ns."""
    simulate: bool = False
    """Whether to run in simulation mode instead of real hardware. Defaults to False."""
    simulation_duration_ns: int = 10_000
    """Duration of simulation in nanoseconds. Defaults to 10,000 ns."""
    timeout: int = 100
    """Timeout in seconds for the measurement. Defaults to 100 seconds."""
    load_data_id: Optional[int] = None
    """Optional ID of previously saved data to load instead of running new measurement. Defaults to None."""
    reset_type: Literal["active", "thermal"] = "active"
    """Type of qubit reset to use - "active" or "thermal". Defaults to "active"."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination in readout. Defaults to False."""
    coupler_flux: float = 0.05
    """Coupler flux value"""
    amp_start: float = 0.0
    """Starting amplitude for the control qubit drive."""
    amp_end: float = 2.0
    """Ending amplitude for the control qubit drive."""
    amp_step: float = 0.01
    """Amplitude step size for the control qubit drive."""


node = QualibrationNode(name="50d_three_tone_spectroscopy_coupler_Rabi", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

num_qubit_pairs = len(qubit_pairs)
qubit_pair_names = [qp.name for qp in qubit_pairs]

num_qubit_pairs = len(qubit_pairs)

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# The frequency sweep around the resonator resonance frequency
amps = np.arange(node.parameters.amp_start, node.parameters.amp_end, node.parameters.amp_step)

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
coupler_IFs = {
    qp.name: qp.coupler.RF_frequency - qp.qubit_control.xy.opx_output.upconverter_frequency for qp in qubit_pairs
}

with program() as multi_res_spec_vs_flux:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubit_pairs)
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_stream_target = [declare_stream() for _ in range(num_qubit_pairs)]
    dc = declare(fixed)  # QUA variable for the flux bias
    amp = declare(fixed)  # QUA variable for the readout frequency

    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit_pairs[0].qubit_target)
        for qp in qubit_pairs:
            qp.coupler.set_dc_offset(node.parameters.coupler_flux)
            wait(1000)
        align()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        for i, qp in enumerate(qubit_pairs):
            with for_(*from_array(amp, amps)):

                # Qubit initialization
                qubit_control = qp.qubit_control
                qubit_target = qp.qubit_target

                # Update the qubit frequency
                if node.parameters.reset_type == "active":
                    active_reset_simple(qubit_control)
                    active_reset_simple(qubit_target)

                else:
                    qubit_control.wait(qubit_control.thermalization_time * u.ns)
                    qubit_target.wait(qubit_target.thermalization_time * u.ns)
                    qp.align()

                # update the frequency of the control qubit to couler drive frequency
                qubit_control.xy.update_frequency(coupler_IFs[qp.name])

                target_pulse_duration = (
                    node.parameters.target_pulse_duration_in_ns * u.ns
                    if node.parameters.target_pulse_duration_in_ns is not None
                    else qubit_target.xy.operations[node.parameters.target_drive_operation].length * u.ns
                )
                control_pulse_duration = (
                    node.parameters.control_pulse_duration_in_ns * u.ns
                    if node.parameters.control_pulse_duration_in_ns is not None
                    else qubit_control.xy.operations[node.parameters.control_drive_operation].length * u.ns
                )

                qp.align()

                # Drive coupler through qubit with a strong drive
                qubit_control.xy.play(
                    node.parameters.control_drive_operation,
                    amplitude_scale=amp,
                    duration=control_pulse_duration,
                )

                # Apply a probe tone to target qubit
                qubit_target.xy.play(
                    node.parameters.target_drive_operation,
                    amplitude_scale=node.parameters.target_pulse_amplitude,
                    duration=target_pulse_duration,
                )
                qp.align()

                # Measure target qubit and save data
                if node.parameters.use_state_discrimination:
                    readout_state(qp.qubit_target, state_target[i])
                    save(state_target[i], state_stream_target[i])
                else:
                    qp.qubit_target.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
        # Measure sequentially
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_stream_target[i].buffer(len(amps)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(amps)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amps)).average().save(f"Q{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_flux, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(multi_res_spec_vs_flux)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    if node.parameters.load_data_id is not None:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
    else:
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"amp": amps})
        if not node.parameters.use_state_discrimination:
            ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        ds = ds.assign_coords(
            {
                "abs_amp": (
                    ["qubit", "amp"],
                    np.array(
                        [
                            qp.qubit_control.xy.operations[node.parameters.control_drive_operation].amplitude * amps
                            for qp in qubit_pairs
                        ]
                    ),
                )
            }
        )

    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    fit_results = {}

    if node.parameters.use_state_discrimination:
        fit = fit_oscillation(ds.state, "amp")
    else:
        fit = fit_oscillation(ds.I, "amp")

    fit_evals = oscillation(
        ds.amp,
        fit.sel(fit_vals="a"),
        fit.sel(fit_vals="f"),
        fit.sel(fit_vals="phi"),
        fit.sel(fit_vals="offset"),
    )

    # Save fitting results
    for qp in qubit_pairs:
        fit_results[qp.name] = {}
        f_fit = fit.loc[qp.name].sel(fit_vals="f")
        phi_fit = fit.loc[qp.name].sel(fit_vals="phi")
        phi_fit = phi_fit - np.pi * (phi_fit > np.pi / 2)
        factor = float(1.0 * (np.pi - phi_fit) / (2 * np.pi * f_fit))
        new_pi_amp = qp.qubit_control.xy.operations[node.parameters.control_drive_operation].amplitude * factor
        limits = instrument_limits(qp.qubit_control.xy)
        if new_pi_amp < limits.max_x180_wf_amplitude:
            print(f"amplitude for Pi pulse is modified by a factor of {factor:.2f}")
            print(f"new amplitude is {1e3 * new_pi_amp:.2f} {limits.units} \n")
            fit_results[qp.name]["Pi_amplitude"] = new_pi_amp
        else:
            print(f"Fitted amplitude too high, new amplitude is {limits.max_x180_wf_amplitude} \n")
            fit_results[qp.name]["Pi_amplitude"] = limits.max_x180_wf_amplitude

    node.results["fit_results"] = fit_results

    # %% {Plotting}
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qp].state.plot(ax=ax, x="amp_mV")
            ax.plot(ds.abs_amp.loc[qp] * 1e3, fit_evals.loc[qp])
            ax.set_ylabel("Target qubit state")
        else:
            (ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qp].I * 1e3).plot(ax=ax, x="amp_mV")
            ax.plot(ds.abs_amp.loc[qp] * 1e3, 1e3 * fit_evals.loc[qp])
            ax.set_ylabel("Trans. amp. IQ_abs")

        ax.set_title(qp["qubit"])
        ax.set_xlabel("Amplitude")

    grid.fig.suptitle(f"Coupler Rabi Oscillations \n Coupler flux = {node.parameters.coupler_flux *1e3} mV")
    plt.tight_layout()
    plt.show()
    node.results["coupler_rabi"] = grid.fig

    # %% {Save_results}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()

# %%
