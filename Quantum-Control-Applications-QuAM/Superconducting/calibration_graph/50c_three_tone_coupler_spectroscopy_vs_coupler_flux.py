"""
THREE-TONE COUPLER SPECTROSCOPY VERSUS FLUX

Overview:
    In the absence of a dedicated readout circuit for the coupler, we use three-tone spectroscopy
    with an target qubit to find the coupler frequency.

Methdology:
    We begin by applying a weak probe tone to the target qubit at its transition frequency and monitor its
    state by sending a continuous wave to its readout resonator. At the same time, we strongly drive the coupler
    through the drive line of the control qubit. When the control qubit drive tone is in resonance with the coupler frequency, the
    coupler gets partially excited, and the target qubit frequency shifts down due to dispersive
    interaction between the target qubit and the coupler. As a result, the weak probe tone driving the target
    qubit no longer excites it, leading to a change in the readout signal of the ancilla qubit. This
    effectively maps the coupler state to the readout signal of the target qubit. The coupler flux is
    sweeped using coupler flux pulse to map out the whole coupler spectrum.

Prerequisites:
    - Calibrations of the readout of the target qubit

Before proceeding to the next node:
    - Save the current state and the coupler frequency vs flux plot
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset_simple, readout_state
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):
    """Parameters for three-tone coupler spectroscopy versus coupler flux calibration."""

    qubit_pairs: Optional[List[str]] = ["coupler_q2_q3"]
    """List of qubit pair names to measure. Defaults to ["coupler_q2_q3"]."""
    control_drive_operation: Literal["x180_Square", "x180"] = "x180_Square"
    """Type of control qubit drive operation."""
    control_pulse_duration_in_ns: int = 800
    """Duration of the control qubit pulse in ns."""
    control_pulse_amplitude: float = 0.2
    """Amplitude scale for the control qubit pulse."""
    target_drive_operation: str = "saturation"
    """Type of operation to perform on the target qubit (e.g., "saturation", "x180"). Defaults to "saturation"."""
    target_pulse_amplitude: Optional[float] = 0.005  # 0.05  # 0.004, 0.02
    """Relative amplitude factor for the target drive pulse. Defaults to 0.005."""
    target_pulse_duration_in_ns: Optional[int] = 1000
    """Duration of the target qubit pulse in nanoseconds. Defaults to 1000 ns."""
    num_averages: int = 200
    """Number of times to average each measurement point. Defaults to 1000."""
    frequency_span_in_mhz: float = 900
    """Total frequency span to sweep in MHz. Defaults to 200 MHz."""
    frequency_step_in_mhz: float = 2.0
    """Frequency step size in MHz. Defaults to 0.2 MHz."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    """Whether to apply the same flux bias to all qubits or independently. Can be "joint" or "independent". Defaults to "joint"."""
    coupler_flux_min: float = 0.0
    """Minimum coupler flux bias value. Defaults to -0.2."""
    coupler_flux_max: float = 0.3
    """Maximum coupler flux bias value. Defaults to 0.2."""
    coupler_flux_num_step: float = 51
    """Number of flux bias steps to scan. Defaults to 21."""
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
    RF_frequency_startpoint: Optional[float] = 7.2e9
    """Starting RF frequency of coupler in Hz for the scan. Defaults to 6.80 GHz."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination in readout. Defaults to False."""
    use_flux_pulse: bool = False
    """Whether to use flux pulse for coupler. Defaults to False."""


node = QualibrationNode(name="50c_Three_Tone_Coupler_Spectroscopy_vs_coupler_flux", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}

# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

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
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

fluxes = np.linspace(
    node.parameters.coupler_flux_min, node.parameters.coupler_flux_max, node.parameters.coupler_flux_num_step
)

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
coupler_RFs = {
    qp.name: (
        node.parameters.RF_frequency_startpoint
        if node.parameters.RF_frequency_startpoint is not None
        else qp.coupler.RF_frequency
    )
    for qp in qubit_pairs
}
coupler_IFs = {
    qp.name: coupler_RFs[qp.name] - qp.qubit_control.xy.opx_output.upconverter_frequency for qp in qubit_pairs
}

with program() as multi_res_spec_vs_flux:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubit_pairs)
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_stream_target = [declare_stream() for _ in range(num_qubit_pairs)]
    flux_pulse = declare(fixed)  # QUA variable for the flux bias
    df = declare(int)  # QUA variable for the readout frequency

    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit_pairs[0].qubit_target)
    align()
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        for i, qp in enumerate(qubit_pairs):
            with for_(*from_array(flux_pulse, fluxes)):  # type: ignore
                with for_(*from_array(df, dfs)):  # type: ignore
                    
                    qp.coupler.set_dc_offset(flux_pulse)
                    wait(1000)
                    qp.align()  

                    # Qubit initialization
                    qubit_control = qp.qubit_control
                    qubit_target = qp.qubit_target
                    
                    if node.parameters.reset_type == "active":
                        active_reset_simple(qubit_control)
                        active_reset_simple(qubit_target)
                        qp.align()
                    else:
                        qubit_control.wait(qubit_control.thermalization_time * u.ns)
                        qubit_target.wait(qubit_target.thermalization_time * u.ns)
                        qp.align()

                    # update the frequency of the control qubit to couler drive frequency
                    qubit_control.xy.update_frequency(df + coupler_IFs[qp.name])
                    
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
                        amplitude_scale=node.parameters.control_pulse_amplitude,
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
                state_stream_target[i].buffer(len(dfs)).buffer(len(fluxes)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(dfs)).buffer(len(fluxes)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).buffer(len(fluxes)).average().save(f"Q{i + 1}")


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
    node.machine = machine
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
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    else:
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"freq": dfs, "flux": fluxes})
        if not node.parameters.use_state_discrimination:
            ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # Add the coupler  RF frequency to the dataset coordinates for plotting
        RF_freq = np.array([dfs + coupler_RFs[qp.name] for qp in qubit_pairs])
        ds = ds.assign_coords({"freq_full_control": (["qubit", "freq"], RF_freq)})
        ds.freq_full_control.attrs["long_name"] = "Frequency"
        ds.freq_full_control.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}

    # Find the frequency for which ds.IQ_abs or ds.state is minimum
    if node.parameters.use_state_discrimination:
        min_idx = ds.state.argmin(dim="freq")
    else:
        min_idx = ds.IQ_abs.argmin(dim="freq")

    min_freqs = ds.freq_full_control.isel(freq=min_idx)

    # %% {Plotting}
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)

    for ax, qp in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            ds.assign_coords(freq_GHz=ds.freq_full_control / 1e9).sel(qubit=qp["qubit"]).state.plot(
                ax=ax, y="freq_GHz", x="flux"
            )
        else:
            ds.assign_coords(freq_GHz=ds.freq_full_control / 1e9).sel(qubit=qp["qubit"]).IQ_abs.plot(
                ax=ax, y="freq_GHz", x="flux"
            )

        ax.set_title(qp["qubit"])
        ax.set_ylabel("Frequency (GHz)")
        ax.set_xlabel("Coupler Flux (V)")
    grid.fig.suptitle(f"Coupler spectroscopy")
    plt.tight_layout()
    plt.show()
    node.results["coupler_spectroscopy"] = grid.fig

    # %% {Save_results}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()


# %%
