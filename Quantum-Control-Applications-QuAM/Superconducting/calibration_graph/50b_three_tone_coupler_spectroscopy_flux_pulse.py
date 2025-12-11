"""
THREE-TONE COUPLER SPECTROSCOPY WITH COUPLER FLUX PULSE

Overview:
    In the absence of a dedicated readout circuit for the coupler, we use three-tone spectroscopy
    with an target qubit to find the coupler frequency. The coupler frequency is set
    by using a coupler flux pulse.

Methdology:
    We begin by applying a weak probe tone to the target qubit at its transition frequency and monitor its
    state by sending a continuous wave to its readout resonator.
    At the same time, we strongly drive the coupler through the drive line
    of the control qubit. When the control qubit drive tone is in resonance with the coupler frequency, the
    coupler gets partially excited, and the target qubit frequency shifts down due to dispersive
    interaction between the target qubit and the coupler. As a result, the weak probe tone driving the target
    qubit no longer excites it, leading to a change in the readout signal of the ancilla qubit. This
    effectively maps the coupler state to the readout signal of the target qubit.

Prerequisites:
    - Calibrations of the readout of the target qubit

State update:
    - Coupler RF frequency: `qubit_pair.coupler.RF_frequency`
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
from quam_libs.trackable_object import tracked_updates
import warnings


# %% {Node_parameters}
class Parameters(NodeParameters):
    """Parameters for three-tone coupler spectroscopy versus coupler flux calibration."""

    qubit_pairs: Optional[List[str]] = ["coupler_q2_q3"]
    """List of qubit pair names to measure."""
    operation: str = "saturation"
    """Type of operation to perform on the target qubit (e.g., "saturation", "x180"). Defaults to "saturation"."""
    operation_amplitude_factor: Optional[float] = 0.005  # 0.05  # 0.004, 0.02
    """Relative amplitude factor for the operation pulse. Defaults to 0.005."""
    operation_len_in_ns: Optional[int] = 1000
    """Duration of the operation pulse in nanoseconds. Defaults to 1000 ns."""
    num_averages: int = 1000
    """Number of times to average each measurement point. Defaults to 1000."""
    frequency_span_in_mhz: float = 300
    """Total frequency span to sweep in MHz. Defaults to 200 MHz."""
    frequency_step_in_mhz: float = 0.2
    """Frequency step size in MHz. Defaults to 0.2 MHz."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    """Whether to apply the same flux bias to all qubits or independently. Can be "joint" or "independent". Defaults to "joint"."""
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
    RF_frequency_startpoint: Optional[float] = 6.85e9
    """Starting RF frequency of coupler in Hz for the scan. Defaults to 6.80 GHz."""
    use_state_discrimination: bool = False
    """Whether to use state discrimination in readout. Defaults to False."""
    coupler_flux: float = 0.15
    """Coupler flux value set  """


node = QualibrationNode(name="50b_three_tone_coupler_spectroscopy_flux_pulse", parameters=Parameters())


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

operation = node.parameters.operation
operation_len = node.parameters.operation_len_in_ns

if node.parameters.operation_amplitude_factor:
    operation_amp = node.parameters.operation_amplitude_factor
else:
    operation_amp = 1.0

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
# coupler_IFs = {}

# for qp in qubit_pairs:
#     # Decide if updating the LO is needed depending on the detuning request
#     qubit_control = qp.qubit_control

#     if (
#         coupler_RFs[qp.name]
#         + node.parameters.frequency_span_in_mhz
#         - qubit_control.xy.LO_frequency
#         > 400e6):
#           warnings.warn(
#             "Control qubit LO has been changed to reach desired coupler RF frequency"
#             )
#         # track the LO and IF changes to revert later
#           with tracked_updates(qubit_control, auto_revert=True, dont_assign_to_none=False) as q_upd:
#                 lo_frequency = coupler_RFs[qp.name]
#                 if (q_upd.xy.opx_output.band == 3) and (lo_frequency < 6.5e9):
#                     raise ValueError("Requested Coupler RF is not in the given MW FEM band")
#                 elif (q_upd.xy.opx_output.band == 2) and (lo_frequency < 4.5e9):
#                     raise ValueError("Requested Coupler RF is not in the given MW FEM band")
#                 print(f"Updating {q_upd.name} LO to {lo_frequency}")
#                 q_upd.xy.opx_output.upconverter_frequency = lo_frequency

#     else:
#         warnings.warn(
#             f"Control qubit LO is kept at {qubit_control.xy.LO_frequency}"
#             )
#     coupler_IFs[qp.name]= coupler_RFs[qp.name] - qp.qubit_control.xy.opx_output.upconverter_frequency

with program() as multi_res_spec_vs_flux:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubit_pairs)
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_stream_target = [declare_stream() for _ in range(num_qubit_pairs)]
    dc = declare(fixed)  # QUA variable for the flux bias
    df = declare(int)  # QUA variable for the readout frequency

    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit_pairs[0].qubit_target)
    align()
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        for i, qp in enumerate(qubit_pairs):
            with for_(*from_array(df, dfs)):  # type: ignore

                qubit_control = qp.qubit_control
                qubit_target = qp.qubit_target

                # Update the qubit frequency
                qubit_control.xy.update_frequency(qubit_control.xy.intermediate_frequency)
                if node.parameters.reset_type == "active":
                    active_reset_simple(qubit_control)
                    active_reset_simple(qubit_target)
                    qp.align()
                else:
                    qubit_control.reset_qubit_thermal()
                    qubit_target.reset_qubit_thermal()
                    qp.align()

                # update the frequency of the control qubit
                qubit_control.xy.update_frequency(df + coupler_IFs[qp.name])

                control_duration = (qubit_control.xy.operations["x180_Square"].length + 800) * u.ns

                target_duration = (
                    operation_len * u.ns
                    if operation_len is not None
                    else qubit_target.xy.operations[operation].length * u.ns
                )

                readout_duration = qubit_target.resonator.operations["readout"].length * u.ns

                qp.align()
                # Change the coupler flux
                qp.coupler.play(
                    "const",
                    amplitude_scale=node.parameters.coupler_flux / qp.coupler.operations["const"].amplitude,
                    duration=20 + control_duration + target_duration,
                )
                # wait for the flux to settle
                wait(20)
                qubit_control.xy.play("x180_Square", amplitude_scale=0.5, duration=control_duration)
                qubit_target.xy.play(
                    operation,
                    amplitude_scale=operation_amp,
                    duration=target_duration,
                )

                qp.align()
                if node.parameters.use_state_discrimination:
                    # Qubit readout
                    readout_state(qp.qubit_target, state_target[i])
                    # save data
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
                state_stream_target[i].buffer(len(dfs)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_flux, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"freq": dfs})
        if not node.parameters.use_state_discrimination:
            ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # Add the coupler  RF frequency to the dataset coordinates for plotting
        RF_freq = np.array([dfs + coupler_RFs[qp.name] for qp in qubit_pairs])
        ds = ds.assign_coords({"freq_full_control": (["qubit", "freq"], RF_freq)})
        ds.freq_full_control.attrs["long_name"] = "Frequency"
        ds.freq_full_control.attrs["units"] = "GHz"
        flux_full = np.array(
            [(node.parameters.coupler_flux + qp.coupler.decouple_offset) for qp in qubit_pairs]
        )
        ds = ds.assign_coords({"flux_full": (["qubit"], flux_full)})
        ds.flux_full.attrs["long_name"] = "Coupler Flux Bias"
        ds.flux_full.attrs["units"] = "V"
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}

    # Find the frequency for which ds.IQ_abs is minimum using xarray's reduction methods
    if node.parameters.use_state_discrimination:
        min_idx = ds.state.argmin(dim="freq")
    else:
        min_idx = ds.IQ_abs.argmax(dim="freq")

    min_freqs = ds.freq_full_control.isel(freq=min_idx)
    for qp in qubit_pairs:
        print(
            f"coupler Frequency for {qp.name} at flux {node.parameters.coupler_flux} V: {min_freqs.sel(qubit=qp.name).values*1e-9:.2f} GHz"
        )

    # %% {Plotting}
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            ds.assign_coords(freq_GHz=ds.freq_full_control / 1e9).sel(qubit=qp["qubit"]).state.plot(ax=ax, x="freq_GHz")
        else:
            ds.assign_coords(freq_GHz=ds.freq_full_control / 1e9).sel(qubit=qp["qubit"]).IQ_abs.plot(
                ax=ax, x="freq_GHz"
            )
        # ds.assign_coords(freq_GHz=ds.freq_full_control / 1e9).sel(qubit=qp["qubit"]).state.plot(ax=ax, x="freq_GHz")
        ax.axvline(1e-9 * min_freqs.sel(qubit=qp["qubit"]), color="red", linestyle="--", alpha=0.5)
        ax.set_title(f"{qp['qubit']}: \n Coupler flux = {ds.flux_full.sel(qubit=qp['qubit']).values * 1e3} mV")
        ax.set_xlabel("Frequency (GHz)")

    grid.fig.suptitle(f"Coupler spectroscopy")
    plt.tight_layout()
    plt.show()
    node.results["coupler_spectroscopy"] = grid.fig

    # %% {Update_state}
    for qp in qubit_pairs:
        qp.coupler.RF_frequency = float(min_freqs.sel(qubit=qp.name))
    # %% {Save_results}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%
