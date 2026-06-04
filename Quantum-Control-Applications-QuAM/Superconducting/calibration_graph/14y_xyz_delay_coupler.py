# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.xyz_delay import (
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from quam_libs.macros import qua_declaration, active_reset
from qualang_tools.units import unit
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qm import SimulationConfig
from quam_libs.components import QuAM
from qualang_tools.bakery import baking
from typing import List, Literal, Optional
# %% {Description}
description = """
        XY-Z DELAY CALIBRATION
This calibration determines the relative delay between the microwave XY control line (e.g. x180) and the coupler Z line (flux)
for each qubit. The goal is to ensure that the coupler flux pulse reaches the qubit at the same time as the XY drive and correct
for any latency differences between the two control lines. By applying the XY and coupler Z pulses simultaneously, the qubit
frequency is shifted during the XY rotation, which can lead to incomplete rotations if the pulses are not properly aligned
in time. By inserting variable leading / trailing zeros around the fixed XY and coupler Z pulse shapes, the sequence scans the
relative timing at 1ns resolution and measures the qubit state for two initial preparations
(|e> created by an initial x180 and |g> with identity). The resulting population (or I/Q) versus relative timing is
fitted to extract the optimal flux delay that best aligns the coupler Z pulse with the qubit XY drive.

Prerequisites:
    - Having calibrated a pi-pulse (x180) for the given qubit.
    - Having found the anticrossing point of the qubit frequency versus coupler flux bias.
    - (Optional) State discrimination calibrated if use_state_discrimination = True.

State update:
    - Adds extracted flux delay (fit_results[qubit]["flux_delay"]) to q.z.opx_output.delay per successful qubit.
"""
qubit_pair_index = 2  # [1, 2]

class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q3"]
    """List of qubit names to calibrate. If None or empty, all active qubits are used."""
    coupler: Optional[List[str]] = f"coupler_q%s_q%s" % (qubit_pair_index, qubit_pair_index + 1)
    """Name of the coupler to use for flux pulsing."""
    num_shots: int = 50
    """Number of averages to perform. Default is 50."""
    zeros_before_after_pulse: int = 80
    """Number of zeros before and after the flux pulse to see the rising time. Default is 60ns"""
    reset_coupler_bias: bool = False
    """Whether to reset the coupler bias to 0V before each measurement (True)"""
    z_pulse_amplitude: float = -0.098
    """Amplitude of the Z pulse. It should to close to the antincrossing point. 
    It is relative to the decoupling offset if reset_coupler_bias is False, otherwise it is relative to 0V. Default is -0.1V"""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    """Flux point setting strategy: 'joint', 'independent', or 'pairwise'. Default is 'joint'."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for measurement (True) or acquire I/Q data (False). Default is True."""
    reset_type_active_or_thermal: Literal["active", "thermal"] = "thermal"
    """Type of qubit reset to use before each measurement: 'active' or 'thermal'. Default is 'thermal'."""
    timeout: int = 100
    """Timeout for QOP session in seconds. Default is 100s."""
    load_data_id:str = None
    """If provided, loads data from the specified dataset ID instead of running the experiment."""
    simulate:str = None
    """If True, simulates the QUA program instead of executing it on hardware."""
    multiplexed: bool = False
    """Whether to multiplex qubit measurements (True) or align after each qubit (False). Default is False."""
     

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode(
    name="14y_xyz_delay_coupler",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


def baked_flux_xy_segments(config: dict, waveform: List[float], qb, coupler, zeros_each_side: int):
    """Create baked XY+Z (flux) pulse segments for all relative shifts.

    Parameters
    ----------
    config : dict
        Full QUA configuration dict.
    waveform : list[float]
        Flux (Z) pulse samples (without padding) matching x180 length.
    qb : AnyTransmon-like
        Qubit object providing access to xy channel.
    coupler : AnyFluxTunableLike
        Coupler object providing access to flux channel.
    zeros_each_side : int
        Number of zeros before and after (total scan range = 2 * zeros_each_side).

    Returns
    -------
    list
        List of baking objects, each representing one relative timing segment.
    """
    pulse_segments = []
    total = 2 * zeros_each_side
    i_key = f"{qb.xy.operations['x180'].name}.wf.I"
    q_key = f"{qb.xy.operations['x180'].name}.wf.Q"
    I_samples = config["waveforms"][i_key]["samples"]
    Q_samples = config["waveforms"][q_key]["samples"]
    for i in range(total):
        with baking(config, padding_method="symmetric_l") as b:
            wf = [0.0] * i + waveform + [0.0] * (total - i)
            zeros = [0.0] * zeros_each_side
            I_wf = zeros + I_samples + zeros
            Q_wf = zeros + Q_samples + zeros
            assert len(wf) == len(I_wf) == len(Q_wf), "Flux and XY padded waveforms must have identical length"
            b.add_op("flux_pulse", coupler.name, wf)
            b.add_op("x180", qb.xy.name, [I_wf, Q_wf])
            b.play("flux_pulse", coupler.name)
            b.play("x180", qb.xy.name)
        pulse_segments.append(b)
    return pulse_segments


# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QUAM class from the state file
machine= QuAM.load()
node.machine = machine

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    
# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

coupler = machine.qubit_pairs[node.parameters.coupler].coupler
reset_coupler_bias = node.parameters.reset_coupler_bias

node.namespace["qubits"] = qubits
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_active_or_thermal

n_avg = node.parameters.num_shots  # Number of averages (used in the QUA averaging loop)
total_zeros = 2 * node.parameters.zeros_before_after_pulse  # Total number of delay positions scanned (± range)

flux_waveform_list = {}  # Will store per-qubit flux pulse sample lists prior to baking

for qubit in qubits:
    flux_waveform_list[qubit.xy.name] = [node.parameters.z_pulse_amplitude] * qubit.xy.operations["x180"].length

delay_segments = {}
# Baked flux pulse segments with 1ns resolution

for i, qubit in enumerate(qubits):
    delay_segments[qubit.xy.name] = baked_flux_xy_segments(
        config,
        flux_waveform_list[qubit.xy.name],
        qubit,
        coupler,
        node.parameters.zeros_before_after_pulse,
    )
    print(f"Baked waveform for {qubit.xy.name}")

node.namespace["config"] = config
relative_time = np.arange(
    -node.parameters.zeros_before_after_pulse, node.parameters.zeros_before_after_pulse, 1
)  # x-axis for plotting - Must be in ns.
number_of_segments = 2 * node.parameters.zeros_before_after_pulse

n_avg = node.parameters.num_shots  # The number of averages

# Register the sweep axes to be added to the dataset when fetching data
node.namespace["sweep_axes"] = {
    "qubit": qubits,
    "init_state": xr.DataArray(["e", "g"], attrs={"long_name": "initial qubit state", "units": "a.u."}),
    "relative_time": xr.DataArray(
        relative_time, attrs={"long_name": "relative delay between pulses", "units": "ns"}
    ),
}

with program() as qua_prog:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    if node.parameters.use_state_discrimination:
        state = [declare(bool) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
    segment = declare(int)  # QUA variable for the flux pulse segment index
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    npi = declare(int)  # QUA variable for the number of qubit pulses
    count = declare(int)  # QUA variable for counting the qubit pulses
   
    # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
    for qubit in qubits:
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
    align()
    if reset_coupler_bias:
        coupler.set_dc_offset(0.0)
    else:
        coupler.to_decouple_idle()
    wait(1000)

    # --- Batch over qubits (allows time-multiplexed execution respecting hardware constraints)
    for i, qubit in enumerate(qubits):

        # --- Averaging loop
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)  # Save current average index for live progress

            # --- Initial state preparation loop: prepare |e> (via x180) and |g> (idle) for contrast
            for init_state in ["e", "g"]:
                # --- Relative delay scan loop (index over baked XY+Z aligned segments)
                with for_(segment, 0, segment < number_of_segments, segment + 1):

                    # 1. Reset qubits to ground state
                    if reset_type == "active":
                        active_reset(qubit)
                    elif reset_type == "thermal":
                        qubit.wait(4 * qubit.thermalization_time * u.ns)
                    else:
                        raise ValueError(f"Unrecognized reset type {reset_type}.")

                    # 2. State preparation: excited (x180) or ground (wait same duration)
                    if init_state == "e":
                        qubit.xy.play("x180")
                    elif init_state == "g":
                        qubit.xy.wait(qubit.xy.operations["x180"].length)

                    qubit.align()
                    # 3. Optional coarse pre-wait (accounts for leading padding before fine scan)
                    qubit.wait(node.parameters.zeros_before_after_pulse // 4)
                    # 4. Apply baked XY+Z segment with specific relative shift
                    with switch_(segment):
                        for j in range(0, number_of_segments):
                            with case_(j):
                                delay_segments[qubit.xy.name][j].run()

                    qubit.align()
                    # 5. Measurement (state discrimination or I/Q acquisition)
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    if node.parameters.use_state_discrimination:
                        assign(state[i], I[i] > qubit.resonator.operations["readout"].threshold)
                        save(state[i], state_st[i])
                    else:
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

        if not node.parameters.multiplexed:
            align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].boolean_to_int().buffer(number_of_segments).buffer(2).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(number_of_segments).buffer(2).average().save(f"I{i + 1}")
                    Q_st[i].buffer(number_of_segments).buffer(2).average().save(f"Q{i + 1}")


# %% {Simulate}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, qua_prog, simulation_config)
    # Plot the simulated samples
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Update the node & save
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    # Open a quantum machine to execute the QUA program
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(qua_prog)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)


# %% {Execute}

# %% {Data_fetching_and_dataset_creation}

if not node.parameters.simulate:
    if node.parameters.load_data_id is not None:
        ds, machine, json_data, qubits, node.parameters = load_dataset(
            node.parameters.load_data_id, parameters=node.parameters
        )
        node.namespace["qubits"] = qubits
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"relative_time": relative_time, "init_state": ["e", "g"]})
    # Add the dataset to the node
    node.results = {"ds_raw": ds}


# %% {Analyse_data}
if reset_coupler_bias:
    coupler_set_point = 0.0 + node.parameters.z_pulse_amplitude
else:
    coupler_set_point = coupler.decouple_offset + node.parameters.z_pulse_amplitude
node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
log_fitted_results(fit_results, log_callable=node.log)
# Convert to dict format for storage and create outcomes
node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }

# %% {Plot_data}
fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
plt.suptitle(f"Qubit XY - Coupler Z Delay Calibration,  Coupler flux (full) : {coupler_set_point * 1e3} mV")
plt.show()
# Store the generated figures
node.results["figures"] = {
    "delay_scan": fig_raw_fit,
}


# %% {Update_state}
if node.parameters.load_data_id is  None:
    for q in node.namespace["qubits"]:
        if node.outcomes[q.name] == "failed":
            continue
        else:
            # Update the qubit flux delay
            coupler.opx_output.delay += int(node.results["fit_results"][q.name]["flux_delay"])


# %% {Save_results}
node.outcomes = {q.name: "successful" for q in qubits}
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
node.save()

# %%
