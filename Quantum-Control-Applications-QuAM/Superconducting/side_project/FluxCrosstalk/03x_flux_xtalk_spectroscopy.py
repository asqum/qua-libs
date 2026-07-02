# %%
"""
        Z LINE CROSSTALK MEASUREMENT

Measure flux crosstalk between two Z lines by sweeping:

    detector_z (self compensation)
    crosstalk_z (disturbing line)

The resonance shift line gives the crosstalk slope.

Result:
    detector_z = k * crosstalk_z + b
    crosstalk = -1 / k
"""

# %% Imports
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM, Transmon, TransmonPair

from quam_libs.macros import qua_declaration, active_reset, readout_state, active_reset_simple
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import peaks_dips

from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from qm.qua import *
from qm import SimulationConfig

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional, List


# %% Node parameters
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ["q4", "q5"]
    source: str = "coupler_q4_q5"

    num_averages: int = 300

    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.01  #0.004, 0.02
    operation_len_in_ns: float = 100000

    source_flux_span: float = 0.6
    num_flux_points: int = 51

    expect_crosstalk: float = 0.1
    qubits_bias: float = 0
    qubits_freq_shift: float = 0  #MHz

    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    reset_type: Literal["active", "thermal"] = "active"

    simulate: bool = False
    simulation_duration_ns: int = 2000

    timeout: int = 100
    load_data_id: Optional[int] = None


node = QualibrationNode(
    name="03x_flux_xtalk",
    parameters=Parameters()
)


# %% Initialize QuAM
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
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

# Remove any qubit whose id matches the source
for qubit in qubits[:]:  # iterate over a copy to safely remove
    if qubit.id == node.parameters.source:
        print(f"Removing qubit '{qubit.id}' because it matches the source '{node.parameters.source}'")
        qubits.remove(qubit)
try:
    source = machine.qubits[node.parameters.source]
except:
    try:
        source = machine.qubit_pairs[node.parameters.source]
    except:
        raise ValueError("node.parameters.source is not a qubit or a coupler")

num_qubits = len(qubits)
# %% QUA program
n_avg = node.parameters.num_averages  # The number of averages
operation = node.parameters.operation  # The qubit operation to play

# Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
operation_len = node.parameters.operation_len_in_ns

if node.parameters.operation_amplitude_factor:
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
else:
    operation_amp = 1.0
crosstalk = node.parameters.expect_crosstalk
source_z_range = node.parameters.source_flux_span
z_step = node.parameters.num_flux_points
qubits_z_shift = node.parameters.qubits_bias
qubits_f_shift = int(node.parameters.qubits_freq_shift * u.MHz)
source_dzs = np.linspace(-source_z_range / 2, source_z_range / 2, z_step)
qubits_dzs = crosstalk * source_dzs + qubits_z_shift
flux_point = node.parameters.flux_point_joint_or_independent

with program() as flux_xtalk:

    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)

    s_dz = declare(fixed)
    q_dz = declare(fixed)

    machine.apply_all_couplers_to_min()

    for i, qubit in enumerate(qubits):
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            qubit.z.settle()
            # if isinstance(source, TransmonPair):
            #     source.coupler.settle()
            # else:
            #     source.z.settle()
            align()


        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(s_dz, source_dzs)):
                with for_(*from_array(q_dz, qubits_dzs)):
                    if not node.parameters.simulate:
                        if node.parameters.reset_type == "active":
                            active_reset(qubit, "readout")
                        else:
                            qubit.resonator.wait(qubit.thermalization_time * u.ns)
                            qubit.align()
                    # Update the qubit frequency
                    qubit.xy.update_frequency(qubit.xy.intermediate_frequency + qubits_f_shift, keep_phase=True)
                    duration = operation_len * u.ns if operation_len is not None else qubit.xy.operations[operation].length * u.ns
                    
                    align()
                    if isinstance(source, TransmonPair):
                        source.coupler.play("const", amplitude_scale=s_dz / source.coupler.operations["const"].amplitude, duration=duration)
                    else:
                        source.z.play("const", amplitude_scale=s_dz / source.z.operations["const"].amplitude, duration=duration)
                    qubit.z.play("const", amplitude_scale=q_dz / qubit.z.operations["const"].amplitude, duration=duration)
                    qubit.xy.play(operation,amplitude_scale=operation_amp,duration=duration,)
                    align()
                    qubit.xy.update_frequency(qubit.xy.intermediate_frequency, keep_phase=True)
                    align()

                    # Measure the state of the resonators
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
                    # Update the qubit frequency


    with stream_processing():

        n_st.save("n")
        for i, detector in enumerate(qubits):
            I_st[i].buffer(len(qubits_dzs)).buffer(len(source_dzs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(qubits_dzs)).buffer(len(source_dzs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, flux_xtalk, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples, plot=True)
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
        job = qm.execute(flux_xtalk)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is not None:
        ds, machine, json_data, detector, source, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"qubits_z": qubits_dzs, "source_z": source_dzs})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        ds = ds.assign_coords(qubits_z=ds.qubits_z, source_z=ds.source_z)  # convert to V
        ds.qubits_z.attrs = {"long_name": "qubits z bias", "units": "V"}
        ds.source_z.attrs = {"long_name": "source z bias", "units": "V"}

    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # Find the resonance dips for each flux point
    peaks = peaks_dips(ds.I, dim="qubits_z", prominence_factor=6)
    # Fit the result with a parabola
    linear_fit_results = peaks.position.polyfit("source_z", 1)
    # Try to fit again with a smaller prominence factor (may need some adjustment)
    if np.any(np.isnan(np.concatenate(linear_fit_results.polyfit_coefficients.values))):
        # Find the resonance dips for each flux point
        peaks = peaks_dips(ds.I, dim="qubits_z", prominence_factor=4)
        # Fit the result with a parabola
        linear_fit_results = peaks.position.polyfit("source_z", 1)
    # Extract relevant fitted parameters
    coeff = linear_fit_results.polyfit_coefficients
    fitted = coeff.sel(degree=1) * ds.source_z + coeff.sel(degree=0)
    qubits_bias = coeff[1]
    xtalk = -coeff[0]

    # Save fitting results
    fit_results = {}
    for q in qubits:
        fit_results[q.name] = {}
        if not np.isnan(xtalk.sel(qubit=q.name).values):
            if flux_point == "independent":
                offset = q.z.independent_offset
            elif flux_point == "joint":
                offset = q.z.joint_offset
            else:
                offset = 0.0
            print(f"flux crosstalk from {source.name} to {q.name} is {xtalk.sel(qubit=q.name).values:.2f}")

            fit_results[q.name]["qubit_bias"] = float(qubits_bias.sel(qubit=q.name).values)
            fit_results[q.name]["crosstalk"] = float(xtalk.sel(qubit=q.name).values)
        else:
            print(f"No fit for qubit {q.name}")
            fit_results[q.name]["qubit_bias"] = np.nan
            fit_results[q.name]["crosstalk"] = np.nan
    node.results["fit_results"] = fit_results

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds.loc[qubit].I.plot(
            ax=ax, add_colorbar=False, x="source_z", y="qubits_z", robust=True
        )
        fitted.loc[qubit].plot(ax=ax, linewidth=0.5, ls="--", color="r")
        peaks.position.loc[qubit].plot(ax=ax, ls="", marker=".", color="g", ms=0.5)
        ax.set_ylabel("qubit z bias (V)")
        ax.set_title(f"{qubit['qubit']}:{xtalk.sel(qubit=qubit['qubit']).values:.4f}")
        ax.set_xlabel("source z bias (V)")

    grid.fig.suptitle(f"crosstalk from {source.name}")
    plt.tight_layout()
    plt.show()
    node.results["figure_raw"] = grid.fig

    # %% {Update_state}
    # if node.parameters.load_data_id is None:
    #     with node.record_state_updates():
    #         for q in qubits:
    #             if not np.isnan(xtalk.sel(qubit=q.name).values):
    #                 q.xy.intermediate_frequency += fit_results[q.name]["drive_freq"]
    #                 q.freq_vs_flux_01_quad_term = fit_results[q.name]["quad_term"]

    # %% {Save_results}
    node.results["ds"] = ds
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()

# %%