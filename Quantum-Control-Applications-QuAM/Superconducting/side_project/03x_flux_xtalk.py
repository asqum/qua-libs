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
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state, active_reset_simple
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset

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

    detector: Optional[List[str]] = None
    source: Optional[List[str]] = None

    num_averages: int = 200

    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.1 #0.004, 0.02
    operation_len_in_ns: float = 100

    source_flux_span: float = 0.4
    num_flux_points: int = 51

    expect_crosstalk: float = 0.5
    detector_bias: float = 0

    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    reset_type: Literal["active", "thermal"] = "thermal"
    use_state_discrimination: bool = False

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
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.source is None or node.parameters.source == "":
    sources = machine.active_qubits
else:
    sources = [machine.qubits[q] for q in node.parameters.qubits]

if node.parameters.detector is None or node.parameters.detector == "":
    detectors = machine.active_qubits
else:
    detectors = [machine.qubits[q] for q in node.parameters.qubits]

num_detector = len(detectors)
num_source = len(sources)
num_qubit = num_detector*num_source
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
detector_z_shift = node.parameters.detector_bias
source_dzs = np.arange(-source_z_range // 2, source_z_range // 2, z_step)
detector_dzs = crosstalk * source_dzs + detector_z_shift
flux_point = node.parameters.flux_point_joint_or_independent

with program() as flux_xtalk:

    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubit)

    s_dz = declare(fixed)
    d_dz = declare(fixed)

    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubit)]
        state_st = [declare_stream() for _ in range(num_qubit)]


    machine.apply_all_couplers_to_min()

    for i, detector in enumerate(detectors):
        for j, source in enumerate(sources):

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(s_dz, source_dzs)):

                    with for_(*from_array(d_dz, detector_dzs)):

                        machine.set_all_fluxes(flux_point=flux_point, target=detector)
                        detector.z.settle()
                        source.z.settle()
                        detector.align()
                        source.align()

                        if node.parameters.reset_type == "active":
                            active_reset(detector, "readout")
                        else:
                            detector.resonator.wait(detector.thermalization_time * u.ns)
                            detector.align()
                        
                        duration = operation_len * u.ns if operation_len is not None else detector.xy.operations[operation].length * u.ns

                        source.z.play("const", amplitude_scale=d_dz / source.z.operations["const"].amplitude, duration=duration)
                        detector.z.play("const", amplitude_scale=d_dz / detector.z.operations["const"].amplitude, duration=duration)
                        detector.xy.play(operation,amplitude_scale=operation_amp,duration=duration,)
                        source.align()
                        detector.align()
                        # Measure the state of the resonators
                        if node.parameters.use_state_discrimination:
                            readout_state(detector, state[i*num_source+j])
                            save(state[i*num_source+j], state_st[i*num_source+j])
                        else:
                            detector.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            # save data
                            save(I[i*num_source+j], I_st[i*num_source+j])
                            save(Q[i*num_source+j], Q_st[i*num_source+j])
                        # Wait for the qubits to decay to the ground state
                        detector.resonator.wait(machine.depletion_time * u.ns)

    with stream_processing():

        n_st.save("n")
        for i, detector in enumerate(detectors):
            for j, source in enumerate(sources):
                if node.parameters.use_state_discrimination:
                    state_st[i*num_source+j].buffer(len(detector_dzs)).buffer(len(source_dzs)).average().save(f"state{i + 1}")
                else:
                    I_st[i*num_source+j].buffer(len(detector_dzs)).buffer(len(source_dzs)).average().save(f"I{i + 1}")
                    Q_st[i*num_source+j].buffer(len(detector_dzs)).buffer(len(source_dzs)).average().save(f"Q{i + 1}")


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
    node.machine = machine
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
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"flux": dcs, "freq": dfs})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([shift + dfs + q.xy.RF_frequency for q in qubits]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # Find the resonance dips for each flux point
    peaks = peaks_dips(ds.I, dim="freq", prominence_factor=6)
    # Fit the result with a parabola
    parabolic_fit_results = peaks.position.polyfit("flux", 2)
    # Try to fit again with a smaller prominence factor (may need some adjustment)
    if np.any(np.isnan(np.concatenate(parabolic_fit_results.polyfit_coefficients.values))):
        # Find the resonance dips for each flux point
        peaks = peaks_dips(ds.I, dim="freq", prominence_factor=4)
        # Fit the result with a parabola
        parabolic_fit_results = peaks.position.polyfit("flux", 2)
    # Extract relevant fitted parameters
    coeff = parabolic_fit_results.polyfit_coefficients
    fitted = coeff.sel(degree=2) * ds.flux**2 + coeff.sel(degree=1) * ds.flux + coeff.sel(degree=0)
    flux_shift = -coeff[1] / (2 * coeff[0])
    freq_shift = coeff.sel(degree=2) * flux_shift**2 + coeff.sel(degree=1) * flux_shift + coeff.sel(degree=0)

    # Save fitting results
    fit_results = {}
    for q in qubits:
        fit_results[q.name] = {}
        if not np.isnan(flux_shift.sel(qubit=q.name).values):
            if flux_point == "independent":
                offset = q.z.independent_offset
            elif flux_point == "joint":
                offset = q.z.joint_offset
            else:
                offset = 0.0
            print(f"flux offset for qubit {q.name} is {offset*1e3 + flux_shift.sel(qubit = q.name).values*1e3:.0f} mV")
            print(f"(shift of  {flux_shift.sel(qubit = q.name).values*1e3:.0f} mV)")
            print(
                f"Drive frequency for {q.name} is {(freq_shift.sel(qubit = q.name).values + q.xy.RF_frequency)/1e9:.3f} GHz"
            )
            print(f"(shift of {freq_shift.sel(qubit = q.name).values/1e6:.0f} MHz)")
            print(f"quad term for qubit {q.name} is {float(coeff.sel(degree = 2, qubit = q.name)/1e9):.3e} GHz/V^2 \n")
            fit_results[q.name]["flux_shift"] = float(flux_shift.sel(qubit=q.name).values)
            fit_results[q.name]["drive_freq"] = float(freq_shift.sel(qubit=q.name).values)
            fit_results[q.name]["quad_term"] = float(coeff.sel(degree=2, qubit=q.name))
        else:
            print(f"No fit for qubit {q.name}")
            fit_results[q.name]["flux_shift"] = np.nan
            fit_results[q.name]["drive_freq"] = np.nan
            fit_results[q.name]["quad_term"] = np.nan
    node.results["fit_results"] = fit_results

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        freq_ref = machine.qubits[qubit["qubit"]].xy.RF_frequency
        ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].I.plot(
            ax=ax, add_colorbar=False, x="flux", y="freq_GHz", robust=True
        )
        ((fitted + freq_ref) / 1e9).loc[qubit].plot(ax=ax, linewidth=0.5, ls="--", color="r")
        ax.plot(flux_shift.loc[qubit], ((freq_shift.loc[qubit] + freq_ref) / 1e9), "r*")
        ((peaks.position.loc[qubit] + freq_ref) / 1e9).plot(ax=ax, ls="", marker=".", color="g", ms=0.5)
        ax.set_ylabel("Freq (GHz)")
        ax.set_xlabel("Flux (V)")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Qubit spectroscopy vs flux ")

    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for q in qubits:
                if not np.isnan(flux_shift.sel(qubit=q.name).values):
                    if flux_point == "independent":
                        q.z.independent_offset += fit_results[q.name]["flux_shift"]
                        if "c" in q.id: # for coupler-test case
                            q.z.joint_offset += fit_results[q.name]["flux_shift"]
                            q.z.independent_offset = q.z.joint_offset - q.phi0_voltage / 2 
                    elif flux_point == "joint":
                        q.z.joint_offset += fit_results[q.name]["flux_shift"]
                    q.xy.intermediate_frequency += fit_results[q.name]["drive_freq"]
                    q.freq_vs_flux_01_quad_term = fit_results[q.name]["quad_term"]

    # %% {Save_results}
    node.results["ds"] = ds
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%