"""
        RESONATOR SPECTROSCOPY VERSUS FLUX
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various readout intermediate dfs and flux biases.
The resonator frequency as a function of flux bias is then extracted and fitted so that the parameters can be stored in the state.

This information can then be used to adjust the readout frequency for the maximum and minimum frequency points.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the relevant flux biases in the state.
    - Save the current state
"""

# %% {Imports}
from datetime import datetime
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.experiments.res_spec_vs_qubit_flux.analysis import fit_resonator_spectroscopy_vs_flux
from quam_libs.experiments.res_spec_vs_qubit_flux.plotting import plot_resonator_spectroscopy_vs_qubit_flux
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_oscillation
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import warnings


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 10
    min_flux_offset_in_v: float = -1.0
    max_flux_offset_in_v: float = 1.0
    num_flux_points: int = 201
    frequency_span_in_mhz: float = 10
    frequency_step_in_mhz: float = 0.1
    smoothing_filter_size_in_mhz: float = 0.1
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    input_line_impedance_in_ohm: float = 50
    line_attenuation_in_db: float = 0
    update_flux_min: bool = False
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = 144

node = QualibrationNode(name="02b_Resonator_Spectroscopy_vs_Flux", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
if any([q.z is None for q in qubits]):
    warnings.warn("Found qubits without a flux line. Skipping")

qubits = [q for q in qubits if q.z is not None]
resonators = [qubit.resonator for qubit in qubits]
num_qubits = len(qubits)

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Flux bias sweep in V
dcs = np.linspace(
    node.parameters.min_flux_offset_in_v,
    node.parameters.max_flux_offset_in_v,
    node.parameters.num_flux_points,
)
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
update_flux_min = node.parameters.update_flux_min  # Update the min flux point

with program() as multi_res_spec_vs_flux:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    dc = declare(fixed)  # QUA variable for the flux bias
    df = declare(int)  # QUA variable for the readout frequency

    for i, qubit in enumerate(qubits):
        # resonator of the qubit
        rr = resonators[i]
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit, do_align=False)
        align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(dc, dcs)):
                # Flux sweeping by tuning the OPX dc offset associated with the flux_line element
                qubit.z.set_dc_offset(dc)
                qubit.z.settle()
                align()
                with for_(*from_array(df, dfs)):
                    # Update the resonator frequencies for resonator
                    update_frequency(rr.name, df + rr.intermediate_frequency)
                    # readout the resonator
                    rr.measure("readout", qua_vars=(I[i], Q[i]))
                    # wait for the resonator to relax
                    rr.wait(machine.depletion_time * u.ns)
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

        # Measure sequentially
        # align(*[rr.name for rr in resonators])
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].buffer(len(dfs)).buffer(len(dcs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).buffer(len(dcs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_flux, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
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
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs, "flux": dcs})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        RF_freq = np.array([dfs + q.resonator.RF_frequency for q in qubits])
        ds = ds.assign_coords({"freq_full": (["qubit", "freq"], RF_freq)})
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
        # Add the current axis of each qubit to the dataset coordinates for plotting
        current = np.array([ds.flux.values / node.parameters.input_line_impedance_in_ohm for q in qubits])
        ds = ds.assign_coords({"current": (["qubit", "flux"], current)})
        ds.current.attrs["long_name"] = "Current"
        ds.current.attrs["units"] = "A"
        # Add attenuated current to dataset
        attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
        attenuated_current = ds.current * attenuation_factor
        ds = ds.assign_coords({"attenuated_current": (["qubit", "flux"], attenuated_current.values)})
        ds.attenuated_current.attrs["long_name"] = "Attenuated Current"
        ds.attenuated_current.attrs["units"] = "A"
    # Add the dataset to the node
    node.results = {"ds": ds}

    ds = ds.transpose("qubit", "freq", "flux")

    fit = fit_resonator_spectroscopy_vs_flux(ds, node.parameters.smoothing_filter_size_in_mhz)
    node.results["fit_results"] = fit

    # %% {Plotting}
    figs = plot_resonator_spectroscopy_vs_qubit_flux(ds, fit, qubits)
    node.results.update(figs)
    plt.show()

    ds = ds.transpose("qubit", "flux", "freq")

    # %% {Update_state}
    if not node.parameters.load_data_id:
        with node.record_state_updates():
            for q in qubits:
                if q.name not in fit:
                    continue

                if flux_point == "independent":
                    q.z.independent_offset = fit[q.name]["insensitive"]
                    q.z.reset_offset = fit[q.name]["crossing"]
                else:
                    raise NotImplementedError()

                if update_flux_min:
                    q.z.min_offset = fit[q.name]["minimum"]

                q.resonator.intermediate_frequency += fit[q.name]["frequency_at_insensitive"]

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()
