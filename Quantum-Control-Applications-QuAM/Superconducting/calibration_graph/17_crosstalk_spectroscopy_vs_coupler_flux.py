# %% {Imports}
import warnings

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualang_tools.loops import from_array

from qualibrate import QualibrationNode
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset
from qualang_tools.results import progress_counter, fetching_tool

from calibration_utils.crosstalk_spectroscopy_vs_flux.program import (
    get_expected_frequency_at_flux_detuning,
    get_flux_detuning_in_v
)
from calibration_utils.crosstalk_spectroscopy_vs_flux.analysis import fit_lorentzian_peaks
from calibration_utils.crosstalk_spectroscopy_vs_flux.plotting import plot_analysis, add_node_info_subtitle
from calibration_utils.crosstalk_spectroscopy_vs_flux.fitting import fit_linear
from copy import deepcopy
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualibrate import QualibrationNode, NodeParameters
from typing import Literal, Optional, List
from qm import SimulationConfig
from quam_libs.lib.qua_datasets import convert_IQ_to_V

# %% {Description}

description = """
Qubit Spectroscopy for Crosstalk Calibration
This experiment performs qubit spectroscopy while sweeping the flux bias of a neighboring qubit or tunable coupler,
in order to map the target qubit’s frequency response as a function of the other element’s flux bias.
The resulting frequency–flux map is used to extract and compensate for flux crosstalk.

Purpose:
    - Measure the dependence of the target qubit’s f_01 on a neighboring qubit or coupler’s flux bias.
    - Determine the crosstalk slope (∂f_target/∂Φ_neighbor) for building a crosstalk compensation matrix.
    - Verify and refine flux bias settings to isolate control channels.

Prerequisites:
    - XY vs. Z channel delay correctly calibrated.
    - Mixer or Octave calibration completed (nodes 01a or 01b).
    - Readout parameters calibrated (nodes 02a, 02b, and/or 02c).
    - Target qubit frequency calibrated at its nominal flux point (03a_qubit_spectroscopy.py).
    - Flux operating points defined for both the target and the neighboring element
      (e.g., qubit.z.flux_point and coupler.z.flux_point).

State Update:
    - Measured f_01 of the target qubit vs. neighbor flux bias.
    - Extracted crosstalk coefficients for compensation.
    - Updated flux bias offsets for independent or joint control: q.z.independent_offset or q.z.joint_offset.
"""

class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    target_qubits: Optional[List[str]] = ["q10"]
    """Target qubit"""
    qubit_pairs: Optional[List[str]] = ["coupler_q9_q10"]
    """aggressor qubit"""
    num_shots: int = 100
    """Number of averages to perform. Default is 50."""
    operation: str = "saturation"
    """Operation to perform. Default is "saturation"."""
    operation_amplitude_factor: float = 0.005
    """Amplitude factor for the operation. Default is 0.1."""
    operation_len_in_ns: Optional[int] = None
    """Length of the operation in ns. Default is the predefined pulse length."""
    target_qubit_frequency_span: float = 20
    """Frequency span of target qubit in MHz. Default is 100 MHz."""
    frequency_num_points: float = 51
    """Frequency number of points."""
    flux_span: float = 0.01
    """Minimum target flux bias offset in volts. Default is -0.01 V."""
    flux_num_points: int = 101
    """Number of flux points. Default is 51."""
    flux_pulse_padding_in_ns: float = 2000
    """Extra padding time between the flux pulse and pi-pulse, which is also doubled and added to the duration of the flux pulse"""
    manual_flux_detuning_in_v: float = 0.06
    """Target qubit's flux detuning when the mode is set to manual."""
    load_data_id: Optional[int] = None
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    flux_detuning_mode: Literal["auto_for_linear_response", "auto_fill_sweep_window", "manual"] = "manual"
    """Strategy for choosing the target qubit's flux detuning."""



node = QualibrationNode(
    name="17_crosstalk_spectroscopy_vs_flux",
    parameters=Parameters(),
)

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)

# Instantiate the QUAM class from the state file
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
num_qubits = len(qubits)

node.target_qubits = target_qubits = [machine.qubits[q] for q in node.parameters.target_qubits]
node.qubit_pairs = qubit_pairs = [machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]
combined_element_list =  target_qubits + qubit_pairs 

# %% {Create_QUA_program}
n_avg = node.parameters.num_shots  # The number of averages
reset_type = node.parameters.reset_type_thermal_or_active
operation = node.parameters.operation  # The qubit operation to play
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
operation_len = node.parameters.operation_len_in_ns
if node.parameters.operation_amplitude_factor:
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
else:
    operation_amp = 1.0

# Qubit detuning sweep with respect to their resonance frequencies
target_qubit_frequency_span = node.parameters.target_qubit_frequency_span * u.MHz
frequency_num_points = node.parameters.frequency_num_points

dfs = np.linspace(-target_qubit_frequency_span/2, target_qubit_frequency_span/2, frequency_num_points)

# Target sweep grid
dcs = np.linspace(
    -node.parameters.flux_span,
   node.parameters.flux_span,
    node.parameters.flux_num_points,
)

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'


# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Get the target and aggressor qubits

if any([q.z is None for q in qubits]):
    warnings.warn("Found qubits without a flux line. Skipping")
flux_pulse_padding = node.parameters.flux_pulse_padding_in_ns

with program() as crosstalk_vs_flux:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)  # QUA variable for the qubit frequency
    dc = declare(fixed)  # QUA variable for the flux dc level
    comp_flux_qubit = declare(float)
    # Target qubit flux detunings
    flux_detunings = {
            q.name: get_flux_detuning_in_v(node.parameters, q) for q in target_qubits
        }
    for i, target_qubit in enumerate(target_qubits):
             
        flux_detuning = flux_detunings[target_qubit.name]
        expected_frequency = get_expected_frequency_at_flux_detuning(target_qubit, flux_detuning)
        expected_frequency_offset = expected_frequency - target_qubit.xy.RF_frequency
        
        for j, element in enumerate(combined_element_list):
             
            if target_qubit.name == element.name: 
                aggressor_z_line = target_qubit.z
            else:
                aggressor_z_line = element.coupler
             
            # Initialize the QPU in terms of flux points (flux-tunable transmons and/or tunable couplers)
            for qubit in qubits:
                machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            set_dc_offset(target_qubit.z.name, "single", target_qubit.z.independent_offset + flux_detuning)
            target_qubit.z.settle()
            target_qubit.align()
            # if "coupler" in element.name:
            #     element.coupler.set_dc_offset(-0.3)
            #     wait(200)
            #     element.align()
            
            
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    # Update the qubit frequency
                    target_qubit.xy.update_frequency(df + target_qubit.xy.intermediate_frequency-30e6)
                    with for_(*from_array(dc, dcs)):
                        if node.parameters.reset_type_thermal_or_active == "active":
                            # active_reset(qubit, "readout")
                            active_reset(target_qubit, "readout")
                        else:
                            target_qubit.resonator.wait(qubit.thermalization_time * u.ns)
                            element.align()
                        if "coupler" in element.name:
                            if "coupler_qubit_crosstalk" in element.extras:
                                assign(comp_flux_qubit, element.extras["coupler_qubit_crosstalk"] * dc )
                            else:
                                assign(comp_flux_qubit, 0.0)
                        # Flux sweeping for a qubit
                        duration = operation_len * u.ns if operation_len is not None else target_qubit.xy.operations[operation].length * u.ns
                        # Bring the qubit to the desired point during the saturation pulse
                        if "coupler" in element.name:
                            target_qubit.z.play(
                                "const", 
                                amplitude_scale = comp_flux_qubit / qubit.z.operations["const"].amplitude, 
                                duration = duration +  2 * (flux_pulse_padding // 4)
                            )
                        aggressor_z_line.play(
                            "const", 
                            amplitude_scale = dc / aggressor_z_line.operations["const"].amplitude, 
                            duration = duration +  2 * (flux_pulse_padding // 4)
                            )
                        # let the flux settle
                        target_qubit.xy.wait(200)
                        target_qubit.xy.play(
                            operation,
                            amplitude_scale=operation_amp,
                            duration=duration,
                        )
                        target_qubit.align()
                        # QUA macro to read the state of the active resonators
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        # Wait for the qubits to decay to the ground state
                        qubit.resonator.wait(machine.depletion_time * u.ns)

            align()

    with stream_processing():
        n_st.save("n")
        for i, target_qubit in enumerate(target_qubits):
            I_st[i].buffer(len(dcs)).buffer(len(dfs)).buffer(n_avg).map(FUNCTIONS.average()).buffer(len(combined_element_list)).save(f"I{i + 1}")
            Q_st[i].buffer(len(dcs)).buffer(len(dfs)).buffer(n_avg).map(FUNCTIONS.average()).buffer(len(combined_element_list)).save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, crosstalk_vs_flux, simulation_config)
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
        job = qm.execute(crosstalk_vs_flux)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)


# %% {Data_fetching_and_dataset_creation}
aggressors = []
for q in combined_element_list:
            aggressors.append(q.name)
if not node.parameters.simulate:
    if node.parameters.load_data_id is not None:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, target_qubits, {"flux_bias": dcs, "detuning": dfs, "aggressor": aggressors})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, target_qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        ds = ds.assign_coords(
            {
                "full_freq": (
                    ["qubit", "detuning"],
                    np.array([dfs + q.xy.RF_frequency for q in target_qubits]),
                )
            }
        )
        ds.full_freq.attrs["long_name"] = "Frequency"
        ds.full_freq.attrs["units"] = "GHz"
        ds = ds.assign_coords(
            {
                "flux_qubit_full": (
                    ["qubit", "flux_bias"],
                    np.array([dcs + q.z.independent_offset for q in target_qubits]),
                )
            }
        )
        ds.flux_qubit_full.attrs["long_name"] = "flux_qubit_full"
        ds.flux_qubit_full.attrs["units"] = "V"
    # Add the dataset to the node
    node.results = {"ds": ds}

# %% {Analyse_data}
def analyse_data(ds, target_qubits, combined_element_list):
    peak_results = fit_lorentzian_peaks(ds, target_qubits, combined_element_list)
    fit_results = {str(target_qubit_name.data): {} for target_qubit_name in peak_results.qubit}

    for pair in peak_results.pair:
        peak_freq = peak_results.sel(pair=pair).peak_frequencies
        flux_bias = peak_freq.flux_bias
        target_qubit_name = str(pair.qubit.data)
        aggressor_qubit_name = str(pair.aggressor.data)
        slope, intercept, inlier_mask = fit_linear(flux_bias, peak_freq)
        fit_results[target_qubit_name][aggressor_qubit_name] = dict(
                linear_fit_slope=slope,
                linear_fit_intercept=intercept,
                linear_fit_inlier_mask=inlier_mask,
                success=True
                )
    return ds, fit_results, peak_results

def append_crosstalk_coefficient(fit_results):
    # make a deep copy if you don’t want to mutate in place
    results = deepcopy(fit_results)

    target_qubits = list(results.keys())
    for target in target_qubits:
        self_slope = results[target][target].get("linear_fit_slope", np.nan)

        for aggressor in results[target].keys():
            slope = results[target][aggressor].get("linear_fit_slope", np.nan)
            if self_slope == 0 or np.isnan(self_slope):
                coeff = np.nan
            else:
                coeff = slope / self_slope

            # add back into dict
            results[target][aggressor]["crosstalk_coefficient"] = coeff
    return results



node.results["ds_fit"], fit_results, peak_results = analyse_data(node.results["ds"], target_qubits, combined_element_list)
node.results["peak_results"] = peak_results
fit_results = append_crosstalk_coefficient(fit_results)
node.results["fit_results"] = {k: v for k, v in fit_results.items()}
node.results["flux_detunings"] = flux_detunings
        
# %% {Plot_data}
fig = plot_analysis(
    node.results["ds"], node.results["peak_results"], node.results["fit_results"],
    node.results.get("flux_detunings"), machine.qubits
)
add_node_info_subtitle(node, fig)

node.results["figures"] = {"main": fig}

plt.show()


# %% {Update_state}

"""Update the relevant parameters if the qubit data analysis was successful."""
with node.record_state_updates():
    for target_qubit_name, target_qubit_results in node.results["fit_results"].items():
        for aggressor_qubit_name, fit_result in target_qubit_results.items():
            if target_qubit_name == aggressor_qubit_name:
                continue

            # Find the qubit objects
            target_qubit = node.machine.qubits[target_qubit_name]
            aggressor_qubit = node.machine.qubits[aggressor_qubit_name]

            target_output = target_qubit.z.opx_output
            aggressor_output = aggressor_qubit.z.opx_output

            # Update crosstalk coefficients
            if target_output.fem_id == aggressor_output.fem_id and target_output.controller_id == aggressor_output.controller_id:
                if not target_output.crosstalk:
                    target_output.crosstalk = {}
                if not aggressor_output.port_id in target_output.crosstalk or np.isnan(target_output.crosstalk[aggressor_output.port_id]):
                    target_output.crosstalk[aggressor_output.port_id] = 0
                target_output.crosstalk[aggressor_output.port_id] += fit_result["crosstalk_coefficient"]

            else:
                node.log(f"Couldn't compensate crosstalk between {target_qubit.name} and {aggressor_qubit.name}, "
                            f"since they are on different fems ({target_output.controller_id, target_output.fem_id} and "
                            f"{aggressor_output.controller_id, aggressor_output.fem_id}) respectively.")

# %% {Save_results}
node.results["ds"] = ds
node.outcomes = {q.name: "successful" for q in qubits}
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
node.results['peak_results'] = node.results['peak_results'].reset_index('pair')
node.save()

# %%
