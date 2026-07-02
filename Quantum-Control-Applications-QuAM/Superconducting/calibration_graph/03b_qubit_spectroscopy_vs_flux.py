# %%
"""
        QUBIT SPECTROSCOPY VERSUS FLUX
This sequence involves doing a qubit spectroscopy for several flux biases in order to exhibit the qubit frequency
versus flux response.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Identification of the approximate qubit frequency ("qubit_spectroscopy").

Before proceeding to the next node:
    - Update the qubit frequency, in the state.
    - Update the relevant flux points in the state.
    - Update the frequency vs flux quadratic term in the state.
    - Save the current state
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubits_from_node,
)
from quam_libs.lib.fit import peaks_dips
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

    qubits: Optional[List[str]] = ['q4']
    num_averages: int = 500
    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.002 #0.004, 0.02 # q6:3e-3, q7:1e-2, q8:3e-3, q9:***,
    operation_len_in_ns: Optional[int] = None
    frequency_span_in_mhz: float = 200 #12, 120
    frequency_step_in_mhz: float = 1 #0.1, 1
    frequency_shift_in_mhz: float = 0 #0  
    min_flux_offset_in_v: float = -0.2 ##-0.042
    max_flux_offset_in_v: float = 0.0 #0.042
    num_flux_points: int = 41
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True


node = QualibrationNode(name="03b_Qubit_Spectroscopy_vs_Flux", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
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
num_qubits = len(qubits)

# selected coupler to drive flux from: 
# qp = machine.qubit_pairs["coupler_q1_q2"]


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
operation = node.parameters.operation  # The qubit operation to play
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
operation_len = node.parameters.operation_len_in_ns
if node.parameters.operation_amplitude_factor:
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
else:
    operation_amp = 1.0
# Qubit detuning sweep with respect to their resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
shift = int(node.parameters.frequency_shift_in_mhz * u.MHz)
dfs = np.arange(-span//2, span//2, step, dtype=np.int32)
# Flux bias sweep
dcs = np.linspace(
    node.parameters.min_flux_offset_in_v,
    node.parameters.max_flux_offset_in_v,
    node.parameters.num_flux_points,
)
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as multi_qubit_spec_vs_flux:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)  # QUA variable for the qubit frequency
    dc = declare(fixed)  # QUA variable for the flux dc level

    machine.apply_all_couplers_to_min()
    for i, qubit in enumerate(qubits):

        # Fixed qubit for debugging unknown flux-dependency: 
        fixed_qubit = machine.qubits[qubit.name]
        c = machine.qubits['q2'].z #machine.qubit_pairs['coupler_q5_q6'].coupler
        # fixed_qubit = machine.qubits["q4"]
        # for q in machine.active_qubits:
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
        qubit.z.settle()
        qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            with for_(*from_array(df, dfs)):
                # Update the qubit frequency
                fixed_qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + shift, keep_phase=True)
                with for_(*from_array(dc, dcs)):
                    # Flux sweeping for a qubit
                    duration = operation_len * u.ns if operation_len is not None else qubit.xy.operations[operation].length * u.ns
                    # Bring the qubit to the desired point during the saturation pulse
                    # qubit.z.play("const", amplitude_scale=dc / qubit.z.operations["const"].amplitude, duration=duration)
                    qubit.z.play("const", amplitude_scale=dc / qubit.z.operations["const"].amplitude, duration=duration)
                    # qp.coupler.play("const", amplitude_scale=dc / qubit.z.operations["const"].amplitude, duration=duration)
                    # Apply saturation pulse to all qubits
                    fixed_qubit.xy.play(
                        operation,
                        amplitude_scale=operation_amp,
                        duration=duration,
                    )
                    qubit.align()
                    # QUA macro to read the state of the active resonators
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
                    # Wait for the qubits to decay to the ground state
                    qubit.resonator.wait(machine.depletion_time * u.ns)

        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            I_st[i].buffer(len(dcs)).buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dcs)).buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_qubit_spec_vs_flux, simulation_config)
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
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(multi_qubit_spec_vs_flux)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is not None:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubits = resolve_qubits_from_node(machine, node)
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
    peaks = peaks_dips(ds.I, dim="freq", prominence_factor=15)
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
            quad_term = float(coeff.sel(degree=2, qubit=q.name))
            flux_shift_q = float(flux_shift.sel(qubit=q.name).values)
            drive_freq = float(freq_shift.sel(qubit=q.name).values)
            sweetspot_freq = drive_freq + q.xy.RF_frequency
            sweet_offset = offset + flux_shift_q
            flux_phase_ratio = np.sqrt(-quad_term / sweetspot_freq * 4)
            print(f"flux offset for qubit {q.name} is {offset*1e3 + flux_shift_q*1e3:.0f} mV")
            print(f"(shift of  {flux_shift_q*1e3:.0f} mV)")
            print(
                f"Drive frequency for {q.name} is {sweetspot_freq/1e9:.3f} GHz"
            )
            print(f"(shift of {drive_freq/1e6:.0f} MHz)")
            print(f"quad term for qubit {q.name} is {quad_term/1e9:.3e} GHz/V^2")
            print(f"sweetspot_freq for {q.name} is {sweetspot_freq/1e9:.6f} GHz")
            print(f"sweet_offset for {q.name} is {sweet_offset*1e3:.3f} mV")
            print(f"flux/phase ratio for {q.name} is {flux_phase_ratio:.6e} \n")
            fit_results[q.name]["flux_shift"] = flux_shift_q
            fit_results[q.name]["drive_freq"] = drive_freq
            fit_results[q.name]["quad_term"] = quad_term
            fit_results[q.name]["sweetspot_freq"] = sweetspot_freq
            fit_results[q.name]["sweet_offset"] = sweet_offset
            fit_results[q.name]["flux_phase_ratio"] = flux_phase_ratio
        else:
            print(f"No fit for qubit {q.name}")
            fit_results[q.name]["flux_shift"] = np.nan
            fit_results[q.name]["drive_freq"] = np.nan
            fit_results[q.name]["quad_term"] = np.nan
            fit_results[q.name]["sweetspot_freq"] = np.nan
            fit_results[q.name]["sweet_offset"] = np.nan
            fit_results[q.name]["flux_phase_ratio"] = np.nan
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
                # if q.name in ['q3', 'q5']:
                if not np.isnan(flux_shift.sel(qubit=q.name).values):
                    if flux_point == "independent":
                        q.z.independent_offset += fit_results[q.name]["flux_shift"]
                        if "c" in q.id: # for coupler-test case
                            q.z.joint_offset += fit_results[q.name]["flux_shift"]
                            q.z.independent_offset = q.z.joint_offset - q.phi0_voltage / 2 
                    elif flux_point == "joint":
                        q.z.joint_offset += fit_results[q.name]["flux_shift"] / 2
                    q.xy.intermediate_frequency += fit_results[q.name]["drive_freq"]
                    q.freq_vs_flux_01_quad_term = fit_results[q.name]["quad_term"]
                    q.extras["sweetspot_freq"] = fit_results[q.name]["sweetspot_freq"]
                    q.extras["sweet_offset"] = fit_results[q.name]["sweet_offset"]
                    q.extras["flux_phase_ratio"] = fit_results[q.name]["flux_phase_ratio"]

    # %% {Save_results}
    node.results["ds"] = ds
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()

# %%
