# %%
"""
    Coupler spectroscopy to find the resonant frequency of the coupler.
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters

from quam_libs.components import QuAM
from quam_libs.lib.instrument_limits import instrument_limits
from quam_libs.macros import qua_declaration, readout_state_coupler
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
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

    couplers: str = 'coupler_q3_q4'
    detector_qb:str = 'q4'
    driver_qb:str = 'q3'
    c_sweet_flux_amp: float = -0.235
    num_averages: int = 5000
    operation: str = "long_x180_cp"
    operation_amplitude_factor: Optional[float] = 1    #0.004, 0.0004
    operation_len_in_ns: Optional[int] = None
    frequency_span_in_mhz: float = 150 #200, 4, 800
    frequency_step_in_mhz: float = 1 #0.25, 0.01
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    target_peak_width: Optional[float] = 1e6 #1e6
    arbitrary_flux_bias: Optional[float] = None
    arbitrary_qubit_frequency_in_ghz: Optional[float] = None
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False


node = QualibrationNode(name="03x_Coupler_Spectroscopy", parameters=Parameters())


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


drive_q = [machine.qubits[node.parameters.driver_qb]]
detector_q = [machine.qubits[node.parameters.detector_qb]]
coupler = [machine.qubit_pairs[node.parameters.couplers]] # currently supports 1 coupler a time only.




# %% {QUA_program}
operation = node.parameters.operation  # The qubit operation to play
n_avg = node.parameters.num_averages  # The number of averages
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
dfs = np.arange(-span // 2, +span // 2, step, dtype=np.int32)
flux_point = node.parameters.flux_point_joint_or_independent
qubit_freqs = {q.name: q.xy.RF_frequency for q in drive_q}  # for opx

# Set the qubit frequency for a given flux point
# if node.parameters.arbitrary_flux_bias is not None:
#     arb_flux_bias_offset = {q.name: node.parameters.arbitrary_flux_bias for q in qubits}
#     detunings = {q.name: q.freq_vs_flux_01_quad_term * arb_flux_bias_offset[q.name] ** 2 for q in qubits}
# elif node.parameters.arbitrary_qubit_frequency_in_ghz is not None:
#     detunings = {
#         q.name: 1e9 * node.parameters.arbitrary_qubit_frequency_in_ghz - qubit_freqs[q.name] for q in qubits
#     }
#     arb_flux_bias_offset = {q.name: np.sqrt(detunings[q.name] / q.freq_vs_flux_01_quad_term) for q in qubits}

# else:
arb_flux_bias_offset = {q.name: 0.0 for q in drive_q}
detunings = {q.name: 0.0 for q in drive_q}


target_peak_width = node.parameters.target_peak_width
if target_peak_width is None:
    target_peak_width = (
        3e6  # the desired width of the response to the saturation pulse (including saturation amp), in Hz
    )

with program() as qubit_spec:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    state = [declare(int) for _ in range(len(detector_q))]
    state_st = [declare_stream() for _ in range(len(detector_q))]
    df = declare(int)  # QUA variable for the qubit frequency
    n_st = declare_stream()
    n = declare(int)
    if not node.parameters.simulate:
        machine.apply_all_couplers_to_min()
    for i, qubit in enumerate(drive_q):
        max_freq = dfs[-1] + qubit.xy.intermediate_frequency
        min_freq = dfs[0] + qubit.xy.intermediate_frequency
        assert max_freq <= 400e6 and min_freq >= -400e6, (
            f"{qubit.name} IF span out of range: min={min_freq/1e6:.2f} MHz, "
            f"max={max_freq/1e6:.2f} MHz (limit ±400 MHz), please adjust the frequency span.")
        
        # Bring the active qubits to the desired frequency point
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
        qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            for play_pi in [1, 0]:
                with for_(*from_array(df, dfs)):
                    # Update the qubit frequency
                    qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + detunings[qubit.name])
                    
                    # qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + detunings[qubit.name] + 700e6) # for coupler search 
                    wait(qubit.thermalization_time * u.ns)
                    align()
                    if not node.parameters.simulate:
                        duration = operation_len * u.ns if operation_len is not None else (qubit.xy.operations[operation].length + qubit.z.settle_time) * u.ns
                    else:
                        duration = 100 *u.ns
                    # # Bring the qubit to the desired point during the saturation pulse
                    # qubit.z.play("const", amplitude_scale=arb_flux_bias_offset[qubit.name] / qubit.z.operations["const"].amplitude, duration=duration)
                    # # Play the saturation pulse
                    # qubit.xy.wait(qubit.z.settle_time * u.ns)
                    align()
                    if play_pi:
                        print("YES PI")
                        detector_q[i].xy.play('x180')
                    else:
                        print("NO PI")
                    align()
                    coupler[0].coupler.play("const", amplitude_scale=node.parameters.c_sweet_flux_amp/coupler[0].coupler.operations["const"].amplitude, duration=duration)
                    qubit.xy.play(
                        operation,
                        amplitude_scale=operation_amp,
                        duration=duration,
                    )
                    align()
                    
                    # readout the resonator
                    readout_state_coupler(detector_q[i], state[i], method='zz-pi')
                    save(state[i], state_st[i])
                    # Wait for the qubit to decay to the ground state
                    detector_q[i].resonator.wait(machine.depletion_time * u.ns)

    with stream_processing():
        n_st.save("n")
        for i in range(len(drive_q)):
            state_st[i].buffer(len(dfs)).buffer(len([1, 0])).average().save(f"state{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, qubit_spec, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(qubit_spec)
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
        ds = fetch_results_as_xarray(job.result_handles, drive_q, {"freq": dfs, "detector_pre_state":[1,0]})
        # Convert IQ data into volts
        # ds = convert_IQ_to_V(ds, detector_q)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) and phase
        # ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # ds = ds.assign({"phase": np.arctan2(ds.Q, ds.I)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([dfs + qubit_freqs[q.name] + detunings[q.name] for q in drive_q]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}
    # %%
    print(np.array(ds.state.sel(detector_pre_state=1)).shape)
    plt.plot(ds.freq_full[0], ds.state.sel(detector_pre_state=1)[0], label='with pi pulse')
    plt.plot(ds.freq_full[0], ds.state.sel(detector_pre_state=0)[0], label='without pi pulse')
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Measured state")  
    plt.legend() 
    plt.show()                 

    # %% {Data_analysis}
    # search for frequency for which the amplitude the farthest from the mean to indicate the approximate location of the peak
    shifts = np.abs((ds.state - ds.state.mean(dim="freq"))).idxmax(dim="freq")
    # Find the rotation angle to align the separation along the 'I' axis
    # angle = np.arctan2(
    #     ds.sel(freq=shifts).Q - ds.Q.mean(dim="freq"),
    #     ds.sel(freq=shifts).I - ds.I.mean(dim="freq"),
    # )
    # # rotate the data to the new I axis
    # ds = ds.assign({"I_rot" : ds.I * np.cos(angle) + ds.Q * np.sin(angle)})
    # Find the peak with minimal prominence as defined, if no such peak found, returns nan
    result = peaks_dips(ds.state, dim="freq", prominence_factor=5)
    # The resonant RF frequency of the qubits
    abs_freqs = dict(
        [
            (
                q.name,
                ds.freq_full.sel(freq = result.position.sel(qubit=q.name).values).sel(qubit=q.name).values,
            )
            for q in drive_q if not np.isnan(result.sel(qubit=q.name).position.values)
        ]
    )

    # Save fitting results
    fit_results = {}
    for q in drive_q:
        fit_results[q.name] = {}
        if not np.isnan(result.sel(qubit=q.name).position.values):
            fit_results[q.name]["fit_successful"] = True
            Pi_length = q.xy.operations["x180"].length
            used_amp = q.xy.operations["saturation"].amplitude * operation_amp
            print(
                f"Drive frequency for {q.name} is "
                f"{(result.sel(qubit = q.name).position.values + q.xy.RF_frequency) / 1e9:.6f} GHz"
            )
            fit_results[q.name]["drive_freq"] = result.sel(qubit=q.name).position.values + q.xy.RF_frequency
            print(f"(shift of {result.sel(qubit = q.name).position.values/1e6:.3f} MHz)")
            factor_cw = float(target_peak_width / result.sel(qubit=q.name).width.values)
            factor_pi = np.pi / (result.sel(qubit=q.name).width.values * Pi_length * 1e-9)
            print(f"Found a peak width of {result.sel(qubit = q.name).width.values/1e6:.2f} MHz")
            print(
                f"To obtain a peak width of {target_peak_width/1e6:.1f} MHz the cw amplitude is modified "
                f"by {factor_cw:.2f} to {factor_cw * used_amp / operation_amp * 1e3:.0f} mV"
            )
            print(
                f"To obtain a Pi pulse at {Pi_length} ns the Rabi amplitude is modified by {factor_pi:.2f} "
                f"to {factor_pi*used_amp*1e3:.0f} mV"
            )
            # print(f"readout angle for qubit {q.name}: {angle.sel(qubit = q.name).values:.4}")
            print()
        else:
            fit_results[q.name]["fit_successful"] = False
            print(f"Failed to find a peak for {q.name}")
            print()
    node.results["fit_results"] = fit_results

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in detector_q])
    approx_peak = result.base_line + result.amplitude * (1 / (1 + ((ds.freq - result.position) / result.width) ** 2))
    for ax, qubit in grid_iter(grid):
        # Plot the line
        (ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].state).plot(ax=ax, x="freq_GHz")
        # Identify the resonance peak
        if not np.isnan(result.sel(qubit=qubit["qubit"]).position.values):
            ax.plot(
                abs_freqs[qubit["qubit"]] / 1e9,
                ds.loc[qubit].sel(freq=result.loc[qubit].position.values, method="nearest").state,
                ".r",
            )
            # # Identify the width
            (approx_peak.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit]).plot(
                ax=ax, x="freq_GHz", linewidth=0.5, linestyle="--"
            )
        ax.set_xlabel("Driving freq [GHz]")
        ax.set_ylabel("State Porbability")
        ax.set_title(node.parameters.couplers)
    grid.fig.suptitle("Coupler spectroscopy (amplitude)")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for q in drive_q:
                print(q.name)
                if not np.isnan(result.sel(qubit=q.name).position.values):
                    if flux_point == "arbitrary":
                        q.arbitrary_intermediate_frequency = float(
                            result.sel(qubit=q.name).position.values + detunings[q.name] + q.xy.intermediate_frequency
                        )
                        q.z.arbitrary_offset = arb_flux_bias_offset[q.name]
                    else:
                        q.xy.intermediate_frequency += float(result.sel(qubit=q.name).position.values)
                        q.xy.operations["x180_cp"] = {
                            "length": 100,
                            "digital_marker": "ON",
                            "axis_angle": 0,
                            "amplitude": 0.2,
                            "alpha": -0.7601,
                            "anharmonicity": f"#/qubits/{q.name}/anharmonicity",
                            "__class__": "quam.components.pulses.DragCosinePulse"}
                        q.xy.operations["x90_cp"] = {
                            "length": 100,
                            "digital_marker": "ON",
                            "axis_angle": 0,
                            "amplitude": 0.1,
                            "alpha": -0.7601,
                            "anharmonicity": f"#/qubits/{q.name}/anharmonicity",
                            "__class__": "quam.components.pulses.DragCosinePulse"}
        node.results["ds"] = ds

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in coupler}
        node.results["initial_parameters"] = node.parameters.model_dump()
        
        node.save()


# %%
