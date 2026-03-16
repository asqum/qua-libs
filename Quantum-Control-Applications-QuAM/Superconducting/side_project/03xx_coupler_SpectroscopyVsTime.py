# %%
"""
    Coupler spectroscopy to find the resonant frequency of the coupler.

    Prerequisites:
        - Having a aSWAP operation for you detector_qb in state.json with amplitude is a half flux period, length is about 400 ns which can be manually extended in the future if needed.
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
from quam.components import pulses
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import xarray as xr

# %% {Node_parameters}
class Parameters(NodeParameters):

    couplers: str = 'coupler_q4_q5'
    num_averages: int = 500
    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.05    # 0.05 , 0.1 good
    operation_len_in_ns: Optional[int] = None
    Driving_LO_GHz:float|None = None      # None use state recorded. Otherwise, use this value as the new LO (and will be updated into state)
    frequency_span_in_mhz: float = 400 #200, 4, 800
    frequency_step_in_mhz: float = 2 #0.25, 0.01
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    target_peak_width: Optional[float] = 1e6 #1e6
    arbitrary_flux_bias: Optional[float] = None
    arbitrary_qubit_frequency_in_ghz: Optional[float] = None
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    repetitions:int = 100


node = QualibrationNode(name="03x_Coupler_Spectroscopy", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations

coupler = [machine.qubit_pairs[node.parameters.couplers]] # currently supports 1 coupler a time only.
drive_q = [machine.qubits[coupler[0].extras["RD"]["driven_q"]]]
detector_q = [machine.qubits[coupler[0].extras["RD"]["readout_q"]]]
# Change driving LO
if node.parameters.load_data_id is None and not node.parameters.simulate:
    drive_LO_original = {drive_q[0].name: drive_q[0].xy.opx_output.upconverter_frequency}
    if node.parameters.Driving_LO_GHz is None:
        LO_to_plot = coupler[0].extras["RD"]["LO"]
        drive_q[0].xy.opx_output.upconverter_frequency = coupler[0].extras["RD"]["LO"]
    else:
        LO_to_plot = node.parameters.Driving_LO_GHz*1e9
        drive_q[0].xy.opx_output.upconverter_frequency = node.parameters.Driving_LO_GHz*1e9
    if "swap_direction" in coupler[0].extras["RD"]:
        detector_q[0].z.operations['aSWAP'].slope_direction = coupler[0].extras["RD"]["swap_direction"]
    



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
        max_freq = dfs[-1] + coupler[0].extras["RD"]["IF"]
        min_freq = dfs[0] + coupler[0].extras["RD"]["IF"]
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
            with for_(*from_array(df, dfs)):
                # Update the qubit frequency
                qubit.xy.update_frequency(df + coupler[0].extras["RD"]["IF"] + detunings[qubit.name])
                
                # qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + detunings[qubit.name] + 700e6) # for coupler search 
                qubit.align()
                if not node.parameters.simulate:
                    duration = operation_len * u.ns if operation_len is not None else (qubit.xy.operations[operation].length + qubit.z.settle_time) * u.ns
                else:
                    duration = 100 *u.ns
                # # Bring the qubit to the desired point during the saturation pulse
                # qubit.z.play("const", amplitude_scale=arb_flux_bias_offset[qubit.name] / qubit.z.operations["const"].amplitude, duration=duration)
                # # Play the saturation pulse
                # qubit.xy.wait(qubit.z.settle_time * u.ns)
                align()
                qubit.xy.play(
                    operation,
                    amplitude_scale=operation_amp,
                    duration=duration,
                )
                # coupler[0].coupler.play(
                #     "const",
                #     amplitude_scale=0,
                #     duration=duration
                # )
        

                # readout the resonator
                readout_state_coupler(detector_q[i], state[i], method='aswap')
                save(state[i], state_st[i])
                # Wait for the qubit to decay to the ground state
                detector_q[i].resonator.wait(machine.depletion_time * u.ns)

    with stream_processing():
        n_st.save("n")
        for i in range(len(drive_q)):
            state_st[i].buffer(len(dfs)).average().save(f"state{i + 1}")


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
else:
    if node.parameters.load_data_id is None:
        dss = {}
        for ii in range(node.parameters.repetitions):
            try:
                config = machine.generate_config()
                qmm = machine.connect()


                with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                    job = qm.execute(qubit_spec)
                    results = fetching_tool(job, ["n"], mode="live")
                    while results.is_processing():
                        # Fetch results
                        n = results.fetch_all()[0]
                        # Progress bar
                        progress_counter(n, n_avg, start_time=results.start_time)

                ds = fetch_results_as_xarray(job.result_handles, drive_q, {"freq": dfs})
                
                ds = ds.assign_coords(
                    {
                        "freq_full": (
                            ["qubit", "freq"],
                            np.array([dfs + coupler[0].extras["RD"]["IF"] + LO_to_plot + detunings[q.name] for q in drive_q]),
                        )
                    }
                )
                ds.freq_full.attrs["long_name"] = "Frequency"
                ds.freq_full.attrs["units"] = "GHz"

                current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dss[current_time_str] = ds
            except:
                break
            
        time_labels = [pd.to_datetime(t) for t in dss.keys()]
        datasets = list(dss.values())  
        combined_ds = xr.concat(datasets, dim=pd.Index(time_labels, name='timestamp'))
        ds = combined_ds.sortby('timestamp')
        node.results["ds"] = ds

    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)


# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:

    if node.parameters.load_data_id is None:
        for q in drive_q:
            a_ds = ds.sel(qubit=q.name)

            plt.figure(figsize=(12, 6))


            heatmap = a_ds.state.plot(
                x='freq_full', 
                y='timestamp', 
                cmap='viridis',    # 推薦使用 viridis 或 inferno
                cbar_kwargs={'label': 'State Value'}
            )

            
            plt.title(f"Spectroscopy vs Time (Coupler: {coupler[0].name})")
            plt.xlabel("Driving Frequency [GHz]")
            plt.ylabel("Timestamp")

            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x*1e-9:.3f}')

            plt.tight_layout()
            plt.grid()
            node.results[f"figure_{q.name}"] = plt.gcf()
            plt.show()

        # %% {Save_results}
        for q in drive_q:
            q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
        for q in detector_q:
            q.z.operations['aSWAP'].slope_direction = -1
        node.outcomes = {q.name: "successful" for q in coupler}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()


# %%
