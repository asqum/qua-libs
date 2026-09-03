# %%
"""
        QUBIT SPECTROSCOPY
This sequence involves sending a saturation pulse to the qubit, placing it in a mixed state,
and then measuring the state of the resonator across various qubit drive intermediate frequencies dfs.
In order to facilitate the qubit search, the qubit pulse duration and amplitude can be changed manually in the QUA
program directly from the node parameters.

The data is post-processed to determine the qubit resonance frequency and the width of the peak.

Note that it can happen that the qubit is excited by the image sideband or LO leakage instead of the desired sideband.
This is why calibrating the qubit mixer is highly recommended.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Set the flux bias to the desired working point, independent, joint or arbitrary, in the state.
    - Configuration of the saturation pulse amplitude and duration to transition the qubit into a mixed state.

Before proceeding to the next node:
    - Update the qubit frequency in the state, as well as the expected x180 amplitude and IQ rotation angle.
    - Save the current state
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters

from quam_libs.components import QuAM
from quam_libs.lib.instrument_limits import instrument_limits
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

    qubits: Optional[List[str]] = ['q6']
    num_averages: int = 200
    operation_amplitude_factor: Optional[float] = 1  #0.004, 0.0004
    frequency_span_in_mhz: float = 200 #200, 4, 800
    frequency_step_in_mhz: float = 5 #0.25, 0.01
    ROamp_max_dB: float = -1 # must <= 6 (ratio = 2)
    ROamp_min_dB: float = -30
    sampling_scale :Literal['lin','log'] = 'lin'
    amp_pts:int = 30
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    target_peak_width: Optional[float] = 30e6 #1e6
    arbitrary_flux_bias: Optional[float] = None
    arbitrary_qubit_frequency_in_ghz: Optional[float] = None
    simulate: bool = False
    simulation_duration_ns: int = 10_000
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    xy_ro_overlap: bool = True


node = QualibrationNode(name="03dx_Overlap2tone_ROamp", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine

# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

operation = 'saturation'

original_paras = {}
if node.parameters.xy_ro_overlap:
    for q in qubits:
        original_paras[q.name] = {"saturation_length":q.xy.operations[operation].length, "readout_length":q.resonator.operations['readout'].length}
        q.xy.operations[operation].length = 100_000
        q.resonator.operations['readout'].length = 100_000

        if q.xy.thread == q.resonator.thread:
            original_paras[q.name]['readout_thread'] = q.resonator.thread
            q.resonator.thread = q.xy.thread*4

# Generate the OPX and Octave configurations
config = machine.generate_config()


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
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
qubit_freqs = {q.name: q.xy.RF_frequency for q in qubits}  # for opx

if node.parameters.ROamp_max_dB > 6:
    raise ValueError("The maximum readout amplitude scale must <= 6 !")

if node.parameters.sampling_scale == 'log':
    amps = np.geomspace(
        10 ** (node.parameters.ROamp_min_dB/20.0),
        10 ** (node.parameters.ROamp_max_dB/20.0),
        node.parameters.amp_pts,
    )
else:
    amps = np.linspace(
        10 ** (node.parameters.ROamp_min_dB/20.0),
        10 ** (node.parameters.ROamp_max_dB/20.0),
        node.parameters.amp_pts,
    )


# %%
# Set the qubit frequency for a given flux point
if node.parameters.arbitrary_flux_bias is not None:
    arb_flux_bias_offset = {q.name: node.parameters.arbitrary_flux_bias for q in qubits}
    detunings = {q.name: q.freq_vs_flux_01_quad_term * arb_flux_bias_offset[q.name] ** 2 for q in qubits}
elif node.parameters.arbitrary_qubit_frequency_in_ghz is not None:
    detunings = {
        q.name: 1e9 * node.parameters.arbitrary_qubit_frequency_in_ghz - qubit_freqs[q.name] for q in qubits
    }
    arb_flux_bias_offset = {q.name: np.sqrt(detunings[q.name] / q.freq_vs_flux_01_quad_term) for q in qubits}

else:
    arb_flux_bias_offset = {q.name: 0.0 for q in qubits}
    detunings = {q.name: 0.0 for q in qubits}


target_peak_width = node.parameters.target_peak_width
if target_peak_width is None:
    target_peak_width = (
        3e6  # the desired width of the response to the saturation pulse (including saturation amp), in Hz
    )

with program() as qubit_spec:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)  # QUA variable for the qubit frequency
    amp = declare(fixed)
    machine.apply_all_couplers_to_min()
    for i, qubit in enumerate(qubits):
        max_freq = dfs[-1] + qubit.xy.intermediate_frequency
        min_freq = dfs[0] + qubit.xy.intermediate_frequency
        assert max_freq <= 400e6 and min_freq >= -400e6, (
            f"{qubit.name} IF span out of range: min={min_freq/1e6:.2f} MHz, "
            f"max={max_freq/1e6:.2f} MHz (limit ±400 MHz), please adjust the frequency span.")
        
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
        qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                # Update the qubit frequency
                qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + detunings[qubit.name])
                with for_each_(amp, amps):
                    qubit.align()
                    qubit.xy.play(operation,amplitude_scale=operation_amp)
                    # readout the resonator
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]), amplitude_scale=amp)
                    # Wait for the qubit to decay to the ground state
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
                    qubit.resonator.wait(10 * machine.depletion_time * u.ns)

        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            I_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns)  # In clock cycles = 4ns
    job = qmm.simulate(config, qubit_spec, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    

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
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubits = resolve_qubits_from_node(machine, node)
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"amp": amps, "freq": dfs})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) and phase
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([dfs + qubit_freqs[q.name] + detunings[q.name] for q in qubits]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    
    for ax, qubit in grid_iter(grid):
        da = ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].I
        da_norm = da - da.mean(dim="freq")
        da_norm = (da - da.mean(dim="freq")) / da.std(dim="freq")
        da_norm.plot(ax=ax, add_colorbar=False, x="amp", y="freq_GHz", robust=True)
        ax.set_ylabel("Freq (GHz)")
        ax.set_xlabel("RO Amplitude factor")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Qubit spectroscopy with RO power (Overlap)")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Save_results}
    for q in qubits:
        q.resonator.operations['readout'].length = original_paras[q.name]['readout_length']
        q.resonator.thread = original_paras[q.name]['readout_thread']

    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    
    node.save()


# %%
