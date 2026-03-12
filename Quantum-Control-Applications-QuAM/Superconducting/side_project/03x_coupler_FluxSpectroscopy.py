# %%
"""
The transition frequency flux spectrum for the target coupler.

Prerequisites:
    - the driving frequency for the target coupler.

Recommended readout strategy: 'aswap' for better contrast because the ZZ interaction might be small.
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
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
    couplers: str = 'coupler_q4_q5' #'coupler_q3_q4'
    readout_strategy: Literal['zz-pi', 'aswap'] = 'aswap'
    num_averages: int = 500
    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.02 #0.004, 0.02 # q6:3e-3, q7:1e-2, q8:3e-3, q9:***,
    operation_len_in_ns: Optional[int] = None
    Driving_LO_GHz: float|None = None # 3.18
    frequency_span_in_mhz: float = 100 #12, 120
    frequency_step_in_mhz: float = 2 #0.1, 1
    frequency_shift_in_mhz: float = 0 #0  
    min_flux_offset_in_v: float = -0.025 ##-0.042
    max_flux_offset_in_v: float = 0.025 #0.042
    num_flux_points: int = 51
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False


node = QualibrationNode(name="03x_coupler_Spectroscopy_vs_Flux", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()


# Get the relevant QuAM components
coupler = [machine.qubit_pairs[node.parameters.couplers]] # currently supports 1 coupler a time only.
drive_q = [machine.qubits[coupler[0].extras["RD"]["driven_q"]]]
detector_q = [machine.qubits[coupler[0].extras["RD"]["readout_q"]]]
# Change driving LO
if node.parameters.load_data_id is None and not node.parameters.simulate:
    drive_LO_original = {drive_q[0].name: drive_q[0].xy.opx_output.upconverter_frequency}
    if node.parameters.Driving_LO_GHz is None:
        drive_q[0].xy.opx_output.upconverter_frequency = coupler[0].extras["RD"]["LO"]
        LO_to_plot = coupler[0].extras["RD"]["LO"]
    else:
        LO_to_plot = node.parameters.Driving_LO_GHz * 1e9
        drive_q[0].xy.opx_output.upconverter_frequency = node.parameters.Driving_LO_GHz * 1e9

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()



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
    _, _, _, _, n, n_st = qua_declaration(num_qubits=len(detector_q))
    state = [declare(int) for _ in range(len(detector_q))]
    state_st = [declare_stream() for _ in range(len(detector_q))]
    df = declare(int)  # QUA variable for the qubit frequency
    dc = declare(fixed)  # QUA variable for the flux dc level

    machine.apply_all_couplers_to_min()
    for i, qubit in enumerate(drive_q):

        # Fixed qubit for debugging unknown flux-dependency: 
        fixed_qubit = machine.qubits[qubit.name]
        c = coupler[0].coupler
        
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
        qubit.z.settle()
        qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            with for_(*from_array(df, dfs)):
                # Update the qubit frequency
                fixed_qubit.xy.update_frequency(df + coupler[0].extras["RD"]["IF"] + shift, keep_phase=True)
                with for_(*from_array(dc, dcs)):
                    # Flux sweeping for a qubit
                    duration = operation_len * u.ns if operation_len is not None else qubit.xy.operations[operation].length * u.ns
                    # Bring the qubit to the desired point during the saturation pulse
                    # qubit.z.play("const", amplitude_scale=dc / qubit.z.operations["const"].amplitude, duration=duration)
                    c.play("const", amplitude_scale=dc / qubit.z.operations["const"].amplitude, duration=duration)
                    # qp.coupler.play("const", amplitude_scale=dc / qubit.z.operations["const"].amplitude, duration=duration)
                    # Apply saturation pulse to all qubits
                    fixed_qubit.xy.play(
                        operation,
                        amplitude_scale=operation_amp,
                        duration=duration,
                    )
                    qubit.align()
                    # QUA macro to read the state of the active resonators
                    readout_state_coupler(detector_q[i], state[i], method=node.parameters.readout_strategy)
                    save(state[i], state_st[i])
                    # Wait for the qubit to decay to the ground state
                    # Wait for the qubits to decay to the ground state
                    qubit.resonator.wait(machine.depletion_time * u.ns)

        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(drive_q):
            state_st[i].buffer(len(dcs)).buffer(len(dfs)).average().save(f"state{i + 1}")



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
    node.machine = machine
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
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, drive_q, {"flux": dcs, "freq": dfs})
        # Convert IQ data into volts
        # ds = convert_IQ_to_V(ds, qubits)
        # # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        # ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([shift + dfs + coupler[0].extras["RD"]["IF"] + LO_to_plot for q in drive_q]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # Find the resonance dips for each flux point
    peaks = peaks_dips(ds.state, dim="freq", prominence_factor=6)
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
    for q in drive_q:
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
                f"Drive frequency for {q.name} is {(freq_shift.sel(qubit = q.name).values + coupler[0].extras['RD']['IF'] + LO_to_plot)/1e9:.3f} GHz"
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
    grid = QubitGrid(ds, [q.grid_location for q in drive_q])

    for ax, qubit in grid_iter(grid):
        freq_ref = coupler[0].extras["RD"]["IF"] + LO_to_plot #machine.qubits[qubit["qubit"]].xy.RF_frequency
        ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].state.plot(
            ax=ax, add_colorbar=False, x="flux", y="freq_GHz", robust=True
        )
        ((fitted + freq_ref) / 1e9).loc[qubit].plot(ax=ax, linewidth=0.5, ls="--", color="r")
        ax.plot(flux_shift.loc[qubit], ((freq_shift.loc[qubit] + freq_ref) / 1e9), "r*")
        ((peaks.position.loc[qubit] + freq_ref) / 1e9).plot(ax=ax, ls="", marker=".", color="g", ms=0.5)
        ax.set_ylabel("Freq (GHz)")
        ax.set_xlabel("Flux (V)")
        ax.set_title(node.parameters.couplers)
    grid.fig.suptitle("coupler spectroscopy vs flux ")

    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    if node.parameters.load_data_id is None and not node.parameters.simulate:
        with node.record_state_updates():
            for q in drive_q:
                if node.parameters.Driving_LO_GHz is not None:
                    coupler[0].extras["RD"]["LO"] = node.parameters.Driving_LO_GHz * 1e9

        # %% {Save_results}
        for q in drive_q:
            q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
        node.results["ds"] = ds
        node.outcomes = {q.name: "successful" for q in drive_q}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()

# %%
