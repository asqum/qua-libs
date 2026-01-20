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
from datetime import datetime, timezone, timedelta
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset_simple
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
from scipy.signal import find_peaks
from quam_libs.lib.fit import fit_oscillation


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ["q1"]
    qubit_pair: str = "coupler_q1_q2"
    num_averages: int = 50
    frequency_span_in_mhz: float = 15
    frequency_step_in_mhz: float = 0.05
    min_flux_offset_in_v: float = -0.5
    max_flux_offset_in_v: float = 0.5
    num_flux_points: int = 101
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None


node = QualibrationNode(name="02e_resonator_spectroscopy_vs_coupler_flux", parameters=Parameters())
#node_id = get_node_id()


if node.parameters.qubit_pair is None:
    raise ValueError("Please specify the qubit_pair name")

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

qubit_pair = machine.qubit_pairs[node.parameters.qubit_pair]

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = qubit_pair.qubit_control
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)
resonators = [qubit.resonator for qubit in qubits]


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Qubit detuning sweep with respect to their resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, span // 2, step)
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
    comp_flux_qubit = declare(float)
    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        rr = resonators[i]
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(dc, dcs)):
                if "coupler_qubit_crosstalk" in qubit_pair.extras:
                    assign(comp_flux_qubit, qubit_pair.extras["coupler_qubit_crosstalk"] * dc )
                else:
                    assign(comp_flux_qubit, 0.0)
                # Flux sweeping for a qubit
                duration = operation_len * u.ns if operation_len is not None else qubit.xy.operations[operation].length * u.ns
                # Bring the qubit to the desired point during the saturation pulse
                qubit_pair.coupler.set_dc_offset(dc)
                qubit_pair.align()
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
        align(*[rr.name for rr in resonators])

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            I_st[i].buffer(len(dfs)).buffer(len(dcs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).buffer(len(dcs)).average().save(f"Q{i + 1}")


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
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
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
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs, "flux": dcs})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([dfs + q.resonator.RF_frequency for q in qubits]),
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

    ds = ds.assign_coords(freq_GHz=ds.freq_full / 1e9)
    for ax, qubit in grid_iter(grid):
        ds.loc[qubit].IQ_abs.plot(ax=ax, x="flux", y="freq_GHz", robust=True, add_colorbar=False)
        ax.legend(fontsize=8)
        ax.set_ylabel("Freq (GHz)")
        ax.set_xlabel("Coupler flux (V)")
        ax.set_title(f"{qubit['qubit']} - {qubit_pair.coupler.name}")
    grid.fig.suptitle(f"Resonator spectroscopy vs coupler flux")
    
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig


    # %% {Save_results}
    node.results["ds"] = ds
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%
