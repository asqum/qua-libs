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
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubit_pairs_from_node,
    resolve_qubits_from_node,
)
from calibration_utils.resonator_vs_coupler_flux.analysis import (
    analyze_decouple_offsets,
    format_decouple_offset_summary,
    match_loaded_qubit_pair_dataset,
    prepare_fetched_qubit_pair_dataset,
    qubits_for_pair,
)
from calibration_utils.resonator_vs_coupler_flux.plot import plot_decouple_offset_maps
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

class SweepingParameters(NodeParameters):

    qubits: Optional[List[str]] = None  # List of qubits to perform the measurement on, if None, it will be inferred from the qubit pairs
    qubit_pairs: Optional[List[str]] = ["coupler_q1_q2", "coupler_q2_q3","coupler_q3_q4","coupler_q4_q5"]  # List of qubit pair names to perform the measurement on, if None, it will be performed on all available pairs
    num_averages: int = 50
    frequency_span_in_mhz: float = 25
    frequency_step_in_mhz: float = 0.1
    min_flux_offset_in_v: float = -0.5
    max_flux_offset_in_v: float = 0.5
    num_flux_points: int = 201
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    # dip_edge_points: int = 8
    # dip_smooth_window: int = 1
    # dip_prominence_fraction: float = 0.10
    # dip_min_distance: int = 10
    # fit_weight_power: float = 1.5
    # fit_f_scale: float = 0.30
    # enforce_fc_max_above_fr: bool = True
    # min_fc_max_above_fr_in_mhz: float = 500
    # fit_selection_min_valid_dips: int = 5
    update_coupler_decouple: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None  # If specified, it will load the data from the given node id instead of executing the QUA program


class FittingParameters(NodeParameters):
    detrend_fit_min_percentile: int = 35
    row_span_prominence_fraction: float = 0.05
    minima_distance: int = 5

class Parameters(SweepingParameters, FittingParameters):
    pass

node = QualibrationNode(name="02e_resonator_spectroscopy_vs_coupler_flux", parameters=Parameters())
#node_id = get_node_id()


if node.parameters.qubit_pairs is None:
    raise ValueError("Please specify the qubit_pair name")

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

if node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

num_qubit_pairs = len(qubit_pairs)


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
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=2 * num_qubit_pairs)
    df = declare(int)  # QUA variable for the qubit frequency
    dc = declare(fixed)  # QUA variable for the flux dc level
    comp_flux_qubit = declare(float)
    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qp.qubit_control)
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(dc, dcs)):
                if "coupler_qubit_crosstalk" in qp.extras:
                    assign(comp_flux_qubit, qp.extras["coupler_qubit_crosstalk"] * dc)
                else:
                    assign(comp_flux_qubit, 0.0)
                # Flux sweeping for a qubit
                # Bring the qubit to the desired point during the saturation pulse
                qp.coupler.set_dc_offset(dc)
                qp.align()
                for role_index, qubit in enumerate(qubits_for_pair(qp)):
                    stream_index = 2 * i + role_index
                    rr = qubit.resonator
                    with for_(*from_array(df, dfs)):
                        # Update the resonator frequencies for resonator
                        update_frequency(rr.name, df + rr.intermediate_frequency)
                        # readout the resonator
                        rr.measure("readout", qua_vars=(I[stream_index], Q[stream_index]))
                        # wait for the resonator to relax
                        rr.wait(machine.depletion_time * u.ns)
                        # save data
                        save(I[stream_index], I_st[stream_index])
                        save(Q[stream_index], Q_st[stream_index])
        # Measure sequentially
        align(*[qubit.resonator.name for qubit in qubits_for_pair(qp)])

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            I_st[2 * i].buffer(len(dfs)).buffer(len(dcs)).average().save(f"I_control{i + 1}")
            Q_st[2 * i].buffer(len(dfs)).buffer(len(dcs)).average().save(f"Q_control{i + 1}")
            I_st[2 * i + 1].buffer(len(dfs)).buffer(len(dcs)).average().save(f"I_target{i + 1}")
            Q_st[2 * i + 1].buffer(len(dfs)).buffer(len(dcs)).average().save(f"Q_target{i + 1}")


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
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
        ds, qubit_pairs = match_loaded_qubit_pair_dataset(ds, qubit_pairs)
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"freq": dfs, "flux": dcs})
        ds = prepare_fetched_qubit_pair_dataset(ds, qubit_pairs, dfs, dcs)
       
    # Add the dataset to the node
    node.results = {"ds": ds}
    # %% {Analyzing}
    decouple_analysis = analyze_decouple_offsets(
        ds,
        qubit_pairs,
        fit_min_percentile=node.parameters.detrend_fit_min_percentile,
        prominence_fraction=node.parameters.row_span_prominence_fraction,
        distance=node.parameters.minima_distance,
    )
    print(format_decouple_offset_summary(decouple_analysis))
    node.results["decouple_offsets_V"] = decouple_analysis.decouple_offsets_V()

    # %% {Plotting}
    node.results["figure"] = plot_decouple_offset_maps(
        decouple_analysis,
        data_id=node.parameters.load_data_id,
        show=False,
    )

    # %% {Update_state}
    if node.parameters.update_coupler_decouple and node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qp in qubit_pairs:
                qp.coupler.decouple_offset = decouple_analysis.pairs[qp.name].decouple_offset_V

    # %% {Save_results}
    node.results["ds"] = ds
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()

# %%
