# %%
"""
Optimization for the aSWAP pulse duration. It will try different pluse length for aSWAP and applied a pi-pulse for each length. The best pulse duration is the one that gives the highest state probability.

Prerequisites:
    - Coupler's π-pulse calibrated.

"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state_coupler
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):

    coupler: str = 'coupler_q4_q5'
    num_averages: int = 500
    min_length_ns:int = 100
    max_length_ns:int = 1000
    length_step_ns:int = 8
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    simulation_duration_ns: int = 500
    timeout: int = 100
    load_data_id: Optional[int] = None
    

node = QualibrationNode(name="07x_aSWAP_DuraOptimize", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
coupler = [machine.qubit_pairs[node.parameters.coupler]] # currently supports 1 coupler a time only.
drive_q = [machine.qubits[coupler[0].extras["RD"]["driven_q"]]]
detector_q = [machine.qubits[coupler[0].extras["RD"]["readout_q"]]]

# Change driving LO
if node.parameters.load_data_id is None and not node.parameters.simulate:
    drive_LO_original = {drive_q[0].name: drive_q[0].xy.opx_output.upconverter_frequency}
    drive_q[0].xy.opx_output.upconverter_frequency = coupler[0].extras["RD"]["LO"]
    if "swap_direction" in coupler[0].extras["RD"]:
        detector_q[0].z.operations['aSWAP'].slope_direction = coupler[0].extras["RD"]["swap_direction"]

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = 'thermal' #node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
state_discrimination = True

# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)


duras = np.arange(
    4*(node.parameters.min_length_ns//4),
    4*(node.parameters.max_length_ns//4),
    4*(node.parameters.length_step_ns//4),
)//4




with program() as power_rabi:
    _, _, _, _, n, n_st = qua_declaration(num_qubits=len(detector_q))
    
    state = [declare(int) for _ in range(len(detector_q))]
    state_stream = [declare_stream() for _ in range(len(detector_q))]
    du = declare(int)

    if not node.parameters.simulate:
        machine.apply_all_couplers_to_min()
    for i, qubit in enumerate(drive_q):
        # Bring the active qubits to the minimum frequency point
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
        qubit.xy.update_frequency(coupler[0].extras["RD"]["IF"])
        qubit.align()


        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(du, duras.astype(int))):
                # Initialize the qubits
            
                if not node.parameters.simulate:
                    if qubit.thermalization_time//5 > coupler[0].extras['T1']*1e9:
                        wait(qubit.thermalization_time * u.ns)
                    else:
                        wait(5*coupler[0].extras['T1']*1e9 * u.ns)


                align()
                qubit.xy.play('x180_cp')

                readout_state_coupler(detector_q[i], state[i], method='aswap', assign_aswap_duration=du)
                save(state[i], state_stream[i])


    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(drive_q):
                state_stream[i].buffer(len(duras)).average().save(f"state{i + 1}")
            

# %% {Simulate_or_execute}
if not node.parameters.load_data_id:
    
    # Generate the OPX and Octave configurations
    config = machine.generate_config()
    
    # Open Communication with the QOP
    if node.parameters.load_data_id is None:
        qmm = machine.connect()
    if node.parameters.simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)  # In clock cycles = 4ns
        job = qmm.simulate(config, power_rabi, simulation_config)
        # Get the simulated samples and plot them for all controllers
        samples = job.get_simulated_samples()
        samples.con1.plot()
        node.results = {"figure": plt.gcf()}
        wf_report = job.get_simulated_waveform_report()
        wf_report.create_plot(samples, plot=True, save_path=None)
        node.save()
    
    else:

        if node.parameters.load_data_id is None:
            with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                job = qm.execute(power_rabi)
                results = fetching_tool(job, ["n"], mode="live")
                while results.is_processing():
                    # Fetch results
                    n = results.fetch_all()[0]
                    # Progress bar
                    progress_counter(n, n_avg, start_time=results.start_time)

        
    if not node.parameters.simulate:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, drive_q, {"duration": duras*4})
        
else:
    ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
    # Add the dataset to the node
    # 
# %%
if not node.parameters.simulate and node.parameters.load_data_id is None:
    data = ds.state
    grid_names = [q.grid_location for q in drive_q]
    grid = QubitGrid(ds, grid_names)
    for ax, qubit in grid_iter(grid):
        ds.sel(qubit = qubit['qubit']).state.plot(ax = ax, marker='o', linestyle='')
        ax.set_xlabel("Duration (ns)")
        ax.set_title(f"{coupler[0].name} aSWAP duration")
        ax.set_ylabel("|1> population")
        ax.grid()
        
    plt.tight_layout()
    plt.show()
    node.results['figure'] = grid.fig

# %% {Save_results}
if node.parameters.load_data_id is None and not node.parameters.simulate:
    for q in drive_q:
        q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
    for q in detector_q:
        q.z.operations['aSWAP'].slope_direction = -1 # always at -1
    
    node.outcomes = {q.name: "successful" for q in drive_q}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%

