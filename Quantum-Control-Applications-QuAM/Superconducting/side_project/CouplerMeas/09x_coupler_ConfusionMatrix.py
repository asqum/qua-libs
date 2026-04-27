# %%
"""
CONFUSION MATRIX

Prerequisites:
    - the π-pulse calibrated for the target coupler.

Updates:
    - Figures only.

"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state_coupler, active_reset_coupler
from quam_libs.lib.instrument_limits import instrument_limits
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_oscillation, oscillation
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

    coupler: str = 'coupler_q4_q5'
    shots: int = 2048*5
    reset_type: Literal["thermal","active"] = 'thermal'
    simulate: bool = False
    simulation_duration_ns: int = 8000
    timeout: int = 100
    load_data_id: Optional[int] = None

node = QualibrationNode(name="09x_coupler_ConfusionMatrix", parameters=Parameters())


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
if not node.parameters.simulate and node.parameters.load_data_id is None:
    aswap_dir_update_is_q = True
    drive_LO_original = {drive_q[0].name: drive_q[0].xy.opx_output.upconverter_frequency}
    drive_q[0].xy.opx_output.upconverter_frequency = coupler[0].extras["RD"]["LO"]
    if "swap_direction" in coupler[0].extras["RD"]:
        detector_q[0].z.operations['aSWAP'].slope_direction = coupler[0].extras["RD"]["swap_direction"]
    if 'strategy' not in coupler[0].extras["RD"]:
        readout_strategy = 'aswap'
    else:
        readout_strategy = coupler[0].extras["RD"]["strategy"]
    if coupler[0].extras["RD"]["aswap_supplier"].lower() == 'c':
        print("*** aSWAP is applied on coupler itself !")
        if not hasattr(coupler[0].coupler.operations, "aSWAP"):
            raise  LookupError(f"aSWAP operation now is not in {coupler[0].name}.coupler.operation, please add it to unlock the ability for coupler's measurement!")
        aswaper = coupler[0]
        coupler[0].coupler.operations['aSWAP'].slope_direction = coupler[0].extras["RD"]["swap_direction"]
        aswap_dir_update_is_q = False
    else:
        aswaper = None

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()



# %% {QUA_program}
n_avg = node.parameters.shots  # The number of averages

flux_point =  'independent' 
prepare_state = np.array([0, 1])
state_discrimination = True

operation_exact_name = f"x180_{coupler[0].name}"

with program() as power_rabi:
    _, _, _, _, n, n_st = qua_declaration(num_qubits=len(detector_q))
    
    state = [declare(int) for _ in range(len(detector_q))]
    state_stream = [declare_stream() for _ in range(len(detector_q))]
    dummy_state = [declare(int) for _ in range(len(detector_q))]
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    instr_idx = declare(int)  # QUA variable for the number of qubit pulses
    

    if not node.parameters.simulate:
        machine.apply_all_couplers_to_min()
    for i, qubit in enumerate(drive_q):
        # Bring the active qubits to the minimum frequency point
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
        
        # update LO
        qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            for prepared in prepare_state:
                
                # Initialize the qubits
                qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                if node.parameters.reset_type == 'thermal':
                    if not node.parameters.simulate:
                        wait(10*coupler[0].extras['T1']*1e9 * u.ns)
                        active_reset(detector_q[i])
                        active_reset(qubit) 
                else:
                    active_reset(detector_q[i])
                    active_reset(qubit)
                    readout_state_coupler(detector_q[i], dummy_state[i], method=readout_strategy)
                    align()
                    active_reset(detector_q[i])
                    active_reset(qubit)

                qubit.xy.update_frequency(coupler[0].extras["RD"]["IF"])
                align()

                # Play coupler's pi pulse
                if prepared:
                    qubit.xy.play(operation_exact_name)
                align()
                
                    
                align()
                # wait(250)
                readout_state_coupler(detector_q[i], state[i], flux_applied_target=aswaper, method=readout_strategy)
                save(state[i], state_stream[i])


    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(drive_q): 
            state_stream[i].buffer(len(prepare_state)).buffer(n_avg).save(f"state{i + 1}")
            


# %% {Simulate_or_execute}
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

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(power_rabi)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, drive_q, {"state_prepared": prepare_state, "N":np.linspace(1, n_avg, n_avg)})
    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
    # Add the dataset to the node
    node.results = {"ds": ds}


    # %% {Plotting}
    import seaborn as sns
    grid = QubitGrid(ds, [q.grid_location for q in drive_q])
    for ax, qubit in grid_iter(grid):
        sub_ds = ds.state.sel(qubit=qubit['qubit'])

        # population about excited
        prob_1 = sub_ds.astype(float).mean(dim='N')

        p01 = prob_1.sel(state_prepared=0).values # 0 prepared, 1 measured 
        p11 = prob_1.sel(state_prepared=1).values 
        
        matrix = [
            [1 - p01, p01],  # Prepared 0
            [1 - p11, p11]   # Prepared 1
        ]
        ro_fidelity = 0.5*(1-p01+p11)
        sns.heatmap(
            matrix, 
            annot=True, 
            fmt=".3f", 
            cmap="Blues",        # 你也可以換成 "RdBu_r" 或 "YlGnBu"
            vmin=0,              # 強制設定最小值為 0
            vmax=1,              # 強制設定最大值為 1
            xticklabels=['0', '1'], 
            yticklabels=['0', '1'],
            ax=ax,
            cbar=True,           # 暫時顯示 cbar 檢查對比範圍
            linewidths=1,        # 增加格子間的白線，讓邊界更清晰
            linecolor='black',
            annot_kws={"weight": "bold", "size": 12}
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
        ax.set_title(f"{coupler[0].name}\n RO_fidelity={ro_fidelity:.1%}")
        ax.set_xlabel("Measured State")
        ax.set_ylabel("Prepared State")
    
    grid.fig.suptitle("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    if not node.parameters.simulate:
        if node.parameters.load_data_id is None:
            for q in drive_q:
                q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
            if aswap_dir_update_is_q:
                for q in detector_q:
                    q.z.operations['aSWAP'].slope_direction = -1
            else:
                for c in coupler:
                    c.coupler.operations['aSWAP'].slope_direction = -1
        node.outcomes = {q.name: "successful" for q in drive_q}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()

# %%
