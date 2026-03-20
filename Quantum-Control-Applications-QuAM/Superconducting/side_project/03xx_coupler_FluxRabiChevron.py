# %%
"""
RABI CHEVRON flux version
Use the same pi-pulse condition with the flux varied for the coupler, optimize the flux offset this current driving frequency.  

Prerequisites:
    - x180_cp calibrated

Recommended node parameters
    - x180cp_dura_scaling = [3, 9]
    -flux_span_V: float = 0.025

Next steps before going to the next node:
    - Update coupler's decouple offset in the state.
    - Save the current state by calling machine.save("quam")

"""

# %% {Imports}
from quam_libs.lib.fit import peaks_dips
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state_coupler
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, List
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):

    coupler: str = 'coupler_q4_q5'
    num_averages: int = 500
    flux_span_V: float = 0.02 # 0.05
    flux_pts: int = 100
    x180cp_dura_scaling:List[int] = [3, 9] # [3, 9]
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(name="03xx_coupler_FluxRabiChevron", parameters=Parameters())


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
if not node.parameters.simulate:
    drive_LO_original = {drive_q[0].name: drive_q[0].xy.opx_output.upconverter_frequency}
    drive_q[0].xy.opx_output.upconverter_frequency = coupler[0].extras["RD"]["LO"]
    if "swap_direction" in coupler[0].extras["RD"]:
        detector_q[0].z.operations['aSWAP'].slope_direction = coupler[0].extras["RD"]["swap_direction"]




# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()

num_qubits = len(drive_q)

z_rising_buffer_time_ns = 200

# %% {QUA_program}

n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

span = node.parameters.flux_span_V
dcs = np.linspace(-span / 2, +span / 2, node.parameters.flux_pts, dtype=float)
current_x180cp_length_qua = {}
pi_dura_scal = node.parameters.x180cp_dura_scaling
for qubit in drive_q:
        # Check if the qubit has the required operations
        if hasattr(qubit.xy.operations, "x180_cp"):
            current_x180cp_length_qua[qubit.name] = int(qubit.xy.operations["x180_cp"].length)//4
        else:
            raise ValueError(f"x180_cp hadn't been calibrated for {qubit.name}! ")

with program() as EF_PR_Chevron:
    _, _, _, _, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    dc = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    dura_scal = declare(int)

    if not node.parameters.simulate:
        machine.apply_all_couplers_to_min()

    for i, qubit in enumerate(drive_q):
        # Bring the active qubits to the minimum frequency point
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
        
        qubit.xy.update_frequency(coupler[0].extras["RD"]["IF"])
        align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            with for_(*from_array(dc, dcs)):

                with for_each_(dura_scal, pi_dura_scal):

                    if not node.parameters.simulate:
                        if qubit.thermalization_time//5 > coupler[0].extras['T1']*1e9:
                            wait(qubit.thermalization_time * u.ns)
                        else:
                            wait(5*coupler[0].extras['T1']*1e9 * u.ns)

                    align()
                    coupler[0].coupler.play("const", amplitude_scale=dc/ coupler[0].coupler.operations['const'].amplitude, duration=current_x180cp_length_qua[qubit.name]*dura_scal+z_rising_buffer_time_ns//4)
                    qubit.xy.wait(z_rising_buffer_time_ns//4)
                    qubit.xy.play("x180_cp", amplitude_scale=1/dura_scal, duration=current_x180cp_length_qua[qubit.name]*dura_scal)
                    align()

                    # readout
                    readout_state_coupler(detector_q[i], state[i], method='aswap')
                    save(state[i], state_st[i])

                align()

            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(drive_q):
            state_st[i].buffer(len(pi_dura_scal)).buffer(len(dcs)).average().save(f"state{i + 1}")
            


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, EF_PR_Chevron, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(EF_PR_Chevron)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

    # %% {Data_fetching_and_dataset_creation}
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, coupler, {"dura_scales":pi_dura_scal, "flux": dcs})
    
    # Add the dataset to the node
    node.results = {"ds": ds}


    # %% {Data_analysis}
    result = peaks_dips(
        ds.state, dim="flux", prominence_factor=5, remove_baseline=False
    )
    
    
    fit_results = {}
    for q in coupler:
        fit_results[q.name] = {}
    
        if not np.isnan(result.sel(qubit=q.name).position.values.all()):
            fit_results[q.name]["fit_successful"] = True
            fit_results[q.name]["correct_additional_flux"] = np.average(result.sel(qubit=q.name).position.values)
        else:
            fit_results[q.name]["fit_successful"] = False
            print(f"Failed to find a peak for {q.name}\n")

    node.results["fit_results"] = fit_results

    # %% {Plot}
    grid_names, qubit_pair_names = grid_pair_names(coupler)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit in grid_iter(grid):
        ds.sel(qubit=qubit['qubit']).state.plot(
        x="flux", 
        hue="dura_scales",
        ax=ax
        )
        ax.axvline(
            np.average(result.sel(qubit=qubit["qubit"]).position.values),
            color="r",
            linestyle="--",
        )
        ax.grid()
        ax.set_xlabel("Additional Flux (V)")
        

    
    plt.suptitle("Flux Rabi Chevron")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig


    # %% {Update_state}
    for q in coupler:
        with node.record_state_updates():
            if fit_results[q.name]["fit_successful"]:
                q.coupler.decouple_offset += fit_results[q.name]["correct_additional_flux"]

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
