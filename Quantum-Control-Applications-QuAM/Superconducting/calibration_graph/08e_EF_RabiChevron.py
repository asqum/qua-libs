# %%
"""
RABI CHEVRON
Prepare qubit at E-state and applied EF-x180 pulse with different duration to calibrate a preciser EF-frequency (anharmonicity). 

Prerequisites:
    - EF_x180 calibrated
    - gef IQ blobs calibrated

Next steps before going to the next node:
    - Update the qubit anharmonicity in the state.
    - Save the current state by calling machine.save("quam")
"""


# %% {Imports}
import dataclasses
from quam_libs.lib.fit import peaks_dips
from qualibrate import QualibrationNode, NodeParameters
from quam.components import pulses
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset_gef, readout_state_gef
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
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

    qubits: Optional[List[str]] = None
    num_averages: int = 600
    frequency_span_in_mhz: float = 5
    frequency_step_in_mhz: float = 0.05
    EF_x180_dura_scaling:List[int] = [200, 500]
    reset_type: Literal["active", "thermal"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(name="08e_EF_RabiChevron", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

for q in qubits:
    # Check if an optimized GEF frequency exists
    if not hasattr(q, "GEF_frequency_shift"):
        q.GEF_frequency_shift = 0


# %% {QUA_program}
operation = 'EF_x180'  # The qubit operation to play
n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, +span // 2, step, dtype=np.int32)

current_ef_x180_length_qua = {}
pi_dura_scal = node.parameters.EF_x180_dura_scaling
for qubit in qubits:
        # Check if the qubit has the required operations
        if hasattr(qubit.xy.operations, "EF_x180"):
            current_ef_x180_length_qua[qubit.name] = int(qubit.xy.operations["EF_x180"].length)//4
        else:
            raise ValueError(f"EF_x180 hadn't been calibrated for {qubit.name}! ")

with program() as EF_PR_Chevron:
    _, _, _, _, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    df = declare(int)  # QUA variable for the qubit drive amplitude pre-factor
    dura_scal = declare(int)

    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            machine.apply_all_couplers_to_min()
            qubit.z.to_independent_idle()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()

        # Wait for the flux bias to settle
        for qb in qubits:
            qb.z.settle()

        align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            with for_each_(dura_scal, pi_dura_scal):

                with for_(*from_array(df, dfs)):


                    if node.parameters.reset_type == 'thermal':
                        wait(qubit.thermalization_time * u.ns)
                    else:
                        active_reset_gef(qubit)

                    wait(4)
                    # Reset the qubit frequency
                    update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
                    # Drive the qubit to the excited state
                    wait(4)
                    qubit.xy.play("x180")
                    # Update the qubit frequency to scan around the expected f_12
                    qubit.align()
                    update_frequency(
                        qubit.xy.name, qubit.xy.intermediate_frequency - qubit.anharmonicity + df
                    )
                    wait(4)
                    qubit.xy.play("EF_x180", amplitude_scale=1/dura_scal, duration=current_ef_x180_length_qua[qubit.name]*dura_scal)
                    wait(4)
                    # Reset the qubit frequency, play the rest to ground
                    update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
                    wait(4)
                    qubit.xy.play("x180")
                    align()

                    # readout
                    readout_state_gef(qubit, state[i])
                    save(state[i], state_st[i])

                align()

            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            state_st[i].buffer(len(dfs)).buffer(len(pi_dura_scal)).average().save(f"state{i + 1}")
            


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
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs, "dura_scales":pi_dura_scal})
    
    ds = ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "freq"],
                np.array([dfs + q.xy.RF_frequency - q.anharmonicity for q in qubits]),
            )
        }
    )
    ds.freq_full.attrs["long_name"] = "Frequency"
    ds.freq_full.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}


    # %% {Data_analysis}
    result = peaks_dips(
        ds.state, dim="freq", prominence_factor=5, remove_baseline=False
    )
    
    
    fit_results = {}
    for q in qubits:
        fit_results[q.name] = {}
    
        if not np.isnan(result.sel(qubit=q.name).position.values.all()):
            fit_results[q.name]["fit_successful"] = True
            fit_results[q.name]["optimized_anharmonicity"] = q.anharmonicity - np.average(result.sel(qubit=q.name).position.values)
        else:
            fit_results[q.name]["fit_successful"] = False
            print(f"Failed to find a peak for {q.name}\n")

    node.results["fit_results"] = fit_results


    # %% {Plot}
    grid_names = [q.grid_location for q in qubits]
    grid = QubitGrid(ds, grid_names)
    for ax, qubit in grid_iter(grid):
        ds.sel(qubit=qubit['qubit']).state.plot(
        x="freq", 
        hue="dura_scales",
        ax=ax
        )
        ax.axvline(
            np.average(result.sel(qubit=qubit["qubit"]).position.values),
            color="r",
            linestyle="--",
        )
        ax.grid()

    
    plt.suptitle("EF states Rabi Chevron")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig


    # %% {Update_state}
    for q in qubits:
        with node.record_state_updates():
            if fit_results[q.name]["fit_successful"]:
                q.xy.operations['EF_x180'].anharmonicity = fit_results[q.name]["optimized_anharmonicity"]
                q.anharmonicity = fit_results[q.name]["optimized_anharmonicity"]

    # %% {Save_results}
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%
