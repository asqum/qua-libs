"""
        IQ BLOBS
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit
in the |g> state) and then after applying a pi pulse to the qubit (bringing the qubit to the |e> state) successively.
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the separation between |g> and |e> states
      aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit.wait(qubit.thermalization_time * u.ns) spectroscopy, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the state.
    - Update the g -> e thresholds (threshold & rus_threshold) in the state.
    - Update the confusion matrices in the state.
    - Save the current state
"""


# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import List, Literal, Optional
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.experiments.iq_blobs.fetch_dataset import fetch_dataset
from quam_libs.experiments.iq_blobs.parameters import Parameters
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.macros import qua_declaration, active_reset
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.analysis.discriminator import two_state_discriminator
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from time import time 
from scipy.stats import norm
from quam_libs.lib.qubit_thermometer import repetition_data, StateDiscrimination, prepare_dataset_for_qcat



# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None #The qubit to be measured. If None, all active qubits will be measured
    num_runs: int = 2048*2
    flux_point_joint_or_independent_or_arbitrary: Literal['joint', 'independent'] = 'independent'   
    simulate: bool = False
    timeout: int = 100
    use_state_discrimination: bool = True
    reset_type: Literal['thermal'] = "thermal"
    load_data_id: Optional[int] = None
    multiplexed: bool = 1
    
    

node = QualibrationNode(
    name="07st_QubitThermometer_histogram",
    parameters=Parameters()
)


# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)

machine = QuAM.load()
node.machine = machine

if node.parameters.load_data_id is None:
    qmm = machine.connect()

qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)

config = machine.generate_config()

# %% {QUA_program}
n_runs = node.parameters.num_runs
flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary
reset_type = node.parameters.reset_type
operation_name = 'readout'

with program() as iq_blobs:
    reset_global_phase()
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)

    for multiplexed_qubits in qubits.batch():
        machine.set_all_fluxes(flux_point=flux_point, target=list(multiplexed_qubits.values())[0])

        with for_(n, 0, n < n_runs, n + 1):
            save(n, n_st)

            if not node.parameters.simulate:
                # measure ground-state IQ blob for all qubits
                for i, qubit in multiplexed_qubits.items():
                    if reset_type == "active":
                        active_reset(qubit)
                        # active_reset_gef(qubit)
                    elif reset_type == "thermal":
                        qubit.wait(2 * qubit.thermalization_time * u.ns)
                    else:
                        raise ValueError(f"Unrecognized reset type {reset_type}.")

            align(*[q.xy.name for q in multiplexed_qubits.values()] +
                   [q.resonator.name for q in multiplexed_qubits.values()] +
                   [q.z.name for q in multiplexed_qubits.values()])

            for i, qubit in multiplexed_qubits.items():
                qubit.resonator.measure(operation_name, qua_vars=(I_g[i], Q_g[i]))
                qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])

            if not node.parameters.simulate:
                # measure excited-state IQ blob for all qubits
                align(*[q.xy.name for q in multiplexed_qubits.values()] +
                       [q.resonator.name for q in multiplexed_qubits.values()] +
                       [q.z.name for q in multiplexed_qubits.values()])

                for i, qubit in multiplexed_qubits.items():
                        if reset_type == "active":
                            active_reset(qubit)
                        elif reset_type == "thermal":
                            qubit.wait(2*qubit.thermalization_time * u.ns)
                        else:
                            raise ValueError(f"Unrecognized reset type {reset_type}.")

            align(*[q.xy.name for q in multiplexed_qubits.values()] +
                   [q.resonator.name for q in multiplexed_qubits.values()] +
                   [q.z.name for q in multiplexed_qubits.values()])

            for i, qubit in multiplexed_qubits.items():
                qubit.xy.play("x180")
                qubit.resonator.wait(qubit.xy.operations["x180"].length * u.ns) # qubit.align()
                qubit.resonator.measure(operation_name, qua_vars=(I_e[i], Q_e[i]))
                qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                save(I_e[i], I_e_st[i])
                save(Q_e[i], Q_e_st[i])
            if node.parameters.multiplexed:
                align(*[q.xy.name for q in multiplexed_qubits.values()] +
                [q.resonator.name for q in multiplexed_qubits.values()] +
                [q.z.name for q in multiplexed_qubits.values()])
            else:
                align()
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].save_all(f"I_g{i + 1}")
            Q_g_st[i].save_all(f"Q_g{i + 1}")
            I_e_st[i].save_all(f"I_e{i + 1}")
            Q_e_st[i].save_all(f"Q_e{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    samples, fig = simulate_and_plot(qmm, config, iq_blobs, node.parameters)
    node.results = {"figure": fig}
    node.save()
else:
    if node.parameters.load_data_id is None:

        dss = []
        start = time()
        
        target_counts = 1
        current_success = 0
        max_retries = target_counts + 5  # 設定一個總嘗試上限，避免無限迴圈
        attempts = 0

        while current_success < target_counts and attempts < max_retries:
            attempts += 1
            try: # prevent getting a unpredictable error like connection failed or qmm closed.
                with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                    job = qm.execute(iq_blobs)
                    for i in range(num_qubits):
                        results = fetching_tool(job, ["n"], mode="live")
                        while results.is_processing():
                            n = results.fetch_all()[0]
                            if target_counts <= 5:
                                progress_counter(n, n_runs, start_time=results.start_time)
                ds = fetch_dataset(job, qubits, node.parameters)
                dss.append(ds)
                current_success += 1
                print(f"Counts: {current_success} (Total attempts: {attempts})")
            except Exception as e:
                print(f"Attempt {attempts} failed: {e}. Skipping...")
                if (attempts - current_success) > 5:
                    print("Too many consecutive failures. Stopping experiment.")
                    break
             
        end = time()
        print(f"Total {round(end-start,1)} sec for {current_success} counts")
        ds = xr.concat(dss, dim='iteration')
        node.results = {"ds": ds, "figs": {}, "results": {}}
        reload_qbs = False
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"] 
        machine = node.machine
        reload_qbs = True


# %% {analysis}
if not node.parameters.simulate:
    models = {q.name:[] for q in qubits}
    trained_params = {}
    RO_fidelity = {q.name:[] for q in qubits}

    for iteration in range(ds.dims['iteration']):
        dss = prepare_dataset_for_qcat(ds.isel(iteration=iteration))
        
        sep_data = repetition_data(dss, repetition_dim="qubit")

    
        for sq_data in sep_data:
            qubit_name = sq_data["qubit"].values.item()

            analysis = StateDiscrimination(sq_data)
            analysis._start_analysis()
            models[qubit_name].append(analysis)
            trained_params[qubit_name] = analysis.analysis_result['trained_paras']    # save trained parameters for each qubit
            (p00, p01), (p10, p11) = analysis.analysis_result['gaussian_norms']
            RO_fidelity[qubit_name].append(1 - 0.5*(p01+p10))
    
    for q in qubits:
        node.results['results'][q.name] = {}
        
        node.results['results'][q.name]["RO_fidelity"] = np.average(RO_fidelity[q.name])
        node.results['results'][q.name]["GMM_mean"] = trained_params[q.name]['mean'].tolist()
        node.results['results'][q.name]["GMM_std"] = trained_params[q.name]['std']

    #%% {Plot}
    mu_collection, sig_collection = {}, {}
    if reload_qbs:
        qubits = [machine.qubits[q] for q in list(models.keys())]
   
    ## prepare ground 1D histogram
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        Teff = models[qubit['qubit']][0].show_thermal_analysis(qubit['qubit'], machine.qubits[qubit['qubit']].xy.RF_frequency*1e-9, ax)
        node.results["results"][qubit['qubit']]["Teff_mK"] = Teff
        mu_collection[qubit['qubit']] = Teff
    plt.suptitle("Ground State Preparation")
    plt.tight_layout()
    plt.show()
    node.results["Pg_1DHistogran"] = grid.fig
    ## raw data plot
    grid_2 = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid_2): 
        models[qubit['qubit']][0].show_scatter_analysis(qubit['qubit'], ax)
    plt.suptitle("IQ Blobs")
    plt.tight_layout()
    plt.show()
    node.results["Colored_Scatter"] = grid_2.fig
    ## outliers
    grid_3 = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid_3): 
        models[qubit['qubit']][0].show_outliers_analysis(qubit['qubit'], ax)
    plt.suptitle("Outliers Detection")
    plt.tight_layout()
    plt.show()
    node.results["Outliers_Detection"] = grid_3.fig
    

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qubit in qubits:
                qubit.extras['GMM_mean'] = node.results['results'][qubit.name]["GMM_mean"]
                qubit.extras['GMM_std'] = node.results['results'][qubit.name]["GMM_std"]

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.save()          


# %%