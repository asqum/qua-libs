"""
        QUBIT THERMOMETER
This sequence measures the resonator 'N' times after thermalization (qubit in |g>), without driving the qubit.
The IQ blob is fitted with the 2D Gaussian mixture model calibrated by 07st_ReadoutGMMCalibration (centers and std
fixed, only the weights are fitted), giving the excited-state population P_e, which is converted to an effective
qubit temperature Teff through the Boltzmann distribution at the qubit frequency.
With histo_num > 1 the measurement is repeated and the Teff distribution is fitted with a normal distribution.

Note: P_e also contains readout-induced excitation, so Teff is an upper bound of the qubit temperature.

Prerequisites:
    - Having run 07st_ReadoutGMMCalibration with the current readout settings (qubit.extras GMM_mean/GMM_std).
    - Set the desired flux bias

State update (only when histo_num >= 100):
    - qubit.extras['Teff_mK'] and qubit.extras['Teff_mK_dev'].
"""


# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import List, Literal, Optional
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.experiments.iq_blobs.fetch_dataset import fetch_dataset_ground_only
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.macros import qua_declaration
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import restore_load_data_id, resolve_qubits_from_node
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm.qua import *
from time import time 
from scipy.stats import norm
from quam_libs.lib.qubit_thermometer import repetition_data, StateDiscrimination, prepare_ground_state_for_qcat, PetoT



# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ['q2'] #The qubit to be measured. If None, all active qubits will be measured
    num_runs: int = 2048*2
    flux_point_joint_or_independent_or_arbitrary: Literal['joint', 'independent'] = 'joint'   
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = 1
    histo_num:int = 1
    

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



for q in qubits:
    if q.extras.get("GMM_mean") is None or q.extras.get("GMM_std") is None:
        raise ValueError(f"Qubit {q.name} is missing GMM parameters. Please run the 'ReadoutGMMCalibration' node first.")

# Set driving to minimum power only for generating the config, then restore the machine immediately
# so that the modified values can never be saved into the state.
original_params = {}
try:
    for q in qubits:
        original_params[q.name] = {
            "XY-FSP": q.xy.opx_output.full_scale_power_dbm,
            "x180_amp": q.xy.operations["x180"].amplitude,
            "x90_amp": q.xy.operations["x90"].amplitude,
        }
        q.xy.opx_output.full_scale_power_dbm = -11
        q.xy.operations["x180"].amplitude = 0.0
        q.xy.operations["x90"].amplitude = 0.0

    config = machine.generate_config()
finally:
    for q in qubits:
        if q.name in original_params:
            q.xy.opx_output.full_scale_power_dbm = original_params[q.name]['XY-FSP']
            q.xy.operations["x180"].amplitude = original_params[q.name]['x180_amp']
            q.xy.operations["x90"].amplitude = original_params[q.name]['x90_amp']

# %% {QUA_program}
n_runs = node.parameters.num_runs
flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary
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
                    qubit.wait(2 * qubit.thermalization_time * u.ns)
                    

            align(*[q.xy.name for q in multiplexed_qubits.values()] +
                   [q.resonator.name for q in multiplexed_qubits.values()] +
                   [q.z.name for q in multiplexed_qubits.values()])

            for i, qubit in multiplexed_qubits.items():
                qubit.resonator.measure(operation_name, qua_vars=(I_g[i], Q_g[i]))
                qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])

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


# %% {Simulate_or_execute}
if node.parameters.simulate:
    samples, fig = simulate_and_plot(qmm, config, iq_blobs, node.parameters)
    node.results = {"figure": fig}
    node.save()
else:
    if node.parameters.load_data_id is None:

        dss = []
        start = time()
        
        target_counts = node.parameters.histo_num
        current_success = 0
        max_retries = target_counts + 10  # 設定一個總嘗試上限，避免無限迴圈
        attempts = 0

        while current_success < target_counts and attempts < max_retries:
            attempts += 1
            try: # prevent getting a unpredictable error like connection failed or qmm closed.
                with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                    job = qm.execute(iq_blobs)
                    results = fetching_tool(job, ["n"], mode="live")
                    while results.is_processing():
                        n = results.fetch_all()[0]
                        if target_counts <= 5:
                            progress_counter(n, n_runs, start_time=results.start_time)
                ds = fetch_dataset_ground_only(job, qubits, node.parameters)
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
        if not dss:
            raise RuntimeError(f"All {attempts} attempts failed, no data was acquired.")
        ds = xr.concat(dss, dim='iteration')
        node.results = {"ds": ds, "figs": {}, "results": {}}
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        # load_from_id restores the original run's parameters (load_data_id=None); keep it so the state is not updated
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubits = resolve_qubits_from_node(machine, node)
        node.results["results"] = {}


# %% {analysis}
if not node.parameters.simulate:
    models = {q.name:[] for q in qubits}
    thermal_population = {q.name:[] for q in qubits}
    Teff = {q.name:[] for q in qubits}

    for iteration in range(ds.sizes['iteration']):
        dss = prepare_ground_state_for_qcat(ds.isel(iteration=iteration))
        
        sep_data = repetition_data(dss, repetition_dim="qubit")

    
        for sq_data in sep_data:
            qubit_name = sq_data["qubit"].values.item()
            gmm_mean, gmm_std = np.array(machine.qubits[qubit_name].extras["GMM_mean"]), machine.qubits[qubit_name].extras["GMM_std"]
            # Rename n_runs to shot_idx if present
            # sq_data = sq_data.rename({'N': 'shot_idx'})
            # print(sq_data)
            analysis = StateDiscrimination(sq_data, user_mean=gmm_mean, user_std=gmm_std)
            analysis._start_analysis()
            models[qubit_name].append(analysis)
           
            (p00, p01), (p10_dummy, p11_dummy) = analysis.analysis_result['gaussian_norms']
            thermal_population[qubit_name].append(p01)  
            Teff[qubit_name].append(PetoT(p01, 2*np.pi*machine.qubits[qubit_name].xy.RF_frequency))
            
            # figs_dict = analysis._plot_results(qubit_name, None)
    
    # Teff_mK is always the mean of the per-iteration Teff (PetoT is nonlinear, so it differs from PetoT(mean P_e))
    for q in qubits:
        node.results['results'][q.name] = {}
        node.results['results'][q.name]["thermal_population"] = float(np.average(thermal_population[q.name]))
        node.results['results'][q.name]["Teff_mK"] = float(np.average(Teff[q.name]))
        

    #%% {Plot}
    mu_collection, sig_collection = {}, {}
    if node.parameters.histo_num == 1:
        ## prepare ground 1D histogram
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            models[qubit['qubit']][0].show_thermal_analysis(qubit['qubit'], machine.qubits[qubit['qubit']].xy.RF_frequency*1e-9, ax)
        plt.suptitle("Ground State Preparation")
        plt.tight_layout()
        plt.show()
        node.results["Pg_1DHistogram"] = grid.fig
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
    else:
        n_histo = ds.sizes['iteration']
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            data = np.array(Teff[qubit['qubit']])
            data = data[np.isfinite(data)]
            counts, bins, _ = ax.hist(data, bins=15, alpha=0.7, color='skyblue', edgecolor='white', label='Counts')
            ### Normal distribution
            mu, sigma = norm.fit(data)
            bin_width = bins[1] - bins[0]
            scaling_factor = len(data) * bin_width
            x = np.linspace(min(data), max(data), 100)
            p = norm.pdf(x, mu, sigma) * scaling_factor  
            mu_collection[qubit['qubit']], sig_collection[qubit['qubit']] = mu, sigma
            node.results["results"][qubit['qubit']]["Teff_mK"] = mu
            node.results["results"][qubit['qubit']]["Teff_mK_dev"] = sigma
            ### Plot
            ax.plot(x, p, 'r-', lw=2, label='Normal Fit')
            ax.set_title(f"{qubit['qubit']}")
            ax.set_xlabel("Teff (mK)")
            ax.set_ylabel("Counts")
            ax.grid(axis='y', alpha=0.3)
            
            stats_text = (
                rf"$T = {mu:.1f} \pm {sigma:.2f}$ mK"
            )
            ax.text(
                0.05, 0.95, stats_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
            )
        plt.suptitle(f"Qubit Thermometer Statistics, #={n_histo}", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        node.results["Temperature_statistics"] = grid.fig
        from numpy import random
        idx = random.randint(n_histo)
        ## prepare ground 1D histogram of one randomly picked iteration
        grid_3 = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid_3):
            models[qubit['qubit']][idx].show_thermal_analysis(qubit['qubit'], machine.qubits[qubit['qubit']].xy.RF_frequency*1e-9, ax)
            
        plt.suptitle(f"Ground State Preparation (iteration {idx})")
        plt.tight_layout()
        plt.show()
        node.results[f"Pg_1DHistogram_idx_{idx}"] = grid_3.fig
        ## raw data plot
        grid_2 = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid_2): 
            models[qubit['qubit']][idx].show_scatter_analysis(qubit['qubit'], ax)
        plt.suptitle("IQ Blobs")
        plt.tight_layout()
        plt.show()
        node.results[f"Colored_Scatter_idx_{idx}"] = grid_2.fig

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qubit in qubits:
                if node.parameters.histo_num >= 100:
                    qubit.extras['Teff_mK'] = mu_collection[qubit.name]
                    qubit.extras['Teff_mK_dev'] = sig_collection[qubit.name]

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.save()          


# %%