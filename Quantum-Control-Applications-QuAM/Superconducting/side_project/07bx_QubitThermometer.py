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

from qualibrate import QualibrationNode
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
node = QualibrationNode(
    name="07bx_QubitThermometer",
    parameters=Parameters(
        qubits=None,
        multiplexed=0,
        flux_point_joint_or_independent="independent",
        num_runs=4096*1,
        load_data_id=None,
        simulate=False,
        simulation_duration_ns=1000,
        use_waveform_report=False,
    )
)
# statistics number
histo_num:int = 100

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)

machine = QuAM.load()
# machine.network["port"] = int(access_port)

# print(f"Machine access port :{access_port}")
if node.parameters.load_data_id is None:
    qmm = machine.connect()

qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)

config = machine.generate_config()

# %% {QUA_program}
n_runs = node.parameters.num_runs
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
operation_name = node.parameters.operation_name

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
    node.machine = machine
    node.save()
else:
    if node.parameters.load_data_id is None:

        dss = []
        start = time()
        
        target_counts = histo_num
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
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]


# %% {analysis}
if not node.parameters.simulate:
    models = {q.name:[] for q in qubits}
    RO_fidelity = {q.name:[] for q in qubits}

    for iteration in range(ds.dims['iteration']):
        dss = prepare_dataset_for_qcat(ds.isel(iteration=iteration))
        
        sep_data = repetition_data(dss, repetition_dim="qubit")

    
        for sq_data in sep_data:
            qubit_name = sq_data["qubit"].values.item()

            # Rename n_runs to shot_idx if present
            # sq_data = sq_data.rename({'N': 'shot_idx'})
            # print(sq_data)
            analysis = StateDiscrimination(sq_data)
            analysis._start_analysis()
            models[qubit_name].append(analysis)
           
            (p00, p01), (p10, p11) = analysis.analysis_result['gaussian_norms']
            RO_fidelity[qubit_name].append(1 - 0.5*(p01+p10))
            
            # figs_dict = analysis._plot_results(qubit_name, None)
    
    for q in qubits:
        node.results['results'][q.name] = {}
        node.results['results'][q.name]["RO_fidelity"] = np.average(RO_fidelity[q.name])
        
        

    #%% {Plot}
    mu_collection, sig_collection = {}, {}
    if histo_num == 1:
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
    else:
        from quam_libs.lib.qubit_thermometer import PetoT
        Teff = {q.name:[] for q in qubits}
        for q in qubits:
            for iter in range(len(models[q.name])):
                (p00, p01), (p10, p11) = models[q.name][iter].analysis_result['gaussian_norms']
                Teff[q.name].append(PetoT(p01, 2*np.pi*q.xy.RF_frequency))
        tot_c = 0
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        
        for ax, qubit in grid_iter(grid):
            data = Teff[qubit['qubit']]
            tot_c = len(data)
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
                f"$T = {mu:.1f} \pm {sigma:.2f}$ mK\n"
            )
            ax.text(
                0.05, 0.95, stats_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
            )
        plt.suptitle(f"Qubit Thermometer Statistics, #={tot_c}", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        node.results["Temperature_statistics"] = grid.fig

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qubit in qubits:
                qubit.extras['Teff_mK'] = mu_collection[qubit.name]
                if histo_num != 1:
                    qubit.extras['Teff_mK_dev'] = sig_collection[qubit.name]

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()          


# %%