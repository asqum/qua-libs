# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.components.transmon_pair import Transmon, TransmonPair
from quam_libs.lib import find_c_with_q
from quam_libs.macros import qua_declaration, active_reset, active_reset_simple
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.analysis import two_state_discriminator
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List, Dict
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib.animation as animation
from sklearn.mixture import GaussianMixture


# %%
def IQ_observer(qubits:List[Transmon], qubit_pairs:List[TransmonPair], dcs:np.ndarray, n_runs:int=4096, z_rising_time_ns:int=1000, reset_type:Literal['thermal', 'active']='thermal', simulation:bool = False):
    ro_time = 0
    xy_time = 0
    num_qubits = len(qubits)
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)

    z_rising_time = z_rising_time_ns//4
    dc = declare(fixed)
    
    # flux crosstalk compensate coef
    fcc_flux_amp = {q.name: declare(fixed, value=0.0) for q in qubits}
    
    
    # Check readout and driving duration
    for q in qubits:
        if q.xy.operations["x180"].length//4 >= xy_time:
            xy_time = q.xy.operations["x180"].length//4
        if q.resonator.operations["readout"].length//4 >= ro_time:
            ro_time = q.resonator.operations["readout"].length//4


    align()     

    with for_(*from_array(dc, dcs)): 
    
        # reset
        for i, q in enumerate(qubits):
            assign(fcc_flux_amp[q.name], 0.0)

        # flux crosstalk compenstation
        for qp in qubit_pairs:
            if "FCC" in qp.extras:   
                for q_name in qp.extras['FCC']:    
                    if q_name in fcc_flux_amp:
                        assign(fcc_flux_amp[q_name], fcc_flux_amp[q_name]+qp.extras["FCC"][q_name] * dc)
            


        with for_(n, 0, n < n_runs, n + 1):
            # ground iq blobs for all qubits
            save(n, n_st)
        
            """ Prepare Ground |0> """
            for i, qubit in enumerate(qubits):
                if reset_type == "active":
                    active_reset_simple(qubit, "readout")
                elif reset_type == "thermal":
                    if not simulation:
                        qubit.wait(qubit.thermalization_time * u.ns)
                    else:
                        qubit.wait(16 * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")

            align()

            for i, qp in enumerate(qubit_pairs):
                
                qp.coupler.play(
                    "const", 
                    amplitude_scale = dc / qp.coupler.operations["const"].amplitude, 
                    duration = z_rising_time + ro_time
                ) 
                
            
            for i, qubit in enumerate(qubits):
                qubit.z.play(
                    "const", 
                    amplitude_scale = fcc_flux_amp[qubit.name] / qubit.z.operations["const"].amplitude, 
                    duration = z_rising_time + ro_time
                )
                qubit.resonator.wait(z_rising_time)
                qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]))


                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])
            
            align()
            
            """ Prepare Excited |1> """

            for i, qubit in enumerate(qubits):
                if reset_type == "active":
                    active_reset_simple(qubit, "readout")
                elif reset_type == "thermal":
                    if not simulation:
                        qubit.wait(qubit.thermalization_time * u.ns)
                    else:
                        qubit.wait(16 * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")
            
            align()

            
            for i, qp in enumerate(qubit_pairs):
                qp.coupler.play(
                    "const", 
                    amplitude_scale = dc / qp.coupler.operations["const"].amplitude, 
                    duration = z_rising_time + xy_time + ro_time
                )

            
            for i, qubit in enumerate(qubits):
                qubit.z.play(
                    "const", 
                    amplitude_scale = fcc_flux_amp[qubit.name] / qubit.z.operations["const"].amplitude, 
                    duration = z_rising_time + xy_time + ro_time
                )
                qubit.resonator.wait(z_rising_time + xy_time)
                qubit.xy.wait(z_rising_time)
                qubit.xy.play("x180")
            
            
                qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]))

                save(I_e[i], I_e_st[i])
                save(Q_e[i], Q_e_st[i])

            align()

    streaming = {"Ig":I_g_st, "Qg":Q_g_st, "Ie":I_e_st, "Qe":Q_e_st,  "n":n_st}
    
    return streaming
    

def make_raw_data_anime(ds:xr.Dataset, qubits:list[Transmon], frame_per_s:float=1.0):
    from matplotlib.axes import Axes
    num_qubits = len(qubits)
    # global maxima and minima
    all_I = ds.I.values.flatten()
    all_Q = ds.Q.values.flatten()
    I_min, I_max = all_I.min(), all_I.max()
    Q_min, Q_max = all_Q.min(), all_Q.max()
    
    # window Buffer
    margin_I = (I_max - I_min) * 0.1
    margin_Q = (Q_max - Q_min) * 0.1
    lim_I = (I_min - margin_I, I_max + margin_I)
    lim_Q = (Q_min - margin_Q, Q_max + margin_Q)

    # 2. create figure
    fig, axes = plt.subplots(
        ncols=num_qubits, 
        nrows=1, 
        figsize=(5 * num_qubits, 6), 
        squeeze=False 
    )
    axes = axes[0] 

    
    def update(frame_idx):
        amplitude = float(ds.amplitude[frame_idx])
        
        for i, q in enumerate(list(qubits)):
            ax:Axes = axes[i]
            ax.clear() 
            
           
            ds_q = ds.sel(qubit=q.name).isel(amplitude=frame_idx)
            
            
            ax.plot(ds_q.I.sel(state=0), ds_q.Q.sel(state=0), ".", 
                    color="tab:blue", alpha=0.3, label="Ground", markersize=2)
            ax.plot(ds_q.I.sel(state=1), ds_q.Q.sel(state=1), ".", 
                    color="tab:orange", alpha=0.3, label="Excited", markersize=2)
            
            
            ax.set_title(f"{q.name}\nCoupler Voltage: {amplitude:.4f} V\n")
            ax.set_xlabel("I")
            if i == 0: ax.set_ylabel("Q") 
            
            
            ax.set_xlim(lim_I)
            ax.set_ylim(lim_Q)

            # ax.axis("equal") 
            ax.legend(loc="upper right", markerscale=3)
            ax.grid(True, alpha=0.3)
            

    ani = animation.FuncAnimation(fig, update, frames=len(ds.amplitude), interval=1000/frame_per_s)

    
    plt.close()
    return ani


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q1', 'q2']
    num_runs: int = 4096
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    c_min_v: float = -0.2
    c_max_v: float = 0.2
    v_nums: int = 100
    outliers_threshold: float = 0.98
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    z_rising_time_ns:int = 1000
    gif_fps:int|None = 3 # assign None to skip save raw data gif


node = QualibrationNode(name="07c_Readout_Power_Optimization", parameters=Parameters())


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

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

qubit_pairs = find_c_with_q(qubit_list=[q.name for q in qubits], coupler_list=machine.active_qubit_pairs)

# %% {QUA_program}
n_runs = node.parameters.num_runs  # Number of runs
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
dcs = np.linspace(node.parameters.c_min_v, node.parameters.c_max_v, node.parameters.v_nums)




with program() as iq_blobs_snr:

    if not node.parameters.simulate:
        machine.apply_all_couplers_to_min()
        for i, qubit in enumerate(qubits):

            # Bring the active qubits to the desired frequency point
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
            qubit.align()
    
    streaming = IQ_observer(qubits, qubit_pairs, dcs, n_runs, node.parameters.z_rising_time_ns, reset_type, node.parameters.simulate)  

    with stream_processing():
        streaming['n'].save("n")
        for i in range(num_qubits):
            streaming['Ig'][i].buffer(n_runs).buffer(len(dcs)).save(f"I_g{i + 1}")
            streaming['Qg'][i].buffer(n_runs).buffer(len(dcs)).save(f"Q_g{i + 1}")
            streaming['Ie'][i].buffer(n_runs).buffer(len(dcs)).save(f"I_e{i + 1}")
            streaming['Qe'][i].buffer(n_runs).buffer(len(dcs)).save(f"Q_e{i + 1}")


if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, iq_blobs_snr, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    data_list = ["n"]
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(iq_blobs_snr)
        results = fetching_tool(job, data_list, mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_runs, start_time=results.start_time)


# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_results_as_xarray(
            job.result_handles, 
            qubits, 
            {"N": np.linspace(1, n_runs, n_runs), "amplitude": dcs}
        )
        ds = ds.assign_coords({"readout_amp": (["qubit", "amplitude"], np.array([dcs * 1 for q in qubits]))})
        ds_rearranged = xr.Dataset()


        # merge I, Q
        ds_rearranged["I"] = xr.concat([ds.I_g, ds.I_e], dim="state").assign_coords(state=[0, 1])
        ds_rearranged["Q"] = xr.concat([ds.Q_g, ds.Q_e], dim="state").assign_coords(state=[0, 1])

    
        

        for var in ds.coords:
            if var not in ds_rearranged.coords:
                ds_rearranged[var] = ds[var]

        for var in ds.data_vars:
            if var not in ["I_g", "I_e", "Q_g", "Q_e"]:
                ds_rearranged[var] = ds[var]

        ds = ds_rearranged
    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)


    node.results = {"ds": ds, "results": {}, "figs": {}}

    # %% {Data_analysis}
    def apply_fit_gmm(I, Q):
        I_mean = np.mean(I, axis=1)
        Q_mean = np.mean(Q, axis=1)
        means_init = [[I_mean[0], Q_mean[0]], [I_mean[1], Q_mean[1]]]
        precisions_init = [1 / ((np.mean(np.var(I, axis=1)) + np.mean(np.var(Q, axis=1))) / 2)] * 2
        clf = GaussianMixture(
            n_components=2,
            covariance_type="spherical",
            means_init=means_init,
            precisions_init=precisions_init,
            tol=1e-5,
            reg_covar=1e-12,
        )
        X = np.array([np.array(I).flatten(), np.array(Q).flatten()]).T
        clf.fit(X)
        ground_purity = np.sum(clf.predict(np.array([I[0], Q[0]]).T) == 0) / len(I[0])


        meas_fidelity = (
            np.sum(clf.predict(np.array([I[0], Q[0]]).T) == 0) / len(I[0])
            + np.sum(clf.predict(np.array([I[1], Q[1]]).T) == 1) / len(I[1])
        ) / 2
        loglikelihood = clf.score_samples(X)
        max_ll = np.max(loglikelihood)
        outliers = np.sum(loglikelihood > np.log(0.01) + max_ll) / len(X)
        return np.array([meas_fidelity, outliers, ground_purity])

    fit_res = xr.apply_ufunc(
        apply_fit_gmm,
        ds.I,
        ds.Q,
        input_core_dims=[["state", "N"], ["state", "N"]],
        output_core_dims=[["result"]],
        vectorize=True,
    )

    fit_res = fit_res.assign_coords(result=["meas_fidelity", "outliers", "ground_fidelity"])

    plot_individual = False
    best_data = {}

    best_amp = {}
    thrs = node.parameters.outliers_threshold
    for q in qubits:
        fit_res_q = fit_res.sel(qubit=q.name)

        while True:
            valid_amps = fit_res_q.amplitude[(fit_res_q.sel(result="outliers") >= thrs)]

            if len(valid_amps.values) == 0:
                thrs *= 0.97
            else:
                break

        amps_fidelity = fit_res_q.sel(amplitude=valid_amps.values, result="meas_fidelity")
        
        best_amp[q.name] = float(amps_fidelity.readout_amp[amps_fidelity.argmax()])
        
        print(f"amp for {q.name} is {best_amp[q.name]}")
        node.results["results"][q.name] = {}
        node.results["results"][q.name]["best_amp"] = best_amp[q.name]

        # Select data for the best amplitude
        best_amp_data = ds.sel(qubit=q.name, amplitude=float(amps_fidelity.idxmax()))
        best_data[q.name] = best_amp_data

        # Extract I and Q data for ground and excited states
        I_g = best_amp_data.I.sel(state=0)
        Q_g = best_amp_data.Q.sel(state=0)
        I_e = best_amp_data.I.sel(state=1)
        Q_e = best_amp_data.Q.sel(state=1)
        angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
            I_g, Q_g, I_e, Q_e, True, b_plot=plot_individual
        )
        I_rot = I_g * np.cos(angle) - Q_g * np.sin(angle)
        hist = np.histogram(I_rot, bins=100)
        RUS_threshold = hist[1][1:][np.argmax(hist[0])]
        if plot_individual:
            fig = plt.gcf()
            plt.show()
            node.results["figs"][q.name] = fig
        node.results["results"][q.name]["angle"] = float(angle)
        node.results["results"][q.name]["threshold"] = float(threshold)
        node.results["results"][q.name]["fidelity"] = float(fidelity)
        node.results["results"][q.name]["confusion_matrix"] = np.array([[gg, ge], [eg, ee]])
        node.results["results"][q.name]["rus_threshold"] = float(RUS_threshold)


    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        # 1. 取出該 Qubit 的數據
        # fit_res 的結構应该是 (qubit, readout_amp, result)
        # 我們先選定 qubit
        data = fit_res.sel(qubit=qubit["qubit"])
        
        # 2. 建立雙 Y 軸 (左邊畫 Fidelity, 右邊畫 Outliers)
        ax_right = ax.twinx()
        
        # --- 左軸: 畫 Meas Fidelity 與 Ground Fidelity ---
        # 取出 X 軸數據
        x_vals = data.readout_amp.values
        
        # 畫 Meas Fidelity (平均保真度)
        l1, = ax.plot(x_vals, data.sel(result="meas_fidelity"), 
                    color="C0", label="Meas Fidelity", linestyle="-")
        
        # 畫 Ground Fidelity (Ground Purity)
        # 建議: 如果這兩條線太接近，可以用透明度 alpha 或點線區分
        l2, = ax.plot(x_vals, data.sel(result="ground_fidelity"), 
                    color="green", label="Ground Purity")

        # --- 右軸: 畫 Outliers ---
        l3, = ax_right.plot(x_vals, (1-data.sel(result="outliers"))*100, 
                            color="red", label="Outliers",alpha=0.4)

        # 3. 標示最佳振幅點 (Best Amplitude)
        # 畫一條垂直黑線
        best_amp_val = best_amp[qubit["qubit"]]
        l_line = ax.axvline(best_amp_val, color="k", linestyle=":", label="Best Amp")


        
        # 4. 設定 Label 與 標題
        ax.set_xlabel("Coupler Flux Pulse Amplitude (V)")
        ax.set_ylabel("Fidelity", color="C0")
        ax_right.set_ylabel("Outliers Ratio (%)", color="C3")
        # ax_right.set_yscale('log')
        ax_right.set_ylim(0, 10)
        # 設定右軸刻度顏色，方便辨識
        ax.tick_params(axis='y', labelcolor="C0")
        ax_right.tick_params(axis='y', labelcolor="C3")

        ax.set_title(f"Qubit: {qubit['qubit']}")

        # 5. 合併 Legend (圖例)
        # 因為用了兩個軸，普通的 ax.legend() 只會顯示左軸的
        lines = [l1, l2, l3, l_line]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best', fontsize='small')
        ax.grid()
    grid.fig.suptitle("Fidelity and inlier probability VS coupler Bias")
    grid.fig.set_size_inches(15, 8)
    plt.tight_layout()
    plt.show()
    node.results["figure_assignment_fid"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds_q = best_data[qubit["qubit"]]
        qn = qubit["qubit"]
        ax.plot(
            1e3
            * (
                ds_q.I.sel(state=0) * np.cos(node.results["results"][qn]["angle"])
                - ds_q.Q.sel(state=0) * np.sin(node.results["results"][qn]["angle"])
            ),
            1e3
            * (
                ds_q.I.sel(state=0) * np.sin(node.results["results"][qn]["angle"])
                + ds_q.Q.sel(state=0) * np.cos(node.results["results"][qn]["angle"])
            ),
            ".",
            alpha=0.1,
            label="Ground",
            markersize=1,
        )
        ax.plot(
            1e3
            * (
                ds_q.I.sel(state=1) * np.cos(node.results["results"][qn]["angle"])
                - ds_q.Q.sel(state=1) * np.sin(node.results["results"][qn]["angle"])
            ),
            1e3
            * (
                ds_q.I.sel(state=1) * np.sin(node.results["results"][qn]["angle"])
                + ds_q.Q.sel(state=1) * np.cos(node.results["results"][qn]["angle"])
            ),
            ".",
            alpha=0.1,
            label="Excited",
            markersize=1,
        )
        ax.axvline(
            1e3 * node.results["results"][qn]["rus_threshold"], color="k", linestyle="--", lw=0.5, label="RUS Threshold"
        )
        ax.axvline(1e3 * node.results["results"][qn]["threshold"], color="r", linestyle="--", lw=0.5, label="Threshold")
        ax.axis("equal")
        ax.set_xlabel("I [mV]")
        ax.set_ylabel("Q [mV]")
        ax.set_title(qubit["qubit"])

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    grid.fig.suptitle("g.s. and e.s. discriminators (rotated)")
    plt.tight_layout()

    node.results["figure_IQ_blobs"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        confusion = node.results["results"][qubit["qubit"]]["confusion_matrix"]
        ax.imshow(confusion)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels=["|g>", "|e>"])
        ax.set_yticklabels(labels=["|g>", "|e>"])
        ax.set_ylabel("Prepared")
        ax.set_xlabel("Measured")
        ax.text(0, 0, f"{100 * confusion[0][0]:.1f}%", ha="center", va="center", color="k")
        ax.text(1, 0, f"{100 * confusion[0][1]:.1f}%", ha="center", va="center", color="w")
        ax.text(0, 1, f"{100 * confusion[1][0]:.1f}%", ha="center", va="center", color="w")
        ax.text(1, 1, f"{100 * confusion[1][1]:.1f}%", ha="center", va="center", color="k")
        ax.set_title(qubit["qubit"])

    grid.fig.suptitle("g.s. and e.s. fidelity")
    plt.tight_layout()
    plt.show()
    node.results["figure_fidelities"] = grid.fig


    # %% {Update_state}
    if node.parameters.load_data_id is None:
        # with node.record_state_updates():
        #     for qubit in qubits:
        #         qubit.resonator.operations["readout"].integration_weights_angle -= float(
        #             node.results["results"][qubit.name]["angle"]
        #         )
        #         qubit.resonator.operations["readout"].threshold = float(node.results["results"][qubit.name]["threshold"])
        #         qubit.resonator.operations["readout"].rus_exit_threshold = float(
        #             node.results["results"][qubit.name]["rus_threshold"]
        #         )
        #         qubit.resonator.operations["readout"].amplitude = float(node.results["results"][qubit.name]["best_amp"])
        #         qubit.resonator.confusion_matrix = node.results["results"][qubit.name]["confusion_matrix"].tolist()
        pass

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()
        if node.parameters.gif_fps is not None: 
            from qualibrate_config.resolvers import get_qualibrate_config_path, get_qualibrate_config
            from quam_libs.compat import get_node_dir_path
            import os
            qs = get_qualibrate_config(get_qualibrate_config_path())
            base_path = qs.storage.location

            node_dir = get_node_dir_path(node.snapshot_idx, base_path)
            ani = make_raw_data_anime(ds, qubits, node.parameters.gif_fps)
        
            ani.save(os.path.join(node_dir, f"coupler_flux_sweep.gif"), writer='pillow', fps=node.parameters.gif_fps)


# %%
