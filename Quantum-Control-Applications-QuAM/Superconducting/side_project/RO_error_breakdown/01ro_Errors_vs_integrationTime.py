# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import math
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
from qualang_tools.loops import from_array
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from time import time 
from scipy.stats import norm
from quam_libs.lib.qubit_thermometer import repetition_data, StateDiscrimination, prepare_dataset_for_qcat
from quam_libs.experiments.T1_patch.T1_statics import run_t1_and_get_ds, fit_decay_exp


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ['q3', 'q4', 'q5'] #The qubit to be measured. If None, all active qubits will be measured
    num_runs: int = 4096
    flux_point_joint_or_independent_or_arbitrary: Literal['joint', 'independent'] = 'joint'   
    simulate: bool = False
    timeout: int = 100
    use_state_discrimination: bool = False
    reset_type: Literal['thermal'] = "thermal"
    load_data_id: Optional[int] = 2929
    multiplexed: bool = 0
    min_integrate_time_ns:int = 800
    max_integrate_time_ns:int = 3380 
    time_step:int = 52
    num_repeats:int = 10
    measure_T1_also:bool = True
    interleaved: bool = True # if interleaved, T1 statics will be measured after each repeat of readout_length
    
    

node = QualibrationNode(
    name="01ro_separation_vs_integration",
    parameters=Parameters()
)

u = unit(coerce_to_integer=True)

machine = QuAM.load()


if node.parameters.load_data_id is None:
    qmm = machine.connect()

qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)

lens = np.arange(node.parameters.min_integrate_time_ns, node.parameters.max_integrate_time_ns, node.parameters.time_step).tolist()

print("Total integration time points: ", len(lens))

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
                    qubit.wait(2 * qubit.thermalization_time * u.ns)
                    

            align(*[q.xy.name for q in multiplexed_qubits.values()] +
                   [q.resonator.name for q in multiplexed_qubits.values()] +
                   [q.z.name for q in multiplexed_qubits.values()])

            for i, qubit in multiplexed_qubits.items():
                qubit.resonator.measure(operation_name, qua_vars=(I_g[i], Q_g[i]))
                qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])

            
            align(*[q.xy.name for q in multiplexed_qubits.values()] +
                [q.resonator.name for q in multiplexed_qubits.values()] +
                [q.z.name for q in multiplexed_qubits.values()])

            if not node.parameters.simulate:
                for i, qubit in multiplexed_qubits.items():
                    qubit.wait(2*qubit.thermalization_time * u.ns)

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
    config = machine.generate_config()
    samples, fig = simulate_and_plot(qmm, config, iq_blobs, node.parameters)
    node.results = {"figure": fig}
    node.save()
else:
    if node.parameters.load_data_id is None:
        num_repeats = node.parameters.num_repeats
        start = time()
        fault_tolerance = 5
        dss = []
        did_run_lens = []
        T1_dss = []
        k = 0
        while k < len(lens):
            inte_l = lens[k]
            for q in qubits:
                q.resonator.operations.readout.length = inte_l * u.ns
             
            try:  # prevent getting an unpredictable error like connection failed or qmm closed.
                repeat_dss = []
                if node.parameters.interleaved:
                    repeat_T1_dss = [] 
                
                # 2. 使用變數 num_repeats
                for repeat in range(num_repeats):
                    
                    with qm_session(qmm, machine.generate_config(), timeout=node.parameters.timeout) as qm:
                        job = qm.execute(iq_blobs)
                        for i in range(num_qubits):
                            results = fetching_tool(job, ["n"], mode="live")
                            while results.is_processing():
                                n = results.fetch_all()[0]
                                if num_repeats < 5:
                                    progress_counter(n, n_runs, start_time=results.start_time)
                    
                    if num_repeats > 4:
                        print(f"\r{round(inte_l)} ns, repeat {repeat + 1}/{num_repeats} done.", end="", flush=True)
                    
                    ds_single = fetch_dataset(job, qubits, node.parameters)
                    repeat_dss.append(ds_single)

                    if node.parameters.interleaved and node.parameters.measure_T1_also:
                        ds_t1 = run_t1_and_get_ds(statics_num=1, multiplexing=node.parameters.multiplexed)
                        repeat_T1_dss.append(ds_t1)

                

                # 3. 將同一個 readout_length 下的多次 repeat 拼接出 "repeat" 維度
                ds_inte_l = xr.concat(repeat_dss, dim="repeat")
                dss.append(ds_inte_l)

                if node.parameters.measure_T1_also: 
                    if not node.parameters.interleaved:
                        ds_t1 = run_t1_and_get_ds(statics_num=num_repeats, multiplexing=node.parameters.multiplexed)
                    else:
                        # 將同一個 readout_length 下的多次 repeat 拼接出 "iteration" 維度
                        combined_ds = xr.concat(repeat_T1_dss, dim='iteration')
                        ds_t1 = combined_ds.assign_coords(
                            iteration=np.arange(len(combined_ds.iteration))
                        )
                    T1_dss.append(ds_t1)

                # Record
                did_run_lens.append(inte_l)
                k += 1
                
            except Exception as e:
                if fault_tolerance < 0:
                    print("Run out all the tolerance, we stop here.")
                    break

                fault_tolerance -= 1
                print(f"\nAttempt failed: {e}. Try it again...\n")

        end = time()
        
        # 合併後的 ds 將包含 (readout_length, repeat, ...) 等維度
        ds = xr.concat(dss, dim="readout_length").assign_coords(readout_length=did_run_lens)
        if node.parameters.measure_T1_also:
            T1_tot_ds = xr.concat(T1_dss, dim="readout_length").assign_coords(readout_length=did_run_lens)
            node.results = {"ds": ds, "T1_all_ds": T1_tot_ds, "figs": {}, "results": {}}
        else:
            node.results = {"ds": ds, "figs": {}, "results": {}}

        reload_qbs = False
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"] 
        did_run_lens = ds.readout_length.values.tolist()
        try:
            T1_tot_ds = node.results["T1_all_ds"]
        except:
            T1_tot_ds = None
        
        machine = node.machine
        reload_qbs = True

# %%
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        try:
            print(f"Total execution time: {round(end - start)} seconds")
        except:
            pass


# %% {analysis}
if not node.parameters.simulate:
    # models 結構調整為 [qubit_name][readout_length_idx][repeat_idx]
    models = {q.name: [] for q in qubits}
    trained_params = {}
    
    # 儲存平均值（用於擬合與主要繪圖）
    sep_error = {q.name: [] for q in qubits}
    sep_error_std = {q.name: [] for q in qubits}  # 記錄 repeat 間的標準差
    blob_sigma = {q.name: [] for q in qubits}
    blob_dist = {q.name: [] for q in qubits}
    p01_p10_diff = {q.name: [] for q in qubits}
    
    T1ss = {q.name: [] for q in qubits}
    mean_T1s = {q.name: [] for q in qubits}
    std_T1s = {q.name: [] for q in qubits}
    T1_fits = {q.name: [] for q in qubits}

    for readout_len in range(ds.dims['readout_length']):
        ds_len = ds.isel(readout_length=readout_len)
        
        len_models = {q.name: [] for q in qubits}
        len_sep_error = {q.name: [] for q in qubits}
        len_blob_sigma = {q.name: [] for q in qubits}
        len_blob_dist = {q.name: [] for q in qubits}
        len_p01_p10_diff = {q.name: [] for q in qubits}

        # 針對當前 readout_length 下的每一個 repeat 進行 State Discrimination
        for rep in range(ds_len.dims['repeat']):
            ds_rep = ds_len.isel(repeat=rep)
            dss = prepare_dataset_for_qcat(ds_rep)
            sep_data = repetition_data(dss, repetition_dim="qubit")

            for sq_data in sep_data:
                qubit_name = sq_data["qubit"].values.item()

                analysis = StateDiscrimination(sq_data)
                analysis._start_analysis()
                len_models[qubit_name].append(analysis)
                
                t_params = analysis.analysis_result['trained_paras']
                means = np.array(t_params['mean'])
                std = t_params['std']
                
                d = np.linalg.norm(means[1] - means[0])
                snr = d / std
                overlap_error = 0.5 * math.erfc(snr / (2 * math.sqrt(2)))
                (_, p01), (p10, _) = analysis.analysis_result['gaussian_norms']

                len_blob_sigma[qubit_name].append(std)
                len_blob_dist[qubit_name].append(d)
                len_sep_error[qubit_name].append(overlap_error)
                len_p01_p10_diff[qubit_name].append(p10 - p01)

        # 統計該 readout_length 下跨 repeat 的平均值與標準差
        for q in qubits:
            q_name = q.name
            models[q_name].append(len_models[q_name])
            
            sep_error[q_name].append(np.mean(len_sep_error[q_name]))
            sep_error_std[q_name].append(np.std(len_sep_error[q_name]))
            blob_sigma[q_name].append(np.mean(len_blob_sigma[q_name]))
            blob_dist[q_name].append(np.mean(len_blob_dist[q_name]))
            p01_p10_diff[q_name].append(np.mean(len_p01_p10_diff[q_name]))

        # T1 分析維持原樣
        if node.parameters.measure_T1_also:
            a_T1_ds = T1_tot_ds.isel(readout_length=readout_len)
            qbs = a_T1_ds.qubit.values
            iterations = a_T1_ds.iteration.values
            
            t1_collection = {q: [] for q in qbs}
            fit_collection = {q: [] for q in qbs}
        
            for q_name in qbs:
                for iter_val in iterations:
                    ds_sub = a_T1_ds.sel(qubit=q_name, iteration=iter_val)
                    try:
                        fit_data = fit_decay_exp(ds_sub.state, "idle_time")
                    except:
                        fit_data = fit_decay_exp(ds_sub.I, "idle_time")
        
                    fit_collection[q_name].append(fit_data)
                    decay_val = fit_data.sel(fit_vals="decay").values
                    tau_val = -1 / decay_val
                    t1_collection[q_name].append(tau_val)

                mean_T1s[q_name].append(np.average(t1_collection[q_name]))
                std_T1s[q_name].append(np.std(t1_collection[q_name]))
                T1ss[q_name].append(t1_collection[q_name])
                T1_fits[q_name].append(fit_collection[q_name])

    for q in qubits:
        node.results['results'][q.name] = {}
        node.results['results'][q.name]['separation_errors'] = sep_error[q.name]
        node.results['results'][q.name]['separation_errors_std'] = sep_error_std[q.name]

# %%
lens = did_run_lens
grid_100 = QubitGrid(ds, [q.grid_location for q in qubits], size=10)
for ax, qubit in grid_iter(grid_100):
    ax.axis("off")
    ax.set_title(qubit['qubit'])
    w, h = 0.42, 0.40

    cata = {"g": {"i": [], "q": []}, "e": {"i": [], "q": []}}
    figs = {
        "gi": ax.inset_axes([0.0, 0.60, w, h]),
        "gq": ax.inset_axes([0.58, 0.60, w, h]),
        "ei": ax.inset_axes([0.0, 0.00, w, h]),
        "eq": ax.inset_axes([0.58, 0.00, w, h])
    }

    # 遍歷所有的 readout_length 以及其下的 repeats
    for len_idx in range(len(models[qubit['qubit']])):
        for rep_idx in range(len(models[qubit['qubit']][len_idx])):
            m = models[qubit['qubit']][len_idx][rep_idx]
            indexed_means = sorted(enumerate(m.analysis_result['trained_paras']['mean']), key=lambda x: x[1][0])

            for i, (idx, center) in enumerate(indexed_means):
                if idx == 0:
                    cata['g']['i'].append(center[0])
                    cata['g']['q'].append(center[1])
                else:
                    cata['e']['i'].append(center[0])
                    cata['e']['q'].append(center[1])

    # 畫圖時需要注意 lens 與 cata 的長度對齊 (若畫散佈圖可將 lens 重複 N 次，或改為繪製 scatter)
    lens_expanded = np.repeat(lens, node.parameters.num_repeats)
    for state in ['g', 'e']:
        for signal in ['i', 'q']:
            figs[state+signal].scatter(lens_expanded, cata[state][signal], alpha=0.6, s=15)
            figs[state+signal].set_title(f"|{state.upper()}>, {signal.upper()}-Channel")

plt.tight_layout()
plt.show()


#%% {Plot}
from matplotlib.axes import Axes
from scipy.optimize import curve_fit
from scipy.special import erfc
def fit_sep_error(t, A):
    return 0.5 * erfc(A * np.sqrt(t))
def fit_sqrt_t(t, A, B):
    return A / np.sqrt(t) / B
def fit_linear(t, a, b):
    return a*t + b

T1s = {q.name: q.extras["T1"]*1e6 for q in qubits}
T1_devs = {q.name: q.extras["T1_dev"]*1e6 for q in qubits}

grid = QubitGrid(ds, [q.grid_location for q in qubits], size=6)
grid.fig.set_size_inches(6 * num_qubits, 5)
for ax, qubit in grid_iter(grid):

    # 1. 計算跨 repeat 的各項指標 mean 與 std
    ro_fidelity = []
    ro_fidelity_std = []
    p01_p10_diff_mean = []
    p01_p10_diff_std = []

    for len_models in models[qubit["qubit"]]:
        ro_fid_reps = []
        p01_p10_diff_reps = []
        for m in len_models:
            (p00, p01), (p10, p11) = m.analysis_result["gaussian_norms"]
            ro_fid_reps.append(0.5 * (p00 + p11))
            p01_p10_diff_reps.append(p10 - p01)
        
        # Readout Fidelity 平均值與標準差
        ro_fidelity.append(np.mean(ro_fid_reps))
        ro_fidelity_std.append(np.std(ro_fid_reps))

        # P(0|1)-P(1|0) 平均值與標準差
        p01_p10_diff_mean.append(np.mean(p01_p10_diff_reps))
        p01_p10_diff_std.append(np.std(p01_p10_diff_reps))

    ax: Axes
    # 轉換為百分比 [%]
    err_pct = 100 * np.array(sep_error[qubit['qubit']])
    err_std_pct = 100 * np.array(sep_error_std[qubit['qubit']])

    err_relax_pct = 100 * np.array(p01_p10_diff_mean)
    err_relax_std_pct = 100 * np.array(p01_p10_diff_std)

    ro_fidelity_pct = 100 * (1 - np.array(ro_fidelity))
    ro_fidelity_std_pct = 100 * np.array(ro_fidelity_std)

    p0_guess = 1.0 / np.sqrt(np.mean(np.array(lens)))

    # 2. 進行非線性最小二乘法擬合
    popt_erfc, _ = curve_fit(
            fit_sep_error, 
            lens, 
            sep_error[qubit['qubit']], 
            p0=[p0_guess], 
            bounds=(0, np.inf)
        )

    popt_linear, _ = curve_fit(
        fit_linear, 
        lens, 
        p01_p10_diff_mean
    )
    
    # 3. 產生平滑的 X 軸數據點（讓擬合線看起來平滑）
    lens_smooth = np.linspace(min(lens), max(lens), 200)
    err_fit_smooth_erfc = fit_sep_error(lens_smooth, *popt_erfc)
    linear_p01_p10_diff = fit_linear(lens_smooth, *popt_linear)

    T1_estima_us = ((1/popt_linear[0])/2)*1e-3
    estimate_control_error = popt_linear[1] * 100
    
    # T1 measured
    Tro_over_2T1 = []
    Tro_over_2T1_err = []
    if node.parameters.measure_T1_also:
        
        for idx, a_len in enumerate(lens):
            
            Tro_over_2T1.append(1 - np.exp(-a_len / (2 * mean_T1s[qubit['qubit']][idx] * 1e3)))
            # relative std
            std_pct = std_T1s[qubit['qubit']][idx] / mean_T1s[qubit['qubit']][idx]
            Tro_over_2T1_err.append(Tro_over_2T1[idx] * std_pct)

        popt_ro_T1, _ = curve_fit(
            fit_linear, 
            lens, 
            Tro_over_2T1
        )

        T1_estim = 1e-3 / (2 * popt_ro_T1[0])

        ax.errorbar(
            lens, 
            100*np.array(Tro_over_2T1), 
            yerr=100*np.array(Tro_over_2T1_err), 
            fmt='*', 
            label=f"Error Estimation, T1 measured ~ {round(T1_estim, 1)} µs", 
            color='green'
        )
        ax.plot(lens_smooth, 100*fit_linear(lens_smooth, *popt_ro_T1), 'r--')
        x_lim = [max(err_pct), max(err_relax_pct), max(100 * np.array(Tro_over_2T1)), max(ro_fidelity_pct)]
    else:
        x_lim = [max(err_pct), max(err_relax_pct), max(ro_fidelity_pct)]

    # ==================== 左軸繪圖 (Errors [%]) ====================
    # A. Separation Error (帶 Error Bar)
    ax.errorbar(
        lens, 
        err_pct, 
        yerr=err_std_pct, 
        fmt='o', 
        capsize=3, 
        label="Separation Error"
    )
    
    # B. P(0|1)-P(1|0) Relaxation Error (由 scatter 改為帶 Error Bar 的 errorbar)
    ax.errorbar(
        lens, 
        err_relax_pct, 
        yerr=err_relax_std_pct, 
        fmt='^', 
        capsize=3, 
        label=f"P(0|1)-P(1|0), T1 estimated ~ {round(T1_estima_us,1)} µs \n estimated control error ~ {round(estimate_control_error,2)} %"
    )

    # C. T1 測量比值與擬合線
    
    ax.plot(lens_smooth, 100 * linear_p01_p10_diff, color='brown', linestyle='--')

    # 座標軸設定
    ax.set_xlim(min(lens) - 0.05 * (max(lens) - min(lens)), max(lens) + 0.05 * (max(lens) - min(lens)))   
    ax.set_ylim(-0.1, 1.1 * max(x_lim))    
    ax.set_xlabel("Integration Time [ns]")
    ax.set_ylabel("Error [%]")
    ax.set_title(f"{qubit['qubit']}")
    ax.grid()
    # ==================== 右軸繪圖 (Total Readout Error [%]) ====================
    #ax2 = ax.twinx()

    color_ro = "tab:purple"  # 使用紫色區隔右軸
    # Total RO Error (由 scatter 改為帶 Error Bar 的 errorbar)
    ax.errorbar(
        lens,
        ro_fidelity_pct,
        yerr=ro_fidelity_std_pct,
        fmt="s",
        color=color_ro,
        capsize=3,
        label="Total RO Error",
    )
    
    # 設定右側 Y 軸標籤與範圍
    # ax2.set_ylabel("Total Readout Error [%]", color=color_ro, fontsize=10)
    # ax2.tick_params(axis="y", labelcolor=color_ro)
    # ax2.set_ylim(0, max(ro_fidelity_pct) * 1.2 if len(ro_fidelity_pct) > 0 else 1)

    # ==================== 合併左右兩軸圖例 ====================
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),  # x=0.5 居中，y=-0.22 移動至 X 軸標籤下方
        ncol=2,                       # 2 欄矩陣排列
        fontsize=9,                   # 稍微縮小字體以適應變動的寬度
        framealpha=0.9,               # 半透明背景框
        handletextpad=0.3,            # 圖標與文字間距
        columnspacing=0.6,            # 兩欄間距
    )

    ax.grid(True)

# ==================== 迴圈外版面動態佈局 ====================
plt.suptitle("Error VS Integration time", y=0.98)

# 先自動排版
plt.tight_layout()

# 動態設定間距：保留 28% 底部空間給 Legend，並拉開左右子圖間距 (wspace) 避免 Legend 相撞
plt.subplots_adjust(bottom=0.28, wspace=0.35)

plt.show()

node.results["Errors_vs_time"] = grid.fig

# %%
grid_1 = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid_1):

    # 1. 提煉跨 repeat 的 blob_sigma 平均值與標準差
    sigma_means = []
    sigma_stds = []
    for len_models in models[qubit["qubit"]]:
        sigmas = [m.analysis_result['trained_paras']['std'] for m in len_models]
        sigma_means.append(np.mean(sigmas))
        sigma_stds.append(np.std(sigmas))

    sigma_means = np.array(sigma_means)
    sigma_stds = np.array(sigma_stds)

    # 2. 進行 1/√t 非線性擬合 (使用跨 repeat 的平均值)
    popt_sqrt, _ = curve_fit(
        fit_sqrt_t, 
        lens, 
        sigma_means
    )

    lens_smooth = np.linspace(min(lens), max(lens), 200)
    sigma_fit_smooth_sqrt = fit_sqrt_t(lens_smooth, *popt_sqrt)

    # 3. 將 ax.scatter 改為帶 Error Bar 的 ax.errorbar (注意 yerr 也需 * 1000 轉為 mV)
    ax.errorbar(
        lens, 
        1000 * sigma_means, 
        yerr=1000 * sigma_stds, 
        fmt='o', 
        capsize=3
    )
    ax.plot(lens_smooth, 1000 * sigma_fit_smooth_sqrt, 'r--', label=f"Fit: 1/√t")
    ax.set_xlabel("Integration Time [ns]")
    ax.set_ylabel("Blob Sigma [mV]")
    ax.set_title(qubit['qubit'])
    ax.legend()
    ax.grid(True)

plt.suptitle("Noise VS Integration time")
plt.tight_layout()
plt.show()
node.results["blob_sigma_vs_time"] = grid_1.fig


# Grid 2: Signal VS Integration time
grid_2 = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid_2):

    # 1. 提煉跨 repeat 的 blob_dist 平均值與標準差
    dist_means = []
    dist_stds = []
    for len_models in models[qubit["qubit"]]:
        dists = []
        for m in len_models:
            means = np.array(m.analysis_result['trained_paras']['mean'])
            dists.append(np.linalg.norm(means[1] - means[0]))
        dist_means.append(np.mean(dists))
        dist_stds.append(np.std(dists))

    dist_means = np.array(dist_means)
    dist_stds = np.array(dist_stds)

    # 2. 將 ax.scatter 改為帶 Error Bar 的 ax.errorbar
    ax.errorbar(
        lens, 
        1000 * dist_means, 
        yerr=1000 * dist_stds, 
        fmt='o', 
        capsize=3
    )

    ax.set_xlabel("Integration Time [ns]")
    ax.set_ylabel("Blobs Distance [mV]")
    ax.set_title(qubit['qubit'])
    # ax.legend()
    ax.grid(True)

plt.suptitle("Signal VS Integration time")
plt.tight_layout()
plt.show()
node.results["blob_distance_vs_time"] = grid_2.fig


# %% 檢查每個 Qubit 的 T1 散佈情況與離群值
if node.parameters.measure_T1_also:
    grid_t1 = QubitGrid(ds, [q.grid_location for q in qubits], size=6)
    for ax, qubit in grid_iter(grid_t1):
        q_name = qubit['qubit']
        
        # 1. 畫出每一個 iteration / repeat 計算出的 T1 散佈點
        for idx, a_len in enumerate(lens):
            # 取出該 readout_length 下的所有 T1 值 (轉成 µs)
            t1_vals = np.array(T1ss[q_name][idx])  
            
            # 繪製半透明的個案散佈點
            ax.scatter([a_len] * len(t1_vals), t1_vals, color='tab:blue', alpha=0.3, s=15)
        
        # 2. 疊加平均值與標準差 Error Bar
        mean_t1_us = np.array(mean_T1s[q_name])
        std_t1_us = np.array(std_T1s[q_name])
        ax.errorbar(lens, mean_t1_us, yerr=std_t1_us, fmt='o-', color='tab:red', capsize=3, label="Mean T1 ± Std")
        
        ax.set_xlabel("Integration Time [ns]")
        ax.set_ylabel("T1 [µs]")
        ax.set_title(f"{q_name} - {round(np.median(mean_t1_us), 2)} ± {round(np.std(mean_t1_us), 2)} µs")
        ax.grid(True)
        ax.legend()

    plt.suptitle("T1 Value Distribution VS Integration Time")
    plt.tight_layout()
    plt.show()
    node.results["t1_distribution_vs_time"] = grid_t1.fig


    for q in qubits:
        target_qubit = q.name  # 指定檢查的 qubit
        num_lens_to_show = max(1, int(len(lens)/10))  # 抽查 10% 代表性的 readout length
        selected_indices = np.linspace(0, len(lens)-1, num_lens_to_show, dtype=int)

        fig, axes = plt.subplots(1, num_lens_to_show, figsize=(3.5 * num_lens_to_show, 3.5), sharey=True)

        for i, len_idx in enumerate(selected_indices):
            ax = axes[i] if num_lens_to_show > 1 else axes
            a_len = lens[len_idx]
            
            # 從 T1_tot_ds 擷取數據
            a_T1_ds = T1_tot_ds.isel(readout_length=len_idx).sel(qubit=target_qubit)
            idle_times = a_T1_ds.idle_time.values
            
            # 畫出該 readout length 下每一次 iteration 的 raw trace
            for iter_val in a_T1_ds.iteration.values:
                ds_sub = a_T1_ds.sel(iteration=iter_val)
                signal = ds_sub.state.values if 'state' in ds_sub else ds_sub.I.values
                ax.plot(idle_times, signal, alpha=0.3)
                
            ax.set_title(f"RO Length: {a_len} ns")
            ax.set_xlabel("Idle Time [us]")
            if i == 0:
                ax.set_ylabel("State / Signal [a.u.]")
            ax.grid(True)

        plt.suptitle(f"{target_qubit} Raw T1 Decay Traces")
        plt.tight_layout()
        plt.show()
        node.results[f"{target_qubit}_Raw_T1_Decay_Traces"] = fig

# %%
def get_gap_population(self, bridge_sigma=2.5):
    """
    計算落在 BRIDGE_SIGMA Gap 區間的 Population 比率。
    
    Returns:
        tuple: (pop_ratio_0, pop_ratio_1)
            - pop_ratio_0: prepared_state = 0 落在 Gap 的比例 (熱激發 Thermal Excitation)
            - pop_ratio_1: prepared_state = 1 落在 Gap 的比例 (能量弛豫 T1 Decay)
            若兩高斯峰重疊過大 (b0 >= b1) 或資料缺失，則回傳 (np.nan, np.nan)
    """
    if self.analysis_result is None or 'trained_paras' not in self.analysis_result:
        return np.nan, np.nan

    # --- 1. 取得 x, y 座標與網格步長 dx, dy ---
    x = self.hist_dataset['x'].values
    y = self.hist_dataset['y'].values
    dx = (x[1] - x[0]) if len(x) > 1 else 1.0
    dy = (y[1] - y[0]) if len(y) > 1 else 1.0
    bin_area = dx * dy

    # --- 2. 計算各 State 沿 Q 軸積分後的 1D 機率分布 ---
    prob_1d_list = []
    for state in self.hist_dataset.coords['prepared_state'].values:
        density_2d = self.hist_dataset['density'].sel(prepared_state=state).values
        prob_2d = density_2d * bin_area
        prob_1d = prob_2d.sum(axis=0)
        prob_1d_list.append(prob_1d)

    # --- 3. 提取擬合的高斯參數 (mean & sigma) ---
    trained_paras = self.analysis_result['trained_paras']
    means = trained_paras['mean']
    covs = trained_paras['covariance']

    if len(means) < 2:
        return np.nan, np.nan

    mu_xs, sigma_xs = [], []
    for k in range(len(means)):
        mean_k = np.array(means[k])
        mu_x = mean_k[0] if mean_k.ndim > 0 else float(mean_k)
        
        cov_k = covs
        var_x = cov_k[0][0] if isinstance(cov_k, (list, np.ndarray)) else float(cov_k)
        sigma_x = np.sqrt(var_x)
        
        mu_xs.append(mu_x)
        sigma_xs.append(sigma_x)

    # 自動識別左側 (idx0, state 0) 與右側 (idx1, state 1) 高斯
    idx0, idx1 = (0, 1) if mu_xs[0] < mu_xs[1] else (1, 0)

    # --- 4. 計算 Sigma 邊界 ---
    b0_sig = mu_xs[idx0] + bridge_sigma * sigma_xs[idx0]  # State 0 (+kσ)
    b1_sig = mu_xs[idx1] - bridge_sigma * sigma_xs[idx1]  # State 1 (-kσ)

    # --- 5. Overlap 判斷：當 State 0 右邊界 >= State 1 左邊界，代表 Overlap 過大 ---
    if b0_sig >= b1_sig:
        return np.nan, np.nan

    # --- 6. 計算落在 Gap [b0_sig, b1_sig] 的 Raw Data Population ---
    mask_gap = (x >= b0_sig) & (x <= b1_sig)
    
    pop_ratio_0 = prob_1d_list[idx0][mask_gap].sum()  # prepared = |0>
    pop_ratio_1 = prob_1d_list[idx1][mask_gap].sum()  # prepared = |1>

    return float(pop_ratio_0), float(pop_ratio_1)

# %% {Checking Raws}
if node.parameters.measure_T1_also:
    grid_checking = QubitGrid(ds, [q.grid_location for q in qubits], size=6)
    for ax, qubit in grid_iter(grid_checking):
        q_name = qubit['qubit']
        offset_mean_collection = []
        offset_std_collection = []
        offset_raw_collection = []
        for a_t1_fit_result in T1_fits[q_name]:
            offset_raws = [] 
            for repeat in range(len(a_t1_fit_result)):
                offset_raws.append(a_t1_fit_result[repeat].sel(fit_vals="offset"))

            offset_mean_collection.append(np.mean(offset_raws))
            offset_std_collection.append(np.std(offset_raws))
            offset_raw_collection.append(offset_raws)       

        ax.errorbar(lens, offset_mean_collection, yerr=offset_std_collection, alpha=0.5, label="Offset from T1 Fit", fmt='o', zorder=2)
        x_raws = [x_val for x_val, raw_vals in zip(lens, offset_raw_collection) for _ in raw_vals]
        y_raws = [y for raw_vals in offset_raw_collection for y in raw_vals]

        # 繪製紅點散佈圖
        ax.scatter(x_raws, y_raws, color='red', s=15, alpha=0.2, label='Raw Data', zorder=3)
        ax.set_title(f"{q_name}")
        ax.set_xlabel("Integration Time [ns]")  
        ax.set_ylabel("T1 Decay Offset", fontsize=12)
        ax.legend()
        ax.grid()
    grid_checking.fig.suptitle("T1 Decay Offset Check")
    plt.show()
    node.results[f"T1_Decay_Offset_Check"] = grid_checking.fig

    grid_checking = QubitGrid(ds, [q.grid_location for q in qubits], size=6)
    for ax, qubit in grid_iter(grid_checking):
        q_name = qubit['qubit']
        mean_collection = []
        std_collection = []
        raw_collection = []
        for a_t1_fit_result in T1_fits[q_name]:
            raws = [] 
            for repeat in range(len(a_t1_fit_result)):
                decay = a_t1_fit_result[repeat].sel(fit_vals="decay")
                decay_res = a_t1_fit_result[repeat].sel(fit_vals="decay_decay")
                tau = -1 / a_t1_fit_result[repeat].sel(fit_vals="decay")
                tau_error_pct = -1 * (np.sqrt(decay_res) / decay)
                raws.append(100*tau_error_pct)

            mean_collection.append(np.mean(raws))
            std_collection.append(np.std(raws))
            raw_collection.append(raws)       

        ax.errorbar(lens, mean_collection, yerr=std_collection, alpha=0.5, label="fit error", fmt='o', zorder=2)
        x_raws = [x_val for x_val, raw_vals in zip(lens, raw_collection) for _ in raw_vals]
        y_raws = [y for raw_vals in raw_collection for y in raw_vals]

        # 繪製紅點散佈圖
        ax.scatter(x_raws, y_raws, color='red', s=15, alpha=0.2, label='Raw Data', zorder=3)
        ax.set_title(f"{q_name} - T1 Fitting Confidence")
        ax.set_xlabel("Integration Time [ns]")  
        ax.set_ylabel("Fitting Error ($\epsilon / \mu$) [%]", fontsize=12)
        ax.legend()
        ax.grid()
        
    grid_checking.fig.suptitle("T1 Fitting Check")
    plt.show()
    node.results[f"T1_Fitting_error_Check"] = grid_checking.fig
    
for state in [0,1]:
    grid_checking = QubitGrid(ds, [q.grid_location for q in qubits], size=6)
    for ax, qubit in grid_iter(grid_checking):
        q_name = qubit['qubit']
        mean_collection = []
        std_collection = []
        raw_collection = []
        for ms in models[q_name]:
            raws = []
            for m in ms:
                max_outlier = 100*m.analysis_result['outlier_probability'][state]
                raws.append(max_outlier)
            mean_collection.append(np.mean(raws))
            std_collection.append(np.std(raws))
            raw_collection.append(raws) 

        

        ax.errorbar(lens, mean_collection, yerr=std_collection, label="Outlier", fmt='o', zorder=2)
        x_raws = [x_val for x_val, raw_vals in zip(lens, raw_collection) for _ in raw_vals]
        y_raws = [y for raw_vals in raw_collection for y in raw_vals]

        # 繪製紅點散佈圖
        ax.scatter(x_raws, y_raws, color='red', s=15, alpha=0.2, label='Raw Data', zorder=3)

        mean_collection = []
        std_collection = []
        raw_collection = []
        for ms in models[q_name]:
            raws = []
            for m in ms:
                population_in_bridge = get_gap_population(m)
                raws.append(100*population_in_bridge[state])
            mean_collection.append(np.mean(raws))
            std_collection.append(np.std(raws))
            raw_collection.append(raws) 

        ax.errorbar(lens, mean_collection, yerr=std_collection, label="Outliers in the bridge", fmt='o', zorder=2, color='brown')
        x_raws = [x_val for x_val, raw_vals in zip(lens, raw_collection) for _ in raw_vals]
        y_raws = [y for raw_vals in raw_collection for y in raw_vals]

        # 繪製紅點散佈圖
        ax.scatter(x_raws, y_raws, color='orange', s=15, alpha=0.2, zorder=3)
        



        ax.set_title(f"{q_name}")
        ax.set_xlabel("Integration Time [ns]")  
        ax.set_ylabel("Outlier Population [%]", fontsize=12)
        ax.legend()
        ax.grid()
        
    grid_checking.fig.suptitle(f"Prepared {state} Outlier Population Check")
    plt.show()
    node.results[f"T1_Outlier_Check_prepared_{state}"] = grid_checking.fig


# %% {Save_results}
node.results["initial_parameters"] = node.parameters.model_dump()
node.save()      


# %%
