# %%
"""
Optimization for the aSWAP pulse duration. It will try different pluse length for aSWAP and perfrom a power Rabi experiment for each length. The best pulse duration is the one that gives the highest state probability (amplitude) for the Rabi oscillation.

Prerequisites:
    - The power rabi for target coupler.

"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state_coupler
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
from time import time
import xarray as xr

# %% {Node_parameters}
class Parameters(NodeParameters):

    coupler: str = 'coupler_q4_q5'
    num_averages: int = 500
    min_length_ns:int = 200
    max_length_ns:int = 300
    length_step_ns:int = 12
    min_amp_factor: float = 0.0 #0.001
    max_amp_factor: float = 1.79 #2.0
    amp_factor_step: float = 1.79/100 
    debug:bool = False
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    simulation_duration_ns: int = 500
    timeout: int = 100
    load_data_id: Optional[int] = None
    

node = QualibrationNode(name="04xx_aSWAP_DuraOptimize", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
coupler = [machine.qubit_pairs[node.parameters.coupler]] # currently supports 1 coupler a time only.
drive_q = [machine.qubits[coupler[0].extras["RD"]["driven_q"]]]
detector_q = [machine.qubits[coupler[0].extras["RD"]["readout_q"]]]

max_length_ns = detector_q[0].z.operations["aSWAP"].length if node.parameters.max_length_ns > detector_q[0].z.operations["aSWAP"].length else node.parameters.max_length_ns

# Change driving LO
drive_LO_original = {drive_q[0].name: drive_q[0].xy.opx_output.upconverter_frequency}
original_truncate_len = {detector_q[0].name: detector_q[0].z.operations['aSWAP'].truncate_len}
drive_q[0].xy.opx_output.upconverter_frequency = coupler[0].extras["RD"]["LO"]

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
N_pi = 1  # Number of applied Rabi pulses sweep
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = 'thermal' #node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
state_discrimination = True
operation = 'x180_cp'  # The qubit operation to play
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(
    node.parameters.min_amp_factor,
    node.parameters.max_amp_factor,
    node.parameters.amp_factor_step,
)

duras = np.arange(
    4*(node.parameters.min_length_ns//4),
    4*(max_length_ns//4),
    4*(node.parameters.length_step_ns//4),
)

if node.parameters.debug:
    duras = np.array([200]*10)

# Number of applied Rabi pulses sweep
if N_pi > 1:
    if operation in ["x180_cp"]:
        N_pi_vec = np.arange(1, N_pi, 2).astype("int")
    elif operation in ["x90_cp"]:
        N_pi_vec = np.arange(2, N_pi, 4).astype("int")
    else:
        raise ValueError(f"Unrecognized operation {operation}.")
else:
    N_pi_vec = np.linspace(1, N_pi, N_pi).astype("int")[::2]


with program() as power_rabi:
    _, _, _, _, n, n_st = qua_declaration(num_qubits=len(detector_q))
    
    state = [declare(int) for _ in range(len(detector_q))]
    state_stream = [declare_stream() for _ in range(len(detector_q))]
    
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    count = declare(int)  # QUA variable for counting the qubit pulses

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
        print(detector_q[0].z.operations['aSWAP'].truncate_len)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(a, amps)):
                # Initialize the qubits
                
                if not node.parameters.simulate:
                    if qubit.thermalization_time//5 > coupler[0].extras['T1']*1e9:
                        wait(qubit.thermalization_time * u.ns)
                    else:
                        wait(5*coupler[0].extras['T1']*1e9 * u.ns)

                # for a better RO fidelity
                # active_reset(detector_q[i], "readout")
                # align()

                # Loop for error amplification (perform many qubit pulses)
                align()
                qubit.xy.play(operation, amplitude_scale=a)

                readout_state_coupler(detector_q[i], state[i], method='aswap')
                save(state[i], state_stream[i])


    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(drive_q):
                state_stream[i].buffer(len(amps)).average().save(f"state{i + 1}")
            

# %% {Simulate_or_execute}
if not node.parameters.load_data_id:
    dss = []
    start = time()
    for i, truncate_dura in enumerate(duras):
        
        # detector_q[0].z.operations['aSWAP'].truncate_len = truncate_dura
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
            d = fetch_results_as_xarray(job.result_handles, drive_q, {"amp": amps})
            # 1. 確保維度乾淨 (移除可能存在的純量 qubit 標籤，避免衝突)
            if 'qubit' in d.coords and d.coords['qubit'].ndim == 0:
                d = d.drop_vars('qubit')

            # 2. 強制建立 duration 維度
            d = d.expand_dims(duration=[i])

            # 3. 重新建立 qubit 維度座標 (確保它是 array 而不是 scalar)
            # 假設 drive_q 裡面有多個 qubit 名稱
            q_names = [q.name for q in drive_q]
            d = d.assign_coords(qubit=q_names)

            # 4. 現在可以安全地定義 abs_amp 了
            abs_amp_data = np.array([q.xy.operations[operation].amplitude * amps for q in drive_q])
            d = d.assign_coords(
                abs_amp=(["qubit", "amp"], abs_amp_data)
            )
            dss.append(d)
    ds = xr.concat(dss, dim='duration')
    node.results = {"ds": ds}
    end = time()
else:
    ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
    # Add the dataset to the node
    # 
# %%
print(f"{round(end-start,1)} sec.")
# 隨機挑兩個 duration，檢查它們的內容是否真的不同
print(ds.state.sel(duration=0).values[:5])
print(ds.state.sel(duration=1).values[:5])


# %%
fit_res = fit_oscillation(ds.state, dim="amp")
def plot_fitting_verification(ds, fit_res, q_name):
    # --- 關鍵修正 1: 強制將 fit_res 的座標順序對齊 ds ---
    # 這樣可以確保兩者的 duration 順序完全一致
    fit_res = fit_res.reindex_like(ds.state, method=None)
    
    durations = ds.duration.values
    num_plots = len(durations)
    
    cols = 5
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), constrained_layout=True)
    axes = axes.flatten()
    
    x_amp = ds.amp.values
    
    for i, d in enumerate(durations):
        ax = axes[i]
        
        # 取得數據
        y_data = ds.state.sel(qubit=q_name, duration=d).values
        
        # --- 關鍵修正 2: 使用 .item() 確保提取的是數值而非 DataArray ---
        # 這樣可以避免計算 y_fit 時發生維度廣播錯誤
        params = fit_res.sel(qubit=q_name, duration=d)
        
        try:
            # 提取擬合參數並轉為 float
            a = float(params.sel(fit_vals="a"))
            f = float(params.sel(fit_vals="f"))
            phi = float(params.sel(fit_vals="phi"))
            offset = float(params.sel(fit_vals="offset"))
            
            # 3. 計算擬合曲線
            x_fit = np.linspace(x_amp.min(), x_amp.max(), 200)
            y_fit = oscillation(x_fit, a, f, phi, offset)
            
            # 4. 繪圖
            ax.scatter(x_amp, y_data, s=10, label="Data", color="black", alpha=0.5)
            ax.plot(x_fit, y_fit, label="Fit", color="red", lw=1.5)
            
        except (ValueError, TypeError):
            # 如果該點擬合失敗 (NaN)，至少把原始資料畫出來
            ax.scatter(x_amp, y_data, s=10, color="gray", alpha=0.3)
            ax.set_facecolor('#fff0f0') # 失敗的格線塗成淡紅色

        ax.set_title(f"Dur: {d:.2f}")
        if i == 0: ax.legend()
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle(f"Fitting Verification for Qubit: {q_name}", fontsize=16)
    plt.show()

# 使用範例
# 假設你的擬合結果儲存在 fit_res
plot_fitting_verification(ds, fit_res, ds.qubit.values[0])


# %% {Data_analysis & plot
from matplotlib.ticker import AutoMinorLocator
if not node.parameters.simulate:
    
    fit_results = {}
    # 執行擬合
    fit_res_raw = fit_oscillation(ds.state, dim="amp")
    
    # --- 【修正點 1】強制對齊座標順序 ---
    # 確保 fit_res 的 duration 順序跟 ds.state 完全一模一樣
    fit_res = fit_res_raw.reindex_like(ds.state)
    
    fit_results[coupler[0].name] = fit_res

    def plot_heatmap_with_snr(ds, fit_res, q_name):
        da_q = ds.state.sel(qubit=q_name)
        durations = ds.duration.values
        amps = ds.amp.values
        
        snr_db_list = []
        
        for d in durations:
            y_data = da_q.sel(duration=d).values
            p = fit_res.sel(qubit=q_name, duration=d)
            
            # --- 【修正點 2】使用 float() 提取純量，避免 metadata 干擾 ---
            try:
                a = float(p.sel(fit_vals="a"))
                f = float(p.sel(fit_vals="f"))
                phi = float(p.sel(fit_vals="phi"))
                offset = float(p.sel(fit_vals="offset"))
                
                y_fit = oscillation(amps, a, f, phi, offset)
                residual = y_data - y_fit
                noise = np.std(residual)
                
                if noise > 1e-10 and a > 0:
                    snr_db = 20 * np.log10(a / noise)
                else:
                    snr_db = 0
            except:
                snr_db = 0
                
            snr_db_list.append(snr_db)

        fig, ax1 = plt.subplots(figsize=(10, 8))
    
        # 繪製 Heatmap
        im = ax1.pcolormesh(amps, durations, da_q.values, cmap="RdBu_r", shading='auto', alpha=0.9)
        ax1.set_xlabel("pi Amplitude (amp)", color="black")
        ax1.set_ylabel("Truncated duration (ns)") # 修正標籤名稱
        
        ax1.yaxis.set_minor_locator(AutoMinorLocator(n=5))
        # 將 Grid 設淡一點，不然會遮住數據
        ax1.grid(True, axis='y', color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        # 建立上方的 SNR X 軸
        ax2 = ax1.twiny() 
        ax2.scatter(snr_db_list, durations, color="green", marker='*', s=40, label="SNR (dB)")
        ax2.set_xlabel("Fitting SNR (dB)", color="green", labelpad=10)
        ax2.grid(True, axis='x', color='green', linestyle=':', linewidth=1, alpha=0.4)
        ax2.tick_params(axis='x', labelcolor='green')
        
        # --- 【修正點 3】更穩健的 X 軸範圍 ---
        max_snr = np.max(snr_db_list) if len(snr_db_list) > 0 else 10
        ax2.set_xlim(0, max(max_snr * 1.1, 10)) # 至少給 10dB 的範圍
        
        fig.colorbar(im, ax=ax1, label="Population", pad=0.1)
        plt.title(f"aSWAP duration mapping - {coupler[0].name}\n", pad=25)
        
        return fig

    # 呼叫函式
    fig = plot_heatmap_with_snr(ds, fit_res, ds.qubit.values[0])
    node.results["figure"] = fig
    plt.show()
    # %% {Save_results}
    
    if node.parameters.load_data_id is None:

        with node.record_state_updates():
            for q in drive_q:
                q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
            for q in detector_q:
                q.z.operations['aSWAP'].truncate_len = original_truncate_len[q.name]
        
        node.outcomes = {q.name: "successful" for q in drive_q}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()

# %%
print(dss)
# %%
for ds in dss:
    plt.plot(ds.state.values[0][0])
    plt.show()
# %%
