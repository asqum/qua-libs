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
from scipy.interpolate import UnivariateSpline
def auto_optimize_readout_voltage(ds, qubit_name, plot=True):
    """
    自動估算平滑係數並尋找最佳讀取電壓。
    """
    # 1. 取得數據並轉為 dB
    snr_data = ds.snr.sel(qubit=qubit_name).mean(dim="N")
    snr_db = 20 * np.log10(snr_data.where(snr_data > 0, 1e-6))
    
    v_data = ds.voltage.values
    snr_values = snr_db.values
    m = len(v_data)

    # 2. 自動估算雜訊 (Sigma)
    # 透過計算相鄰點的差值來估算高頻雜訊
    sigma = np.std(np.diff(snr_values)) 
    
    # 3. 根據經驗公式設定自動 s
    # s = m * sigma^2 是統計學上的經典建議值
    auto_s = m * (sigma**2)
    
    # 限制 s 的範圍，避免過度平滑或完全沒平滑
    auto_s = np.clip(auto_s, 0.5, 20) 

    # 4. 進行擬合
    spline = UnivariateSpline(v_data, snr_values, s=auto_s)
    v_fine = np.linspace(v_data.min(), v_data.max(), 1000)
    snr_smooth = spline(v_fine)
    
    best_v = v_fine[np.argmax(snr_smooth)]
    max_snr_db = np.max(snr_smooth)

    # 5. 繪圖
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(v_data, snr_values, 'b.', label='Raw Data', alpha=0.3)
        plt.plot(v_fine, snr_smooth, 'r-', lw=2, label=f'Auto Spline (s={auto_s:.2f})')
        plt.axvline(best_v, color='green', linestyle='--', 
                   label=f'Best V: {best_v:.3f}V\nSNR: {max_snr_db:.2f} dB')
        plt.title(f"{qubit_name}: Auto-optimized SNR")
        plt.xlabel("Coupler Voltage (V)")
        plt.ylabel("SNR (dB)")
        plt.legend()
        plt.show()

    return float(best_v)

def verify_snr(ds, qubit_name, voltage_val):
        # 1. 提取特定 qubit 與電壓下的數據 (state: 0=Ground, 1=Excited)
        data = ds.sel(qubit=qubit_name).sel(voltage=voltage_val, method="nearest")
        
        Ig = data.I.sel(state=0)
        Qg = data.Q.sel(state=0)
        Ie = data.I.sel(state=1)
        Qe = data.Q.sel(state=1)

        # 2. 計算平均值 (Center of the clouds)
        mu_g = np.array([Ig.mean().item(), Qg.mean().item()])
        mu_e = np.array([Ie.mean().item(), Qe.mean().item()])
        
        # 3. 計算距離平方 (Signal Power)
        dist_sq = np.sum((mu_e - mu_g)**2)
        
        # 4. 計算變異數 (Noise Power)
        # Var(Total) = Var(I) + Var(Q)
        var_g = Ig.var().item() + Qg.var().item()
        var_e = Ie.var().item() + Qe.var().item()
        var_max = max(var_g, var_e)
        
        # 5. 計算 SNR 與 SNR dB
        snr_python = np.sqrt(dist_sq / var_max)
        snr_db_python = 20 * np.log10(snr_python)
        
        # 6. 讀取 QUA 計算的 SNR dB (取 N 次實驗的平均值)
        snr_db_qua = ds.snr_db.sel(qubit=qubit_name).sel(voltage=voltage_val, method="nearest").mean(dim='N').item()

        # absolute error 
        abs_error = round(abs(snr_db_python-snr_db_qua),2)
        
        return abs_error


def SNR_realtime_calc(sum_G:dict, sum_E:dict, sum_G_sq:list, sum_E_sq:list, total_counts:int, snr_st:list, ret:bool=False):
    """ 
    Calculate SNR and return the dict contained qubit individual SNR and over all qubits averaged SNR.
    """
    
    num_qubits = len(sum_G_sq)
    inv_n = 1.0 / total_counts
    inv_num_qubits = 1.0 / num_qubits

   
    n = declare(int)
    snr_final = declare(fixed)
    avg_snr = declare(fixed, value=0.0)
    
    
    each_snr = [declare(fixed, value=0.0) for _ in range(num_qubits)]

    
    m_Ig, m_Qg = declare(fixed), declare(fixed)
    m_Ie, m_Qe = declare(fixed), declare(fixed)
    m_sq_g, m_sq_e = declare(fixed), declare(fixed)
    dist_sq = declare(fixed)
    var_g, var_e, var_max = declare(fixed), declare(fixed), declare(fixed)
    
    snr_db_guard = 0.03 
    
    # --- Loop over qubits ---
    for i in range(num_qubits):
        
        # Reset calculation variables not needed, strictly overwritten by assign below

        # 1. Calculate Means
        assign(m_Ig, sum_G['I'][i] * inv_n)
        assign(m_Qg, sum_G['Q'][i] * inv_n)
        assign(m_Ie, sum_E['I'][i] * inv_n)
        assign(m_Qe, sum_E['Q'][i] * inv_n)
        assign(m_sq_g, sum_G_sq[i] * inv_n)
        assign(m_sq_e, sum_E_sq[i] * inv_n)

        # 2. Dist^2
        assign(dist_sq, (m_Ie - m_Ig)*(m_Ie - m_Ig) + (m_Qe - m_Qg)*(m_Qe - m_Qg))
        
        # 3. Variance
        assign(var_g, m_sq_g - (m_Ig*m_Ig + m_Qg*m_Qg))
        assign(var_e, m_sq_e - (m_Ie*m_Ie + m_Qe*m_Qe))

        # Protection
        with if_(var_g < 1e-9): assign(var_g, 1e-9)
        with if_(var_e < 1e-9): assign(var_e, 1e-9)
        
        # 4. SNR Calc
        with if_(var_e > var_g):
            assign(var_max, var_e)
        with else_():
            assign(var_max, var_g)
        
        with if_(var_max > 1e-9):
            assign(snr_final, Math.sqrt(Math.div(dist_sq, var_max)))
            with if_(snr_final < snr_db_guard):
                assign(snr_final, snr_db_guard)
        with else_():
            with if_(dist_sq > 1e-9):
                assign(snr_final, 7.99)
            with else_():
                assign(snr_final, snr_db_guard)

        # 5. Save Stream (match stream processing structure)
        with for_(n, 0, n < total_counts, n + 1):
            save(snr_final, snr_st[i])
        
        # 6. Accumulate Average SNR 
        assign(each_snr[i], snr_final)
        # AVG
        assign(avg_snr, avg_snr + snr_final * inv_num_qubits)
    
    if ret:
        return {"individual": each_snr, "averaged": avg_snr}, snr_st
    else:
        return None

def SNR_observer(qubits:List[Transmon], qubit_pairs:List[TransmonPair], dcs:np.ndarray, n_runs:int=4096, z_rising_time_ns:int=1000, reset_type:Literal['thermal', 'active']='thermal', simulation:bool = False):
    ro_time = 0
    xy_time = 0
    num_qubits = len(qubits)
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)
    snr_st = [declare_stream() for _ in range(len(qubits))]
    z_rising_time = z_rising_time_ns//4
    dc = declare(fixed)
    
    # flux crosstalk compensate coef
    fcc_flux_amp = {q.name: declare(fixed, value=0.0) for q in qubits}
    
    # averaging and square sum accumulator
    sum_Ig = [declare(fixed, value=0.0) for _ in range(num_qubits)]
    sum_Qg = [declare(fixed, value=0.0) for _ in range(num_qubits)]
    sum_Ie = [declare(fixed, value=0.0) for _ in range(num_qubits)]
    sum_Qe = [declare(fixed, value=0.0) for _ in range(num_qubits)]
    sum_sq_g = [declare(fixed, value=0.0) for _ in range(num_qubits)]
    sum_sq_e = [declare(fixed, value=0.0) for _ in range(num_qubits)]
    
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
            assign(sum_Ig[i], 0.0)
            assign(sum_Qg[i], 0.0)
            assign(sum_Ie[i], 0.0)
            assign(sum_Qe[i], 0.0)
            assign(sum_sq_g[i], 0.0)
            assign(sum_sq_e[i], 0.0)
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

                assign(sum_Ig[i], sum_Ig[i] + I_g[i])
                assign(sum_Qg[i], sum_Qg[i] + Q_g[i])
                assign(sum_sq_g[i], sum_sq_g[i] + (I_g[i] * I_g[i] + Q_g[i] * Q_g[i]))

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

                assign(sum_Ie[i], sum_Ie[i] + I_e[i])
                assign(sum_Qe[i], sum_Qe[i] + Q_e[i])
                assign(sum_sq_e[i], sum_sq_e[i] + (I_e[i]*I_e[i] + Q_e[i]*Q_e[i]))

                save(I_e[i], I_e_st[i])
                save(Q_e[i], Q_e_st[i])

            align()
            
        SNR_realtime_calc(sum_G={"I":sum_Ig,"Q":sum_Qg}, sum_E={"I":sum_Ie,"Q":sum_Qe}, sum_G_sq=sum_sq_g, sum_E_sq=sum_sq_e, total_counts=n_runs, snr_st=snr_st)
        

    streaming = {"Ig":I_g_st, "Qg":Q_g_st, "Ie":I_e_st, "Qe":Q_e_st, "SNR":snr_st, "n":n_st}
    
    return streaming
    




# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q1', 'q2']
    num_runs: int = 4096
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    c_min_v: float = -0.2
    c_max_v: float = 0.2
    v_nums: int = 100
    outliers_threshold: float = 0.98
    plot_raw: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    z_rising_time_ns:int = 1000


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
    
    streaming = SNR_observer(qubits, qubit_pairs, dcs, n_runs, node.parameters.z_rising_time_ns, reset_type, node.parameters.simulate)  

    with stream_processing():
        streaming['n'].save("n")
        for i in range(num_qubits):
            streaming['Ig'][i].buffer(n_runs).buffer(len(dcs)).save(f"I_g{i + 1}")
            streaming['Qg'][i].buffer(n_runs).buffer(len(dcs)).save(f"Q_g{i + 1}")
            streaming['Ie'][i].buffer(n_runs).buffer(len(dcs)).save(f"I_e{i + 1}")
            streaming['Qe'][i].buffer(n_runs).buffer(len(dcs)).save(f"Q_e{i + 1}")

            # SNR
            streaming['SNR'][i].buffer(n_runs).buffer(len(dcs)).save(f"snr{i + 1}")


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
    for i in range(num_qubits):
        data_list.extend([f"I_g{i+1}", f"Q_g{i+1}", f"I_e{i+1}", f"Q_e{i+1}", f"snr{i+1}"])
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
            {"N": np.linspace(1, n_runs, n_runs), "voltage": dcs}
        )

        ds_rearranged = xr.Dataset()

        # merge I, Q
        ds_rearranged["I"] = xr.concat([ds.I_g, ds.I_e], dim="state").assign_coords(state=[0, 1])
        ds_rearranged["Q"] = xr.concat([ds.Q_g, ds.Q_e], dim="state").assign_coords(state=[0, 1])

    
        ds = ds.assign_coords({"voltage": (["qubit", "voltage"], np.array([dcs * 1 for q in qubits]))})

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

    #%% Plot
    ## Plot SNR
    best_amp = {}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    ds["snr_db"] = 20 * np.log10(ds.snr.where(ds.snr > 0, 1e-6))
    for ax, qubit in grid_iter(grid):
        node.results["results"][qubit["qubit"]] = {}
        optimal_voltage = auto_optimize_readout_voltage(ds, qubit["qubit"], False)
        node.results["results"][qubit["qubit"]]["best_coupler_voltage"] = optimal_voltage
        ds.snr_db.sel(qubit=qubit["qubit"]).mean(dim="N").plot(ax=ax)
        ax.axvline(optimal_voltage, color="k", linestyle="dashed")
        ax.set_xlabel("Coupler flux pulse amplitude (V)")
        ax.set_ylabel("SNR (dB)")
        ax.set_title(f'{qubit["qubit"]}, coupler at {round(optimal_voltage, 3)} is good')
        ax.grid()
    grid.fig.suptitle("Coupler flux pulse vs Readout SNR")
    
    plt.tight_layout()
    plt.show()
    node.results["figs"][f"Figure_SNR"] = grid.fig

    # %%
    ## Plot Raw 
    
    #%%
    if node.parameters.plot_raw:
    
        # 1. 預先計算所有數據的全局最大/最小值，以固定座標軸 (避免畫面跳動)
        # 這樣才能看出 Cloud 真的在移動，而不是座標軸在伸縮
        all_I = ds.I.values.flatten()
        all_Q = ds.Q.values.flatten()
        I_min, I_max = all_I.min(), all_I.max()
        Q_min, Q_max = all_Q.min(), all_Q.max()
        
        # 稍微放寬一點邊界 (Buffer)
        margin_I = (I_max - I_min) * 0.1
        margin_Q = (Q_max - Q_min) * 0.1
        lim_I = (I_min - margin_I, I_max + margin_I)
        lim_Q = (Q_min - margin_Q, Q_max + margin_Q)

        # 2. 建立畫布 (只需要一列，因為我們會隨著時間更新這列)
        fig, axes = plt.subplots(
            ncols=num_qubits, 
            nrows=1, 
            figsize=(5 * num_qubits, 6), 
            squeeze=False # 確保 axes 總是 2D array，方便索引
        )
        axes = axes[0] # 取出第一列

        # 3. 定義更新函數 (每一幀要做的事情)
        def update(frame_idx):
            voltage = float(ds.voltage[frame_idx])
            
            for i, q in enumerate(list(qubits)):
                ax = axes[i]
                ax.clear() # 清除上一幀的圖
                
                # 選取當前電壓點的數據
                ds_q = ds.sel(qubit=q.name).isel(voltage=frame_idx)
                
                # 繪製 Ground (state=0) 和 Excited (state=1)
                # 使用 scatter 或是 plot('.', ...)
                ax.plot(ds_q.I.sel(state=0), ds_q.Q.sel(state=0), ".", 
                        color="tab:blue", alpha=0.3, label="Ground", markersize=2)
                ax.plot(ds_q.I.sel(state=1), ds_q.Q.sel(state=1), ".", 
                        color="tab:orange", alpha=0.3, label="Excited", markersize=2)
                
                # 計算該點的 SNR 誤差 (呼叫你原本的 verify_snr)
                err = verify_snr(ds, q.name, voltage)
                qua_snr = ds.snr_db.sel(qubit=q.name).sel(voltage=voltage, method="nearest").mean(dim='N').item()
                # 設定標題與標籤
                ax.set_title(f"{q.name}\nCoupler Voltage: {voltage:.4f} V\nSNR: {qua_snr:.2f} dB, error: {err:.3f} dB")
                ax.set_xlabel("I")
                if i == 0: ax.set_ylabel("Q") # 只在最左邊顯示 Y 軸標籤
                
                # 關鍵：固定座標軸範圍
                ax.set_xlim(lim_I)
                ax.set_ylim(lim_Q)
    
                # ax.axis("equal") # 如果希望比例尺 1:1 可開啟，但可能會破壞固定範圍
                ax.legend(loc="upper right", markerscale=3)
                ax.grid(True, alpha=0.3)
                

        # 4. 建立動畫
        # frames=len(ds.voltage) 代表總共有多少個電壓點
        # interval=200 代表每一幀停留 200ms (即每秒 5 張圖)
        ani = animation.FuncAnimation(fig, update, frames=len(ds.voltage), interval=1000)

        # 5. 存檔 (儲存為 GIF 或 MP4)
    
        
        # 存成 GIF (通常最通用，不需要額外安裝 ffmpeg)
        # ani.save(save_path, writer='pillow', fps=5)
        
        # 如果你有安裝 ffmpeg，也可以存成 mp4 (檔案較小，畫質較好)
        # ani.save("coupler_flux_sweep.mp4", writer='ffmpeg', fps=5)
        
        plt.close() # 關閉畫布，避免在 Jupyter Notebook 中重複顯示靜態圖
        
        # 將動畫物件存入 node results (雖然動畫很難直接序列化存入，但可以存路徑)
       


    # Plot Centroid vs coupler bias
    plt.figure(figsize=(8, 6))
    for q in ds.qubit.values:
        # 先對 N 維度取平均，得到每顆球的中心點 (Centroid)
        avg_I = ds.I.sel(qubit=q, state=0).mean(dim="N")
        avg_Q = ds.Q.sel(qubit=q, state=0).mean(dim="N")
        
        plt.plot(avg_I, avg_Q, '.-', label=f"Qubit {q} path")
        # 標註起點 (第一個 DC 點)
        plt.annotate("Start", (avg_I[0], avg_Q[0]))

    plt.title("IQ Centroid Drift with DC Flux")
    plt.xlabel("Average I"); plt.ylabel("Average Q")
    plt.legend()
    plt.axis('equal')
    plt.show()
    

    dSNR = []
    max_dSNR = {"q":"", "voltage":0, "dsnr":-1000}
    import pandas as pd
    for voltage in ds.voltage:
        for q in list(qubits):
            dsnr = verify_snr(ds, q.name, float(voltage))
            dSNR.append(dsnr)

            if dsnr >= max_dSNR['dsnr']:
                max_dSNR['q'] = q.name
                max_dSNR['voltage'] = float(voltage)
                max_dSNR['dsnr'] = dsnr

    ds_q = ds.sel(qubit=max_dSNR['q'], voltage=max_dSNR['voltage'])

    plt.plot(ds_q.I.sel(state=0), ds_q.Q.sel(state=0), ".", alpha=0.2, label="Ground", markersize=2)
    plt.plot(ds_q.I.sel(state=1), ds_q.Q.sel(state=1), ".", alpha=0.2, label="Excited", markersize=2)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title(f"{max_dSNR['q']}, C @ {round(max_dSNR['voltage'],4)}, QUA-SNR= {round(ds.snr_db.sel(qubit=max_dSNR['q']).sel(voltage=max_dSNR['voltage'], method='nearest').mean(dim='N').item(), 2)} dB, SNR error (Max)= { max_dSNR['dsnr']} dB")
    plt.axis("equal")
    plt.show()
    node.results["figs"][f"Max_SNR_calculation_error"] = plt.gcf()
    se = pd.Series(np.array(dSNR))
    print(" *** All qubits |QUA_SNR - Post_SNR| statistics, unit: dB *** ")
    print(se.describe())


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


# %%
if node.parameters.plot_raw: 
    from qualibrate_config.resolvers import get_qualibrate_config_path, get_qualibrate_config
    from quam_libs.compat import get_node_dir_path
    import os
    qs = get_qualibrate_config(get_qualibrate_config_path())
    base_path = qs.storage.location

    node_dir = get_node_dir_path(node.snapshot_idx, base_path)
   
    ani.save(os.path.join(node_dir, f"coupler_flux_sweep.gif"), writer='pillow', fps=5)



# 假設 node.parameters.plot_raw 為 True

# %%
if node.parameters.plot_raw:
    wnat_flux_locs = [0] # what flux amplitude want to plot
    fig, axes = plt.subplots(
        ncols=num_qubits,
        nrows=len(wnat_flux_locs),
        sharex=False,
        sharey=False,
        squeeze=False,
        figsize=(5 * num_qubits, 5 * len(wnat_flux_locs)),
    )
    for amplitude, ax1 in zip(wnat_flux_locs, axes):
        for q, ax2 in zip(list(qubits), ax1):
            ds_q = ds.sel(qubit=q.name).sel(voltage=amplitude, method='nearest')
            ax2.plot(ds_q.I.sel(state=0), ds_q.Q.sel(state=0), ".", alpha=0.2, label="Ground", markersize=2)
            ax2.plot(ds_q.I.sel(state=1), ds_q.Q.sel(state=1), ".", alpha=0.2, label="Excited", markersize=2)
            ax2.set_xlabel("I")
            ax2.set_ylabel("Q")
            ax2.set_title(f"{q.name}\nCoupler Voltage: {amplitude:.4f}")
            ax2.axis("equal")
            ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# %%
