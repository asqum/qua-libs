# %%
"""
C to Q Flux Crosstalk Compensation (FCC) measurement.
Measure the flux crosstalk from coupler to its neighboring qubits. like coupler_q1_q2 to q1 or q2, etc.
At this moment, only measure one coupler in a single time
The measurement process inherited from 64a

Noticed that if you want to check the target qubit defined in coupler, we need to bias it to make its transition frequeny get lower than the other. This bias voltage is named 'target_q_bias' in node.parameters and we need to manually search it at this moment.

* Once the flux crosstalk is fit, it will be saved into a dict in this coupler's extra with a name FCC and its key will be the target qubit.
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state_gef, readout_state, active_reset_simple
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter
from sklearn.linear_model import RANSACRegressor
# %%
from scipy.stats import linregress
def fit_single_slice_with_error(ds_slice, qubit_label, y_limit=None, sigma=2.0):
    """
    對單一 duration 切片進行 RANSAC 擬合，並計算斜率誤差。
    """
    # --- A. 數據整理 (與之前相同) ---
    # 確保維度是 (Coupler, Qubit)
    ds_slice = ds_slice.transpose("flux_coupler", "flux_qubit")
    
    # 處理 y_limit
    if y_limit is not None:
        ds_slice = ds_slice.where(ds_slice.flux_coupler * 1e3 <= y_limit, drop=True)
        
    flux_qubit_mV = ds_slice.flux_qubit.values * 1e3
    flux_coupler_mV = ds_slice.flux_coupler.values * 1e3
    signal_values = ds_slice.values

    # --- B. 找特徵線 (與之前相同) ---
    background_level = np.median(signal_values)
    is_finding_min = (background_level - np.min(signal_values)) > (np.max(signal_values) - background_level)
    
    smoothed_signal = gaussian_filter(signal_values, sigma=[0, sigma])
    
    valid_x, valid_y = [], []
    for i, y_val in enumerate(flux_coupler_mV):
        row_data = smoothed_signal[i, :]
        target_idx = np.argmin(row_data) if is_finding_min else np.argmax(row_data)
        valid_x.append(flux_qubit_mV[target_idx])
        valid_y.append(y_val)

    # 轉為 numpy array
    X = np.array(valid_y).reshape(-1, 1) # Coupler (Y軸當自變數)
    y = np.array(valid_x)                # Qubit (X軸當應變數)

    if len(X) < 5: # 點太少無法計算誤差
        return np.nan, np.nan

    # --- C. RANSAC 篩選 ---
    ransac = RANSACRegressor(random_state=42, residual_threshold=2.0)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    y_pred_all = ransac.predict(X)
        
        
    residuals = y - y_pred_all.ravel() # .ravel() 轉成 1D
        
    # 3. MAD (Median Absolute Deviation)
    # 這是 RANSAC 內部用來判斷雜訊的標準，這代表了數據的"整體頻寬"或"雜訊大小"
    # approch a sigma in normal distribution through * 1.4826 
    mad_error = np.median(np.abs(residuals - np.median(residuals))) * 1.4826


    # --- D. 關鍵步驟：使用 Scipy 計算標準誤差 ---
    # RANSAC 告訴我們哪些點是乾淨的，我們用這些乾淨的點來算統計誤差
    if np.sum(inlier_mask) < 3:
        return np.nan, np.nan

    # linregress(x, y) -> 注意這裡輸入都要是 1D array
    # 我們的 X 是 Coupler (自變數), y 是 Qubit Offset (應變數)
    slope_result = linregress(X[inlier_mask].ravel(), y[inlier_mask])
    
    x_std = np.std(X[inlier_mask]) # X軸的擴展範圍
    n_samples = len(X[inlier_mask]) # 點的數量
    
    if x_std == 0 or n_samples == 0:
        return np.nan, np.nan

    slope_uncertainty = mad_error / (x_std * np.sqrt(n_samples))

    # slope_result.slope = 斜率
    # slope_result.stderr = 斜率的標準誤差
    return slope_result.slope, slope_uncertainty


def analyze_crosstalk_vs_duration(ds, qubit_label, state='control', state_discriminator:bool=True, y_limit=None, sigma:float=2.0):
    
    # 1. 準備儲存容器
    durations = ds.duration.values
    slopes = []
    errors = []
    
    if state_discriminator:
        target_var = 'state_target' if state.lower() != 'control' else 'state_control'
    else:
        target_var = 'I_target' if state.lower() != 'control' else 'I_control'
    print(f"Starting analysis for {len(durations)} duration points...")

    # 2. 遍歷每一個 duration
    for d in durations:
        # 選取特定的 duration 和 qubit，這裡會降維成 2D (coupler, flux_qubit)
        ds_slice = ds[target_var].sel(qubit=qubit_label, duration=d)
        
        # 呼叫上面的核心運算
        slope, err = fit_single_slice_with_error(ds_slice, qubit_label, y_limit, sigma)
        
        slopes.append(slope)
        errors.append(err)
        
        # (選用) 顯示進度
        # print(f"Duration {d}: Slope={slope:.4f} +/- {err:.4f}")

    # 轉成 numpy array 方便繪圖
    slopes = np.array(slopes)
    errors = np.array(errors)

    # 3. 畫圖 (Duration vs Slope with Errorbar)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 使用 errorbar
    ax.errorbar(durations, slopes*100, yerr=errors*100, 
                fmt='-o',        # 線型 + 點
                color='royalblue',
                ecolor='tomato', # 誤差棒顏色
                capsize=5,       # 誤差棒帽子大小
                lw=2, markersize=6,
                label='Crosstalk')

    # 美化圖表
    ax.set_title(f"{ds.attrs['coupler_z_waveform']} Crosstalk vs. Duration ({qubit_label})")
    ax.set_xlabel("Duration [ns]") # 假設單位是 ns，請確認
    ax.set_ylabel("Crosstalk [%]")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig, durations, slopes, errors


def plot_all_durations_heatmap(ds, qubit_label, state='control', y_limit=None, sigma=2.0, cols=4):
    """
    將每個 Duration 的 Crosstalk Heatmap 與擬合線畫在同一張大圖上。
    
    參數:
    - cols: 一列要畫幾張圖 (預設 4 張)
    """
    
    durations = ds.duration.values
    n_plots = len(durations)
    
    # 1. 計算子圖排列 (Rows x Cols)
    rows = int(np.ceil(n_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3.5), constrained_layout=True)
    
    # 將 axes 展平成 1D 陣列，方便迴圈操作
    if n_plots > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    target_var = 'state_target' if state.lower() != 'control' else 'state_control'

    print(f"Plotting {n_plots} heatmaps for {qubit_label}...")

    # 2. 遍歷每一個 Duration
    for i, d in enumerate(durations):
        ax = axes_flat[i]
        
        # --- A. 數據提取 (Dimension Safe) ---
        # 這裡做跟之前一樣的切片與轉置
        ds_slice = ds[target_var].sel(qubit=qubit_label, duration=d)
        ds_slice = ds_slice.transpose("flux_coupler", "flux_qubit") # 強制 (Y, X)
        
        if y_limit is not None:
            ds_slice = ds_slice.where(ds_slice.flux_coupler * 1e3 <= y_limit, drop=True)

        flux_qubit_mV = ds_slice.flux_qubit.values * 1e3
        flux_coupler_mV = ds_slice.flux_coupler.values * 1e3
        signal_values = ds_slice.values

        # --- B. 快速擬合邏輯 (為了畫紅線) ---
        # 判斷找亮還是找暗
        background = np.median(signal_values)
        is_finding_min = (background - np.min(signal_values)) > (np.max(signal_values) - background)
        
        # 高斯平滑
        smoothed = gaussian_filter(signal_values, sigma=[0, sigma])
        
        # 找點
        valid_x, valid_y = [], []
        for r, y_val in enumerate(flux_coupler_mV):
            row_data = smoothed[r, :]
            idx = np.argmin(row_data) if is_finding_min else np.argmax(row_data)
            valid_x.append(flux_qubit_mV[idx])
            valid_y.append(y_val)
            
        X = np.array(valid_y).reshape(-1, 1) # Coupler
        y = np.array(valid_x)                # Qubit

        # --- C. 繪圖 ---
        # 1. 畫底圖 Heatmap
        pcm = ax.pcolormesh(flux_qubit_mV, flux_coupler_mV, signal_values, cmap='viridis', shading='auto')
        
        # 2. 執行 RANSAC 並畫線 (如果有足夠的點)
        slope_str = "Fit: N/A"
        line_color = 'gray'
        
        if len(X) > 5:
            try:
                ransac = RANSACRegressor(random_state=42, residual_threshold=2.0)
                ransac.fit(X, y)
                
                # 取得斜率與截距
                slope = ransac.estimator_.coef_[0]
                slope_str = f"Slope: {slope:.4f}"
                line_color = 'red'
                
                # 畫擬合線
                line_y = np.linspace(min(flux_coupler_mV), max(flux_coupler_mV), 100).reshape(-1, 1)
                line_x = ransac.predict(line_y)
                ax.plot(line_x, line_y, color='red', linestyle='--', lw=2, alpha=0.8)
                
                # 畫被選中的點 (Inliers) - 讓你確認它抓得準不準
                inlier_mask = ransac.inlier_mask_
                ax.scatter(y[inlier_mask], X[inlier_mask], s=2, c='lime', alpha=0.6)
                
            except Exception as e:
                slope_str = "Fit Failed"
        
        # --- D. 標示與美化 ---
        ax.set_title(f"Dur: {d:.1f} ns | {slope_str}")
        ax.set_xlabel("Qubit Flux [mV]")
        ax.set_ylabel("Coupler Flux [mV]")
        
        # 固定範圍，方便視覺比較
        ax.set_xlim(min(flux_qubit_mV), max(flux_qubit_mV))
        ax.set_ylim(min(flux_coupler_mV), max(flux_coupler_mV))

    # 3. 清理多餘的空白子圖
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.suptitle(f"Crosstalk Analysis per Duration - {qubit_label}", fontsize=14)
    plt.show()

def make_unique_multiples_of_4(arr:np.ndarray):
    """
    將 array 轉為 4 的倍數，並保證嚴格遞增（不重複）。
    策略：先四捨五入，若發生重疊或逆序，則強制推擠到下一個 4 的倍數。
    """
    # 1. 先進行基礎的四捨五入對齊
    # 這是為了讓數字盡可能接近原始分佈
    result = (np.round(arr / 4) * 4).astype(int)
    
    # 2. 迭代修正（解決重複問題）
    # 因為 linspace 是排序好的，我們只要確保 result[i] > result[i-1]
    for i in range(1, len(result)):
        if result[i] <= result[i-1]:
            # 如果當前數字 小於等於 前一個數字 (發生碰撞)
            # 就強制設定為 前一個數字 + 4
            result[i] = result[i-1] + 4
            
    return result


from matplotlib.animation import FuncAnimation

def animate_crosstalk_evolution(ds, qubit_label, state='control', state_discriminator:bool=True, y_limit=None, sigma=2.0, interval=200, save_path=None):
    """
    製作 Crosstalk Heatmap 隨 Duration 變化的動畫 (已修正維度錯誤)。
    """
    
    durations = ds.duration.values
    if state_discriminator:
        target_var = 'state_target' if state.lower() != 'control' else 'state_control'
    else:
        target_var = 'I_target' if state.lower() != 'control' else 'I_control'

    # --- 1. 準備工作與鎖定範圍 (修正處) ---
    # 先針對 qubit 和 state 進行切片，這樣維度就會剩下 (duration, flux_coupler, flux_qubit)
    ds_target = ds[target_var].sel(qubit=qubit_label)
    
    # 為了讓動畫顏色穩定，需找出全域的最大最小值
    global_min = ds_target.min().values
    global_max = ds_target.max().values
    
    # 預先取得座標軸範圍
    # 先取 duration=0，剩下的維度應該只有 (flux_coupler, flux_qubit)
    # 使用 transpose 強制確保順序為 Y(Coupler), X(Qubit)
    ds_temp = ds_target.isel(duration=0).transpose("flux_coupler", "flux_qubit")
    
    if y_limit is not None:
        ds_temp = ds_temp.where(ds_temp.flux_coupler * 1e3 <= y_limit, drop=True)
        
    flux_qubit_mV = ds_temp.flux_qubit.values * 1e3
    flux_coupler_mV = ds_temp.flux_coupler.values * 1e3
    xlims = (min(flux_qubit_mV), max(flux_qubit_mV))
    ylims = (min(flux_coupler_mV), max(flux_coupler_mV))
    line_y_range = np.linspace(ylims[0], ylims[1], 100).reshape(-1, 1)

    # --- 2. 初始化畫布 ---
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # 畫初始空圖
    pcm = ax.pcolormesh(flux_qubit_mV, flux_coupler_mV, np.zeros_like(ds_temp.values), 
                        cmap='viridis', shading='auto', vmin=global_min, vmax=global_max)
    cbar = fig.colorbar(pcm, ax=ax, label='Signal (a.u.)')
    ax.set_aspect('auto')

    print(f"Generating animation for {qubit_label}: {len(durations)} frames...")

    # --- 3. 動畫更新核心函數 ---
    def update(frame_idx):
        d = durations[frame_idx]
        ax.cla() # 清除當前 Axes

        # A. 數據切片 (這裡已經不需要再 sel qubit 了，因為 ds_target 已經切過了)
        # 只要切 duration 即可
        ds_slice = ds_target.sel(duration=d)
        ds_slice = ds_slice.transpose("flux_coupler", "flux_qubit") # 強制 (Y, X)
        
        if y_limit is not None:
            ds_slice = ds_slice.where(ds_slice.flux_coupler * 1e3 <= y_limit, drop=True)

        current_signal = ds_slice.values

        # B. 擬合邏輯
        background = np.median(current_signal)
        # 簡單判斷對比度是否足夠 (避免全雜訊時擬合)
        data_range = np.max(current_signal) - np.min(current_signal)
        is_finding_min = (background - np.min(current_signal)) > (np.max(current_signal) - background)
        
        smoothed = gaussian_filter(current_signal, sigma=[0, sigma])
        
        valid_x, valid_y = [], []
        for r, y_val in enumerate(flux_coupler_mV):
            row_data = smoothed[r, :]
            idx = np.argmin(row_data) if is_finding_min else np.argmax(row_data)
            valid_x.append(flux_qubit_mV[idx])
            valid_y.append(y_val)
            
        X = np.array(valid_y).reshape(-1, 1) # Coupler
        y = np.array(valid_x)                # Qubit

        # C. 重新繪圖
        ax.pcolormesh(flux_qubit_mV, flux_coupler_mV, current_signal, 
                      cmap='viridis', shading='auto', vmin=global_min, vmax=global_max)
        
        title_suffix = "Fit: N/A"
        
        # 2. 畫擬合結果 (加入保護機制：如果對比度太低就不擬合)
        if len(X) > 5:
            try:
                ransac = RANSACRegressor(random_state=42, residual_threshold=2.0)
                ransac.fit(X, y)
                slope = ransac.estimator_.coef_[0]
                title_suffix = f"Slope: {100*slope:.1f}"
                print(slope)
                line_x_pred = ransac.predict(line_y_range)
                ax.plot(line_x_pred, line_y_range, color='red', linestyle='--', lw=2.5, alpha=0.9)
                
                inlier_mask = ransac.inlier_mask_
                ax.scatter(y[inlier_mask], X[inlier_mask], s=15, c='lime', edgecolors='k', lw=0.5, alpha=0.8, label='Inliers')

            except Exception:
                title_suffix = "Fit Failed"
        ax.grid()
        ax.set_title(f"{qubit_label} | Duration: {d:.0f} ns | {title_suffix} %", fontsize=12, fontweight='bold')
        ax.set_xlabel(f"{ds.attrs['target_q']} Flux amplitude [mV]")
        ax.set_ylabel("Coupler Flux [mV]")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        # ax.legend(loc='upper right', frameon=True, fontsize='small') # 可選

    # --- 4. 建立與儲存 ---
    ani = FuncAnimation(fig, update, frames=len(durations), interval=interval, blit=False)
    plt.close(fig)

    if save_path:
        print(f"Saving animation to {save_path}...")
        try:
            if save_path.endswith('.gif'):
                ani.save(save_path, writer='pillow', fps=1000//interval)
            elif save_path.endswith('.mp4'):
                ani.save(save_path, writer='ffmpeg', fps=1000//interval, dpi=150)
            print("Done.")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Please ensure 'ffmpeg' is installed for mp4, or use .gif")
        
    return ani

# %% {Node_parameters}
class Parameters(NodeParameters):
    
    z_source_c:List[str] = ['coupler_q3_q4']
    target_q:str = 'q4'
    control_flux_min:float = -0.025
    control_flux_max:float = 0.02
    qubit_flux_step : float = 0.0005

    
    source_flux_max:float = 0.15
    source_flux_min:float = -0.05
    
    exam_FCC:bool = False
    num_averages: int = 1500
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "independent"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 1200
    load_data_id: Optional[int] = None
    use_state_discrimination:bool = True
    operation:Literal['cz', 'iswap'] = 'iswap'
    pulse_duration_pts: int = 15   # 100*100*10 *300 runs = 280s (active) 
    target_q_bias:float = 0.2      # if target_q freq is higher than control_q. Applied when coupler's control_q is the assigned target_q
    force_apply_bias:bool = False

node = QualibrationNode(
    name="SP_FCC", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()


operation = 'cardinal'


z_sources = [machine.qubit_pairs[c] for c in node.parameters.z_source_c]
z_source_flux_pt = 'joint'

if  node.parameters.target_q == z_sources[0].qubit_control.name:
    q_ctrl = z_sources[0].qubit_control
    q_target = z_sources[0].qubit_target
    q_bias_apply = False 
elif  node.parameters.target_q == z_sources[0].qubit_target.name:
    q_ctrl = z_sources[0].qubit_target
    q_target = z_sources[0].qubit_control
    q_bias_apply = True
# if you want to check the unconnected qubits
else:
    for c in machine.active_qubit_pairs:
        if node.parameters.target_q == c.qubit_control:
            q_ctrl = c.qubit_control
            q_target = c.qubit_target
            q_bias_apply = False 
            break # directly break this for loop
        else:
            q_ctrl = c.qubit_target
            q_target = c.qubit_control
            q_bias_apply = True

if not q_bias_apply:
    q_bias_apply = node.parameters.force_apply_bias

bias_wait_time = 800 // 4
print(f"Ctrl: {q_ctrl.name}, Target: {q_target.name}, CT-flip: {q_bias_apply}")

elements_to_reset = [q_ctrl, q_target]

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
# %%

####################
# Helper functions #
####################


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
target_q_bias = node.parameters.target_q_bias
flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'
# Loop parameters
fluxes_source = np.linspace(node.parameters.source_flux_min, node.parameters.source_flux_max+0.0001, 20)
fluxes_qubit = np.arange(node.parameters.control_flux_min, node.parameters.control_flux_max+0.0001, node.parameters.qubit_flux_step) -0.05 
duras = make_unique_multiples_of_4(np.linspace(80, 2080, node.parameters.pulse_duration_pts))
qua_duras = duras//4


with program() as CPhase_Oscillations:
    n = declare(int)
    flux_source = declare(float)
    flux_qubit = declare(float)
    comp_flux_qubit = declare(float)
    n_st = declare_stream()
    qua_pulse_duration = declare(int)
        
    
    if node.parameters.use_state_discrimination:
        state_target = [declare(int) for _ in range(len(z_sources))]
        state_control = [declare(int) for _ in range(len(z_sources))]
        state_st_target = [declare_stream() for _ in range(len(z_sources))]
        state_st_control = [declare_stream() for _ in range(len(z_sources))]
    else:
        I_control = [declare(float) for _ in range(len(z_sources))]
        Q_control = [declare(float) for _ in range(len(z_sources))]
        I_target = [declare(float) for _ in range(len(z_sources))]
        Q_target = [declare(float) for _ in range(len(z_sources))]
        I_st_control = [declare_stream() for _ in range(len(z_sources))]
        Q_st_control = [declare_stream() for _ in range(len(z_sources))]
        I_st_target = [declare_stream() for _ in range(len(z_sources))]
        Q_st_target = [declare_stream() for _ in range(len(z_sources))]

    for i, z_source in enumerate(z_sources):
        
        if not node.parameters.simulate:
            machine.apply_all_couplers_to_min()
            for q in elements_to_reset:
                machine.set_all_fluxes(flux_point, q)
            machine.set_all_fluxes(z_source_flux_pt, z_source)
            wait(1000)

    
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)  

            for idx, qua_pulse_duration in enumerate(qua_duras):   
                
                with for_(*from_array(flux_source, fluxes_source)):
                    with for_(*from_array(flux_qubit, fluxes_qubit)):

                        if node.parameters.exam_FCC and "FCC" in z_source.extras:
                            try:
                                assign(comp_flux_qubit, flux_qubit + z_source.extras["FCC"][q_ctrl.name] * flux_source )
                            except:
                                print(f"Control qubit {q_ctrl} is not in {z_source}'s FCC, use 0 compensation instead.")
                                assign(comp_flux_qubit, flux_qubit) 
                        else:
                            assign(comp_flux_qubit, flux_qubit) 


                        # reset
                        for j, qubit in enumerate(elements_to_reset):
                            if node.parameters.reset_type == "active":
                                active_reset(qubit, "readout")
                                # active_reset_simple(qubit, "readout")
                            else:
                                if not node.parameters.simulate:
                                    qubit.wait(qubit.thermalization_time * u.ns)
                                else:
                                    qubit.wait(16 * u.ns)
                        
                        align()
                        # setting both qubits ot the initial state
                        q_ctrl.xy.play("x180")
                        if node.parameters.operation == 'cz':
                            q_target.xy.play("x180")
                        align()
                        if q_bias_apply:
                            q_target.z.play("const", amplitude_scale = target_q_bias / q_target.z.operations["const"].amplitude, duration = qua_pulse_duration+bias_wait_time)
                            q_ctrl.z.wait(bias_wait_time)
                            z_source.coupler.wait(bias_wait_time)
                        
                        q_ctrl.z.play(operation, amplitude_scale = comp_flux_qubit / q_ctrl.z.operations[operation].amplitude, duration = qua_pulse_duration)                
                        z_source.coupler.play(operation, amplitude_scale = flux_source / z_source.coupler.operations[operation].amplitude, duration = qua_pulse_duration)
                        align()
                        wait(20)
                        # readout
                        if node.parameters.use_state_discrimination:
                            if node.parameters.operation == 'cz':
                                # readout_state_gef(q_ctrl, state_control[i])
                                readout_state(q_ctrl, state_control[i])
                            else:
                                readout_state(q_ctrl, state_control[i])
                            wait(4)
                            z_source.align()
                            wait(4)
                            readout_state(q_target, state_target[i])
                            align()
                            save(state_control[i], state_st_control[i])
                            save(state_target[i], state_st_target[i])
                        else:
                            q_ctrl.resonator.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                            q_target.resonator.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                            save(I_control[i], I_st_control[i])
                            save(Q_control[i], Q_st_control[i])
                            save(I_target[i], I_st_target[i])
                            save(Q_target[i], Q_st_target[i])
                        
        align()
        
    with stream_processing():
        n_st.save("n")
        for i, z_source in enumerate(z_sources):
            if node.parameters.use_state_discrimination:
                state_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_source)).buffer(len(duras)).average().save(f"state_control{i + 1}")
                state_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_source)).buffer(len(duras)).average().save(f"state_target{i + 1}")
            else:
                I_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_source)).buffer(len(duras)).average().save(f"I_control{i + 1}")
                Q_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_source)).buffer(len(duras)).average().save(f"Q_control{i + 1}")
                I_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_source)).buffer(len(duras)).average().save(f"I_target{i + 1}")
                Q_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_source)).buffer(len(duras)).average().save(f"Q_target{i + 1}")
            
# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    # node.machine = machine
    # node.save()
elif node.parameters.load_data_id is None:
    
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(CPhase_Oscillations)

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
        ds = fetch_results_as_xarray(job.result_handles, z_sources, {  "flux_qubit": fluxes_qubit, "flux_coupler": fluxes_source, "duration":duras})
        ds = ds.assign_coords(flux_qubit_full=ds["flux_qubit"].broadcast_like(ds))
        ds = ds.assign_coords(flux_coupler_full=ds["flux_coupler"].broadcast_like(ds))
        ds.attrs["target_q"] = node.parameters.target_q
        ds.attrs["coupler_z_waveform"] = operation
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}

print(ds.dims)
# %%
node.results["results"] = {}

## HARD CODED FROM EXPERIMENT
    
    
# %% {Plotting}
plot_state:Literal['control', 'target'] = 'control'
if not node.parameters.simulate:
    for qp in z_sources:
        node.results["results"][qp.name] = {}
        fig, dura, slopes, errs = analyze_crosstalk_vs_duration(ds, qp.name, state=plot_state, state_discriminator=node.parameters.use_state_discrimination, y_limit=None, sigma=12.0)
        # node.results["results"][qp.name][node.parameters.target_q] = slop
        if fig is not None:
            node.results[f'figure_{plot_state}'] = fig





# %% {Update_state}
if not node.parameters.simulate:
    pass
                    
# %% {Save_results}
if not node.parameters.simulate:    
    node.outcomes = {q.name: "successful" for q in z_sources}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()

    from qualibrate_config.resolvers import get_qualibrate_config_path, get_qualibrate_config
    from qualibrate.utils.node.path_solver import get_node_dir_path
    import os
    qs = get_qualibrate_config(get_qualibrate_config_path())
    base_path = qs.storage.location

    node_dir = get_node_dir_path(node.snapshot_idx, base_path)

    # {Check raw data
    for qp in z_sources:
        save_path = os.path.join(node_dir, f"{qp.name}to{node.parameters.target_q}_FC_vs_duration.gif")
        animate_crosstalk_evolution(ds, qp.name, state=plot_state, state_discriminator=node.parameters.use_state_discrimination, sigma=12.0, y_limit=None, interval=1000, save_path=save_path)
        
# %%
