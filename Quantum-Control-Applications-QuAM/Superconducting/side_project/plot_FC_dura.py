import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import RANSACRegressor
from scipy.stats import linregress
from quam_libs.lib.save_utils import load_dataset
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

    # --- D. 關鍵步驟：使用 Scipy 計算標準誤差 ---
    # RANSAC 告訴我們哪些點是乾淨的，我們用這些乾淨的點來算統計誤差
    if np.sum(inlier_mask) < 3:
        return np.nan, np.nan

    # linregress(x, y) -> 注意這裡輸入都要是 1D array
    # 我們的 X 是 Coupler (自變數), y 是 Qubit Offset (應變數)
    slope_result = linregress(X[inlier_mask].ravel(), y[inlier_mask])
    
    # slope_result.slope = 斜率
    # slope_result.stderr = 斜率的標準誤差
    return slope_result.slope, slope_result.stderr


def analyze_crosstalk_vs_duration(ds, qubit_label, state='control', y_limit=None):
    
    # 1. 準備儲存容器
    durations = ds.duration.values
    slopes = []
    errors = []
    
    target_var = 'state_target' if state.lower() != 'control' else 'state_control'
    print(f"Starting analysis for {len(durations)} duration points...")

    # 2. 遍歷每一個 duration
    for d in durations:
        # 選取特定的 duration 和 qubit，這裡會降維成 2D (coupler, flux_qubit)
        ds_slice = ds[target_var].sel(qubit=qubit_label, duration=d)
        
        # 呼叫上面的核心運算
        slope, err = fit_single_slice_with_error(ds_slice, qubit_label, y_limit)
        
        slopes.append(slope)
        errors.append(err)
        
        # (選用) 顯示進度
        # print(f"Duration {d}: Slope={slope:.4f} +/- {err:.4f}")

    # 轉成 numpy array 方便繪圖
    slopes = np.array(slopes)
    errors = np.array(errors)
    
    return durations, slopes, errors

def plot(qubit_label:str, dict_to_plot:dict):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 使用 errorbar
    for item in dict_to_plot:
        ax.errorbar(dict_to_plot[item]['durations'], dict_to_plot[item]['slopes']*100, yerr=dict_to_plot[item]['errors']*100, 
                    fmt='-o',        # 線型 + 點
                    color='royalblue',
                    ecolor='tomato', # 誤差棒顏色
                    capsize=5,       # 誤差棒帽子大小
                    lw=2, markersize=6,
                    label=f'{item} waveform Crosstalk')

    # 美化圖表
    ax.set_ylim(-15, -8)
    ax.set_title(f"Z pulse waveform Crosstalk vs. Duration ({qubit_label})")
    ax.set_xlabel("Duration [ns]") # 假設單位是 ns，請確認
    ax.set_ylabel("Crosstalk [%]")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    return fig


file_a = xr.open_dataset('/home/ratiswu/Qualibrate_data/2Q1C/2026-01-17/#52_SP_FCC_153216/ds.h5')
file_b = xr.open_dataset('/home/ratiswu/Qualibrate_data/2Q1C/2026-01-17/#55_SP_FCC_162759/ds.h5')
dss = {"Square":file_a, "Bipolar":file_b}
to_plot = {}
plot_state = 'target'

    
for waveform in dss:
    to_plot[waveform] = {}
    to_plot[waveform]['durations'], to_plot[waveform]['slopes'], to_plot[waveform]['errors'] = analyze_crosstalk_vs_duration(dss[waveform], 'coupler_q1_q2', state=plot_state, y_limit=None)


fig = plot('coupler_q1_q2', to_plot)

# %%
