# %%
import os, re
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import RANSACRegressor
from scipy.stats import linregress
from qualibrate  import NodeParameters, QualibrationNode
# %%
def fit_single_slice_with_error(ds_slice, y_limit=None, sigma=2.0):
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


def analyze_crosstalk_vs_duration(ds, state='control', state_discriminator:bool=True, y_limit=None, sigma:float=2.0):
    qubit_label = ds.qubit.values[0]
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
        slope, err = fit_single_slice_with_error(ds_slice, y_limit, sigma)
        
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
    if 'coupler_z_waveform' in ds.attrs:
        ax.set_title(f"{ds.attrs['coupler_z_waveform']} Crosstalk vs. Duration ({qubit_label})")
    else:
        ax.set_title(f"Flux Crosstalk vs. Duration ({qubit_label})")
    ax.set_xlabel("Duration [ns]") # 假設單位是 ns，請確認
    ax.set_ylabel("Crosstalk [%]")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig, durations, slopes, errors


def plot_combined(source_qubit_label:str, dict_to_plot:dict, target_qubit_label:str, normalize_Y:bool=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    results = {}
    for i, item in enumerate(dict_to_plot):
        # 1. 準備原始數據
        x_raw = np.array(dict_to_plot[item]['durations'])
        y_raw = np.array(dict_to_plot[item]['slopes']) * 100 
        y_err_raw = np.array(dict_to_plot[item]['errors']) * 100
        
        # 2. 【第一步】找出誤差最大的點並刪除 (Outlier Removal)
        idx_max_err = np.argmax(y_err_raw)
        
        # 產生過濾後的數據
        x = np.delete(x_raw, idx_max_err)
        y = np.delete(y_raw, idx_max_err)
        y_err = np.delete(y_err_raw, idx_max_err)
        ori_y_to_fit = y
        current_color = colors[i % len(colors)]
        
        if normalize_Y:
            y_min = np.min(y)
            y_max = np.max(y)
            y = (y - y_min) / (y_max - y_min)
            y_err = y_err / (y_max - y_min)
        

        # 4. 【第二步】加權擬合 (Weighted Fit on remaining data)
        # 權重 = 1 / 誤差 (誤差越小，權重越大)
        weights = 1 / (y_err + 1e-9)
        
        # 使用 numpy.polyfit 進行加權擬合 (degree=1 代表線性)
        # w=weights 參數告訴 numpy 哪些點比較重要
        p, cov = np.polyfit(x, y, deg=1, w=weights, cov=True)
        
        slope, intercept = p
        

        
        # 計算加權 R^2
        y_fit = slope * x + intercept
        ss_res = np.sum(weights * (y - y_fit)**2)
        ss_tot = np.sum(weights * (y - np.average(y, weights=weights))**2)
        r_squared = 1 - (ss_res / ss_tot)

        results[item] = {
            'slope': slope,          # 單位: % / ns
            'intercept': intercept,  # 單位: %
            'r_squared': r_squared
        }
        
        # 5. 畫趨勢線
        # 3. 繪圖 (只畫過濾後的點)
        if normalize_Y:
            norm_factor = y_max - y_min
            p, cov = np.polyfit(x, ori_y_to_fit, deg=1, w=weights, cov=True)
            slope, intercept = p
        else:
            norm_factor = 0.0


        
        ax.errorbar(x, y, yerr=y_err, fmt='o', capsize=5, lw=2, markersize=6,
                    color=current_color, label=f'{item}, \n$y={slope*1000:.2f}x{intercept:+.3f}, R^2={r_squared:.3f}$\n norm={norm_factor:.2f}% \n')
        ax.plot(x, y_fit, linestyle='--', alpha=0.7, color=current_color)

    ax.set_title(f"{source_qubit_label} -> {target_qubit_label} Time dependent Flux Crosstalk (slope amplified by 1k)")
    ax.set_xlabel("Duration [ns]")
    ax.set_ylabel("Flux Crosstalk [%]" if not normalize_Y else "Normalized Flux Crosstalk [a.u.]")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 調整 Legend
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0) # 若圖例太大擋住，可改 loc='upper left' 或其他位置
    
    plt.tight_layout()
    plt.show()
    return fig, results

def predict_zero_slope(results_dict: dict):
    """
    分析不同 P 值對應的 Slope，預測 Slope 為 0 時的 P 值。
    並計算該預測模型的 R^2 (決定係數)。
    """
    p_values = []
    slopes = []
    
    # 1. 解析字典，提取 P 值與 Slope
    print(f"{'P-Value':<10} | {'Slope':<15}")
    print("-" * 30)
    
    for label, data in results_dict.items():
        # 抓取 "P=" 後面的數字
        match = re.search(r"P=(\d+)", label)
        if match:
            p_val = float(match.group(1))
            slope_val = data['slope']
            
            p_values.append(p_val)
            slopes.append(slope_val)
            print(f"{p_val:<10.0f} | {slope_val:<15.5f}")
            
    if len(p_values) < 2:
        print("❌ 數據點不足，無法進行預測 (至少需要 2 個不同的 P 值)")
        return None

    # 轉成 numpy array
    x = np.array(p_values)
    y = np.array(slopes)

    # 2. 建立線性模型 (Slope vs P)
    # linregress 回傳的 r_value 就是相關係數
    m, c, r_value, p_val_stat, std_err = linregress(x, y)
    
    # 【關鍵修改】計算 R^2
    r_squared = r_value**2

    # 3. 預測 Slope = 0 時的 P 值
    # 0 = m * P_target + c  =>  P_target = -c / m
    target_p = -c / m
    
    # --- 繪圖展示結果 ---
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # 畫測量點
    ax.scatter(x, y, s=80, label='Measured Slopes', color='blue', zorder=5)
    
    # 畫擬合線
    x_range = np.linspace(min(x.min(), target_p) - 5, max(x.max(), target_p) + 5, 100)
    y_fit = m * x_range + c


    ax.plot(x_range, y_fit, 'r--', alpha=0.6, label=f'$R^2={r_squared:.2f}$')
    
    # 標記預測點 (Zero Crossing)
    ax.scatter([target_p], [0], color='green', marker='*', s=200, zorder=10, 
               label=f'Predicted P for 0 Slope: {target_p:.2f}%')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(target_p, color='green', linestyle=':', alpha=0.3)

    ax.set_title("Prediction of Bipolar composition for Zero Time dependency Flux Crosstalk")
    ax.set_xlabel("Bipolar Positive Pole Ratio [%]")
    ax.set_ylabel("Flux Crosstalk Time Dependency [%/ns]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    
    return fig, target_p/100


class Parameters(NodeParameters):
    test:str = "test"
node = QualibrationNode(name="Combine_TimeDepFC_vs_Waveform", parameters=Parameters())
node.results = {}


dss = {
    # "Bipolar P=40%":'/home/ratiswu/Qualibrate_data/2Q1C/2026-01-18/#85_SP_FCC_145243/ds.h5',
    # "Bipolar P=50%":"/home/ratiswu/Qualibrate_data/2Q1C/2026-01-18/#90_SP_FCC_150203/ds.h5",
    # "Bipolar P=60%":"/home/ratiswu/Qualibrate_data/2Q1C/2026-01-18/#96_SP_FCC_151124/ds.h5",
    # "Bipolar P=70%":"/home/ratiswu/Qualibrate_data/2Q1C/2026-01-18/#102_SP_FCC_151829/ds.h5",
    # "Bipolar P=75%":"/home/ratiswu/Qualibrate_data/2Q1C/2026-01-18/#120_SP_FCC_160231/ds.h5",
    # "Bipolar P=80%":"/home/ratiswu/Qualibrate_data/2Q1C/2026-01-18/#104_SP_FCC_152419/ds.h5",
    # "2q1c, q1 in q1_q2 - Square":"/home/ratiswu/Qualibrate_data/2Q1C/2026-01-17/#48_SP_FCC_152401/ds.h5",
    "10Qv2, Bipolar P=100%" : "/home/ratiswu/Qualibrate_data/10Q9Cv2_q1q5/2026-01-31/#294_SP_FCC_150346/ds.h5",
    "10Qv2, Bipolar P=75% ":"/home/ratiswu/Qualibrate_data/10Q9Cv2_q1q5/2026-01-31/#330_SP_FCC_182817/ds.h5"
}

plot_state = ['target', 'control']

to_plot = {}

# used for compa fig title only
target_q = 'q' 
source = 'coupler'    

for i, waveform in enumerate(list(dss.keys())):
    to_plot[waveform] = {}
    _, to_plot[waveform]['durations'], to_plot[waveform]['slopes'], to_plot[waveform]['errors'] = analyze_crosstalk_vs_duration(xr.open_dataset(dss[waveform]), state=plot_state[i], y_limit=None, sigma=12.0)



node.results["Hybrid_Compa_fig"], analysis_results = plot_combined(source, to_plot, target_q, normalize_Y=True)
node.results["references"] = {}
for i in dss:
    node.results["references"][i] = os.path.split(dss[i])[0]

node.results["Time dependency"] = analysis_results
if len(list(dss.keys())) >= 3:
    node.results["Bipolar_best_positive_ratio_fig"], node.results["best_p_value"] = predict_zero_slope(analysis_results)

node.save()
# %%
t = np.arange(80) / 80
y = 1 * 0.5 * (1 - np.cos(np.pi * t))
y2 = np.ones(80)
y3 = 1 * 0.5 * (1 + np.cos(np.pi * t))
y4 = -1 * 0.5 * (1 - np.cos(np.pi * t))
y5 = -1*np.ones(80)
y6 = -1 * 0.5 * (1 + np.cos(np.pi * t))

plt.plot(np.arange(80*6),np.concatenate([y, y2, y3, y4, y5, y6]))
plt.grid()
plt.show()
# %%
