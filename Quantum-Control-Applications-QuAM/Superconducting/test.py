import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
# %%
df = pd.read_csv("/Users/ratiswu/Downloads/For_Po_An.csv")

x_data = np.array(df['Pulse time (ns)'])
y_data_q4 = np.array(df['C -> Q4 (%)'])
y_data_q5 = np.array(df['C -> Q5 (%)'])

# %%
def f_log_lin(x, a, b): 
    return a * np.log10(x) + b

def f_log_poly2(x, a, b, c): 
    return a * (np.log10(x)**2) + b * np.log10(x) + c

def f_log_poly3(x, a, b, c, d): 
    return a * (np.log10(x)**3) + b * (np.log10(x)**2) + c * np.log10(x) + d

def f_power(x, a, b, c): 
    return a * (x**b) + c

def f_exp(x, a, b, c):
    # 為了避免 x=100000 時 e^x 爆掉，內部運算時稍微縮放，但不影響最終曲線形狀
    return a * np.exp(b * (x / 100000.0)) + c

# 將模型打包成字典方便迴圈處理
models = {
    'Linear': f_log_lin,
    # 'Log-Poly2': f_log_poly2,
    # 'Log-Poly3': f_log_poly3,
    'Power-Law': f_power,
    'Exponential': f_exp
}
def get_formula_string(name, popt, keep_digits:int=2):
    """根據模型名稱與參數，回傳格式化的公式字串"""
    # fmt1 用於首項 (不需要強制顯示正號)
    fmt1 = f".{keep_digits}g"
    # fmt2 用於後續項 (強制顯示 +/- 符號，解決 "+ -4.8" 的問題)
    fmt2 = f"+.{keep_digits}g"
    
    if name == 'Linear' or name == 'Log-Linear':
        return rf"$y = {popt[0]:{fmt1}}\log_{{10}}(x) {popt[1]:{fmt2}}$"
    
    elif name == 'Log-Poly2':
        return rf"$y = {popt[0]:{fmt1}}(\log_{{10}}(x))^2 {popt[1]:{fmt2}}\log_{{10}}(x) {popt[2]:{fmt2}}$"
    
    elif name == 'Log-Poly3':
        return rf"$y = {popt[0]:{fmt1}}(\log_{{10}}(x))^3 {popt[1]:{fmt2}}(\log_{{10}}(x))^2 {popt[2]:{fmt2}} \log_{{10}}(x) {popt[3]:{fmt2}}$"
    
    elif name == 'Power-Law':
        # LaTeX 的次方需要 {} 包裝，在 f-string 中要寫成 {{ }}
        return rf"$y = {popt[0]:{fmt1}}x^{{ {popt[1]:{fmt1}} }} {popt[2]:{fmt2}}$"
    
    elif name == 'Exponential':
        return rf"$y = {popt[0]:{fmt1}}e^{{ {popt[1]:{fmt1}}(x/100000) }} {popt[2]:{fmt2}}$"
    
    return "Unknown Formula"

def find_absolute_best_fit(x, y, target_name):
    best_name = ""
    best_func = None
    best_params = None
    highest_r2 = -float('inf')  # R² 可能為負，所以初始值設為極小
    
    print(f"--- {target_name} 擬合分數 (R²) 排行榜 ---")
    
    for name, func in models.items():
        try:
            popt, _ = curve_fit(func, x, y, maxfev=10000)
            
            # 計算預測值並計算 R²
            y_pred = func(x, *popt)
            r2 = r2_score(y, y_pred)
            
            print(f"{name:<15} : R² = {r2:.4f}")
            
            # 更新最佳紀錄 (尋找最大的 R²)
            if r2 > highest_r2:
                highest_r2 = r2
                best_name = name
                best_func = func
                best_params = popt
        except Exception:
            print(f"{name:<15} : 擬合失敗")
    best_formula = get_formula_string(best_name, best_params)
    print(f">> {target_name} 最佳選擇: {best_name} (R²: {highest_r2:.4f})\n")
    return best_func, best_params, f"{best_formula} (R²:{highest_r2:.3f})"

# %%
# 4. 執行與繪圖
plt.figure(figsize=(12, 7))
x_fit = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 200)

# 處理 Q4
best_f_q4, params_q4, label_q4 = find_absolute_best_fit(x_data, y_data_q4, "Q4")
plt.scatter(x_data, y_data_q4, color='#1f77b4', marker='o', s=70)
plt.plot(x_fit, best_f_q4(x_fit, *params_q4), color='#1f77b4', linestyle='-', linewidth=2, 
         label=f'q4: {label_q4}')

# 處理 Q5
best_f_q5, params_q5, label_q5 = find_absolute_best_fit(x_data, y_data_q5, "Q5")
plt.scatter(x_data, y_data_q5, color='#ff7f0e', marker='X', s=70)
plt.plot(x_fit, best_f_q5(x_fit, *params_q5), color='#ff7f0e', linestyle='--', linewidth=2, 
         label=f'q5: {label_q5}')

# 5. 圖表視覺優化
plt.xscale('log')
plt.xlabel('Pulse time (ns)', fontsize=12)
plt.ylabel('Flux Crosstalk (%)', fontsize=12)
plt.title('Pulse Time Dependent Flux Crosstalk\n Flux Source: Coupler_q4_q5', fontsize=15)
plt.grid(True, which='both', linestyle=':', alpha=0.7)
plt.legend(fontsize=11, frameon=True, shadow=True)
plt.tight_layout()
plt.show()

# %%