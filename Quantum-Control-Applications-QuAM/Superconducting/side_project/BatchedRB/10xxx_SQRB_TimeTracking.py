import time, math
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.axes import Axes

# %%
def to_float_clean(x):
    s = str(x).strip().lstrip("'")
    return float(s)


def execution(save_path:str, elapse_time_min:int|None, num_runs:int|str, cooldown_sec:float, load_path:str|None=None):
    if load_path is None:
        from quam_libs.experiments.SQRB import run_SQRB
        # folder path to save the file
        out_dir = Path(save_path) #Change
        out_dir.mkdir(parents=True, exist_ok=True)

        # csv file name (contains the time to avoid overlap)

        session_ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")
        wide_csv = out_dir / f"SQEPC_{session_ts}.csv" 
        i = 0
        header_qubits = None  
        start = time.time()
        while True:
            i += 1
            ts = datetime.now().isoformat(timespec="seconds")
            print(f"[{i:02d}/{num_runs}] {ts} Starting run…")

            try:
                EPG, EPC = run_SQRB()
                qubit_names = EPC.qubit.values
                fluctuations = EPC.values
                # create column names for qubits, determined by the first run
                if header_qubits is None:
                    header_qubits = list(qubit_names)
                    with open(wide_csv, "w", newline="") as f:
                        csv.writer(f).writerow(["run_index", "timestamp", *header_qubits])

                with open(wide_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    name_to_val = {n: to_float_clean(v) for n, v in zip(qubit_names, fluctuations)}
                    row_vals = [name_to_val[q] for q in header_qubits]  
                    writer.writerow([i, ts, *row_vals])

                print(f"[{i:02d}/{num_runs}] appended to {wide_csv.name}")

            except Exception as e:
                print(f"[{i:02d}/{num_runs}] error: {e!r}, continuing to the next run…")

            time.sleep(cooldown_sec)
            end = time.time()
            if elapse_time_min is None:
                if not isinstance(num_runs, str):
                    if isinstance(num_runs, int):
                        if i == num_runs :
                            break
            else:
                if (end - start)/60 > float(elapse_time_min):
                    break
            

        print(f"All runs finished: {wide_csv}") 
    
    else:
        csv_path = Path(load_path)
        x_axis = "timestamp"   # can be "run_index" or "timestamp"  
        base_dir = csv_path.parent
        # create a subfolder using the CSV file name
        out_dir = base_dir / csv_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # read CSV and validate columns
        df = pd.read_csv(csv_path)
        if "run_index" not in df.columns or "timestamp" not in df.columns:
            raise ValueError("CSV must include columns: run_index, timestamp, and at least one qubit column")

        # check and set x-axis
        if x_axis not in ("run_index", "timestamp"):
            raise ValueError("x_axis must be either 'run_index' or 'timestamp'")
        if x_axis == "timestamp":
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # identify qubit columns (exclude run_index and timestamp)
        qubit_cols = [c for c in df.columns if c not in ("run_index", "timestamp")]
        if not qubit_cols:
            raise ValueError("No qubit columns found")

        # convert to numeric values (non-numeric become NaN)
        for c in qubit_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        n = len(qubit_cols)
        ncols = 2 if n > 1 else 1        # 如果只有 1 個 qubit，就只要 1 欄
        nrows = math.ceil(n / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axs = np.atleast_1d(axs).flatten()

        for i, q in enumerate(qubit_cols):
            ax:Axes = axs[i]
            sub = df[[x_axis, q]].dropna()
            if sub.empty:
                continue

            ax.scatter(sub[x_axis], sub[q], marker="o")
            # --- 補充的部分 ---
            # 1. 計算平均值與標準差
            mean_val = sub[q].mean()
            std_val = sub[q].std()
            
            # 2. 加上 mean 的 hlines
            ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'$\mu$={mean_val:.2f}')
            
            # 3. 加上 +std, -std 區間標記 (alpha=0.5)
            # axhspan 會自動填滿給定 y 區間的整個 x 軸範圍
            ax.axhspan(mean_val - std_val, mean_val + std_val, color='red', alpha=0.5, label=f'$\sigma$={std_val:.2f}')
            
            # 顯示圖例 (把 mean 的數值也秀出來)
            ax.legend()
        
        for j in range(len(qubit_cols), len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axs = np.atleast_1d(axs).flatten()

        for i, q in enumerate(qubit_cols):
            ax = axs[i]
    
            # 記得要在這裡取出對應的數據並排除 NaN
            sub_q = df[q].dropna()
            if sub_q.empty:
                continue
                
            # --- 補充的部分 ---
            # 1. 畫出 Histogram (bins='auto' 會使用 Freedman-Diaconis rule 自動決定最佳柱數)
            # density=True 讓面積總和為 1，方便疊加機率密度函數 (PDF)
            ax.hist(sub_q, bins='auto', density=True, alpha=0.6, color='#1f77b4', edgecolor='white', label='Data')
            
            # 2. 擬合並畫出 Gaussian 分布曲線
            mu, std = norm.fit(sub_q)           # 取得這組數據的 mu 和 std
            xmin, xmax = ax.get_xlim()          # 取得當前圖表的 X 軸範圍
            x = np.linspace(xmin, xmax, 100)    # 產生平滑的 X 陣列
            p = norm.pdf(x, mu, std)            # 計算對應的 Gaussian PDF
            
            # 畫出曲線
            ax.plot(x, p, 'k--', linewidth=1.5, label=f'Fit ($\mu$={mu:.3f}, $\sigma$={std:.3f})')
            
            ax.set_title(f'Distribution of {q}')
            ax.legend()
        for j in range(len(qubit_cols), len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()

# %%
elapse_time_min:None|float = 60 # if None: num_runs decides when to stop. Else: keep running within the elapse time (in minutes)
num_runs = 1 #Change an integer. If you wanna keep tracking, set it as 'inf'.
cooldown_sec = 5 # Change
save_path = '/home/ratiswu/Qualibrate_data/as-qpu-10Q9Cv2/SQRB_Tracking'
load_data_path = '/home/ratiswu/Qualibrate_data/as-qpu-10Q9Cv2/SQRB_Tracking/SQEPC_2026-04-12T14-10-33.csv' # if None: run experiments, else: load data to plot

execution(save_path, elapse_time_min, num_runs, cooldown_sec, load_data_path)