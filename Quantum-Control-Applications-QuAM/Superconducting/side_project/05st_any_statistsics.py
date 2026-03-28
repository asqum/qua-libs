import time
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd
from qualibrate import QualibrationNode, NodeParameters
import matplotlib.pyplot as plt

from node_for_repeat import run_flux_offset, run_T1, run_T2_echo, run_IQ_blobs


def to_float_clean(x):
    s = str(x).strip().lstrip("'")
    return float(s)

# --- 可修改的參數 ---
num_runs = None   # None means keep running until Ctrl+C
cooldown_sec = 0  # 每次間隔秒數

# --- 設定每個 function 的參數 ---
parameters_for_functions = {
    "flux_offset": {
        "qubits": ["q1", "q2"],
        "num_averages": 500,
        "frequency_detuning_in_mhz": 8.0,
        "min_wait_time_in_ns": 16,
        "max_wait_time_in_ns": 500,
        "wait_time_step_in_ns": 10,
        "flux_span": 0.04,
        "flux_step": 0.001,
        "flux_point_joint_or_independent": "independent",
        "simulate": False,
        "simulation_duration_ns": 2500,
        "timeout": 100,
        "load_data_id": None,
        "multiplexed": False,
    },
    "T1": {
        "qubits": ["q1", "q2"],
        "num_averages": 300,
        "min_wait_time_in_ns": 16,
        "max_wait_time_in_ns": 90000,
        "wait_time_step_in_ns": 300,
        "flux_point_joint_or_independent_or_arbitrary": "independent",
        "reset_type": "thermal",
        "use_state_discrimination": False,
        "simulate": False,
        "simulation_duration_ns": 2500,
        "timeout": 100,
        "load_data_id": None,
        "multiplexed": True,
    },
    "T2_echo": {
        "qubits": ["q1", "q2"],
        "num_averages": 1000,
        "min_wait_time_in_ns": 16,
        "max_wait_time_in_ns": 10000,
        "wait_time_step_in_ns": 50,
        "flux_point_joint_or_independent_or_arbitrary": "independent",
        "reset_type": "thermal",
        "use_state_discrimination": True,
        "simulate": False,
        "simulation_duration_ns": 2500,
        "timeout": 100,
        "load_data_id": None,
        "multiplexed": True,
    },
    "IQ_blobs": {
        "qubits": ["q1", "q2"],
        "num_runs": 3000,
        "flux_point_joint_or_independent": "independent",
        "reset_type": "thermal",
        "operation_name": "readout",
        "simulate": False,
        "simulation_duration_ns": 1000,
        "timeout": 100,
        "load_data_id": None,
        "multiplexed": False
    },
}

# --- 選擇每次要跑的 function ---
functions_to_run = {
    "flux_offset": run_flux_offset,
    "T1": run_T1,
    "T2_echo": run_T2_echo,
    "IQ_blobs": run_IQ_blobs,
}

# --- 資料儲存路徑 ---
out_dir = Path(r"D:\RCCI\Data\251027_DR4_5Q4C#2")
out_dir.mkdir(parents=True, exist_ok=True)
session_ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")
excel_path = out_dir / f"qubit_measurements_{session_ts}.xlsx"

# ======= Configuration =======
excel_path = out_dir
x_axis = "run_index"   # 可改成 "timestamp"
# =============================

# --- 暫存資料結構 ---
# run_data[function][result_name] = list of dict (每次 run 的資料)
run_data = {}

print("Press Ctrl+C to stop running.\n")

i = 0
try:
    while True if num_runs is None else i < num_runs:
        i += 1
        ts = datetime.now().isoformat(timespec="seconds")
        run_label = f"{i}" if num_runs is None else f"{i:02d}/{num_runs}"
        print(f"[{run_label}] {ts} Starting run…")

        try:
            for fname, func in functions_to_run.items():
                params = parameters_for_functions.get(fname, None)
                result = func(params) if params is not None else func()

                # 最後一個元素是 qubits_name，其餘是 (list, name)
                *values_with_names, qubits_name = result

                if fname not in run_data:
                    run_data[fname] = {}

                for vals, name in values_with_names:
                    if name not in run_data[fname]:
                        run_data[fname][name] = []

                    row = {"run_index": i, "timestamp": ts}
                    for q_idx, q in enumerate(qubits_name):
                        row[q] = to_float_clean(vals[q_idx])
                    run_data[fname][name].append(row)

                print(f"[{run_label}] {fname}: stored results.")

        except Exception as e:
            print(f"[{run_label}] error: {e!r}, continuing…")

        time.sleep(cooldown_sec)

except KeyboardInterrupt:
    print("\n🟡 Detected Ctrl+C — saving results…")

finally:
    # --- 寫入 Excel，每個回傳 list 對應一個分頁 ---
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for fname, result_dict in run_data.items():
            for result_name, records in result_dict.items():
                sheet_name = f"{result_name}"  # 使用回傳的名字
                df = pd.DataFrame(records)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✅ All data saved to: {excel_path}")

# =========plot===============

base_dir = excel_path.parent
out_dir = base_dir / excel_path.stem
out_dir.mkdir(parents=True, exist_ok=True)

# --- 讀取 Excel 的所有工作表 ---
sheets = pd.read_excel(excel_path, sheet_name=None)

for fname, df in sheets.items():
    print(f"Processing sheet: {fname}")

    if "run_index" not in df.columns or "timestamp" not in df.columns:
        continue

    qubit_cols = [c for c in df.columns if c not in ("run_index", "timestamp")]
    if not qubit_cols:
        continue

    # timestamp 轉 datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # 逐 qubit 畫折線圖（多次測量變化）
    for q in qubit_cols:
        sub = df[[x_axis, q]].dropna()
        if sub.empty:
            continue
        fig = plt.figure()
        plt.plot(sub[x_axis], sub[q], marker="o")
        plt.title(f"{fname} – {q} time trend")
        plt.xlabel(x_axis)
        plt.ylabel(fname)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.xticks(rotation=90)
        fig.savefig(out_dir / f"{fname}_{q}_line_{x_axis}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- 各 qubit 的統計柱狀圖（Histogram） ---
        data = sub[q].dropna().values
        mean_val = np.mean(data)
        std_val = np.std(data)

        fig_hist = plt.figure()
        plt.hist(data, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
        plt.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.2f}")
        plt.axvline(mean_val + std_val, color="orange", linestyle="--", linewidth=1, label=f"+1σ = {mean_val+std_val:.2f}")
        plt.axvline(mean_val - std_val, color="orange", linestyle="--", linewidth=1, label=f"-1σ = {mean_val-std_val:.2f}")

        plt.title(f"{fname} – {q} Distribution\nMean = {mean_val:.2f}, Std = {std_val:.2f}")
        plt.xlabel(f"{fname} value")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        fig_hist.savefig(out_dir / f"{fname}_{q}_hist.png", dpi=150, bbox_inches="tight")
        plt.close(fig_hist)

    # 所有 qubit 同圖（時間趨勢）
    fig_all = plt.figure()
    for q in qubit_cols:
        sub = df[[x_axis, q]].dropna()
        if sub.empty:
            continue
        plt.plot(sub[x_axis], sub[q], marker="o", label=q)
    plt.title(f"{fname} – All Qubits Time Trend")
    plt.xlabel(x_axis)
    plt.ylabel(fname)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.xticks(rotation=90)
    fig_all.savefig(out_dir / f"{fname}_all_qubits_line_{x_axis}.png", dpi=150, bbox_inches="tight")
    plt.close(fig_all)

print(f"✅ All charts (line + per-qubit histogram) saved to: {out_dir.resolve()}")
