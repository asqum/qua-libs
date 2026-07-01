from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ======= Configuration =======
excel_path = Path(r"D:\RCCI\Data\251027_DR4_5Q4C#2\qubit_measurements_2025-10-28T13-10-26.xlsx")
x_axis = "run_index"   # 可改成 "timestamp"
# =============================

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
