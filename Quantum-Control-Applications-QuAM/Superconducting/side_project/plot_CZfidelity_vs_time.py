from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
# ======= Configuration =======
csv_path = Path("/home/ratiswu/Qualibrate_data/5Q4C_Qcage/2025-11-14/#9a9a_2QRB_Tracking/CZ_Fidelity_Tracking_2025-11-14T21-15-45/CZ_Fidelity_Tracking_2025-11-14T21-15-45.csv")
base_dir = Path("/home/ratiswu/Qualibrate_data/5Q4C_Qcage/2025-11-14/#9a9a_2QRB_Tracking/CZ_Fidelity_Tracking_2025-11-14T21-15-45")
x_axis = "timestamp"   # can be "run_index" or "timestamp"
# =============================

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

# plot each qubit separately
for q in qubit_cols:
    sub = df[[x_axis, q]].dropna()
    if sub.empty:
        continue
    fig = plt.figure()
    
    mean, sd = round(np.mean(np.array(sub[q]*100)),1), round(np.std(np.array(sub[q]*100)),1)
    
    plt.plot(sub[x_axis], sub[q]*100,zorder=1)
    # plt.plot(sub[x_axis], savgol_filter(sub[q]*100,window_length=31,polyorder=5),zorder=2,label='filtered')
    plt.hlines(mean-sd, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),linestyles="--",colors='pink',label='Deviation')
    plt.hlines(mean+sd, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),linestyles="--",colors='pink')
    plt.hlines(mean, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),colors='red',label="mean")
    plt.scatter(sub.loc[sub[q].idxmax(), x_axis],np.max(sub[q]*100),c='red',marker="*",s=100,label=f"best {round(np.max(sub[q]*100),1)} % at {sub.loc[sub[q].idxmax(), x_axis]}",zorder=3)
    plt.title(f"{q} CZ fidelity = {mean} $\pm$ {sd} %")
    plt.xlabel(x_axis)
    plt.ylabel("CZ fidelity (%)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(rotation=90)
    plt.legend(fontsize=6)
    fig.savefig(out_dir / f"{q}_line_{x_axis}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)