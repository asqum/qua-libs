from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from scipy.signal import savgol_filter
# ======= Configuration =======
x_axis = "timestamp"   # can be "run_index" or "timestamp"
base_dir = Path("/home/ratiswu/Qualibrate_data/5Q4C_Qcage/2025-12-09/#0a0a_2QRB_Tracking")
## CZ tracking
csv_path = Path("/home/ratiswu/Qualibrate_data/5Q4C_Qcage/2025-12-09/#0a0a_2QRB_Tracking/CZ_Fidelity_Tracking_2025-12-09T00-05-38.csv")
## OFFSET tracking
offset_csv:Path|None = Path("/home/ratiswu/Qualibrate_data/5Q4C_Qcage/2025-12-09/#0a0a_2QRB_Tracking/Offset_Tracking_2025-12-09T00-05-38.csv")
## OFFSET start point
offset_when_calibration:dict = {"q2": 0.13452111192651628, "q3": 0.15909386915732268}
# =============================

def norm_1darray(y):
    y = np.array(y)*1000 # convert to mV
    modulation = np.max(y) - np.min(y)
    return modulation, savgol_filter((y - np.min(y)) / modulation, window_length=11, polyorder=3)

my_colors = [
    '#1f77b4',  # Blue (深)
    '#ff7f0e',  # Orange (預設第二色)
    '#2ca02c',  # Green (預設第三色)
    '#d62728',  # Red (深)
    '#9467bd',  # Purple (預設第五色)
    '#8c564b'   # Brown (預設第六色)
]


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


## OFFSET csv
if offset_csv is not None:
    # read CSV and validate columns
    offset_df = pd.read_csv(offset_csv)
    if "run_index" not in offset_df.columns or "timestamp" not in offset_df.columns:
        raise ValueError("CSV must include columns: run_index, timestamp, and at least one qubit column")

    # check and set x-axis
    if x_axis not in ("run_index", "timestamp"):
        raise ValueError("x_axis must be either 'run_index' or 'timestamp'")
    if x_axis == "timestamp":
        offset_df["timestamp"] = pd.to_datetime(offset_df["timestamp"], errors="coerce")

    # identify qubit columns (exclude run_index and timestamp)
    offset_qubit_cols = [c for c in offset_df.columns if c not in ("run_index", "timestamp")]
    if not offset_qubit_cols:
        raise ValueError("No qubit columns found")

    # convert to numeric values (non-numeric become NaN)
    for c in offset_qubit_cols:
        offset_df[c] = pd.to_numeric(offset_df[c], errors="coerce")

# plot each qubit separately
for q in qubit_cols:
    if offset_csv is None:
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
    else:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)
        # Plot CZ trend
        sub = df[[x_axis, q]].dropna()
        if sub.empty:
            continue
        ax0:Axes = axes[0]
        mean, sd = round(np.mean(np.array(sub[q]*100)),1), round(np.std(np.array(sub[q]*100)),1)
        ax0.plot(sub[x_axis], sub[q]*100,zorder=1)
        ax0.hlines(mean-sd, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),linestyles="--",colors='pink',label='Deviation')
        ax0.hlines(mean+sd, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),linestyles="--",colors='pink')
        ax0.hlines(mean, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),colors='red',label="mean")
        ax0.scatter(sub.loc[sub[q].idxmax(), x_axis],np.max(sub[q]*100),c='red',marker="*",s=100,label=f"best {round(np.max(sub[q]*100),1)} % at {sub.loc[sub[q].idxmax(), x_axis]}",zorder=3)
        ax0.set_title(f"{q} CZ fidelity = {mean} $\pm$ {sd} %")
       
        ax0.set_ylabel("CZ fidelity (%)")
        ax0.grid(True, linestyle="--", alpha=0.4)
        
        ax0.legend(fontsize=6)
        
        # Plot offset trend
        qs = q.split("_")[1:]
        ax1:Axes = axes[1]
        for idx, a_q in enumerate(qs):
            offset_sub = offset_df[[x_axis, a_q]].dropna()
            if sub.empty:
                continue
            mod, y_normed = norm_1darray(np.array(offset_sub[a_q]))
            ax1.plot(offset_sub[x_axis], y_normed, label=f"{a_q}, mod={round(mod, 1)} mV", c=my_colors[idx])
            if offset_when_calibration != {}:
                if a_q in offset_when_calibration:
                    offset_ref = (offset_when_calibration[a_q] - min(np.array(offset_sub[a_q])))/mod
                    ax1.hlines(y=offset_ref*1000, xmin=min(offset_sub[x_axis]), xmax=max(offset_sub[x_axis]), label=f"{a_q} CZ calibrate offset", linestyle="--", alpha=0.7, colors=my_colors[idx])
        ax1.set_xlabel(x_axis)
        ax1.set_ylabel("Normalized independent_offset")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(fontsize=6)
        plt.xticks(rotation=90)
        fig.savefig(out_dir / f"{q}_line_{x_axis}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)