from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from scipy.signal import savgol_filter
# ======= Configuration =======
x_axis = "timestamp"   # can be "run_index" or "timestamp"
base_dir = Path("/home/ratiswu/Qualibrate_data/5Q4C_Qcage/2025-12-12/#1a1a_2QRB_Tracking")
## OFFSET start point
offset_when_calibration:dict = {"q2": 0.13466227687115415, "q3": 0.16022179335949935}
# =============================


def sort_csv(folder_path: str | Path):
    folder = Path(folder_path)
    sorted_pths = {}
    types = 0
    for p in folder.glob("*.csv"):
        name = p.name

        if name.startswith("BellState_Tracking_"):
            sorted_pths["Bell Fidelity"] = p
            types += 1
        elif name.startswith("CZ_Fidelity_Tracking_"):
            sorted_pths["CZ Fidelity"] = p
            types += 1
        elif name.startswith("Offset_Tracking_"):
            sorted_pths["Offset"] = p
            types += 1
        elif name.startswith("Standard2Q_Fidelity_Tracking_"):
            sorted_pths["Standard Fidelity"] = p
            types += 1
            
    order = ["CZ Fidelity", "Bell Fidelity", "Standard Fidelity", "Offset"]

    sorted_pths = {k: sorted_pths[k] for k in order if k in sorted_pths}
    return sorted_pths, types


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

pths, num = sort_csv(base_dir)
print(pths)
fig, axes = plt.subplots(nrows=num, ncols=1, figsize=(10, 8), sharex=True)
pair = ""
for idx, cata in enumerate(list(pths.keys())):
    ax:Axes = axes[idx]
    df = pd.read_csv(pths[cata])
    if "run_index" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("CSV must include columns: run_index, timestamp, and at least one qubit column")
    # # check and set x-axis
    if x_axis not in ("run_index", "timestamp"):
        raise ValueError("x_axis must be either 'run_index' or 'timestamp'")
    if x_axis == "timestamp":
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # # identify qubit columns (exclude run_index and timestamp)
    qubit_cols = [c for c in df.columns if c not in ("run_index", "timestamp")]
    if not qubit_cols:
        raise ValueError("No qubit columns found")

    # convert to numeric values (non-numeric become NaN)
    for c in qubit_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    
    if cata != 'Offset':
        for q in qubit_cols:
            sub = df[[x_axis, q]].dropna()
            if sub.empty:
                continue
            mean, sd = round(np.mean(np.array(sub[q]*100)),1), round(np.std(np.array(sub[q]*100)),1)
            ax.plot(sub[x_axis], sub[q]*100,zorder=1)
            # plt.plot(sub[x_axis], savgol_filter(sub[q]*100,window_length=31,polyorder=5),zorder=2,label='filtered')
            ax.hlines(mean-sd, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),linestyles="--",colors='pink',label='Deviation')
            ax.hlines(mean+sd, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),linestyles="--",colors='pink')
            ax.hlines(mean, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),colors='red',label="mean")
            ax.scatter(sub.loc[sub[q].idxmax(), x_axis],np.max(sub[q]*100),c='red',marker="*",s=100,label=f"best {round(np.max(sub[q]*100),1)} % at {sub.loc[sub[q].idxmax(), x_axis]}",zorder=3)
        
            if cata == 'CZ Fidelity':
                ax.set_title(f"{q} CZ fidelity = {mean} $\pm$ {sd} %")
                ax.set_ylabel("CZ fidelity (%)")
            elif cata == 'Standard Fidelity':
                ax.set_title(f"{q} 2Q standard fidelity = {mean} $\pm$ {sd} %")
                ax.set_ylabel("2Q fidelity (%)")
            else:
                ax.set_title(f"{q} Bell-State fidelity = {mean} $\pm$ {sd} %")
                ax.set_ylabel("Bell-State fidelity (%)")
            pair = q

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=6)
    else:
        print(pair)
        
        qs = pair.split("_")[1:]
        for idx, a_q in enumerate(qs):
            sub = df[[x_axis, a_q]].dropna()
            if sub.empty:
                continue
            mod, y_normed = norm_1darray(np.array(sub[a_q]))
            ax.plot(sub[x_axis], y_normed, label=f"{a_q}, mod={round(mod, 1)} mV", c=my_colors[idx])
            if offset_when_calibration != {}:
                if a_q in offset_when_calibration:
                    offset_ref = (offset_when_calibration[a_q] - min(np.array(sub[a_q])))/mod
                    ax.hlines(y=offset_ref*1000, xmin=min(sub[x_axis]), xmax=max(sub[x_axis]), label=f"{a_q} CZ calibrate offset", linestyle="--", alpha=0.7, colors=my_colors[idx])
            ax.legend(fontsize=6)
        
        ax.set_xlabel(x_axis)
        ax.set_ylabel("Normalized offset")
        ax.grid(True, linestyle="--", alpha=0.4)
        



    fig.savefig(base_dir / f"line_{x_axis}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# # create a subfolder using the CSV file name
# out_dir = base_dir / csv_path.stem
# out_dir.mkdir(parents=True, exist_ok=True)



# # read CSV and validate columns
# df = pd.read_csv(csv_path)
# if "run_index" not in df.columns or "timestamp" not in df.columns:
#     raise ValueError("CSV must include columns: run_index, timestamp, and at least one qubit column")

# # check and set x-axis
# if x_axis not in ("run_index", "timestamp"):
#     raise ValueError("x_axis must be either 'run_index' or 'timestamp'")
# if x_axis == "timestamp":
#     df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# # identify qubit columns (exclude run_index and timestamp)
# qubit_cols = [c for c in df.columns if c not in ("run_index", "timestamp")]
# if not qubit_cols:
#     raise ValueError("No qubit columns found")

# # convert to numeric values (non-numeric become NaN)
# for c in qubit_cols:
#     df[c] = pd.to_numeric(df[c], errors="coerce")


# ## OFFSET csv
# if offset_csv is not None:
#     # read CSV and validate columns
#     offset_df = pd.read_csv(offset_csv)
#     if "run_index" not in offset_df.columns or "timestamp" not in offset_df.columns:
#         raise ValueError("CSV must include columns: run_index, timestamp, and at least one qubit column")

#     # check and set x-axis
#     if x_axis not in ("run_index", "timestamp"):
#         raise ValueError("x_axis must be either 'run_index' or 'timestamp'")
#     if x_axis == "timestamp":
#         offset_df["timestamp"] = pd.to_datetime(offset_df["timestamp"], errors="coerce")

#     # identify qubit columns (exclude run_index and timestamp)
#     offset_qubit_cols = [c for c in offset_df.columns if c not in ("run_index", "timestamp")]
#     if not offset_qubit_cols:
#         raise ValueError("No qubit columns found")

#     # convert to numeric values (non-numeric become NaN)
#     for c in offset_qubit_cols:
#         offset_df[c] = pd.to_numeric(offset_df[c], errors="coerce")

# # plot each qubit separately
# for q in qubit_cols:
#     if offset_csv is None:
#         sub = df[[x_axis, q]].dropna()
#         if sub.empty:
#             continue
#         fig = plt.figure()
#         mean, sd = round(np.mean(np.array(sub[q]*100)),1), round(np.std(np.array(sub[q]*100)),1)
#         plt.plot(sub[x_axis], sub[q]*100,zorder=1)
#         # plt.plot(sub[x_axis], savgol_filter(sub[q]*100,window_length=31,polyorder=5),zorder=2,label='filtered')
#         plt.hlines(mean-sd, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),linestyles="--",colors='pink',label='Deviation')
#         plt.hlines(mean+sd, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),linestyles="--",colors='pink')
#         plt.hlines(mean, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),colors='red',label="mean")
#         plt.scatter(sub.loc[sub[q].idxmax(), x_axis],np.max(sub[q]*100),c='red',marker="*",s=100,label=f"best {round(np.max(sub[q]*100),1)} % at {sub.loc[sub[q].idxmax(), x_axis]}",zorder=3)
#         plt.title(f"{q} CZ fidelity = {mean} $\pm$ {sd} %")
#         plt.xlabel(x_axis)
#         plt.ylabel("CZ fidelity (%)")
#         plt.grid(True, linestyle="--", alpha=0.4)
#         plt.xticks(rotation=90)
#         plt.legend(fontsize=6)
#         fig.savefig(out_dir / f"{q}_line_{x_axis}.png", dpi=150, bbox_inches="tight")
#         plt.close(fig)
#     else:
#         fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)
#         # Plot CZ trend
#         sub = df[[x_axis, q]].dropna()
#         if sub.empty:
#             continue
#         ax0:Axes = axes[0]
#         mean, sd = round(np.mean(np.array(sub[q]*100)),1), round(np.std(np.array(sub[q]*100)),1)
#         ax0.plot(sub[x_axis], sub[q]*100,zorder=1)
#         ax0.hlines(mean-sd, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),linestyles="--",colors='pink',label='Deviation')
#         ax0.hlines(mean+sd, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),linestyles="--",colors='pink')
#         ax0.hlines(mean, xmin=np.min(sub[x_axis]),xmax=np.max(sub[x_axis]),colors='red',label="mean")
#         ax0.scatter(sub.loc[sub[q].idxmax(), x_axis],np.max(sub[q]*100),c='red',marker="*",s=100,label=f"best {round(np.max(sub[q]*100),1)} % at {sub.loc[sub[q].idxmax(), x_axis]}",zorder=3)
#         ax0.set_title(f"{q} CZ fidelity = {mean} $\pm$ {sd} %")
       
#         ax0.set_ylabel("CZ fidelity (%)")
#         ax0.grid(True, linestyle="--", alpha=0.4)
        
#         ax0.legend(fontsize=6)
        
#         # Plot offset trend
#         qs = q.split("_")[1:]
#         ax1:Axes = axes[1]
#         for idx, a_q in enumerate(qs):
#             offset_sub = offset_df[[x_axis, a_q]].dropna()
#             if sub.empty:
#                 continue
#             mod, y_normed = norm_1darray(np.array(offset_sub[a_q]))
#             ax1.plot(offset_sub[x_axis], y_normed, label=f"{a_q}, mod={round(mod, 1)} mV", c=my_colors[idx])
#             if offset_when_calibration != {}:
#                 if a_q in offset_when_calibration:
#                     offset_ref = (offset_when_calibration[a_q] - min(np.array(offset_sub[a_q])))/mod
#                     ax1.hlines(y=offset_ref*1000, xmin=min(offset_sub[x_axis]), xmax=max(offset_sub[x_axis]), label=f"{a_q} CZ calibrate offset", linestyle="--", alpha=0.7, colors=my_colors[idx])
#         ax1.set_xlabel(x_axis)
#         ax1.set_ylabel("Normalized independent_offset")
#         ax1.grid(True, linestyle="--", alpha=0.4)
#         ax1.legend(fontsize=6)
#         plt.xticks(rotation=90)
#         fig.savefig(out_dir / f"{q}_line_{x_axis}.png", dpi=150, bbox_inches="tight")
#         plt.close(fig)