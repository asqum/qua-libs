import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import savgol_filter


# manually specified 
files = {"DR4,5q4c-1":"/home/ratiswu/Flux_drift/5q4c-1_DR4/independent_offset_2025-11-08T17-18-10.csv", "DR4,5q4c-2":"/home/ratiswu/Flux_drift/5q4c-2_DR4/independent_offset_2025-11-08T17-44-25.csv"}
fig_save_folder = "/home/ratiswu/Flux_drift/5q4c-1_DR4"
plt.figure(figsize=(10, 6))

# =====================================================================================================================================


class Artist:
    def __init__(self):
        pass

    def assign_color(self, index:int):
        cs = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        return cs[index % len(cs)]

    def assign_linestyle(self, index:int):
        linestyles = [None,'-', '--', '-.', ':']
        return linestyles[index % len(linestyles)]

    def assign_marker(self, index:int):
        markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+']
        return markers[index % len(markers)]
    
    def assign_marksize(self,total_pts:int,  scale_factor:float=2.0):
        return scale_factor/np.log10(total_pts)


def norm_1darray(y):
    y = np.array(y)*1000 # convert to mV
    modulation = np.max(y) - np.min(y)
    return modulation, savgol_filter((y - np.min(y)) / modulation, window_length=11, polyorder=3)



color_i = 0
for sample_name in files:
    df = pd.read_csv(files[sample_name])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in df.columns:
        if col not in ["run_index", "timestamp"]:
            color_i += 1
            y = df[col].values
            mod, y_norm = norm_1darray(y)
            plt.plot(df["timestamp"], y_norm, label=f"{sample_name}_{col} mod = {round(mod,1)} mV ", linestyle=Artist().assign_linestyle(color_i), c=Artist().assign_color(color_i))
            

plt.xlabel("Timestamp")
plt.ylabel("Normalized Offset (mV)")
plt.title("Smoothened, Flux offset vs Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(fig_save_folder, "flux_offset_vs_time.png"), dpi=450)
plt.close()
# plt.show()