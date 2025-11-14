import time
import csv
from datetime import datetime
from pathlib import Path

from node_for_repeat import run_once  

def to_float_clean(x):
    s = str(x).strip().lstrip("'")
    return float(s)

num_runs = 'inf' # Change, 'inf' or an integer 
cooldown_sec = 10 #Chnage

i = 0
# folder path to save the file
out_dir = Path('/home/ratiswu/Qualibrate_data/5Q4C_Qcage/2025-11-14/#9a9a_2QRB_Tracking') #Chnage
out_dir.mkdir(parents=True, exist_ok=True)

# csv file name (contains the time to avoid overlap)
session_ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")
wide_csv = out_dir / f"CZ_Fidelity_Tracking_{session_ts}.csv" 

header_qubits = None  

start = time.time()
# for i in range(1, num_runs + 1):
while True:
    i += 1
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{i:02d}/{num_runs}] {ts} Starting run…")

    try:
        CZ_fidelity, pair_names = run_once()
        print(pair_names)
        # create column names for qubits, determined by the first run
        if header_qubits is None:
            header_qubits = list(pair_names)
            with open(wide_csv, "w", newline="") as f:
                csv.writer(f).writerow(["run_index", "timestamp", *header_qubits])
            f.close()

        with open(wide_csv, "a", newline="") as f:
            writer = csv.writer(f)
            name_to_val = {n: to_float_clean(v) for n, v in zip(pair_names, CZ_fidelity)}
            row_vals = [name_to_val[q] for q in header_qubits]  
            writer.writerow([i, ts, *row_vals])
        f.close()

        print(f"[{i:02d}/{num_runs}] appended to {wide_csv.name}")

    except Exception as e:
        print(f"[{i:02d}/{num_runs}] error: {e!r}, continuing to the next run…")

    time.sleep(cooldown_sec)
    each_time = time.time()
    if each_time - start > 3*24*3600:
        break
    if not isinstance(num_runs, str):
        if isinstance(num_runs, int):
            if i == num_runs :
                break

print(f"All runs finished: {wide_csv}")
final_time = time.time()


# Just make sure to close qms
from quam_libs.components import QuAM
machine = QuAM.load()
qmm = machine.connect() # Use this line if you want to connect to the QOP defined in the state file
qmm.close_all_qms()

print(f"Total elapsed {final_time-start} secs for CD = {cooldown_sec} secs and {i} runs.")

