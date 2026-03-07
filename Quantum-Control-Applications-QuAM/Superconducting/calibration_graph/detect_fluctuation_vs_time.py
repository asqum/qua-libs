import time
import csv
from datetime import datetime
from pathlib import Path

from node_for_repeat import run_once  

def to_float_clean(x):
    s = str(x).strip().lstrip("'")
    return float(s)

num_runs = 'inf' #Change an integer. If you wanna keep tracking, set it as 'inf'.
cooldown_sec = 5 #Change

# folder path to save the file
out_dir = Path('/home/ratiswu/Qualibrate_data/10Q9Cv2_q1q5/2026-02-01/#a123_OFFSET_tracking') #Change
out_dir.mkdir(parents=True, exist_ok=True)

# csv file name (contains the time to avoid overlap)
session_ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")
wide_csv = out_dir / f"independent_offset_{session_ts}.csv" 
i = 0
header_qubits = None  
while True:
    i += 1
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{i:02d}/{num_runs}] {ts} Starting run…")

    try:
        fluctuations, qubit_names = run_once()

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
    if not isinstance(num_runs, str):
        if isinstance(num_runs, int):
            if i == num_runs :
                break

print(f"All runs finished: {wide_csv}")


# Just make sure to close qms
from quam_libs.components import QuAM
machine = QuAM.load()
qmm = machine.connect() # Use this line if you want to connect to the QOP defined in the state file
qmm.close_all_qms()

