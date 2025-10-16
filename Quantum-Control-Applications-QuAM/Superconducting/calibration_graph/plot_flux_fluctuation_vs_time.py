from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ======= Configuration =======
csv_path = Path(r"d:\qm_code\as\qua-libs\Quantum-Control-Applications-QuAM\Superconducting\data\independent_offset_2025-10-17T02-50-34.csv")
base_dir = Path(r"d:\qm_code\as\qua-libs\Quantum-Control-Applications-QuAM\Superconducting\data")
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
    plt.plot(sub[x_axis], sub[q], marker="o")
    plt.title(f"{q} independent_offset")
    plt.xlabel(x_axis)
    plt.ylabel("independent_offset")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(rotation=90)
    fig.savefig(out_dir / f"{q}_line_{x_axis}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# plot all qubits together
fig_all = plt.figure()
for q in qubit_cols:
    sub = df[[x_axis, q]].dropna()
    if sub.empty:
        continue
    plt.plot(sub[x_axis], sub[q], marker="o", label=q)
plt.title("All qubits – independent_offset")
plt.xlabel(x_axis)
plt.ylabel("independent_offset")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.xticks(rotation=90)
fig_all.savefig(out_dir / f"all_qubits_line_{x_axis}.png", dpi=150, bbox_inches="tight")
plt.close(fig_all)

print(f"Done. Charts saved to: {out_dir.resolve()}")