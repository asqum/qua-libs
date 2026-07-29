"""
Set xy gate length for all active qubits in the current state.json:
  - Set x180.length to TARGET_LENGTH (ns)
  - Scale x180.amplitude and x90.amplitude by old_length / TARGET_LENGTH
    so pulse area (rotation angle) is approximately preserved

x90.length (and other gates that reference x180.length) follow via QuAM references.
"""
from quam_libs.components import QuAM

# Parameter: target x180 pulse length in ns (must be a multiple of 4)
target_qubits = ['q3','q4','q5','q6','q7','q8']
TARGET_LENGTH = 16

machine = QuAM.load()
if target_qubits is None or target_qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in target_qubits]

if TARGET_LENGTH % 4 != 0:
    raise ValueError(f"TARGET_LENGTH must be a multiple of 4 ns, got {TARGET_LENGTH}")

for q in qubits:
    x180 = q.xy.operations["x180"]
    x90 = q.xy.operations["x90"]
    old_length = int(x180.length)
    if old_length <= 0:
        raise ValueError(f"{q.name} x180.length must be > 0, got {old_length}")

    scale = old_length / TARGET_LENGTH
    old_x180_amp = float(x180.amplitude)
    old_x90_amp = float(x90.amplitude)
    new_x180_amp = old_x180_amp * scale
    new_x90_amp = old_x90_amp * scale

    print(
        f"{q.name}: length {old_length} -> {TARGET_LENGTH} (scale={scale:.6g}), "
        f"x180.amp {old_x180_amp:.6g} -> {new_x180_amp:.6g}, "
        f"x90.amp {old_x90_amp:.6g} -> {new_x90_amp:.6g}"
    )
    x180.length = TARGET_LENGTH
    x180.amplitude = new_x180_amp
    x90.amplitude = new_x90_amp

machine.save()
print(f"Saved state with gate length {TARGET_LENGTH} ns for {len(qubits)} active qubit(s).")
