"""
Reset x180 / x90 pulse parameters for all active qubits in the current state.json:
  alpha = -1.0, detuning = 0.0
"""
from quam_libs.components import QuAM

target_qubits = ['q3','q4','q5','q6','q7','q8']

machine = QuAM.load()
if target_qubits is None or target_qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in target_qubits]

for q in qubits:
    for op in ("x180", "x90"):
        pulse = q.xy.operations[op]
        print(
            f"{q.name} {op}: alpha {pulse.alpha} -> -1.0, "
            f"detuning {getattr(pulse, 'detuning', None)} -> 0.0"
        )
        pulse.alpha = -1.0
        pulse.detuning = 0.0

machine.save()
print(f"Saved state with updated x180/x90 for {len(qubits)} active qubit(s).")
