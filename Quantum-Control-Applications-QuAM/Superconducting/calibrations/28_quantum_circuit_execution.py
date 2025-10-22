from qiskit import QuantumCircuit
from quam_libs.components import QuAM
from quam_libs.experiments.qiskit_circuit import run_qiskit_to_qua_program

machine = QuAM.load()
n_shots = 1024
optimization_level = 1
target_qubit_indices = [0, 1]
target_qubits = [machine.active_qubits[i] for i in target_qubit_indices]

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure(0, 0)
qc.measure(1, 1)
results = run_qiskit_to_qua_program(qc, machine, target_qubits, n_shots, optimization_level)
print(results)

