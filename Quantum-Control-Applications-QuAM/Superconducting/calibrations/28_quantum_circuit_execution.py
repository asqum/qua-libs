from qiskit import QuantumCircuit, transpile
from quam_libs.components import QuAM
from quam_libs.experiments.qiskit_circuit import run_qiskit_to_qua_program, create_target
machine = QuAM.load()
n_shots = 1024
manual_transpile = True
target_qubit_indices = [0, 1]
target_qubits = [machine.active_qubits[i] for i in target_qubit_indices]

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure(0, 0)
qc.measure(1, 1)

if manual_transpile:
    optimization_level = 1
    target = create_target(machine)
    # Transpile the circuit to the target (Optional: if not done here, will be done in the run_qiskit_to_qua_program function)
    qc = transpile(qc, target=target, initial_layout=target_qubit_indices, optimization_level=optimization_level)
    results = run_qiskit_to_qua_program(qc, machine, n_shots=n_shots)

else:
    optimization_level = 1 # Default optimization level is 1, has to be specified if manual_transpile is False
    results = run_qiskit_to_qua_program(qc, machine, target_qubits, n_shots, optimization_level)

print(results)

# Results in the form: {'c1": {"00": 512, "11": 512}, "c0": {"00": 512, "11": 512}, ...}