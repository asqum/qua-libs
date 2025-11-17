from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from quam_libs.components import QuAM
from quam_libs.experiments.qiskit_circuit import design_qua_program_from_qiskit, run_qua_program_and_return_results, create_target
machine = QuAM.load()
qmm = machine.connect()
qm = qmm.open_qm(machine.generate_config(), close_other_machines=True)
n_shots = 1024
manual_transpile = True
target_qubit_indices = [0,1,2,3,4]
target_qubits = [machine.active_qubits[i] for i in target_qubit_indices]

for qubit in target_qubits:
    qubit.macros['reset'].reset_type='thermalize'
    qubit.macros['reset'].thermalize_time = qubit.thermalization_time

multiplexed_measurement = Instruction("multiplexed_measurement", num_qubits=len(target_qubits), num_clbits=len(target_qubits), params=[])
qc = QuantumCircuit(len(target_qubits), len(target_qubits))
# qc.x(0)
qc.h(0)
# qc.cx(0, 1)
# qc.measure(0, 0)
qc.append(multiplexed_measurement, range(len(target_qubits)), range(len(target_qubits)))
# qc.measure(1, 1)
# qc.measure_all()

if manual_transpile:
    optimization_level = 1
    target = create_target(machine)
    # Transpile the circuit to the target (Optional: if not done here, will be done in the run_qiskit_to_qua_program function)
    qc = transpile(qc, target=target, initial_layout=target_qubit_indices, optimization_level=optimization_level)
    prog = design_qua_program_from_qiskit(qc, machine, n_shots=n_shots)
    results = run_qua_program_and_return_results(prog, machine, qc, n_shots, qm)

else:
    optimization_level = 1 # Default optimization level is 1, has to be specified if manual_transpile is False
    prog = design_qua_program_from_qiskit(qc, machine, target_qubits, n_shots, optimization_level)
    results = run_qua_program_and_return_results(prog, machine, qc, n_shots, qm)

print(results)

# Results in the form: {'c1": {"00": 512, "11": 512}, "c0": {"00": 512, "11": 512}, ...}